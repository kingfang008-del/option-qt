#!/usr/bin/env python3
"""Outcome-first rule mining on extended-hours ≥8% rallies.

1) Find pre/AH sessions whose peak return from session open ≥ 8%.
2) Exit rule (no fixed clock): after the run, flatten when price enters a plateau
   (no new high for N minutes and drawdown from high-water within band).
3) Reverse-engineer early-path factors (VP/vol z, accel, efficiency, time)
   that discriminate winners vs non-winners and rank rules by win-rate / expectancy
   under plateau exits.
"""
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from qqq_btc.tools.validate_stock_ext_hours_ignition_sw import (
    NY,
    add_causal_tod_z,
    add_window_features,
    list_symbols,
    load_1min,
    month_range,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="/mnt/s990/data/all_data/stocks_15s_parquet")
    p.add_argument("--start-month", default="2024-01")
    p.add_argument("--end-month", default="2025-06")
    p.add_argument("--max-symbols", type=int, default=0)
    p.add_argument("--lookback-days", type=int, default=20)
    p.add_argument("--winner-ret", type=float, default=0.08, help="peak cum-ret from session open")
    p.add_argument("--onset-ret", type=float, default=0.015, help="early mark on way to peak")
    p.add_argument("--plateau-mins", type=int, default=10, help="minutes without new high")
    p.add_argument("--plateau-band", type=float, default=0.004, help="max dd from HWM to call flat")
    p.add_argument("--z-vol", type=float, default=2.0)
    p.add_argument("--z-vp", type=float, default=2.0)
    p.add_argument("--min-win-ret", type=float, default=0.002)
    p.add_argument(
        "--output-dir",
        default="qqq_btc/results/stock_ext_hours_winner8_plateau_rules",
    )
    return p.parse_args()


def session_slice(df: pd.DataFrame, session: str) -> pd.DataFrame:
    if session == "pre":
        m = (df["tod"] >= pd.Timestamp("04:00").time()) & (df["tod"] < pd.Timestamp("09:30").time())
    else:
        m = (df["tod"] >= pd.Timestamp("16:00").time()) & (df["tod"] < pd.Timestamp("20:00").time())
    return df.loc[m].copy()


def find_plateau_exit(
    closes: np.ndarray,
    peak_idx: int,
    plateau_mins: int,
    plateau_band: float,
) -> int:
    """After peak_idx (first time at session peak), find plateau start index.

    Plateau: last `plateau_mins` bars all within `plateau_band` of running HWM
    since peak, and no new HWM in that window.
    Returns exit index (inclusive), or last index if never.
    """
    n = len(closes)
    if peak_idx >= n - 1:
        return n - 1
    hwm = closes[peak_idx]
    flat_run = 0
    for i in range(peak_idx + 1, n):
        px = closes[i]
        if px > hwm * (1.0 + 1e-12):
            hwm = px
            flat_run = 0
            continue
        dd = (hwm - px) / hwm if hwm > 0 else 0.0
        if dd <= plateau_band:
            flat_run += 1
        else:
            # broke down from plateau band — exit at previous bar if had flat, else now
            if flat_run >= plateau_mins:
                return i - 1
            flat_run = 0
        if flat_run >= plateau_mins:
            return i
    return n - 1


def process_symbol(df_1m: pd.DataFrame, symbol: str, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (winner_events, onset_candidates)."""
    if df_1m.empty:
        return pd.DataFrame(), pd.DataFrame()

    winners = []
    onsets = []

    for session in ("pre", "ah"):
        sess = session_slice(df_1m, session)
        if sess.empty:
            continue
        sess = add_window_features(sess, (5, 10, 15))
        sess["sess_open"] = sess.groupby("date_str")["close"].transform("first")
        sess["cum_ret"] = sess["close"] / sess["sess_open"] - 1.0
        z_cols = [f"vol_{w}" for w in (5, 10, 15)] + [f"vp_{w}" for w in (5, 10, 15)]
        sess = add_causal_tod_z(sess, z_cols, args.lookback_days)

        for date_str, day in sess.groupby("date_str", sort=True):
            day = day.sort_values("timestamp").reset_index(drop=True)
            if len(day) < 30:
                continue
            cum = day["cum_ret"].to_numpy(dtype=float)
            closes = day["close"].to_numpy(dtype=float)
            peak_ret = float(np.nanmax(cum))
            peak_idx = int(np.nanargmax(cum))
            is_winner = peak_ret >= args.winner_ret

            # long-side early unusual only (拉升初期，不做下跌腿)
            unusual = (
                ((day["z_vol_10"] >= args.z_vol) | (day["z_vp_10"] >= args.z_vp))
                & (day["ret_10"] >= args.min_win_ret)
                & (day["accel_10"].fillna(0) > 0)
                & (day["cum_ret"] >= 0)
            )
            early_unusual = (unusual.fillna(False) & (day["cum_ret"] <= args.onset_ret * 1.5)).to_numpy()
            u_hits = np.where(early_unusual)[0]
            u_pos = int(u_hits[0]) if len(u_hits) else None

            # onset: first time cum reaches +onset_ret (up move)
            onset_hits = np.where(cum >= args.onset_ret)[0]
            onset_idx = int(onset_hits[0]) if len(onset_hits) else None

            entry_pos = u_pos if u_pos is not None else None
            if entry_pos is None and onset_idx is not None:
                # only keep if still "early" and path is up
                if cum[onset_idx] <= args.onset_ret * 1.5:
                    entry_pos = onset_idx
            if entry_pos is None:
                if is_winner:
                    soft = np.where(cum >= min(0.005, args.onset_ret))[0]
                    entry_pos = int(soft[0]) if len(soft) else 0
                else:
                    continue

            # require long-side entry
            if cum[entry_pos] < -0.002 and day.iloc[entry_pos]["ret_10"] < 0:
                if not is_winner:
                    continue

            # plateau exit after the path peak (for winners) or after local peak post-entry
            if is_winner:
                # peak of whole session for anatomy; exit after that peak's plateau
                exit_pos = find_plateau_exit(closes, peak_idx, args.plateau_mins, args.plateau_band)
            else:
                # for non-winner candidates: peak after entry within session
                post = cum.copy()
                post[: entry_pos + 1] = -np.inf
                if not np.isfinite(post).any() or np.isneginf(post).all():
                    local_peak = entry_pos
                else:
                    local_peak = int(np.nanargmax(post))
                exit_pos = find_plateau_exit(closes, local_peak, args.plateau_mins, args.plateau_band)

            entry_px = float(closes[entry_pos])
            exit_px = float(closes[exit_pos])
            if entry_px <= 0:
                continue
            direction = 1.0  # long the rally path
            trade_ret = direction * (exit_px / entry_px - 1.0)
            mfe = float(np.nanmax(cum[entry_pos : exit_pos + 1]) - cum[entry_pos]) if exit_pos >= entry_pos else float("nan")
            mae = float(np.nanmin(cum[entry_pos : exit_pos + 1]) - cum[entry_pos]) if exit_pos >= entry_pos else float("nan")

            row = day.iloc[entry_pos]
            feat = {
                "symbol": symbol,
                "session": session,
                "date_str": date_str,
                "is_winner8": int(is_winner),
                "peak_ret": peak_ret,
                "entry_ts": str(row["timestamp"]),
                "entry_mod": int(row["mod"]),
                "entry_hour": int(row["timestamp"].hour),
                "entry_cum": float(row["cum_ret"]),
                "exit_pos_frac": float(exit_pos / max(len(day) - 1, 1)),
                "hold_mins": int(exit_pos - entry_pos),
                "trade_ret_plateau": trade_ret,
                "mfe_from_entry": mfe,
                "mae_from_entry": mae,
                "hit_onset_mark": int(onset_idx is not None),
                "entry_via_unusual": int(u_pos is not None),
            }
            for w in (5, 10, 15):
                for c in (f"ret_{w}", f"accel_{w}", f"z_vol_{w}", f"z_vp_{w}", f"eff_{w}"):
                    v = row.get(c, np.nan)
                    feat[c] = float(v) if pd.notna(v) else float("nan")
            # path quality to peak (winners) / to local peak
            feat["eff_to_peak"] = float(
                abs(peak_ret - float(row["cum_ret"]))
                / max(float(np.abs(np.diff(cum[entry_pos : peak_idx + 1])).sum()) if peak_idx > entry_pos else 1e-8, 1e-8)
            ) if is_winner else float("nan")

            if is_winner:
                winners.append(feat)
            # all onset/unusual candidates for rule mining (includes winners)
            if u_pos is not None or (onset_idx is not None and float(row["cum_ret"]) <= args.onset_ret * 1.5):
                onsets.append(feat)

    return pd.DataFrame(winners), pd.DataFrame(onsets)


def _worker(payload: tuple) -> tuple[pd.DataFrame, pd.DataFrame]:
    root_s, sym, months, args_dict = payload
    ns = argparse.Namespace(**args_dict)
    df = load_1min(Path(root_s), sym, months)
    return process_symbol(df, sym, ns)


def factor_bins(s: pd.Series, qs=(0.33, 0.67)) -> pd.Series:
    try:
        return pd.qcut(s, q=[0, qs[0], qs[1], 1.0], labels=["low", "mid", "high"], duplicates="drop")
    except Exception:
        return pd.Series(["na"] * len(s), index=s.index)


def rule_table(onsets: pd.DataFrame, winner_col: str = "is_winner8") -> pd.DataFrame:
    """Rank simple factor rules by P(winner) and plateau trade stats among fires."""
    rows = []
    if onsets.empty:
        return pd.DataFrame()

    def add_rule(name: str, mask: pd.Series):
        sub = onsets.loc[mask]
        if len(sub) < 30:
            return
        rows.append(
            {
                "rule": name,
                "n": int(len(sub)),
                "p_winner8": float(sub[winner_col].mean()),
                "avg_peak": float(sub["peak_ret"].mean()),
                "avg_trade_plateau": float(sub["trade_ret_plateau"].mean()),
                "med_trade_plateau": float(sub["trade_ret_plateau"].median()),
                "wr_trade_gt0": float((sub["trade_ret_plateau"] > 0).mean()),
                "wr_trade_gt2pct": float((sub["trade_ret_plateau"] > 0.02).mean()),
                "avg_hold_mins": float(sub["hold_mins"].mean()),
            }
        )

    add_rule("baseline_all_onset", pd.Series(True, index=onsets.index))

    # single factor high tercile
    for col in ["z_vol_10", "z_vp_10", "accel_10", "ret_10", "eff_10", "z_vol_5", "z_vp_5", "eff_5"]:
        if col not in onsets.columns:
            continue
        b = factor_bins(onsets[col].abs() if col.startswith("accel") or col.startswith("ret") else onsets[col])
        # for signed ret/accel use raw high (bullish)
        if col.startswith("ret") or col.startswith("accel"):
            b = factor_bins(onsets[col])
        for lab in ["low", "mid", "high"]:
            if lab in set(b.astype(str)):
                add_rule(f"{col}={lab}", b.astype(str) == lab)

    # directional combinations (long-side early)
    m_vol = onsets["z_vol_10"] >= 2.0
    m_vp = onsets["z_vp_10"] >= 2.0
    m_acc = onsets["accel_10"] > 0
    m_ret = onsets["ret_10"] > 0
    m_eff = onsets["eff_10"] >= onsets["eff_10"].median()
    m_early_hour_pre = onsets["entry_hour"] <= 6
    m_late_pre = onsets["entry_hour"] >= 8
    m_cum_small = onsets["entry_cum"].abs() <= 0.02
    m_via_u = onsets["entry_via_unusual"] == 1

    add_rule("z_vol10>=2 & ret10>0", m_vol & m_ret)
    add_rule("z_vp10>=2 & ret10>0", m_vp & m_ret)
    add_rule("z_vol10>=2 & accel>0 & ret>0", m_vol & m_acc & m_ret)
    add_rule("z_vp10>=2 & accel>0 & ret>0", m_vp & m_acc & m_ret)
    add_rule("vol+vp>=2 & ret>0", m_vol & m_vp & m_ret)
    add_rule("vol+vp+accel+eff", m_vol & m_vp & m_acc & m_ret & m_eff)
    add_rule("vol+vp+accel & hour<=6", m_vol & m_vp & m_acc & m_ret & m_early_hour_pre)
    add_rule("vol+vp+accel & hour>=8", m_vol & m_vp & m_acc & m_ret & m_late_pre)
    add_rule("eff_high & z_vol>=2 & ret>0", m_eff & m_vol & m_ret)
    add_rule("eff_high & z_vp>=2 & ret>0", m_eff & m_vp & m_ret)
    add_rule("unusual_entry & cum<=2%", m_via_u & m_cum_small)
    add_rule("unusual & vol+vp & cum<=2%", m_via_u & m_vol & m_vp & m_cum_small)
    add_rule("unusual & vol>=3 & accel>0", m_via_u & (onsets["z_vol_10"] >= 3) & m_acc & m_ret)

    # stronger z
    add_rule("z_vol10>=3 & ret>0", (onsets["z_vol_10"] >= 3) & m_ret)
    add_rule("z_vp10>=3 & ret>0", (onsets["z_vp_10"] >= 3) & m_ret)
    add_rule("z_vol10>=4 & ret>0", (onsets["z_vol_10"] >= 4) & m_ret)

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    # score: prioritize winner hit-rate then plateau expectancy
    out["score"] = out["p_winner8"] * 0.6 + out["wr_trade_gt0"] * 0.25 + out["avg_trade_plateau"].clip(lower=-0.05, upper=0.2) * 5 * 0.15
    return out.sort_values(["p_winner8", "avg_trade_plateau"], ascending=False).reset_index(drop=True)


def winner_anatomy(winners: pd.DataFrame) -> dict:
    if winners.empty:
        return {}
    g = winners
    return {
        "n": int(len(g)),
        "by_session": g.groupby("session").size().astype(int).to_dict(),
        "peak_ret": {
            "mean": float(g["peak_ret"].mean()),
            "median": float(g["peak_ret"].median()),
            "p90": float(g["peak_ret"].quantile(0.9)),
        },
        "plateau_trade": {
            "avg": float(g["trade_ret_plateau"].mean()),
            "median": float(g["trade_ret_plateau"].median()),
            "wr_gt0": float((g["trade_ret_plateau"] > 0).mean()),
            "avg_hold_mins": float(g["hold_mins"].mean()),
            "capture_vs_peak": float((g["trade_ret_plateau"] / g["peak_ret"].clip(lower=1e-6)).median()),
        },
        "entry_features_mean": {
            c: float(g[c].mean())
            for c in [
                "z_vol_10",
                "z_vp_10",
                "ret_10",
                "accel_10",
                "eff_10",
                "entry_cum",
                "entry_hour",
            ]
            if c in g.columns
        },
        "entry_hour_hist": g.groupby("entry_hour").size().astype(int).to_dict(),
        "frac_entry_via_unusual": float(g["entry_via_unusual"].mean()),
    }


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    root = Path(args.root)
    months = month_range(args.start_month, args.end_month)
    symbols = list_symbols(root, args.max_symbols)
    print(
        f"[winner8] symbols={len(symbols)} peak>={args.winner_ret:.0%} "
        f"plateau={args.plateau_mins}m/{args.plateau_band:.2%}",
        flush=True,
    )

    payloads = [(str(root), sym, months, vars(args)) for sym in symbols]
    win_parts, onset_parts = [], []
    n_workers = min(8, max(1, (os.cpu_count() or 4) // 2))
    done = 0
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futs = {ex.submit(_worker, p): p[1] for p in payloads}
        for fut in as_completed(futs):
            sym = futs[fut]
            done += 1
            try:
                w, o = fut.result()
                if w is not None and not w.empty:
                    win_parts.append(w)
                if o is not None and not o.empty:
                    onset_parts.append(o)
            except Exception as e:
                print(f"[warn] {sym}: {e}", flush=True)
            if done % 20 == 0 or done == 1:
                print(
                    f"[winner8] {done}/{len(symbols)} last={sym} "
                    f"winners={sum(len(x) for x in win_parts)} onsets={sum(len(x) for x in onset_parts)}",
                    flush=True,
                )

    winners = pd.concat(win_parts, ignore_index=True) if win_parts else pd.DataFrame()
    onsets = pd.concat(onset_parts, ignore_index=True) if onset_parts else pd.DataFrame()
    winners.to_parquet(out / "winners.parquet", index=False)
    onsets.to_parquet(out / "onset_candidates.parquet", index=False)

    anatomy = {
        "all": winner_anatomy(winners),
        "pre": winner_anatomy(winners[winners["session"] == "pre"]) if len(winners) else {},
        "ah": winner_anatomy(winners[winners["session"] == "ah"]) if len(winners) else {},
    }

    rules_all = rule_table(onsets)
    rules_pre = rule_table(onsets[onsets["session"] == "pre"]) if len(onsets) else pd.DataFrame()
    rules_ah = rule_table(onsets[onsets["session"] == "ah"]) if len(onsets) else pd.DataFrame()
    rules_all.to_csv(out / "rules_all.csv", index=False)
    rules_pre.to_csv(out / "rules_pre.csv", index=False)
    rules_ah.to_csv(out / "rules_ah.csv", index=False)

    # winner vs non-winner factor contrast at onset
    contrast = []
    if len(onsets):
        for col in ["z_vol_10", "z_vp_10", "ret_10", "accel_10", "eff_10", "z_vol_5", "z_vp_5", "entry_cum", "entry_hour"]:
            if col not in onsets.columns:
                continue
            w = onsets.loc[onsets["is_winner8"] == 1, col]
            l = onsets.loc[onsets["is_winner8"] == 0, col]
            contrast.append(
                {
                    "factor": col,
                    "winner_mean": float(w.mean()) if len(w) else float("nan"),
                    "loser_mean": float(l.mean()) if len(l) else float("nan"),
                    "winner_med": float(w.median()) if len(w) else float("nan"),
                    "loser_med": float(l.median()) if len(l) else float("nan"),
                    "lift_mean": float(w.mean() - l.mean()) if len(w) and len(l) else float("nan"),
                }
            )
    contrast_df = pd.DataFrame(contrast).sort_values("lift_mean", ascending=False) if contrast else pd.DataFrame()
    if len(contrast_df):
        contrast_df.to_csv(out / "factor_contrast.csv", index=False)

    summary = {
        "experiment": "stock_ext_hours_winner8_plateau_rules",
        "config": vars(args),
        "n_winners": int(len(winners)),
        "n_onset_candidates": int(len(onsets)),
        "n_symbols_winners": int(winners["symbol"].nunique()) if len(winners) else 0,
        "winner_anatomy": anatomy,
        "top_rules_all": rules_all.head(15).to_dict(orient="records") if len(rules_all) else [],
        "top_rules_pre": rules_pre.head(10).to_dict(orient="records") if len(rules_pre) else [],
        "top_rules_ah": rules_ah.head(10).to_dict(orient="records") if len(rules_ah) else [],
        "factor_contrast": contrast_df.to_dict(orient="records") if len(contrast_df) else [],
        "notes": [
            "Winners = session peak cum-ret from open >= 8%",
            "Exit = plateau after peak (no new high for N mins within band of HWM)",
            "Rules ranked on onset candidates by P(hit 8%) and plateau-exit trade stats",
        ],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(
        json.dumps(
            {
                "n_winners": summary["n_winners"],
                "n_onsets": summary["n_onset_candidates"],
                "anatomy_all": anatomy.get("all", {}),
                "top_rules": summary["top_rules_all"][:8],
                "contrast_top": summary["factor_contrast"][:8],
            },
            indent=2,
            default=str,
        )
    )
    print(f"results -> {out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Optimize extended-hours rules for plateau-exit expectancy / drawdown.

Not hunting P(hit 8%). Entry = early unusual VP/accel (long-side).
Exit = plateau after local peak (no new high for N mins within HWM band).
Rank rules by: avg trade, win-rate, payoff, MAE, and simple day-basket account.
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
    add_causal_tod_z,
    add_window_features,
    list_symbols,
    load_1min,
    month_range,
)
from qqq_btc.tools.validate_stock_ext_hours_winner8_plateau_rules import (
    find_plateau_exit,
    session_slice,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="/mnt/s990/data/all_data/stocks_15s_parquet")
    p.add_argument("--start-month", default="2024-01")
    p.add_argument("--end-month", default="2025-06")
    p.add_argument("--max-symbols", type=int, default=0)
    p.add_argument("--lookback-days", type=int, default=20)
    p.add_argument("--z-vol", type=float, default=2.0)
    p.add_argument("--z-vp", type=float, default=2.0)
    p.add_argument("--min-ret", type=float, default=0.002)
    p.add_argument("--max-entry-cum", type=float, default=0.02, help="early only: |cum| at entry")
    p.add_argument("--plateau-mins", type=int, default=10)
    p.add_argument("--plateau-band", type=float, default=0.004)
    p.add_argument(
        "--output-dir",
        default="qqq_btc/results/stock_ext_hours_plateau_expectancy",
    )
    return p.parse_args()


def process_symbol(df_1m: pd.DataFrame, symbol: str, args: argparse.Namespace) -> pd.DataFrame:
    if df_1m.empty:
        return pd.DataFrame()
    rows = []
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
            if len(day) < 40:
                continue
            # long-side early unusual
            unusual = (
                ((day["z_vol_10"] >= args.z_vol) | (day["z_vp_10"] >= args.z_vp))
                & (day["ret_10"] >= args.min_ret)
                & (day["accel_10"].fillna(0) > 0)
                & (day["cum_ret"] >= 0)
                & (day["cum_ret"] <= args.max_entry_cum)
            ).fillna(False)
            hits = np.where(unusual.to_numpy())[0]
            if len(hits) == 0:
                continue
            entry_pos = int(hits[0])
            closes = day["close"].to_numpy(dtype=float)
            cum = day["cum_ret"].to_numpy(dtype=float)

            # local peak after entry, then plateau exit
            post = cum.copy()
            post[: entry_pos + 1] = -np.inf
            if not np.isfinite(post).any() or np.all(np.isneginf(post)):
                local_peak = entry_pos
            else:
                local_peak = int(np.nanargmax(post))
            exit_pos = find_plateau_exit(closes, local_peak, args.plateau_mins, args.plateau_band)
            entry_px = float(closes[entry_pos])
            exit_px = float(closes[exit_pos])
            if entry_px <= 0:
                continue

            path = closes[entry_pos : exit_pos + 1] / entry_px - 1.0
            trade_ret = float(exit_px / entry_px - 1.0)
            mfe = float(np.nanmax(path)) if len(path) else float("nan")
            mae = float(np.nanmin(path)) if len(path) else float("nan")
            row = day.iloc[entry_pos]
            hour = int(row["timestamp"].hour)
            bucket = (
                "04-06"
                if hour < 6
                else ("06-08" if hour < 8 else ("08-09" if hour < 16 else ("16-17" if hour < 17 else "17-20")))
            )
            rows.append(
                {
                    "symbol": symbol,
                    "session": session,
                    "date_str": date_str,
                    "entry_ts": str(row["timestamp"]),
                    "entry_hour": hour,
                    "entry_bucket": bucket,
                    "entry_cum": float(row["cum_ret"]),
                    "hold_mins": int(exit_pos - entry_pos),
                    "trade_ret": trade_ret,
                    "mfe": mfe,
                    "mae": mae,
                    "peak_after_entry": float(np.nanmax(cum[entry_pos:]) - cum[entry_pos]),
                    "z_vol_5": float(row["z_vol_5"]) if pd.notna(row["z_vol_5"]) else float("nan"),
                    "z_vol_10": float(row["z_vol_10"]) if pd.notna(row["z_vol_10"]) else float("nan"),
                    "z_vol_15": float(row["z_vol_15"]) if pd.notna(row["z_vol_15"]) else float("nan"),
                    "z_vp_5": float(row["z_vp_5"]) if pd.notna(row["z_vp_5"]) else float("nan"),
                    "z_vp_10": float(row["z_vp_10"]) if pd.notna(row["z_vp_10"]) else float("nan"),
                    "z_vp_15": float(row["z_vp_15"]) if pd.notna(row["z_vp_15"]) else float("nan"),
                    "ret_5": float(row["ret_5"]) if pd.notna(row["ret_5"]) else float("nan"),
                    "ret_10": float(row["ret_10"]) if pd.notna(row["ret_10"]) else float("nan"),
                    "ret_15": float(row["ret_15"]) if pd.notna(row["ret_15"]) else float("nan"),
                    "accel_5": float(row["accel_5"]) if pd.notna(row["accel_5"]) else float("nan"),
                    "accel_10": float(row["accel_10"]) if pd.notna(row["accel_10"]) else float("nan"),
                    "accel_15": float(row["accel_15"]) if pd.notna(row["accel_15"]) else float("nan"),
                    "eff_5": float(row["eff_5"]) if pd.notna(row["eff_5"]) else float("nan"),
                    "eff_10": float(row["eff_10"]) if pd.notna(row["eff_10"]) else float("nan"),
                    "eff_15": float(row["eff_15"]) if pd.notna(row["eff_15"]) else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def _worker(payload: tuple) -> pd.DataFrame:
    root_s, sym, months, args_dict = payload
    ns = argparse.Namespace(**args_dict)
    return process_symbol(load_1min(Path(root_s), sym, months), sym, ns)


def trade_stats(sub: pd.DataFrame, label: str) -> dict:
    if sub.empty:
        return {"label": label, "n": 0}
    r = sub["trade_ret"]
    mae = sub["mae"]
    return {
        "label": label,
        "n": int(len(sub)),
        "avg_ret": float(r.mean()),
        "median_ret": float(r.median()),
        "win_rate": float((r > 0).mean()),
        "wr_gt1pct": float((r > 0.01).mean()),
        "avg_mfe": float(sub["mfe"].mean()),
        "avg_mae": float(mae.mean()),
        "p05_mae": float(mae.quantile(0.05)),
        "payoff": float(r[r > 0].mean() / abs(r[r <= 0].mean())) if (r <= 0).any() and (r > 0).any() else float("nan"),
        "avg_hold_mins": float(sub["hold_mins"].mean()),
        "avg_peak_after": float(sub["peak_after_entry"].mean()),
    }


def day_account(sub: pd.DataFrame, frac: float = 0.1) -> dict:
    if sub.empty:
        return {"n_days": 0}
    day = sub.groupby("date_str")["trade_ret"].mean().sort_index()
    eq = np.cumprod(1.0 + frac * day.to_numpy())
    peaks = np.maximum.accumulate(np.r_[1.0, eq])[:-1]
    return {
        "n_days": int(len(day)),
        "avg_day": float(day.mean()),
        "account_ret": float(eq[-1] - 1.0),
        "max_dd": float((eq / peaks - 1.0).min()),
        "win_day_rate": float((day > 0).mean()),
    }


def score_row(st: dict, acct: dict) -> float:
    """Higher better: expectancy, WR, mild DD penalty."""
    if st.get("n", 0) < 80:
        return -1e9
    avg = st.get("avg_ret", 0.0) or 0.0
    wr = st.get("win_rate", 0.0) or 0.0
    mae = abs(st.get("avg_mae", 0.0) or 0.0)
    dd = abs(acct.get("max_dd", 0.0) or 0.0)
    # emphasize avg_ret and low adverse excursion
    return avg * 100 + wr * 0.15 - mae * 40 - dd * 20


def build_rules(trades: pd.DataFrame) -> pd.DataFrame:
    rules = []

    def add(name: str, mask: pd.Series):
        sub = trades.loc[mask]
        st = trade_stats(sub, name)
        ac = day_account(sub)
        st.update({f"acct_{k}": v for k, v in ac.items()})
        st["score"] = score_row(st, ac)
        rules.append(st)

    add("baseline", pd.Series(True, index=trades.index))

    # z thresholds
    for z in (2.0, 2.5, 3.0, 4.0):
        add(f"z_vol10>={z}", trades["z_vol_10"] >= z)
        add(f"z_vp10>={z}", trades["z_vp_10"] >= z)
        add(f"z_vol10>={z} & z_vp10>={z}", (trades["z_vol_10"] >= z) & (trades["z_vp_10"] >= z))

    # ret / accel caps (avoid chase)
    for lo, hi, tag in [(0.002, 0.006, "mild"), (0.006, 0.015, "mid"), (0.015, 0.05, "strong")]:
        add(f"ret10_{tag}", (trades["ret_10"] >= lo) & (trades["ret_10"] < hi))
        add(f"accel10_{tag}", (trades["accel_10"] >= lo) & (trades["accel_10"] < hi))

    # efficiency
    eff_med = trades["eff_10"].median()
    add("eff10>=med", trades["eff_10"] >= eff_med)
    add("eff10<med", trades["eff_10"] < eff_med)
    add("eff10>=0.6", trades["eff_10"] >= 0.6)
    add("eff10>=0.7", trades["eff_10"] >= 0.7)

    # entry cum earlyness
    add("cum<=0.5%", trades["entry_cum"] <= 0.005)
    add("cum<=1%", trades["entry_cum"] <= 0.01)
    add("cum 1-2%", (trades["entry_cum"] > 0.01) & (trades["entry_cum"] <= 0.02))

    # time buckets
    for b in sorted(trades["entry_bucket"].dropna().unique()):
        add(f"bucket={b}", trades["entry_bucket"] == b)

    # composite candidates aimed at expectancy
    mild = (trades["ret_10"] >= 0.002) & (trades["ret_10"] < 0.008)
    mid_vol = (trades["z_vol_10"] >= 2.0) & (trades["z_vol_10"] < 4.0)
    high_eff = trades["eff_10"] >= 0.55
    low_mae_proxy = trades["ret_10"] < 0.01  # smaller ignition
    add("mild_ret & mid_vol & eff>=0.55", mild & mid_vol & high_eff)
    add("mild_ret & z_vp>=2 & eff>=0.55", mild & (trades["z_vp_10"] >= 2) & high_eff)
    add("mild_ret & z_vol>=2 & cum<=1%", mild & (trades["z_vol_10"] >= 2) & (trades["entry_cum"] <= 0.01))
    add("mild_ret & bucket=04-06", mild & (trades["entry_bucket"] == "04-06"))
    add("mild_ret & bucket=06-08", mild & (trades["entry_bucket"] == "06-08"))
    add("mild_ret & bucket=16-17", mild & (trades["entry_bucket"] == "16-17"))
    add("z_vol 2-3 & ret mild & eff>=0.6", mid_vol & mild & (trades["eff_10"] >= 0.6))
    add("z_vp>=2 & ret mild & accel mild", mild & (trades["z_vp_10"] >= 2) & (trades["accel_10"] < 0.008))
    add("avoid_strong_ret", trades["ret_10"] < 0.01)
    add("avoid_strong_ret & z_vol>=2", (trades["ret_10"] < 0.01) & (trades["z_vol_10"] >= 2))
    add("avoid_strong_ret & eff>=0.6", (trades["ret_10"] < 0.01) & (trades["eff_10"] >= 0.6))
    add(
        "core: mild+vol2-4+eff0.55+cum<=1%",
        mild & mid_vol & high_eff & (trades["entry_cum"] <= 0.01),
    )
    add(
        "core2: mild+vp>=2+eff0.6+cum<=1%",
        mild & (trades["z_vp_10"] >= 2) & (trades["eff_10"] >= 0.6) & (trades["entry_cum"] <= 0.01),
    )
    add(
        "core3: avoid_strong + vol>=2 + eff>=0.55 + hour<=6",
        low_mae_proxy & (trades["z_vol_10"] >= 2) & high_eff & (trades["entry_hour"] <= 6),
    )

    out = pd.DataFrame(rules)
    return out.sort_values("score", ascending=False).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    root = Path(args.root)
    months = month_range(args.start_month, args.end_month)
    symbols = list_symbols(root, args.max_symbols)
    print(f"[plateau-exp] symbols={len(symbols)} early_cum<={args.max_entry_cum} exit=plateau", flush=True)

    payloads = [(str(root), sym, months, vars(args)) for sym in symbols]
    parts = []
    n_workers = min(8, max(1, (os.cpu_count() or 4) // 2))
    done = 0
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futs = {ex.submit(_worker, p): p[1] for p in payloads}
        for fut in as_completed(futs):
            sym = futs[fut]
            done += 1
            try:
                tr = fut.result()
                if tr is not None and not tr.empty:
                    parts.append(tr)
            except Exception as e:
                print(f"[warn] {sym}: {e}", flush=True)
            if done % 20 == 0 or done == 1:
                print(f"[plateau-exp] {done}/{len(symbols)} last={sym} n={sum(len(x) for x in parts)}", flush=True)

    trades = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    trades.to_parquet(out / "trades.parquet", index=False)

    summary_blocks = {}
    all_rules = {}
    for session in ("all", "pre", "ah"):
        sub = trades if session == "all" else trades[trades["session"] == session]
        rules = build_rules(sub) if len(sub) else pd.DataFrame()
        if len(rules):
            rules.to_csv(out / f"rules_{session}.csv", index=False)
        all_rules[session] = rules.head(15).to_dict(orient="records") if len(rules) else []
        summary_blocks[session] = {
            "baseline": trade_stats(sub, "baseline"),
            "account_baseline": day_account(sub),
            "top_rules": all_rules[session][:8],
        }

    # recommended = best score on pre and ah separately with n>=100
    rec = {}
    for session in ("pre", "ah"):
        fp = out / f"rules_{session}.csv"
        if not fp.exists():
            continue
        r = pd.read_csv(fp)
        r = r[r["n"] >= 100].sort_values("score", ascending=False)
        rec[session] = r.head(5).to_dict(orient="records") if len(r) else []

    summary = {
        "experiment": "stock_ext_hours_plateau_expectancy",
        "config": vars(args),
        "n_trades": int(len(trades)),
        "n_symbols": int(trades["symbol"].nunique()) if len(trades) else 0,
        "by_session": trades.groupby("session").size().astype(int).to_dict() if len(trades) else {},
        "blocks": summary_blocks,
        "recommended_n>=100": rec,
        "notes": [
            "Objective: plateau-exit expectancy and drawdown, not P(hit 8%)",
            "Entry: first long-side unusual VP/accel with cum<=max_entry_cum",
            "Exit: local peak then plateau (no new high N mins within band)",
            "score = 100*avg_ret + 0.15*WR - 40*|avg_mae| - 20*|max_dd|",
        ],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(
        json.dumps(
            {
                "n_trades": summary["n_trades"],
                "baseline_all": summary_blocks["all"]["baseline"],
                "recommended": rec,
            },
            indent=2,
            default=str,
        )
    )
    print(f"results -> {out}")


if __name__ == "__main__":
    main()

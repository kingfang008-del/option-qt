#!/usr/bin/env python3
"""High-elasticity subset + mild rules + plateau exit, walk-forward.

Universe (causal): each month pick symbols with most extended-hours big moves
in prior train window (peak>=5% or winner8), top-K elastic names.
Rules: mild ignition only; exit plateau within pre/AH session.
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


RULES = {
    "pre_mild_0406": {
        "session": "pre",
        "bucket": "04-06",
        "z_vol_lo": 2.0,
        "z_vol_hi": 4.0,
        "z_vp": 2.0,
        "ret_lo": 0.002,
        "ret_hi": 0.008,
        "max_cum": 0.01,
        "min_eff": 0.55,
    },
    "pre_mild_z2_cum1": {
        "session": "pre",
        "bucket": None,
        "z_vol_lo": 2.0,
        "z_vol_hi": 99.0,
        "z_vp": 2.0,
        "ret_lo": 0.002,
        "ret_hi": 0.008,
        "max_cum": 0.01,
        "min_eff": 0.0,
    },
    "pre_core_mild": {
        "session": "pre",
        "bucket": None,
        "z_vol_lo": 2.0,
        "z_vol_hi": 4.0,
        "z_vp": 2.0,
        "ret_lo": 0.002,
        "ret_hi": 0.008,
        "max_cum": 0.01,
        "min_eff": 0.55,
    },
    "ah_mild_z2_cum1": {
        "session": "ah",
        "bucket": None,
        "z_vol_lo": 2.0,
        "z_vol_hi": 99.0,
        "z_vp": 2.0,
        "ret_lo": 0.002,
        "ret_hi": 0.008,
        "max_cum": 0.01,
        "min_eff": 0.0,
    },
    "ah_mild_eff": {
        "session": "ah",
        "bucket": None,
        "z_vol_lo": 2.0,
        "z_vol_hi": 4.0,
        "z_vp": 2.0,
        "ret_lo": 0.002,
        "ret_hi": 0.008,
        "max_cum": 0.01,
        "min_eff": 0.55,
    },
    "ah_mild_cum2": {
        "session": "ah",
        "bucket": None,
        "z_vol_lo": 2.0,
        "z_vol_hi": 99.0,
        "z_vp": 2.0,
        "ret_lo": 0.002,
        "ret_hi": 0.008,
        "max_cum": 0.02,
        "min_eff": 0.0,
    },
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="/mnt/s990/data/all_data/stocks_15s_parquet")
    p.add_argument("--start-month", default="2024-01")
    p.add_argument("--end-month", default="2025-06")
    p.add_argument("--max-symbols", type=int, default=0)
    p.add_argument("--lookback-days", type=int, default=20)
    p.add_argument("--train-months", type=int, default=3)
    p.add_argument("--elastic-top-k", type=int, default=40)
    p.add_argument("--big-move-ret", type=float, default=0.05, help="peak session ret for elastic score")
    p.add_argument("--plateau-mins", type=int, default=10)
    p.add_argument("--plateau-band", type=float, default=0.004)
    p.add_argument(
        "--output-dir",
        default="qqq_btc/results/stock_elastic_mild_plateau_wf",
    )
    return p.parse_args()


def bucket_of(hour: int) -> str:
    if hour < 6:
        return "04-06"
    if hour < 8:
        return "06-08"
    if hour < 16:
        return "08-09"
    if hour < 17:
        return "16-17"
    return "17-20"


def match_rule(row: pd.Series, cfg: dict) -> bool:
    if row.get("session") != cfg["session"]:
        return False
    if cfg.get("bucket") and row.get("entry_bucket") != cfg["bucket"]:
        return False
    zv = row.get("z_vol_10", np.nan)
    if not (cfg["z_vol_lo"] <= zv < cfg["z_vol_hi"]):
        return False
    if not (row.get("z_vol_10", 0) >= cfg["z_vol_lo"] or row.get("z_vp_10", 0) >= cfg["z_vp"]):
        return False
    r = row.get("ret_10", np.nan)
    if not (cfg["ret_lo"] <= r < cfg["ret_hi"]):
        return False
    if row.get("entry_cum", 99) > cfg["max_cum"]:
        return False
    if cfg["min_eff"] > 0 and row.get("eff_10", 0) < cfg["min_eff"]:
        return False
    return True


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
        sess["month"] = sess["timestamp"].dt.strftime("%Y-%m")

        for date_str, day in sess.groupby("date_str", sort=True):
            day = day.sort_values("timestamp").reset_index(drop=True)
            if len(day) < 25:
                continue
            unusual = (
                ((day["z_vol_10"] >= 2.0) | (day["z_vp_10"] >= 2.0))
                & (day["ret_10"] >= 0.002)
                & (day["accel_10"].fillna(0) > 0)
                & (day["cum_ret"] >= 0)
                & (day["cum_ret"] <= 0.02)
            ).fillna(False)
            if not unusual.any():
                continue
            entry_pos = int(np.where(unusual.to_numpy())[0][0])
            row = day.iloc[entry_pos]
            closes = day["close"].to_numpy(float)
            cum = day["cum_ret"].to_numpy(float)
            peak_idx = int(np.nanargmax(cum))
            exit_pos = find_plateau_exit(closes, peak_idx, args.plateau_mins, args.plateau_band)
            if exit_pos <= entry_pos:
                exit_pos = min(entry_pos + 5, len(closes) - 1)
            entry_px = float(closes[entry_pos])
            if entry_px <= 0:
                continue
            trade_ret = float(closes[exit_pos] / entry_px - 1.0)
            peak_ret = float(np.nanmax(cum))
            hour = int(row["timestamp"].hour)
            base = {
                "symbol": symbol,
                "session": session,
                "date_str": date_str,
                "month": str(row["month"]),
                "entry_bucket": bucket_of(hour),
                "entry_cum": float(row["cum_ret"]),
                "peak_ret": peak_ret,
                "big_move": int(peak_ret >= args.big_move_ret),
                "trade_ret": trade_ret,
                "hold_mins": int(exit_pos - entry_pos),
                "z_vol_10": float(row["z_vol_10"]) if pd.notna(row["z_vol_10"]) else float("nan"),
                "z_vp_10": float(row["z_vp_10"]) if pd.notna(row["z_vp_10"]) else float("nan"),
                "ret_10": float(row["ret_10"]) if pd.notna(row["ret_10"]) else float("nan"),
                "eff_10": float(row["eff_10"]) if pd.notna(row["eff_10"]) else float("nan"),
            }
            for rname, cfg in RULES.items():
                if match_rule(pd.Series({**base, "session": session}), cfg):
                    rec = base.copy()
                    rec["rule"] = rname
                    rows.append(rec)
    return pd.DataFrame(rows)


def _worker(payload: tuple) -> pd.DataFrame:
    root_s, sym, months, args_dict = payload
    return process_symbol(load_1min(Path(root_s), sym, months), sym, argparse.Namespace(**args_dict))


def elastic_universe(train: pd.DataFrame, top_k: int) -> set[str]:
    if train.empty:
        return set()
    score = train.groupby("symbol").agg(
        big_moves=("big_move", "sum"),
        n=("trade_ret", "size"),
        avg_peak=("peak_ret", "mean"),
    )
    score["score"] = score["big_moves"] * 2 + score["avg_peak"].clip(0, 0.2) * 10
    score = score.sort_values("score", ascending=False)
    return set(score.head(top_k).index.tolist())


def pick_rules(train: pd.DataFrame, syms: set[str]) -> dict[str, str]:
    sub = train[train["symbol"].isin(syms)]
    out = {"pre": "pre_mild_0406", "ah": "ah_mild_z2_cum1"}
    for sess in ("pre", "ah"):
        best, best_sc = None, -1e9
        for rname in [k for k, v in RULES.items() if v["session"] == sess]:
            g = sub[(sub["session"] == sess) & (sub["rule"] == rname)]
            if len(g) < 8:
                continue
            avg = float(g["trade_ret"].mean())
            wr = float((g["trade_ret"] > 0).mean())
            sc = avg * 100 + wr * 0.15
            if sc > best_sc:
                best_sc, best = sc, rname
        if best:
            out[sess] = best
    return out


def summarize(trades: pd.DataFrame, label: str) -> dict:
    if trades.empty:
        return {"label": label, "n": 0}
    r = trades["trade_ret"]
    return {
        "label": label,
        "n": int(len(trades)),
        "avg_ret": float(r.mean()),
        "median_ret": float(r.median()),
        "win_rate": float((r > 0).mean()),
        "wr_ge_2pct": float((r >= 0.02).mean()),
        "wr_ge_3pct": float((r >= 0.03).mean()),
        "avg_hold_mins": float(trades["hold_mins"].mean()),
    }


def day_account(trades: pd.DataFrame, frac: float = 0.5) -> dict:
    if trades.empty:
        return {"n_days": 0, "account_ret": 0.0, "max_dd": 0.0}
    day = trades.groupby("date_str")["trade_ret"].mean().sort_index()
    eq = np.cumprod(1.0 + frac * day.to_numpy())
    peaks = np.maximum.accumulate(np.r_[1.0, eq])[:-1]
    return {
        "n_days": int(len(day)),
        "avg_day": float(day.mean()),
        "account_ret": float(eq[-1] - 1.0),
        "max_dd": float((eq / peaks - 1.0).min()),
        "win_day_rate": float((day > 0).mean()),
        "position_frac": frac,
    }


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    months = month_range(args.start_month, args.end_month)
    symbols = list_symbols(Path(args.root), args.max_symbols)
    print(f"[elastic-wf] load {len(symbols)} symbols", flush=True)

    payloads = [(str(args.root), s, months, vars(args)) for s in symbols]
    parts = []
    with ProcessPoolExecutor(max_workers=min(8, max(1, (os.cpu_count() or 4) // 2))) as ex:
        futs = {ex.submit(_worker, p): p[1] for p in payloads}
        for i, fut in enumerate(as_completed(futs), 1):
            tr = fut.result()
            if tr is not None and not tr.empty:
                parts.append(tr)
            if i % 50 == 0:
                print(f"[elastic-wf] {i}/{len(symbols)}", flush=True)

    all_tr = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    all_tr.to_parquet(out / "all_candidates.parquet", index=False)

    # fixed elastic list from first train window (reference)
    fixed_syms = elastic_universe(
        all_tr[all_tr["month"].isin(months[: args.train_months])], args.elastic_top_k
    )
    fixed_pre = all_tr[
        (all_tr["symbol"].isin(fixed_syms))
        & (all_tr["session"] == "pre")
        & (all_tr["rule"] == "pre_mild_0406")
    ]
    fixed_ah = all_tr[
        (all_tr["symbol"].isin(fixed_syms))
        & (all_tr["session"] == "ah")
        & (all_tr["rule"] == "ah_mild_z2_cum1")
    ]
    fixed_all = pd.concat([fixed_pre, fixed_ah], ignore_index=True)

    uniq = sorted(all_tr["month"].unique()) if len(all_tr) else []
    wf_trades = []
    wf_meta = []
    for i in range(args.train_months, len(uniq)):
        test_m = uniq[i]
        train_ms = uniq[i - args.train_months : i]
        train = all_tr[all_tr["month"].isin(train_ms)]
        test = all_tr[all_tr["month"] == test_m]
        syms = elastic_universe(train, args.elastic_top_k)
        picks = pick_rules(train, syms)
        for sess in ("pre", "ah"):
            sub = test[
                (test["symbol"].isin(syms))
                & (test["session"] == sess)
                & (test["rule"] == picks[sess])
            ].copy()
            if sub.empty:
                continue
            sub["wf_month"] = test_m
            sub["picked_rule"] = picks[sess]
            sub["elastic_universe"] = ",".join(sorted(syms)[:15]) + ("..." if len(syms) > 15 else "")
            wf_trades.append(sub)
            wf_meta.append(
                {
                    "month": test_m,
                    "session": sess,
                    "picked_rule": picks[sess],
                    "n_elastic": len(syms),
                    "n_trades": len(sub),
                    "avg_ret": float(sub["trade_ret"].mean()),
                    "wr": float((sub["trade_ret"] > 0).mean()),
                }
            )

    wf = pd.concat(wf_trades, ignore_index=True) if wf_trades else pd.DataFrame()
    wf.to_parquet(out / "wf_trades.parquet", index=False)

    summary = {
        "experiment": "stock_elastic_mild_plateau_wf",
        "config": vars(args),
        "fixed_elastic_universe_sample": sorted(fixed_syms)[:25],
        "n_fixed_elastic_symbols": len(fixed_syms),
        "fixed_pre_mild_0406": {
            "stats": summarize(fixed_pre, "fixed_pre"),
            "account_50pct": day_account(fixed_pre, 0.5),
            "account_10pct": day_account(fixed_pre, 0.1),
        },
        "fixed_ah_mild_z2_cum1": {
            "stats": summarize(fixed_ah, "fixed_ah"),
            "account_50pct": day_account(fixed_ah, 0.5),
            "account_10pct": day_account(fixed_ah, 0.1),
        },
        "fixed_combined": {
            "stats": summarize(fixed_all, "fixed_combined"),
            "account_50pct": day_account(fixed_all, 0.5),
            "account_10pct": day_account(fixed_all, 0.1),
        },
        "walk_forward_oos": {
            "stats": summarize(wf, "wf_all"),
            "account_50pct": day_account(wf, 0.5),
            "account_10pct": day_account(wf, 0.1),
            "by_session": {
                "pre": summarize(wf[wf["session"] == "pre"], "wf_pre") if len(wf) else {},
                "ah": summarize(wf[wf["session"] == "ah"], "wf_ah") if len(wf) else {},
            },
            "monthly": wf_meta,
        },
        "notes": [
            "Elastic universe: causal top-K by prior big moves (peak>=5%)",
            "Pre default: mild_0406; AH default: mild_z2_cum1",
            "Exit: plateau within same extended session",
        ],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    pd.DataFrame(wf_meta).to_csv(out / "wf_monthly.csv", index=False)

    print(json.dumps(
        {
            "fixed_combined": summary["fixed_combined"],
            "wf_oos": summary["walk_forward_oos"]["stats"],
            "wf_acct50": summary["walk_forward_oos"]["account_50pct"],
            "wf_by_session": summary["walk_forward_oos"]["by_session"],
        },
        indent=2,
        default=str,
    ))
    print(f"results -> {out}")


if __name__ == "__main__":
    main()

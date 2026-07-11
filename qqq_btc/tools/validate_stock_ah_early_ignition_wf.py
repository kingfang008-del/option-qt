#!/usr/bin/env python3
"""Walk-forward: AH strong early ignition + cum<=2% + plateau exit.

Target: capture +3~5% moves in after-hours, not P(hit 8%).
Walk-forward: rolling train months pick rule params, test on next month (causal).
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
    p.add_argument("--train-months", type=int, default=3)
    p.add_argument("--max-entry-cum", type=float, default=0.02)
    p.add_argument("--early-window-mins", type=int, default=30)
    p.add_argument("--plateau-mins", type=int, default=10)
    p.add_argument("--plateau-band", type=float, default=0.004)
    p.add_argument(
        "--output-dir",
        default="qqq_btc/results/stock_ah_early_ignition_wf",
    )
    return p.parse_args()


RULE_GRID = [
    {"name": "strong_z3_cum1", "z_vol": 3.0, "z_vp": 3.0, "min_ret": 0.003, "max_cum": 0.01},
    {"name": "strong_z3_cum2", "z_vol": 3.0, "z_vp": 3.0, "min_ret": 0.003, "max_cum": 0.02},
    {"name": "strong_z25_cum2", "z_vol": 2.5, "z_vp": 2.5, "min_ret": 0.003, "max_cum": 0.02},
    {"name": "mild_z2_cum1", "z_vol": 2.0, "z_vp": 2.0, "min_ret": 0.002, "max_cum": 0.01},
    {"name": "mild_z2_cum2", "z_vol": 2.0, "z_vp": 2.0, "min_ret": 0.002, "max_cum": 0.02},
    {"name": "vp_heavy_z4", "z_vol": 2.5, "z_vp": 4.0, "min_ret": 0.003, "max_cum": 0.02},
    {"name": "vol_heavy_z4", "z_vol": 4.0, "z_vp": 2.5, "min_ret": 0.003, "max_cum": 0.02},
]


def process_symbol(df_1m: pd.DataFrame, symbol: str, args: argparse.Namespace) -> pd.DataFrame:
    if df_1m.empty:
        return pd.DataFrame()
    sess = session_slice(df_1m, "ah")
    if sess.empty:
        return pd.DataFrame()
    sess = add_window_features(sess, (5, 10, 15))
    sess["sess_open"] = sess.groupby("date_str")["close"].transform("first")
    sess["cum_ret"] = sess["close"] / sess["sess_open"] - 1.0
    z_cols = [f"vol_{w}" for w in (5, 10, 15)] + [f"vp_{w}" for w in (5, 10, 15)]
    sess = add_causal_tod_z(sess, z_cols, args.lookback_days)
    sess["month"] = sess["timestamp"].dt.strftime("%Y-%m")

    rows = []
    for date_str, day in sess.groupby("date_str", sort=True):
        day = day.sort_values("timestamp").reset_index(drop=True)
        if len(day) < 25:
            continue
        closes = day["close"].to_numpy(float)
        cum = day["cum_ret"].to_numpy(float)
        peak_idx = int(np.nanargmax(cum))
        peak_ret = float(cum[peak_idx])
        t0 = day.iloc[0]["timestamp"]
        mins = (day["timestamp"] - t0).dt.total_seconds() / 60.0
        early = day.loc[mins <= args.early_window_mins]
        if len(early) < 5:
            continue

        for rule in RULE_GRID:
            unusual = (
                ((early["z_vol_10"] >= rule["z_vol"]) | (early["z_vp_10"] >= rule["z_vp"]))
                & (early["ret_10"] >= rule["min_ret"])
                & (early["accel_10"].fillna(0) > 0)
                & (early["cum_ret"] >= 0)
                & (early["cum_ret"] <= rule["max_cum"])
            ).fillna(False)
            if not unusual.any():
                continue
            entry_pos = int(np.where(unusual.to_numpy())[0][0])
            entry_row = day.iloc[entry_pos]
            if float(entry_row["cum_ret"]) > args.max_entry_cum:
                continue
            exit_pos = find_plateau_exit(closes, peak_idx, args.plateau_mins, args.plateau_band)
            if exit_pos <= entry_pos:
                exit_pos = min(entry_pos + 5, len(closes) - 1)
            entry_px = float(closes[entry_pos])
            exit_px = float(closes[exit_pos])
            if entry_px <= 0:
                continue
            trade_ret = exit_px / entry_px - 1.0
            rows.append(
                {
                    "symbol": symbol,
                    "date_str": date_str,
                    "month": str(entry_row["month"]),
                    "rule": rule["name"],
                    "entry_cum": float(entry_row["cum_ret"]),
                    "entry_min": float(mins.iloc[entry_pos]),
                    "peak_ret": peak_ret,
                    "trade_ret": trade_ret,
                    "hold_mins": int(exit_pos - entry_pos),
                    "hit_3pct": int(trade_ret >= 0.03),
                    "hit_5pct": int(trade_ret >= 0.05),
                    "z_vol_10": float(entry_row["z_vol_10"]) if pd.notna(entry_row["z_vol_10"]) else float("nan"),
                    "z_vp_10": float(entry_row["z_vp_10"]) if pd.notna(entry_row["z_vp_10"]) else float("nan"),
                    "ret_10": float(entry_row["ret_10"]) if pd.notna(entry_row["ret_10"]) else float("nan"),
                    "eff_10": float(entry_row["eff_10"]) if pd.notna(entry_row["eff_10"]) else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def _worker(payload: tuple) -> pd.DataFrame:
    root_s, sym, months, args_dict = payload
    ns = argparse.Namespace(**args_dict)
    return process_symbol(load_1min(Path(root_s), sym, months), sym, ns)


def pick_rule(train: pd.DataFrame) -> str:
    """Select rule on train by score favoring avg_ret, wr>=3%, low n penalty."""
    if train.empty:
        return RULE_GRID[1]["name"]
    best = None
    best_score = -1e9
    for rule, g in train.groupby("rule"):
        if len(g) < 15:
            continue
        avg = float(g["trade_ret"].mean())
        wr = float((g["trade_ret"] > 0).mean())
        wr3 = float((g["trade_ret"] >= 0.03).mean())
        score = avg * 100 + wr * 0.1 + wr3 * 0.5 - 0.001 * max(0, 80 - len(g))
        if score > best_score:
            best_score = score
            best = rule
    return best or RULE_GRID[1]["name"]


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
        "wr_ge_3pct": float((r >= 0.03).mean()),
        "wr_ge_5pct": float((r >= 0.05).mean()),
        "p95": float(r.quantile(0.95)),
        "avg_peak": float(trades["peak_ret"].mean()),
        "avg_hold_mins": float(trades["hold_mins"].mean()),
    }


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    months = month_range(args.start_month, args.end_month)
    symbols = list_symbols(Path(args.root), args.max_symbols)
    print(f"[ah-wf] symbols={len(symbols)} AH early ignition walk-forward", flush=True)

    payloads = [(str(args.root), s, months, vars(args)) for s in symbols]
    parts = []
    with ProcessPoolExecutor(max_workers=min(8, max(1, (os.cpu_count() or 4) // 2))) as ex:
        futs = {ex.submit(_worker, p): p[1] for p in payloads}
        done = 0
        for fut in as_completed(futs):
            done += 1
            tr = fut.result()
            if tr is not None and not tr.empty:
                parts.append(tr)
            if done % 50 == 0:
                print(f"[ah-wf] load {done}/{len(symbols)}", flush=True)

    all_tr = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    all_tr.to_parquet(out / "all_trades.parquet", index=False)

    # fixed rule: strong_z3_cum2 (user thesis)
    fixed = all_tr[all_tr["rule"] == "strong_z3_cum2"].copy()
    fixed_stats = summarize(fixed, "fixed_strong_z3_cum2")
    fixed_acct_50 = day_account(fixed, 0.5)
    fixed_acct_10 = day_account(fixed, 0.1)

    # walk-forward by month
    uniq_months = sorted(all_tr["month"].unique()) if len(all_tr) else []
    wf_rows = []
    wf_trades = []
    for i in range(args.train_months, len(uniq_months)):
        test_m = uniq_months[i]
        train_ms = uniq_months[i - args.train_months : i]
        train = all_tr[all_tr["month"].isin(train_ms)]
        test = all_tr[all_tr["month"] == test_m]
        if test.empty:
            continue
        picked = pick_rule(train)
        sub = test[test["rule"] == picked].copy()
        sub["picked_rule"] = picked
        sub["train_months"] = ",".join(train_ms)
        wf_trades.append(sub)
        st = summarize(sub, f"wf_{test_m}")
        st["picked_rule"] = picked
        st["train_months"] = train_ms
        wf_rows.append(st)

    wf = pd.concat(wf_trades, ignore_index=True) if wf_trades else pd.DataFrame()
    wf.to_parquet(out / "wf_trades.parquet", index=False)
    wf_stats = summarize(wf, "walk_forward_oos")
    wf_acct_50 = day_account(wf, 0.5)
    wf_acct_10 = day_account(wf, 0.1)

    # monthly oos table
    monthly = []
    if not wf.empty:
        for m, g in wf.groupby("month"):
            monthly.append(
                {
                    "month": m,
                    "picked_rule": g["picked_rule"].iloc[0],
                    "n": int(len(g)),
                    "avg_ret": float(g["trade_ret"].mean()),
                    "wr": float((g["trade_ret"] > 0).mean()),
                    "wr_ge_3pct": float((g["trade_ret"] >= 0.03).mean()),
                }
            )

    # rule leaderboard in-sample full period (reference only)
    lb = []
    for rule, g in all_tr.groupby("rule"):
        lb.append(summarize(g, rule))
    lb_df = pd.DataFrame(lb).sort_values("avg_ret", ascending=False)
    lb_df.to_csv(out / "rule_leaderboard.csv", index=False)

    summary = {
        "experiment": "stock_ah_early_ignition_walkforward",
        "config": vars(args),
        "n_all_trades": int(len(all_tr)),
        "n_symbols": int(all_tr["symbol"].nunique()) if len(all_tr) else 0,
        "fixed_rule": {
            "name": "strong_z3_cum2",
            "stats": fixed_stats,
            "account_50pct": fixed_acct_50,
            "account_10pct": fixed_acct_10,
        },
        "walk_forward": {
            "stats": wf_stats,
            "account_50pct": wf_acct_50,
            "account_10pct": wf_acct_10,
            "monthly_oos": monthly,
            "rule_picks": {r["label"].replace("wf_", ""): r["picked_rule"] for r in wf_rows},
        },
        "rule_leaderboard_top5": lb_df.head(5).to_dict(orient="records") if len(lb_df) else [],
        "notes": [
            "Session: after-hours only (16:00-20:00 ET)",
            "Entry: first strong VP/vol ignition within first 30 min, cum<=max_cum",
            "Exit: plateau within same AH session",
            "WF: pick rule on prior train_months, trade next month OOS",
        ],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    pd.DataFrame(wf_rows).to_csv(out / "wf_monthly_stats.csv", index=False)
    pd.DataFrame(monthly).to_csv(out / "wf_monthly_oos.csv", index=False)

    print(json.dumps(
        {
            "fixed": {"stats": fixed_stats, "acct50": fixed_acct_50},
            "wf_oos": {"stats": wf_stats, "acct50": wf_acct_50},
            "monthly": monthly[-6:],
        },
        indent=2,
        default=str,
    ))
    print(f"results -> {out}")


if __name__ == "__main__":
    main()

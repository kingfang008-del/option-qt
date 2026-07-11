#!/usr/bin/env python3
"""Onset entry + state split (early continue / extended fade).

Not chase-high: enter at first unusual VP/accel while the session move is still small.
State:
  early     — onset with small session cum-ret & modest window ret → continue to 09:15
  extended  — unusual VP but already ran a lot → fade to 09:15
  router    — early continue + extended fade

Exit: 09:15 ET (AH = next day 09:15). Universe: full stocks_15s_parquet.
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
    daily_account,
    list_symbols,
    load_1min,
    month_range,
    next_trading_date,
    ols_fit,
    px_at,
    session_mask,
    summarize,
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
    p.add_argument("--min-ret", type=float, default=0.002, help="min |window ret| to arm onset")
    p.add_argument(
        "--max-early-ret",
        type=float,
        default=0.010,
        help="|window ret| above this at first hit => extended (already ran)",
    )
    p.add_argument(
        "--max-early-cum",
        type=float,
        default=0.008,
        help="|session cum ret| above this at first hit => extended",
    )
    p.add_argument(
        "--output-dir",
        default="qqq_btc/results/stock_ext_hours_onset_state",
    )
    return p.parse_args()


def unusual_mask(day: pd.DataFrame, args: argparse.Namespace, w: int = 10) -> pd.Series:
    z_vol = day[f"z_vol_{w}"]
    z_vp = day[f"z_vp_{w}"]
    ret = day[f"ret_{w}"]
    accel = day[f"accel_{w}"]
    unusual = (z_vol >= args.z_vol) | (z_vp >= args.z_vp)
    return (
        unusual.fillna(False)
        & (ret.abs() >= args.min_ret)
        & (np.sign(accel.fillna(0.0)) == np.sign(ret.fillna(0.0)))
        & (ret.fillna(0.0) != 0)
    )


def classify_state(cum_ret: float, win_ret: float, args: argparse.Namespace) -> str:
    """early = 拉升初期; extended = 已经走出一截再触发."""
    if abs(cum_ret) > args.max_early_cum or abs(win_ret) > args.max_early_ret:
        return "extended"
    return "early"


def process_symbol(df_1m: pd.DataFrame, symbol: str, args: argparse.Namespace) -> pd.DataFrame:
    if df_1m.empty:
        return pd.DataFrame()
    dates = sorted(df_1m["date_str"].unique())
    trades = []

    for session in ("pre", "ah"):
        if session == "pre":
            mask = (df_1m["tod"] >= pd.Timestamp("04:00").time()) & (
                df_1m["tod"] < pd.Timestamp("09:30").time()
            )
        else:
            mask = (df_1m["tod"] >= pd.Timestamp("16:00").time()) & (
                df_1m["tod"] < pd.Timestamp("20:00").time()
            )
        sess = df_1m.loc[mask].copy()
        if sess.empty:
            continue
        sess = add_window_features(sess, (5, 10, 15))
        # session cum ret from first bar of that session/day
        sess["sess_open"] = sess.groupby("date_str")["close"].transform("first")
        sess["cum_ret"] = sess["close"] / sess["sess_open"] - 1.0
        z_cols = [f"vol_{w}" for w in (5, 10, 15)] + [f"vp_{w}" for w in (5, 10, 15)]
        sess = add_causal_tod_z(sess, z_cols, args.lookback_days)

        for date_str, day_all in sess.groupby("date_str", sort=True):
            day = day_all.loc[session_mask(day_all["tod"], session)]
            if len(day) < 20:
                continue
            # prefer 10m window for onset; fallback 5 then 15
            hit_row = None
            hit_w = None
            for w in (10, 5, 15):
                m = unusual_mask(day, args, w=w)
                if m.any():
                    hit_row = day.loc[m].iloc[0]
                    hit_w = w
                    break
            if hit_row is None:
                continue

            win_ret = float(hit_row[f"ret_{hit_w}"])
            cum_ret = float(hit_row["cum_ret"])
            state = classify_state(cum_ret, win_ret, args)
            direction_cont = float(np.sign(win_ret))
            if direction_cont == 0:
                continue

            if session == "pre":
                exit_date = date_str
            else:
                exit_date = next_trading_date(dates, date_str)
                if exit_date is None:
                    continue
            exit_px = px_at(df_1m, exit_date, "09:15")
            entry_px = float(hit_row["close"])
            if not np.isfinite(exit_px) or entry_px <= 0:
                continue

            raw_fwd = exit_px / entry_px - 1.0  # unsigned path
            # continue = follow ignition dir; fade = opposite
            fwd_continue = direction_cont * raw_fwd
            fwd_fade = -direction_cont * raw_fwd
            # router: early continue, extended fade
            fwd_router = fwd_continue if state == "early" else fwd_fade

            hour = int(hit_row["timestamp"].hour)
            bucket = "04-06" if hour < 6 else ("06-08" if hour < 8 else ("08-09" if hour < 16 else "16-20"))

            trades.append(
                {
                    "symbol": symbol,
                    "session": session,
                    "date_str": date_str,
                    "exit_date": exit_date,
                    "entry_ts": str(hit_row["timestamp"]),
                    "entry_hour_bucket": bucket,
                    "hit_w": hit_w,
                    "state": state,
                    "direction_cont": direction_cont,
                    "cum_ret": cum_ret,
                    "win_ret": win_ret,
                    "z_vol": float(hit_row[f"z_vol_{hit_w}"]) if pd.notna(hit_row[f"z_vol_{hit_w}"]) else float("nan"),
                    "z_vp": float(hit_row[f"z_vp_{hit_w}"]) if pd.notna(hit_row[f"z_vp_{hit_w}"]) else float("nan"),
                    "accel": float(hit_row[f"accel_{hit_w}"]) if pd.notna(hit_row[f"accel_{hit_w}"]) else float("nan"),
                    "eff": float(hit_row[f"eff_{hit_w}"]) if pd.notna(hit_row[f"eff_{hit_w}"]) else float("nan"),
                    "fwd_continue": fwd_continue,
                    "fwd_fade": fwd_fade,
                    "fwd_router": fwd_router,
                }
            )
    return pd.DataFrame(trades)


def _worker(payload: tuple) -> pd.DataFrame:
    root_s, sym, months, args_dict = payload
    ns = argparse.Namespace(**args_dict)
    df = load_1min(Path(root_s), sym, months)
    return process_symbol(df, sym, ns)


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    root = Path(args.root)
    months = month_range(args.start_month, args.end_month)
    symbols = list_symbols(root, args.max_symbols)
    print(
        f"[onset-state] symbols={len(symbols)} early_cum<{args.max_early_cum} "
        f"early_win<{args.max_early_ret} exit=09:15",
        flush=True,
    )

    payloads = [(str(root), sym, months, vars(args)) for sym in symbols]
    all_tr = []
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
                    all_tr.append(tr)
            except Exception as e:
                print(f"[warn] {sym}: {e}", flush=True)
            if done % 20 == 0 or done == 1:
                n = sum(len(x) for x in all_tr)
                print(f"[onset-state] {done}/{len(symbols)} last={sym} cum_trades={n}", flush=True)

    trades = pd.concat(all_tr, ignore_index=True) if all_tr else pd.DataFrame()
    trades.to_parquet(out / "trades.parquet", index=False)

    stats = []
    accts = {}
    regs = {}
    compare = {}

    for session in ("pre", "ah"):
        g = trades[trades["session"] == session]
        early = g[g["state"] == "early"]
        ext = g[g["state"] == "extended"]

        for label, sub, col in [
            ("all_continue", g, "fwd_continue"),
            ("all_fade", g, "fwd_fade"),
            ("early_continue", early, "fwd_continue"),
            ("early_fade", early, "fwd_fade"),
            ("extended_continue", ext, "fwd_continue"),
            ("extended_fade", ext, "fwd_fade"),
            ("router", g, "fwd_router"),
        ]:
            key = f"{session}/{label}"
            stats.append(summarize(sub[col], f"{key}/to_0915"))
            accts[key] = daily_account(sub.rename(columns={col: "fwd_to_0915"}), "fwd_to_0915")

        # regressability of router / early continue
        xcols = ["z_vol", "z_vp", "accel", "win_ret", "cum_ret", "eff"]
        regs[f"{session}/early_continue"] = ols_fit(early, "fwd_continue", xcols)
        regs[f"{session}/router"] = ols_fit(g, "fwd_router", xcols)
        regs[f"{session}/extended_fade"] = ols_fit(ext, "fwd_fade", xcols)

        compare[session] = {
            "n_all": int(len(g)),
            "n_early": int(len(early)),
            "n_extended": int(len(ext)),
            "early_frac": float(len(early) / len(g)) if len(g) else 0.0,
            "early_continue": summarize(early["fwd_continue"], "early_cont"),
            "extended_fade": summarize(ext["fwd_fade"], "ext_fade"),
            "router": summarize(g["fwd_router"], "router"),
            "all_continue": summarize(g["fwd_continue"], "all_cont"),
            "all_fade": summarize(g["fwd_fade"], "all_fade"),
            "account_router": accts[f"{session}/router"],
            "account_early_continue": accts[f"{session}/early_continue"],
            "account_extended_fade": accts[f"{session}/extended_fade"],
            "ols_early": regs[f"{session}/early_continue"],
            "ols_router": regs[f"{session}/router"],
        }

        # hour bucket on early continue
        if not early.empty:
            by_h = (
                early.groupby("entry_hour_bucket")["fwd_continue"]
                .agg(n="count", avg="mean", wr=lambda s: (s > 0).mean())
                .reset_index()
            )
            compare[session]["early_by_hour"] = by_h.to_dict(orient="records")

    summary = {
        "experiment": "stock_ext_hours_onset_state",
        "config": vars(args),
        "n_trades": int(len(trades)),
        "n_symbols": int(trades["symbol"].nunique()) if len(trades) else 0,
        "state_counts": trades.groupby(["session", "state"]).size().astype(int).to_dict()
        if len(trades)
        else {},
        "trade_stats": stats,
        "accounts_10pct": accts,
        "ols": regs,
        "compare": compare,
        "notes": [
            "Onset = first unusual VP/accel in session (not chase after large run)",
            "early: |cum|<=max_early_cum and |win_ret|<=max_early_ret → continue",
            "extended: already large move when signal fires → fade",
            "router = early continue + extended fade; exit always 09:15",
        ],
    }
    # fix tuple keys in state_counts for json
    sc = {}
    if len(trades):
        for (a, b), v in trades.groupby(["session", "state"]).size().items():
            sc[f"{a}/{b}"] = int(v)
    summary["state_counts"] = sc

    pd.DataFrame(stats).to_csv(out / "trade_stats.csv", index=False)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"compare": compare}, indent=2, default=str))
    print(f"results -> {out}")


if __name__ == "__main__":
    main()

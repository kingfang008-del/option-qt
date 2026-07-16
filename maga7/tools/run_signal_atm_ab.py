#!/usr/bin/env python3
"""A/B: day_lock vs signal_atm (same TopK / stable gates).

Default uses day_iv synthetic quotes for BOTH arms so the only difference
is contract selection. Optional --also-1s-baseline runs production day_lock
on 1s quotes for reference (not a pure A/B).
"""
from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from maga7.common.config import load_profile
from maga7.common.replay import run_offline_replay


def _write_arm(out_dir: Path, name: str, result: dict) -> dict:
    d = out_dir / name
    d.mkdir(parents=True, exist_ok=True)
    summary = result["summary"]
    (d / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    result["trades"].to_csv(d / "trades.csv", index=False)
    result["daily"].to_csv(d / "daily.csv", index=False)
    return summary


def _big_move_slice(trades: pd.DataFrame, threshold: float = 0.02) -> dict:
    if trades is None or trades.empty or "sig_spot" not in trades.columns:
        return {}
    # Approximate open→signal move via |K_day - spot| is not open; use |sig_strike-sig_spot|
    # Better: compare day_lock vs signal tickers when both present
    out = {}
    if "day_lock_ticker" in trades.columns and "ticker" in trades.columns:
        same = trades["day_lock_ticker"].astype(str).str.replace("O:", "", regex=False) == trades[
            "ticker"
        ].astype(str).str.replace("O:", "", regex=False)
        out["pct_same_as_day_lock"] = float(same.mean()) if len(trades) else float("nan")
        out["pct_diff_contract"] = float((~same).mean()) if len(trades) else float("nan")
    if "sig_strike" in trades.columns and "sig_spot" in trades.columns:
        m = trades.dropna(subset=["sig_strike", "sig_spot"])
        if len(m):
            moneyness = (m["sig_strike"] - m["sig_spot"]).abs() / m["sig_spot"]
            out["median_abs_moneyness"] = float(moneyness.median())
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Mag7 day_lock vs signal_atm A/B")
    p.add_argument(
        "--base-profile",
        default=str(ROOT / "maga7/CONFIG/strategy_profiles/m5c_qqq_onlywin_stable_v1.json"),
    )
    p.add_argument("--scheme", default="m5_circuit", choices=["single", "m5", "m5_circuit"])
    p.add_argument("--quote-source", default="day_iv", choices=["day_iv", "1s", "auto"])
    p.add_argument("--half-spread", type=float, default=0.01)
    p.add_argument("--start-date", default=None)
    p.add_argument("--end-date", default=None)
    p.add_argument("--tag", default="signal_atm_ab_jan_jul")
    p.add_argument(
        "--also-1s-baseline",
        action="store_true",
        help="also run production day_lock on 1s quotes (reference KPI)",
    )
    args = p.parse_args()

    base = load_profile(args.base_profile)
    if args.start_date:
        base["date_range"]["start"] = args.start_date
    if args.end_date:
        base["date_range"]["end"] = args.end_date

    out_dir = Path(base["_paths"]["results_dir"]) / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    # Shared stock/regime load happens inside each run; acceptable for research.
    arms = []

    day_lock = deepcopy(base)
    day_lock.setdefault("trade", {})
    day_lock["trade"]["contract_mode"] = "day_lock"
    day_lock["trade"]["quote_source"] = args.quote_source
    day_lock["trade"]["day_iv_half_spread_frac"] = float(args.half_spread)

    signal_atm = deepcopy(base)
    signal_atm.setdefault("trade", {})
    signal_atm["trade"]["contract_mode"] = "signal_atm"
    signal_atm["trade"]["quote_source"] = args.quote_source
    signal_atm["trade"]["day_iv_half_spread_frac"] = float(args.half_spread)

    print(f"running day_lock ({args.quote_source})...")
    r_lock = run_offline_replay(day_lock, scheme=args.scheme)
    s_lock = _write_arm(out_dir, f"day_lock_{args.quote_source}", r_lock)
    arms.append({"arm": f"day_lock_{args.quote_source}", **s_lock, **_big_move_slice(r_lock["trades"])})

    print(f"running signal_atm ({args.quote_source})...")
    r_sig = run_offline_replay(signal_atm, scheme=args.scheme)
    s_sig = _write_arm(out_dir, f"signal_atm_{args.quote_source}", r_sig)
    arms.append({"arm": f"signal_atm_{args.quote_source}", **s_sig, **_big_move_slice(r_sig["trades"])})

    if args.also_1s_baseline:
        print("running day_lock (1s production reference)...")
        prod = deepcopy(base)
        prod.setdefault("trade", {})
        prod["trade"]["contract_mode"] = "day_lock"
        prod["trade"]["quote_source"] = "1s"
        r_prod = run_offline_replay(prod, scheme=args.scheme)
        s_prod = _write_arm(out_dir, "day_lock_1s_ref", r_prod)
        arms.append({"arm": "day_lock_1s_ref", **s_prod})

    board = pd.DataFrame(arms)
    board.to_csv(out_dir / "scoreboard.csv", index=False)
    (out_dir / "scoreboard.json").write_text(
        json.dumps(arms, indent=2, default=str), encoding="utf-8"
    )

    # Pairwise trade overlap diagnostics when both have trades
    if len(r_lock["trades"]) and len(r_sig["trades"]):
        a = r_lock["trades"].copy()
        b = r_sig["trades"].copy()
        keys = ["date", "symbol", "dir", "n_in_day"]
        a["k"] = a[keys].astype(str).agg("|".join, axis=1)
        b["k"] = b[keys].astype(str).agg("|".join, axis=1)
        m = a.merge(b, on="k", suffixes=("_lock", "_sig"), how="inner")
        if len(m):
            m["same_ticker"] = (
                m["ticker_lock"].astype(str).str.replace("O:", "", regex=False)
                == m["ticker_sig"].astype(str).str.replace("O:", "", regex=False)
            )
            diag = {
                "n_paired": int(len(m)),
                "pct_same_ticker": float(m["same_ticker"].mean()),
                "mean_ret_lock": float(m["ret_lock"].mean()),
                "mean_ret_sig": float(m["ret_sig"].mean()),
                "mean_ret_delta": float((m["ret_sig"] - m["ret_lock"]).mean()),
                "mean_ret_delta_when_diff": float(
                    (m.loc[~m["same_ticker"], "ret_sig"] - m.loc[~m["same_ticker"], "ret_lock"]).mean()
                )
                if (~m["same_ticker"]).any()
                else None,
            }
            (out_dir / "paired_diag.json").write_text(json.dumps(diag, indent=2), encoding="utf-8")
            m.to_csv(out_dir / "paired_trades.csv", index=False)
            print(json.dumps(diag, indent=2))

    print(board.to_string(index=False))
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""T+30 Rule-A sensitivity on causal stock 1s (not pre-agg 1m cache).

Rebuilds mf10/streak/vol_z from ``/mnt/s990/data/raw_1s/stocks`` via
``build_stock_by_from_1s`` (1s→left-labeled 1m). Keeps
``bar_availability_delay_seconds=60`` so bars are only used after the minute
closes (same causal clock as the frozen T30 baseline).

Sweeps streak / from_prev / vol_z / peer_align_min on dual windows and ranks
rules by dual-window scoreboard.

Example:
  PYTHONPATH=. python -m maga7.tools.run_t30_sensitivity_stock1s \\
    --tag research_t30_sensitivity_stock1s_dual
"""
from __future__ import annotations

import argparse
import copy
import itertools
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from maga7.common.config import load_profile
from maga7.common.replay import run_offline_replay
from maga7.common.stock_1s import build_stock_by_from_1s, coverage_report, session_dates

T30 = (
    "maga7/CONFIG/strategy_profiles/"
    "single_qqq_open_ladder_atm5otm_t30_rails_p20_googl_peer3_v1.json"
)

WINDOWS = (
    ("jan_mar", "2026-01-02", "2026-03-31"),
    ("may_jul", "2026-05-01", "2026-07-22"),
)


def _parse_floats(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def _parse_ints(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _slice_stock_by(
    stock_by: dict[str, pd.DataFrame], start: str, end: str
) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for sym, df in stock_by.items():
        if df is None or df.empty:
            continue
        sub = df[(df["date"].astype(str) >= start) & (df["date"].astype(str) <= end)]
        if not sub.empty:
            out[sym] = sub.reset_index(drop=True)
    return out


def _ok_window(s: dict[str, Any], *, min_n: int) -> bool:
    try:
        return (
            int(s.get("n_trades") or 0) >= min_n
            and float(s.get("total_ret") or -1) > 0
            and float(s.get("maxdd") or -1) > -0.50
        )
    except Exception:
        return False


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", default=T30)
    ap.add_argument("--tag", default="research_t30_sensitivity_stock1s_dual")
    ap.add_argument("--scheme", default="single", choices=["single", "m5", "m5_circuit"])
    ap.add_argument("--streak", default="6,8,10")
    ap.add_argument("--from-prev", default="0.015,0.02,0.025")
    ap.add_argument("--vol-z", default="0.5,1.0,1.5")
    ap.add_argument("--peer-min", default="3", help="comma peer_align_min values")
    ap.add_argument("--min-n", type=int, default=15, help="min trades per window")
    ap.add_argument(
        "--delay-sec",
        type=int,
        default=60,
        help="bar_availability_delay_seconds (60 for left-labeled 1m)",
    )
    args = ap.parse_args(argv)

    profile = load_profile(args.profile)
    # Pure T+30 baseline clock
    profile["trade"]["exit_mode"] = "none"
    profile["trade"]["hold_minutes"] = 30
    profile["trade"]["bar_availability_delay_seconds"] = int(args.delay_sec)

    start_all = min(w[1] for w in WINDOWS)
    end_all = max(w[2] for w in WINDOWS)
    profile["date_range"]["start"] = start_all
    profile["date_range"]["end"] = end_all

    out_dir = Path(profile["_paths"]["results_dir"]) / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    dates = session_dates(start_all, end_all)
    print(
        f"building stock_by from 1s {start_all}..{end_all} "
        f"days≈{len(dates)} delay={args.delay_sec}s …",
        flush=True,
    )
    stock_by = build_stock_by_from_1s(profile, dates=dates, include_refs=True)
    cov = coverage_report(stock_by, dates=dates, symbols=list(profile["symbols"]))
    (out_dir / "stock_1s_coverage.json").write_text(json.dumps(cov, indent=2), encoding="utf-8")
    print(
        f"symbols={list(stock_by)} bars={sum(len(v) for v in stock_by.values())}",
        flush=True,
    )

    streaks = _parse_ints(args.streak)
    fps = _parse_floats(args.from_prev)
    vzs = _parse_floats(args.vol_z)
    peers = _parse_ints(args.peer_min)

    score_rows: list[dict[str, Any]] = []
    n_cells = len(list(itertools.product(streaks, fps, vzs, peers)))
    print(
        f"grid cells={n_cells} windows={len(WINDOWS)} "
        f"streak={streaks} fp={fps} vz={vzs} peer={peers}",
        flush=True,
    )

    cell_i = 0
    for streak, fp, vz, peer in itertools.product(streaks, fps, vzs, peers):
        cell_i += 1
        win_stats: dict[str, dict[str, Any]] = {}
        for wname, w0, w1 in WINDOWS:
            cfg = copy.deepcopy(profile)
            cfg["date_range"]["start"] = w0
            cfg["date_range"]["end"] = w1
            cfg["signal"]["streak_min"] = int(streak)
            cfg["signal"]["from_prev_abs"] = float(fp)
            cfg["signal"]["vol_z_min"] = float(vz)
            cfg["signal"]["peer_align_min"] = int(peer)
            cfg["trade"]["exit_mode"] = "none"
            cfg["trade"]["hold_minutes"] = 30
            cfg["trade"]["bar_availability_delay_seconds"] = int(args.delay_sec)
            sb = _slice_stock_by(stock_by, w0, w1)
            result = run_offline_replay(cfg, scheme=args.scheme, stock_by=sb)
            s = result["summary"]
            win_stats[wname] = s
            print(
                f"[{cell_i}/{n_cells} {wname}] s={streak} fp={fp:.3f} vz={vz:.1f} "
                f"peer={peer} → ret={s.get('total_ret', 0)*100:+.1f}% "
                f"dd={s.get('maxdd', 0)*100:.1f}% n={s.get('n_trades')} "
                f"win={float(s.get('trade_win') or 0)*100:.0f}%",
                flush=True,
            )

        is_base = (
            streak == 8
            and abs(fp - 0.02) < 1e-12
            and abs(vz - 1.0) < 1e-12
            and peer == 3
        )
        dual = all(
            _ok_window(win_stats[w], min_n=int(args.min_n)) for w, _, _ in WINDOWS
        )
        row: dict[str, Any] = {
            "streak_min": streak,
            "from_prev_abs": fp,
            "vol_z_min": vz,
            "peer_align_min": peer,
            "is_baseline": is_base,
            "dual_pass": dual,
            "stock_source": "raw_1s",
            "delay_sec": int(args.delay_sec),
            "hold_minutes": 30,
            "exit_mode": "none",
        }
        for wname, _, _ in WINDOWS:
            s = win_stats[wname]
            for k in (
                "total_ret",
                "maxdd",
                "n_trades",
                "trade_win",
                "trade_exp",
                "n_peer_block",
                "n_regime_block",
            ):
                row[f"{wname}_{k}"] = s.get(k)
            # per-window calmar-like
            ret = float(s.get("total_ret") or 0)
            dd = abs(float(s.get("maxdd") or 0))
            row[f"{wname}_calmar_like"] = ret / max(dd, 0.05)

        # dual score: worse-window calmar, then sum ret
        row["dual_calmar_min"] = min(
            float(row["jan_mar_calmar_like"] or 0),
            float(row["may_jul_calmar_like"] or 0),
        )
        row["dual_ret_sum"] = float(row.get("jan_mar_total_ret") or 0) + float(
            row.get("may_jul_total_ret") or 0
        )
        score_rows.append(row)
        if dual:
            print(
                f"  *** DUAL PASS s={streak} fp={fp} vz={vz} peer={peer} "
                f"sum_ret={row['dual_ret_sum']*100:+.1f}% "
                f"calmar_min={row['dual_calmar_min']:.2f}",
                flush=True,
            )

    score = pd.DataFrame(score_rows)
    score.to_csv(out_dir / "scoreboard.csv", index=False)

    dual_df = score[score["dual_pass"]].copy()
    if len(dual_df):
        dual_df = dual_df.sort_values(
            ["dual_calmar_min", "dual_ret_sum"], ascending=[False, False]
        )
    else:
        # fallback ranking even if dual_pass empty
        dual_df = score.sort_values(
            ["dual_calmar_min", "dual_ret_sum"], ascending=[False, False]
        )

    best = dual_df.iloc[0].to_dict() if len(dual_df) else None
    baseline_row = None
    base_hits = score[score["is_baseline"]]
    if len(base_hits):
        baseline_row = base_hits.iloc[0].to_dict()

    summary = {
        "profile": args.profile,
        "stock_source": "/mnt/s990/data/raw_1s/stocks",
        "hold_minutes": 30,
        "exit_mode": "none",
        "delay_sec": int(args.delay_sec),
        "windows": [list(w) for w in WINDOWS],
        "grid": {
            "streak": streaks,
            "from_prev": fps,
            "vol_z": vzs,
            "peer_min": peers,
        },
        "n_cells": int(len(score)),
        "n_dual_pass": int(score["dual_pass"].sum()) if len(score) else 0,
        "verdict": "PASS" if int(score["dual_pass"].sum() or 0) > 0 else "REJECT",
        "baseline_frozen": baseline_row,
        "best_dual": best,
        "top_dual": dual_df.head(15).to_dict(orient="records") if len(dual_df) else [],
        "note": (
            "Rule-A features rebuilt from causal 1s→1m; delay=60s for left-label "
            "availability. Pure T+30 (no hold_extend). Optimal = max min(calmar) "
            "among dual_pass, else best dual_calmar_min overall."
        ),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    (out_dir / "best.json").write_text(
        json.dumps({"best_dual": best, "baseline": baseline_row}, indent=2, default=str),
        encoding="utf-8",
    )

    print("\n=== baseline (s8/fp0.02/vz1/peer3) on stock_1s ===", flush=True)
    print(json.dumps(baseline_row, indent=2, default=str), flush=True)
    print(
        f"\n=== best dual (n_pass={summary['n_dual_pass']}) verdict={summary['verdict']} ===",
        flush=True,
    )
    print(json.dumps(best, indent=2, default=str), flush=True)
    if len(dual_df):
        cols = [
            c
            for c in [
                "streak_min",
                "from_prev_abs",
                "vol_z_min",
                "peer_align_min",
                "dual_pass",
                "dual_calmar_min",
                "dual_ret_sum",
                "jan_mar_total_ret",
                "jan_mar_maxdd",
                "jan_mar_n_trades",
                "may_jul_total_ret",
                "may_jul_maxdd",
                "may_jul_n_trades",
                "is_baseline",
            ]
            if c in dual_df.columns
        ]
        print("\n=== top 15 ===", flush=True)
        print(dual_df[cols].head(15).to_string(index=False), flush=True)
    print(f"wrote {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

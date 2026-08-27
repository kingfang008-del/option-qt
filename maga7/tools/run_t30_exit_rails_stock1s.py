#!/usr/bin/env python3
"""T+30 Rule-A exit-rails ablation on causal 1s→1m (entry frozen).

Keeps frozen entry: streak=8 / fp=2% / vz=1.0 / peer3 / delay=60.
Sweeps option-price rails ``tp_mult`` × ``sl_mult`` × ``hold_minutes`` under
``exit_mode=none`` (same book as causal baseline: hit TP/SL or clock flatten).

Example:
  PYTHONPATH=. python -m maga7.tools.run_t30_exit_rails_stock1s \\
    --tag research_t30_exit_rails_stock1s_dual
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
from maga7.tools.run_t30_sensitivity_stock1s import WINDOWS, _ok_window, _slice_stock_by

T30 = (
    "maga7/CONFIG/strategy_profiles/"
    "single_qqq_open_ladder_atm5otm_t30_rails_p20_googl_peer3_v1.json"
)


def _parse_floats(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def _parse_ints(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", default=T30)
    ap.add_argument("--tag", default="research_t30_exit_rails_stock1s_dual")
    ap.add_argument("--scheme", default="single", choices=["single", "m5", "m5_circuit"])
    ap.add_argument("--tp", default="1.4,1.6,2.0,2.5")
    ap.add_argument("--sl", default="0.35,0.4,0.5,0.65")
    ap.add_argument("--hold", default="20,30,45")
    ap.add_argument("--min-n", type=int, default=15)
    ap.add_argument("--delay-sec", type=int, default=60)
    args = ap.parse_args(argv)

    profile = load_profile(args.profile)
    # Freeze entry + pure rails book
    profile["trade"]["exit_mode"] = "none"
    profile["trade"]["bar_availability_delay_seconds"] = int(args.delay_sec)
    profile["signal"]["streak_min"] = 8
    profile["signal"]["from_prev_abs"] = 0.02
    profile["signal"]["vol_z_min"] = 1.0
    profile["signal"]["peer_align_min"] = 3

    start_all = min(w[1] for w in WINDOWS)
    end_all = max(w[2] for w in WINDOWS)
    profile["date_range"]["start"] = start_all
    profile["date_range"]["end"] = end_all

    out_dir = Path(profile["_paths"]["results_dir"]) / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    dates = session_dates(start_all, end_all)
    print(
        f"building stock_by 1s→1m {start_all}..{end_all} delay={args.delay_sec}s …",
        flush=True,
    )
    stock_by = build_stock_by_from_1s(profile, dates=dates, include_refs=True)
    cov = coverage_report(stock_by, dates=dates, symbols=list(profile["symbols"]))
    (out_dir / "stock_1s_coverage.json").write_text(
        json.dumps(cov, indent=2), encoding="utf-8"
    )
    print(f"bars={sum(len(v) for v in stock_by.values())}", flush=True)

    tps = _parse_floats(args.tp)
    sls = _parse_floats(args.sl)
    holds = _parse_ints(args.hold)
    cells = list(itertools.product(tps, sls, holds))
    print(
        f"exit grid cells={len(cells)} tp={tps} sl={sls} hold={holds} "
        f"(entry frozen s8/fp2/vz1/peer3)",
        flush=True,
    )

    score_rows: list[dict[str, Any]] = []
    for i, (tp, sl, hold) in enumerate(cells, 1):
        win_stats: dict[str, dict[str, Any]] = {}
        for wname, w0, w1 in WINDOWS:
            cfg = copy.deepcopy(profile)
            cfg["date_range"]["start"] = w0
            cfg["date_range"]["end"] = w1
            cfg["trade"]["tp_mult"] = float(tp)
            cfg["trade"]["sl_mult"] = float(sl)
            cfg["trade"]["hold_minutes"] = int(hold)
            cfg["trade"]["exit_mode"] = "none"
            cfg["trade"]["bar_availability_delay_seconds"] = int(args.delay_sec)
            sb = _slice_stock_by(stock_by, w0, w1)
            result = run_offline_replay(cfg, scheme=args.scheme, stock_by=sb)
            s = result["summary"]
            win_stats[wname] = s
            print(
                f"[{i}/{len(cells)} {wname}] tp={tp:.2f} sl={sl:.2f} H={hold} → "
                f"ret={float(s.get('total_ret') or 0)*100:+.1f}% "
                f"dd={float(s.get('maxdd') or 0)*100:.1f}% "
                f"n={s.get('n_trades')} "
                f"win={float(s.get('trade_win') or 0)*100:.0f}% "
                f"exp={float(s.get('trade_exp') or 0)*100:+.1f}%",
                flush=True,
            )

        is_base = (
            abs(tp - 1.6) < 1e-12 and abs(sl - 0.4) < 1e-12 and hold == 30
        )
        dual = all(
            _ok_window(win_stats[w], min_n=int(args.min_n)) for w, _, _ in WINDOWS
        )
        row: dict[str, Any] = {
            "tp_mult": tp,
            "sl_mult": sl,
            "hold_minutes": hold,
            "is_baseline": is_base,
            "dual_pass": dual,
            "stock_source": "raw_1s",
            "delay_sec": int(args.delay_sec),
            "exit_mode": "none",
            "entry": "s8_fp02_vz1_peer3",
        }
        for wname, _, _ in WINDOWS:
            s = win_stats[wname]
            for k in (
                "total_ret",
                "maxdd",
                "n_trades",
                "trade_win",
                "trade_exp",
            ):
                row[f"{wname}_{k}"] = s.get(k)
            ret = float(s.get("total_ret") or 0)
            dd = abs(float(s.get("maxdd") or 0))
            row[f"{wname}_calmar_like"] = ret / max(dd, 0.05)

        row["dual_calmar_min"] = min(
            float(row["jan_mar_calmar_like"] or 0),
            float(row["may_jul_calmar_like"] or 0),
        )
        row["dual_ret_sum"] = float(row.get("jan_mar_total_ret") or 0) + float(
            row.get("may_jul_total_ret") or 0
        )
        row["dual_win_min"] = min(
            float(row.get("jan_mar_trade_win") or 0),
            float(row.get("may_jul_trade_win") or 0),
        )
        row["dual_dd_worst"] = min(
            float(row.get("jan_mar_maxdd") or 0),
            float(row.get("may_jul_maxdd") or 0),
        )
        score_rows.append(row)
        if dual:
            mark = " BASE" if is_base else ""
            print(
                f"  *** DUAL PASS{mark} tp={tp} sl={sl} H={hold} "
                f"sum_ret={row['dual_ret_sum']*100:+.1f}% "
                f"calmar_min={row['dual_calmar_min']:.2f} "
                f"win_min={row['dual_win_min']*100:.0f}%",
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
        dual_df = score.sort_values(
            ["dual_calmar_min", "dual_ret_sum"], ascending=[False, False]
        )

    best = dual_df.iloc[0].to_dict() if len(dual_df) else None
    base_hits = score[score["is_baseline"]]
    baseline_row = base_hits.iloc[0].to_dict() if len(base_hits) else None

    # Beat baseline? stricter: dual_pass + better calmar_min than baseline
    beat = None
    if baseline_row is not None and len(dual_df):
        bcal = float(baseline_row.get("dual_calmar_min") or 0)
        cand = dual_df[
            (~dual_df["is_baseline"])
            & (dual_df["dual_calmar_min"] > bcal + 1e-9)
        ]
        if len(cand):
            beat = cand.iloc[0].to_dict()

    by_win = score.sort_values(
        ["dual_win_min", "dual_ret_sum"], ascending=[False, False]
    )

    summary = {
        "profile": args.profile,
        "stock_source": "/mnt/s990/data/raw_1s/stocks",
        "entry_frozen": "streak8/fp0.02/vz1.0/peer3",
        "exit_mode": "none",
        "delay_sec": int(args.delay_sec),
        "windows": [list(w) for w in WINDOWS],
        "grid": {"tp": tps, "sl": sls, "hold": holds},
        "n_cells": int(len(score)),
        "n_dual_pass": int(score["dual_pass"].sum()) if len(score) else 0,
        "baseline": baseline_row,
        "best_dual": best,
        "beat_baseline": beat,
        "verdict": (
            "BEAT"
            if beat is not None
            else (
                "KEEP_BASELINE"
                if baseline_row is not None
                and best is not None
                and bool(best.get("is_baseline"))
                else ("PASS" if int(score["dual_pass"].sum() or 0) > 0 else "REJECT")
            )
        ),
        "top_dual": dual_df.head(15).to_dict(orient="records") if len(dual_df) else [],
        "top_by_win": by_win.head(10).to_dict(orient="records") if len(by_win) else [],
        "note": (
            "Entry frozen at causal Rule-A. Exit = option mid/fill rails "
            "tp_mult/sl_mult or hold_minutes clock. Compare dual_calmar_min "
            "vs baseline tp1.6/sl0.4/H30."
        ),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )

    print("\n=== baseline tp1.6/sl0.4/H30 ===", flush=True)
    print(json.dumps(baseline_row, indent=2, default=str), flush=True)
    print(f"\n=== best dual verdict={summary['verdict']} ===", flush=True)
    print(json.dumps(best, indent=2, default=str), flush=True)
    if beat is not None:
        print("\n=== beat baseline ===", flush=True)
        print(json.dumps(beat, indent=2, default=str), flush=True)
    cols = [
        c
        for c in [
            "tp_mult",
            "sl_mult",
            "hold_minutes",
            "is_baseline",
            "dual_pass",
            "dual_calmar_min",
            "dual_ret_sum",
            "dual_win_min",
            "dual_dd_worst",
            "jan_mar_total_ret",
            "jan_mar_maxdd",
            "jan_mar_trade_win",
            "may_jul_total_ret",
            "may_jul_maxdd",
            "may_jul_trade_win",
        ]
        if c in dual_df.columns
    ]
    print("\n=== top 15 dual_calmar ===", flush=True)
    print(dual_df[cols].head(15).to_string(index=False), flush=True)
    print(f"wrote {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Multi-window tick validation for stock_flow_opt (trades/tick pricing).

Windows (intersect available tick dates):
  Feb_Apr · May1_Jul9 · Jul10_23 (Jul = discovery pocket)

Cells (fixed, no grid chase):
  frozen   baseline + tp25/sl20/h900
  exit_cand volz15  + tp10/sl25/h900
  + small neighbors for context

Promote gate: Feb_Apr AND May1_Jul9 both keep_edge
  (n>=min_n, mean>0, add>0, day_win>=0.55). Jul reported but not required
  for OOS dual (it is the pocket that selected the cell).

Example:
  PYTHONPATH=. python -m maga7.tools.run_stock_flow_tick_dual \\
    --tag research_stock_flow_tick_dual_feb_jul
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from maga7.common.config import load_profile
from maga7.common.open_lock import load_multidte_lock_index, resolve_otm_rungs
from maga7.common.option_flow import DEFAULT_TICK_ROOT, tick_dates
from maga7.tools.run_stock_flow_exit_ablation import (
    PROFILE,
    SESSIONS,
    SESSIONS_OPEN1H,
    _collect_arms,
    _score_arms,
)

# name, start, end, prefix, role
WINDOWS = (
    ("Feb_Apr", "2026-02-01", "2026-04-30", "fa", "oos"),
    ("May1_Jul9", "2026-05-01", "2026-07-09", "mj", "oos"),
    ("Jul10_23", "2026-07-10", "2026-07-23", "jul", "pocket"),
)

CELLS = (
    ("frozen_baseline", "baseline", 0.25, 0.20, 900),
    ("exit_cand_volz15_tp10_sl25", "volz15", 0.10, 0.25, 900),
    ("volz15_tp12_sl25", "volz15", 0.12, 0.25, 900),
    ("volz15_tp08_sl25", "volz15", 0.08, 0.25, 900),
    ("volz15_frozen_exit", "volz15", 0.25, 0.20, 900),
    ("baseline_tp10_sl25", "baseline", 0.10, 0.25, 900),
)


def _keep_edge(st: dict[str, Any], *, min_n: int) -> bool:
    return bool(
        int(st.get("n") or 0) >= int(min_n)
        and float(st.get("mean") or 0) > 0
        and float(st.get("add") or 0) > 0
        and float(st.get("day_win") or 0) >= 0.55
    )


def _hit_win(st: dict[str, Any], *, min_n: int, target_win: float) -> bool:
    return bool(
        int(st.get("n") or 0) >= int(min_n)
        and float(st.get("win") or 0) >= float(target_win)
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", default=PROFILE)
    ap.add_argument("--tag", default="research_stock_flow_tick_dual_feb_jul")
    ap.add_argument("--tick-root", default=str(DEFAULT_TICK_ROOT))
    ap.add_argument(
        "--sessions",
        default="fullday",
        choices=("fullday", "open1h"),
        help="fullday=09:35–15:30 multi-session; open1h=09:35–10:30 only",
    )
    ap.add_argument("--stride-sec", type=int, default=5)
    ap.add_argument("--rearm-gap-sec", type=int, default=60)
    ap.add_argument("--slip", type=float, default=0.01)
    ap.add_argument("--position-frac", type=float, default=0.10)
    ap.add_argument("--max-concurrent", type=int, default=4)
    ap.add_argument("--cooldown-minutes", type=float, default=1.0)
    ap.add_argument("--target-win", type=float, default=0.55)
    ap.add_argument("--min-n-fa", type=int, default=80)
    ap.add_argument("--min-n-mj", type=int, default=80)
    ap.add_argument("--min-n-jul", type=int, default=40)
    # open1h has fewer arms — allow lower floors
    ap.add_argument("--min-n-fa-open1h", type=int, default=40)
    ap.add_argument("--min-n-mj-open1h", type=int, default=40)
    ap.add_argument("--min-n-jul-open1h", type=int, default=20)
    args = ap.parse_args(argv)

    sess_tuple = SESSIONS_OPEN1H if args.sessions == "open1h" else SESSIONS
    if args.sessions == "open1h":
        args.min_n_fa = int(args.min_n_fa_open1h)
        args.min_n_mj = int(args.min_n_mj_open1h)
        args.min_n_jul = int(args.min_n_jul_open1h)

    min_n_map = {
        "Feb_Apr": int(args.min_n_fa),
        "May1_Jul9": int(args.min_n_mj),
        "Jul10_23": int(args.min_n_jul),
    }

    all_dates = tick_dates(args.tick_root)
    win_dates: dict[str, list[str]] = {}
    for wname, a, b, _, _ in WINDOWS:
        win_dates[wname] = [d for d in all_dates if a <= d <= b]
        print(f"window {wname}: n_dates={len(win_dates[wname])} {a}..{b}", flush=True)
        if not win_dates[wname]:
            print(f"ERROR empty window {wname}", flush=True)
            return 2

    prof = load_profile(args.profile)
    paths = prof["_paths"]
    symbols = list(prof.get("symbols") or [])
    stock_1s = Path(paths.get("stock_1s_root") or "/mnt/s990/data/raw_1s/stocks").expanduser()
    tick_root = Path(args.tick_root)
    lock = load_multidte_lock_index(Path(paths["open_locked_map"]).expanduser())
    otm = resolve_otm_rungs(prof, default=3)
    out = Path(paths["results_dir"]) / args.tag
    out.mkdir(parents=True, exist_ok=True)

    entry_needed = sorted({c[1] for c in CELLS})
    arms_by_win_entry: dict[tuple[str, str], list] = {}
    for wname, dates in win_dates.items():
        for en in entry_needed:
            vz = 1.5 if en == "volz15" else None
            print(f"collecting arms {wname}/{en}…", flush=True)
            arms = _collect_arms(
                dates=dates,
                symbols=symbols,
                stock_1s=stock_1s,
                tick_root=tick_root,
                lock=lock,
                otm=otm,
                require_volz=vz,
                stride_sec=int(args.stride_sec),
                rearm_gap_sec=int(args.rearm_gap_sec),
                sessions=sess_tuple,
            )
            arms_by_win_entry[(wname, en)] = arms
            print(f"  arms={len(arms)}", flush=True)

    rows: list[dict[str, Any]] = []
    by_cell: dict[str, dict[str, Any]] = {}

    for cname, en, tp, sl, h in CELLS:
        cell_sum: dict[str, Any] = {
            "name": cname,
            "entry": en,
            "tp": tp,
            "sl": sl,
            "max_hold_sec": h,
        }
        oos_edge = True
        oos_win = True
        all_edge = True
        n_edge_windows = 0
        for wname, _, _, prefix, role in WINDOWS:
            min_n = min_n_map[wname]
            st, sized = _score_arms(
                arms_by_win_entry[(wname, en)],
                tp=float(tp),
                sl=float(sl),
                max_hold_sec=int(h),
                slip=float(args.slip),
                position_frac=float(args.position_frac),
                max_concurrent=int(args.max_concurrent),
                cooldown_minutes=float(args.cooldown_minutes),
            )
            edge = _keep_edge(st, min_n=min_n)
            hitw = _hit_win(st, min_n=min_n, target_win=float(args.target_win))
            if edge:
                n_edge_windows += 1
            all_edge = all_edge and edge
            if role == "oos":
                oos_edge = oos_edge and edge
                oos_win = oos_win and hitw
            cell_sum[f"{prefix}_n"] = st.get("n")
            cell_sum[f"{prefix}_win"] = st.get("win")
            cell_sum[f"{prefix}_mean"] = st.get("mean")
            cell_sum[f"{prefix}_day_win"] = st.get("day_win")
            cell_sum[f"{prefix}_add"] = st.get("add")
            cell_sum[f"{prefix}_keep_edge"] = edge
            cell_sum[f"{prefix}_hit_win"] = hitw
            cell_sum[f"{prefix}_n_dates"] = len(win_dates[wname])
            if sized:
                pd.DataFrame(sized).to_csv(out / f"trades_{cname}_{wname}.csv", index=False)
            rows.append(
                {
                    "cell": cname,
                    "window": wname,
                    "role": role,
                    "entry": en,
                    "tp": tp,
                    "sl": sl,
                    "max_hold_sec": h,
                    "min_n": min_n,
                    "keep_edge": edge,
                    "hit_win": hitw,
                    **st,
                }
            )

        cell_sum["oos_dual_keep_edge"] = oos_edge
        cell_sum["oos_dual_hit_win"] = oos_win
        cell_sum["triple_keep_edge"] = all_edge
        cell_sum["n_edge_windows"] = n_edge_windows
        if oos_edge and oos_win:
            cell_sum["verdict"] = "OOS_DUAL_PASS"
        elif oos_edge:
            cell_sum["verdict"] = "OOS_DUAL_EDGE_WIN_SHORT"
        elif n_edge_windows == 1:
            cell_sum["verdict"] = "SINGLE_WINDOW_ONLY"
        elif n_edge_windows >= 2:
            cell_sum["verdict"] = "PARTIAL_WINDOWS"
        else:
            cell_sum["verdict"] = "MULTI_FAIL"
        by_cell[cname] = cell_sum

    score = pd.DataFrame(rows)
    score.to_csv(out / "scoreboard_by_window.csv", index=False)
    cells_df = pd.DataFrame(list(by_cell.values()))
    cells_df.to_csv(out / "scoreboard_dual.csv", index=False)

    cand = by_cell["exit_cand_volz15_tp10_sl25"]
    frozen = by_cell["frozen_baseline"]
    overall = cand["verdict"]
    if overall.startswith("OOS_DUAL"):
        pass
    elif frozen["verdict"].startswith("OOS_DUAL"):
        overall = f"FROZEN_{frozen['verdict']}_CAND_{cand['verdict']}"

    summary = {
        "windows": [
            {
                "name": w[0],
                "start": w[1],
                "end": w[2],
                "prefix": w[3],
                "role": w[4],
                "n_tick_dates": len(win_dates[w[0]]),
                "dates": win_dates[w[0]],
            }
            for w in WINDOWS
        ],
        "target_win": args.target_win,
        "min_n": min_n_map,
        "overall_verdict": overall,
        "exit_candidate": cand,
        "frozen_baseline": frozen,
        "cells": by_cell,
        "sessions": args.sessions,
        "session_bounds": [list(x) for x in sess_tuple],
        "note": (
            "Tick pricing (last±slip). OOS dual = Feb_Apr ∩ May1_Jul9 keep_edge. "
            "Jul10_23 is discovery pocket. Quote FillSpec still required before wire."
        ),
    }
    print(f"sessions={args.sessions} bounds={list(sess_tuple)}", flush=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    print("\n=== multi-window scoreboard ===", flush=True)
    cols = [
        c
        for c in [
            "name",
            "verdict",
            "fa_n",
            "fa_win",
            "fa_add",
            "fa_day_win",
            "mj_n",
            "mj_win",
            "mj_add",
            "mj_day_win",
            "jul_n",
            "jul_win",
            "jul_add",
            "jul_day_win",
        ]
        if c in cells_df.columns
    ]
    print(cells_df[cols].to_string(index=False), flush=True)
    print(f"\noverall: {overall}", flush=True)
    print(
        f"exit_cand: FA win={cand.get('fa_win')} add={cand.get('fa_add')} | "
        f"MJ win={cand.get('mj_win')} add={cand.get('mj_add')} | "
        f"Jul win={cand.get('jul_win')} add={cand.get('jul_add')} → {cand.get('verdict')}",
        flush=True,
    )
    print(f"wrote {out}", flush=True)
    return 0 if str(overall).startswith("OOS_DUAL") else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""L2 washout_reclaim neighborhood: wash_drop_min × allow_baseline_opposite.

Does not retune defaults — only scores locked grid vs L0/L1 on dual windows.
Gate: each cell's total_ret >= 0.90 × L0 on both strong and weak windows.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from maga7.common.config import load_profile
from maga7.common.replay import run_offline_replay

L2_DEFAULT = (
    "maga7/CONFIG/strategy_profiles/"
    "single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_watchdog_hunter_washout_reclaim_v1.json"
)


def _run(prof: dict, *, start: str, end: str) -> dict:
    p = copy.deepcopy(prof)
    p["date_range"] = {"start": start, "end": end}
    res = run_offline_replay(p, scheme="single")
    s = res["summary"]
    return {
        "total_ret": float(s["total_ret"]),
        "maxdd": float(s["maxdd"]),
        "n_trades": int(s["n_trades"]),
        "n_hunt_trades": int(s.get("n_hunt_trades") or 0),
        "n_hunt_signals": int(s.get("n_hunt_signals") or 0),
        "trade_win": s.get("trade_win"),
        "end_equity": s.get("end_equity"),
        "watchdog_state_counts": s.get("watchdog_state_counts") or {},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", default=L2_DEFAULT)
    ap.add_argument("--out", default="maga7/results/watchdog/hunter_wd_neighborhood")
    ap.add_argument("--strong-start", default="2026-05-01")
    ap.add_argument("--strong-end", default="2026-07-17")
    ap.add_argument("--weak-start", default="2026-02-01")
    ap.add_argument("--weak-end", default="2026-04-30")
    ap.add_argument("--gate", type=float, default=0.90, help="min vs L0 total_ret")
    args = ap.parse_args()

    base = load_profile(args.profile)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    drops = [0.012, 0.015, 0.018]
    opps = [False, True]
    windows = {
        "strong": (args.strong_start, args.strong_end),
        "weak": (args.weak_start, args.weak_end),
    }

    # Anchors: L0 (watchdog off) and L1 (hunter off) from same base profile
    anchors: dict[str, dict[str, dict]] = {"strong": {}, "weak": {}}
    for wname, (start, end) in windows.items():
        p0 = copy.deepcopy(base)
        p0.setdefault("watchdog", {})["enabled"] = False
        if isinstance(p0.get("watchdog"), dict) and "hunter" in p0["watchdog"]:
            p0["watchdog"]["hunter"]["enabled"] = False
        print(f"anchor L0 {wname}...", flush=True)
        anchors[wname]["L0"] = _run(p0, start=start, end=end)

        p1 = copy.deepcopy(base)
        p1.setdefault("watchdog", {})["enabled"] = True
        p1["watchdog"].setdefault("hunter", {})["enabled"] = False
        print(f"anchor L1 {wname}...", flush=True)
        anchors[wname]["L1"] = _run(p1, start=start, end=end)

    rows = []
    for wd in drops:
        for opp in opps:
            tag = f"wd{int(wd * 1000):02d}_{'opp' if opp else 'mutex'}"
            for wname, (start, end) in windows.items():
                p = copy.deepcopy(base)
                p.setdefault("watchdog", {})["enabled"] = True
                h = p["watchdog"].setdefault("hunter", {})
                h["enabled"] = True
                h["wash_drop_min"] = float(wd)
                h["allow_baseline_opposite"] = bool(opp)
                h["mutex_scope"] = "symbol_dir" if opp else "symbol"
                print(f"run {tag} {wname}...", flush=True)
                r = _run(p, start=start, end=end)
                l0 = anchors[wname]["L0"]["total_ret"]
                l1 = anchors[wname]["L1"]["total_ret"]
                vs_l0 = r["total_ret"] / l0 if abs(l0) > 1e-12 else float("nan")
                vs_l1 = r["total_ret"] / l1 if abs(l1) > 1e-12 else float("nan")
                rows.append(
                    {
                        "tag": tag,
                        "window": wname,
                        "wash_drop_min": wd,
                        "allow_baseline_opposite": opp,
                        "vs_L0": vs_l0,
                        "vs_L1": vs_l1,
                        "pass_gate": bool(vs_l0 >= float(args.gate)),
                        **r,
                    }
                )

    # pivot summary
    sb = pd.DataFrame(rows)
    sb.to_csv(out / "neighborhood.csv", index=False)
    pivot = (
        sb.pivot_table(
            index=["wash_drop_min", "allow_baseline_opposite"],
            columns="window",
            values=["total_ret", "vs_L0", "vs_L1", "n_hunt_trades", "pass_gate"],
            aggfunc="first",
        )
        .sort_index()
    )
    pivot.to_csv(out / "neighborhood_pivot.csv")

    # cell passes both windows?
    cell_pass = []
    for (wd, opp), g in sb.groupby(["wash_drop_min", "allow_baseline_opposite"]):
        ok = bool(g["pass_gate"].all())
        cell_pass.append(
            {
                "wash_drop_min": wd,
                "allow_baseline_opposite": opp,
                "pass_both": ok,
                "vs_L0_strong": float(g.loc[g.window == "strong", "vs_L0"].iloc[0]),
                "vs_L0_weak": float(g.loc[g.window == "weak", "vs_L0"].iloc[0]),
                "vs_L1_strong": float(g.loc[g.window == "strong", "vs_L1"].iloc[0]),
                "vs_L1_weak": float(g.loc[g.window == "weak", "vs_L1"].iloc[0]),
                "is_default": bool(abs(wd - 0.015) < 1e-12 and opp),
            }
        )
    cells = pd.DataFrame(cell_pass)
    cells.to_csv(out / "cell_pass.csv", index=False)

    summary = {
        "gate_vs_L0": float(args.gate),
        "default": {"wash_drop_min": 0.015, "allow_baseline_opposite": True},
        "anchors": anchors,
        "cells": cell_pass,
        "verdict": (
            "PASS_NEIGHBORHOOD"
            if all(c["pass_both"] for c in cell_pass if c["is_default"])
            and sum(1 for c in cell_pass if c["pass_both"]) >= 4
            else "REVIEW_NEIGHBORHOOD"
        ),
        "note": (
            "PASS if default cell passes both windows and ≥4/6 cells pass "
            "(no cliff). REVIEW otherwise — do not promote L2."
        ),
    }
    # refine verdict: default must pass; majority of grid pass; no cell < 0.85 if neighbor of default
    default_ok = next(c["pass_both"] for c in cell_pass if c["is_default"])
    n_pass = sum(1 for c in cell_pass if c["pass_both"])
    near_cliff = any(
        (not c["pass_both"]) and abs(c["wash_drop_min"] - 0.015) < 1e-12 for c in cell_pass
    )
    if default_ok and n_pass >= 4 and not near_cliff:
        summary["verdict"] = "PASS_NEIGHBORHOOD"
    elif default_ok:
        summary["verdict"] = "PASS_DEFAULT_WEAK_GRID"
    else:
        summary["verdict"] = "FAIL_NEIGHBORHOOD"

    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str))
    print(cells.to_string(index=False))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

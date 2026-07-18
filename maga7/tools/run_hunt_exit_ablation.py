#!/usr/bin/env python3
"""P2.1: Hunt-only exit / size ablations — dual window + 2025H2 OOS.

Does not mutate L2 default profile. Goal: find a Hunt-exit cell with
2025H2 L2 >= L1 and strong/weak vs L0 >= 0.95.
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

L2 = (
    "maga7/CONFIG/strategy_profiles/"
    "single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_watchdog_hunter_washout_reclaim_v1.json"
)
L1 = (
    "maga7/CONFIG/strategy_profiles/"
    "single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_watchdog_v1.json"
)
H2 = (
    "maga7/CONFIG/strategy_profiles/"
    "single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_2025h2_v1.json"
)

WINDOWS = {
    "strong": ("2026-05-01", "2026-07-17", None),
    "weak": ("2026-02-01", "2026-04-30", None),
    "oos_2025h2": ("2025-07-01", "2025-12-31", H2),
}

VARIANTS: dict[str, dict] = {
    "l2_default": {},
    "hold15_noext": {
        "hold_minutes": 15,
        "exit_mode": "none",
        "hold_extend_minutes": 15,
    },
    "hold20_noext": {
        "hold_minutes": 20,
        "exit_mode": "none",
        "hold_extend_minutes": 20,
    },
    "sl70": {"sl_mult": 0.70},
    "mae25": {
        "early_exit_mode": "mae_cut",
        "mae_cut_ret": 0.25,
        "mae_cut_mfe_bypass": 0.20,
        "mae_cut_min_hold_minutes": 5,
    },
    "mae20_fast": {
        "early_exit_mode": "mae_cut",
        "mae_cut_ret": 0.20,
        "mae_cut_mfe_bypass": 0.15,
        "mae_cut_min_hold_minutes": 3,
    },
    "hold20_mae25": {
        "hold_minutes": 20,
        "hold_extend_minutes": 20,
        "exit_mode": "hold_extend+mae_cut",
        "mae_cut_ret": 0.25,
        "mae_cut_mfe_bypass": 0.20,
        "mae_cut_min_hold_minutes": 5,
    },
    "size10": {"position_frac": 0.10},
    "hold15_size10": {
        "hold_minutes": 15,
        "exit_mode": "none",
        "position_frac": 0.10,
    },
    "mae25_size10": {
        "early_exit_mode": "mae_cut",
        "mae_cut_ret": 0.25,
        "mae_cut_mfe_bypass": 0.20,
        "mae_cut_min_hold_minutes": 5,
        "position_frac": 0.10,
    },
}


def _tot_from_daily(daily: pd.DataFrame) -> float:
    eq = 1.0
    for r in daily["day_ret"].astype(float):
        eq *= 1.0 + float(r)
    return eq - 1.0


def _run(prof: dict, *, start: str, end: str) -> dict:
    p = copy.deepcopy(prof)
    p["date_range"] = {"start": start, "end": end}
    res = run_offline_replay(p, scheme="single")
    s = res["summary"]
    daily = res["daily"]
    trades = res["trades"]
    total = _tot_from_daily(daily) if not daily.empty else float(s["total_ret"])
    n_hunt = int(s.get("n_hunt_trades") or 0)
    hunt_mean = float("nan")
    if not trades.empty and "route" in trades.columns:
        h = trades[trades["route"].astype(str) == "hunt"]
        if len(h):
            hunt_mean = float(h["ret"].mean())
            n_hunt = len(h)
    return {
        "total_ret": total,
        "maxdd": float(s["maxdd"]),
        "n_trades": int(s["n_trades"]),
        "n_hunt_trades": n_hunt,
        "hunt_mean_ret": hunt_mean,
        "end_equity": float(s.get("end_equity") or 100.0 * (1.0 + total)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="maga7/results/watchdog/hunt_exit_ablation_p21")
    ap.add_argument("--gate-vs-l0", type=float, default=0.95)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    l2_src = load_profile(L2)
    l1_src = load_profile(L1)

    # Anchors L0/L1 per window
    anchors: dict[str, dict] = {}
    for wname, (start, end, base_path) in WINDOWS.items():
        base = load_profile(base_path) if base_path else copy.deepcopy(l2_src)
        # L0
        p0 = copy.deepcopy(base)
        p0["watchdog"] = copy.deepcopy(l2_src.get("watchdog") or {})
        p0["watchdog"]["enabled"] = False
        print(f"anchor L0 {wname}...", flush=True)
        a0 = _run(p0, start=start, end=end)
        # L1
        p1 = copy.deepcopy(base)
        p1["watchdog"] = copy.deepcopy(l1_src.get("watchdog") or {})
        p1["watchdog"]["enabled"] = True
        p1["watchdog"].setdefault("hunter", {})["enabled"] = False
        print(f"anchor L1 {wname}...", flush=True)
        a1 = _run(p1, start=start, end=end)
        anchors[wname] = {"L0": a0, "L1": a1}

    rows = []
    for vname, patch in VARIANTS.items():
        for wname, (start, end, base_path) in WINDOWS.items():
            base = load_profile(base_path) if base_path else copy.deepcopy(l2_src)
            p = copy.deepcopy(base)
            p["watchdog"] = copy.deepcopy(l2_src.get("watchdog") or {})
            p["watchdog"]["enabled"] = True
            h = p["watchdog"].setdefault("hunter", {})
            h["enabled"] = True
            h.update(patch)
            print(f"run {vname} {wname}...", flush=True)
            r = _run(p, start=start, end=end)
            l0 = anchors[wname]["L0"]["total_ret"]
            l1 = anchors[wname]["L1"]["total_ret"]
            vs_l0 = r["total_ret"] / l0 if abs(l0) > 1e-12 else float("nan")
            vs_l1 = r["total_ret"] / l1 if abs(l1) > 1e-12 else float("nan")
            rows.append(
                {
                    "variant": vname,
                    "window": wname,
                    "vs_L0": vs_l0,
                    "vs_L1": vs_l1,
                    "pass_vs_L0": bool(vs_l0 >= float(args.gate_vs_l0)),
                    "ge_L1": bool(r["total_ret"] + 1e-12 >= l1),
                    "patch": json.dumps(patch),
                    **r,
                }
            )

    sb = pd.DataFrame(rows)
    sb.to_csv(out / "scoreboard.csv", index=False)

    # per-variant rollup
    roll = []
    for vname, g in sb.groupby("variant"):
        strong = g[g.window == "strong"].iloc[0]
        weak = g[g.window == "weak"].iloc[0]
        oos = g[g.window == "oos_2025h2"].iloc[0]
        ok = bool(
            strong["pass_vs_L0"]
            and weak["pass_vs_L0"]
            and oos["ge_L1"]
            and oos["pass_vs_L0"]
        )
        roll.append(
            {
                "variant": vname,
                "strong_ret": strong["total_ret"],
                "strong_vs_L0": strong["vs_L0"],
                "weak_ret": weak["total_ret"],
                "weak_vs_L0": weak["vs_L0"],
                "oos_ret": oos["total_ret"],
                "oos_vs_L0": oos["vs_L0"],
                "oos_vs_L1": oos["vs_L1"],
                "oos_ge_L1": bool(oos["ge_L1"]),
                "hunt_mean_strong": strong["hunt_mean_ret"],
                "hunt_mean_oos": oos["hunt_mean_ret"],
                "promote_candidate": ok,
            }
        )
    roll_df = pd.DataFrame(roll).sort_values(
        ["promote_candidate", "oos_vs_L1", "strong_vs_L0"], ascending=[False, False, False]
    )
    roll_df.to_csv(out / "rollup.csv", index=False)

    cands = roll_df[roll_df["promote_candidate"]]["variant"].tolist()
    summary = {
        "gate_vs_L0": float(args.gate_vs_l0),
        "goal": "oos_2025h2 L2 >= L1 and dual-window vs L0 >= gate",
        "anchors": {
            w: {k: v["total_ret"] for k, v in a.items()} for w, a in anchors.items()
        },
        "promote_candidates": cands,
        "best": roll_df.iloc[0].to_dict() if len(roll_df) else None,
        "verdict": (
            "PASS_P21_CANDIDATE"
            if cands
            else "FAIL_P21_NO_CANDIDATE"
        ),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str))
    print(roll_df.to_string(index=False))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Ablate faster cut-loss (mtm_floor / mf_flip) on extend_mtm_only peer3.

Stack soft exits on hold_extend via exit_mode='hold_extend+…'.
Also compare standalone soft exits and full_day overlays.
"""
from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from maga7.common.config import load_profile
from maga7.common.replay import run_offline_replay

PEER3 = (
    ROOT
    / "maga7"
    / "CONFIG"
    / "strategy_profiles"
    / "single_qqq_open_ladder_atm5otm_t30_rails_p20_googl_peer3_v1.json"
)

FOCUS = {"2026-07-07", "2026-07-08", "2026-07-09"}


def _extend_base(trade: dict[str, Any]) -> None:
    trade["hold_minutes"] = 30
    trade["hold_extend_minutes"] = 45
    trade["hold_extend_mtm_min"] = 0.0
    trade["hold_extend_require_mf"] = False
    trade["bar_availability_delay_seconds"] = 60
    trade["day_circuit"] = None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--profile", default=str(PEER3))
    p.add_argument("--start-date", default="2026-05-01")
    p.add_argument("--end-date", default="2026-07-13")
    p.add_argument("--tag", default="early_cut_ablation_extend_mtm_peer3_may_jul")
    args = p.parse_args()

    base = load_profile(args.profile)
    base["date_range"]["start"] = args.start_date
    base["date_range"]["end"] = args.end_date
    _extend_base(base.setdefault("trade", {}))
    base["trade"]["exit_mode"] = "hold_extend"

    out = Path(base["_paths"]["results_dir"]) / args.tag
    out.mkdir(parents=True, exist_ok=True)

    full_day = {
        "event_calendar_block": True,
        "event_calendar": "default",
        "event_blackout_sessions": 0,
    }

    variants: list[tuple[str, dict[str, Any], dict[str, Any]]] = [
        ("extend_mtm_only", {}, {}),
        # stack on extend
        ("ext_mtm_floor_h10", {"exit_mode": "hold_extend+mtm_floor", "exit_min_hold_minutes": 10}, {}),
        ("ext_mtm_floor_h5", {"exit_mode": "hold_extend+mtm_floor", "exit_min_hold_minutes": 5}, {}),
        ("ext_mf_flip_g60", {"exit_mode": "hold_extend+mf_flip", "exit_mf_grace_seconds": 60}, {}),
        ("ext_mf_reversal_h10", {"exit_mode": "hold_extend+mf_reversal", "exit_min_hold_minutes": 10}, {}),
        ("ext_floor_h5_mf_flip", {
            "exit_mode": "hold_extend+mtm_floor+mf_flip",
            "exit_min_hold_minutes": 5,
            "exit_mf_grace_seconds": 60,
        }, {}),
        # replace extend (pure soft)
        ("mtm_floor_h10_t30", {"exit_mode": "mtm_floor", "exit_min_hold_minutes": 10, "hold_extend_minutes": 30}, {}),
        ("mf_flip_g60_t30", {"exit_mode": "mf_flip", "exit_mf_grace_seconds": 60, "hold_extend_minutes": 30}, {}),
        # best calendar + best early cut (filled after first pass if needed)
        ("full_day", {}, full_day),
        ("full_day_ext_floor_h5", {
            "exit_mode": "hold_extend+mtm_floor",
            "exit_min_hold_minutes": 5,
        }, full_day),
        ("full_day_ext_mf_flip", {
            "exit_mode": "hold_extend+mf_flip",
            "exit_mf_grace_seconds": 60,
        }, full_day),
    ]

    scoreboard: list[dict[str, Any]] = []
    for name, trade_over, reg_over in variants:
        prof = deepcopy(base)
        for k, v in trade_over.items():
            prof["trade"][k] = v
        prof.setdefault("regime", {})
        for k, v in reg_over.items():
            prof["regime"][k] = v
        print(f"==> {name}", flush=True)
        result = run_offline_replay(prof, scheme="single")
        s = result["summary"]
        tr = result["trades"]
        focus = tr.loc[tr["date"].astype(str).isin(FOCUS)] if len(tr) else tr
        reason_vc = (
            tr["reason"].astype(str).value_counts().to_dict() if len(tr) else {}
        )
        focus_detail = []
        if len(focus):
            for r in focus.itertuples(index=False):
                focus_detail.append(
                    f"{r.date} {r.symbol} {r.dir} ret={float(r.ret):+.1%} {r.reason}"
                )
        row = {
            "name": name,
            "exit_mode": s.get("exit_mode"),
            "total_ret": float(s["total_ret"]),
            "maxdd": float(s["maxdd"]),
            "n_trades": int(s["n_trades"]),
            "trade_win": float(s["trade_win"]),
            "trade_exp": float(s["trade_exp"]),
            "focus_ret_sum": float(focus["ret"].sum()) if len(focus) else 0.0,
            "n_mtm_floor": int(reason_vc.get("MTM_FLOOR", 0)),
            "n_mf_flip": int(reason_vc.get("MF_FLIP", 0)),
            "reason_counts": reason_vc,
            "focus_trades": focus_detail,
            "end_equity": float(s["end_equity"]),
        }
        scoreboard.append(row)
        sub = out / name
        sub.mkdir(parents=True, exist_ok=True)
        (sub / "summary.json").write_text(json.dumps(s, indent=2), encoding="utf-8")
        tr.to_csv(sub / "trades.csv", index=False)
        result["daily"].to_csv(sub / "daily.csv", index=False)
        print(
            f"    ret={row['total_ret']:+.1%} dd={row['maxdd']:.1%} n={row['n_trades']} "
            f"floor={row['n_mtm_floor']} flip={row['n_mf_flip']} focus={row['focus_ret_sum']:+.2f}",
            flush=True,
        )
        for line in focus_detail:
            print(f"      {line}", flush=True)

    base_ret = scoreboard[0]["total_ret"]
    for row in scoreboard:
        row["uplift_vs_extend_pp"] = float(row["total_ret"] - base_ret)

    (out / "scoreboard.json").write_text(
        json.dumps(scoreboard, indent=2, default=str), encoding="utf-8"
    )
    pd.DataFrame(
        [
            {
                "name": r["name"],
                "exit_mode": r["exit_mode"],
                "total_ret": r["total_ret"],
                "maxdd": r["maxdd"],
                "n_trades": r["n_trades"],
                "n_mtm_floor": r["n_mtm_floor"],
                "n_mf_flip": r["n_mf_flip"],
                "focus_ret_sum": r["focus_ret_sum"],
                "uplift_vs_extend_pp": r["uplift_vs_extend_pp"],
                "end_equity": r["end_equity"],
            }
            for r in scoreboard
        ]
    ).to_csv(out / "scoreboard.csv", index=False)

    summary = {
        "period": f"{args.start_date}..{args.end_date}",
        "base": "extend_mtm_only peer3 day_circuit=null",
        "best_ret": max(scoreboard, key=lambda r: r["total_ret"])["name"],
        "best_dd": max(scoreboard, key=lambda r: r["maxdd"])["name"],
        "best_focus": max(scoreboard, key=lambda r: r["focus_ret_sum"])["name"],
        "scoreboard": [
            {
                "name": r["name"],
                "total_ret": r["total_ret"],
                "maxdd": r["maxdd"],
                "n_trades": r["n_trades"],
                "n_mtm_floor": r["n_mtm_floor"],
                "n_mf_flip": r["n_mf_flip"],
                "focus_ret_sum": r["focus_ret_sum"],
                "uplift_vs_extend_pp": r["uplift_vs_extend_pp"],
            }
            for r in scoreboard
        ],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()

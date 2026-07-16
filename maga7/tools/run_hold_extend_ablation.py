#!/usr/bin/env python3
"""Ablate conditional hold extend (T30→T45) on peer3 causal baseline.

Variants (all delay=60, rails, Mag7+GOOGL peer_min3):
  - baseline_t30_rails
  - t45_rails (unconditional upper-bound control)
  - extend_mtm_mf (MTM>=0 + mf10 aligned)
  - extend_mtm_only (MTM>=0, ignore mf)
  - extend_mtm5_mf (MTM>=5% + mf10 aligned)
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


def _metrics(summary: dict[str, Any], trades: pd.DataFrame) -> dict[str, Any]:
    reasons = {}
    if trades is not None and not trades.empty and "reason" in trades.columns:
        reasons = {str(k): int(v) for k, v in trades["reason"].value_counts().items()}
    n_ext = int(reasons.get("T+45", 0))
    return {
        "total_ret": float(summary["total_ret"]),
        "maxdd": float(summary["maxdd"]),
        "n_trades": int(summary["n_trades"]),
        "trade_win": float(summary["trade_win"]),
        "trade_exp": float(summary["trade_exp"]),
        "end_equity": float(summary["end_equity"]),
        "exit_mode": summary.get("exit_mode"),
        "n_peer_block": summary.get("n_peer_block"),
        "reason_counts": reasons,
        "n_t45_exits": n_ext,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--profile", default=str(PEER3))
    p.add_argument("--start-date", default="2026-05-01")
    p.add_argument("--end-date", default="2026-07-13")
    p.add_argument("--tag", default="hold_extend_ablation_mag7_googl_peer3_may_jul")
    args = p.parse_args()

    base = load_profile(args.profile)
    base["date_range"]["start"] = args.start_date
    base["date_range"]["end"] = args.end_date
    base.setdefault("trade", {})["bar_availability_delay_seconds"] = 60

    out = Path(base["_paths"]["results_dir"]) / args.tag
    out.mkdir(parents=True, exist_ok=True)

    variants: list[tuple[str, dict[str, Any]]] = [
        ("baseline_t30_rails", {"exit_mode": "none", "hold_minutes": 30}),
        ("t45_rails", {"exit_mode": "none", "hold_minutes": 45}),
        (
            "extend_mtm_mf",
            {
                "exit_mode": "hold_extend",
                "hold_minutes": 30,
                "hold_extend_minutes": 45,
                "hold_extend_mtm_min": 0.0,
                "hold_extend_require_mf": True,
            },
        ),
        (
            "extend_mtm_only",
            {
                "exit_mode": "hold_extend",
                "hold_minutes": 30,
                "hold_extend_minutes": 45,
                "hold_extend_mtm_min": 0.0,
                "hold_extend_require_mf": False,
            },
        ),
        (
            "extend_mtm5_mf",
            {
                "exit_mode": "hold_extend",
                "hold_minutes": 30,
                "hold_extend_minutes": 45,
                "hold_extend_mtm_min": 0.05,
                "hold_extend_require_mf": True,
            },
        ),
    ]

    scoreboard: list[dict[str, Any]] = []
    for name, overrides in variants:
        prof = deepcopy(base)
        for k, v in overrides.items():
            prof["trade"][k] = v
        print(f"==> {name}", flush=True)
        result = run_offline_replay(prof, scheme="single")
        row = {"name": name, **overrides, **_metrics(result["summary"], result["trades"])}
        scoreboard.append(row)
        sub = out / name
        sub.mkdir(parents=True, exist_ok=True)
        (sub / "summary.json").write_text(
            json.dumps(result["summary"], indent=2), encoding="utf-8"
        )
        result["trades"].to_csv(sub / "trades.csv", index=False)
        result["daily"].to_csv(sub / "daily.csv", index=False)
        print(
            f"    ret={row['total_ret']:+.1%} dd={row['maxdd']:.1%} "
            f"n={row['n_trades']} t45={row['n_t45_exits']}",
            flush=True,
        )

    base_ret = next(r["total_ret"] for r in scoreboard if r["name"] == "baseline_t30_rails")
    for row in scoreboard:
        row["uplift_vs_baseline_pp"] = float(row["total_ret"] - base_ret)

    (out / "scoreboard.json").write_text(json.dumps(scoreboard, indent=2), encoding="utf-8")
    pd.DataFrame(
        [
            {
                "name": r["name"],
                "exit_mode": r.get("exit_mode"),
                "hold_minutes": r.get("hold_minutes"),
                "hold_extend_minutes": r.get("hold_extend_minutes"),
                "hold_extend_mtm_min": r.get("hold_extend_mtm_min"),
                "hold_extend_require_mf": r.get("hold_extend_require_mf"),
                "total_ret": r["total_ret"],
                "maxdd": r["maxdd"],
                "n_trades": r["n_trades"],
                "trade_win": r["trade_win"],
                "trade_exp": r["trade_exp"],
                "n_t45_exits": r["n_t45_exits"],
                "uplift_vs_baseline_pp": r["uplift_vs_baseline_pp"],
            }
            for r in scoreboard
        ]
    ).to_csv(out / "scoreboard.csv", index=False)

    best = max(scoreboard, key=lambda r: r["total_ret"])
    summary = {
        "period": f"{args.start_date}..{args.end_date}",
        "delay_seconds": 60,
        "profile": str(args.profile),
        "baseline_ret": base_ret,
        "best_by_ret": {
            "name": best["name"],
            "total_ret": best["total_ret"],
            "maxdd": best["maxdd"],
            "n_trades": best["n_trades"],
            "n_t45_exits": best["n_t45_exits"],
        },
        "note": "Conditional extend at T30 if MTM/mf gate passes; rails kept throughout.",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

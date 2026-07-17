#!/usr/bin/env python3
"""Ablate event_calendar_block on extend_mtm_only (peer3, no day_circuit).

Base matches archived hold_extend extend_mtm_only daily.csv (+401% / 53 trades).

Variants:
  - extend_mtm_only
  - core_day          (NVDA earn / SpaceX IPO / FOMC decision)
  - core_plus1        (core + next session)
  - full_day          (yield spike week + IPO + FOMC cluster)
  - full_plus1
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

FOCUS = {
    "2026-05-20",
    "2026-05-21",
    "2026-06-12",
    "2026-06-16",
    "2026-06-18",
    "2026-07-07",
    "2026-07-08",
    "2026-07-09",
}


def _extend_mtm_only(trade: dict[str, Any]) -> None:
    trade["exit_mode"] = "hold_extend"
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
    p.add_argument("--tag", default="event_calendar_ablation_extend_mtm_peer3_may_jul")
    args = p.parse_args()

    base = load_profile(args.profile)
    base["date_range"]["start"] = args.start_date
    base["date_range"]["end"] = args.end_date
    _extend_mtm_only(base.setdefault("trade", {}))

    out = Path(base["_paths"]["results_dir"]) / args.tag
    out.mkdir(parents=True, exist_ok=True)

    variants: list[tuple[str, dict[str, Any]]] = [
        ("extend_mtm_only", {}),
        (
            "core_day",
            {
                "event_calendar_block": True,
                "event_calendar": "core",
                "event_blackout_sessions": 0,
            },
        ),
        (
            "core_plus1",
            {
                "event_calendar_block": True,
                "event_calendar": "core",
                "event_blackout_sessions": 1,
            },
        ),
        (
            "full_day",
            {
                "event_calendar_block": True,
                "event_calendar": "default",
                "event_blackout_sessions": 0,
            },
        ),
        (
            "full_plus1",
            {
                "event_calendar_block": True,
                "event_calendar": "default",
                "event_blackout_sessions": 1,
            },
        ),
    ]

    scoreboard: list[dict[str, Any]] = []
    for name, reg_over in variants:
        prof = deepcopy(base)
        prof.setdefault("regime", {})
        for k, v in reg_over.items():
            prof["regime"][k] = v
        print(f"==> {name} {reg_over}", flush=True)
        result = run_offline_replay(prof, scheme="single")
        s = result["summary"]
        tr = result["trades"]
        focus = float(tr.loc[tr["date"].isin(FOCUS), "ret"].sum()) if len(tr) else 0.0
        # which focus dates still traded
        focus_left = sorted(set(tr.loc[tr["date"].isin(FOCUS), "date"].astype(str)))
        row = {
            "name": name,
            **reg_over,
            "total_ret": float(s["total_ret"]),
            "maxdd": float(s["maxdd"]),
            "n_trades": int(s["n_trades"]),
            "trade_win": float(s["trade_win"]),
            "trade_exp": float(s["trade_exp"]),
            "n_event_block": int(s.get("n_event_block") or 0),
            "event_blackout_dates": s.get("event_blackout_dates") or [],
            "focus_cluster_ret_sum": focus,
            "focus_dates_still_traded": focus_left,
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
            f"event_days={row['n_event_block']} focus={focus:+.2f} "
            f"blackout={row['event_blackout_dates']}",
            flush=True,
        )

    base_ret = scoreboard[0]["total_ret"]
    for row in scoreboard:
        row["uplift_vs_extend_pp"] = float(row["total_ret"] - base_ret)

    (out / "scoreboard.json").write_text(json.dumps(scoreboard, indent=2), encoding="utf-8")
    pd.DataFrame(
        [
            {
                "name": r["name"],
                "event_calendar": r.get("event_calendar"),
                "event_blackout_sessions": r.get("event_blackout_sessions"),
                "total_ret": r["total_ret"],
                "maxdd": r["maxdd"],
                "n_trades": r["n_trades"],
                "n_event_block": r["n_event_block"],
                "focus_cluster_ret_sum": r["focus_cluster_ret_sum"],
                "uplift_vs_extend_pp": r["uplift_vs_extend_pp"],
                "end_equity": r["end_equity"],
            }
            for r in scoreboard
        ]
    ).to_csv(out / "scoreboard.csv", index=False)

    best = max(scoreboard, key=lambda r: r["total_ret"])
    best_dd = max(scoreboard, key=lambda r: r["maxdd"])
    summary = {
        "period": f"{args.start_date}..{args.end_date}",
        "base": "extend_mtm_only peer3 day_circuit=null",
        "extend_baseline_ret": base_ret,
        "best_by_ret": {k: best[k] for k in ("name", "total_ret", "maxdd", "n_trades")},
        "best_by_maxdd": {k: best_dd[k] for k in ("name", "total_ret", "maxdd", "n_trades")},
        "note": (
            "core=NVDA earnings AH + SpaceX IPO + FOMC decision; "
            "full adds 05-19 yield spike, 05-21 post-earn, 06-16/18 FOMC cluster."
        ),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

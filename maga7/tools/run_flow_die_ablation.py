#!/usr/bin/env python3
"""Ablate post-entry flow / flow+MTM soft exits vs Mag7+GOOGL T+30 rails.

``flow_die``: after min_hold, exit if cum favorable stock net$ <= floor.
``flow_mtm``: same, but only when option MTM ret also <= mtm_floor.
  UP = inflow (net$); DN = outflow (-net$).
"""
from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from maga7.common.config import load_profile
from maga7.common.replay import run_offline_replay

BASE = (
    ROOT
    / "maga7"
    / "CONFIG"
    / "strategy_profiles"
    / "single_qqq_open_ladder_atm5otm_t30_rails_p20_googl_v1.json"
)

STREAK_DAYS = {
    "2026-05-06",
    "2026-05-07",
    "2026-05-08",
    "2026-05-11",
    "2026-05-20",
    "2026-05-21",
    "2026-05-22",
    "2026-07-07",
    "2026-07-08",
    "2026-07-09",
}


def _run(profile: dict[str, Any], scheme: str = "single") -> dict[str, Any]:
    result = run_offline_replay(profile, scheme=scheme)
    s = result["summary"]
    trades = result["trades"]
    reasons = (
        trades["reason"].value_counts().to_dict()
        if not trades.empty and "reason" in trades.columns
        else {}
    )
    streak_ret = None
    if not trades.empty:
        td = trades.copy()
        td["date"] = td["date"].astype(str)
        streak = td[td["date"].isin(STREAK_DAYS)]
        streak_ret = float(streak["ret"].mean()) if not streak.empty else None
    daily = result["daily"]
    streak_day_ret = None
    if not daily.empty and "date" in daily.columns:
        d = daily.copy()
        d["date"] = d["date"].astype(str)
        sd = d[d["date"].isin(STREAK_DAYS)]
        if not sd.empty and "day_ret" in sd.columns:
            streak_day_ret = float((1 + sd["day_ret"].fillna(0)).prod() - 1)
    return {
        "total_ret": float(s["total_ret"]),
        "maxdd": float(s["maxdd"]),
        "n_trades": int(s["n_trades"]),
        "trade_win": float(s["trade_win"]),
        "trade_exp": float(s["trade_exp"]),
        "end_equity": float(s["end_equity"]),
        "reasons": {str(k): int(v) for k, v in reasons.items()},
        "streak_trade_mean_ret": streak_ret,
        "streak_days_compound": streak_day_ret,
        "summary": s,
        "trades": trades,
        "daily": daily,
    }


def _variants() -> list[tuple[str, dict[str, Any]]]:
    return [
        ("baseline_rails", {}),
        (
            "flow_die_h5",
            {
                "trade": {
                    "exit_mode": "flow_die",
                    "exit_min_hold_minutes": 5,
                    "flow_cum_floor": 0.0,
                }
            },
        ),
        (
            "flow_mtm_h5",
            {
                "trade": {
                    "exit_mode": "flow_mtm",
                    "exit_min_hold_minutes": 5,
                    "flow_cum_floor": 0.0,
                    "mtm_floor_ret": 0.0,
                }
            },
        ),
        (
            "flow_mtm_h8",
            {
                "trade": {
                    "exit_mode": "flow_mtm",
                    "exit_min_hold_minutes": 8,
                    "flow_cum_floor": 0.0,
                    "mtm_floor_ret": 0.0,
                }
            },
        ),
        (
            "flow_mtm_h10",
            {
                "trade": {
                    "exit_mode": "flow_mtm",
                    "exit_min_hold_minutes": 10,
                    "flow_cum_floor": 0.0,
                    "mtm_floor_ret": 0.0,
                }
            },
        ),
        (
            "flow_mtm_h5_m5",
            {
                "trade": {
                    "exit_mode": "flow_mtm",
                    "exit_min_hold_minutes": 5,
                    "flow_cum_floor": 0.0,
                    "mtm_floor_ret": -0.05,
                }
            },
        ),
    ]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--profile", default=str(BASE))
    p.add_argument("--start-date", default="2026-05-01")
    p.add_argument("--end-date", default="2026-07-13")
    p.add_argument("--tag", default="flow_mtm_ablation_mag7_googl_may_jul")
    p.add_argument("--scheme", default="single")
    args = p.parse_args()

    base = load_profile(args.profile)
    base["date_range"]["start"] = args.start_date
    base["date_range"]["end"] = args.end_date
    base.setdefault("trade", {})["bar_availability_delay_seconds"] = 60
    out = Path(base["_paths"]["results_dir"]) / args.tag
    out.mkdir(parents=True, exist_ok=True)

    scoreboard: list[dict[str, Any]] = []
    for name, patch in _variants():
        prof = deepcopy(base)
        for section, vals in patch.items():
            prof.setdefault(section, {}).update(vals)
        print(f"[{args.tag}] {name} ...", flush=True)
        got = _run(prof, args.scheme)
        row = {
            "name": name,
            "exit_mode": prof.get("trade", {}).get("exit_mode", "none"),
            "exit_min_hold_minutes": prof.get("trade", {}).get("exit_min_hold_minutes"),
            "flow_cum_floor": prof.get("trade", {}).get("flow_cum_floor"),
            "mtm_floor_ret": prof.get("trade", {}).get("mtm_floor_ret"),
            **{
                k: got[k]
                for k in (
                    "total_ret",
                    "maxdd",
                    "n_trades",
                    "trade_win",
                    "trade_exp",
                    "end_equity",
                    "streak_trade_mean_ret",
                    "streak_days_compound",
                )
            },
            "reasons": got["reasons"],
        }
        scoreboard.append(row)
        print(
            f"  ret={row['total_ret']:+.2%} dd={row['maxdd']:.2%} "
            f"n={row['n_trades']} win={row['trade_win']:.1%} exp={row['trade_exp']:+.2%} "
            f"streak_tr={row['streak_trade_mean_ret']!s} "
            f"streak_days={row['streak_days_compound']!s} reasons={row['reasons']}",
            flush=True,
        )
        sub = out / name
        sub.mkdir(exist_ok=True)
        (sub / "summary.json").write_text(json.dumps(got["summary"], indent=2), encoding="utf-8")
        got["trades"].to_csv(sub / "trades.csv", index=False)
        got["daily"].to_csv(sub / "daily.csv", index=False)

    import pandas as pd

    pd.DataFrame(scoreboard).to_csv(out / "scoreboard.csv", index=False)
    baseline = scoreboard[0]
    best = max(scoreboard, key=lambda r: r["total_ret"])
    viable = [
        r
        for r in scoreboard[1:]
        if r["total_ret"] > baseline["total_ret"] and r["maxdd"] >= baseline["maxdd"] - 0.05
    ]
    pick = max(viable, key=lambda r: r["total_ret"]) if viable else None
    summary = {
        "period": f"{args.start_date}..{args.end_date}",
        "baseline": baseline,
        "best_by_ret": best,
        "pick_uplift_dd_ok": pick,
        "uplift_pp": {r["name"]: r["total_ret"] - baseline["total_ret"] for r in scoreboard[1:]},
        "promote": bool(pick is not None),
        "note": (
            "flow_die: cum fav net$<=floor. flow_mtm/flow_soft: same AND option MTM<=mtm_floor. "
            "Stock bars delayed by bar_availability_delay_seconds=60."
        ),
    }
    (out / "scoreboard.json").write_text(json.dumps(scoreboard, indent=2), encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in summary if k != "note"}, indent=2))
    print(f"→ {out}")


if __name__ == "__main__":
    main()

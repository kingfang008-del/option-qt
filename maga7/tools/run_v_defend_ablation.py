#!/usr/bin/env python3
"""Ablate anti-V entry (streak_max) and option-MTM floor exit vs Mag7+GOOGL rails.

Variants:
  A: streak_max=12 (ban streak>=13)
  B: exit_mode=mtm_floor after min_hold=10m when MTM ret<=0
  A+B: both

Does not replace the frozen Mag7 OTM5 T+30 baseline unless scoreboard wins.
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


def _run(profile: dict[str, Any], scheme: str = "single") -> dict[str, Any]:
    result = run_offline_replay(profile, scheme=scheme)
    s = result["summary"]
    reasons = (
        result["trades"]["reason"].value_counts().to_dict()
        if not result["trades"].empty and "reason" in result["trades"].columns
        else {}
    )
    return {
        "total_ret": float(s["total_ret"]),
        "maxdd": float(s["maxdd"]),
        "n_trades": int(s["n_trades"]),
        "trade_win": float(s["trade_win"]),
        "trade_exp": float(s["trade_exp"]),
        "end_equity": float(s["end_equity"]),
        "reasons": {str(k): int(v) for k, v in reasons.items()},
        "summary": s,
        "trades": result["trades"],
        "daily": result["daily"],
    }


def _variants() -> list[tuple[str, dict[str, Any]]]:
    return [
        ("baseline_mag7_googl", {}),
        ("A_streak_max12", {"signal": {"streak_max": 12}}),
        (
            "B_mtm_floor_h10",
            {
                "trade": {
                    "exit_mode": "mtm_floor",
                    "exit_min_hold_minutes": 10,
                    "mtm_floor_ret": 0.0,
                }
            },
        ),
        (
            "AB_streak12_mtm_floor",
            {
                "signal": {"streak_max": 12},
                "trade": {
                    "exit_mode": "mtm_floor",
                    "exit_min_hold_minutes": 10,
                    "mtm_floor_ret": 0.0,
                },
            },
        ),
        (
            "B2_mtm_floor_m5pct_h10",
            {
                "trade": {
                    "exit_mode": "mtm_floor",
                    "exit_min_hold_minutes": 10,
                    "mtm_floor_ret": -0.05,
                }
            },
        ),
    ]


def _apply(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    prof = deepcopy(base)
    for section, vals in patch.items():
        prof.setdefault(section, {}).update(vals)
    return prof


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--profile", default=str(BASE))
    p.add_argument("--start-date", default="2026-05-01")
    p.add_argument("--end-date", default="2026-07-13")
    p.add_argument("--tag", default="v_defend_ablation_mag7_googl_may_jul")
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
        prof = _apply(base, patch)
        print(f"[{args.tag}] {name} ...", flush=True)
        got = _run(prof, args.scheme)
        row = {
            "name": name,
            "streak_max": prof.get("signal", {}).get("streak_max"),
            "exit_mode": prof.get("trade", {}).get("exit_mode", "none"),
            "exit_min_hold_minutes": prof.get("trade", {}).get("exit_min_hold_minutes"),
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
                )
            },
            "reasons": got["reasons"],
        }
        scoreboard.append(row)
        print(
            f"  ret={row['total_ret']:+.2%} dd={row['maxdd']:.2%} "
            f"n={row['n_trades']} win={row['trade_win']:.1%} exp={row['trade_exp']:+.2%} "
            f"reasons={row['reasons']}",
            flush=True,
        )
        sub = out / name
        sub.mkdir(exist_ok=True)
        (sub / "summary.json").write_text(json.dumps(got["summary"], indent=2), encoding="utf-8")
        got["trades"].to_csv(sub / "trades.csv", index=False)
        got["daily"].to_csv(sub / "daily.csv", index=False)

    import pandas as pd

    sb = pd.DataFrame(scoreboard)
    sb.to_csv(out / "scoreboard.csv", index=False)
    baseline = scoreboard[0]
    best = max(scoreboard, key=lambda r: r["total_ret"])
    # prefer ret uplift with DD not worse than baseline by >5pp
    viable = [
        r
        for r in scoreboard[1:]
        if r["total_ret"] > baseline["total_ret"]
        and r["maxdd"] >= baseline["maxdd"] - 0.05
    ]
    pick = max(viable, key=lambda r: r["total_ret"]) if viable else None
    summary = {
        "period": f"{args.start_date}..{args.end_date}",
        "universe": base.get("symbols"),
        "baseline": baseline,
        "best_by_ret": best,
        "pick_uplift_dd_ok": pick,
        "uplift_pp": {
            r["name"]: r["total_ret"] - baseline["total_ret"] for r in scoreboard[1:]
        },
        "promote": bool(pick is not None and pick["total_ret"] > baseline["total_ret"]),
        "note": (
            "A=streak_max12; B=mtm_floor after 10m if option MTM<=0; "
            "B2=floor at -5%. Promote only if uplift with DD within 5pp of baseline."
        ),
    }
    (out / "scoreboard.json").write_text(json.dumps(scoreboard, indent=2), encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in summary if k != "note"}, indent=2))
    print(f"→ {out}")


if __name__ == "__main__":
    main()

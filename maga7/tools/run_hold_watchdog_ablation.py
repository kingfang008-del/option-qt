#!/usr/bin/env python3
"""Ablate mid-hold QQQ shock flatten (hold_watchdog) on L2+TT1-05 baseline.

Variants: off (baseline) vs qqq_adverse thresholds, optional MTM gate.
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

DEFAULT_PROFILE = (
    "maga7/CONFIG/strategy_profiles/"
    "single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1.json"
)


def _row(name: str, result: dict[str, Any]) -> dict[str, Any]:
    s = result["summary"]
    tr = result["trades"]
    reasons = (
        tr["reason"].value_counts().to_dict()
        if tr is not None and not tr.empty and "reason" in tr.columns
        else {}
    )
    n_shock = int(reasons.get("HOLD_SHOCK", 0))
    shock_ret = float("nan")
    if n_shock and not tr.empty:
        shock_ret = float(tr.loc[tr["reason"] == "HOLD_SHOCK", "ret"].mean())
    return {
        "arm": name,
        "n_trades": s.get("n_trades"),
        "total_ret": s.get("total_ret"),
        "maxdd": s.get("maxdd"),
        "trade_win": s.get("trade_win"),
        "trade_exp": s.get("trade_exp"),
        "n_hold_shock": n_shock,
        "shock_mean_ret": shock_ret,
        "end_equity": s.get("end_equity"),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--profile", default=DEFAULT_PROFILE)
    p.add_argument("--start-date", default="2026-05-01")
    p.add_argument("--end-date", default="2026-07-17")
    p.add_argument("--tag", default="hold_watchdog_ablation_peer3_may_jul")
    args = p.parse_args()

    base = load_profile(args.profile)
    base["date_range"]["start"] = args.start_date
    base["date_range"]["end"] = args.end_date
    out = Path(base["_paths"]["results_dir"]) / args.tag
    out.mkdir(parents=True, exist_ok=True)

    variants: list[tuple[str, dict[str, Any]]] = [
        ("00_off", {"enabled": False}),
        (
            "01_qqq80bps",
            {
                "enabled": True,
                "qqq_adverse_from_entry": 0.008,
                "min_hold_seconds": 60,
                "require_option_mtm_max": None,
            },
        ),
        (
            "02_qqq100bps",
            {
                "enabled": True,
                "qqq_adverse_from_entry": 0.010,
                "min_hold_seconds": 60,
                "require_option_mtm_max": None,
            },
        ),
        (
            "03_qqq150bps",
            {
                "enabled": True,
                "qqq_adverse_from_entry": 0.015,
                "min_hold_seconds": 60,
                "require_option_mtm_max": None,
            },
        ),
        (
            "04_qqq80bps_mtm0",
            {
                "enabled": True,
                "qqq_adverse_from_entry": 0.008,
                "min_hold_seconds": 60,
                "require_option_mtm_max": 0.0,
            },
        ),
        (
            "05_qqq80bps_grace180",
            {
                "enabled": True,
                "qqq_adverse_from_entry": 0.008,
                "min_hold_seconds": 180,
                "require_option_mtm_max": None,
            },
        ),
    ]

    scoreboard: list[dict[str, Any]] = []
    for name, hwd in variants:
        prof = deepcopy(base)
        prof.setdefault("trade", {})["hold_watchdog"] = hwd
        print(f"==> {name}", flush=True)
        result = run_offline_replay(prof, scheme="single")
        arm = out / name
        arm.mkdir(parents=True, exist_ok=True)
        (arm / "summary.json").write_text(
            json.dumps(result["summary"], indent=2, default=str), encoding="utf-8"
        )
        result["trades"].to_csv(arm / "trades.csv", index=False)
        result["daily"].to_csv(arm / "daily.csv", index=False)
        row = _row(name, result)
        scoreboard.append(row)
        print(
            f"    total={row['total_ret']:+.1%} dd={row['maxdd']:.1%} "
            f"n={row['n_trades']} shock={row['n_hold_shock']} "
            f"shock_exp={row['shock_mean_ret']}",
            flush=True,
        )
        pd.DataFrame(scoreboard).to_csv(out / "scoreboard.csv", index=False)

    sb = pd.DataFrame(scoreboard)
    if not sb.empty and "00_off" in set(sb["arm"]):
        base_ret = float(sb.loc[sb["arm"] == "00_off", "total_ret"].iloc[0])
        sb["d_total"] = sb["total_ret"] - base_ret
        sb.to_csv(out / "scoreboard.csv", index=False)
    (out / "scoreboard.json").write_text(
        sb.to_json(orient="records", indent=2), encoding="utf-8"
    )
    (out / "README.md").write_text(
        "\n".join(
            [
                "# Hold Watchdog ablation (QQQ adverse from entry)",
                "",
                f"Window {args.start_date} → {args.end_date}",
                "",
                sb.to_markdown(index=False, floatfmt=".4f")
                if hasattr(sb, "to_markdown")
                else sb.to_string(index=False),
                "",
                "Acceptance: lift MaxDD / cut toxic days without wrecking total_ret "
                "(prefer ≥95% of off-arm total_ret).",
            ]
        ),
        encoding="utf-8",
    )
    print("wrote", out / "scoreboard.csv", flush=True)


if __name__ == "__main__":
    main()

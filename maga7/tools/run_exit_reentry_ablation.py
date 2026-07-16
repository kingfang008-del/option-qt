#!/usr/bin/env python3
"""Ablate exit / reentry under causal delay=60.

Focus: first-only vs m5 reentry, mf_flip vs fixed T+hold, with/without TP/SL rails.
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

PROD = (
    ROOT
    / "maga7"
    / "CONFIG"
    / "strategy_profiles"
    / "m5c_qqq_onlywin_open_ladder_atm5otm_mf_flip_p20_v1.json"
)


def _run(profile: dict[str, Any], scheme: str) -> dict[str, Any]:
    result = run_offline_replay(profile, scheme=scheme)
    s = result["summary"]
    return {
        "total_ret": float(s["total_ret"]),
        "maxdd": float(s["maxdd"]),
        "n_trades": int(s["n_trades"]),
        "trade_win": float(s["trade_win"]),
        "trade_exp": float(s["trade_exp"]),
        "end_equity": float(s["end_equity"]),
        "exit_mode": s.get("exit_mode"),
        "scheme": scheme,
        "trades": result["trades"],
        "daily": result["daily"],
        "summary": s,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--profile", default=str(PROD))
    p.add_argument("--start-date", default="2026-05-01")
    p.add_argument("--end-date", default="2026-07-13")
    p.add_argument("--tag", default="exit_reentry_ablation_may_jul")
    args = p.parse_args()

    base = load_profile(args.profile)
    base["date_range"]["start"] = args.start_date
    base["date_range"]["end"] = args.end_date
    base.setdefault("trade", {})["bar_availability_delay_seconds"] = 60

    out = Path(base["_paths"]["results_dir"]) / args.tag
    out.mkdir(parents=True, exist_ok=True)

    # name -> (scheme, trade overrides)
    variants: list[tuple[str, str, dict[str, Any]]] = [
        ("prod_m5c_mf_flip", "m5_circuit", {}),
        ("single_mf_flip", "single", {"exit_mode": "mf_flip"}),
        ("single_t30_rails", "single", {"exit_mode": "none", "hold_minutes": 30}),
        ("single_t60_rails", "single", {"exit_mode": "none", "hold_minutes": 60}),
        (
            "single_t30_norails",
            "single",
            {"exit_mode": "none", "hold_minutes": 30, "tp_mult": 999.0, "sl_mult": 0.0},
        ),
        (
            "single_t60_norails",
            "single",
            {"exit_mode": "none", "hold_minutes": 60, "tp_mult": 999.0, "sl_mult": 0.0},
        ),
        (
            "m5c_onlywin_t30_rails",
            "m5_circuit",
            {"exit_mode": "none", "hold_minutes": 30},
        ),
        (
            "m5c_onlywin_t30_norails",
            "m5_circuit",
            {"exit_mode": "none", "hold_minutes": 30, "tp_mult": 999.0, "sl_mult": 0.0},
        ),
        (
            "m5c_allreenter_t30_norails",
            "m5_circuit",
            {
                "exit_mode": "none",
                "hold_minutes": 30,
                "tp_mult": 999.0,
                "sl_mult": 0.0,
                "reentry_mode": "always",
                "only_reenter_after_win": False,
            },
        ),
        (
            "single_t30_norails_volz20",
            "single",
            {
                "exit_mode": "none",
                "hold_minutes": 30,
                "tp_mult": 999.0,
                "sl_mult": 0.0,
            },
        ),
    ]

    scoreboard = []
    best = None
    for name, scheme, overrides in variants:
        prof = deepcopy(base)
        for k, v in overrides.items():
            if k in ("vol_z_min",):
                continue
            prof.setdefault("trade", {})[k] = v
        if name.endswith("volz20"):
            prof.setdefault("signal", {})["vol_z_min"] = 2.0
        print(f"running {name} ...", flush=True)
        got = _run(prof, scheme)
        row = {
            "name": name,
            "scheme": scheme,
            "hold_minutes": int(prof["trade"].get("hold_minutes", 30)),
            "exit_mode": str(prof["trade"].get("exit_mode", "mf_flip")),
            "tp_mult": float(prof["trade"].get("tp_mult", 1.6)),
            "sl_mult": float(prof["trade"].get("sl_mult", 0.4)),
            "vol_z_min": float(prof.get("signal", {}).get("vol_z_min", 1.0)),
            "reentry_mode": str(prof["trade"].get("reentry_mode", "")),
            "total_ret": got["total_ret"],
            "maxdd": got["maxdd"],
            "n_trades": got["n_trades"],
            "trade_win": got["trade_win"],
            "trade_exp": got["trade_exp"],
            "end_equity": got["end_equity"],
        }
        scoreboard.append(row)
        print(
            f"  {name}: ret={row['total_ret']:+.2%} dd={row['maxdd']:.2%} "
            f"n={row['n_trades']} win={row['trade_win']:.1%} exp={row['trade_exp']:+.2%}",
            flush=True,
        )
        sub = out / name
        sub.mkdir(parents=True, exist_ok=True)
        (sub / "summary.json").write_text(json.dumps(got["summary"], indent=2), encoding="utf-8")
        got["trades"].to_csv(sub / "trades.csv", index=False)
        got["daily"].to_csv(sub / "daily.csv", index=False)
        if best is None or row["total_ret"] > best["total_ret"]:
            best = row

    # risk-adjusted pick: prefer higher ret with maxdd > -0.45, else best ret
    viable = [r for r in scoreboard if r["maxdd"] > -0.45]
    pick = max(viable, key=lambda r: r["total_ret"]) if viable else best

    summary = {
        "period": f"{args.start_date}..{args.end_date}",
        "delay_seconds": 60,
        "baseline": next(r for r in scoreboard if r["name"] == "prod_m5c_mf_flip"),
        "best_by_ret": best,
        "pick_ret_with_dd_gt_-45": pick,
        "uplift_vs_baseline_pp": {
            "best_ret": (best["total_ret"] - scoreboard[0]["total_ret"]) if best else None,
            "pick": (pick["total_ret"] - scoreboard[0]["total_ret"]) if pick else None,
        },
        "note": (
            "All runs causal bar_availability_delay_seconds=60, open_ladder, concurrent p20. "
            "norails = tp_mult=999/sl_mult=0 pure time exit."
        ),
    }
    import pandas as pd

    pd.DataFrame(scoreboard).to_csv(out / "scoreboard.csv", index=False)
    (out / "scoreboard.json").write_text(json.dumps(scoreboard, indent=2), encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in summary if k != "note"}, indent=2))
    print(f"→ {out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Ablate day-over-day QQQ flip / put-VIXY gates on peer3 causal baseline.

Research only — does not mutate the frozen causal baseline profile.

Variants (trade rails / delay=60 fixed; only ``regime`` overrides):
  - baseline                 (qqq_align only)
  - put_vixy_z0              (DN requires vixy_z >= 0)
  - qqq_mf10_align
  - flip_block               (skip entries on QQQ from_prev day-sign flip)
  - flip_scale50             (half size on flip day)
  - flip_block_put0          (flip block + put z>=0)
  - flip_scale50_put0
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

LOSS_FOCUS = {
    "2026-05-20",
    "2026-05-21",
    "2026-06-12",
    "2026-06-16",
    "2026-06-18",
    "2026-07-07",
    "2026-07-08",
    "2026-07-09",
}


def _metrics(summary: dict[str, Any], trades: pd.DataFrame) -> dict[str, Any]:
    focus_ret = 0.0
    if trades is not None and not trades.empty:
        focus_ret = float(trades.loc[trades["date"].isin(LOSS_FOCUS), "ret"].sum())
    return {
        "total_ret": float(summary["total_ret"]),
        "maxdd": float(summary["maxdd"]),
        "n_trades": int(summary["n_trades"]),
        "trade_win": float(summary["trade_win"]),
        "trade_exp": float(summary["trade_exp"]),
        "end_equity": float(summary["end_equity"]),
        "n_regime_block": int(summary.get("n_regime_block") or 0),
        "n_regime_scale": int(summary.get("n_regime_scale") or 0),
        "focus_cluster_ret_sum": focus_ret,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--profile", default=str(PEER3))
    p.add_argument("--start-date", default="2026-05-01")
    p.add_argument("--end-date", default="2026-07-13")
    p.add_argument("--tag", default="regime_flip_ablation_mag7_googl_peer3_may_jul")
    p.add_argument(
        "--with-hold-extend",
        action="store_true",
        help="Also run the winning variant under exit_mode=hold_extend mtm-only.",
    )
    args = p.parse_args()

    base = load_profile(args.profile)
    base["date_range"]["start"] = args.start_date
    base["date_range"]["end"] = args.end_date
    base.setdefault("trade", {})["bar_availability_delay_seconds"] = 60

    out = Path(base["_paths"]["results_dir"]) / args.tag
    out.mkdir(parents=True, exist_ok=True)

    variants: list[tuple[str, dict[str, Any]]] = [
        ("baseline", {}),
        ("put_vixy_z0", {"put_vixy_z_min": 0.0}),
        ("qqq_mf10_align", {"qqq_mf10_align": True}),
        ("flip_block", {"qqq_day_flip_mode": "block"}),
        (
            "flip_scale50",
            {"qqq_day_flip_mode": "scale", "qqq_day_flip_scale": 0.5},
        ),
        (
            "flip_block_put0",
            {"qqq_day_flip_mode": "block", "put_vixy_z_min": 0.0},
        ),
        (
            "flip_scale50_put0",
            {
                "qqq_day_flip_mode": "scale",
                "qqq_day_flip_scale": 0.5,
                "put_vixy_z_min": 0.0,
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
        row = {"name": name, **reg_over, **_metrics(result["summary"], result["trades"])}
        scoreboard.append(row)
        sub = out / name
        sub.mkdir(parents=True, exist_ok=True)
        (sub / "summary.json").write_text(
            json.dumps(result["summary"], indent=2), encoding="utf-8"
        )
        result["trades"].to_csv(sub / "trades.csv", index=False)
        result["daily"].to_csv(sub / "daily.csv", index=False)
        print(
            f"    ret={row['total_ret']:+.1%} dd={row['maxdd']:.1%} n={row['n_trades']} "
            f"block={row['n_regime_block']} scale={row['n_regime_scale']} "
            f"focus_sum={row['focus_cluster_ret_sum']:+.2f}",
            flush=True,
        )

    if args.with_hold_extend:
        # Stack hold_extend mtm-only on baseline + best non-baseline by MaxDD.
        extend_names = ["baseline"]
        ranked = sorted(
            [r for r in scoreboard if r["name"] != "baseline"],
            key=lambda r: r["maxdd"],
            reverse=True,
        )
        if ranked:
            extend_names.append(ranked[0]["name"])
        for base_name in extend_names:
            reg_over = next(v for n, v in variants if n == base_name)
            name = f"extend_mtm_only__{base_name}"
            prof = deepcopy(base)
            prof.setdefault("regime", {})
            for k, v in reg_over.items():
                prof["regime"][k] = v
            prof["trade"]["exit_mode"] = "hold_extend"
            prof["trade"]["hold_minutes"] = 30
            prof["trade"]["hold_extend_minutes"] = 45
            prof["trade"]["hold_extend_mtm_min"] = 0.0
            prof["trade"]["hold_extend_require_mf"] = False
            print(f"==> {name} {reg_over}", flush=True)
            result = run_offline_replay(prof, scheme="single")
            row = {
                "name": name,
                **reg_over,
                "exit_mode": "hold_extend",
                **_metrics(result["summary"], result["trades"]),
            }
            scoreboard.append(row)
            sub = out / name
            sub.mkdir(parents=True, exist_ok=True)
            (sub / "summary.json").write_text(
                json.dumps(result["summary"], indent=2), encoding="utf-8"
            )
            result["trades"].to_csv(sub / "trades.csv", index=False)
            result["daily"].to_csv(sub / "daily.csv", index=False)
            print(
                f"    ret={row['total_ret']:+.1%} dd={row['maxdd']:.1%} n={row['n_trades']}",
                flush=True,
            )

    base_ret = next(r["total_ret"] for r in scoreboard if r["name"] == "baseline")
    for row in scoreboard:
        row["uplift_vs_baseline_pp"] = float(row["total_ret"] - base_ret)

    (out / "scoreboard.json").write_text(json.dumps(scoreboard, indent=2), encoding="utf-8")
    pd.DataFrame(scoreboard).to_csv(out / "scoreboard.csv", index=False)

    best = max(scoreboard, key=lambda r: r["total_ret"])
    best_dd = max(scoreboard, key=lambda r: r["maxdd"])  # maxdd less negative is better
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
        },
        "best_by_maxdd": {
            "name": best_dd["name"],
            "total_ret": best_dd["total_ret"],
            "maxdd": best_dd["maxdd"],
            "n_trades": best_dd["n_trades"],
        },
        "note": (
            "qqq_day_flip compares sign(qqq_from_prev) vs prior session end; "
            "put_vixy_z_min blocks DN when VIXY z is compressed. Research only."
        ),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

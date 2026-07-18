#!/usr/bin/env python3
"""Ablate mf10 + fast companion window (early_on_mf_fast) on peer3 causal baseline.

Variants keep trade rails / delay=60 / Mag7+GOOGL peer_min3 fixed; only signal
timing changes:

  - baseline_mf10          (streak_min=8, no early path)
  - early_mf3_s5           (mf_fast=3, streak_min_fast=5)
  - early_mf3_s6           (mf_fast=3, streak_min_fast=6)
  - early_mf5_s5           (mf_fast=5, streak_min_fast=5)
  - early_mf5_s6           (mf_fast=5, streak_min_fast=6)

Research only — does not mutate the frozen causal baseline profile.
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
    return {
        "total_ret": float(summary["total_ret"]),
        "maxdd": float(summary["maxdd"]),
        "n_trades": int(summary["n_trades"]),
        "trade_win": float(summary["trade_win"]),
        "trade_exp": float(summary["trade_exp"]),
        "end_equity": float(summary["end_equity"]),
        "n_peer_block": summary.get("n_peer_block"),
    }


def _sig_advance_vs_baseline(
    base_trades: pd.DataFrame, variant_trades: pd.DataFrame
) -> dict[str, Any]:
    """Median minutes earlier on overlapping (date, symbol, dir) fires."""
    keys = ["date", "symbol", "dir"]
    need = {"sig_ts", *keys}
    if base_trades is None or variant_trades is None:
        return {"n_overlap": 0}
    if base_trades.empty or variant_trades.empty:
        return {"n_overlap": 0}
    if not need.issubset(base_trades.columns) or not need.issubset(variant_trades.columns):
        return {"n_overlap": 0}
    b = base_trades[list(need)].copy()
    v = variant_trades[list(need)].copy()
    b["sig_ts"] = pd.to_datetime(b["sig_ts"])
    v["sig_ts"] = pd.to_datetime(v["sig_ts"])
    m = b.merge(v, on=keys, suffixes=("_base", "_var"))
    if m.empty:
        return {"n_overlap": 0}
    delta_min = (m["sig_ts_base"] - m["sig_ts_var"]).dt.total_seconds() / 60.0
    return {
        "n_overlap": int(len(m)),
        "median_min_earlier": float(delta_min.median()),
        "mean_min_earlier": float(delta_min.mean()),
        "n_strictly_earlier": int((delta_min > 0).sum()),
        "n_same_ts": int((delta_min == 0).sum()),
        "n_later": int((delta_min < 0).sum()),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--profile", default=str(PEER3))
    p.add_argument("--start-date", default="2026-05-01")
    p.add_argument("--end-date", default="2026-07-13")
    p.add_argument("--tag", default="mf_fast_early_ablation_mag7_googl_peer3_may_jul")
    args = p.parse_args()

    base = load_profile(args.profile)
    base["date_range"]["start"] = args.start_date
    base["date_range"]["end"] = args.end_date
    base.setdefault("trade", {})["bar_availability_delay_seconds"] = 60

    out = Path(base["_paths"]["results_dir"]) / args.tag
    out.mkdir(parents=True, exist_ok=True)

    variants: list[tuple[str, dict[str, Any]]] = [
        ("baseline_mf10", {}),
        (
            "early_mf3_s5",
            {
                "mf_fast_window": 3,
                "early_on_mf_fast": True,
                "streak_min_fast": 5,
            },
        ),
        (
            "early_mf3_s6",
            {
                "mf_fast_window": 3,
                "early_on_mf_fast": True,
                "streak_min_fast": 6,
            },
        ),
        (
            "early_mf5_s5",
            {
                "mf_fast_window": 5,
                "early_on_mf_fast": True,
                "streak_min_fast": 5,
            },
        ),
        (
            "early_mf5_s6",
            {
                "mf_fast_window": 5,
                "early_on_mf_fast": True,
                "streak_min_fast": 6,
            },
        ),
    ]

    scoreboard: list[dict[str, Any]] = []
    base_trades: pd.DataFrame | None = None
    for name, sig_overrides in variants:
        prof = deepcopy(base)
        for k, v in sig_overrides.items():
            prof["signal"][k] = v
        print(f"==> {name} {sig_overrides}", flush=True)
        result = run_offline_replay(prof, scheme="single")
        trades = result["trades"]
        if name == "baseline_mf10":
            base_trades = trades
        advance = _sig_advance_vs_baseline(base_trades, trades) if name != "baseline_mf10" else {}
        row = {
            "name": name,
            **sig_overrides,
            **_metrics(result["summary"], trades),
            **{f"sig_{k}": v for k, v in advance.items()},
        }
        scoreboard.append(row)
        sub = out / name
        sub.mkdir(parents=True, exist_ok=True)
        (sub / "summary.json").write_text(
            json.dumps(result["summary"], indent=2), encoding="utf-8"
        )
        trades.to_csv(sub / "trades.csv", index=False)
        result["daily"].to_csv(sub / "daily.csv", index=False)
        print(
            f"    ret={row['total_ret']:+.1%} dd={row['maxdd']:.1%} "
            f"n={row['n_trades']} advance={advance}",
            flush=True,
        )

    base_ret = next(r["total_ret"] for r in scoreboard if r["name"] == "baseline_mf10")
    for row in scoreboard:
        row["uplift_vs_baseline_pp"] = float(row["total_ret"] - base_ret)

    (out / "scoreboard.json").write_text(json.dumps(scoreboard, indent=2), encoding="utf-8")
    pd.DataFrame(scoreboard).to_csv(out / "scoreboard.csv", index=False)

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
        },
        "note": (
            "mf10 remains the primary streak window; mf_fast is a companion that "
            "can fire early when streak_min_fast <= streak < streak_min and fast "
            "window aligns. Not promoted to causal baseline."
        ),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

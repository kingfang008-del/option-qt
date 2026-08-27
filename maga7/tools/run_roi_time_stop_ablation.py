#!/usr/bin/env python3
"""Ablation: V0-style option ROI progress time-stops (MID/TIME)."""
from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from maga7.common.config import load_profile
from maga7.common.replay import run_offline_replay

PROFILE = (
    "maga7/CONFIG/strategy_profiles/"
    "single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1.json"
)

VARIANTS = {
    # Classic V0: 15m & 30m must show +5% option ROI
    "v0_15_05_30_05": {
        "enabled": True,
        "rails": [{"mins": 15, "min_roi": 0.05}, {"mins": 30, "min_roi": 0.05}],
    },
    # Softer: only require non-negative at 15m
    "mid15_roi0": {
        "enabled": True,
        "rails": [{"mins": 15, "min_roi": 0.0}],
    },
    # Soft progress: +2% by 15m
    "mid15_roi02": {
        "enabled": True,
        "rails": [{"mins": 15, "min_roi": 0.02}],
    },
    # Later gate only
    "late30_roi05": {
        "enabled": True,
        "rails": [{"mins": 30, "min_roi": 0.05}],
    },
    # Soft late: flat/red at 30m (before/at extend decision)
    "late30_roi0": {
        "enabled": True,
        "rails": [{"mins": 30, "min_roi": 0.0}],
    },
    # Combo with morning r5 scale
    "mid15_roi0_plus_r5": {
        "roi_time_stop": {
            "enabled": True,
            "rails": [{"mins": 15, "min_roi": 0.0}],
        },
        "morning_r5_scale": {
            "enabled": True,
            "tod_start": "10:31",
            "tod_end": "11:00",
            "lookback_bars": 5,
            "min_signed_ret": 0.0005,
            "scale": 0.5,
        },
    },
}


def _metrics(summary: dict, daily) -> dict:
    import pandas as pd

    d = daily.copy()
    n_le5 = int((pd.to_numeric(d["day_ret"], errors="coerce") <= -0.05).sum())
    return {
        "n_trades": summary.get("n_trades"),
        "total_ret": summary.get("total_ret"),
        "maxdd": summary.get("maxdd"),
        "day_win": summary.get("day_win"),
        "trade_win": summary.get("trade_win"),
        "n_day_le_m5": n_le5,
        "n_roi_time_stop": summary.get("n_roi_time_stop"),
        "n_morn_r5_scale": summary.get("n_morn_r5_scale"),
        "n_trade_tox": summary.get("n_trade_tox"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", default=PROFILE)
    ap.add_argument("--start-date", default="2026-02-01")
    ap.add_argument("--end-date", default="2026-07-16")
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--tag-prefix", default="research_roi_time_stop_ablation")
    args = ap.parse_args()

    base = load_profile(args.profile)
    base["date_range"]["start"] = args.start_date
    base["date_range"]["end"] = args.end_date
    results_dir = Path(base["_paths"]["results_dir"])
    names = [v.strip() for v in args.variants.split(",") if v.strip()]
    table = []
    for name in names:
        if name not in VARIANTS:
            raise SystemExit(f"unknown variant {name}; choose {list(VARIANTS)}")
        prof = deepcopy(base)
        trade = prof.setdefault("trade", {})
        spec = VARIANTS[name]
        if "roi_time_stop" in spec or "morning_r5_scale" in spec:
            for k, v in spec.items():
                trade[k] = v
        else:
            trade["roi_time_stop"] = dict(spec)
        tag = f"{args.tag_prefix}_{name}_{args.start_date[5:7]}_{args.end_date[5:7]}"
        out = results_dir / tag
        out.mkdir(parents=True, exist_ok=True)
        print(f"=== {name} → {out} ===", flush=True)
        result = run_offline_replay(prof, scheme="single")
        summary = result["summary"]
        (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        result["trades"].to_csv(out / "trades.csv", index=False)
        result["daily"].to_csv(out / "daily.csv", index=False)
        m = _metrics(summary, result["daily"])
        m["variant"] = name
        m["out"] = str(out)
        table.append(m)
        print(json.dumps(m, indent=2), flush=True)

    cmp = results_dir / f"{args.tag_prefix}_compare.json"
    cmp.write_text(json.dumps(table, indent=2), encoding="utf-8")
    print(f"wrote {cmp}", flush=True)


if __name__ == "__main__":
    main()

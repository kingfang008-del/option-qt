#!/usr/bin/env python3
"""Ablation: Mag7–QQQ corr_rewire day-level size scale (causal)."""
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
    "rewire25_s05": {
        "enabled": True,
        "asof": "10:30",
        "event_bars": 60,
        "calm_bars": 180,
        "rewire_min": 0.25,
        "rho_event_min": None,
        "edge_density_min": None,
        "action": "scale",
        "scale": 0.5,
    },
    "rewire20_s05": {
        "enabled": True,
        "asof": "10:30",
        "event_bars": 60,
        "calm_bars": 180,
        "rewire_min": 0.20,
        "action": "scale",
        "scale": 0.5,
    },
    "rho_le30_s05": {
        "enabled": True,
        "asof": "10:30",
        "event_bars": 60,
        "calm_bars": 180,
        "rewire_min": None,
        "rho_event_min": 0.30,
        "action": "scale",
        "scale": 0.5,
    },
    "edge_le35_s05": {
        "enabled": True,
        "asof": "10:30",
        "event_bars": 60,
        "calm_bars": 180,
        "rewire_min": None,
        "edge_density_min": 0.35,
        "action": "scale",
        "scale": 0.5,
    },
    "combo_rewire20_or_rho30": {
        "enabled": True,
        "asof": "10:30",
        "event_bars": 60,
        "calm_bars": 180,
        "rewire_min": 0.20,
        "rho_event_min": 0.30,
        "action": "scale",
        "scale": 0.5,
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
        "n_corr_rewire_days": summary.get("n_corr_rewire_days"),
        "n_corr_rewire_scale": summary.get("n_corr_rewire_scale"),
        "n_corr_rewire_block": summary.get("n_corr_rewire_block"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", default=PROFILE)
    ap.add_argument("--start-date", default="2026-02-01")
    ap.add_argument("--end-date", default="2026-07-16")
    ap.add_argument(
        "--variants",
        default=",".join(VARIANTS),
        help="comma list of variant keys",
    )
    ap.add_argument("--tag-prefix", default="research_corr_rewire_ablation")
    args = ap.parse_args()

    base = load_profile(args.profile)
    base["date_range"]["start"] = args.start_date
    base["date_range"]["end"] = args.end_date
    results_dir = Path(base["_paths"]["results_dir"])
    names = [v.strip() for v in args.variants.split(",") if v.strip()]

    table = []
    for name in names:
        if name not in VARIANTS:
            raise SystemExit(f"unknown variant {name}; choose from {list(VARIANTS)}")
        prof = deepcopy(base)
        prof.setdefault("trade", {})["corr_rewire"] = dict(VARIANTS[name])
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

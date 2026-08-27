#!/usr/bin/env python3
"""Ablation: entry-side adverse volume share (block / scale)."""
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
    "scale55_w120": {
        "entry_adv_vol": {
            "enabled": True,
            "action": "scale",
            "window_seconds": 120,
            "max_share": 0.55,
            "scale": 0.5,
        }
    },
    "scale60_w120": {
        "entry_adv_vol": {
            "enabled": True,
            "action": "scale",
            "window_seconds": 120,
            "max_share": 0.60,
            "scale": 0.5,
        }
    },
    "block55_w120": {
        "entry_adv_vol": {
            "enabled": True,
            "action": "block",
            "window_seconds": 120,
            "max_share": 0.55,
        }
    },
    "block60_w120": {
        "entry_adv_vol": {
            "enabled": True,
            "action": "block",
            "window_seconds": 120,
            "max_share": 0.60,
        }
    },
    "scale55_morn": {
        "entry_adv_vol": {
            "enabled": True,
            "action": "scale",
            "window_seconds": 120,
            "max_share": 0.55,
            "scale": 0.5,
            "tod_start": "10:31",
            "tod_end": "11:00",
        }
    },
    "scale55_w60": {
        "entry_adv_vol": {
            "enabled": True,
            "action": "scale",
            "window_seconds": 60,
            "max_share": 0.55,
            "scale": 0.5,
        }
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
        "n_entry_adv_vol_block": summary.get("n_entry_adv_vol_block"),
        "n_entry_adv_vol_scale": summary.get("n_entry_adv_vol_scale"),
    }


def _window_metrics(daily, start: str, end: str) -> dict:
    import pandas as pd

    d = daily.copy()
    d["date"] = pd.to_datetime(d["date"]).dt.strftime("%Y-%m-%d")
    sub = d[(d["date"] >= start) & (d["date"] <= end)]
    if sub.empty:
        return {"total_ret": None, "maxdd": None, "n_le5": 0}
    r = sub["day_ret"].astype(float)
    eq = (1 + r).cumprod()
    peak = eq.cummax()
    return {
        "total_ret": float(eq.iloc[-1] - 1),
        "maxdd": float((eq / peak - 1).min()),
        "n_le5": int((r <= -0.05).sum()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", default=PROFILE)
    ap.add_argument("--start-date", default="2026-02-01")
    ap.add_argument("--end-date", default="2026-07-16")
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--tag-prefix", default="research_entry_adv_vol_ablation")
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
        for k, v in VARIANTS[name].items():
            trade[k] = v
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
        m["weak_feb_mar"] = _window_metrics(result["daily"], "2026-02-01", "2026-03-31")
        m["strong_may_jul"] = _window_metrics(result["daily"], "2026-05-01", "2026-07-16")
        table.append(m)
        print(json.dumps(m, indent=2), flush=True)

    cmp = results_dir / f"{args.tag_prefix}_compare.json"
    cmp.write_text(json.dumps(table, indent=2), encoding="utf-8")
    print(f"wrote {cmp}", flush=True)


if __name__ == "__main__":
    main()

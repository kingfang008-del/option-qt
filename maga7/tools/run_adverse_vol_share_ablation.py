#!/usr/bin/env python3
"""Ablation: short-window adverse volume share soft gate."""
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

BASE = {
    "enabled": True,
    "opt_mtm_max": 0.0,
    "extend_max_cut": False,
}

VARIANTS = {
    "soft_s55_w180": {
        "adverse_vol_share": {
            **BASE,
            "mode": "soft_exit",
            "check_seconds": 180,
            "window_seconds": 180,
            "min_share": 0.55,
        }
    },
    "soft_s60_w180": {
        "adverse_vol_share": {
            **BASE,
            "mode": "soft_exit",
            "check_seconds": 180,
            "window_seconds": 180,
            "min_share": 0.60,
        }
    },
    "tox_s55_w180": {
        "adverse_vol_share": {
            **BASE,
            "mode": "tox_tighten",
            "check_seconds": 180,
            "window_seconds": 180,
            "min_share": 0.55,
            "tight_cut_ret": 0.15,
            "tight_mfe_bypass": 0.03,
        }
    },
    "tox_s60_w180": {
        "adverse_vol_share": {
            **BASE,
            "mode": "tox_tighten",
            "check_seconds": 180,
            "window_seconds": 180,
            "min_share": 0.60,
            "tight_cut_ret": 0.15,
            "tight_mfe_bypass": 0.03,
        }
    },
    "soft_s55_w120": {
        "adverse_vol_share": {
            **BASE,
            "mode": "soft_exit",
            "check_seconds": 120,
            "window_seconds": 120,
            "min_share": 0.55,
        }
    },
    "soft_s55_stk": {
        "adverse_vol_share": {
            **BASE,
            "mode": "soft_exit",
            "check_seconds": 180,
            "window_seconds": 180,
            "min_share": 0.55,
            "require_stock_adverse": True,
            "stock_adverse_max": -0.0005,
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
        "n_adv_vol": summary.get("n_adv_vol"),
        "n_adv_vol_armed": summary.get("n_adv_vol_armed"),
        "n_trade_tox": summary.get("n_trade_tox"),
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
    ap.add_argument("--tag-prefix", default="research_adverse_vol_share_ablation")
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

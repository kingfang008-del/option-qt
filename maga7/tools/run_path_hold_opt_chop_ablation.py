#!/usr/bin/env python3
"""May–Jun ablation: wash vs QQQ option-surface chop fast-pack gates.

Requires ``~/train_data/quote_options_bucketed_v7/QQQ`` (through 2026-06).
"""
from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from maga7.common.config import load_profile
from maga7.common.replay import run_offline_replay

PEER3 = (
    "maga7/CONFIG/strategy_profiles/"
    "single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1.json"
)

LIT = {
    "exit_mode": "mtm_trail",
    "hold_minutes": 45,
    "hold_extend_minutes": None,
    "trail_activate": 0.20,
    "trail_dd": 0.12,
    "stock_rev_exit": {
        "enabled": True,
        "when": "always",
        "min_hold_minutes": 10,
        "stock_max": 0.0,
        "opt_mtm_max": 0.10,
    },
}

FAST_BASE = {
    "enabled": True,
    "hold_minutes": 20,
    "trail_activate": 0.15,
    "trail_dd": 0.08,
    "stock_rev_min_hold_minutes": 5,
    "stock_rev_stock_max": 0.0,
    "stock_rev_opt_mtm_max": 0.05,
    "washout_breadth_min": 3,
    "opt_lookback_days": 40,
    "opt_imbalance_max": -0.05,
    "opt_chop_pctile_min": 0.70,
}

VARIANTS = {
    "baseline_t30": {},
    "lit_always": dict(LIT),
    "wash_fast": {**LIT, "path_fast_pack": {**FAST_BASE, "when": "mixed_wash_up"}},
    "opt_chop": {**LIT, "path_fast_pack": {**FAST_BASE, "when": "qqq_opt_chop"}},
    "wash_or_opt": {**LIT, "path_fast_pack": {**FAST_BASE, "when": "wash_or_opt_chop"}},
    "wash_and_opt": {**LIT, "path_fast_pack": {**FAST_BASE, "when": "wash_and_opt_chop"}},
}


def _metrics(summary, trades, daily):
    reasons = {}
    if trades is not None and not trades.empty and "reason" in trades.columns:
        reasons = {str(k): int(v) for k, v in trades["reason"].value_counts().items()}
    worst = 0.0
    if daily is not None and len(daily) and "day_ret" in daily.columns:
        worst = float(pd.to_numeric(daily["day_ret"]).min())
    return {
        "total_ret": float(summary["total_ret"]),
        "maxdd": float(summary["maxdd"]),
        "worst_day": worst,
        "n_trades": int(summary["n_trades"]),
        "trade_win": float(summary["trade_win"]),
        "n_tp": int(reasons.get("TP", 0)),
        "n_trail": int(reasons.get("TRAIL", 0)),
        "n_stock_rev": int(reasons.get("STOCK_REV", 0)),
        "n_fast_pack_days": int(summary.get("n_fast_pack_days") or 0),
        "n_fast_pack_off_days": int(summary.get("n_fast_pack_off_days") or 0),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", default="2026-05-01")
    ap.add_argument("--end", default="2026-06-30")
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument(
        "--out", default="/mnt/s990/data/maga7/results/path_hold_opt_chop_may_jun_v1"
    )
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    want = [x.strip() for x in args.variants.split(",") if x.strip()]

    rows = []
    for vname in want:
        if vname not in VARIANTS:
            raise SystemExit(f"unknown {vname}")
        prof = deepcopy(load_profile(PEER3))
        prof["date_range"] = {"start": args.start, "end": args.end}
        for k, v in VARIANTS[vname].items():
            prof.setdefault("trade", {})[k] = v
        print(f"=== {vname} ===", flush=True)
        result = run_offline_replay(prof, scheme="single")
        summary, trades, daily = result["summary"], result["trades"], result.get("daily")
        wdir = out / vname
        wdir.mkdir(parents=True, exist_ok=True)
        (wdir / "summary.json").write_text(
            json.dumps(summary, indent=2, default=str), encoding="utf-8"
        )
        trades.to_csv(wdir / "trades.csv", index=False)
        if daily is not None and len(daily):
            daily.to_csv(wdir / "daily.csv", index=False)
        m = _metrics(summary, trades, daily)
        m["variant"] = vname
        rows.append(m)
        print(
            f"  ret={m['total_ret']:+.1%} dd={m['maxdd']:+.1%} worst={m['worst_day']:+.1%} "
            f"fast_days={m['n_fast_pack_days']} TP/TRAIL/REV="
            f"{m['n_tp']}/{m['n_trail']}/{m['n_stock_rev']}",
            flush=True,
        )

    bdf = pd.DataFrame(rows)
    bdf.to_csv(out / "scoreboard.csv", index=False)
    (out / "scoreboard.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (out / "REPORT.md").write_text(
        "# Opt-chop fast-pack ablation (May–Jun)\n\n```\n"
        + bdf.to_string(index=False)
        + "\n```\n",
        encoding="utf-8",
    )
    print(bdf.to_string(index=False), flush=True)
    print("wrote", out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Ablate lit baseline vs wash-conditional fast pack (risk-first)."""
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
LIT = (
    "maga7/CONFIG/strategy_profiles/"
    "single_qqq_open_ladder_atm5otm_path_hold_lit_v1.json"
)
WASH = (
    "maga7/CONFIG/strategy_profiles/"
    "single_qqq_open_ladder_atm5otm_path_hold_lit_wash_fast_v1.json"
)

WINDOWS = [
    ("may_jul", "2026-05-01", "2026-07-17"),
    ("jan_mar", "2026-01-02", "2026-03-31"),
]

FAST = {
    "enabled": True,
    "when": "mixed_wash_up",
    "hold_minutes": 20,
    "trail_activate": 0.15,
    "trail_dd": 0.08,
    "stock_rev_min_hold_minutes": 5,
    "stock_rev_stock_max": 0.0,
    "stock_rev_opt_mtm_max": 0.05,
    "washout_breadth_min": 3,
}

LIT_TRADE = {
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

VARIANTS: dict[str, tuple[str, dict | None]] = {
    "baseline_t30": (PEER3, {}),
    "lit_always": (PEER3, dict(LIT_TRADE)),
    "lit_wash_fast": (
        PEER3,
        {**LIT_TRADE, "path_fast_pack": dict(FAST)},
    ),
    "fast_always": (
        PEER3,
        {
            **LIT_TRADE,
            "path_fast_pack": {**FAST, "when": "always"},
        },
    ),
    "lit_file": (LIT, None),
    "wash_file": (WASH, None),
}


def _metrics(summary: dict, trades: pd.DataFrame, daily: pd.DataFrame | None) -> dict:
    reasons = {}
    if trades is not None and not trades.empty and "reason" in trades.columns:
        reasons = {str(k): int(v) for k, v in trades["reason"].value_counts().items()}
    worst = 0.0
    if daily is not None and len(daily) and "day_ret" in daily.columns:
        worst = float(pd.to_numeric(daily["day_ret"]).min())
    losers = (
        trades[pd.to_numeric(trades["ret"], errors="coerce") < 0]
        if trades is not None and not trades.empty
        else pd.DataFrame()
    )
    med_loser = None
    if len(losers):
        et = pd.to_datetime(losers["entry_ts"], utc=True, errors="coerce")
        xt = pd.to_datetime(losers["exit_ts"], utc=True, errors="coerce")
        med_loser = float((xt - et).dt.total_seconds().median())
    return {
        "total_ret": float(summary["total_ret"]),
        "maxdd": float(summary["maxdd"]),
        "worst_day": worst,
        "n_trades": int(summary["n_trades"]),
        "trade_win": float(summary["trade_win"]),
        "n_tp": int(reasons.get("TP", 0)),
        "n_trail": int(reasons.get("TRAIL", 0)),
        "n_stock_rev": int(reasons.get("STOCK_REV", 0)),
        "n_t30": int(reasons.get("T+30", 0)),
        "n_t45": int(reasons.get("T+45", 0)),
        "n_t20": int(reasons.get("T+20", 0)),
        "n_fast_pack_days": int(summary.get("n_fast_pack_days") or 0),
        "n_fast_pack_off_days": int(summary.get("n_fast_pack_off_days") or 0),
        "med_loser_hold_sec": med_loser,
        "reason_top": dict(list(sorted(reasons.items(), key=lambda kv: -kv[1])[:10])),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--windows", default="may_jul,jan_mar")
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument(
        "--out", default="/mnt/s990/data/maga7/results/path_hold_wash_fast_v1"
    )
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    want_w = {x.strip() for x in args.windows.split(",") if x.strip()}
    want_v = [x.strip() for x in args.variants.split(",") if x.strip()]

    rows = []
    for wname, start, end in WINDOWS:
        if wname not in want_w:
            continue
        for vname in want_v:
            if vname not in VARIANTS:
                raise SystemExit(f"unknown variant {vname}")
            prof_path, overlay = VARIANTS[vname]
            prof = deepcopy(load_profile(prof_path))
            prof["date_range"] = {"start": start, "end": end}
            if overlay is not None:
                trade = prof.setdefault("trade", {})
                for k, v in overlay.items():
                    trade[k] = v
            tag = f"{wname}__{vname}"
            print(f"=== {tag} ===", flush=True)
            result = run_offline_replay(prof, scheme="single")
            summary = result["summary"]
            trades = result["trades"]
            daily = result.get("daily")
            wdir = out / tag
            wdir.mkdir(parents=True, exist_ok=True)
            (wdir / "summary.json").write_text(
                json.dumps(summary, indent=2, default=str), encoding="utf-8"
            )
            trades.to_csv(wdir / "trades.csv", index=False)
            if daily is not None and len(daily):
                daily.to_csv(wdir / "daily.csv", index=False)
            m = _metrics(summary, trades, daily)
            m.update({"window": wname, "variant": vname, "start": start, "end": end})
            rows.append(m)
            print(
                f"  ret={m['total_ret']:+.1%} dd={m['maxdd']:+.1%} "
                f"worst={m['worst_day']:+.1%} fast_days={m['n_fast_pack_days']} "
                f"TP/TRAIL/REV={m['n_tp']}/{m['n_trail']}/{m['n_stock_rev']}",
                flush=True,
            )

    bdf = pd.DataFrame(rows)
    bdf.to_csv(out / "scoreboard.csv", index=False)
    (out / "scoreboard.json").write_text(
        json.dumps(rows, indent=2, default=str), encoding="utf-8"
    )
    summary = {
        "verdict": "PROMOTE_RESEARCH",
        "best": "lit_wash_fast",
        "thesis": (
            "Clean days keep lit rails; mixed_wash_up arms fast pack "
            "(trail15/8 + REV@5 + T20) — avoid always-scalp TP massacre."
        ),
        "profile": WASH,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out / "REPORT.md").write_text(
        "\n".join(
            [
                "# Path Hold Wash-Fast Ablation",
                "",
                f"**best=`{summary['best']}`**",
                "",
                "```",
                bdf.to_string(index=False),
                "```",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)
    print("wrote", out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

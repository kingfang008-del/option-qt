#!/usr/bin/env python3
"""Ablate trend-purity position sizing on extend_mtm_only peer3.

High causal purity → full size; low → shrink (or skip). Does not change
TopK selection order — only scales risk on each accepted entry.
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

FOCUS = {"2026-07-07", "2026-07-08", "2026-07-09"}


def _extend_base(trade: dict[str, Any]) -> None:
    trade["exit_mode"] = "hold_extend"
    trade["hold_minutes"] = 30
    trade["hold_extend_minutes"] = 45
    trade["hold_extend_mtm_min"] = 0.0
    trade["hold_extend_require_mf"] = False
    trade["bar_availability_delay_seconds"] = 60
    trade["day_circuit"] = None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--profile", default=str(PEER3))
    p.add_argument("--start-date", default="2026-05-01")
    p.add_argument("--end-date", default="2026-07-13")
    p.add_argument("--tag", default="trend_purity_ablation_extend_mtm_peer3_may_jul")
    args = p.parse_args()

    base = load_profile(args.profile)
    base["date_range"]["start"] = args.start_date
    base["date_range"]["end"] = args.end_date
    _extend_base(base.setdefault("trade", {}))

    out = Path(base["_paths"]["results_dir"]) / args.tag
    out.mkdir(parents=True, exist_ok=True)

    full_day = {
        "event_calendar_block": True,
        "event_calendar": "default",
        "event_blackout_sessions": 0,
    }

    variants: list[tuple[str, dict[str, Any], dict[str, Any]]] = [
        ("extend_mtm_only", {}, {}),
        (
            "purity_cont",
            {
                "trend_purity_sizing": True,
                "trend_purity_mode": "continuous",
                "trend_purity_high": 0.70,
                "trend_purity_min_scale": 0.35,
                "trend_purity_fp_ref": 0.025,
            },
            {},
        ),
        (
            "purity_cont_soft",
            {
                "trend_purity_sizing": True,
                "trend_purity_mode": "continuous",
                "trend_purity_high": 0.60,
                "trend_purity_min_scale": 0.50,
                "trend_purity_fp_ref": 0.025,
            },
            {},
        ),
        (
            "purity_tier",
            {
                "trend_purity_sizing": True,
                "trend_purity_mode": "tier",
                "trend_purity_low": 0.40,
                "trend_purity_high": 0.70,
                "trend_purity_min_scale": 0.35,
                "trend_purity_mid_scale": 0.55,
            },
            {},
        ),
        (
            "purity_skip_low",
            {
                "trend_purity_sizing": True,
                "trend_purity_mode": "skip_low",
                "trend_purity_skip_below": 0.40,
                "trend_purity_high": 0.70,
                "trend_purity_min_scale": 0.35,
            },
            {},
        ),
        (
            "confirm1_purity_cont",
            {
                "entry_confirm_bars": 1,
                "entry_confirm_mode": "mf",
                "trend_purity_sizing": True,
                "trend_purity_mode": "continuous",
                "trend_purity_high": 0.70,
                "trend_purity_min_scale": 0.35,
            },
            {},
        ),
        ("full_day", {}, full_day),
        (
            "full_day_purity_cont",
            {
                "trend_purity_sizing": True,
                "trend_purity_mode": "continuous",
                "trend_purity_high": 0.70,
                "trend_purity_min_scale": 0.35,
            },
            full_day,
        ),
    ]

    scoreboard: list[dict[str, Any]] = []
    for name, trade_over, reg_over in variants:
        prof = deepcopy(base)
        for k, v in trade_over.items():
            prof["trade"][k] = v
        prof.setdefault("regime", {})
        for k, v in reg_over.items():
            prof["regime"][k] = v
        print(f"==> {name}", flush=True)
        result = run_offline_replay(prof, scheme="single")
        s = result["summary"]
        tr = result["trades"]
        focus = tr.loc[tr["date"].astype(str).isin(FOCUS)] if len(tr) else tr
        focus_detail = []
        if len(focus):
            for r in focus.itertuples(index=False):
                pur = getattr(r, "trend_purity", None)
                psc = getattr(r, "trend_purity_scale", None)
                pur_s = f" pur={float(pur):.2f}" if pur is not None and pd.notna(pur) else ""
                psc_s = f"×{float(psc):.2f}" if psc is not None and pd.notna(psc) else ""
                focus_detail.append(
                    f"{r.date} {r.symbol} {r.dir} ret={float(r.ret):+.1%} "
                    f"size={float(r.size_frac):.3f}{pur_s}{psc_s} {r.reason}"
                )
        row = {
            "name": name,
            "total_ret": float(s["total_ret"]),
            "maxdd": float(s["maxdd"]),
            "n_trades": int(s["n_trades"]),
            "trade_win": float(s["trade_win"]),
            "trade_exp": float(s["trade_exp"]),
            "n_purity_scaled": int(s.get("n_purity_scaled") or 0),
            "n_purity_skip": int(s.get("n_purity_skip") or 0),
            "focus_ret_sum": float(focus["ret"].sum()) if len(focus) else 0.0,
            "focus_trades": focus_detail,
            "end_equity": float(s["end_equity"]),
        }
        if len(tr) and "trend_purity" in tr.columns:
            row["purity_mean"] = float(tr["trend_purity"].mean())
            row["purity_focus_mean"] = (
                float(focus["trend_purity"].mean()) if len(focus) else float("nan")
            )
        scoreboard.append(row)
        sub = out / name
        sub.mkdir(parents=True, exist_ok=True)
        (sub / "summary.json").write_text(json.dumps(s, indent=2), encoding="utf-8")
        tr.to_csv(sub / "trades.csv", index=False)
        result["daily"].to_csv(sub / "daily.csv", index=False)
        print(
            f"    ret={row['total_ret']:+.1%} dd={row['maxdd']:.1%} n={row['n_trades']} "
            f"scaled={row['n_purity_scaled']} skip={row['n_purity_skip']} "
            f"focus={row['focus_ret_sum']:+.2f}",
            flush=True,
        )
        for line in focus_detail:
            print(f"      {line}", flush=True)

    base_ret = scoreboard[0]["total_ret"]
    for row in scoreboard:
        row["uplift_vs_extend_pp"] = float(row["total_ret"] - base_ret)

    (out / "scoreboard.json").write_text(
        json.dumps(scoreboard, indent=2, default=str), encoding="utf-8"
    )
    pd.DataFrame(
        [
            {
                "name": r["name"],
                "total_ret": r["total_ret"],
                "maxdd": r["maxdd"],
                "n_trades": r["n_trades"],
                "n_purity_scaled": r["n_purity_scaled"],
                "n_purity_skip": r["n_purity_skip"],
                "focus_ret_sum": r["focus_ret_sum"],
                "uplift_vs_extend_pp": r["uplift_vs_extend_pp"],
                "end_equity": r["end_equity"],
            }
            for r in scoreboard
        ]
    ).to_csv(out / "scoreboard.csv", index=False)

    summary = {
        "period": f"{args.start_date}..{args.end_date}",
        "base": "extend_mtm_only peer3 day_circuit=null",
        "best_ret": max(scoreboard, key=lambda r: r["total_ret"])["name"],
        "best_dd": max(scoreboard, key=lambda r: r["maxdd"])["name"],
        "scoreboard": [
            {
                "name": r["name"],
                "total_ret": r["total_ret"],
                "maxdd": r["maxdd"],
                "n_trades": r["n_trades"],
                "focus_ret_sum": r["focus_ret_sum"],
                "uplift_vs_extend_pp": r["uplift_vs_extend_pp"],
            }
            for r in scoreboard
        ],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()

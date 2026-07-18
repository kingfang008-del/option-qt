#!/usr/bin/env python3
"""Ablate deferred TopK commit auction on extend_mtm_only peer3.

Until commit_tod, Rule-A fires only enter a candidate pool (all_first universe).
At commit clock, rank by score and enter TopK at commit+delay (not at first-fire time).
Optional post_commit_fill lets later fires take remaining concurrent slots.
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
    p.add_argument("--tag", default="topk_commit_ablation_extend_mtm_peer3_may_jul")
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
        ("commit_1100_fp", {
            "topk_commit_tod": "11:00",
            "topk_rank": "abs_from_prev",
            "topk_post_commit_fill": True,
        }, {}),
        ("commit_1130_fp", {
            "topk_commit_tod": "11:30",
            "topk_rank": "abs_from_prev",
            "topk_post_commit_fill": True,
        }, {}),
        ("commit_1200_fp", {
            "topk_commit_tod": "12:00",
            "topk_rank": "abs_from_prev",
            "topk_post_commit_fill": True,
        }, {}),
        ("commit_1230_fp", {
            "topk_commit_tod": "12:30",
            "topk_rank": "abs_from_prev",
            "topk_post_commit_fill": True,
        }, {}),
        ("commit_1130_peer_fp", {
            "topk_commit_tod": "11:30",
            "topk_rank": "peer_fp",
            "topk_post_commit_fill": True,
        }, {}),
        ("commit_1130_fp_nofill", {
            "topk_commit_tod": "11:30",
            "topk_rank": "abs_from_prev",
            "topk_post_commit_fill": False,
        }, {}),
        ("full_day", {}, full_day),
        ("full_day_commit_1130_fp", {
            "topk_commit_tod": "11:30",
            "topk_rank": "abs_from_prev",
            "topk_post_commit_fill": True,
        }, full_day),
        ("full_day_commit_1230_fp", {
            "topk_commit_tod": "12:30",
            "topk_rank": "abs_from_prev",
            "topk_post_commit_fill": True,
        }, full_day),
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
                focus_detail.append(
                    f"{r.date} {r.symbol} {r.dir} ret={float(r.ret):+.1%} "
                    f"{r.reason} entry={r.entry_ts}"
                )
        row = {
            "name": name,
            **{k: trade_over.get(k) for k in (
                "topk_commit_tod", "topk_rank", "topk_post_commit_fill"
            )},
            "total_ret": float(s["total_ret"]),
            "maxdd": float(s["maxdd"]),
            "n_trades": int(s["n_trades"]),
            "trade_win": float(s["trade_win"]),
            "trade_exp": float(s["trade_exp"]),
            "n_commit_selected": int(s.get("n_commit_selected") or 0),
            "n_commit_pool": int(s.get("n_commit_pool") or 0),
            "focus_ret_sum": float(focus["ret"].sum()) if len(focus) else 0.0,
            "focus_trades": focus_detail,
            "end_equity": float(s["end_equity"]),
        }
        scoreboard.append(row)
        sub = out / name
        sub.mkdir(parents=True, exist_ok=True)
        (sub / "summary.json").write_text(json.dumps(s, indent=2), encoding="utf-8")
        tr.to_csv(sub / "trades.csv", index=False)
        result["daily"].to_csv(sub / "daily.csv", index=False)
        print(
            f"    ret={row['total_ret']:+.1%} dd={row['maxdd']:.1%} n={row['n_trades']} "
            f"sel={row['n_commit_selected']}/{row['n_commit_pool']} "
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
                "topk_commit_tod": r.get("topk_commit_tod"),
                "topk_rank": r.get("topk_rank"),
                "topk_post_commit_fill": r.get("topk_post_commit_fill"),
                "total_ret": r["total_ret"],
                "maxdd": r["maxdd"],
                "n_trades": r["n_trades"],
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
        "best_focus": max(scoreboard, key=lambda r: r["focus_ret_sum"])["name"],
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

#!/usr/bin/env python3
"""Apply inprove2.txt filters on extend_mtm_only (peer3) research base.

Base matches ``hold_extend_ablation_.../extend_mtm_only``:
  peer3 + hold_extend T30→T45 MTM>=0 (no mf confirm) + delay=60 rails.

Filters from preprocess/raw_data_deal/inprove2.txt (skew skipped — needs Greeks):
  1) SI >= 0.57 / 0.43
  2) PE min_ratio 0.5
  3) TOD mf10 z-score >= 2.0 (new)
  + recommended SI + TOD combo
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

FOCUS = {
    "2026-05-20",
    "2026-05-21",
    "2026-06-12",
    "2026-06-16",
    "2026-06-18",
    "2026-07-07",
    "2026-07-08",
    "2026-07-09",
}


def _extend_mtm_only(trade: dict[str, Any]) -> None:
    trade["exit_mode"] = "hold_extend"
    trade["hold_minutes"] = 30
    trade["hold_extend_minutes"] = 45
    trade["hold_extend_mtm_min"] = 0.0
    trade["hold_extend_require_mf"] = False
    trade["bar_availability_delay_seconds"] = 60


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--profile", default=str(PEER3))
    p.add_argument("--start-date", default="2026-05-01")
    p.add_argument("--end-date", default="2026-07-13")
    p.add_argument("--tag", default="inprove2_extend_mtm_ablation_peer3_may_jul")
    args = p.parse_args()

    base = load_profile(args.profile)
    base["date_range"]["start"] = args.start_date
    base["date_range"]["end"] = args.end_date
    _extend_mtm_only(base.setdefault("trade", {}))

    out = Path(base["_paths"]["results_dir"]) / args.tag
    out.mkdir(parents=True, exist_ok=True)

    variants: list[tuple[str, dict[str, Any]]] = [
        ("extend_mtm_only", {}),
        ("si_0_57", {"si_min": 0.57}),
        ("si_0_43", {"si_min": 0.43}),
        ("pe_0_5", {"pe_min_ratio": 0.5, "pe_window": 10, "pe_lookback_bars": 780}),
        (
            "tod_z_2",
            {"tod_mf_z_min": 2.0, "tod_mf_z_lookback_days": 20},
        ),
        (
            "tod_z_1_5",
            {"tod_mf_z_min": 1.5, "tod_mf_z_lookback_days": 20},
        ),
        (
            "si057_tod_z2",
            {"si_min": 0.57, "tod_mf_z_min": 2.0, "tod_mf_z_lookback_days": 20},
        ),
        (
            "si043_tod_z2",
            {"si_min": 0.43, "tod_mf_z_min": 2.0, "tod_mf_z_lookback_days": 20},
        ),
        (
            "si043_pe05",
            {
                "si_min": 0.43,
                "pe_min_ratio": 0.5,
                "pe_window": 10,
                "pe_lookback_bars": 780,
            },
        ),
        (
            "si057_pe05_tod_z2",
            {
                "si_min": 0.57,
                "pe_min_ratio": 0.5,
                "pe_window": 10,
                "pe_lookback_bars": 780,
                "tod_mf_z_min": 2.0,
                "tod_mf_z_lookback_days": 20,
            },
        ),
    ]

    scoreboard: list[dict[str, Any]] = []
    for name, sig_over in variants:
        prof = deepcopy(base)
        prof.setdefault("signal", {}).update(sig_over)
        print(f"==> {name} {sig_over}", flush=True)
        result = run_offline_replay(prof, scheme="single")
        s = result["summary"]
        tr = result["trades"]
        focus = float(tr.loc[tr["date"].isin(FOCUS), "ret"].sum()) if len(tr) else 0.0
        row = {
            "name": name,
            **sig_over,
            "total_ret": float(s["total_ret"]),
            "maxdd": float(s["maxdd"]),
            "n_trades": int(s["n_trades"]),
            "trade_win": float(s["trade_win"]),
            "trade_exp": float(s["trade_exp"]),
            "n_si_block": int(s.get("n_si_block") or 0),
            "n_pe_block": int(s.get("n_pe_block") or 0),
            "n_tod_z_block": int(s.get("n_tod_z_block") or 0),
            "n_peer_block": int(s.get("n_peer_block") or 0),
            "focus_cluster_ret_sum": focus,
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
            f"si/pe/tod={row['n_si_block']}/{row['n_pe_block']}/{row['n_tod_z_block']} "
            f"focus={focus:+.2f}",
            flush=True,
        )

    base_ret = scoreboard[0]["total_ret"]
    for row in scoreboard:
        row["uplift_vs_extend_pp"] = float(row["total_ret"] - base_ret)

    (out / "scoreboard.json").write_text(json.dumps(scoreboard, indent=2), encoding="utf-8")
    pd.DataFrame(scoreboard).to_csv(out / "scoreboard.csv", index=False)
    best = max(scoreboard, key=lambda r: r["total_ret"])
    best_dd = max(scoreboard, key=lambda r: r["maxdd"])
    summary = {
        "period": f"{args.start_date}..{args.end_date}",
        "base": "extend_mtm_only peer3",
        "source": "preprocess/raw_data_deal/inprove2.txt",
        "extend_baseline_ret": base_ret,
        "best_by_ret": {k: best[k] for k in ("name", "total_ret", "maxdd", "n_trades")},
        "best_by_maxdd": {k: best_dd[k] for k in ("name", "total_ret", "maxdd", "n_trades")},
        "note": "Option skew filter not implemented (needs minute Greeks). SI/PE/TOD only.",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Tight washout-gate variants (after b3@0.3% proved too loose)."""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from maga7.tools.run_washout_gate_scoreboard import _run, _with_router
from maga7.common.config import load_profile
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--profile",
        default="maga7/CONFIG/strategy_profiles/single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1.json",
    )
    ap.add_argument("--experts", default="maga7/CONFIG/regime_router/experts_v1.json")
    ap.add_argument("--out", default="maga7/results/regime_router/washout_gate_scoreboard_tight")
    args = ap.parse_args()

    base = load_profile(args.profile)
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    ep = args.experts

    def wash(rule, expert, *, wd=0.008, b=5, **extra):
        return {
            "enabled": True,
            "mode": "rule",
            "rule": rule,
            "asof": "10:30",
            "wash_window_end": "10:00",
            "wash_drop_min": wd,
            "washout_breadth_min": b,
            "washout_expert": expert,
            "experts_path": ep,
            **extra,
        }

    variants = {
        "baseline": None,
        "reclaim_disp55": {
            "enabled": True,
            "mode": "rule",
            "rule": "reclaim_disp55",
            "asof": "10:30",
            "experts_path": ep,
        },
        "wd8_b5_dn": wash("washout_breadth", "washout_gate_dn"),
        "wd8_b5_both": wash("washout_breadth", "washout_gate_both"),
        "wd8_b5_halt": wash("washout_breadth", "washout_gate_halt"),
        "wd12_b6_both": wash("washout_breadth", "washout_gate_both", wd=0.012, b=6),
        "wd12_b6_halt": wash("washout_breadth", "washout_gate_halt", wd=0.012, b=6),
        "wd8_b5_and_reclaim_both": wash("washout_and_reclaim", "washout_gate_both"),
        "wd8_b5_and_reclaim_halt": wash("washout_and_reclaim", "washout_gate_halt"),
        "wd8_b5_or_reclaim_dn": wash("washout_or_reclaim", "washout_gate_dn"),
    }

    windows = [
        ("strong_may_jul", "2026-05-01", "2026-07-17"),
        ("weak_feb_apr", "2026-02-01", "2026-04-30"),
    ]
    rows = []
    for vname, router in variants.items():
        if router is None:
            prof = copy.deepcopy(base)
            rr = dict(prof.get("regime_router") or {})
            rr["enabled"] = False
            prof["regime_router"] = rr
        else:
            prof = _with_router(base, router)
        for wname, start, end in windows:
            tag = f"{vname}__{wname}"
            print(f"=== {tag} ===", flush=True)
            r = _run(prof, start=start, end=end, tag=tag, out=out)
            r["variant"] = vname
            r["window"] = wname
            rows.append(r)
            print(
                f"  ret={r['total_ret']:.3f} maxdd={r['maxdd']:.3f} n={r['n_trades']} "
                f"expert_days={r['n_router_expert_days']} counts={r['router_day_counts']} "
                f"0717={r['day_ret_0717']}",
                flush=True,
            )

    board = pd.DataFrame(rows)
    base_ret = {r["window"]: r["total_ret"] for r in rows if r["variant"] == "baseline"}
    board["vs_baseline"] = board.apply(
        lambda r: (r["total_ret"] / base_ret[r["window"]]) if base_ret.get(r["window"]) else None,
        axis=1,
    )
    board.to_csv(out / "scoreboard.csv", index=False)
    (out / "summary.json").write_text(
        json.dumps(
            {"scoreboard": board.to_dict(orient="records"), "note": "tight washout gates"},
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    cols = [
        "variant",
        "window",
        "total_ret",
        "maxdd",
        "n_trades",
        "n_router_expert_days",
        "day_ret_0717",
        "vs_baseline",
    ]
    print(board[cols].to_string(index=False))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

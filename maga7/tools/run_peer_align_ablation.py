#!/usr/bin/env python3
"""Ablate Mag7 peer same-direction breadth filter vs Mag7+GOOGL T+30 rails.

Require >=K peers with mf10 (or streak) aligned at signal time before entry.
"""
from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from maga7.common.config import load_profile
from maga7.common.replay import run_offline_replay

BASE = (
    ROOT
    / "maga7"
    / "CONFIG"
    / "strategy_profiles"
    / "single_qqq_open_ladder_atm5otm_t30_rails_p20_googl_v1.json"
)

MAG7 = ["NVDA", "TSLA", "AAPL", "AMZN", "META", "MSFT", "AMD"]

STREAK_DAYS = {
    "2026-05-06",
    "2026-05-07",
    "2026-05-08",
    "2026-05-11",
    "2026-05-20",
    "2026-05-21",
    "2026-05-22",
    "2026-07-07",
    "2026-07-08",
    "2026-07-09",
}


def _run(profile: dict[str, Any], scheme: str = "single") -> dict[str, Any]:
    result = run_offline_replay(profile, scheme=scheme)
    s = result["summary"]
    trades = result["trades"]
    daily = result["daily"]
    streak_ret = None
    streak_days = None
    if not trades.empty:
        td = trades.copy()
        td["date"] = td["date"].astype(str)
        st = td[td["date"].isin(STREAK_DAYS)]
        streak_ret = float(st["ret"].mean()) if not st.empty else None
    if not daily.empty:
        d = daily.copy()
        d["date"] = d["date"].astype(str)
        sd = d[d["date"].isin(STREAK_DAYS)]
        if not sd.empty:
            streak_days = float((1 + sd["day_ret"].fillna(0)).prod() - 1)
    return {
        "total_ret": float(s["total_ret"]),
        "maxdd": float(s["maxdd"]),
        "n_trades": int(s["n_trades"]),
        "trade_win": float(s["trade_win"]),
        "trade_exp": float(s["trade_exp"]),
        "end_equity": float(s["end_equity"]),
        "n_peer_block": int(s.get("n_peer_block") or 0),
        "streak_trade_mean_ret": streak_ret,
        "streak_days_compound": streak_days,
        "summary": s,
        "trades": trades,
        "daily": daily,
    }


def _variants() -> list[tuple[str, dict[str, Any]]]:
    return [
        ("baseline", {}),
        (
            "peer_mf10_min3_all",
            {"signal": {"peer_align_min": 3, "peer_align_mode": "mf10"}},
        ),
        (
            "peer_mf10_min4_all",
            {"signal": {"peer_align_min": 4, "peer_align_mode": "mf10"}},
        ),
        (
            "peer_mf10_min3_mag7",
            {
                "signal": {
                    "peer_align_min": 3,
                    "peer_align_mode": "mf10",
                    "peer_symbols": MAG7,
                }
            },
        ),
        (
            "peer_mf10_min4_mag7",
            {
                "signal": {
                    "peer_align_min": 4,
                    "peer_align_mode": "mf10",
                    "peer_symbols": MAG7,
                }
            },
        ),
        (
            "peer_streak_min3_all",
            {"signal": {"peer_align_min": 3, "peer_align_mode": "streak"}},
        ),
        (
            "peer_mf_fp_min3_all",
            {"signal": {"peer_align_min": 3, "peer_align_mode": "mf_fp"}},
        ),
    ]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--profile", default=str(BASE))
    p.add_argument("--start-date", default="2026-05-01")
    p.add_argument("--end-date", default="2026-07-13")
    p.add_argument("--tag", default="peer_align_ablation_mag7_googl_may_jul")
    p.add_argument("--scheme", default="single")
    args = p.parse_args()

    base = load_profile(args.profile)
    base["date_range"]["start"] = args.start_date
    base["date_range"]["end"] = args.end_date
    base.setdefault("trade", {})["bar_availability_delay_seconds"] = 60
    out = Path(base["_paths"]["results_dir"]) / args.tag
    out.mkdir(parents=True, exist_ok=True)

    scoreboard: list[dict[str, Any]] = []
    for name, patch in _variants():
        prof = deepcopy(base)
        for section, vals in patch.items():
            prof.setdefault(section, {}).update(vals)
        print(f"[{args.tag}] {name} ...", flush=True)
        got = _run(prof, args.scheme)
        row = {
            "name": name,
            "peer_align_min": prof.get("signal", {}).get("peer_align_min"),
            "peer_align_mode": prof.get("signal", {}).get("peer_align_mode"),
            "peer_symbols": prof.get("signal", {}).get("peer_symbols"),
            **{
                k: got[k]
                for k in (
                    "total_ret",
                    "maxdd",
                    "n_trades",
                    "trade_win",
                    "trade_exp",
                    "end_equity",
                    "n_peer_block",
                    "streak_trade_mean_ret",
                    "streak_days_compound",
                )
            },
        }
        scoreboard.append(row)
        print(
            f"  ret={row['total_ret']:+.2%} dd={row['maxdd']:.2%} n={row['n_trades']} "
            f"win={row['trade_win']:.1%} peer_block={row['n_peer_block']} "
            f"streak_days={row['streak_days_compound']}",
            flush=True,
        )
        sub = out / name
        sub.mkdir(exist_ok=True)
        (sub / "summary.json").write_text(json.dumps(got["summary"], indent=2), encoding="utf-8")
        got["trades"].to_csv(sub / "trades.csv", index=False)
        got["daily"].to_csv(sub / "daily.csv", index=False)

    import pandas as pd

    pd.DataFrame(scoreboard).to_csv(out / "scoreboard.csv", index=False)
    baseline = scoreboard[0]
    best = max(scoreboard, key=lambda r: r["total_ret"])
    viable = [
        r
        for r in scoreboard[1:]
        if r["total_ret"] > baseline["total_ret"] and r["maxdd"] >= baseline["maxdd"] - 0.05
    ]
    pick = max(viable, key=lambda r: r["total_ret"]) if viable else None
    summary = {
        "period": f"{args.start_date}..{args.end_date}",
        "baseline": baseline,
        "best_by_ret": best,
        "pick_uplift_dd_ok": pick,
        "uplift_pp": {r["name"]: r["total_ret"] - baseline["total_ret"] for r in scoreboard[1:]},
        "promote": bool(pick is not None),
        "note": (
            "peer_align: at feature_ts count symbols with mf10/streak/mf_fp aligned to entry dir; "
            "block if count < peer_align_min. Includes self."
        ),
    }
    (out / "scoreboard.json").write_text(json.dumps(scoreboard, indent=2), encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in summary if k != "note"}, indent=2))
    print(f"→ {out}")


if __name__ == "__main__":
    main()

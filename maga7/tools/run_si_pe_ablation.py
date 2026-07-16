#!/usr/bin/env python3
"""Ablate SI (strict sync) + price-efficiency entry filters vs Mag7+GOOGL rails."""
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

GOOGL_PROF = (
    ROOT
    / "maga7/CONFIG/strategy_profiles/single_qqq_open_ladder_atm5otm_t30_rails_p20_googl_v1.json"
)
MAG7_PROF = (
    ROOT
    / "maga7/CONFIG/strategy_profiles/single_qqq_open_ladder_atm5otm_t30_rails_p20_v1.json"
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
    streak_days = None
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
        "n_peer_block": int(s.get("n_peer_block") or 0),
        "n_si_block": int(s.get("n_si_block") or 0),
        "n_pe_block": int(s.get("n_pe_block") or 0),
        "streak_days_compound": streak_days,
        "summary": s,
        "trades": trades,
        "daily": daily,
    }


def _variants() -> list[tuple[str, dict[str, Any]]]:
    mag7_peers = {"peer_symbols": MAG7, "peer_align_mode": "mf10"}
    return [
        ("baseline", {}),
        ("peer_min3", {**mag7_peers, "peer_align_min": 3}),
        ("peer_min5", {**mag7_peers, "peer_align_min": 5}),
        ("peer_min6", {**mag7_peers, "peer_align_min": 6}),
        ("si_0_57", {**mag7_peers, "si_min": 0.57}),  # ~>=6/7 same sign
        ("si_0_43", {**mag7_peers, "si_min": 0.43}),  # ~>=5/7
        ("pe_0_5", {"pe_min_ratio": 0.5, "pe_window": 10, "pe_lookback_bars": 780}),
        ("pe_0_75", {"pe_min_ratio": 0.75, "pe_window": 10, "pe_lookback_bars": 780}),
        (
            "peer3_pe05",
            {
                **mag7_peers,
                "peer_align_min": 3,
                "pe_min_ratio": 0.5,
                "pe_window": 10,
                "pe_lookback_bars": 780,
            },
        ),
        (
            "si043_pe05",
            {
                **mag7_peers,
                "si_min": 0.43,
                "pe_min_ratio": 0.5,
                "pe_window": 10,
                "pe_lookback_bars": 780,
            },
        ),
    ]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--profile", default=str(GOOGL_PROF))
    p.add_argument("--start-date", default="2026-05-01")
    p.add_argument("--end-date", default="2026-07-13")
    p.add_argument("--tag", default="si_pe_ablation_mag7_googl_may_jul")
    p.add_argument("--also-mag7-jan-jul", action="store_true")
    p.add_argument("--scheme", default="single")
    args = p.parse_args()

    jobs = [(args.profile, args.start_date, args.end_date, args.tag)]
    if args.also_mag7_jan_jul:
        jobs.append(
            (
                str(MAG7_PROF),
                "2026-01-02",
                "2026-07-13",
                "si_pe_ablation_mag7_jan_jul",
            )
        )

    for prof_path, start, end, tag in jobs:
        base = load_profile(prof_path)
        base["date_range"]["start"] = start
        base["date_range"]["end"] = end
        base.setdefault("trade", {})["bar_availability_delay_seconds"] = 60
        out = Path(base["_paths"]["results_dir"]) / tag
        out.mkdir(parents=True, exist_ok=True)

        scoreboard: list[dict[str, Any]] = []
        for name, sig_patch in _variants():
            prof = deepcopy(base)
            prof.setdefault("signal", {}).update(sig_patch)
            print(f"[{tag}] {name} ...", flush=True)
            got = _run(prof, args.scheme)
            row = {
                "name": name,
                **{k: got[k] for k in (
                    "total_ret", "maxdd", "n_trades", "trade_win", "trade_exp",
                    "n_peer_block", "n_si_block", "n_pe_block", "streak_days_compound",
                )},
                "signal": sig_patch,
            }
            scoreboard.append(row)
            print(
                f"  ret={row['total_ret']:+.2%} dd={row['maxdd']:.2%} n={row['n_trades']} "
                f"win={row['trade_win']:.1%} peer/si/pe_block="
                f"{row['n_peer_block']}/{row['n_si_block']}/{row['n_pe_block']} "
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
            r for r in scoreboard[1:]
            if r["total_ret"] > baseline["total_ret"] and r["maxdd"] >= baseline["maxdd"] - 0.05
        ]
        pick = max(viable, key=lambda r: r["total_ret"]) if viable else None
        summary = {
            "period": f"{start}..{end}",
            "profile": prof_path,
            "baseline": {k: baseline[k] for k in baseline if k != "signal"},
            "best_by_ret": {k: best[k] for k in best if k != "signal"},
            "pick_uplift_dd_ok": (
                {k: pick[k] for k in pick if k != "signal"} if pick else None
            ),
            "uplift_pp": {r["name"]: r["total_ret"] - baseline["total_ret"] for r in scoreboard[1:]},
            "promote": bool(pick is not None),
        }
        (out / "scoreboard.json").write_text(json.dumps(scoreboard, indent=2, default=str), encoding="utf-8")
        (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2, default=str))
        print(f"→ {out}")


if __name__ == "__main__":
    main()

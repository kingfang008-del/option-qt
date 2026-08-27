#!/usr/bin/env python3
"""Re-run research_baseline offline replay on causal 1s→1m stock bars.

Rebuilds mf/streak/regime frames from ``paths.stock_1s_root`` (left-labeled 1m,
``bar_availability_delay_seconds=60``). Compares to the cached-1m headline
May–Jul ``total_ret≈18.56`` (+1856%).

Example:
  PYTHONPATH=. python -m maga7.tools.run_research_baseline_stock1s_replay \\
    --tag research_baseline_stock1s_replay_dual
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from maga7.common.config import load_profile
from maga7.common.replay import run_offline_replay
from maga7.common.stock_1s import (
    build_stock_by_from_1s,
    coverage_report,
    regime_gate_from_1s,
    session_dates,
)

PROFILE = (
    "maga7/CONFIG/strategy_profiles/"
    "single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1.json"
)

# Headline from docs/research_full_day_peer3_baseline.md (preagg 1m cache)
HEADLINE = {
    "may_jul_0716": {
        "total_ret": 18.556,
        "maxdd": -0.0537,
        "n_trades": 57,
        "trade_win": 0.6842,
        "note": "preagg 1m May–Jul →07-16 L2+05+sl55+tt600d",
    },
}

WINDOWS = (
    ("may_jul_0716", "2026-05-01", "2026-07-16"),  # headline apples-to-apples
    ("may_jul_0722", "2026-05-01", "2026-07-22"),  # extend with later tape
    ("jan_mar", "2026-01-02", "2026-03-31"),
    ("apr_jul_0716", "2026-04-01", "2026-07-16"),
)


def _slice_stock_by(stock_by: dict, start: str, end: str) -> dict:
    out = {}
    for sym, df in stock_by.items():
        if df is None or df.empty:
            continue
        sub = df[(df["date"].astype(str) >= start) & (df["date"].astype(str) <= end)]
        if not sub.empty:
            out[sym] = sub.reset_index(drop=True)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", default=PROFILE)
    ap.add_argument("--tag", default="research_baseline_stock1s_replay_dual")
    ap.add_argument("--scheme", default="single")
    args = ap.parse_args(argv)

    profile = load_profile(args.profile)
    # Ensure causal bar clock (profile already has 60; pin explicitly)
    profile.setdefault("trade", {})["bar_availability_delay_seconds"] = 60

    start_all = min(w[1] for w in WINDOWS)
    end_all = max(w[2] for w in WINDOWS)
    profile["date_range"]["start"] = start_all
    profile["date_range"]["end"] = end_all

    out_dir = Path(profile["_paths"]["results_dir"]) / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    dates = session_dates(start_all, end_all)
    print(
        f"building stock_by 1s→1m {start_all}..{end_all} sessions≈{len(dates)} "
        f"profile={profile.get('profile_id')} rev={profile.get('research_revision')}",
        flush=True,
    )
    stock_by = build_stock_by_from_1s(profile, dates=dates, include_refs=True)
    # Required when stock_root (preagg) is absent: Watchdog Hunt only arms if
    # regime_gate is non-None (replay day loop gates begin_day on it).
    regime_gate = regime_gate_from_1s(profile, stock_by)
    cov = coverage_report(stock_by, dates=dates, symbols=list(profile["symbols"]))
    (out_dir / "stock_1s_coverage.json").write_text(
        json.dumps(cov, indent=2), encoding="utf-8"
    )
    print(
        f"symbols={list(stock_by)} bars={sum(len(v) for v in stock_by.values())} "
        f"regime_gate={'ok' if regime_gate is not None else 'NONE'}",
        flush=True,
    )

    window_rows: dict[str, dict[str, Any]] = {}
    for wname, w0, w1 in WINDOWS:
        cfg = copy.deepcopy(profile)
        cfg["date_range"]["start"] = w0
        cfg["date_range"]["end"] = w1
        cfg["trade"]["bar_availability_delay_seconds"] = 60
        sb = _slice_stock_by(stock_by, w0, w1)
        print(f"\n=== replay {wname} {w0}..{w1} ===", flush=True)
        result = run_offline_replay(
            cfg, scheme=args.scheme, stock_by=sb, regime_gate=regime_gate
        )
        s = result["summary"]
        win_dir = out_dir / wname
        win_dir.mkdir(parents=True, exist_ok=True)
        (win_dir / "summary.json").write_text(
            json.dumps(s, indent=2, default=str), encoding="utf-8"
        )
        result["trades"].to_csv(win_dir / "trades.csv", index=False)
        result["daily"].to_csv(win_dir / "daily.csv", index=False)
        result["topk"].to_csv(win_dir / "topk_signals.csv", index=False)

        row = {
            "window": wname,
            "start": w0,
            "end": w1,
            "stock_source": "raw_1s→1m",
            "delay_sec": 60,
            "total_ret": s.get("total_ret"),
            "maxdd": s.get("maxdd"),
            "n_trades": s.get("n_trades"),
            "trade_win": s.get("trade_win"),
            "trade_exp": s.get("trade_exp"),
            "day_win": s.get("day_win"),
            "n_peer_block": s.get("n_peer_block"),
            "n_regime_block": s.get("n_regime_block"),
            "n_hunt_signals": s.get("n_hunt_signals"),
            "n_hunt_trades": s.get("n_hunt_trades"),
            "n_day_halt": s.get("n_day_halt"),
            "n_event_block": s.get("n_event_block"),
        }
        hl = HEADLINE.get(wname)
        if hl:
            row["headline_total_ret"] = hl["total_ret"]
            row["headline_maxdd"] = hl["maxdd"]
            row["headline_n_trades"] = hl["n_trades"]
            row["headline_trade_win"] = hl["trade_win"]
            tr = float(s.get("total_ret") or 0)
            row["ret_vs_headline"] = tr / hl["total_ret"] if hl["total_ret"] else None
            row["delta_ret"] = tr - hl["total_ret"]
            row["delta_win"] = float(s.get("trade_win") or 0) - hl["trade_win"]
            row["delta_n"] = int(s.get("n_trades") or 0) - int(hl["n_trades"])
        window_rows[wname] = row
        print(
            f"  ret={float(s.get('total_ret') or 0)*100:+.1f}% "
            f"dd={float(s.get('maxdd') or 0)*100:.1f}% "
            f"n={s.get('n_trades')} win={float(s.get('trade_win') or 0)*100:.1f}%",
            flush=True,
        )
        if hl:
            print(
                f"  vs headline +1856%: keep={row['ret_vs_headline']:.3f} "
                f"Δret={row['delta_ret']:+.3f} Δwin={row['delta_win']*100:+.1f}pp "
                f"Δn={row['delta_n']:+d}",
                flush=True,
            )

    mj = window_rows.get("may_jul_0716") or {}
    keep = mj.get("ret_vs_headline")
    if keep is None:
        verdict = "NO_HEADLINE_COMPARE"
    elif keep >= 0.85 and float(mj.get("total_ret") or 0) > 0:
        verdict = "KEEP_NEAR_HEADLINE"
    elif float(mj.get("total_ret") or 0) > 0:
        verdict = "DEGRADED_VS_PREAGG"
    else:
        verdict = "REJECT"

    summary = {
        "profile": args.profile,
        "profile_id": profile.get("profile_id"),
        "research_revision": profile.get("research_revision"),
        "stock_source": "/mnt/s990/data/raw_1s/stocks",
        "bar_clock": "1s→left-labeled 1m + delay=60s",
        "scheme": args.scheme,
        "windows": window_rows,
        "headline_may_jul": HEADLINE["may_jul_0716"],
        "verdict": verdict,
        "note": (
            "Offline research_baseline on causal 1s→1m. Compare may_jul_0716 to "
            "preagg headline total_ret=18.56. Scanner live path is separate "
            "(run_replay_stock_1s)."
        ),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )

    print(f"\n=== verdict={verdict} ===", flush=True)
    print(json.dumps(window_rows, indent=2, default=str), flush=True)
    print(f"wrote {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

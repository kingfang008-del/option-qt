#!/usr/bin/env python3
"""T+30 Rule-A on causal 1s→{5s,15s,60s} bars (HF bar clock).

Rebuilds OHLCV from ``/mnt/s990/data/raw_1s/stocks``, then attaches mf/streak/vol_z
on the chosen bar grid. ``bar_availability_delay_seconds = bar_seconds`` so
left-labeled bars are only used after close.

Two feature regimes per bar size:
  - wall: mf/streak/vol_ma scaled to ~same wall-clock as 1m baseline (10m/8m/20m)
  - hf:   shorter wall-clock pockets (1–5m mf, shorter streak)

Example:
  PYTHONPATH=. python -m maga7.tools.run_t30_hf_bars_stock1s \\
    --tag research_t30_hf_bars_stock1s_dual
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from maga7.common.config import load_profile
from maga7.common.replay import run_offline_replay
from maga7.common.stock_1s import (
    attach_features_stock_by,
    build_bars_by_from_1s,
    coverage_report,
    session_dates,
)

T30 = (
    "maga7/CONFIG/strategy_profiles/"
    "single_qqq_open_ladder_atm5otm_t30_rails_p20_googl_peer3_v1.json"
)

WINDOWS = (
    ("jan_mar", "2026-01-02", "2026-03-31"),
    ("may_jul", "2026-05-01", "2026-07-22"),
)


def _presets_for_bar(bar_seconds: int) -> list[dict[str, Any]]:
    """mf/streak/vol_ma in *bars* for this bar clock."""
    b = int(bar_seconds)
    scale = max(1, 60 // b)  # bars per wall-clock minute
    wall = {
        "name": "wall_eq",
        "mf_window": 10 * scale,
        "streak_min": 8 * scale,
        "vol_ma_window": 20 * scale,
    }
    if b == 60:
        return [wall]  # same as frozen 1m baseline feature windows
    out = [wall]
    for name, mf_m, streak_m, vol_m in (
        ("hf_5m", 5, 4, 10),
        ("hf_2m", 2, 1, 4),
        ("hf_1m", 1, None, 2),  # streak ≈ 40s wall
    ):
        streak_bars = max(2, 40 // b) if streak_m is None else max(2, int(streak_m) * scale)
        out.append(
            {
                "name": name,
                "mf_window": max(2, int(mf_m) * scale),
                "streak_min": streak_bars,
                "vol_ma_window": max(5, int(vol_m) * scale),
            }
        )
    return out


def _slice_stock_by(
    stock_by: dict[str, pd.DataFrame], start: str, end: str
) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for sym, df in stock_by.items():
        if df is None or df.empty:
            continue
        sub = df[(df["date"].astype(str) >= start) & (df["date"].astype(str) <= end)]
        if not sub.empty:
            out[sym] = sub.reset_index(drop=True)
    return out


def _ok_window(s: dict[str, Any], *, min_n: int) -> bool:
    try:
        return (
            int(s.get("n_trades") or 0) >= min_n
            and float(s.get("total_ret") or -1) > 0
            and float(s.get("maxdd") or -1) > -0.50
        )
    except Exception:
        return False


def _parse_ints(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", default=T30)
    ap.add_argument("--tag", default="research_t30_hf_bars_stock1s_dual")
    ap.add_argument("--scheme", default="single", choices=["single", "m5", "m5_circuit"])
    ap.add_argument("--bar-sec", default="5,15,60", help="comma bar seconds")
    ap.add_argument("--hold", default="10,30", help="comma hold_minutes")
    ap.add_argument("--from-prev", default="0.02")
    ap.add_argument("--vol-z", default="1.0")
    ap.add_argument("--peer-min", type=int, default=3)
    ap.add_argument("--min-n", type=int, default=15)
    args = ap.parse_args(argv)

    profile = load_profile(args.profile)
    profile["trade"]["exit_mode"] = "none"

    start_all = min(w[1] for w in WINDOWS)
    end_all = max(w[2] for w in WINDOWS)
    profile["date_range"]["start"] = start_all
    profile["date_range"]["end"] = end_all

    out_dir = Path(profile["_paths"]["results_dir"]) / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    dates = session_dates(start_all, end_all)
    bar_secs = _parse_ints(args.bar_sec)
    holds = _parse_ints(args.hold)
    fp = float(args.from_prev)
    vz = float(args.vol_z)
    peer = int(args.peer_min)

    score_rows: list[dict[str, Any]] = []
    cells: list[tuple[Any, ...]] = []
    for b in bar_secs:
        for preset in _presets_for_bar(b):
            for hold in holds:
                cells.append((b, preset, hold))

    print(
        f"HF bar scan days≈{len(dates)} bar_sec={bar_secs} holds={holds} "
        f"cells={len(cells)} fp={fp} vz={vz} peer={peer}",
        flush=True,
    )

    # Cache raw bars per bar_seconds
    bars_cache: dict[int, dict[str, pd.DataFrame]] = {}
    for b in bar_secs:
        print(f"building bars bar_sec={b} …", flush=True)
        bars_cache[b] = build_bars_by_from_1s(
            profile, dates=dates, include_refs=True, bar_seconds=b
        )
        cov = coverage_report(
            bars_cache[b], dates=dates, symbols=list(profile["symbols"])
        )
        (out_dir / f"coverage_bar{b}.json").write_text(
            json.dumps(cov, indent=2), encoding="utf-8"
        )
        nbar = sum(len(v) for v in bars_cache[b].values())
        print(f"  symbols={list(bars_cache[b])} bars={nbar}", flush=True)

    for i, (b, preset, hold) in enumerate(cells, 1):
        delay = int(b)
        mf = int(preset["mf_window"])
        streak = int(preset["streak_min"])
        vol_ma = int(preset["vol_ma_window"])
        pname = str(preset["name"])

        stock_by = attach_features_stock_by(
            bars_cache[b],
            mf_window=mf,
            vol_ma_window=vol_ma,
            signal_cfg=profile.get("signal") or {},
        )

        win_stats: dict[str, dict[str, Any]] = {}
        for wname, w0, w1 in WINDOWS:
            cfg = copy.deepcopy(profile)
            cfg["date_range"]["start"] = w0
            cfg["date_range"]["end"] = w1
            cfg["signal"]["mf_window"] = mf
            cfg["signal"]["vol_ma_window"] = vol_ma
            cfg["signal"]["streak_min"] = streak
            cfg["signal"]["from_prev_abs"] = fp
            cfg["signal"]["vol_z_min"] = vz
            cfg["signal"]["peer_align_min"] = peer
            cfg["trade"]["exit_mode"] = "none"
            cfg["trade"]["hold_minutes"] = int(hold)
            cfg["trade"]["bar_availability_delay_seconds"] = delay
            sb = _slice_stock_by(stock_by, w0, w1)
            result = run_offline_replay(cfg, scheme=args.scheme, stock_by=sb)
            s = result["summary"]
            win_stats[wname] = s
            print(
                f"[{i}/{len(cells)} {wname}] bar={b}s {pname} "
                f"mf={mf} s={streak} H={hold} → "
                f"ret={float(s.get('total_ret') or 0)*100:+.1f}% "
                f"dd={float(s.get('maxdd') or 0)*100:.1f}% "
                f"n={s.get('n_trades')} "
                f"win={float(s.get('trade_win') or 0)*100:.0f}%",
                flush=True,
            )

        dual = all(
            _ok_window(win_stats[w], min_n=int(args.min_n)) for w, _, _ in WINDOWS
        )
        is_base = b == 60 and pname in ("base_1m", "wall_eq") and hold == 30
        row: dict[str, Any] = {
            "bar_seconds": b,
            "preset": pname,
            "mf_window": mf,
            "streak_min": streak,
            "vol_ma_window": vol_ma,
            "hold_minutes": hold,
            "delay_sec": delay,
            "from_prev_abs": fp,
            "vol_z_min": vz,
            "peer_align_min": peer,
            "is_baseline_1m": is_base,
            "dual_pass": dual,
            "stock_source": "raw_1s",
        }
        for wname, _, _ in WINDOWS:
            s = win_stats[wname]
            for k in (
                "total_ret",
                "maxdd",
                "n_trades",
                "trade_win",
                "trade_exp",
                "n_peer_block",
                "n_regime_block",
            ):
                row[f"{wname}_{k}"] = s.get(k)
            ret = float(s.get("total_ret") or 0)
            dd = abs(float(s.get("maxdd") or 0))
            row[f"{wname}_calmar_like"] = ret / max(dd, 0.05)

        row["dual_calmar_min"] = min(
            float(row["jan_mar_calmar_like"] or 0),
            float(row["may_jul_calmar_like"] or 0),
        )
        row["dual_ret_sum"] = float(row.get("jan_mar_total_ret") or 0) + float(
            row.get("may_jul_total_ret") or 0
        )
        row["dual_win_min"] = min(
            float(row.get("jan_mar_trade_win") or 0),
            float(row.get("may_jul_trade_win") or 0),
        )
        score_rows.append(row)
        if dual:
            print(
                f"  *** DUAL PASS bar={b} {pname} H={hold} "
                f"sum_ret={row['dual_ret_sum']*100:+.1f}% "
                f"win_min={row['dual_win_min']*100:.0f}%",
                flush=True,
            )

    score = pd.DataFrame(score_rows)
    score.to_csv(out_dir / "scoreboard.csv", index=False)

    dual_df = score[score["dual_pass"]].copy()
    if len(dual_df):
        dual_df = dual_df.sort_values(
            ["dual_calmar_min", "dual_ret_sum"], ascending=[False, False]
        )
    else:
        dual_df = score.sort_values(
            ["dual_calmar_min", "dual_ret_sum"], ascending=[False, False]
        )

    best = dual_df.iloc[0].to_dict() if len(dual_df) else None
    base_hits = score[score["is_baseline_1m"] & (score["hold_minutes"] == 30)]
    baseline_row = base_hits.iloc[0].to_dict() if len(base_hits) else None

    # best by win among dual pass / overall
    by_win = score.sort_values(
        ["dual_win_min", "dual_ret_sum"], ascending=[False, False]
    )

    summary = {
        "profile": args.profile,
        "stock_source": "/mnt/s990/data/raw_1s/stocks",
        "exit_mode": "none",
        "windows": [list(w) for w in WINDOWS],
        "bar_sec": bar_secs,
        "holds": holds,
        "from_prev": fp,
        "vol_z": vz,
        "peer_min": peer,
        "n_cells": int(len(score)),
        "n_dual_pass": int(score["dual_pass"].sum()) if len(score) else 0,
        "verdict": "PASS" if int(score["dual_pass"].sum() or 0) > 0 else "REJECT",
        "baseline_1m_H30": baseline_row,
        "best_dual": best,
        "best_by_win": by_win.head(5).to_dict(orient="records") if len(by_win) else [],
        "top_dual": dual_df.head(15).to_dict(orient="records") if len(dual_df) else [],
        "note": (
            "1s→Ns left-labeled bars; delay=bar_seconds. wall_eq scales mf/streak "
            "to ~10m/8m wall-clock; hf_* uses shorter pockets. Pure T+30 clock hold."
        ),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )

    print("\n=== baseline 1m H30 ===", flush=True)
    print(json.dumps(baseline_row, indent=2, default=str), flush=True)
    print(
        f"\n=== best dual (n_pass={summary['n_dual_pass']}) "
        f"verdict={summary['verdict']} ===",
        flush=True,
    )
    print(json.dumps(best, indent=2, default=str), flush=True)
    cols = [
        c
        for c in [
            "bar_seconds",
            "preset",
            "mf_window",
            "streak_min",
            "hold_minutes",
            "dual_pass",
            "dual_calmar_min",
            "dual_ret_sum",
            "dual_win_min",
            "jan_mar_total_ret",
            "jan_mar_trade_win",
            "jan_mar_n_trades",
            "may_jul_total_ret",
            "may_jul_trade_win",
            "may_jul_n_trades",
        ]
        if c in score.columns
    ]
    print("\n=== top by dual_calmar ===", flush=True)
    print(dual_df[cols].head(15).to_string(index=False), flush=True)
    print("\n=== top by dual_win_min ===", flush=True)
    print(by_win[cols].head(10).to_string(index=False), flush=True)
    print(f"wrote {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

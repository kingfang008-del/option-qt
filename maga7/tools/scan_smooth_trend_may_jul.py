#!/usr/bin/env python3
"""Scan Mag7 May–Jul for own-path smooth launches vs significant day moves.

Evaluates whether the 07-20 MSFT-style detector (own-path start, not cross-section
#1) covers most ≥1.5% session legs. Miss analysis suggests complementary gauges.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from maga7.common.config import load_profile
from maga7.common.signals import attach_mf_features, load_stock_month_files
from maga7.common.smooth_trend import (
    SmoothLaunchConfig,
    detect_smooth_launches_day,
    extract_significant_moves,
    match_launches_to_moves,
)

SYMS = ["NVDA", "TSLA", "AAPL", "AMZN", "META", "MSFT", "AMD", "GOOGL"]
MONTHS = ["2026-05", "2026-06", "2026-07"]


def _miss_features(stock_day: pd.DataFrame, move) -> dict:
    """Features at move start + early window for miss analysis."""
    day = stock_day.copy()
    day["timestamp"] = pd.to_datetime(day["timestamp"])
    if getattr(day["timestamp"].dt, "tz", None) is None:
        day["timestamp"] = day["timestamp"].dt.tz_localize("America/New_York")
    else:
        day["timestamp"] = day["timestamp"].dt.tz_convert("America/New_York")
    t0 = pd.Timestamp(move.start_ts)
    # first 15m of the leg
    w = day[(day.timestamp >= t0) & (day.timestamp <= t0 + pd.Timedelta(minutes=15))]
    if len(w) < 5:
        return {"gap": True}
    c = w["close"].astype(float).to_numpy()
    rets = np.diff(c) / c[:-1]
    net = c[-1] / c[0] - 1.0 if move.direction == "UP" else 1.0 - c[-1] / c[0]
    sumabs = float(np.abs(rets).sum()) or 1e-12
    # MF at start if present
    at = day[day.timestamp <= t0]
    mf10 = float(at.iloc[-1]["mf10"]) if len(at) and "mf10" in at.columns and pd.notna(at.iloc[-1]["mf10"]) else None
    vol_z = float(at.iloc[-1]["vol_z"]) if len(at) and "vol_z" in at.columns and pd.notna(at.iloc[-1]["vol_z"]) else None
    return {
        "early15_ret": float(net),
        "early15_eff": float(abs(net) / sumabs),
        "early15_up": float((rets > 0).mean()) if move.direction == "UP" else float((rets < 0).mean()),
        "early15_std": float(np.std(rets)),
        "mf10_at_start": mf10,
        "vol_z_at_start": vol_z,
        "move_ret": float(move.move_ret),
        "leg_minutes": (move.end_ts - move.start_ts).total_seconds() / 60.0,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--profile",
        default=(
            "maga7/CONFIG/strategy_profiles/"
            "single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1.json"
        ),
    )
    ap.add_argument("--start-date", default="2026-05-01")
    ap.add_argument("--end-date", default="2026-07-17")
    ap.add_argument("--min-move", type=float, default=0.015)
    ap.add_argument("--out", default="/mnt/s990/data/maga7/results/research_smooth_trend_may_jul")
    args = ap.parse_args(argv)

    prof = load_profile(args.profile)
    stock_root = Path(prof["_paths"]["stock_root"]).expanduser()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cfg = SmoothLaunchConfig()
    launch_rows: list[dict] = []
    move_rows: list[dict] = []
    match_rows: list[dict] = []
    miss_feat_rows: list[dict] = []

    for sym in SYMS:
        print(f"[load] {sym}", flush=True)
        raw = load_stock_month_files(stock_root, sym, MONTHS)
        if raw.empty:
            print(f"  empty {sym}")
            continue
        raw = attach_mf_features(raw)
        dates = sorted(
            d
            for d in raw["date"].astype(str).unique()
            if args.start_date <= d <= args.end_date
        )
        for date in dates:
            day = raw[raw["date"].astype(str) == date]
            launches = detect_smooth_launches_day(
                day, symbol=sym, date=date, cfg=cfg, directions=("UP", "DN")
            )
            moves = extract_significant_moves(
                day, symbol=sym, date=date, min_move=float(args.min_move)
            )
            for ln in launches:
                launch_rows.append(ln.to_dict())
            for mv in moves:
                move_rows.append(
                    {
                        "date": mv.date,
                        "symbol": mv.symbol,
                        "direction": mv.direction,
                        "start_ts": str(mv.start_ts),
                        "end_ts": str(mv.end_ts),
                        "start_px": mv.start_px,
                        "end_px": mv.end_px,
                        "move_ret": mv.move_ret,
                        "max_adverse": mv.max_adverse,
                    }
                )
            matches = match_launches_to_moves(launches, moves)
            for m in matches:
                mv = m["move"]
                ln = m["launch"]
                row = {
                    "date": mv.date,
                    "symbol": mv.symbol,
                    "direction": mv.direction,
                    "move_ret": mv.move_ret,
                    "move_start": str(mv.start_ts),
                    "move_end": str(mv.end_ts),
                    "hit": bool(m["hit"]),
                    "delay_min": m["delay_min"],
                    "capture_ret": m["capture_ret"],
                    "capture_frac": m["capture_frac"],
                    "detect_ts": None if ln is None else str(ln.detect_ts),
                    "detect_score": None if ln is None else ln.score,
                }
                match_rows.append(row)
                if not m["hit"]:
                    feat = _miss_features(day, mv)
                    miss_feat_rows.append({**row, **feat})

    launches_df = pd.DataFrame(launch_rows)
    moves_df = pd.DataFrame(move_rows)
    match_df = pd.DataFrame(match_rows)
    miss_df = pd.DataFrame(miss_feat_rows)

    launches_df.to_csv(out / "launches.csv", index=False)
    moves_df.to_csv(out / "moves.csv", index=False)
    match_df.to_csv(out / "matches.csv", index=False)
    if not miss_df.empty:
        miss_df.to_csv(out / "misses.csv", index=False)

    # summary metrics
    n_moves = len(match_df)
    n_hit = int(match_df["hit"].sum()) if n_moves else 0
    by_dir = (
        match_df.groupby("direction")
        .agg(n=("hit", "size"), hit=("hit", "sum"), hit_rate=("hit", "mean"), med_delay=("delay_min", "median"))
        .reset_index()
        if n_moves
        else pd.DataFrame()
    )
    # false alarms: launches with no matched move same day/dir/symbol within window
    fa = 0
    if not launches_df.empty and not match_df.empty:
        hit_keys = set(
            zip(
                match_df.loc[match_df.hit, "date"],
                match_df.loc[match_df.hit, "symbol"],
                match_df.loc[match_df.hit, "direction"],
                match_df.loc[match_df.hit, "detect_ts"],
            )
        )
        for _, r in launches_df.iterrows():
            key = (r["date"], r["symbol"], r["direction"], r["detect_ts"])
            if key not in hit_keys:
                # soft FA: no hit move that used this detect_ts
                fa += 1
    # precision-ish: matched launches / all launches
    n_matched_launches = int(match_df["hit"].sum()) if n_moves else 0
    n_launches = len(launches_df)

    # miss typology
    miss_notes = []
    if not miss_df.empty:
        chop = miss_df[(miss_df.get("early15_eff", 0) < 0.35) | (miss_df.get("early15_up", 0) < 0.55)]
        gap = miss_df[miss_df.get("early15_ret", 0).fillna(0) >= 0.008]  # jumped too fast for smooth
        slow = miss_df[miss_df.get("early15_ret", 0).fillna(0) < 0.001]
        miss_notes = {
            "n_miss": int(len(miss_df)),
            "n_choppy_early": int(len(chop)),
            "n_gap_impulse_early": int(len(gap)),
            "n_too_slow_early": int(len(slow)),
            "med_miss_move_ret": float(miss_df["move_ret"].median()),
            "med_hit_move_ret": float(match_df.loc[match_df.hit, "move_ret"].median())
            if n_hit
            else None,
        }

    # 07-20 MSFT sanity
    msft_0720 = match_df[
        (match_df.date == "2026-07-20")
        & (match_df.symbol == "MSFT")
        & (match_df.direction == "UP")
    ]
    sanity = msft_0720.to_dict(orient="records")

    summary = {
        "window": {"start": args.start_date, "end": args.end_date},
        "min_move": args.min_move,
        "cfg": cfg.__dict__,
        "n_moves": n_moves,
        "n_hits": n_hit,
        "hit_rate": (n_hit / n_moves) if n_moves else None,
        "n_launches": n_launches,
        "n_matched_launches": n_matched_launches,
        "launch_precision_proxy": (n_matched_launches / n_launches) if n_launches else None,
        "n_unmatched_launches": fa,
        "by_direction": by_dir.to_dict(orient="records") if len(by_dir) else [],
        "med_delay_min_hits": float(match_df.loc[match_df.hit, "delay_min"].median())
        if n_hit
        else None,
        "med_capture_frac_hits": float(match_df.loc[match_df.hit, "capture_frac"].median())
        if n_hit
        else None,
        "miss_analysis": miss_notes,
        "sanity_msft_20260720": sanity,
        "complementary_indicators": [
            {
                "name": "gap_impulse_detector",
                "when": "early15_ret large but path_eff low / few bars",
                "why": "catches vertical launches smooth-grind detector skips",
            },
            {
                "name": "vwap_reclaim_or_hold",
                "when": "price accepts above/below session VWAP after launch",
                "why": "filters fake smooth grinds that fail acceptance",
            },
            {
                "name": "volume_z_or_participation",
                "when": "vol_z rising with the smooth leg",
                "why": "07-20 MSFT had participation; quiet drifts false-alarm",
            },
            {
                "name": "peer_breadth_confirm",
                "when": "≥N Mag7 same direction (not for start, for size-up)",
                "why": "start stays own-path; breadth upgrades conviction",
            },
            {
                "name": "regime_qqq_align",
                "when": "QQQ same-direction over launch window",
                "why": "reduces single-name noise fades",
            },
            {
                "name": "time_of_day_prior",
                "when": "opens 09:45–11:30 weighted higher",
                "why": "most durable legs start morning; afternoon grind weaker",
            },
        ],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    # REPORT
    lines = [
        "# Smooth Trend Own-Path Scan — May–Jul 2026",
        "",
        f"**Hit rate: `{summary['hit_rate']:.1%}`** ({n_hit}/{n_moves} legs ≥{args.min_move:.1%})",
        f"**Launches: {n_launches}**, precision proxy `{summary['launch_precision_proxy']}`",
        f"**Median delay on hits: {summary['med_delay_min_hits']} min**, "
        f"median capture frac `{summary['med_capture_frac_hits']}`",
        "",
        "## By direction",
        "",
        by_dir.to_markdown(index=False) if len(by_dir) else "(none)",
        "",
        "## Miss typology",
        "",
        "```json",
        json.dumps(miss_notes, indent=2),
        "```",
        "",
        "## 07-20 MSFT sanity",
        "",
        "```json",
        json.dumps(sanity, indent=2, default=str),
        "```",
        "",
        "## Complementary indicators if hit rate insufficient",
        "",
    ]
    for c in summary["complementary_indicators"]:
        lines.append(f"- **{c['name']}**: {c['why']} (`{c['when']}`)")
    lines.append("")
    (out / "REPORT.md").write_text("\n".join(lines))

    print(json.dumps({k: summary[k] for k in summary if k != "complementary_indicators"}, indent=2, default=str))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

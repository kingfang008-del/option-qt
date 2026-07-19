#!/usr/bin/env python3
"""Multi-factor top2 scan (research). Does not trade or change freeze TopK.

Question: can META/NVDA-style days enter multi-factor top2 (causally)?
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from maga7.common.config import load_profile
from maga7.common.macro_unusual import prepare_day
from maga7.common.multifactor_rank import (
    MultiFactorConfig,
    first_top2_entry,
    score_universe_at,
)
from maga7.common.regime import Mag7RegimeGate
from maga7.common.signals import (
    _rule_a_kwargs_from_cfg,
    attach_mf_features,
    first_rule_a_day,
    load_stock_month_files,
)


def _months(start: str, end: str) -> list[str]:
    return [str(p) for p in pd.period_range(start[:7], end[:7], freq="M")]


def _precompute_medians(
    stock_by: dict[str, pd.DataFrame], symbols: list[str], lookback: int
) -> dict[str, dict[str, dict[str, float]]]:
    """symbol -> date -> tod -> cum$  (all days); caller slices lookback."""
    curves: dict[str, dict[str, dict[str, float]]] = {}
    for sym in symbols:
        sdf = stock_by.get(sym)
        if sdf is None or sdf.empty:
            continue
        by_d: dict[str, dict[str, float]] = {}
        for d in sorted(sdf["date"].astype(str).unique()):
            day = prepare_day(sdf, d)
            if day.empty:
                continue
            by_d[str(d)] = {str(r.tod): float(r.cum_dvol) for r in day.itertuples(index=False)}
        curves[sym] = by_d
    return curves


def _med_at(
    curves: dict[str, dict[str, dict[str, float]]],
    *,
    sym: str,
    date: str,
    lookback: int,
) -> dict[str, float]:
    prev = sorted(d for d in curves.get(sym, {}) if d < date)[-int(lookback) :]
    by_tod: dict[str, list[float]] = {}
    for d in prev:
        for tod, v in curves[sym][d].items():
            by_tod.setdefault(tod, []).append(float(v))
    return {tod: float(np.median(vs)) for tod, vs in by_tod.items() if vs}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--profile",
        default=(
            "maga7/CONFIG/strategy_profiles/"
            "single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1.json"
        ),
    )
    ap.add_argument("--start-date", default="2026-07-07")
    ap.add_argument("--end-date", default="2026-07-10")
    ap.add_argument(
        "--focus",
        default="2026-07-08:NVDA:UP,2026-07-09:META:UP",
    )
    ap.add_argument("--fp-gate", type=float, default=0.005)
    ap.add_argument("--step-minutes", type=int, default=1)
    ap.add_argument("--snapshot-tod", default="12:00,12:30,13:00")
    ap.add_argument("--out", default="maga7/results/multifactor_top2_scan")
    args = ap.parse_args()

    prof = load_profile(args.profile)
    symbols = list(prof["symbols"])
    stock_root = Path(os.path.expanduser(prof["paths"]["stock_root"]))
    lb_start = (pd.Timestamp(args.start_date) - pd.Timedelta(days=40)).strftime("%Y-%m-%d")
    load_months = _months(lb_start, args.end_date)

    stock_by: dict[str, pd.DataFrame] = {}
    for sym in symbols + ["QQQ"]:
        raw = load_stock_month_files(stock_root, sym, load_months)
        if raw is None or getattr(raw, "empty", True):
            continue
        stock_by[sym] = attach_mf_features(raw)

    cfg = MultiFactorConfig(fp_gate=float(args.fp_gate))
    print("precomputing cum$ curves...")
    curves = _precompute_medians(stock_by, symbols, cfg.lookback_days)
    rule_kw = _rule_a_kwargs_from_cfg(prof.get("signal") or {})
    gate = Mag7RegimeGate.from_profile(prof, months=load_months)

    focus_out = []
    for part in str(args.focus).split(","):
        part = part.strip()
        if not part:
            continue
        bits = part.split(":")
        if len(bits) != 3:
            continue
        d, sym, direction = bits[0].strip(), bits[1].strip().upper(), bits[2].strip().upper()
        tod_med = {s: _med_at(curves, sym=s, date=d, lookback=cfg.lookback_days) for s in symbols}
        hit = first_top2_entry(
            stock_by,
            date=d,
            symbols=symbols,
            symbol=sym,
            direction=direction,  # type: ignore[arg-type]
            cfg=cfg,
            tod_median_by_sym=tod_med,
            step_minutes=int(args.step_minutes),
        )
        # Rule-A
        sdf = stock_by.get(sym)
        ra = first_rule_a_day(sdf[sdf["date"].astype(str) == d], **rule_kw) if sdf is not None else None
        snaps = {}
        for tod in str(args.snapshot_tod).split(","):
            tod = tod.strip()
            ranked = score_universe_at(
                stock_by,
                date=d,
                symbols=symbols,
                asof_tod=tod,
                direction=direction,  # type: ignore[arg-type]
                cfg=cfg,
                tod_median_by_sym=tod_med,
            )
            snaps[tod] = [
                {
                    "rank": s.rank,
                    "symbol": s.symbol,
                    "score": round(s.score, 3),
                    "fp": round(s.fp, 4),
                    "vol_x": round(s.vol_x, 2),
                    "accel": round(s.accel, 4),
                    "rs": round(s.rs, 4),
                }
                for s in ranked[:4]
            ]
        regime_ok = None
        if hit is not None:
            dec = gate.check(direction, hit.asof)
            regime_ok = {"allow": bool(dec.allow), "reason": dec.reason}
        focus_out.append(
            {
                "date": d,
                "symbol": sym,
                "direction": direction,
                "in_top2": hit is not None,
                "first_top2_tod": None if hit is None else hit.tod,
                "first_top2_rank": None if hit is None else hit.rank,
                "first_top2_fp": None if hit is None else hit.fp,
                "first_top2_score": None if hit is None else hit.score,
                "regime_at_first_top2": regime_ok,
                "rule_a_tod": None
                if ra is None
                else pd.Timestamp(ra["sig_ts"]).strftime("%H:%M"),
                "rule_a_dir": None if ra is None else ra.get("dir"),
                "minutes_before_rule_a": (
                    None
                    if hit is None or ra is None
                    else (pd.Timestamp(ra["sig_ts"]) - hit.asof).total_seconds() / 60.0
                ),
                "snapshots": snaps,
            }
        )

    # Also: earliest Rule-A top2 vs multifactor top2 at each Rule-A fire time on focus dates
    compare_rows = []
    for d in sorted({x["date"] for x in focus_out}):
        # collect first rule-a times that day
        fires = []
        for sym in symbols:
            sdf = stock_by.get(sym)
            if sdf is None:
                continue
            day = sdf[sdf["date"].astype(str) == d]
            hit = first_rule_a_day(day, **rule_kw) if not day.empty else None
            if hit:
                fires.append((pd.Timestamp(hit["sig_ts"]), sym, hit["dir"]))
        fires.sort(key=lambda x: x[0])
        earliest = [f"{t.strftime('%H:%M')}:{s}:{di}" for t, s, di in fires[:2]]
        tod_med = {s: _med_at(curves, sym=s, date=d, lookback=cfg.lookback_days) for s in symbols}
        # multifactor UP top2 at 12:30 (afternoon)
        up = score_universe_at(
            stock_by,
            date=d,
            symbols=symbols,
            asof_tod="12:30",
            direction="UP",
            cfg=cfg,
            tod_median_by_sym=tod_med,
        )
        compare_rows.append(
            {
                "date": d,
                "rule_a_earliest_top2": earliest,
                "mf_up_top2_at_1230": [f"{s.rank}:{s.symbol}" for s in up[:2]],
            }
        )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = {
        "window": f"{args.start_date}..{args.end_date}",
        "cfg": {
            "fp_gate": cfg.fp_gate,
            "accel_minutes": cfg.accel_minutes,
            "weights": cfg.weights,
        },
        "focus": focus_out,
        "compare": compare_rows,
        "note": "Research only — not wired into baseline TopK.",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(json.dumps(summary, indent=2, default=str))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

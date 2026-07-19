#!/usr/bin/env python3
"""Scan macro/unusual candidates (research). Does NOT trade or alter TopK.

Focus: when would META/NVDA-style movers first be flagged vs Rule-A.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from maga7.common.config import load_profile
from maga7.common.macro_unusual import MacroUnusualConfig, prepare_day, scan_macro_day
from maga7.common.regime import Mag7RegimeGate
from maga7.common.signals import (
    _rule_a_kwargs_from_cfg,
    attach_mf_features,
    first_rule_a_day,
    load_stock_month_files,
)


def _precompute_day_curves(
    stock_by: dict[str, pd.DataFrame], symbols: list[str]
) -> dict[str, dict[str, dict[str, float]]]:
    """symbol -> date -> tod -> cum_dvol."""
    out: dict[str, dict[str, dict[str, float]]] = {}
    for sym in symbols:
        sdf = stock_by.get(sym)
        if sdf is None or sdf.empty:
            continue
        by_date: dict[str, dict[str, float]] = {}
        for d in sorted(sdf["date"].astype(str).unique()):
            day = prepare_day(sdf, d)
            if day.empty:
                continue
            by_date[str(d)] = {
                str(r.tod): float(r.cum_dvol) for r in day.itertuples(index=False)
            }
        out[sym] = by_date
    return out


def _median_curve(
    curves: dict[str, dict[str, float]], *, before: str, lookback: int
) -> dict[str, float]:
    prev = sorted(d for d in curves if d < before)[-int(lookback) :]
    by_tod: dict[str, list[float]] = {}
    for d in prev:
        for tod, v in curves[d].items():
            by_tod.setdefault(tod, []).append(float(v))
    import numpy as np

    return {tod: float(np.median(vs)) for tod, vs in by_tod.items() if vs}


def _months(start: str, end: str) -> list[str]:
    idx = pd.period_range(start[:7], end[:7], freq="M")
    return [str(p) for p in idx]


def main() -> int:
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
    ap.add_argument("--vol-ratio-min", type=float, default=1.20)
    ap.add_argument("--fp-min", type=float, default=0.01)
    ap.add_argument("--hold-bars", type=int, default=0)
    ap.add_argument("--only-up", action="store_true")
    ap.add_argument(
        "--focus",
        default="2026-07-08:NVDA,2026-07-09:META",
        help="comma date:SYMBOL pairs for spotlight",
    )
    ap.add_argument("--out", default="maga7/results/macro_unusual_scan")
    args = ap.parse_args()

    prof = load_profile(args.profile)
    symbols = list(prof["symbols"])
    stock_root = Path(os.path.expanduser(prof["paths"]["stock_root"]))
    months = _months(args.start_date, args.end_date)
    # need lookback before start
    lb_start = (pd.Timestamp(args.start_date) - pd.Timedelta(days=40)).strftime("%Y-%m-%d")
    load_months = _months(lb_start, args.end_date)

    stock_by: dict[str, pd.DataFrame] = {}
    for sym in symbols + ["QQQ"]:
        raw = load_stock_month_files(stock_root, sym, load_months)
        if raw is None or getattr(raw, "empty", True):
            continue
        stock_by[sym] = attach_mf_features(raw)

    cfg = MacroUnusualConfig(
        vol_ratio_min=float(args.vol_ratio_min),
        fp_min=float(args.fp_min),
        hold_bars=int(args.hold_bars),
        only_up=bool(args.only_up),
    )
    rule_kw = _rule_a_kwargs_from_cfg(prof.get("signal") or {})
    gate = Mag7RegimeGate.from_profile(prof, months=load_months)

    dates_set: set[str] = set()
    for sym in symbols:
        sdf = stock_by.get(sym)
        if sdf is None or sdf.empty:
            continue
        for d in sdf["date"].astype(str).unique():
            if args.start_date <= str(d) <= args.end_date:
                dates_set.add(str(d))
    dates = sorted(dates_set)

    print("precomputing day cum$ curves...")
    curves = _precompute_day_curves(stock_by, symbols)

    rows: list[dict] = []
    for date in dates:
        tod_med = {
            sym: _median_curve(curves.get(sym, {}), before=date, lookback=cfg.lookback_days)
            for sym in symbols
        }
        cands = scan_macro_day(
            stock_by,
            date=date,
            symbols=symbols,
            cfg=cfg,
            tod_median_by_sym=tod_med,
        )
        rule_a: dict[str, dict] = {}
        for sym in symbols:
            sdf = stock_by.get(sym)
            if sdf is None:
                continue
            day = sdf[sdf["date"].astype(str) == date]
            hit = first_rule_a_day(day, **rule_kw) if not day.empty else None
            if hit:
                rule_a[sym] = hit
        for rank, c in enumerate(cands, start=1):
            ra = rule_a.get(c.symbol)
            dec = gate.check(c.direction, c.sig_ts)
            rows.append(
                {
                    "date": c.date,
                    "symbol": c.symbol,
                    "direction": c.direction,
                    "macro_ts": str(c.sig_ts),
                    "macro_tod": pd.Timestamp(c.sig_ts).strftime("%H:%M"),
                    "fp": c.from_prev,
                    "vol_ratio": c.vol_ratio,
                    "score": c.score,
                    "pool_rank": rank,
                    "regime_allow": bool(dec.allow),
                    "regime_reason": dec.reason,
                    "rule_a_dir": None if ra is None else ra.get("dir"),
                    "rule_a_tod": None
                    if ra is None
                    else pd.Timestamp(ra["sig_ts"]).strftime("%H:%M"),
                    "macro_before_rule_a_min": None
                    if ra is None
                    else (pd.Timestamp(ra["sig_ts"]) - pd.Timestamp(c.sig_ts)).total_seconds()
                    / 60.0,
                }
            )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "candidates.csv", index=False)

    # focus spotlight
    focus_rows = []
    for part in str(args.focus).split(","):
        part = part.strip()
        if not part or ":" not in part:
            continue
        d, sym = part.split(":", 1)
        d, sym = d.strip(), sym.strip().upper()
        sub = df[(df["date"] == d) & (df["symbol"] == sym)]
        day_pool = df[df["date"] == d].sort_values("pool_rank")
        focus_rows.append(
            {
                "date": d,
                "symbol": sym,
                "flagged": bool(len(sub)),
                "detail": None if sub.empty else sub.iloc[0].to_dict(),
                "day_top3": day_pool.head(3).to_dict(orient="records"),
            }
        )

    # coverage vs big overnight-ish movers: |eod fp|>=3% with same-sign macro
    eod_hits = []
    for date in dates:
        for sym in symbols:
            sdf = stock_by.get(sym)
            if sdf is None:
                continue
            day = sdf[sdf["date"].astype(str) == date]
            if day.empty:
                continue
            fp_eod = float(day.iloc[-1]["from_prev"])
            if abs(fp_eod) < 0.03:
                continue
            want = "UP" if fp_eod > 0 else "DN"
            hit = df[(df["date"] == date) & (df["symbol"] == sym) & (df["direction"] == want)]
            eod_hits.append(
                {
                    "date": date,
                    "symbol": sym,
                    "eod_fp": fp_eod,
                    "macro_hit": bool(len(hit)),
                    "macro_tod": None if hit.empty else hit.iloc[0]["macro_tod"],
                    "pool_rank": None if hit.empty else int(hit.iloc[0]["pool_rank"]),
                    "regime_allow": None if hit.empty else bool(hit.iloc[0]["regime_allow"]),
                }
            )
    eod_df = pd.DataFrame(eod_hits)
    if not eod_df.empty:
        eod_df.to_csv(out / "big_eod_coverage.csv", index=False)

    summary = {
        "window": f"{args.start_date}..{args.end_date}",
        "cfg": {
            "vol_ratio_min": cfg.vol_ratio_min,
            "fp_min": cfg.fp_min,
            "hold_bars": cfg.hold_bars,
            "only_up": cfg.only_up,
            "window": f"{cfg.window_start}-{cfg.window_end}",
        },
        "n_candidate_rows": int(len(df)),
        "n_days_with_cand": int(df["date"].nunique()) if len(df) else 0,
        "big_eod_n": int(len(eod_df)) if not eod_df.empty else 0,
        "big_eod_macro_hit_rate": (
            float(eod_df["macro_hit"].mean()) if not eod_df.empty else None
        ),
        "focus": focus_rows,
        "note": "Candidates only — not wired into replay TopK / baseline.",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(json.dumps(summary, indent=2, default=str))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

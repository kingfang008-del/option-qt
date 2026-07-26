#!/usr/bin/env python3
"""Overlay am_pulse sleeve on research baseline — daily P&L comparison.

1) Offline Rule-A peer3 replay (``run_offline_replay``)
2) Am-pulse FO quote FillSpec champion trades (CSV or regenerate)
3) Additive overlay: ``day_ret_comb = day_ret_base + sum(pulse pnl_frac)``
   then re-compound equity.

Example:
  PYTHONPATH=. python -m maga7.tools.run_baseline_am_pulse_overlay \\
    --start-date 2026-05-01 --end-date 2026-07-24 \\
    --tag baseline_am_pulse_overlay_may_jul24 \\
    --pulse-trades /mnt/s990/data/maga7/results/research_am_pulse_quote_dual_v2/trades_dual04_pulse_FO_t0.008_tp0.15_sl0.2_sp0.15_lag5.0.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from maga7.common.config import load_profile
from maga7.common.replay import run_offline_replay

PROFILE = (
    "maga7/CONFIG/strategy_profiles/"
    "single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1.json"
)
DEFAULT_PULSE = (
    "/mnt/s990/data/maga7/results/research_am_pulse_quote_dual_v2/"
    "trades_dual04_pulse_FO_t0.008_tp0.15_sl0.2_sp0.15_lag5.0.csv"
)


def _pulse_daily(pulse: pd.DataFrame) -> pd.DataFrame:
    if pulse is None or pulse.empty:
        return pd.DataFrame(columns=["date", "pulse_n", "pulse_add", "pulse_mean", "pulse_win"])
    t = pulse.copy()
    t["date"] = t["date"].astype(str)
    if "pnl_frac" not in t.columns:
        size = t["size"].astype(float) if "size" in t.columns else 0.10
        t["pnl_frac"] = t["ret"].astype(float) * size.astype(float)
    g = t.groupby("date", as_index=False).agg(
        pulse_n=("ret", "count"),
        pulse_add=("pnl_frac", "sum"),
        pulse_mean=("ret", "mean"),
        pulse_win=("ret", lambda s: float((s.astype(float) > 0).mean())),
    )
    return g


def _compound(daily: pd.DataFrame, ret_col: str) -> tuple[pd.Series, float, float]:
    """Return equity path (start 100), total_ret, maxdd."""
    eq = 100.0
    peak = 100.0
    maxdd = 0.0
    path = []
    for r in daily[ret_col].astype(float).fillna(0.0):
        eq *= 1.0 + float(r)
        peak = max(peak, eq)
        maxdd = min(maxdd, eq / peak - 1.0)
        path.append(eq)
    total = eq / 100.0 - 1.0
    return pd.Series(path, index=daily.index), float(total), float(maxdd)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", default=PROFILE)
    ap.add_argument("--start-date", default="2026-05-01")
    ap.add_argument("--end-date", default="2026-07-24")
    ap.add_argument("--tag", default="baseline_am_pulse_overlay")
    ap.add_argument("--pulse-trades", default=DEFAULT_PULSE)
    ap.add_argument("--scheme", default="single")
    ap.add_argument(
        "--skip-baseline",
        default="",
        help="Reuse existing baseline daily.csv / trades.csv directory",
    )
    args = ap.parse_args(argv)

    profile = load_profile(args.profile)
    profile.setdefault("date_range", {})
    profile["date_range"]["start"] = args.start_date
    profile["date_range"]["end"] = args.end_date
    # Overlay is research additive; keep am_pulse shadow (offline replay ignores live drain).
    out = Path(profile["_paths"]["results_dir"]) / args.tag
    out.mkdir(parents=True, exist_ok=True)

    if args.skip_baseline:
        base_dir = Path(args.skip_baseline)
        daily_b = pd.read_csv(base_dir / "daily.csv")
        trades_b = pd.read_csv(base_dir / "trades.csv")
        summary_b = json.loads((base_dir / "summary.json").read_text(encoding="utf-8"))
        print(f"reused baseline {base_dir}", flush=True)
    else:
        print(
            f"baseline replay {args.start_date}..{args.end_date} …",
            flush=True,
        )
        result = run_offline_replay(profile, scheme=args.scheme)
        daily_b = result["daily"]
        trades_b = result["trades"]
        summary_b = result["summary"]
        trades_b.to_csv(out / "baseline_trades.csv", index=False)
        daily_b.to_csv(out / "baseline_daily.csv", index=False)
        (out / "baseline_summary.json").write_text(
            json.dumps(summary_b, indent=2, default=str), encoding="utf-8"
        )

    pulse_path = Path(args.pulse_trades)
    if not pulse_path.exists():
        raise SystemExit(f"pulse trades missing: {pulse_path}")
    pulse = pd.read_csv(pulse_path)
    pulse["date"] = pulse["date"].astype(str)
    pulse = pulse[
        (pulse["date"] >= args.start_date) & (pulse["date"] <= args.end_date)
    ].copy()
    pulse.to_csv(out / "pulse_trades.csv", index=False)
    pulse_d = _pulse_daily(pulse)

    daily_b = daily_b.copy()
    daily_b["date"] = daily_b["date"].astype(str)
    # Union of calendar days in either book
    all_dates = sorted(
        set(daily_b["date"].tolist()) | set(pulse_d["date"].tolist())
    )
    base_map = daily_b.set_index("date")
    pulse_map = pulse_d.set_index("date") if len(pulse_d) else pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for d in all_dates:
        b = base_map.loc[d] if d in base_map.index else None
        p = pulse_map.loc[d] if (len(pulse_map) and d in pulse_map.index) else None
        day_ret_b = float(b["day_ret"]) if b is not None else 0.0
        n_b = int(b["n"]) if b is not None and "n" in b.index else 0
        pulse_add = float(p["pulse_add"]) if p is not None else 0.0
        pulse_n = int(p["pulse_n"]) if p is not None else 0
        pulse_mean = float(p["pulse_mean"]) if p is not None else float("nan")
        day_ret_c = day_ret_b + pulse_add
        rows.append(
            {
                "date": d,
                "n_base": n_b,
                "day_ret_base": day_ret_b,
                "pulse_n": pulse_n,
                "pulse_add": pulse_add,
                "pulse_mean": pulse_mean,
                "day_ret_comb": day_ret_c,
                "base_win": bool(day_ret_b > 0) if b is not None else None,
                "pulse_win": (
                    bool(pulse_add > 0)
                    if pulse_n > 0
                    else None
                ),
                "comb_win": bool(day_ret_c > 0),
            }
        )
    daily = pd.DataFrame(rows)
    eq_b, tot_b, dd_b = _compound(daily, "day_ret_base")
    eq_c, tot_c, dd_c = _compound(daily, "day_ret_comb")
    daily["equity_base"] = eq_b
    daily["equity_comb"] = eq_c
    daily["delta_vs_base"] = daily["day_ret_comb"] - daily["day_ret_base"]

    daily.to_csv(out / "daily_overlay.csv", index=False)

    # Highlight loss / win days
    loss = daily[daily["day_ret_comb"] < 0].sort_values("day_ret_comb")
    win = daily[daily["day_ret_comb"] > 0].sort_values("day_ret_comb", ascending=False)
    loss.to_csv(out / "daily_loss_days.csv", index=False)
    win.to_csv(out / "daily_win_days.csv", index=False)

    summary = {
        "tag": args.tag,
        "start": args.start_date,
        "end": args.end_date,
        "pulse_cell": pulse_path.name,
        "baseline": {
            "n_trades": int(len(trades_b)),
            "n_days": int(daily_b["date"].nunique()) if len(daily_b) else 0,
            "total_ret": tot_b,
            "maxdd": dd_b,
            "day_win": float((daily["day_ret_base"] > 0).mean()) if len(daily) else None,
            "offline_summary": {
                k: summary_b.get(k)
                for k in ("total_ret", "maxdd", "n_trades", "trade_win", "day_win")
                if isinstance(summary_b, dict)
            },
        },
        "pulse": {
            "n_trades": int(len(pulse)),
            "n_days": int(pulse_d["date"].nunique()) if len(pulse_d) else 0,
            "add": float(pulse["pnl_frac"].sum())
            if len(pulse) and "pnl_frac" in pulse.columns
            else float((pulse["ret"] * pulse.get("size", 0.1)).sum())
            if len(pulse)
            else 0.0,
            "mean_ret": float(pulse["ret"].mean()) if len(pulse) else None,
            "trade_win": float((pulse["ret"] > 0).mean()) if len(pulse) else None,
        },
        "combined": {
            "total_ret": tot_c,
            "maxdd": dd_c,
            "day_win": float((daily["day_ret_comb"] > 0).mean()) if len(daily) else None,
            "n_loss_days": int((daily["day_ret_comb"] < 0).sum()),
            "n_win_days": int((daily["day_ret_comb"] > 0).sum()),
            "lift_total_ret": tot_c - tot_b,
            "worst_day": daily.loc[daily["day_ret_comb"].idxmin()].to_dict()
            if len(daily)
            else None,
            "best_day": daily.loc[daily["day_ret_comb"].idxmax()].to_dict()
            if len(daily)
            else None,
        },
        "note": (
            "Additive overlay: day_ret_comb = day_ret_base + pulse_add "
            "(pulse_add = sum ret×position_frac). Not a single OMS book."
        ),
        "out": str(out),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    # Console: compact daily table
    show = daily[
        [
            "date",
            "n_base",
            "day_ret_base",
            "pulse_n",
            "pulse_add",
            "day_ret_comb",
            "comb_win",
        ]
    ].copy()
    for c in ("day_ret_base", "pulse_add", "day_ret_comb"):
        show[c] = show[c].map(lambda x: f"{100*float(x):+.2f}%")
    print("\n=== daily overlay ===", flush=True)
    print(show.to_string(index=False), flush=True)
    print("\n=== summary ===", flush=True)
    print(json.dumps(summary, indent=2, default=str), flush=True)
    print(f"\nwrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

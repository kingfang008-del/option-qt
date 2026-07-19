#!/usr/bin/env python3
"""Independent stock sleeve replay (research). Does not touch options baseline.

Entry: multi-factor first top2 → long UP / short DN underlying at 1m close.
Default: position_frac=0.25, concurrent=2. Ablates exit / stable_bars.
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
from maga7.common.multifactor_rank import MultiFactorConfig
from maga7.common.signals import attach_mf_features, load_stock_month_files
from maga7.common.stock_sleeve import (
    StockSleeveConfig,
    collect_day_entries,
    replay_stock_sleeve,
)

PROFILE = (
    "maga7/CONFIG/strategy_profiles/"
    "single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1.json"
)
WINDOWS = {
    "strong": ("2026-05-01", "2026-07-17"),
    "weak": ("2026-02-01", "2026-04-30"),
    "focus": ("2026-07-07", "2026-07-10"),
}
EXIT_MODES = ["eod", "window_end", "hold_60", "hold_120"]
STABLE = [1, 5]


def _months(start: str, end: str) -> list[str]:
    return [str(p) for p in pd.period_range(start[:7], end[:7], freq="M")]


def _precompute_curves(
    stock_by: dict[str, pd.DataFrame], symbols: list[str]
) -> dict[str, dict[str, dict[str, float]]]:
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


def _med_map(
    curves: dict[str, dict[str, dict[str, float]]],
    *,
    symbols: list[str],
    dates: list[str],
    lookback: int,
) -> dict[tuple[str, str], dict[str, float]]:
    out: dict[tuple[str, str], dict[str, float]] = {}
    for date in dates:
        for sym in symbols:
            prev = sorted(d for d in curves.get(sym, {}) if d < date)[-int(lookback) :]
            by_tod: dict[str, list[float]] = {}
            for d in prev:
                for tod, v in curves[sym][d].items():
                    by_tod.setdefault(tod, []).append(float(v))
            out[(sym, str(date))] = {
                tod: float(np.median(vs)) for tod, vs in by_tod.items() if vs
            }
    return out


def _trading_dates(
    stock_by: dict[str, pd.DataFrame], symbols: list[str], start: str, end: str
) -> list[str]:
    dates: set[str] = set()
    for sym in symbols:
        sdf = stock_by.get(sym)
        if sdf is None:
            continue
        for d in sdf["date"].astype(str).unique():
            if start <= d <= end:
                dates.add(str(d))
    return sorted(dates)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", default=PROFILE)
    ap.add_argument("--position-frac", type=float, default=0.25)
    ap.add_argument("--concurrent", type=int, default=4)
    ap.add_argument("--max-up", type=int, default=2)
    ap.add_argument("--max-dn", type=int, default=2)
    ap.add_argument(
        "--no-displace",
        action="store_true",
        help="disable causal score displace when side is full",
    )
    ap.add_argument("--fp-gate", type=float, default=0.005)
    ap.add_argument("--cost-bps", type=float, default=1.0)
    ap.add_argument("--step-minutes", type=int, default=1)
    ap.add_argument(
        "--exits",
        default=",".join(EXIT_MODES),
        help="comma list of exit modes to ablate",
    )
    ap.add_argument(
        "--stables",
        default=",".join(str(x) for x in STABLE),
        help="comma list of stable_bars",
    )
    ap.add_argument("--out", default="maga7/results/stock_sleeve_mf_top2")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    prof = load_profile(args.profile)
    symbols = list(prof["symbols"])
    stock_root = Path(os.path.expanduser(prof["paths"]["stock_root"]))
    sig = prof.get("signal") or {}
    win_start = str(sig.get("window_start") or "10:30")
    win_end = str(sig.get("window_end") or "14:00")

    all_start = min(w[0] for w in WINDOWS.values())
    all_end = max(w[1] for w in WINDOWS.values())
    lb_start = (pd.Timestamp(all_start) - pd.Timedelta(days=40)).strftime("%Y-%m-%d")
    load_months = _months(lb_start, all_end)

    print("loading stock months...", load_months[0], "..", load_months[-1])
    stock_by: dict[str, pd.DataFrame] = {}
    for sym in symbols + ["QQQ"]:
        raw = load_stock_month_files(stock_root, sym, load_months)
        if raw is None or getattr(raw, "empty", True):
            print(f"  skip {sym}: no data")
            continue
        stock_by[sym] = attach_mf_features(raw)
        print(f"  {sym}: {len(stock_by[sym])} bars")

    mf_cfg = MultiFactorConfig(
        fp_gate=float(args.fp_gate),
        window_start=win_start,
        window_end=win_end,
    )
    print("precomputing cum$ curves...")
    curves = _precompute_curves(stock_by, symbols)

    exits = [x.strip() for x in str(args.exits).split(",") if x.strip()]
    stables = [int(x.strip()) for x in str(args.stables).split(",") if x.strip()]

    scoreboard = []
    focus_dump: dict[str, list] = {}

    # Precompute medians + entries once per (stable, window); ablate exits cheaply.
    cache_med: dict[str, dict] = {}
    cache_entries: dict[tuple[int, str], dict] = {}
    for wname, (start, end) in WINDOWS.items():
        dates = _trading_dates(stock_by, symbols, start, end)
        print(f"medians {wname} ({len(dates)} days)...")
        cache_med[wname] = {
            "dates": dates,
            "med": _med_map(
                curves, symbols=symbols, dates=dates, lookback=mf_cfg.lookback_days
            ),
        }

    for stable in stables:
        for wname, blob in cache_med.items():
            sleeve0 = StockSleeveConfig(
                position_frac=float(args.position_frac),
                concurrent=int(args.concurrent),
                max_trades_per_day=int(args.concurrent),
                max_up=int(args.max_up),
                max_dn=int(args.max_dn),
                displace=not bool(args.no_displace),
                window_start=win_start,
                window_end=win_end,
                exit_mode="eod",
                stable_bars=int(stable),
                step_minutes=int(args.step_minutes),
                cost_bps=float(args.cost_bps),
            )
            print(f"entries stable={stable} {wname}...")
            cache_entries[(stable, wname)] = collect_day_entries(
                stock_by,
                symbols=symbols,
                dates=blob["dates"],
                mf_cfg=mf_cfg,
                sleeve_cfg=sleeve0,
                tod_median_by_sym_date=blob["med"],
            )

    for exit_mode in exits:
        for stable in stables:
            tag = f"{exit_mode}_s{stable}"
            for wname, blob in cache_med.items():
                dates = blob["dates"]
                med = blob["med"]
                sleeve = StockSleeveConfig(
                    position_frac=float(args.position_frac),
                    concurrent=int(args.concurrent),
                    max_trades_per_day=int(args.concurrent),
                    max_up=int(args.max_up),
                    max_dn=int(args.max_dn),
                    displace=not bool(args.no_displace),
                    window_start=win_start,
                    window_end=win_end,
                    exit_mode=exit_mode,  # type: ignore[arg-type]
                    stable_bars=int(stable),
                    step_minutes=int(args.step_minutes),
                    cost_bps=float(args.cost_bps),
                )
                print(f"run {tag} {wname} ({len(dates)} days)...")
                res = replay_stock_sleeve(
                    stock_by,
                    symbols=symbols,
                    dates=dates,
                    mf_cfg=mf_cfg,
                    sleeve_cfg=sleeve,
                    tod_median_by_sym_date=med,
                    entries_by_date=cache_entries[(stable, wname)],
                )
                s = res["summary"]
                row = {
                    "variant": tag,
                    "exit_mode": exit_mode,
                    "stable_bars": stable,
                    "window": wname,
                    "total_ret": s["total_ret"],
                    "maxdd": s["maxdd"],
                    "n_trades": s["n_trades"],
                    "n_trade_days": s["n_trade_days"],
                    "win_rate": s["win_rate"],
                    "avg_ret": s["avg_ret"],
                }
                scoreboard.append(row)
                print(
                    f"  ret={s['total_ret']:+.2%} dd={s['maxdd']:.2%} "
                    f"n={s['n_trades']} wr={s['win_rate']}"
                )
                sub = out / tag / wname
                sub.mkdir(parents=True, exist_ok=True)
                res["trades"].to_csv(sub / "trades.csv", index=False)
                res["daily"].to_csv(sub / "daily.csv", index=False)
                (sub / "summary.json").write_text(json.dumps(s, indent=2) + "\n")

                if wname == "focus" and not res["trades"].empty:
                    t = res["trades"]
                    cols = [
                        c
                        for c in [
                            "date",
                            "symbol",
                            "dir",
                            "entry_tod",
                            "fp",
                            "ret",
                            "raw_stock_ret",
                            "exit_reason",
                        ]
                        if c in t.columns
                    ]
                    focus_dump[tag] = t[cols].to_dict(orient="records")

    sb = pd.DataFrame(scoreboard)
    sb.to_csv(out / "scoreboard.csv", index=False)

    # pick winner: maximize strong total_ret among variants with weak not catastrophic
    # (report only; no promotion)
    pivot = sb.pivot_table(
        index="variant", columns="window", values="total_ret", aggfunc="first"
    )
    pick = None
    if "strong" in pivot.columns:
        cand = pivot.sort_values("strong", ascending=False)
        pick = str(cand.index[0])
        # prefer also looking at focus META/NVDA if present
    summary = {
        "role": "research_candidate",
        "note": "Independent stock sleeve; options baseline untouched.",
        "position_frac": float(args.position_frac),
        "concurrent": int(args.concurrent),
        "max_up": int(args.max_up),
        "max_dn": int(args.max_dn),
        "fp_gate": float(args.fp_gate),
        "cost_bps": float(args.cost_bps),
        "suggested_by_strong_ret": pick,
        "focus_trades": focus_dump,
        "scoreboard": scoreboard,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print("\nscoreboard:")
    print(sb.to_string(index=False))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

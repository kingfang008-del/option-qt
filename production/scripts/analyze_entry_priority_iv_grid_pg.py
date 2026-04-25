#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Grid search for production entry-priority IV penalty.

This replay keeps the production ranking formula intact:

    score = |alpha|^alpha_power / iv^iv_penalty_power
            * alpha/momentum/MACD/priority/trend multipliers

Only ENTRY_RANK_IV_PENALTY_POWER is changed. The default comparison baseline
is 0.5 (sqrt_iv), because it is the conservative candidate from the pure-alpha
ranking study.
"""

from __future__ import annotations

import argparse
import copy
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import psycopg2  # type: ignore

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from analyze_alpha_priority_pg import (  # noqa: E402
    analyze_candidate_profits,
    analyze_rankings,
    load_alpha_rows,
    load_option_paths,
)

sys.path.insert(0, str(SCRIPT_DIR.parent / "baseline"))
from config import PG_DB_URL  # noqa: E402
from strategy_selector import StrategyConfig  # noqa: E402


@dataclass
class DayResult:
    date: str
    power: float
    selected: int
    samples: int
    avg_max: float
    avg_final: float
    hit10: float
    hit20: float
    pos_final: float


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="生产完整公式 IV penalty grid")
    p.add_argument(
        "--dates",
        required=True,
        help="逗号分隔, 如 2026-03-02,2026-03-03,...",
    )
    p.add_argument("--powers", default="0,0.25,0.5,0.75,1.0,1.25")
    p.add_argument("--baseline-power", type=float, default=0.5)
    p.add_argument("--selected-per-frame", type=int, default=3)
    p.add_argument("--horizon-mins", type=int, default=30)
    p.add_argument("--cooldown-mins", type=int, default=15)
    p.add_argument("--focus-symbols", default="NVDA,MSFT,ADBE,COIN,MSTR,INTC")
    return p.parse_args()


def _fmt_pct(v: float | None) -> str:
    if v is None:
        return "n/a"
    return f"{v * 100:.2f}%"


def _fmt_signed_pct(v: float | None) -> str:
    if v is None:
        return "n/a"
    return f"{v * 100:+.2f}%"


def _stats(vals: list[float]) -> tuple[float, float]:
    if not vals:
        return 0.0, 0.0
    if len(vals) < 2:
        return vals[0], 0.0
    return statistics.mean(vals), statistics.stdev(vals)


def _cfg_with_power(base_cfg: Any, power: float) -> Any:
    cfg = copy.copy(base_cfg)
    setattr(cfg, "ENTRY_RANK_IV_PENALTY_POWER", float(power))
    return cfg


def run_day(date_str: str, powers: list[float], args: argparse.Namespace) -> list[DayResult]:
    conn = psycopg2.connect(PG_DB_URL)
    try:
        alpha_rows = load_alpha_rows(conn, date_str)
        if not alpha_rows:
            return []
        symbols_scope = sorted({str(r["symbol"]) for r in alpha_rows})
        min_ts = min(int(float(r["ts"])) for r in alpha_rows)
        max_ts = max(int(float(r["ts"])) for r in alpha_rows)
        option_paths = load_option_paths(
            conn,
            min_ts - 600,
            max_ts + args.horizon_mins * 60 + 600,
            symbols_scope,
        )
    finally:
        conn.close()

    base_cfg = StrategyConfig()
    out: list[DayResult] = []
    for power in powers:
        cfg = _cfg_with_power(base_cfg, power)
        _, selected_rows = analyze_rankings(
            alpha_rows,
            cfg,
            selected_per_frame=args.selected_per_frame,
        )
        profits = analyze_candidate_profits(
            selected_rows,
            option_paths,
            horizon_mins=args.horizon_mins,
            cooldown_mins=args.cooldown_mins,
        )
        if profits:
            max_rois = [p.max_roi for p in profits]
            final_rois = [p.final_roi for p in profits]
            avg_max = sum(max_rois) / len(max_rois)
            avg_final = sum(final_rois) / len(final_rois)
            hit10 = sum(1 for x in max_rois if x >= 0.10) / len(max_rois)
            hit20 = sum(1 for x in max_rois if x >= 0.20) / len(max_rois)
            pos_final = sum(1 for x in final_rois if x > 0) / len(final_rois)
        else:
            avg_max = avg_final = hit10 = hit20 = pos_final = 0.0
        out.append(
            DayResult(
                date=date_str,
                power=float(power),
                selected=len(selected_rows),
                samples=len(profits),
                avg_max=avg_max,
                avg_final=avg_final,
                hit10=hit10,
                hit20=hit20,
                pos_final=pos_final,
            )
        )
    return out


def _print_daily_matrix(results: list[DayResult], dates: list[str], powers: list[float]) -> None:
    by_key = {(r.date, r.power): r for r in results}
    print("\n## 表 1: 每日 avgMax")
    header = f"{'iv_power':>8} " + " ".join(f"{d[5:]:>8}" for d in dates) + "    mean±std"
    print(header)
    for power in powers:
        vals = [by_key[(d, power)].avg_max for d in dates if (d, power) in by_key]
        mean, std = _stats(vals)
        cells = " ".join(f"{_fmt_pct(by_key[(d, power)].avg_max):>8}" for d in dates if (d, power) in by_key)
        print(f"{power:>8.2f} {cells}    {_fmt_pct(mean)}±{_fmt_pct(std)}")


def _print_summary(results: list[DayResult], dates: list[str], powers: list[float], baseline_power: float) -> None:
    by_power: dict[float, list[DayResult]] = {p: [] for p in powers}
    by_key = {(r.date, r.power): r for r in results}
    for r in results:
        by_power.setdefault(r.power, []).append(r)

    baseline_by_day = {
        d: by_key[(d, baseline_power)]
        for d in dates
        if (d, baseline_power) in by_key
    }
    current_by_day = {
        d: by_key[(d, 0.0)]
        for d in dates
        if (d, 0.0) in by_key
    }

    print(f"\n## 表 2: 汇总 (sqrt_iv baseline = power {baseline_power:.2f})")
    print(
        f"{'iv_power':>8} {'avgMax':>9} {'std':>8} {'avgFinal':>9} "
        f"{'hit10':>8} {'hit20':>8} {'samples':>8} "
        f"{'win_vs_sqrt':>12} {'lift_vs_sqrt':>13} {'lift_vs_0':>10}"
    )
    rows = []
    for power in powers:
        rs = by_power.get(power, [])
        avg_max_vals = [r.avg_max for r in rs]
        avg_final_vals = [r.avg_final for r in rs]
        hit10_vals = [r.hit10 for r in rs]
        hit20_vals = [r.hit20 for r in rs]
        samples_vals = [float(r.samples) for r in rs]
        avg_max, std = _stats(avg_max_vals)
        avg_final, _ = _stats(avg_final_vals)
        hit10, _ = _stats(hit10_vals)
        hit20, _ = _stats(hit20_vals)
        samples, _ = _stats(samples_vals)

        wins_vs_sqrt = 0
        lifts_vs_sqrt = []
        lifts_vs_0 = []
        for d in dates:
            cur = by_key.get((d, power))
            sqrt = baseline_by_day.get(d)
            zero = current_by_day.get(d)
            if cur and sqrt:
                if cur.avg_max > sqrt.avg_max + 1e-12:
                    wins_vs_sqrt += 1
                lifts_vs_sqrt.append(cur.avg_max - sqrt.avg_max)
            if cur and zero:
                lifts_vs_0.append(cur.avg_max - zero.avg_max)
        lift_vs_sqrt, _ = _stats(lifts_vs_sqrt)
        lift_vs_0, _ = _stats(lifts_vs_0)
        rows.append((power, avg_max, std, avg_final, hit10, hit20, samples, wins_vs_sqrt, lift_vs_sqrt, lift_vs_0))

    rows.sort(key=lambda x: (x[1] + 0.3 * x[4] + 0.5 * x[3] - 0.25 * x[2]), reverse=True)
    for power, avg_max, std, avg_final, hit10, hit20, samples, wins, lift_sqrt, lift_0 in rows:
        print(
            f"{power:>8.2f} {_fmt_pct(avg_max):>9} {_fmt_pct(std):>8} {_fmt_pct(avg_final):>9} "
            f"{_fmt_pct(hit10):>8} {_fmt_pct(hit20):>8} {samples:>8.1f} "
            f"{wins}/{len(dates):<9} {_fmt_signed_pct(lift_sqrt):>13} {_fmt_signed_pct(lift_0):>10}"
        )


def _print_split_summary(results: list[DayResult], powers: list[float], label: str, date_prefix: str) -> None:
    subset = [r for r in results if r.date.startswith(date_prefix)]
    if not subset:
        return
    print(f"\n## {label} 分组摘要")
    print(f"{'iv_power':>8} {'days':>5} {'avgMax':>9} {'avgFinal':>9} {'hit10':>8}")
    for power in powers:
        rs = [r for r in subset if r.power == power]
        if not rs:
            continue
        avg_max, _ = _stats([r.avg_max for r in rs])
        avg_final, _ = _stats([r.avg_final for r in rs])
        hit10, _ = _stats([r.hit10 for r in rs])
        print(f"{power:>8.2f} {len(rs):>5} {_fmt_pct(avg_max):>9} {_fmt_pct(avg_final):>9} {_fmt_pct(hit10):>8}")


def main() -> None:
    args = parse_args()
    dates = [d.strip() for d in args.dates.split(",") if d.strip()]
    powers = [float(x.strip()) for x in args.powers.split(",") if x.strip()]
    if args.baseline_power not in powers:
        powers.append(float(args.baseline_power))
        powers = sorted(set(powers))

    print(f"日期: {dates}")
    print(
        f"参数: powers={powers} baseline_power={args.baseline_power} "
        f"selected_per_frame={args.selected_per_frame} horizon={args.horizon_mins}m "
        f"cooldown={args.cooldown_mins}m"
    )
    print("说明: 完整生产公式 replay, 只改变 ENTRY_RANK_IV_PENALTY_POWER。")

    all_results: list[DayResult] = []
    for d in dates:
        print(f"  跑 {d} ...", flush=True)
        day_results = run_day(d, powers, args)
        all_results.extend(day_results)
        if day_results:
            best = max(day_results, key=lambda r: r.avg_max)
            print(f"    best power={best.power:.2f} avgMax={_fmt_pct(best.avg_max)} samples={best.samples}")

    valid_dates = sorted({r.date for r in all_results})
    if not all_results:
        print("没有可用结果。")
        return

    _print_daily_matrix(all_results, valid_dates, powers)
    _print_summary(all_results, valid_dates, powers, args.baseline_power)
    _print_split_summary(all_results, powers, "3 月", "2026-03")
    _print_split_summary(all_results, powers, "4 月", "2026-04")


if __name__ == "__main__":
    main()

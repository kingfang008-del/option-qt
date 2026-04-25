#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""跨多个交易日跑 analyze_alpha_algos_pg 中的所有算法, 汇总均值/标准差,
判断 D3/F2 是真正的稳定优胜者还是单日噪声.

输出三张表:
  1. 每算法每天的 avgMax (横向 = 日期)
  2. 跨日均值: avgMax, hit10, hit20, samples + 标准差
  3. 击败基线天数: A0_raw_abs 作基线, 看每个算法在多少天 avgMax 严格更高
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import psycopg2  # type: ignore

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from analyze_alpha_algos_pg import ALGOS, RunResult, replay  # noqa: E402
from analyze_alpha_priority_pg import (  # noqa: E402
    load_alpha_rows,
    load_option_paths,
)

sys.path.insert(0, str(SCRIPT_DIR.parent / "baseline"))
from config import PG_DB_URL  # noqa: E402


def _fmt_pct(v: float) -> str:
    return f"{v * 100:.1f}%"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="跨多日 alpha 排序算法稳定性测试")
    p.add_argument("--dates", required=True, help="逗号分隔, 例如 2026-04-17,2026-04-20,2026-04-21,2026-04-22,2026-04-23,2026-04-24")
    p.add_argument("--focus-symbols", default="NVDA,MSFT,ADBE,COIN,MSTR,INTC")
    p.add_argument("--selected-per-frame", type=int, default=3)
    p.add_argument("--candidate-horizon-mins", type=int, default=30)
    p.add_argument("--candidate-cooldown-mins", type=int, default=15)
    p.add_argument("--alpha-floor", type=float, default=0.6)
    return p.parse_args()


def run_day(date_str: str, args: argparse.Namespace, focus_symbols: list[str]) -> dict[str, RunResult]:
    conn = psycopg2.connect(PG_DB_URL)
    try:
        alpha_rows = load_alpha_rows(conn, date_str)
        if not alpha_rows:
            return {}
        symbols_scope = sorted({str(r["symbol"]) for r in alpha_rows})
        min_ts = min(int(float(r["ts"])) for r in alpha_rows)
        max_ts = max(int(float(r["ts"])) for r in alpha_rows)
        option_paths = load_option_paths(
            conn, min_ts - 600, max_ts + args.candidate_horizon_mins * 60 + 600,
            focus_symbols + symbols_scope,
        )
    finally:
        conn.close()

    common = dict(
        selected_per_frame=args.selected_per_frame,
        alpha_floor=args.alpha_floor,
        option_paths=option_paths,
        horizon_mins=args.candidate_horizon_mins,
        cooldown_mins=args.candidate_cooldown_mins,
        focus_symbols=focus_symbols,
    )
    out: dict[str, RunResult] = {}
    for family, label, fn in ALGOS:
        out[label] = replay(alpha_rows, family, label, fn, **common)
    return out


def _stats(vals: list[float]) -> tuple[float, float]:
    if not vals:
        return 0.0, 0.0
    mean = sum(vals) / len(vals)
    if len(vals) < 2:
        return mean, 0.0
    return mean, statistics.stdev(vals)


def main() -> None:
    args = parse_args()
    dates = [d.strip() for d in args.dates.split(",") if d.strip()]
    focus_symbols = [s.strip().upper() for s in args.focus_symbols.split(",") if s.strip()]

    print(f"日期: {dates}")
    print(f"参数: alpha_floor={args.alpha_floor} selected_per_frame={args.selected_per_frame} "
          f"horizon={args.candidate_horizon_mins}m cooldown={args.candidate_cooldown_mins}m")

    by_day: dict[str, dict[str, RunResult]] = {}
    for d in dates:
        print(f"  跑 {d} ...", flush=True)
        results = run_day(d, args, focus_symbols)
        if not results:
            print(f"    {d} 无数据, 跳过")
            continue
        by_day[d] = results

    if not by_day:
        print("没有可用日期数据.")
        return

    labels = [lbl for _, lbl, _ in ALGOS]
    valid_dates = sorted(by_day.keys())

    # ===== 表 1: 每算法每天 avgMax =====
    print(f"\n## 表 1: 每算法 avgMax (按日期)")
    header = f"{'label':<22} " + " ".join(f"{d[5:]:>8}" for d in valid_dates) + "    mean±std"
    print(header)
    rows_by_label: dict[str, list[float]] = {}
    for lbl in labels:
        vals = [by_day[d][lbl].avg_max for d in valid_dates]
        rows_by_label[lbl] = vals
        mean, std = _stats(vals)
        cells = " ".join(f"{_fmt_pct(v):>8}" for v in vals)
        print(f"{lbl:<22} {cells}    {_fmt_pct(mean)}±{_fmt_pct(std)}")

    # ===== 表 2: 跨日均值汇总 =====
    print(f"\n## 表 2: 跨日均值 (n={len(valid_dates)} days)")
    header2 = (
        f"{'label':<22} {'avgMax_mean':>12} {'avgMax_std':>11} "
        f"{'hit10_mean':>11} {'hit20_mean':>11} {'avgFin_mean':>12} "
        f"{'samples_mean':>13}"
    )
    print(header2)

    summaries = []
    for lbl in labels:
        days = [by_day[d][lbl] for d in valid_dates]
        avg_max_mean, avg_max_std = _stats([r.avg_max for r in days])
        hit10_mean, _ = _stats([r.hit10 for r in days])
        hit20_mean, _ = _stats([r.hit20 for r in days])
        fin_mean, _ = _stats([r.avg_final for r in days])
        sam_mean = sum(r.samples for r in days) / max(1, len(days))
        summaries.append((lbl, avg_max_mean, avg_max_std, hit10_mean, hit20_mean, fin_mean, sam_mean))

    summaries_sorted = sorted(summaries, key=lambda x: x[1] + 0.5 * x[5] + 0.3 * x[4], reverse=True)
    for lbl, mx, mx_std, h10, h20, fn, sam in summaries_sorted:
        print(
            f"{lbl:<22} {_fmt_pct(mx):>12} {_fmt_pct(mx_std):>11} "
            f"{_fmt_pct(h10):>11} {_fmt_pct(h20):>11} {_fmt_pct(fn):>12} "
            f"{sam:>13.1f}"
        )

    # ===== 表 3: 击败基线 A0_raw_abs 的天数 =====
    print(f"\n## 表 3: 击败基线 A0_raw_abs 的天数 (avgMax 严格更高)")
    print(f"{'label':<22} {'win/total':>10} {'win_rate':>9} {'avg_uplift':>11}")
    for lbl in labels:
        if lbl == "A0_raw_abs":
            continue
        wins = 0
        uplifts = []
        for d in valid_dates:
            base = by_day[d]["A0_raw_abs"].avg_max
            this = by_day[d][lbl].avg_max
            if this > base + 1e-9:
                wins += 1
            uplifts.append(this - base)
        win_rate = wins / max(1, len(valid_dates))
        avg_uplift, _ = _stats(uplifts)
        print(f"{lbl:<22} {wins}/{len(valid_dates):<10} {_fmt_pct(win_rate):>9} {_fmt_pct(avg_uplift):>11}")

    # ===== 推荐 =====
    by_label_score = {
        lbl: mx + 0.5 * fn + 0.3 * h20 - 0.5 * mx_std
        for lbl, mx, mx_std, h10, h20, fn, sam in summaries
    }
    best_label = max(by_label_score, key=lambda k: by_label_score[k])
    base_mean, _ = _stats([by_day[d]["A0_raw_abs"].avg_max for d in valid_dates])
    best_mean, _ = _stats([by_day[d][best_label].avg_max for d in valid_dates])
    print(
        f"\n>>> 跨 {len(valid_dates)} 天综合最优 (含 std 惩罚): {best_label}\n"
        f"    avgMax_mean={_fmt_pct(best_mean)} vs 基线 {_fmt_pct(base_mean)} "
        f"({(best_mean - base_mean) * 100:+.2f} pct points)"
    )


if __name__ == "__main__":
    main()

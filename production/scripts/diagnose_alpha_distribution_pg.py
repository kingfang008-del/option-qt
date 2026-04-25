#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""诊断 3 月 (回测产出) 与 4 月 (实盘产出) alpha_logs 的统计指纹差异.

回答: alpha 分布是否一致? IV 是否一致? cross-section 归一化行为是否一致?

如果 3 月与 4 月 alpha 来自不同计算路径, 之前所有跨数据集结论都需要打折.
"""
from __future__ import annotations

import argparse
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import psycopg2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "baseline"))
from config import PG_DB_URL  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--march-dates", default="2026-03-02,2026-03-03,2026-03-04,2026-03-05,2026-03-06,2026-03-09")
    p.add_argument("--april-dates", default="2026-04-17,2026-04-20,2026-04-21,2026-04-22,2026-04-23,2026-04-24")
    return p.parse_args()


def _quantiles(vals: list[float], qs: Sequence[float]) -> list[float]:
    if not vals:
        return [0.0] * len(qs)
    s = sorted(vals)
    out = []
    for q in qs:
        idx = max(0, min(len(s) - 1, int(len(s) * q)))
        out.append(s[idx])
    return out


def fetch_rows(conn, dates: list[str]) -> list[dict]:
    """逐日 fetch (LIKE 命中 datetime_ny btree index 较稳), 避免一次性扫太多."""
    out: list[dict] = []
    with conn.cursor() as cur:
        for d in dates:
            cur.execute(
                """
                SELECT ts, datetime_ny, symbol, alpha, iv, price, vol_z
                FROM alpha_logs
                WHERE datetime_ny >= %s AND datetime_ny < %s
                """,
                (f"{d} 00:00:00", f"{d} 23:59:59"),
            )
            cols = [c.name for c in cur.description]
            for r in cur.fetchall():
                out.append(dict(zip(cols, r)))
            print(f"    {d}: 累计 {len(out)} 行", flush=True)
    return out


def basic_stats(label: str, rows: list[dict]) -> None:
    alphas = [float(r["alpha"]) for r in rows]
    abs_alphas = [abs(a) for a in alphas]
    ivs = [float(r["iv"]) for r in rows if r["iv"] is not None]
    prices = [float(r["price"]) for r in rows if r["price"]]
    vol_zs = [float(r["vol_z"]) for r in rows if r["vol_z"] is not None]

    p_alpha = _quantiles(abs_alphas, [0.5, 0.75, 0.9, 0.95, 0.99])
    p_iv = _quantiles(ivs, [0.5, 0.9, 0.99])
    print(f"\n========= {label}  rows={len(rows)} =========")
    print(f"  alpha:        mean={statistics.mean(alphas):+.3f}  std={statistics.pstdev(alphas):.3f}")
    print(f"  |alpha|:      mean={statistics.mean(abs_alphas):.3f}  std={statistics.pstdev(abs_alphas):.3f}")
    print(f"  |alpha| 分位: p50={p_alpha[0]:.3f} p75={p_alpha[1]:.3f} p90={p_alpha[2]:.3f} p95={p_alpha[3]:.3f} p99={p_alpha[4]:.3f}")
    n = len(abs_alphas)
    for thr in (0.6, 1.0, 1.5, 2.0, 3.0):
        cnt = sum(1 for a in abs_alphas if a >= thr)
        print(f"    |alpha|>={thr}:   {cnt:>6}  ({cnt/n*100:.2f}%)")
    print(f"  iv:           mean={statistics.mean(ivs):.3f}  median={p_iv[0]:.3f}  p90={p_iv[1]:.3f}  p99={p_iv[2]:.3f}")
    if vol_zs:
        print(f"  vol_z:        mean={statistics.mean(vol_zs):+.3f}  std={statistics.pstdev(vol_zs):.3f}")


def cross_section_check(label: str, rows: list[dict]) -> None:
    """每分钟 28 个 symbols 的 alpha mean/std: 应该是 ~0/~1 才是真正的 cross-section z-score."""
    by_min: dict[int, list[float]] = defaultdict(list)
    for r in rows:
        by_min[int(float(r["ts"]))].append(float(r["alpha"]))

    means, stds, counts = [], [], []
    for ts, alphas in by_min.items():
        if len(alphas) < 5:
            continue
        means.append(statistics.mean(alphas))
        stds.append(statistics.pstdev(alphas))
        counts.append(len(alphas))

    if not means:
        print(f"  [{label}] 无足够 cross-section 数据")
        return
    print(f"\n  ── 每分钟 cross-section 统计 ({label}) ──")
    print(f"  分钟数 = {len(means)}")
    print(f"  alpha mean (per minute) → mean={statistics.mean(means):+.4f}  std={statistics.pstdev(means):.4f}")
    print(f"  alpha std  (per minute) → mean={statistics.mean(stds):.4f}  std={statistics.pstdev(stds):.4f}")
    print(f"  symbols/分钟 → mean={statistics.mean(counts):.1f}  min={min(counts)}  max={max(counts)}")
    p_mean = _quantiles(means, [0.05, 0.5, 0.95])
    p_std = _quantiles(stds, [0.05, 0.5, 0.95])
    print(f"  per-minute mean p05/p50/p95 = {p_mean[0]:+.3f}/{p_mean[1]:+.3f}/{p_mean[2]:+.3f}")
    print(f"  per-minute std  p05/p50/p95 = {p_std[0]:.3f}/{p_std[1]:.3f}/{p_std[2]:.3f}")
    print(f"  → 若是标准 cross-section z-score, 期望 mean≈0, std≈1")


def per_symbol_compare(march_rows: list[dict], april_rows: list[dict],
                        focus: list[str]) -> None:
    print(f"\n========= 同标的 |alpha| 分布对照 =========")
    print(f"{'symbol':<6} {'period':<6} {'n':>5} {'mean':>7} {'p50':>7} {'p90':>7} {'p99':>7} {'>=0.6%':>7} {'>=1.5%':>7}")
    by_period_sym: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in march_rows:
        by_period_sym[("Mar", str(r["symbol"]))].append(abs(float(r["alpha"])))
    for r in april_rows:
        by_period_sym[("Apr", str(r["symbol"]))].append(abs(float(r["alpha"])))

    for sym in focus:
        for period in ("Mar", "Apr"):
            vals = by_period_sym.get((period, sym), [])
            if not vals:
                print(f"{sym:<6} {period:<6} {0:>5} {'n/a':>7}")
                continue
            n = len(vals)
            p = _quantiles(vals, [0.5, 0.9, 0.99])
            ge_06 = sum(1 for v in vals if v >= 0.6) / n
            ge_15 = sum(1 for v in vals if v >= 1.5) / n
            print(
                f"{sym:<6} {period:<6} {n:>5} {statistics.mean(vals):>7.3f} "
                f"{p[0]:>7.3f} {p[1]:>7.3f} {p[2]:>7.3f} {ge_06*100:>6.1f}% {ge_15*100:>6.1f}%"
            )


def time_window_check(label: str, rows: list[dict]) -> None:
    """按交易时段分桶, 看 alpha 强度是否随时间衰减/集中."""
    by_hour: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        hh = str(r["datetime_ny"])[11:13]
        by_hour[hh].append(abs(float(r["alpha"])))
    print(f"\n  ── 按小时 |alpha| 强度 ({label}) ──")
    print(f"  {'hour':<5} {'n':>6} {'mean':>7} {'p90':>7} {'>=0.6%':>7}")
    for hh in sorted(by_hour.keys()):
        vals = by_hour[hh]
        if len(vals) < 10:
            continue
        p = _quantiles(vals, [0.9])
        print(f"  {hh:<5} {len(vals):>6} {statistics.mean(vals):>7.3f} {p[0]:>7.3f} "
              f"{sum(1 for v in vals if v >= 0.6)/len(vals)*100:>6.1f}%")


def main() -> None:
    args = parse_args()
    march_dates = [d.strip() for d in args.march_dates.split(",") if d.strip()]
    april_dates = [d.strip() for d in args.april_dates.split(",") if d.strip()]

    conn = psycopg2.connect(PG_DB_URL)
    try:
        print(f"载入 3 月: {march_dates}")
        march_rows = fetch_rows(conn, march_dates)
        print(f"  rows={len(march_rows)}")
        print(f"载入 4 月: {april_dates}")
        april_rows = fetch_rows(conn, april_dates)
        print(f"  rows={len(april_rows)}")
    finally:
        conn.close()

    basic_stats("3 月 (回测产出)", march_rows)
    basic_stats("4 月 (实盘产出)", april_rows)

    cross_section_check("3 月", march_rows)
    cross_section_check("4 月", april_rows)

    per_symbol_compare(march_rows, april_rows,
                       focus=["NVDA", "MSFT", "ADBE", "TSLA", "AAPL", "QQQ", "INTC", "COIN"])

    time_window_check("3 月", march_rows)
    time_window_check("4 月", april_rows)

    print(f"\n========= 关键诊断结论提示 =========")
    march_abs = [abs(float(r["alpha"])) for r in march_rows]
    april_abs = [abs(float(r["alpha"])) for r in april_rows]
    march_mean = statistics.mean(march_abs)
    april_mean = statistics.mean(april_abs)
    print(f"  |alpha| mean: 3月={march_mean:.3f}  4月={april_mean:.3f}  ratio={march_mean/april_mean:.2f}")
    march_p99 = _quantiles(march_abs, [0.99])[0]
    april_p99 = _quantiles(april_abs, [0.99])[0]
    print(f"  |alpha| p99 : 3月={march_p99:.3f}  4月={april_p99:.3f}  ratio={march_p99/april_p99:.2f}")

    march_thr = sum(1 for v in march_abs if v >= 0.6) / len(march_abs)
    april_thr = sum(1 for v in april_abs if v >= 0.6) / len(april_abs)
    print(f"  |alpha|>=0.6 占比: 3月={march_thr*100:.2f}%  4月={april_thr*100:.2f}%  ratio={march_thr/april_thr:.2f}")
    print(f"\n  → 若 ratio 远离 1.0, 说明 3 月与 4 月 alpha 来自 *不同的计算路径*,")
    print(f"    进而之前所有「跨数据集对比」都需要重新审视: 不是同一个分布!")


if __name__ == "__main__":
    main()

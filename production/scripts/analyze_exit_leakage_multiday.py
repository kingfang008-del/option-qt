#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""跨多日平仓利润漏出 (exit leakage) 分析.

复用 analyze_profit_retention_pg 的订单配对/option mid 重建底座, 但聚焦三件事:

  1. 全局漏出: realized_roi vs hold_peak_roi 的差距, 即 "持仓期间能赚最多的 ROI"
     和 "实际平仓 ROI" 的 gap, 这个 gap 就是所谓 "回吐空间".
  2. 按 exit_reason 分组: 哪类平仓原因丢钱最多?
  3. 按持仓时长分桶: 不同 hold_mins 段的实际/峰值 ROI, 帮助定位最优持仓时长.

输出后接一句 "如果实施 X 改动, 大约能多赚 Y%", 用于决策.
"""
from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Iterable

import psycopg2

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from analyze_profit_retention_pg import (  # noqa: E402
    TradeAnalysis,
    _minute_floor,
    _safe_float,
    build_trade_analyses,
    load_option_paths,
    load_order_events,
    load_stock_paths,
)

sys.path.insert(0, str(SCRIPT_DIR.parent / "baseline"))
from config import PG_DB_URL  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="跨多日平仓利润漏出分析")
    p.add_argument("--dates", required=True, help="逗号分隔, 如 2026-04-17,2026-04-20,2026-04-21,2026-04-22,2026-04-23,2026-04-24")
    p.add_argument("--symbols", default="")
    p.add_argument("--future-mins", type=int, default=10)
    p.add_argument("--whipsaw-threshold", type=float, default=0.08)
    p.add_argument("--capture-threshold", type=float, default=0.60)
    p.add_argument("--top-n", type=int, default=20, help="按 leakage 排序输出 top N")
    return p.parse_args()


def _fmt_pct(v: float | None) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "n/a"
    return f"{v * 100:+.2f}%"


def _fmt_pos(v: float | None) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "n/a"
    return f"{v * 100:.2f}%"


def collect_trades(date_str: str, symbols: list[str], future_mins: int,
                   whipsaw_th: float, capture_th: float) -> list[TradeAnalysis]:
    conn = psycopg2.connect(PG_DB_URL)
    try:
        rows = load_order_events(conn, date_str, symbols)
        if not rows:
            return []
        fill_rows = [r for r in rows if r["action"] == "ORDER_FILLED"]
        if not fill_rows:
            return []
        min_ts = _minute_floor(min(_safe_float(r["ts"]) for r in fill_rows))
        max_ts = _minute_floor(max(_safe_float(r["ts"]) for r in fill_rows)) + future_mins * 60
        event_symbols = sorted({str(r["symbol"]) for r in fill_rows})
        target = symbols or event_symbols
        option_paths = load_option_paths(conn, min_ts, max_ts, target)
        stock_paths = load_stock_paths(conn, min_ts, max_ts, target)
        return build_trade_analyses(
            rows, option_paths, stock_paths,
            future_mins=future_mins,
            whipsaw_threshold=whipsaw_th,
            capture_threshold=capture_th,
        )
    finally:
        conn.close()


def _peak_roi(t: TradeAnalysis) -> float | None:
    if t.hold_peak_mid is None or t.entry_price <= 0:
        return None
    return (t.hold_peak_mid - t.entry_price) / t.entry_price


def _total_peak_roi(t: TradeAnalysis) -> float | None:
    if t.total_peak_mid is None or t.entry_price <= 0:
        return None
    return (t.total_peak_mid - t.entry_price) / t.entry_price


def _normalize_reason(raw: str) -> str:
    r = (raw or "").upper()
    for tag in ("MOMENTUM", "MACD", "FLIP", "IDX_REVERSAL", "CT_TIMEOUT",
                "TARGET", "STOP", "TIME", "TRAILING", "RUNAWAY", "RECONCIL",
                "MANUAL", "EOD", "CIRCUIT", "RUNTIME"):
        if tag in r:
            return tag
    return r.split("|")[0][:24] or "UNKNOWN"


def _bucket_hold(mins: float) -> str:
    if mins < 5:
        return "0-5m"
    if mins < 10:
        return "5-10m"
    if mins < 20:
        return "10-20m"
    if mins < 30:
        return "20-30m"
    if mins < 45:
        return "30-45m"
    if mins < 60:
        return "45-60m"
    return "60m+"


def _summary_block(trades: list[TradeAnalysis]) -> None:
    if not trades:
        print("(空)")
        return
    realized = [t.realized_roi for t in trades]
    hold_peak = [pr for pr in (_peak_roi(t) for t in trades) if pr is not None]
    total_peak = [pr for pr in (_total_peak_roi(t) for t in trades) if pr is not None]
    leakage_hold = [
        (peak - r) for r, peak in zip(realized, (_peak_roi(t) for t in trades)) if peak is not None
    ]
    leakage_total = [
        (peak - r) for r, peak in zip(realized, (_total_peak_roi(t) for t in trades)) if peak is not None
    ]
    print(f"  trades={len(trades)} | win_rate={sum(1 for x in realized if x > 0)/len(realized)*100:.1f}%")
    print(f"  realized_roi  mean={_fmt_pct(mean(realized))} median={_fmt_pct(median(realized))}")
    if hold_peak:
        print(f"  hold_peak_roi mean={_fmt_pos(mean(hold_peak))} median={_fmt_pos(median(hold_peak))}")
    if total_peak:
        print(f"  total_peak_roi mean={_fmt_pos(mean(total_peak))} median={_fmt_pos(median(total_peak))}")
    if leakage_hold:
        print(f"  → 持仓内漏出 mean={_fmt_pos(mean(leakage_hold))} (持仓最高点本可赚的, 没赚到的部分)")
    if leakage_total:
        print(f"  → 含未来漏出 mean={_fmt_pos(mean(leakage_total))} (含 +{0}m 未来观察期)")


def _by_reason(trades: list[TradeAnalysis]) -> None:
    by: dict[str, list[TradeAnalysis]] = defaultdict(list)
    for t in trades:
        by[_normalize_reason(t.exit_reason)].append(t)
    rows = []
    for reason, lst in by.items():
        n = len(lst)
        realized = [t.realized_roi for t in lst]
        peaks = [p for p in (_peak_roi(t) for t in lst) if p is not None]
        leakages = [(p - t.realized_roi) for t, p in zip(lst, (_peak_roi(t) for t in lst)) if p is not None]
        wr = sum(1 for x in realized if x > 0) / max(1, n)
        rows.append((
            reason, n, wr,
            mean(realized) if realized else 0.0,
            mean(peaks) if peaks else 0.0,
            mean(leakages) if leakages else 0.0,
            sum(t.realized_roi for t in lst),
            sum(leakages) if leakages else 0.0,
        ))
    rows.sort(key=lambda r: r[7], reverse=True)
    print(f"\n## 按 exit_reason 分组")
    print(f"{'reason':<14} {'n':>5} {'win%':>6} {'real_mean':>11} {'peak_mean':>11} {'leak_mean':>11} {'sum_real':>10} {'sum_leak':>10}")
    for reason, n, wr, rm, pm, lm, sr, sl in rows:
        print(
            f"{reason:<14} {n:>5} {wr*100:>5.1f}% {_fmt_pct(rm):>11} {_fmt_pos(pm):>11} "
            f"{_fmt_pos(lm):>11} {_fmt_pct(sr):>10} {_fmt_pos(sl):>10}"
        )


def _by_hold_bucket(trades: list[TradeAnalysis]) -> None:
    order = ["0-5m", "5-10m", "10-20m", "20-30m", "30-45m", "45-60m", "60m+"]
    by: dict[str, list[TradeAnalysis]] = defaultdict(list)
    for t in trades:
        by[_bucket_hold(t.hold_mins)].append(t)
    print(f"\n## 按持仓时长分桶 (实际 vs 持仓内峰值)")
    print(f"{'bucket':<8} {'n':>5} {'win%':>6} {'real_mean':>11} {'peak_mean':>11} {'leak_mean':>11}")
    for b in order:
        lst = by.get(b) or []
        if not lst:
            continue
        n = len(lst)
        realized = [t.realized_roi for t in lst]
        peaks = [p for p in (_peak_roi(t) for t in lst) if p is not None]
        leakages = [(p - t.realized_roi) for t, p in zip(lst, (_peak_roi(t) for t in lst)) if p is not None]
        wr = sum(1 for x in realized if x > 0) / max(1, n)
        print(
            f"{b:<8} {n:>5} {wr*100:>5.1f}% {_fmt_pct(mean(realized)):>11} "
            f"{_fmt_pos(mean(peaks)) if peaks else 'n/a':>11} "
            f"{_fmt_pos(mean(leakages)) if leakages else 'n/a':>11}"
        )


def _peak_timing(trades: list[TradeAnalysis]) -> None:
    """统计实际 ROI 与峰值 ROI 的 ratio 分布, 看 capture quality."""
    captures = [t.hold_capture_ratio for t in trades if t.hold_capture_ratio is not None]
    if not captures:
        return
    print(f"\n## 持仓内利润捕获率 (realized / hold_peak)")
    bins = [(-math.inf, 0), (0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0), (1.0, math.inf)]
    counts = [0] * len(bins)
    for c in captures:
        for i, (lo, hi) in enumerate(bins):
            if lo <= c < hi:
                counts[i] += 1
                break
    total = max(1, len(captures))
    labels = ["<0 (亏损)", "0-20%", "20-40%", "40-60%", "60-80%", "80-100%", ">=100%"]
    for label, cnt in zip(labels, counts):
        bar = "█" * int(cnt / total * 50)
        print(f"  {label:<10} {cnt:>4} ({cnt/total*100:>5.1f}%)  {bar}")


def _top_leakage(trades: list[TradeAnalysis], top_n: int) -> None:
    enriched = []
    for t in trades:
        peak = _peak_roi(t)
        if peak is None:
            continue
        leak = peak - t.realized_roi
        enriched.append((leak, peak, t))
    enriched.sort(key=lambda x: x[0], reverse=True)
    print(f"\n## Top {top_n} 单笔漏出最大 (持仓内峰值 - 实际 ROI)")
    print(f"{'date':<11} {'symbol':<6} {'side':<5} {'hold':>6} {'real':>9} {'peak':>9} {'leak':>9} {'reason'}")
    for leak, peak, t in enriched[:top_n]:
        date_part = t.exit_dt[:10]
        print(
            f"{date_part:<11} {t.symbol:<6} {t.option_side:<5} {t.hold_mins:>5.1f}m "
            f"{_fmt_pct(t.realized_roi):>9} {_fmt_pos(peak):>9} {_fmt_pos(leak):>9} "
            f"{_normalize_reason(t.exit_reason)}"
        )


def main() -> None:
    args = parse_args()
    dates = [d.strip() for d in args.dates.split(",") if d.strip()]
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    print(f"日期: {dates}  symbols={symbols or 'ALL'}  future_mins={args.future_mins}")

    all_trades: list[TradeAnalysis] = []
    by_day: dict[str, list[TradeAnalysis]] = {}
    for d in dates:
        print(f"  载入 {d} ...", flush=True)
        ts = collect_trades(d, symbols, args.future_mins, args.whipsaw_threshold, args.capture_threshold)
        by_day[d] = ts
        all_trades.extend(ts)
        print(f"    交易笔数={len(ts)}")

    if not all_trades:
        print("没有可分析的交易.")
        return

    print(f"\n========= 跨日总览 (n={len(all_trades)}) =========")
    _summary_block(all_trades)

    print(f"\n========= 各日总览 =========")
    for d in dates:
        print(f"\n[{d}]")
        _summary_block(by_day.get(d, []))

    _by_reason(all_trades)
    _by_hold_bucket(all_trades)
    _peak_timing(all_trades)
    _top_leakage(all_trades, args.top_n)

    print(f"\n========= 直觉式提升估算 =========")
    realized = [t.realized_roi for t in all_trades]
    peaks = [p for p in (_peak_roi(t) for t in all_trades) if p is not None]
    if peaks and realized:
        sum_real = sum(realized)
        sum_leak = sum((p - t.realized_roi) for t, p in zip(all_trades, (_peak_roi(t) for t in all_trades)) if p is not None)
        print(f"实际收益总和: {_fmt_pct(sum_real)} (累计 ROI 单位)")
        print(f"持仓内最优收益总和: {_fmt_pct(sum_real + sum_leak)}")
        print(f"漏出空间: {_fmt_pos(sum_leak)} (即如果每笔都能在持仓内峰值平仓, 还能多赚的 ROI)")
        if sum_real != 0:
            print(f"漏出 / 实际 = {sum_leak / max(1e-9, abs(sum_real)) * 100:.1f}% (相对 lift 上限)")


if __name__ == "__main__":
    main()

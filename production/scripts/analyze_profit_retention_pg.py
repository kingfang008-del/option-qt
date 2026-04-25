#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from pathlib import Path
from statistics import mean, median
from typing import Any

import psycopg2
from psycopg2.extras import RealDictCursor


def _bootstrap_imports() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    production_dir = repo_root / "production"
    baseline_dir = production_dir / "baseline"
    sys.path.insert(0, str(production_dir))
    sys.path.insert(0, str(baseline_dir))


_bootstrap_imports()
from config import PG_DB_URL  # noqa: E402


CONTRACT_RE = re.compile(r"(\d{6})([CP])(\d{8})")
SENSITIVE_EXIT_TOKENS = ("NO_MOMENTUM", "MACD_FADE", "FLIP", "IDX_REVERSAL", "CT_TIMEOUT")


@dataclass
class TradeLot:
    symbol: str
    qty: float
    entry_ts: float
    entry_dt: str
    entry_price: float
    entry_reason: str
    contract_id: str
    option_side: str
    strike: float | None
    pending_tag: str


@dataclass
class TradeAnalysis:
    symbol: str
    contract_id: str
    option_side: str
    strike: float | None
    qty: float
    entry_dt: str
    exit_dt: str
    hold_mins: float
    entry_price: float
    exit_price: float
    realized_roi: float
    exit_reason: str
    hold_peak_mid: float | None
    future_peak_mid: float | None
    total_peak_mid: float | None
    hold_capture_ratio: float | None
    total_capture_ratio: float | None
    rebound_after_exit_pct: float | None
    stock_exit: float | None
    stock_future_peak: float | None
    stock_rebound_after_exit_pct: float | None
    whipsaw_flag: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="分析策略是否过于灵敏、是否锁住大部分利润")
    parser.add_argument("--date", required=True, help="交易日，格式 2026-04-24")
    parser.add_argument("--symbols", default="", help="逗号分隔，仅分析指定标的")
    parser.add_argument("--future-mins", type=int, default=10, help="卖出后继续观察多少分钟")
    parser.add_argument("--whipsaw-threshold", type=float, default=0.08, help="卖出后再次上涨超过该比例视为疑似震荡洗出")
    parser.add_argument("--capture-threshold", type=float, default=0.60, help="利润捕获率低于该值且出现反弹时，记为疑似过敏")
    parser.add_argument("--csv-out", default="", help="可选，输出逐笔分析 CSV")
    parser.add_argument("--limit", type=int, default=20, help="终端输出的问题交易条数")
    return parser.parse_args()


def _parse_contract(contract_id: str) -> tuple[str, float | None]:
    if not contract_id:
        return "", None
    m = CONTRACT_RE.search(contract_id.replace(" ", ""))
    if not m:
        return "", None
    opt_side = "CALL" if m.group(2) == "C" else "PUT"
    strike = int(m.group(3)) / 1000.0
    return opt_side, strike


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _parse_details(details_json: str) -> dict[str, Any]:
    try:
        payload = json.loads(details_json or "{}")
    except Exception:
        return {}
    strategy_note = payload.get("strategy_note")
    if isinstance(strategy_note, str):
        try:
            payload["strategy_note"] = json.loads(strategy_note)
        except Exception:
            payload["strategy_note"] = {"reason": strategy_note}
    return payload


def _minute_floor(ts: float) -> int:
    return int(float(ts) // 60 * 60)


def _mid_from_bucket(bucket: list[Any]) -> float | None:
    if not bucket:
        return None
    bid = _safe_float(bucket[8] if len(bucket) > 8 else 0.0)
    ask = _safe_float(bucket[9] if len(bucket) > 9 else 0.0)
    px = _safe_float(bucket[0] if len(bucket) > 0 else 0.0)
    if bid > 0 and ask > 0:
        return (bid + ask) / 2.0
    if px > 0:
        return px
    return None


def _pick_bucket_mid(buckets_json: str, option_side: str, strike: float | None, ref_price: float | None) -> float | None:
    try:
        buckets = json.loads(buckets_json or "{}").get("buckets", [])
    except Exception:
        return None
    side_sign = 1 if option_side == "CALL" else -1
    candidates: list[tuple[float, float, float]] = []
    for bucket in buckets:
        if not isinstance(bucket, list) or len(bucket) < 10:
            continue
        delta = _safe_float(bucket[1] if len(bucket) > 1 else 0.0)
        if delta == 0 or (delta > 0) != (side_sign > 0):
            continue
        mid = _mid_from_bucket(bucket)
        if mid is None:
            continue
        bucket_strike = _safe_float(bucket[5] if len(bucket) > 5 else 0.0)
        strike_gap = abs(bucket_strike - float(strike or bucket_strike))
        price_gap = abs(mid - float(ref_price or mid))
        candidates.append((strike_gap, price_gap, mid))
    if not candidates:
        return None
    candidates.sort(key=lambda x: (x[0], x[1]))
    return candidates[0][2]


def _format_pct(v: float | None) -> str:
    if v is None or math.isnan(v):
        return "n/a"
    return f"{v * 100:.1f}%"


def _fetch_rows(conn, query: str, params: tuple[Any, ...]) -> list[dict[str, Any]]:
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(query, params)
        return [dict(row) for row in cur.fetchall()]


def load_order_events(conn, date_str: str, symbols: list[str]) -> list[dict[str, Any]]:
    where_symbols = ""
    params: list[Any] = [f"{date_str}%"]
    if symbols:
        where_symbols = " AND symbol = ANY(%s)"
        params.append(symbols)
    query = f"""
        SELECT ts, datetime_ny, symbol, action, qty, price, details_json
        FROM order_events
        WHERE datetime_ny LIKE %s
          AND action IN ('ORDER_PENDING', 'ORDER_FILLED')
          {where_symbols}
        ORDER BY ts ASC
    """
    rows = _fetch_rows(conn, query, tuple(params))
    for row in rows:
        row["details"] = _parse_details(row.get("details_json", ""))
    return rows


def load_option_paths(conn, min_ts: int, max_ts: int, symbols: list[str]) -> dict[str, dict[int, str]]:
    params: list[Any] = [min_ts, max_ts]
    where_symbols = ""
    if symbols:
        where_symbols = " AND symbol = ANY(%s)"
        params.append(symbols)
    query = f"""
        SELECT symbol, ts, buckets_json
        FROM option_snapshots_1m
        WHERE ts >= %s AND ts <= %s
          {where_symbols}
        ORDER BY symbol ASC, ts ASC
    """
    rows = _fetch_rows(conn, query, tuple(params))
    out: dict[str, dict[int, str]] = defaultdict(dict)
    for row in rows:
        out[str(row["symbol"])][int(row["ts"])] = str(row["buckets_json"])
    return out


def load_stock_paths(conn, min_ts: int, max_ts: int, symbols: list[str]) -> dict[str, dict[int, float]]:
    params: list[Any] = [min_ts, max_ts]
    where_symbols = ""
    if symbols:
        where_symbols = " AND symbol = ANY(%s)"
        params.append(symbols)
    query = f"""
        SELECT symbol, ts, close
        FROM market_bars_1m
        WHERE ts >= %s AND ts <= %s
          {where_symbols}
        ORDER BY symbol ASC, ts ASC
    """
    rows = _fetch_rows(conn, query, tuple(params))
    out: dict[str, dict[int, float]] = defaultdict(dict)
    for row in rows:
        out[str(row["symbol"])][int(row["ts"])] = _safe_float(row["close"])
    return out


def build_trade_analyses(
    rows: list[dict[str, Any]],
    option_paths: dict[str, dict[int, str]],
    stock_paths: dict[str, dict[int, float]],
    future_mins: int,
    whipsaw_threshold: float,
    capture_threshold: float,
) -> list[TradeAnalysis]:
    pending_by_symbol_side: dict[tuple[str, str], deque[dict[str, Any]]] = defaultdict(deque)
    open_lots: dict[str, deque[TradeLot]] = defaultdict(deque)
    trades: list[TradeAnalysis] = []

    for row in rows:
        details = row.get("details", {})
        side = str(details.get("side") or "").upper()
        symbol = str(row["symbol"])

        if row["action"] == "ORDER_PENDING":
            pending_by_symbol_side[(symbol, side)].append(row)
            continue

        if row["action"] != "ORDER_FILLED":
            continue

        while pending_by_symbol_side[(symbol, side)] and _safe_float(pending_by_symbol_side[(symbol, side)][0]["ts"]) < _safe_float(row["ts"]) - 180:
            pending_by_symbol_side[(symbol, side)].popleft()

        pending = pending_by_symbol_side[(symbol, side)][-1] if pending_by_symbol_side[(symbol, side)] else None
        pending_note = (pending or {}).get("details", {}).get("strategy_note", {}) if pending else {}
        reason = str((details.get("strategy_note", {}) or {}).get("reason") or (pending_note or {}).get("reason") or "")
        contract_id = str((pending_note or {}).get("contract_id") or "")
        pending_tag = str((pending_note or {}).get("tag") or "")

        if side == "BUY":
            opt_side, strike = _parse_contract(contract_id)
            if not opt_side:
                opt_side = "CALL" if "CALL" in pending_tag else "PUT"
            open_lots[symbol].append(
                TradeLot(
                    symbol=symbol,
                    qty=_safe_float(row["qty"]),
                    entry_ts=_safe_float(row["ts"]),
                    entry_dt=str(row["datetime_ny"]),
                    entry_price=_safe_float(row["price"]),
                    entry_reason=reason,
                    contract_id=contract_id,
                    option_side=opt_side,
                    strike=strike,
                    pending_tag=pending_tag,
                )
            )
            continue

        if side != "SELL":
            continue

        remaining = _safe_float(row["qty"])
        while remaining > 1e-9 and open_lots[symbol]:
            lot = open_lots[symbol][0]
            matched_qty = min(remaining, lot.qty)
            lot.qty -= matched_qty
            remaining -= matched_qty

            entry_floor = _minute_floor(lot.entry_ts)
            exit_floor = _minute_floor(_safe_float(row["ts"]))
            future_last = exit_floor + int(future_mins) * 60
            option_series: list[tuple[int, float]] = []
            for ts_key, buckets_json in option_paths.get(symbol, {}).items():
                if ts_key < entry_floor or ts_key > future_last:
                    continue
                mid = _pick_bucket_mid(buckets_json, lot.option_side, lot.strike, lot.entry_price)
                if mid is not None:
                    option_series.append((ts_key, mid))
            option_series.sort(key=lambda x: x[0])

            hold_path = [mid for ts_key, mid in option_series if ts_key <= exit_floor]
            future_path = [mid for ts_key, mid in option_series if ts_key > exit_floor]
            total_path = [mid for _, mid in option_series]

            hold_peak = max(hold_path) if hold_path else None
            future_peak = max(future_path) if future_path else None
            total_peak = max(total_path) if total_path else None

            realized_roi = (_safe_float(row["price"]) - lot.entry_price) / max(0.01, lot.entry_price)
            hold_capture = None
            total_capture = None
            if hold_peak and hold_peak > lot.entry_price + 1e-9:
                hold_capture = (_safe_float(row["price"]) - lot.entry_price) / (hold_peak - lot.entry_price)
            if total_peak and total_peak > lot.entry_price + 1e-9:
                total_capture = (_safe_float(row["price"]) - lot.entry_price) / (total_peak - lot.entry_price)

            rebound_after_exit = None
            if future_peak and future_peak > 0.0:
                rebound_after_exit = (future_peak - _safe_float(row["price"])) / max(0.01, _safe_float(row["price"]))

            stock_exit = stock_paths.get(symbol, {}).get(exit_floor)
            stock_future_peak = None
            if stock_paths.get(symbol):
                stock_future_vals = [
                    px for ts_key, px in stock_paths[symbol].items()
                    if ts_key > exit_floor and ts_key <= future_last
                ]
                if stock_future_vals:
                    stock_future_peak = max(stock_future_vals)
            stock_rebound = None
            if stock_exit and stock_future_peak and stock_future_peak > 0:
                stock_rebound = (stock_future_peak - stock_exit) / stock_exit

            whipsaw_flag = (
                any(token in reason for token in SENSITIVE_EXIT_TOKENS)
                and rebound_after_exit is not None
                and rebound_after_exit >= whipsaw_threshold
                and (total_capture is None or total_capture < capture_threshold)
            )

            trades.append(
                TradeAnalysis(
                    symbol=symbol,
                    contract_id=lot.contract_id,
                    option_side=lot.option_side,
                    strike=lot.strike,
                    qty=matched_qty,
                    entry_dt=lot.entry_dt,
                    exit_dt=str(row["datetime_ny"]),
                    hold_mins=max(0.0, (_safe_float(row["ts"]) - lot.entry_ts) / 60.0),
                    entry_price=lot.entry_price,
                    exit_price=_safe_float(row["price"]),
                    realized_roi=realized_roi,
                    exit_reason=reason,
                    hold_peak_mid=hold_peak,
                    future_peak_mid=future_peak,
                    total_peak_mid=total_peak,
                    hold_capture_ratio=hold_capture,
                    total_capture_ratio=total_capture,
                    rebound_after_exit_pct=rebound_after_exit,
                    stock_exit=stock_exit,
                    stock_future_peak=stock_future_peak,
                    stock_rebound_after_exit_pct=stock_rebound,
                    whipsaw_flag=whipsaw_flag,
                )
            )

            if lot.qty <= 1e-9:
                open_lots[symbol].popleft()

    return trades


def print_summary(trades: list[TradeAnalysis], limit: int) -> None:
    if not trades:
        print("没有匹配到完整 round-trip 交易。")
        return

    realized = [t.realized_roi for t in trades]
    hold_capture = [t.hold_capture_ratio for t in trades if t.hold_capture_ratio is not None]
    total_capture = [t.total_capture_ratio for t in trades if t.total_capture_ratio is not None]
    rebound = [t.rebound_after_exit_pct for t in trades if t.rebound_after_exit_pct is not None]
    whipsaw = [t for t in trades if t.whipsaw_flag]

    print("## 总览")
    print(f"交易笔数: {len(trades)}")
    print(f"盈利占比: {sum(1 for x in realized if x > 0) / max(1, len(realized)) * 100:.1f}%")
    print(f"平均持仓: {mean(t.hold_mins for t in trades):.1f} 分钟 | 中位持仓: {median(t.hold_mins for t in trades):.1f} 分钟")
    if hold_capture:
        print(f"持仓内利润捕获率: 均值 {mean(hold_capture):.2f} | 中位 {median(hold_capture):.2f}")
    if total_capture:
        print(f"含卖出后未来机会的利润捕获率: 均值 {mean(total_capture):.2f} | 中位 {median(total_capture):.2f}")
    if rebound:
        print(f"卖出后未来窗口再次上涨: 均值 {_format_pct(mean(rebound))} | 中位 {_format_pct(median(rebound))}")
    print(f"疑似震荡洗出笔数: {len(whipsaw)} / {len(trades)}")

    if whipsaw:
        print("\n## 疑似过敏交易")
        ranked = sorted(
            whipsaw,
            key=lambda x: float(x.rebound_after_exit_pct or 0.0),
            reverse=True,
        )[:limit]
        for trade in ranked:
            print(
                f"{trade.exit_dt} {trade.symbol} {trade.option_side} "
                f"entry={trade.entry_price:.2f} exit={trade.exit_price:.2f} "
                f"roi={_format_pct(trade.realized_roi)} "
                f"future_rebound={_format_pct(trade.rebound_after_exit_pct)} "
                f"capture={trade.total_capture_ratio:.2f if trade.total_capture_ratio is not None else 'n/a'} "
                f"reason={trade.exit_reason}"
            )


def write_csv(path: str, trades: list[TradeAnalysis]) -> None:
    import csv

    rows = [asdict(t) for t in trades]
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

    conn = psycopg2.connect(PG_DB_URL)
    try:
        rows = load_order_events(conn, args.date, symbols)
        if not rows:
            print("当天没有订单事件。")
            return

        fill_rows = [r for r in rows if r["action"] == "ORDER_FILLED"]
        if not fill_rows:
            print("当天没有成交事件。")
            return

        min_ts = _minute_floor(min(_safe_float(r["ts"]) for r in fill_rows))
        max_ts = _minute_floor(max(_safe_float(r["ts"]) for r in fill_rows)) + int(args.future_mins) * 60
        event_symbols = sorted({str(r["symbol"]) for r in fill_rows})

        option_paths = load_option_paths(conn, min_ts, max_ts, symbols or event_symbols)
        stock_paths = load_stock_paths(conn, min_ts, max_ts, symbols or event_symbols)
        trades = build_trade_analyses(
            rows,
            option_paths,
            stock_paths,
            future_mins=args.future_mins,
            whipsaw_threshold=args.whipsaw_threshold,
            capture_threshold=args.capture_threshold,
        )
    finally:
        conn.close()

    print_summary(trades, args.limit)
    if args.csv_out:
        write_csv(args.csv_out, trades)
        print(f"\nCSV 已写出: {args.csv_out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Debug a single MU option position against real PostgreSQL 1s snapshots."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import psycopg2


REPO_ROOT = Path(__file__).resolve().parents[2]
BASELINE_DIR = REPO_ROOT / "production" / "baseline"
sys.path.insert(0, str(BASELINE_DIR))

from config import PG_DB_URL  # noqa: E402
from strategy_config0 import StrategyConfig  # noqa: E402


try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover
    ZoneInfo = None


NY_TZ = ZoneInfo("America/New_York") if ZoneInfo else None


@dataclass
class Trade:
    ts: float
    action: str
    qty: float
    price: float
    tag: str
    reason: str
    details: dict[str, Any]


def _dt(ts: float) -> str:
    if NY_TZ:
        return datetime.fromtimestamp(float(ts), NY_TZ).strftime("%Y-%m-%d %H:%M:%S %Z")
    return datetime.fromtimestamp(float(ts)).isoformat(sep=" ", timespec="seconds")


def _as_details(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        details = raw
    elif isinstance(raw, str) and raw.strip():
        details = json.loads(raw)
    else:
        details = {}
    note = details.get("strategy_note")
    if isinstance(note, str) and note.strip():
        try:
            details["_strategy_note_obj"] = json.loads(note)
        except Exception:
            details["_strategy_note_obj"] = {"raw": note}
    elif isinstance(note, dict):
        details["_strategy_note_obj"] = note
    else:
        details["_strategy_note_obj"] = {}
    return details


def _trade_from_row(row: tuple[Any, ...]) -> Trade:
    ts, action, qty, price, details_raw = row
    details = _as_details(details_raw)
    note = details.get("_strategy_note_obj", {})
    return Trade(
        ts=float(ts),
        action=str(action),
        qty=float(qty or 0.0),
        price=float(price or 0.0),
        tag=str(note.get("tag") or details.get("tag") or ""),
        reason=str(note.get("reason") or details.get("reason") or ""),
        details=details,
    )


def _parse_buckets(raw: Any) -> list[Any]:
    if isinstance(raw, dict):
        return raw.get("buckets") or []
    if isinstance(raw, str) and raw.strip():
        payload = json.loads(raw)
        return payload.get("buckets") or []
    return []


def _fair_price(bucket: list[Any]) -> tuple[float, float, float, float]:
    last = float(bucket[0] or 0.0) if len(bucket) > 0 else 0.0
    bid = float(bucket[8] or 0.0) if len(bucket) > 8 else 0.0
    ask = float(bucket[9] or 0.0) if len(bucket) > 9 else 0.0
    if bid > 0.01 and ask > 0.01:
        px = (bid + ask) / 2.0
    elif bid > 0.01:
        px = bid
    elif ask > 0.01:
        px = ask
    else:
        px = last
    return px, last, bid, ask


def _side_bucket(tag: str) -> tuple[str, int]:
    upper = str(tag or "").upper()
    if "PUT" in upper:
        return "PUT", 0
    return "CALL", 2


def fetch_trades(conn, symbol: str, table: str) -> list[Trade]:
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT ts, action, qty, price, details_json
            FROM {table}
            WHERE symbol = %s
            ORDER BY ts ASC
            """,
            (symbol,),
        )
        return [_trade_from_row(row) for row in cur.fetchall()]


def pair_trades(trades: list[Trade]) -> list[tuple[Trade, Trade]]:
    pairs: list[tuple[Trade, Trade]] = []
    open_trade: Trade | None = None
    for trade in trades:
        action = trade.action.upper()
        if action == "OPEN":
            open_trade = trade
        elif action == "CLOSE" and open_trade is not None:
            pairs.append((open_trade, trade))
            open_trade = None
    return pairs


def choose_pair(pairs: list[tuple[Trade, Trade]], entry_ts: float | None) -> tuple[Trade, Trade]:
    if not pairs:
        raise RuntimeError("没有找到可配对的 OPEN/CLOSE")
    if entry_ts is None:
        return pairs[-1]
    return min(pairs, key=lambda p: abs(p[0].ts - entry_ts))


def fetch_second_path(conn, symbol: str, ymd: str, entry: Trade, close: Trade, bucket_idx: int):
    table = f"option_snapshots_1s_{ymd}"
    rows = []
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT ts, buckets_json
            FROM {table}
            WHERE symbol = %s AND ts BETWEEN %s AND %s
            ORDER BY ts ASC
            """,
            (symbol, entry.ts - 5, close.ts + 5),
        )
        for ts, raw in cur.fetchall():
            buckets = _parse_buckets(raw)
            if bucket_idx >= len(buckets):
                continue
            px, last, bid, ask = _fair_price(buckets[bucket_idx])
            if px <= 0.01:
                continue
            roi = (px - entry.price) / entry.price if entry.price > 0 else 0.0
            rows.append(
                {
                    "ts": float(ts),
                    "px": px,
                    "last": last,
                    "bid": bid,
                    "ask": ask,
                    "roi": roi,
                }
            )
    return rows


def first_ladder_trigger(rows: list[dict[str, float]], ladder: list[tuple[float, float]]):
    max_roi = -999.0
    for row in rows:
        max_roi = max(max_roi, row["roi"])
        for trigger, floor in sorted(ladder, reverse=True):
            if max_roi >= trigger:
                if row["roi"] < floor:
                    return row, max_roi, trigger, floor
                break
    return None


def minute_samples(rows: list[dict[str, float]]) -> list[tuple[int, dict[str, float], float]]:
    by_minute: dict[int, dict[str, float]] = {}
    for row in rows:
        minute_ts = int(row["ts"] // 60) * 60
        by_minute.setdefault(minute_ts, row)
    out = []
    max_roi = -999.0
    for minute_ts in sorted(by_minute):
        row = by_minute[minute_ts]
        max_roi = max(max_roi, row["roi"])
        out.append((minute_ts, row, max_roi))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="用真实 PG 1s 快照调试 MU 开仓后的退出位置")
    parser.add_argument("--date", default="20260501", help="交易日，格式 YYYYMMDD")
    parser.add_argument("--symbol", default="MU")
    parser.add_argument("--table", default="trade_logs_backtest")
    parser.add_argument("--entry-ts", type=float, default=None, help="指定 OPEN ts；不传则使用最后一组配对交易")
    args = parser.parse_args()

    conn = psycopg2.connect(PG_DB_URL)
    try:
        trades = fetch_trades(conn, args.symbol, args.table)
        pairs = pair_trades(trades)
        entry, close = choose_pair(pairs, args.entry_ts)
        side, bucket_idx = _side_bucket(entry.tag)
        rows = fetch_second_path(conn, args.symbol, args.date, entry, close, bucket_idx)
    finally:
        conn.close()

    if not rows:
        raise RuntimeError("没有读取到对应 1s 期权快照")

    peak = max(rows, key=lambda r: r["roi"])
    trough = min(rows, key=lambda r: r["roi"])
    cfg = StrategyConfig()
    ladder_hit = first_ladder_trigger(rows, list(cfg.LADDER_TIGHT))

    print(f"symbol={args.symbol} side={side} bucket={bucket_idx} table={args.table}")
    print(f"OPEN  {_dt(entry.ts)} qty={entry.qty:g} price={entry.price:.4f} tag={entry.tag} reason={entry.reason}")
    print(f"CLOSE {_dt(close.ts)} qty={close.qty:g} price={close.price:.4f} reason={close.reason}")
    print(f"realized_roi={(close.price - entry.price) / entry.price:.2%}")
    print(f"peak_1s {_dt(peak['ts'])} px={peak['px']:.4f} roi={peak['roi']:.2%} bid={peak['bid']:.4f} ask={peak['ask']:.4f}")
    print(f"low_1s  {_dt(trough['ts'])} px={trough['px']:.4f} roi={trough['roi']:.2%} bid={trough['bid']:.4f} ask={trough['ask']:.4f}")
    if ladder_hit:
        row, max_roi, trigger, floor = ladder_hit
        print(
            "first_ladder_1s "
            f"{_dt(row['ts'])} px={row['px']:.4f} roi={row['roi']:.2%} "
            f"max_roi={max_roi:.2%} trigger={trigger:.0%} floor={floor:.0%}"
        )
    else:
        print("first_ladder_1s NONE")

    print("\nminute_samples")
    for minute_ts, row, max_roi in minute_samples(rows):
        print(
            f"{_dt(minute_ts)} sample={_dt(row['ts']).split()[1]} "
            f"px={row['px']:.4f} roi={row['roi']:.2%} max_roi={max_roi:.2%} "
            f"bid={row['bid']:.4f} ask={row['ask']:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Audit V0 exit guards on PostgreSQL historical order/data paths.

This script answers a narrow live-trading question:
- Did V0's ladder / no-momentum / stop-loss guards *theoretically* trigger
  on a filled position path?
- If so, when did the trigger first appear relative to the actual close?

Example:
    python production/scripts/audit_v0_exit_guards_pg.py --date 2026-04-24 --symbols COIN,MSTR
"""

from __future__ import annotations

import argparse
import ast
import math
import sys
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import psycopg2


SCRIPT_DIR = Path(__file__).resolve().parent
BASELINE_DIR = SCRIPT_DIR.parent / "baseline"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(BASELINE_DIR) not in sys.path:
    sys.path.insert(0, str(BASELINE_DIR))

from analyze_profit_retention_pg import (  # noqa: E402
    _format_pct,
    _fetch_rows,
    _minute_floor,
    _parse_contract,
    _parse_details,
    _safe_float,
    load_option_paths,
    load_stock_paths,
)
from config import NY_TZ, PG_DB_URL  # noqa: E402
from strategy_config0 import StrategyConfig  # noqa: E402
from strategy_core_v0 import StrategyCoreV0  # noqa: E402


@dataclass
class Lot:
    symbol: str
    qty: float
    entry_ts: float
    entry_dt: str
    entry_price: float
    entry_stock: float
    entry_reason: str
    contract_id: str
    option_side: str
    strike: float | None
    pending_tag: str


@dataclass
class LotAudit:
    symbol: str
    option_side: str
    entry_dt: str
    actual_exit_dt: str
    entry_price: float
    actual_exit_price: float | None
    actual_exit_reason: str
    hold_mins: float
    peak_roi: float
    actual_exit_roi: float | None
    first_full_exit: tuple[int, float, float, str] | None
    first_profit_exit: tuple[int, float, float, str] | None
    first_no_momentum: tuple[int, float, float, str] | None
    first_hard_stop: tuple[int, float, float, str] | None
    path_points: int


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Audit V0 ladder/no-momentum exits using PGSQL order + 1m option data.")
    p.add_argument("--date", default="2026-04-24", help="NY date, e.g. 2026-04-24")
    p.add_argument("--symbols", default="COIN,MSTR", help="Comma-separated symbols, default COIN,MSTR")
    p.add_argument("--future-mins", type=int, default=30, help="Audit open lots until actual exit or entry+future-mins")
    p.add_argument("--show-path", action="store_true", help="Print per-minute ROI path for flagged lots")
    p.add_argument("--limit", type=int, default=30)
    return p.parse_args()


def load_order_events_with_cancelled(conn, date_str: str, symbols: list[str]) -> list[dict[str, Any]]:
    where_symbols = ""
    params: list[Any] = [f"{date_str}%"]
    if symbols:
        where_symbols = " AND symbol = ANY(%s)"
        params.append(symbols)
    rows = _fetch_rows(
        conn,
        f"""
        SELECT ts, datetime_ny, symbol, action, qty, price, details_json
        FROM order_events
        WHERE datetime_ny LIKE %s
          AND action IN ('ORDER_PENDING', 'ORDER_FILLED', 'ORDER_CANCELLED')
          {where_symbols}
        ORDER BY ts ASC
        """,
        tuple(params),
    )
    for row in rows:
        row["details"] = _parse_details(row.get("details_json", ""))
    return rows


def _dt(ts: float) -> datetime:
    return datetime.fromtimestamp(float(ts), NY_TZ)


def _dt_str(ts: float | int | None) -> str:
    if not ts:
        return "n/a"
    return _dt(float(ts)).strftime("%Y-%m-%d %H:%M:%S")


def _day_bounds(date_str: str) -> tuple[int, int]:
    start = NY_TZ.localize(datetime.strptime(date_str, "%Y-%m-%d"))
    end = start + timedelta(days=1)
    return int(start.timestamp()), int(end.timestamp())


def _row_reason(row: dict[str, Any], pending: dict[str, Any] | None = None) -> str:
    details = row.get("details", {}) or {}
    pending_note = (pending or {}).get("details", {}).get("strategy_note", {}) if pending else {}
    note = details.get("strategy_note", {}) or {}
    return str(note.get("reason") or pending_note.get("reason") or "")


def _row_side(row: dict[str, Any]) -> str:
    return str((row.get("details", {}) or {}).get("side") or "").upper()


def _row_contract(row: dict[str, Any] | None) -> tuple[str, str]:
    if not row:
        return "", ""
    note = (row.get("details", {}) or {}).get("strategy_note", {}) or {}
    return str(note.get("contract_id") or ""), str(note.get("tag") or "")


def build_lots(rows: list[dict[str, Any]], stock_paths: dict[str, dict[int, float]]) -> list[tuple[Lot, dict[str, Any] | None]]:
    pending_by_symbol_side: dict[tuple[str, str], deque[dict[str, Any]]] = defaultdict(deque)
    open_lots: dict[str, deque[Lot]] = defaultdict(deque)
    paired: list[tuple[Lot, dict[str, Any] | None]] = []

    for row in rows:
        details = row.get("details", {}) or {}
        side = str(details.get("side") or "").upper()
        sym = str(row.get("symbol") or "")
        if not sym:
            continue
        if row.get("action") == "ORDER_PENDING":
            pending_by_symbol_side[(sym, side)].append(row)
            continue
        if row.get("action") != "ORDER_FILLED":
            continue

        while pending_by_symbol_side[(sym, side)] and _safe_float(pending_by_symbol_side[(sym, side)][0]["ts"]) < _safe_float(row["ts"]) - 180:
            pending_by_symbol_side[(sym, side)].popleft()
        pending = pending_by_symbol_side[(sym, side)][-1] if pending_by_symbol_side[(sym, side)] else None

        if side == "BUY":
            contract_id, pending_tag = _row_contract(pending)
            option_side, strike = _parse_contract(contract_id)
            if not option_side:
                option_side = "CALL" if "CALL" in pending_tag else "PUT"
            entry_floor = _minute_floor(_safe_float(row["ts"]))
            entry_stock = stock_paths.get(sym, {}).get(entry_floor, 0.0)
            open_lots[sym].append(
                Lot(
                    symbol=sym,
                    qty=_safe_float(row["qty"]),
                    entry_ts=_safe_float(row["ts"]),
                    entry_dt=str(row.get("datetime_ny") or _dt_str(_safe_float(row["ts"]))),
                    entry_price=_safe_float(row["price"]),
                    entry_stock=float(entry_stock or 0.0),
                    entry_reason=_row_reason(row, pending),
                    contract_id=contract_id,
                    option_side=option_side,
                    strike=strike,
                    pending_tag=pending_tag,
                )
            )
            continue

        if side != "SELL":
            continue
        remaining = _safe_float(row["qty"])
        while remaining > 1e-9 and open_lots[sym]:
            lot = open_lots[sym][0]
            matched = min(remaining, lot.qty)
            lot.qty -= matched
            remaining -= matched
            paired.append((lot, row))
            if lot.qty <= 1e-9:
                open_lots[sym].popleft()

    for lots in open_lots.values():
        for lot in lots:
            paired.append((lot, None))
    return paired


def option_path_for_lot(
    lot: Lot,
    option_paths: dict[str, dict[int, str]],
    start_ts: int,
    end_ts: int,
) -> list[tuple[int, float]]:
    points: list[tuple[int, float]] = []
    for ts_key, buckets_json in option_paths.get(lot.symbol, {}).items():
        if ts_key < start_ts or ts_key > end_ts:
            continue
        mid = _pick_bucket_mid_any(buckets_json, lot.option_side, lot.strike, lot.entry_price)
        if mid is not None and mid > 0:
            points.append((int(ts_key), float(mid)))
    points.sort(key=lambda x: x[0])
    return points


def _parse_buckets_any(payload: Any) -> list[Any]:
    if isinstance(payload, dict):
        buckets = payload.get("buckets", [])
        return buckets if isinstance(buckets, list) else []
    if isinstance(payload, list):
        return payload
    text = str(payload or "").strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
    except Exception:
        try:
            import json
            parsed = json.loads(text)
        except Exception:
            return []
    if isinstance(parsed, dict):
        buckets = parsed.get("buckets", [])
        return buckets if isinstance(buckets, list) else []
    return parsed if isinstance(parsed, list) else []


def _bucket_mid(bucket: list[Any]) -> float | None:
    bid = _safe_float(bucket[8] if len(bucket) > 8 else 0.0)
    ask = _safe_float(bucket[9] if len(bucket) > 9 else 0.0)
    px = _safe_float(bucket[0] if len(bucket) > 0 else 0.0)
    if bid > 0 and ask > 0:
        return (bid + ask) / 2.0
    if px > 0:
        return px
    return None


def _pick_bucket_mid_any(payload: Any, option_side: str, strike: float | None, ref_price: float | None) -> float | None:
    buckets = _parse_buckets_any(payload)
    side_sign = 1 if option_side == "CALL" else -1
    candidates: list[tuple[float, float, float]] = []
    for bucket in buckets:
        if not isinstance(bucket, list) or len(bucket) < 10:
            continue
        delta = _safe_float(bucket[1] if len(bucket) > 1 else 0.0)
        if delta == 0 or (delta > 0) != (side_sign > 0):
            continue
        mid = _bucket_mid(bucket)
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


def _mk_ctx(lot: Lot, ts_key: int, mid: float, max_roi: float, stock_px: float) -> dict[str, Any]:
    held_mins = max(0.0, (float(ts_key) - float(lot.entry_ts)) / 60.0)
    pos = {
        "symbol": lot.symbol,
        "dir": 1 if lot.option_side == "CALL" else -1,
        "entry_ts": float(lot.entry_ts),
        "entry_price": float(lot.entry_price),
        "entry_stock": float(lot.entry_stock or stock_px or 0.0),
        "entry_index_trend": 0,
        "entry_spy_roc": 0.0,
        "max_roi": float(max_roi),
    }
    return {
        "symbol": lot.symbol,
        "time": _dt(ts_key),
        "curr_ts": float(ts_key),
        "price": float(stock_px or lot.entry_stock or 0.0),
        "curr_stock": float(stock_px or lot.entry_stock or 0.0),
        "curr_price": float(mid),
        "bid": float(mid),
        "ask": float(mid),
        "alpha_z": 0.0,
        "vol_z": 0.0,
        "stock_roc": 0.0,
        "macd_hist": 0.0,
        "macd_hist_slope": 0.0,
        "spy_roc": 0.0,
        "qqq_roc": 0.0,
        "index_trend": 0,
        "position": pos["dir"],
        "cooldown_until": 0.0,
        "is_ready": True,
        "is_banned": False,
        "held_mins": held_mins,
        "stock_iv": 0.5,
        "holding": pos,
        "spread_divergence": 0.0,
        "snap_roc": 0.0,
        "regime_band": "calm",
        "is_volatile_regime": False,
    }


def audit_lot(
    core: StrategyCoreV0,
    lot: Lot,
    exit_row: dict[str, Any] | None,
    option_paths: dict[str, dict[int, str]],
    stock_paths: dict[str, dict[int, float]],
    future_mins: int,
) -> tuple[LotAudit, list[tuple[int, float, float, str]]]:
    entry_floor = _minute_floor(lot.entry_ts)
    actual_exit_ts = _safe_float((exit_row or {}).get("ts"), 0.0)
    end_ts = _minute_floor(actual_exit_ts) if actual_exit_ts > 0 else entry_floor + future_mins * 60
    path = option_path_for_lot(lot, option_paths, entry_floor, end_ts)

    max_roi = -math.inf
    first_full = None
    first_profit = None
    first_no_mom = None
    first_hard_stop = None
    trace_rows: list[tuple[int, float, float, str]] = []

    for ts_key, mid in path:
        roi = (mid - lot.entry_price) / max(0.01, lot.entry_price)
        max_roi = max(max_roi, roi)
        stock_px = stock_paths.get(lot.symbol, {}).get(ts_key, lot.entry_stock)
        ctx = _mk_ctx(lot, ts_key, mid, max_roi, stock_px)

        full_sig = core.check_exit(ctx)
        full_reason = str((full_sig or {}).get("reason") or "")
        trace_rows.append((ts_key, roi, max_roi, full_reason))
        if first_full is None and full_sig:
            first_full = (ts_key, roi, max_roi, full_reason)

        # Direct guard probes isolate the two guards in question from earlier exits.
        held_mins = ctx["held_mins"]
        core._last_gate_trace = []
        no_mom_sig = core._check_time_and_inactivity_stops(ctx, ctx["holding"], held_mins, roi)
        if first_no_mom is None and no_mom_sig and "NO_MOMENTUM" in str(no_mom_sig.get("reason", "")):
            first_no_mom = (ts_key, roi, max_roi, str(no_mom_sig.get("reason")))

        core._last_gate_trace = []
        profit_sig = core._evaluate_profit_guards(ctx["holding"], roi)
        if first_profit is None and profit_sig:
            first_profit = (ts_key, roi, max_roi, str(profit_sig.get("reason")))

        core._last_gate_trace = []
        stop_sig = core._check_stop_loss_guards(ctx, ctx["holding"], roi, 0.0)
        if first_hard_stop is None and stop_sig:
            first_hard_stop = (ts_key, roi, max_roi, str(stop_sig.get("reason")))

    if max_roi == -math.inf:
        max_roi = 0.0
    actual_exit_price = _safe_float((exit_row or {}).get("price"), 0.0) if exit_row else None
    actual_exit_roi = None
    if actual_exit_price is not None and lot.entry_price > 0:
        actual_exit_roi = (actual_exit_price - lot.entry_price) / lot.entry_price
    hold_mins = ((actual_exit_ts or end_ts) - lot.entry_ts) / 60.0
    audit = LotAudit(
        symbol=lot.symbol,
        option_side=lot.option_side,
        entry_dt=lot.entry_dt,
        actual_exit_dt=str((exit_row or {}).get("datetime_ny") or "OPEN/NO_CLOSE"),
        entry_price=lot.entry_price,
        actual_exit_price=actual_exit_price,
        actual_exit_reason=_row_reason(exit_row or {}, None) if exit_row else "",
        hold_mins=max(0.0, hold_mins),
        peak_roi=max_roi,
        actual_exit_roi=actual_exit_roi,
        first_full_exit=first_full,
        first_profit_exit=first_profit,
        first_no_momentum=first_no_mom,
        first_hard_stop=first_hard_stop,
        path_points=len(path),
    )
    return audit, trace_rows


def _fmt_trigger(t: tuple[int, float, float, str] | None) -> str:
    if not t:
        return "-"
    ts_key, roi, max_roi, reason = t
    return f"{_dt_str(ts_key)} roi={_format_pct(roi)} max={_format_pct(max_roi)} {reason}"


def sell_attempts_for_lot(
    lot: Lot,
    rows: list[dict[str, Any]],
    end_ts: float,
) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        ts = _safe_float(row.get("ts"), 0.0)
        if str(row.get("symbol") or "") != lot.symbol:
            continue
        if ts < lot.entry_ts or ts > end_ts:
            continue
        if _row_side(row) != "SELL":
            continue
        if row.get("action") not in {"ORDER_PENDING", "ORDER_CANCELLED", "ORDER_FILLED"}:
            continue
        out.append(row)
    return sorted(out, key=lambda r: _safe_float(r.get("ts"), 0.0))


def main() -> None:
    args = parse_args()
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    cfg = StrategyConfig()
    core = StrategyCoreV0(cfg)

    day_start, day_end = _day_bounds(args.date)
    conn = psycopg2.connect(PG_DB_URL)
    try:
        rows = load_order_events_with_cancelled(conn, args.date, symbols)
        if not rows:
            print(f"没有找到 {args.date} {symbols or 'ALL'} 的 order_events。")
            return
        stock_paths = load_stock_paths(conn, day_start, day_end + args.future_mins * 60, symbols)
        pairs = build_lots(rows, stock_paths)
        if not pairs:
            print("没有匹配到 BUY ORDER_FILLED。")
            return
        target_symbols = sorted({lot.symbol for lot, _ in pairs})
        option_paths = load_option_paths(conn, day_start, day_end + args.future_mins * 60, target_symbols)
    finally:
        conn.close()

    print("## V0 Exit Guard PG Audit")
    print(f"date={args.date} symbols={','.join(symbols) or 'ALL'} lots={len(pairs)}")
    print(
        "thresholds: "
        f"LADDER_TIGHT={cfg.LADDER_TIGHT[:3]}... "
        f"NO_MOM={cfg.NO_MOMENTUM_MINS}m/{_format_pct(cfg.NO_MOMENTUM_MIN_MAX_ROI)} "
        f"STOP_LOSS={_format_pct(cfg.STOP_LOSS)} ABS={_format_pct(cfg.ABSOLUTE_STOP_LOSS)}"
    )

    audits: list[tuple[LotAudit, list[tuple[int, float, float, str]]]] = []
    pair_audits: list[tuple[Lot, dict[str, Any] | None, LotAudit, list[tuple[int, float, float, str]]]] = []
    for lot, exit_row in pairs:
        audit, trace_rows = audit_lot(core, lot, exit_row, option_paths, stock_paths, args.future_mins)
        audits.append((audit, trace_rows))
        pair_audits.append((lot, exit_row, audit, trace_rows))

    def suspicious(a: LotAudit) -> bool:
        if a.first_profit_exit:
            trigger_ts = a.first_profit_exit[0]
            actual_ts = 0 if a.actual_exit_dt == "OPEN/NO_CLOSE" else int(datetime.strptime(a.actual_exit_dt[:19], "%Y-%m-%d %H:%M:%S").replace(tzinfo=NY_TZ).timestamp())
            return actual_ts <= 0 or trigger_ts + 1 < actual_ts
        if a.first_no_momentum and not a.first_full_exit:
            return True
        return False

    ranked = sorted(
        pair_audits,
        key=lambda x: (
            suspicious(x[2]),
            x[2].peak_roi,
            abs(x[2].actual_exit_roi or 0.0),
        ),
        reverse=True,
    )

    print("\n## Lots")
    shown = 0
    for lot, exit_row, audit, trace_rows in ranked:
        if shown >= args.limit:
            break
        shown += 1
        print(
            f"\n[{shown}] {audit.symbol} {audit.option_side} entry={audit.entry_dt} "
            f"exit={audit.actual_exit_dt} hold={audit.hold_mins:.1f}m points={audit.path_points}"
        )
        print(
            f"    entry_px={audit.entry_price:.2f} exit_px="
            f"{audit.actual_exit_price if audit.actual_exit_price is not None else 'n/a'} "
            f"exit_roi={_format_pct(audit.actual_exit_roi) if audit.actual_exit_roi is not None else 'n/a'} "
            f"peak_roi={_format_pct(audit.peak_roi)} reason={audit.actual_exit_reason or '-'}"
        )
        print(f"    full_strategy_first : {_fmt_trigger(audit.first_full_exit)}")
        print(f"    profit_guard_first  : {_fmt_trigger(audit.first_profit_exit)}")
        print(f"    no_momentum_first   : {_fmt_trigger(audit.first_no_momentum)}")
        print(f"    stop_guard_first    : {_fmt_trigger(audit.first_hard_stop)}")
        actual_exit_ts = _safe_float((exit_row or {}).get("ts"), 0.0)
        attempt_end_ts = actual_exit_ts if actual_exit_ts > 0 else lot.entry_ts + args.future_mins * 60
        attempts = sell_attempts_for_lot(lot, rows, attempt_end_ts)
        filled_attempts = [r for r in attempts if r.get("action") == "ORDER_FILLED"]
        cancelled_attempts = [r for r in attempts if r.get("action") == "ORDER_CANCELLED"]
        print(
            f"    sell_attempts       : total={len(attempts)} filled={len(filled_attempts)} "
            f"cancelled={len(cancelled_attempts)}"
        )
        for row in attempts[:5]:
            print(
                f"      - {row.get('datetime_ny')} {row.get('action')} "
                f"price={_safe_float(row.get('price')):.2f} reason={_row_reason(row, None)}"
            )

        if args.show_path and (audit.first_profit_exit or audit.first_no_momentum):
            print("    path:")
            for ts_key, roi, max_roi, reason in trace_rows:
                mark = "*" if reason else " "
                print(f"      {mark} {_dt_str(ts_key)} roi={_format_pct(roi)} max={_format_pct(max_roi)} {reason}")

    profit_should = [a for a, _ in audits if a.first_profit_exit]
    no_mom_should = [a for a, _ in audits if a.first_no_momentum]
    print("\n## Summary")
    print(f"lots={len(audits)} profit_guard_triggered={len(profit_should)} no_momentum_triggered={len(no_mom_should)}")
    delayed_profit = [a for a in profit_should if a.first_full_exit and a.first_full_exit[0] == a.first_profit_exit[0]]
    print(f"profit_guard_visible_in_full_strategy={len(delayed_profit)}/{len(profit_should)}")
    print("说明: profit_guard_first 有值而实盘未在附近平仓，优先检查 st.max_roi 更新、exit 检查频率、以及 ALPHA_FRAME 是否持续包含该持仓标的。")


if __name__ == "__main__":
    main()

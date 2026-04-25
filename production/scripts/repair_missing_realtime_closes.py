#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Backfill synthetic CLOSE rows for symbols that are still open in trade_logs
but already flat at the broker, and optionally clean stale OMS state layers.

Purpose:
- Clear stale "open positions" reconstructed from OPEN/CLOSE logs
- Preserve an audit trail in Dashboard Recent Trade Logs
- Re-align symbol_state / Redis projections with broker truth on demand

Notes:
- This is still an audit repair, not broker-truth PnL recovery.
- Synthetic CLOSE rows use the outstanding average entry cost as repair price,
  so they neutralize stale open positions without fabricating realized PnL.
- Default mode is REALTIME and default behavior is dry-run.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import psycopg2
import redis


REPO_ROOT = Path(__file__).resolve().parents[2]
BASELINE_DIR = REPO_ROOT / "production" / "baseline"
if str(BASELINE_DIR) not in sys.path:
    sys.path.insert(0, str(BASELINE_DIR))

from config import NY_TZ, OMS_STATE_NAMESPACE, PG_DB_URL, REDIS_CFG  # noqa: E402
from orchestrator_order_state import namespaced_pending_orders_key  # noqa: E402
from orchestrator_state_manager import zero_position_state_row  # noqa: E402
from startup_state_hygiene import is_oms_heartbeat_fresh  # noqa: E402


EPS = 1e-9


@dataclass
class OpenState:
    symbol: str
    qty: float
    avg_cost: float
    last_open_ts: float


@dataclass
class RepairRow:
    ts: float
    datetime_ny: str
    symbol: str
    action: str
    qty: float
    price: float
    details_json: str


def _normalize_mode(value) -> str:
    text = str(value or "").strip().upper()
    return text if text in {"REALTIME", "REALTIME_DRY", "BACKTEST", "SHADOW"} else ""


def _trade_table_for_mode(mode: str) -> str:
    return "trade_logs" if _normalize_mode(mode) == "REALTIME" else "trade_logs_backtest"


def _parse_detail_mode(details_json: str) -> str:
    try:
        payload = json.loads(details_json or "{}")
    except Exception:
        return ""
    return _normalize_mode(payload.get("mode"))


def _ensure_trade_partition(conn, table_name: str, target_date: datetime) -> None:
    start_dt = NY_TZ.localize(datetime.combine(target_date.date(), datetime.min.time()))
    end_dt = start_dt + timedelta(days=1)
    start_ts = int(start_dt.timestamp())
    end_ts = int(end_dt.timestamp())
    suffix = target_date.strftime("%Y%m%d")
    partition_name = f"{table_name}_{suffix}"
    sql = f"""
        CREATE TABLE IF NOT EXISTS {partition_name}
        PARTITION OF {table_name}
        FOR VALUES FROM ({start_ts}) TO ({end_ts});
    """
    with conn.cursor() as cur:
        cur.execute(sql)


def _fetch_trade_rows(conn, table_name: str, start_ts: int, end_ts: int) -> List[dict]:
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT ts, symbol, action, qty, price, details_json
            FROM {table_name}
            WHERE ts >= %s AND ts < %s
            ORDER BY ts ASC
            """,
            (start_ts, end_ts),
        )
        rows = cur.fetchall()
    return [
        {
            "ts": float(ts or 0.0),
            "symbol": str(symbol or ""),
            "action": str(action or ""),
            "qty": float(qty or 0.0),
            "price": float(price or 0.0),
            "details_json": details_json or "",
        }
        for ts, symbol, action, qty, price, details_json in rows
    ]


def _build_open_states(rows: Iterable[dict], expected_mode: str, default_mode: str) -> Dict[str, OpenState]:
    states: Dict[str, OpenState] = {}
    current_mode = _normalize_mode(expected_mode)
    fallback_mode = _normalize_mode(default_mode)
    for row in rows:
        sym = str(row.get("symbol") or "").strip()
        if not sym:
            continue
        row_mode = _parse_detail_mode(row.get("details_json", "")) or fallback_mode
        if current_mode and row_mode != current_mode:
            continue
        action = str(row.get("action") or "").upper()
        qty = abs(float(row.get("qty", 0.0) or 0.0))
        price = float(row.get("price", 0.0) or 0.0)
        if qty <= EPS:
            continue
        if action == "OPEN":
            prev = states.get(sym)
            if prev is None:
                states[sym] = OpenState(symbol=sym, qty=qty, avg_cost=max(price, 0.0), last_open_ts=float(row["ts"]))
            else:
                new_qty = prev.qty + qty
                new_cost = prev.avg_cost
                if new_qty > EPS:
                    new_cost = ((prev.qty * prev.avg_cost) + (qty * max(price, 0.0))) / new_qty
                states[sym] = OpenState(symbol=sym, qty=new_qty, avg_cost=new_cost, last_open_ts=float(row["ts"]))
        elif action == "CLOSE":
            prev = states.get(sym)
            if prev is None:
                continue
            remaining = prev.qty - qty
            if remaining <= EPS:
                states.pop(sym, None)
            else:
                states[sym] = OpenState(symbol=sym, qty=remaining, avg_cost=prev.avg_cost, last_open_ts=prev.last_open_ts)
    return states


def _fetch_live_positions(expected_mode: str) -> Dict[str, float]:
    current_mode = _normalize_mode(expected_mode)
    out: Dict[str, float] = {}
    client = redis.Redis(
        **{k: v for k, v in REDIS_CFG.items() if k in ["host", "port", "db"]},
        decode_responses=False,
    )
    raw_map = client.hgetall("oms:live_positions") or {}
    for raw_sym, raw_val in raw_map.items():
        try:
            sym = raw_sym.decode("utf-8", errors="ignore") if isinstance(raw_sym, (bytes, bytearray)) else str(raw_sym)
            if sym == "____SYSTEM_CASH____":
                continue
            txt = raw_val.decode("utf-8", errors="ignore") if isinstance(raw_val, (bytes, bytearray)) else str(raw_val)
            state = json.loads(txt)
            payload_mode = _normalize_mode(state.get("mode") or state.get("engine_mode") or current_mode)
            if current_mode and payload_mode and payload_mode != current_mode:
                continue
            pos = int(state.get("position", state.get("pos", 0)) or 0)
            qty = abs(float(state.get("qty", 0.0) or 0.0))
            if pos == 0 or qty <= EPS:
                continue
            out[sym] = qty
        except Exception:
            continue
    return out


def _connect_ibkr_with_fallback(ib, host: str = "127.0.0.1", port: Optional[int] = None, preferred_client_ids=None):
    if preferred_client_ids is None:
        preferred_client_ids = [131, 132, 133, 134, 135]
    if port is None:
        try:
            from config import IBKR_PORT  # noqa: E402
            port = int(IBKR_PORT)
        except Exception:
            port = 7497

    tried = []
    last_err = ""
    for cid in preferred_client_ids:
        tried.append(int(cid))
        try:
            if ib.isConnected():
                ib.disconnect()
            ib.connect(host, int(port), clientId=int(cid), timeout=4)
            return True, int(cid), ""
        except Exception as e:
            last_err = str(e)
            continue
    return False, None, f"{last_err} (tried clientId={tried})"


def _fetch_ibkr_live_positions() -> Dict[str, float]:
    out: Dict[str, float] = {}
    try:
        import ib_insync
    except Exception as e:  # pragma: no cover
        raise RuntimeError(f"ib_insync unavailable: {e}") from e

    ib = ib_insync.IB()
    try:
        ok, _cid, err = _connect_ibkr_with_fallback(ib)
        if not ok:
            raise RuntimeError(f"IBKR connect failed: {err}")
        for pos in ib.positions():
            contract = getattr(pos, "contract", None)
            if contract is None:
                continue
            sec_type = str(getattr(contract, "secType", "") or "").upper()
            if sec_type != "OPT":
                continue
            sym = str(getattr(contract, "symbol", "") or "").strip().upper()
            qty = float(getattr(pos, "position", 0.0) or 0.0)
            if not sym or abs(qty) <= EPS:
                continue
            out[sym] = float(out.get(sym, 0.0) or 0.0) + abs(qty)
        return out
    finally:
        try:
            if ib.isConnected():
                ib.disconnect()
        except Exception:
            pass


def _build_repair_rows(
    open_states: Dict[str, OpenState],
    live_positions: Dict[str, float],
    *,
    target_mode: str,
    repair_ts: float,
    symbols: Optional[set[str]] = None,
) -> List[RepairRow]:
    repairs: List[RepairRow] = []
    counter = 0
    for sym in sorted(open_states):
        if symbols and sym not in symbols:
            continue
        state = open_states[sym]
        live_qty = float(live_positions.get(sym, 0.0) or 0.0)
        missing_qty = state.qty - live_qty
        if missing_qty <= EPS:
            continue
        ts = float(repair_ts) + (counter * 0.001)
        counter += 1
        dt_ny = datetime.fromtimestamp(ts, NY_TZ).strftime("%Y-%m-%d %H:%M:%S")
        strategy_note = {
            "reason": "MISSING_CLOSE_REPAIR",
            "repair_source": "oms:live_positions",
            "logged_open_qty": round(state.qty, 6),
            "live_qty": round(live_qty, 6),
            "repair_qty": round(missing_qty, 6),
            "repair_mode": target_mode,
            "repair_note": "Synthetic CLOSE inserted to clear stale OPEN residue after missing close logging.",
        }
        details = {
            "mode": target_mode,
            "fill_ratio": 1.0,
            "fill_duration": 0.0,
            "strategy_note": json.dumps(strategy_note, ensure_ascii=True),
            "synthetic_close_repair": True,
        }
        repairs.append(
            RepairRow(
                ts=ts,
                datetime_ny=dt_ny,
                symbol=sym,
                action="CLOSE",
                qty=round(missing_qty, 6),
                price=round(max(state.avg_cost, 0.01), 6),
                details_json=json.dumps(details, ensure_ascii=True),
            )
        )
    return repairs


def _insert_repairs(conn, table_name: str, repair_rows: List[RepairRow]) -> None:
    if not repair_rows:
        return
    target_dt = datetime.fromtimestamp(repair_rows[0].ts, NY_TZ)
    _ensure_trade_partition(conn, table_name, target_dt)
    with conn.cursor() as cur:
        cur.executemany(
            f"INSERT INTO {table_name} (ts, datetime_ny, symbol, action, qty, price, details_json) VALUES (%s,%s,%s,%s,%s,%s,%s)",
            [
                (row.ts, row.datetime_ny, row.symbol, row.action, row.qty, row.price, row.details_json)
                for row in repair_rows
            ],
        )


def _zero_symbol_state_rows(conn, symbols: Iterable[str], repair_ts: float) -> int:
    symbol_list = [str(s or "").strip().upper() for s in symbols if str(s or "").strip()]
    if not symbol_list:
        return 0
    with conn.cursor() as cur:
        cur.execute("SELECT to_regclass('public.symbol_state')")
        row = cur.fetchone()
        if not row or row[0] is None:
            return 0
        cur.execute(
            """
            SELECT symbol, data
            FROM symbol_state
            WHERE namespace = %s AND symbol = ANY(%s)
            """,
            (OMS_STATE_NAMESPACE, symbol_list),
        )
        rows = cur.fetchall()
        if not rows:
            return 0
        payloads = []
        for sym, data_txt in rows:
            try:
                state = json.loads(data_txt or "{}")
            except Exception:
                state = {}
            cleaned = zero_position_state_row(state)
            cleaned["updated_at"] = float(repair_ts)
            payloads.append((json.dumps(cleaned, ensure_ascii=True), float(repair_ts), str(sym)))
        with conn.cursor() as wcur:
            wcur.executemany(
                """
                UPDATE symbol_state
                SET data = %s, updated_at = %s
                WHERE namespace = %s AND symbol = %s
                """,
                [(data_txt, ts, OMS_STATE_NAMESPACE, sym) for data_txt, ts, sym in payloads],
            )
        return len(payloads)


def _mark_order_state_terminal(conn, symbols: Iterable[str], repair_ts: float) -> int:
    symbol_list = [str(s or "").strip().upper() for s in symbols if str(s or "").strip()]
    if not symbol_list:
        return 0
    with conn.cursor() as cur:
        cur.execute("SELECT to_regclass('public.oms_order_state')")
        row = cur.fetchone()
        if not row or row[0] is None:
            return 0
        cur.execute(
            """
            UPDATE oms_order_state
            SET status = 'REPAIRED',
                is_terminal = TRUE,
                updated_at = %s
            WHERE namespace = %s
              AND symbol = ANY(%s)
              AND is_terminal = FALSE
            """,
            (float(repair_ts), OMS_STATE_NAMESPACE, symbol_list),
        )
        return int(cur.rowcount or 0)


def _cleanup_redis_state(symbols: Iterable[str]) -> Dict[str, int]:
    symbol_list = [str(s or "").strip().upper() for s in symbols if str(s or "").strip()]
    stats = {"live_positions_removed": 0, "pending_orders_removed": 0}
    if not symbol_list:
        return stats
    client = redis.Redis(
        **{k: v for k, v in REDIS_CFG.items() if k in ["host", "port", "db"]},
        decode_responses=False,
    )
    live_fields = []
    raw_map = client.hgetall("oms:live_positions") or {}
    for raw_sym in raw_map.keys():
        sym = raw_sym.decode("utf-8", errors="ignore") if isinstance(raw_sym, (bytes, bytearray)) else str(raw_sym)
        if sym.upper() in symbol_list:
            live_fields.append(raw_sym)
    if live_fields:
        stats["live_positions_removed"] = int(client.hdel("oms:live_positions", *live_fields) or 0)

    pending_key = namespaced_pending_orders_key(OMS_STATE_NAMESPACE)
    raw_pending = client.hgetall(pending_key) or {}
    pending_fields = []
    for raw_key, raw_val in raw_pending.items():
        try:
            txt = raw_val.decode("utf-8", errors="ignore") if isinstance(raw_val, (bytes, bytearray)) else str(raw_val)
            payload = json.loads(txt)
            sym = str(payload.get("symbol", "") or "").strip().upper()
            if sym in symbol_list:
                pending_fields.append(raw_key)
        except Exception:
            continue
    if pending_fields:
        stats["pending_orders_removed"] = int(client.hdel(pending_key, *pending_fields) or 0)
    return stats


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repair missing CLOSE rows from stale realtime trade logs.")
    parser.add_argument("--mode", default="REALTIME", help="Trade mode to repair. Default: REALTIME")
    parser.add_argument("--date", default=datetime.now(NY_TZ).strftime("%Y-%m-%d"), help="NY date to inspect, e.g. 2026-04-24")
    parser.add_argument("--symbols", default="", help="Comma-separated symbol allowlist")
    parser.add_argument("--apply", action="store_true", help="Write synthetic CLOSE rows to PostgreSQL")
    parser.add_argument(
        "--source",
        default="ibkr",
        choices=["ibkr", "oms_live"],
        help="Ground truth source for current live positions. Default: ibkr",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    target_mode = _normalize_mode(args.mode) or "REALTIME"
    table_name = _trade_table_for_mode(target_mode)
    target_date = datetime.strptime(args.date, "%Y-%m-%d")
    start_dt = NY_TZ.localize(datetime.combine(target_date.date(), datetime.min.time()))
    start_ts = int(start_dt.timestamp())
    end_ts = int((start_dt + timedelta(days=1)).timestamp())
    symbol_filter = {s.strip().upper() for s in args.symbols.split(",") if s.strip()}

    conn = psycopg2.connect(PG_DB_URL)
    try:
        rows = _fetch_trade_rows(conn, table_name, start_ts, end_ts)
        open_states = _build_open_states(rows, target_mode, "REALTIME" if table_name == "trade_logs" else target_mode)
        if args.source == "ibkr":
            live_positions = _fetch_ibkr_live_positions()
            repair_source = "ibkr"
        else:
            live_positions = _fetch_live_positions(target_mode)
            repair_source = "oms:live_positions"
        repair_rows = _build_repair_rows(
            open_states,
            live_positions,
            target_mode=target_mode,
            repair_ts=time.time(),
            symbols=symbol_filter or None,
        )
        for row in repair_rows:
            try:
                details = json.loads(row.details_json)
                note = json.loads(details.get("strategy_note", "{}"))
                note["repair_source"] = repair_source
                details["strategy_note"] = json.dumps(note, ensure_ascii=True)
                row.details_json = json.dumps(details, ensure_ascii=True)
            except Exception:
                pass

        heartbeat_fresh = False
        try:
            redis_client = redis.Redis(
                **{k: v for k, v in REDIS_CFG.items() if k in ["host", "port", "db"]},
                decode_responses=False,
            )
            heartbeat_fresh = bool(is_oms_heartbeat_fresh(redis_client))
        except Exception:
            heartbeat_fresh = False

        print(f"mode={target_mode} table={table_name} date={args.date}")
        print(
            f"mode={target_mode} table={table_name} date={args.date} "
            f"source={repair_source} log_open_symbols={len(open_states)} "
            f"live_symbols={len(live_positions)} repair_candidates={len(repair_rows)}"
        )
        if heartbeat_fresh:
            print("WARN oms_heartbeat_fresh=1 ; a running OMS process may temporarily re-broadcast stale in-memory state until reconciliation catches up.")
        for row in repair_rows:
            detail = json.loads(row.details_json)
            note = json.loads(detail.get("strategy_note", "{}"))
            print(
                f"REPAIR {row.symbol} qty={row.qty:.4f} price={row.price:.4f} "
                f"log_open={note.get('logged_open_qty')} live={note.get('live_qty')}"
            )

        if not args.apply:
            print("dry-run only; re-run with --apply to insert synthetic CLOSE rows.")
            return 0

        _insert_repairs(conn, table_name, repair_rows)
        repaired_symbols = [row.symbol for row in repair_rows]
        pg_zeroed = _zero_symbol_state_rows(conn, repaired_symbols, repair_ts=time.time())
        order_rows = _mark_order_state_terminal(conn, repaired_symbols, repair_ts=time.time())
        conn.commit()
        redis_stats = _cleanup_redis_state(repaired_symbols)
        print(
            f"inserted {len(repair_rows)} synthetic CLOSE rows into {table_name}. "
            f"symbol_state_zeroed={pg_zeroed} active_order_rows_marked={order_rows} "
            f"redis_live_removed={redis_stats['live_positions_removed']} "
            f"redis_pending_removed={redis_stats['pending_orders_removed']}"
        )
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())

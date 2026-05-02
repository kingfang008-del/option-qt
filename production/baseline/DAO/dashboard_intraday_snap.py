#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Intraday performance + chart quotes for K-line WS (no Streamlit imports)."""

from __future__ import annotations

import json
import math
from datetime import datetime, time as dt_time
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
import redis

import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import (  # noqa: E402
    HASH_OPTION_SNAPSHOT,
    NY_TZ,
    OMS_STATE_NAMESPACE,
    PG_DB_URL,
    REDIS_CFG,
    RUN_MODE,
    TAG_TO_INDEX,
    TRADING_ENABLED,
)
from dashboard_cash_utils import (  # noqa: E402
    parse_oms_ledger_hash,
    parse_oms_live_cash_payload,
    select_remaining_cash,
)
from utils import serialization_utils as ser  # noqa: E402


def _normalize_trade_mode(value) -> str:
    mode = str(value or "").strip().upper()
    return mode if mode in {"REALTIME", "REALTIME_DRY", "BACKTEST", "SHADOW"} else ""


def _build_trade_mode_mask(df: pd.DataFrame, expected_mode: str) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=bool)
    current_mode = _normalize_trade_mode(expected_mode)
    if not current_mode:
        return pd.Series(True, index=df.index)
    mode_series = df.get("mode", pd.Series(index=df.index, dtype=object)).map(_normalize_trade_mode)
    mask = mode_series.eq(current_mode)
    if current_mode == "REALTIME" and "source_table" in df.columns:
        legacy_live_mask = mode_series.eq("") & df["source_table"].astype(str).eq("trade_logs")
        mask = mask | legacy_live_mask
    return mask.fillna(False)


def _redis_client():
    try:
        return redis.Redis(
            **{k: v for k, v in REDIS_CFG.items() if k in ["host", "port", "db"]},
            decode_responses=False,
        )
    except Exception:
        return None


def _fetch_latest_mock_cash():
    if not TRADING_ENABLED:
        return None
    try:
        conn = psycopg2.connect(PG_DB_URL)
        c = conn.cursor()
        c.execute(
            "SELECT data, updated_at FROM symbol_state WHERE namespace = %s AND symbol = '_GLOBAL_STATE_'",
            (OMS_STATE_NAMESPACE,),
        )
        row = c.fetchone()
        conn.close()
        if row:
            data = json.loads(row[0])
            updated_at = row[1]
            ts_ny = datetime.fromtimestamp(float(updated_at), NY_TZ).date()
            if ts_ny != datetime.now(NY_TZ).date():
                return None
            return float(data.get("mock_cash", 0.0))
    except Exception:
        pass
    return None


def _fetch_live_oms_cash(max_age_sec: float = 120.0):
    expected_modes = [RUN_MODE]
    try:
        r = _redis_client()
        if not r:
            return None, None
        try:
            ledger = r.hgetall("meta:oms_ledger") or {}
            cash, source = parse_oms_ledger_hash(
                ledger,
                max_age_sec=max_age_sec,
                allow_stale=False,
                expected_modes=expected_modes,
            )
            if cash is not None:
                return cash, source
        except Exception:
            pass
        raw = r.hget("oms:live_positions", "____SYSTEM_CASH____")
        cash, source = parse_oms_live_cash_payload(
            raw,
            max_age_sec=max_age_sec,
            allow_stale=False,
            expected_modes=expected_modes,
        )
        if cash is not None:
            return cash, source
    except Exception:
        pass
    return None, None


def fetch_live_oms_positions_map(
    max_age_sec: float = 120.0,
):
    """Mirror dashboard fetch_live_oms_positions (minimal)."""

    def _valid_bucket_tag(raw_tag: str) -> str:
        tag = str(raw_tag or "").strip().upper()
        return tag if tag in TAG_TO_INDEX else ""

    positions = {}
    meta_fresh = False
    try:
        r = _redis_client()
        if not r:
            return positions, meta_fresh
        ledger = r.hgetall("meta:oms_ledger") or {}
        cash_raw = r.hget("oms:live_positions", "____SYSTEM_CASH____")
        ledger_cash, _ = parse_oms_ledger_hash(
            ledger,
            max_age_sec=max_age_sec,
            allow_stale=False,
            expected_modes=[RUN_MODE],
        )
        live_cash, _ = parse_oms_live_cash_payload(
            cash_raw,
            max_age_sec=max_age_sec,
            allow_stale=False,
            expected_modes=[RUN_MODE],
        )
        if ledger_cash is None and live_cash is None:
            return positions, meta_fresh
        meta_fresh = True

        raw_map = r.hgetall("oms:live_positions") or {}
        for raw_sym, raw_val in raw_map.items():
            try:
                sym = raw_sym.decode("utf-8", errors="ignore") if isinstance(raw_sym, (bytes, bytearray)) else str(raw_sym)
                if sym == "____SYSTEM_CASH____":
                    continue
                txt = raw_val.decode("utf-8", errors="ignore") if isinstance(raw_val, (bytes, bytearray)) else str(raw_val)
                state = json.loads(txt)
                pos = int(state.get("position", state.get("pos", 0)) or 0)
                qty = float(state.get("qty", 0.0) or 0.0)
                cost = float(state.get("entry_price", state.get("price", 0.0)) or 0.0)
                stock = float(state.get("entry_stock", state.get("stock", 0.0)) or 0.0)
                if pos == 0 or qty <= 0:
                    continue
                if not bool(state.get("open_fill_confirmed", False)):
                    continue
                opt_type = str(state.get("opt_type", "") or "").strip().lower()
                positions[sym] = {
                    "position": pos,
                    "qty": qty,
                    "cost": cost if math.isfinite(cost) and cost > 0 else 0.0,
                    "stock": stock if math.isfinite(stock) and stock > 0 else 0.0,
                    "tag": _valid_bucket_tag(state.get("tag", "")),
                    "opt_type": opt_type,
                    "contract_id": str(state.get("contract_id", "") or ""),
                    "last_opt_price": float(state.get("last_opt_price", 0.0) or 0.0),
                    "entry_ts": float(state.get("entry_ts", 0.0) or 0.0),
                }
            except Exception:
                continue
    except Exception:
        return positions, False
    return positions, meta_fresh


def _fetch_latest_option_snapshot(symbol: str):
    sym = str(symbol or "").strip().upper()
    if not sym:
        return [], []
    try:
        rds = _redis_client()
        if rds:
            raw = rds.hget(HASH_OPTION_SNAPSHOT, sym)
            if raw:
                snap = ser.unpack(raw)
                if isinstance(snap, dict):
                    b = snap.get("buckets")
                    c = snap.get("contracts", [])
                    return (b if isinstance(b, list) else []), (c if isinstance(c, list) else [])
                if isinstance(snap, list):
                    return snap, []
    except Exception:
        pass

    try:
        conn = psycopg2.connect(PG_DB_URL)
        c = conn.cursor()
        c.execute(
            "SELECT buckets_json FROM option_snapshots_1m WHERE symbol=%s ORDER BY ts DESC LIMIT 1",
            (sym,),
        )
        row = c.fetchone()
        conn.close()
        if row:
            snap = row[0]
            if isinstance(snap, str):
                snap = json.loads(snap)
            if isinstance(snap, dict):
                return snap.get("buckets", []), snap.get("contracts", [])
            if isinstance(snap, list):
                return snap, []
    except Exception:
        pass
    return [], []


def get_bucket_quote(symbol: str, tag: str) -> dict | None:
    idx = TAG_TO_INDEX.get(tag, -1)
    if idx < 0:
        return None
    buckets, contracts = _fetch_latest_option_snapshot(symbol)
    if not buckets or len(buckets) <= idx:
        return None
    row = buckets[idx]
    if not isinstance(row, (list, tuple)) or len(row) < 10:
        return None
    bid = float(row[8] or 0.0) if len(row) > 8 else 0.0
    ask = float(row[9] or 0.0) if len(row) > 9 else 0.0
    last = float(row[0] or 0.0)
    mid = (bid + ask) / 2.0 if bid > 0 and ask > 0 and ask >= bid else (last if last > 0 else 0.0)
    strike = float(row[5] or 0.0) if len(row) > 5 else 0.0
    contract_txt = contracts[idx] if contracts and len(contracts) > idx else ""
    return {
        "bucket_idx": idx,
        "bid": bid,
        "ask": ask,
        "last": last,
        "mid": mid,
        "strike": strike,
        "contract_text": contract_txt,
    }


def fetch_underlying_option_quotes(symbol: str) -> dict:
    out = {}
    for tag in ("CALL_ATM", "CALL_OTM", "PUT_ATM", "PUT_OTM"):
        q = get_bucket_quote(symbol, tag)
        if q:
            out[tag] = q
        else:
            out[tag] = {}
    return out


def _pick_live_price(_last: float, _bid: float, _ask: float) -> float:
    _last = float(_last or 0.0)
    _bid = float(_bid or 0.0)
    _ask = float(_ask or 0.0)
    if _bid > 0.01 and _ask > 0.01 and _ask >= _bid:
        return (_bid + _ask) / 2.0
    if _bid > 0.01:
        return _bid
    if _ask > 0.01:
        return _ask
    if _last > 0.01:
        return _last
    return 0.0


def _effective_bucket_tag(pos: dict) -> str:
    tag_val = str(pos.get("tag", "") or "").strip().upper()
    if tag_val in TAG_TO_INDEX:
        return tag_val
    opt_type = str(pos.get("opt_type", "") or "").strip().lower()
    direction = int(pos.get("position", 0) or 0)
    if opt_type == "call" or direction == 1:
        return "CALL_ATM"
    if opt_type == "put" or direction == -1:
        return "PUT_ATM"
    return ""


def load_today_trades_dataframe(run_mode_hint: str | None = None) -> tuple[pd.DataFrame, str]:
    """Today's merged trade dataframe for current RUN_MODE (+ dual-table merge)."""
    query_date = datetime.now(NY_TZ).date()
    target_start_dt = datetime.combine(query_date, dt_time(0, 0, 0))
    target_start_ts = int(NY_TZ.localize(target_start_dt).timestamp())
    target_end_ts = target_start_ts + 86400

    mode_key = run_mode_hint or RUN_MODE
    current_mode = str(mode_key).strip().upper()
    target_table = "trade_logs" if current_mode == "REALTIME" else "trade_logs_backtest"
    secondary_table = "trade_logs_backtest" if target_table == "trade_logs" else "trade_logs"

    conn = psycopg2.connect(PG_DB_URL)
    sql_primary = f"SELECT * FROM {target_table} WHERE ts >= {target_start_ts} AND ts < {target_end_ts}"
    sql_secondary = f"SELECT * FROM {secondary_table} WHERE ts >= {target_start_ts} AND ts < {target_end_ts}"
    try:
        df_primary = pd.read_sql(sql_primary, conn)
    except Exception:
        df_primary = pd.DataFrame()
    try:
        df_secondary = pd.read_sql(sql_secondary, conn)
    except Exception:
        df_secondary = pd.DataFrame()

    conn.close()

    if not df_primary.empty:
        df_primary["source_table"] = target_table
    if not df_secondary.empty:
        df_secondary["source_table"] = secondary_table
    if df_primary.empty and df_secondary.empty:
        df_all = pd.DataFrame()
    elif df_secondary.empty:
        df_all = df_primary
    elif df_primary.empty:
        df_all = df_secondary
    else:
        df_all = pd.concat([df_primary, df_secondary], ignore_index=True)
        df_all = df_all.drop_duplicates(
            subset=["ts", "symbol", "action", "qty", "price"],
            keep="last",
        )
    if not df_all.empty:
        df_all = df_all.sort_values(by="ts", ascending=True)
    return df_all, current_mode


def build_intraday_snapshot() -> dict:
    """Portfolio-wide metrics + open rows + chart quotes keyed by RUN_MODE."""
    snapshot_ts = float(datetime.now(NY_TZ).timestamp())

    empty = {
        "type": "intraday",
        "ts": snapshot_ts,
        "closed_pnl": 0.0,
        "unrealized_pnl": 0.0,
        "total_market_value": 0.0,
        "remaining_cash": 0.0,
        "cash_source": "",
        "win_rate": 0.0,
        "wins": 0,
        "losses": 0,
        "open_symbols": 0,
        "rows": [],
        "quotes": {},
        "chart_position": None,
        "chart_symbol": "",
        "positions_by_symbol": {},
        "error": "",
    }

    try:
        df_all, current_mode = load_today_trades_dataframe()
    except Exception as exc:
        empty["error"] = str(exc)
        return empty

    if df_all.empty:
        live_cash, live_src = _fetch_live_oms_cash()
        display_cash, cash_src = select_remaining_cash(
            live_cash=live_cash,
            live_cash_source=live_src,
            log_cash=None,
            latest_cash=_fetch_latest_mock_cash(),
            run_mode=RUN_MODE,
        )
        empty["remaining_cash"] = float(display_cash)
        empty["cash_source"] = str(cash_src)
        return empty

    def parse_val(row, key, default=np.nan):
        try:
            d = json.loads(row)
            return d.get(key, default)
        except Exception:
            return default

    for k in [
        "pnl",
        "roi",
        "stock_price",
        "entry_stock",
        "mode",
        "account_cash",
        "alpha_label_ts",
        "alpha_available_ts",
        "order_submit_ts",
        "fill_ts",
        "alpha_to_submit_ms",
        "submit_to_fill_ms",
        "alpha_to_fill_ms",
        "fill_duration",
        "fill_ratio",
    ]:
        df_all[k] = df_all["details_json"].apply(lambda r: parse_val(r, k))

    def extract_note(row, key):
        try:
            d = json.loads(row)
            note_str = d.get("strategy_note", "{}")
            note = json.loads(note_str)
            return note.get(key, "")
        except Exception:
            return ""

    df_all["alpha"] = df_all["details_json"].apply(lambda r: extract_note(r, "alpha"))
    df_all["reason"] = df_all["details_json"].apply(lambda r: extract_note(r, "reason"))
    df_all["tag"] = df_all["details_json"].apply(lambda r: extract_note(r, "tag"))
    df_all = df_all[_build_trade_mode_mask(df_all, RUN_MODE)].copy()

    open_positions = {}
    closed_pnl = 0.0
    wins, losses = 0, 0

    for _, row in df_all.iterrows():
        sym = row["symbol"]
        qty = float(row.get("qty", 0))
        price = float(row.get("price", 0))
        action = row.get("action", "")
        tag = row.get("tag", "")

        if action == "OPEN":
            if sym not in open_positions:
                open_positions[sym] = {"qty": 0, "cost": 0.0, "tag": tag}
            old_qty = open_positions[sym]["qty"]
            old_cost = open_positions[sym]["cost"]
            new_qty = old_qty + qty
            open_positions[sym]["cost"] = (old_qty * old_cost + qty * price) / new_qty if new_qty > 0 else 0
            open_positions[sym]["qty"] = new_qty
            if tag:
                open_positions[sym]["tag"] = tag
        elif action == "CLOSE":
            if sym in open_positions:
                open_positions[sym]["qty"] -= qty
                if open_positions[sym]["qty"] <= 0:
                    del open_positions[sym]
            pnl_val = float(row.get("pnl", 0) if pd.notna(row.get("pnl")) else 0)
            closed_pnl += pnl_val
            if pnl_val > 0:
                wins += 1
            elif pnl_val < 0:
                losses += 1

    log_open_positions = dict(open_positions)
    if current_mode.startswith("REALTIME"):
        live_open, fresh = fetch_live_oms_positions_map()
        if fresh:
            open_positions = live_open
            for sym, pos in open_positions.items():
                log_pos = log_open_positions.get(sym, {})
                if not pos.get("tag") and log_pos.get("tag"):
                    pos["tag"] = log_pos.get("tag", "")
                if pos.get("cost", 0.0) <= 0 and float(log_pos.get("cost", 0.0) or 0.0) > 0:
                    pos["cost"] = float(log_pos.get("cost", 0.0) or 0.0)

    for _sym, pos in open_positions.items():
        if int(pos.get("position", 0) or 0) == 0:
            tg = str(pos.get("tag", "") or "").strip().upper()
            if tg.startswith("CALL"):
                pos["position"] = 1
            elif tg.startswith("PUT"):
                pos["position"] = -1

    r_live = _redis_client()
    unrealized_pnl = 0.0
    total_market_value = 0.0
    rows_out = []

    for sym, pos in open_positions.items():
        live_price = float(pos.get("last_opt_price", 0.0) or 0.0)
        if live_price <= 0:
            live_price = float(pos["cost"])
        tag = _effective_bucket_tag(pos)
        found_live = False

        if r_live:
            try:
                raw = r_live.hget(HASH_OPTION_SNAPSHOT, sym)
                if raw:
                    snap = ser.unpack(raw)
                    buckets = snap.get("buckets", snap) if isinstance(snap, dict) else snap
                    if not isinstance(buckets, list):
                        buckets = []
                    idx = TAG_TO_INDEX.get(tag, -1)
                    if idx != -1 and len(buckets) > idx:
                        bucket = buckets[idx]
                        _last = float(bucket[0])
                        _bid = float(bucket[8]) if len(bucket) > 8 else 0.0
                        _ask = float(bucket[9]) if len(bucket) > 9 else 0.0
                        picked = _pick_live_price(_last, _bid, _ask)
                        if picked > 0.01:
                            live_price = picked
                            found_live = True
            except Exception:
                pass

        if not found_live:
            try:
                pg_conn = psycopg2.connect(PG_DB_URL)
                c_snap = pg_conn.cursor()
                c_snap.execute(
                    "SELECT buckets_json FROM option_snapshots_1m WHERE symbol=%s ORDER BY ts DESC LIMIT 1",
                    (sym,),
                )
                row_snap = c_snap.fetchone()
                if row_snap:
                    json_val = row_snap[0]
                    snap = json_val if isinstance(json_val, dict) else json.loads(json_val) if isinstance(json_val, str) else json_val
                    buckets = snap.get("buckets", snap) if isinstance(snap, dict) else snap
                    idx = TAG_TO_INDEX.get(tag, -1)
                    if idx != -1 and len(buckets) > idx:
                        bucket = buckets[idx]
                        _last = float(bucket[0])
                        _bid = float(bucket[8]) if len(bucket) > 8 else 0.0
                        _ask = float(bucket[9]) if len(bucket) > 9 else 0.0
                        picked = _pick_live_price(_last, _bid, _ask)
                        if picked > 0.01:
                            live_price = picked
                pg_conn.close()
            except Exception:
                pass

        if pos["cost"] > 0:
            price_deviation = abs(live_price - pos["cost"]) / pos["cost"]
            if price_deviation > 5.0:
                live_price = pos["cost"]

        qty_c = float(pos["qty"])
        paper_pnl_sym = (live_price - pos["cost"]) * qty_c * 100
        unrealized_pnl += paper_pnl_sym
        market_value_sym = live_price * qty_c * 100
        total_market_value += market_value_sym

        roi_pct = ((live_price / pos["cost"]) - 1) * 100 if pos["cost"] > 0 else 0.0
        rows_out.append({
            "symbol": sym,
            "tag": tag,
            "qty": qty_c,
            "cost": round(float(pos["cost"]), 2),
            "live": round(float(live_price), 2),
            "mv": round(float(market_value_sym), 2),
            "pnl": round(float(paper_pnl_sym), 2),
            "roi_pct": round(float(roi_pct), 2),
            "position": int(pos.get("position", 0) or 0),
            "contract_id": str(pos.get("contract_id", "") or ""),
            "stock": float(pos.get("stock", 0.0) or 0.0),
        })

    latest_cash = _fetch_latest_mock_cash()
    live_cash, live_src = _fetch_live_oms_cash()
    log_cash = None
    try:
        if "account_cash" in df_all.columns:
            _cash_series = pd.to_numeric(df_all["account_cash"], errors="coerce").dropna()
            if not _cash_series.empty:
                log_cash = float(_cash_series.iloc[-1])
    except Exception:
        log_cash = None

    display_cash, cash_src = select_remaining_cash(
        live_cash=live_cash,
        live_cash_source=live_src,
        log_cash=log_cash,
        latest_cash=latest_cash,
        run_mode=RUN_MODE,
    )

    win_rate = (wins / (wins + losses) * 100) if (wins + losses) > 0 else 0.0

    pos_by_sym = {}
    for k, v in open_positions.items():
        if isinstance(v, dict):
            pos_by_sym[str(k).upper()] = {
                "position": int(v.get("position", 0) or 0),
                "qty": float(v.get("qty", 0.0) or 0.0),
                "cost": float(v.get("cost", 0.0) or 0.0),
                "stock": float(v.get("stock", 0.0) or 0.0),
                "tag": str(v.get("tag", "") or ""),
                "contract_id": str(v.get("contract_id", "") or ""),
                "last_opt_price": float(v.get("last_opt_price", 0.0) or 0.0),
                "opt_type": str(v.get("opt_type", "") or ""),
            }

    out = {
        "type": "intraday",
        "ts": snapshot_ts,
        "closed_pnl": float(closed_pnl),
        "unrealized_pnl": float(unrealized_pnl),
        "total_market_value": float(total_market_value),
        "remaining_cash": float(display_cash),
        "cash_source": str(cash_src),
        "win_rate": float(win_rate),
        "wins": int(wins),
        "losses": int(losses),
        "open_symbols": int(len(open_positions)),
        "rows": rows_out,
        "quotes": {},
        "chart_position": None,
        "chart_symbol": "",
        "positions_by_symbol": pos_by_sym,
        "run_mode": str(current_mode),
        "live_oms_cash": float(live_cash) if live_cash is not None else None,
        "error": "",
    }
    return out


def augment_intraday_for_chart(base: dict, chart_symbol: str) -> dict:
    """Attach quotes + chart leg position (from merged open book)."""
    sym = str(chart_symbol or "").strip().upper()
    out = dict(base)
    out["chart_symbol"] = sym
    if not sym:
        out["quotes"] = {}
        out["chart_position"] = None
        return out
    out["quotes"] = fetch_underlying_option_quotes(sym)
    pmap = out.get("positions_by_symbol") or {}
    ps = pmap.get(sym)
    out["chart_position"] = ps if ps else None
    return out

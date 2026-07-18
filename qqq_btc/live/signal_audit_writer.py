#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
OMS 入场决策 audit CSV —— dry-run 与 strict replay 信号对拍。

默认按美东交易日写入:
  ~/quant_project/shadow/signals_YYYY-MM-DD.csv

环境变量:
  QQQ_BTC_SIGNAL_AUDIT=0   关闭
  QQQ_BTC_SIGNAL_AUDIT_DIR  目录(默认 ~/quant_project/shadow)
"""
from __future__ import annotations

import csv
import logging
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from pytz import timezone

logger = logging.getLogger("qqq_btc.live.signal_audit")

_LOCK = threading.Lock()
_NY = timezone("America/New_York")
_HEADER = (
    "ts",
    "timestamp",
    "symbol",
    "session_bar",
    "kind",
    "decision",
    "leg",
    "edge",
    "threshold",
    "call_edge",
    "put_edge",
    "net_edge_raw",
    "net_edge_q10",
    "spread_pct",
    "trend_fit_ret_30m",
    "spot_close",
    "vwap_log_return",
    "spot_ret_15bar",
    "vix_ret_15bar",
    "vix_level",
    "dyn_threshold",
    "put_dyn_threshold",
    "block_reason",
    "mode",
)


def signal_audit_enabled() -> bool:
    if os.environ.get("QQQ_BTC_SIGNAL_AUDIT", "1").strip().lower() in ("0", "false", "no", "off"):
        return False
    return os.environ.get("QQQ_BTC_LIVE", "").strip().lower() in ("1", "true", "yes", "on")


def default_audit_dir() -> Path:
    raw = os.environ.get("QQQ_BTC_SIGNAL_AUDIT_DIR", "").strip()
    if raw:
        return Path(raw).expanduser()
    return Path.home() / "quant_project" / "shadow"


def audit_path_for_day(day_key: str) -> Path:
    return default_audit_dir() / f"signals_{day_key}.csv"


def _day_key_from_ts(ts: float) -> str:
    dt = datetime.fromtimestamp(float(ts), tz=_NY)
    return dt.strftime("%Y-%m-%d")


def append_signal_audit_row(row: Dict[str, Any], *, day_key: Optional[str] = None) -> None:
    if not signal_audit_enabled():
        return
    ts = row.get("ts")
    if day_key is None:
        try:
            day_key = _day_key_from_ts(float(ts or 0.0))
        except (TypeError, ValueError):
            day_key = datetime.now(tz=_NY).strftime("%Y-%m-%d")
    path = audit_path_for_day(day_key)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = {k: row.get(k, "") for k in _HEADER}
    with _LOCK:
        write_header = not path.exists() or path.stat().st_size == 0
        with path.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=_HEADER)
            if write_header:
                w.writeheader()
            w.writerow(line)


def record_entry_signal_audit(
    *,
    ctx: dict,
    decision: Optional[Any] = None,
    block_reason: str = "",
    session_bar: int = 0,
    dyn_threshold: Optional[float] = None,
    put_dyn_threshold: Optional[float] = None,
    mode: str = "",
) -> Optional[Dict[str, Any]]:
    """记录一次 decide_entry 评估(PASS → kind=ENTER, BLOCK → kind=BLOCK)。"""
    if not signal_audit_enabled():
        return None

    curr_ts = float(ctx.get("curr_ts", 0.0) or 0.0)
    sym = str(ctx.get("symbol", "QQQ") or "QQQ")
    call_edge = float(ctx.get("call_edge", ctx.get("net_edge_raw", 0.0)) or 0.0)
    put_edge = float(ctx.get("put_edge", 0.0) or 0.0)
    edge_raw = float(ctx.get("net_edge_raw", ctx.get("alpha_z", 0.0)) or 0.0)
    q10 = ctx.get("net_edge_q10")
    try:
        q10_f = float(q10) if q10 is not None else ""
    except (TypeError, ValueError):
        q10_f = ""

    bid = float(ctx.get("bid", 0.0) or 0.0)
    ask = float(ctx.get("ask", 0.0) or 0.0)
    mid = float(ctx.get("curr_price", 0.0) or 0.0)
    spread_pct = (ask - bid) / mid if mid > 0.01 and ask >= bid > 0 else ""
    # 优先记录腿别 spread / trend，便于对拍 offline put_trend / put_spread
    put_sp = ctx.get("put_spread_pct")
    call_sp = ctx.get("call_spread_pct")
    try:
        if decision is not None and str(getattr(decision, "leg", "")).upper() == "PUT" and put_sp is not None:
            spread_pct = float(put_sp)
        elif call_sp is not None and (spread_pct == "" or spread_pct is None):
            spread_pct = float(call_sp)
    except (TypeError, ValueError):
        pass
    trend_v = ctx.get("trend_fit_ret_30m", "")
    try:
        trend_v = float(trend_v) if trend_v != "" and trend_v is not None else ""
    except (TypeError, ValueError):
        trend_v = ""

    if decision is not None:
        leg = str(getattr(decision, "leg", "") or "")
        edge = float(getattr(decision, "edge", 0.0) or 0.0)
        threshold = float(getattr(decision, "threshold", 0.0) or 0.0)
        kind = "ENTER"
        decision_flag = "PASS"
        block_reason = ""
    else:
        leg = ""
        edge = ""
        threshold = ""
        kind = "BLOCK"
        decision_flag = "BLOCK"

    try:
        ts_iso = datetime.fromtimestamp(curr_ts, tz=_NY).isoformat() if curr_ts > 0 else ""
    except (OSError, ValueError):
        ts_iso = ""

    row = {
        "ts": curr_ts if curr_ts > 0 else "",
        "timestamp": ts_iso,
        "symbol": sym,
        "session_bar": session_bar,
        "kind": kind,
        "decision": decision_flag,
        "leg": leg,
        "edge": edge,
        "threshold": threshold,
        "call_edge": call_edge,
        "put_edge": put_edge,
        "net_edge_raw": edge_raw,
        "net_edge_q10": q10_f,
        "spread_pct": spread_pct,
        "trend_fit_ret_30m": trend_v,
        "spot_close": ctx.get("spot_close", ""),
        "vwap_log_return": ctx.get("vwap_log_return", ""),
        "spot_ret_15bar": ctx.get("spot_ret_15bar", ""),
        "vix_ret_15bar": ctx.get("vix_ret_15bar", ""),
        "vix_level": ctx.get("vix_level", ""),
        "dyn_threshold": dyn_threshold if dyn_threshold is not None else "",
        "put_dyn_threshold": put_dyn_threshold if put_dyn_threshold is not None else "",
        "block_reason": block_reason,
        "mode": mode,
    }
    append_signal_audit_row(row, day_key=_day_key_from_ts(curr_ts) if curr_ts > 0 else None)
    if decision is not None:
        logger.info(
            "signal_audit PASS %s sb=%s %s edge=%.4f th=%.4f → %s",
            sym,
            session_bar,
            leg,
            float(edge),
            float(threshold),
            audit_path_for_day(_day_key_from_ts(curr_ts)),
        )
    return row

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dashboard 手动期权建仓：与 orchestrator_execution 入场侧一致的
「spread 内 probe + 约 3s fast requote」限价逻辑（同步 / ib_insync）。
"""

from __future__ import annotations

import math
import os
import time
from typing import Any, Tuple

from ib_insync import LimitOrder, Trade


def dashboard_entry_fast_params() -> dict[str, Any]:
    """与 strategy_config0 / orchestrator 默认一致，可用环境变量覆盖。"""
    def _f(name: str, default: float) -> float:
        try:
            return float(os.environ.get(name, str(default)) or default)
        except Exception:
            return float(default)

    enabled = os.environ.get("DASHBOARD_ENTRY_FAST_REQUOTE", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )
    return {
        "enabled": bool(enabled),
        "interval": max(0.05, _f("ENTRY_FAST_REQUOTE_INTERVAL_SECONDS", 0.40)),
        "window": max(_f("ENTRY_FAST_REQUOTE_MAX_SECONDS", 3.0), 0.25),
        "cancel_settle": max(0.0, _f("ENTRY_FAST_REQUOTE_CANCEL_SETTLE_SECONDS", 0.20)),
    }


def _entry_cap_price(entry_ref: float) -> float:
    from config import ENTRY_MAX_REQUOTE_SLIPPAGE_PCT

    ref = float(entry_ref or 0.0)
    if ref <= 0.0:
        return float("inf")
    cap_pct = max(0.0, float(ENTRY_MAX_REQUOTE_SLIPPAGE_PCT))
    return ref * (1.0 + cap_pct)


def _get_entry_limit_price(
    bid: float,
    ask: float,
    base_price: float,
    attempt_no: int,
    limit_buffer_entry: float,
) -> float:
    """Mirror OrchestratorExecution._get_entry_limit_price (meta bid/ask path + fallback)."""
    bid = float(bid or 0.0)
    ask = float(ask or 0.0)
    base_price = float(base_price or 0.0)
    limit_buffer_entry = float(limit_buffer_entry or 1.0)

    if bid > 0.0 and ask > 0.0 and ask >= bid:
        mid = (bid + ask) / 2.0
        spread = max(ask - bid, 0.0)
        inside_spread_probe = min(0.20 + 0.12 * max(int(attempt_no), 0), 0.45)
        raw_candidate = mid + spread * inside_spread_probe
        bid_tick = math.floor(bid * 100.0) / 100.0
        ask_tick = math.ceil(ask * 100.0) / 100.0
        ask_minus_tick = max(round(ask_tick - 0.01, 2), round(bid_tick, 2))
        if ask_minus_tick <= 0.0:
            return 0.0
        candidate = math.floor(raw_candidate * 100.0) / 100.0
        candidate = max(round(bid_tick, 2), candidate)
        candidate = min(candidate, ask_minus_tick)
        return round(candidate, 2)

    if base_price <= 0.0:
        return 0.0
    fallback_buf = max(limit_buffer_entry, 1.0)
    requote_step = 0.005 * max(int(attempt_no), 0)
    return round(base_price * (fallback_buf + requote_step), 2)


def _next_entry_requote_price(
    prev_limit_price: float,
    orch_attempt_no: int,
    cap_price: float,
    bid: float,
    ask: float,
    limit_buffer_entry: float,
) -> float:
    """Mirror OrchestratorExecution._next_entry_requote_price (live bid/ask branch)."""
    from config import ENTRY_REQUOTE_STEP_CAP_PCT

    prev_limit_price = float(prev_limit_price or 0.0)
    if prev_limit_price <= 0:
        return 0.0

    if bid > 0.0 and ask > 0.0:
        candidate = _get_entry_limit_price(
            bid, ask, (bid + ask) / 2.0, attempt_no=0, limit_buffer_entry=limit_buffer_entry
        )
    else:
        candidate = _get_entry_limit_price(
            0.0, 0.0, prev_limit_price, attempt_no=orch_attempt_no, limit_buffer_entry=limit_buffer_entry
        )

    if candidate < 0.05:
        return 0.0
    if candidate > cap_price:
        return 0.0

    step_cap_pct = max(0.0, float(ENTRY_REQUOTE_STEP_CAP_PCT))
    step_cap_price = prev_limit_price * (1.0 + step_cap_pct)
    capped = min(candidate, step_cap_price, cap_price)
    return float(capped)


def _ticker_bid_ask(ib, contract) -> Tuple[float, float]:
    try:
        t = ib.ticker(contract)
        if t is None:
            return 0.0, 0.0
        bid = float(getattr(t, "bid", 0.0) or 0.0)
        ask = float(getattr(t, "ask", 0.0) or 0.0)
        if bid > 0.0 and ask > 0.0 and ask >= bid:
            return bid, ask
    except Exception:
        pass
    return 0.0, 0.0


def _await_cancel_settle(ib, trade: Trade, max_wait: float) -> None:
    wait = max(0.0, float(max_wait or 0.0))
    if wait <= 0:
        return
    deadline = time.time() + wait
    terminal = ("Filled", "Cancelled", "ApiCancelled", "Inactive")
    ib.sleep(min(0.025, wait))
    st = str(getattr(getattr(trade, "orderStatus", None), "status", "") or "")
    if st in terminal:
        return
    while time.time() < deadline:
        st = str(getattr(getattr(trade, "orderStatus", None), "status", "") or "")
        if st in terminal:
            return
        ib.sleep(min(0.05, max(deadline - time.time(), 0.01)))


def place_buy_option_with_entry_requotes(
    ib,
    contract,
    qty: int,
    *,
    snap_bid: float,
    snap_ask: float,
    snap_mid: float,
    account: str = "",
    symbol: str = "",
    tag: str = "",
) -> Tuple[bool, str]:
    """
    首单 + 与 OMS 一致的 cancel/replace 循环（同步）。
    snap_*：Redis/PG 快照兜底；循环内优先用 IB ticker。
    """
    import ib_insync

    qty = int(qty or 0)
    if qty <= 0:
        return False, "qty must be > 0"

    from config import LIMIT_BUFFER_ENTRY

    params = dashboard_entry_fast_params()
    if not params["enabled"]:
        order = ib_insync.MarketOrder("BUY", qty)
        if account:
            order.account = account
        trade = ib.placeOrder(contract, order)
        ib.sleep(1.2)
        status = str(getattr(getattr(trade, "orderStatus", None), "status", "") or "")
        ok = status in {"Submitted", "PreSubmitted", "Filled", "PendingSubmit", "PendingCancel"}
        return ok, f"BUY MKT {qty} {symbol} {tag} status={status or 'Unknown'}"

    try:
        ib.reqMktData(contract, "", False, False)
    except Exception:
        pass
    ib.sleep(0.25)

    bid, ask = _ticker_bid_ask(ib, contract)
    if bid <= 0 or ask <= 0:
        bid, ask = float(snap_bid or 0.0), float(snap_ask or 0.0)
    entry_ref = float(snap_mid or 0.0)
    if entry_ref <= 0 and bid > 0 and ask > 0:
        entry_ref = (bid + ask) / 2.0
    if entry_ref <= 0:
        return False, "cannot resolve entry reference mid for cap"

    cap_price = _entry_cap_price(entry_ref)
    base_for_first = entry_ref
    if bid > 0 and ask > 0:
        base_for_first = (bid + ask) / 2.0

    limit_price = _get_entry_limit_price(bid, ask, base_for_first, 0, LIMIT_BUFFER_ENTRY)
    if limit_price < 0.05:
        return False, f"entry limit too low ({limit_price})"
    if limit_price > cap_price:
        return False, f"initial limit {limit_price:.4f} exceeds cap {cap_price:.4f} (ENTRY_MAX_REQUOTE_SLIPPAGE_PCT)"

    interval = float(params["interval"])
    max_window = float(params["window"])
    cancel_settle = float(params["cancel_settle"])
    max_attempts = max(1, int(math.ceil(max_window / interval)))

    order = LimitOrder("BUY", qty, float(limit_price))
    order.tif = "DAY"
    order.outsideRth = False
    if account:
        order.account = account
    trade = ib.placeOrder(contract, order)

    parts = [f"LMT0={limit_price:.2f}"]

    for attempt_no in range(max_attempts):
        ib.sleep(interval)
        st = str(getattr(getattr(trade, "orderStatus", None), "status", "") or "")
        filled = int(getattr(getattr(trade, "orderStatus", None), "filled", 0) or 0)

        if st == "Filled" and filled >= qty:
            ap = float(getattr(getattr(trade, "orderStatus", None), "avgFillPrice", 0.0) or 0.0)
            return True, (
                f"BUY {qty} {symbol} {tag} FILLED ~{ap:.2f} | entry_requote attempts≤{max_attempts} "
                f"interval={interval:.2f}s window≈{max_attempts * interval:.2f}s | " + " ".join(parts)
            )

        remaining = qty - filled
        if remaining <= 0:
            break

        if st not in ("Cancelled", "ApiCancelled", "Inactive", "Filled"):
            try:
                ib.cancelOrder(trade.order)
            except Exception:
                pass
        _await_cancel_settle(ib, trade, cancel_settle)

        filled = int(getattr(getattr(trade, "orderStatus", None), "filled", 0) or 0)
        remaining = qty - filled
        if remaining <= 0:
            break

        if attempt_no >= max_attempts - 1:
            break

        lb, la = _ticker_bid_ask(ib, contract)
        if lb <= 0 or la <= 0:
            lb, la = float(snap_bid or 0.0), float(snap_ask or 0.0)

        prev_px = limit_price
        try:
            lp = getattr(trade.order, "lmtPrice", None)
            if lp is not None and float(lp) > 0:
                prev_px = float(lp)
        except Exception:
            pass

        next_lmt = _next_entry_requote_price(
            prev_px,
            attempt_no + 1,
            cap_price,
            lb,
            la,
            LIMIT_BUFFER_ENTRY,
        )
        if next_lmt < 0.05:
            parts.append(f"requote_stop_low@{attempt_no + 1}")
            break

        limit_price = next_lmt
        parts.append(f"LMT{attempt_no + 1}={limit_price:.2f}")

        order = LimitOrder("BUY", remaining, float(limit_price))
        order.tif = "DAY"
        order.outsideRth = False
        if account:
            order.account = account
        trade = ib.placeOrder(contract, order)

    st = str(getattr(getattr(trade, "orderStatus", None), "status", "") or "")
    filled = int(getattr(getattr(trade, "orderStatus", None), "filled", 0) or 0)
    ap = float(getattr(getattr(trade, "orderStatus", None), "avgFillPrice", 0.0) or 0.0)
    ok = filled >= qty and st == "Filled"
    msg = (
        f"BUY {symbol} {tag} status={st} filled={filled}/{qty} avg={ap:.2f} | "
        + " ".join(parts)
    )
    return ok, msg

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
OMS 集成 —— monkey-patch legacy orchestrator_execution / execution_engine_v8,
不修改 New_Pro 源码;由 bootstrap 在进程启动时注入。

接线点:
  1. _get_entry_limit_price → oms_adapter.limit_price_from_quote (0.775)
  2. _get_exit_limit_price  → 非 fast_stop/urgent 时用 exit_frac
  3. _evaluate_second_dynamic_exits → disaster_only 时仅 check_disaster_stop
  4. select_entry_candidates_for_frame → 单标的绝对 net_edge,不做截面排序
  5. StrategyCore.check_exit → exit_rails (分钟)
"""
from __future__ import annotations

import logging
import math
from typing import Any, List, Dict, Optional

from qqq_btc.common.exit_rails import ExitRailsConfig, PositionState, check_tick_stops
from qqq_btc.live.oms_adapter import limit_price_from_quote
from qqq_btc.qqq import config as qcfg

logger = logging.getLogger("qqq_btc.live.oms_integration")

_PATCHED = False
_ORIG_ENTRY = None
_ORIG_EXIT = None
_ORIG_TICK_EXITS = None
_ORIG_SELECT_ENTRY = None
_ORIG_CACHE_EXEC = None


def select_entry_candidates_qqq_btc(
    entry_candidates: List[Dict[str, Any]],
    allowed_entries: int,
    cfg,
    *,
    orig_fn=None,
) -> List[Dict[str, Any]]:
    """
    qqq_btc 单标的:decide_entry 已用绝对 net_edge 过滤,不再做截面分池/alpha_strength 排序。
    保留 allowed_entries 上限与 legacy 多标的兼容(>1 候选时才调用 orig_fn)。
    """
    if allowed_entries <= 0 or not entry_candidates:
        return []
    if len(entry_candidates) <= 1:
        return entry_candidates[:allowed_entries]
    if orig_fn is not None:
        return orig_fn(entry_candidates, allowed_entries, cfg)
    return sorted(
        entry_candidates,
        key=lambda x: abs(float((x.get("sig") or {}).get("meta", {}).get("net_edge_raw", x.get("alpha_strength", 0)) or 0)),
        reverse=True,
    )[:allowed_entries]


def _round_option_tick(px: float, *, side: str = "BUY") -> float:
    """与 legacy OMS 一致: 期权限价两位小数,买不穿越 ask。"""
    px = float(px)
    if px <= 0:
        return 0.0
    if side.upper() in ("BUY", "BOT", "LONG"):
        return round(math.floor(px * 100.0) / 100.0, 2)
    return round(math.ceil(px * 100.0) / 100.0, 2)


def entry_limit_price_qqq_btc(sig, base_price, attempt_no=0, *, orig_fn=None):
    """drop-in 替换 OrchestratorExecution._get_entry_limit_price。"""
    bid = float(sig.get("meta", {}).get("bid", 0.0) or 0.0)
    ask = float(sig.get("meta", {}).get("ask", 0.0) or 0.0)
    if bid > 0.0 and ask > 0.0 and ask >= bid:
        raw = limit_price_from_quote(bid, ask, "BUY", qcfg.FILL_MODEL)
        ask_tick = math.ceil(ask * 100.0) / 100.0
        ask_minus_tick = max(round(ask_tick - 0.01, 2), round(math.floor(bid * 100.0) / 100.0, 2))
        candidate = _round_option_tick(raw, side="BUY")
        candidate = max(round(math.floor(bid * 100.0) / 100.0, 2), candidate)
        candidate = min(candidate, ask_minus_tick) if ask_minus_tick > 0 else candidate
        return candidate
    if orig_fn is not None:
        return orig_fn(sig, base_price, attempt_no)
    return 0.0


def exit_limit_price_qqq_btc(
    base_price,
    bid=0.0,
    ask=0.0,
    is_urgent=False,
    attempt_no=0,
    fast_stop=False,
    fast_requote=False,
    *,
    orig_fn=None,
):
    """非紧急出场用 fill_model exit_frac; 止损 fast 路径仍走 legacy。"""
    bid = float(bid or 0.0)
    ask = float(ask or 0.0)
    if fast_stop or is_urgent or fast_requote:
        if orig_fn is not None:
            return orig_fn(
                base_price, bid, ask, is_urgent, attempt_no, fast_stop, fast_requote
            )
        return round(max(bid, 0.01), 2) if bid > 0 else 0.01
    if bid > 0.0 and ask >= bid:
        raw = limit_price_from_quote(bid, ask, "SELL", qcfg.FILL_MODEL)
        floor_bid = round(math.floor(bid * 100.0) / 100.0, 2)
        return max(_round_option_tick(raw, side="SELL"), floor_bid)
    if orig_fn is not None:
        return orig_fn(
            base_price, bid, ask, is_urgent, attempt_no, fast_stop, fast_requote
        )
    return round(max(float(base_price or 0.0), 0.01), 2)


def _position_entry_price(st: Any) -> Optional[float]:
    for attr in ("entry_price", "opt_entry_price", "avg_entry_price"):
        v = getattr(st, attr, None)
        if v is not None:
            try:
                fv = float(v)
                if fv > 0:
                    return fv
            except (TypeError, ValueError):
                pass
    return None


def _quote_option_mid(quote: dict) -> Optional[float]:
    bid = float(quote.get("opt_bid", quote.get("bid", 0.0)) or 0.0)
    ask = float(quote.get("opt_ask", quote.get("ask", 0.0)) or 0.0)
    if bid > 0 and ask >= bid:
        return (bid + ask) / 2.0
    px = float(quote.get("opt_price", quote.get("price", 0.0)) or 0.0)
    return px if px > 0 else None


async def evaluate_disaster_tick_exits(engine, curr_ts: float, rails: ExitRailsConfig | None = None) -> None:
    """秒级风险/浮盈轨:check_tick_stops,不跑 legacy 阶梯/FLASH。"""
    rails = rails or qcfg.EXIT_RAILS
    tick_on = (
        rails.disaster_stop_roi is not None
        or rails.tick_fast_hard_roi is not None
        or rails.tick_profit_trigger_roi is not None
        or bool(rails.tick_profit_ladder)
    )
    if not tick_on:
        return
    for sym, st in engine.states.items():
        if int(getattr(st, "position", 0) or 0) == 0:
            # 平仓后清掉独立 tick_peak,避免下一笔继承
            if hasattr(st, "qqq_btc_tick_peak_roi"):
                st.qqq_btc_tick_peak_roi = 0.0
            continue
        quote = engine._second_quote_for_symbol(sym)
        if not quote:
            continue
        entry_px = _position_entry_price(st)
        mid = _quote_option_mid(quote)
        if entry_px is None or mid is None:
            continue
        peak = float(getattr(st, "qqq_btc_tick_peak_roi", 0.0) or 0.0)
        pos = PositionState(entry_price=entry_px, entry_bar=0, tick_peak_roi=peak)
        reason = check_tick_stops(rails, pos, mid)
        st.qqq_btc_tick_peak_roi = pos.tick_peak_roi
        if not reason:
            continue
        st.qqq_btc_tick_peak_roi = 0.0
        side = int(getattr(st, "position", 1) or 1)
        bid = float(quote.get("opt_bid", quote.get("bid", 0.0)) or 0.0)
        ask = float(quote.get("opt_ask", quote.get("ask", 0.0)) or 0.0)
        exit_sig = {
            "action": "SELL",
            "dir": side,
            "target_side": side,
            "reason": f"QQQ_BTC_{reason}",
            "price": mid,
            "market_price": mid,
            "bid": bid,
            "ask": ask,
            "meta": {"source": "qqq_btc_tick_stop", "roi": mid / entry_px - 1.0},
        }
        logger.warning(
            "⚡ [qqq_btc tick_stop] %s | %s | roi=%.1f%%",
            sym,
            reason,
            (mid / entry_px - 1.0) * 100.0,
        )
        await engine._submit_strategy_order(
            "SELL",
            sym,
            exit_sig,
            float(quote.get("stock_price", getattr(st, "last_price", 0.0)) or 0.0),
            curr_ts,
            -1,
            frame_id=f"1s:disaster:{int(float(curr_ts or 0.0))}",
            allow_delay_queue=False,
        )


def apply_oms_patches(*, tick_exits_mode: str = "disaster_only") -> None:
    global _PATCHED, _ORIG_ENTRY, _ORIG_EXIT, _ORIG_TICK_EXITS, _ORIG_SELECT_ENTRY
    if _PATCHED:
        return

    import execution_engine_v8 as eex
    import orchestrator_execution as oex

    _ORIG_SELECT_ENTRY = eex.select_entry_candidates_for_frame

    def _patched_select(entry_candidates, allowed_entries, cfg):
        return select_entry_candidates_qqq_btc(
            entry_candidates,
            allowed_entries,
            cfg,
            orig_fn=_ORIG_SELECT_ENTRY,
        )

    eex.select_entry_candidates_for_frame = _patched_select
    logger.info("patched select_entry_candidates_for_frame → qqq_btc absolute (no cross-section rank)")

    _ORIG_ENTRY = oex.OrchestratorExecution._get_entry_limit_price
    _ORIG_EXIT = oex.OrchestratorExecution._get_exit_limit_price

    def _patched_entry(self, sig, base_price, attempt_no=0):
        return entry_limit_price_qqq_btc(
            sig, base_price, attempt_no, orig_fn=_ORIG_ENTRY.__get__(self, oex.OrchestratorExecution)
        )

    def _patched_exit(
        self, base_price, bid=0.0, ask=0.0, is_urgent=False, attempt_no=0, fast_stop=False, fast_requote=False
    ):
        return exit_limit_price_qqq_btc(
            base_price,
            bid,
            ask,
            is_urgent,
            attempt_no,
            fast_stop,
            fast_requote,
            orig_fn=_ORIG_EXIT.__get__(self, oex.OrchestratorExecution),
        )

    oex.OrchestratorExecution._get_entry_limit_price = _patched_entry
    oex.OrchestratorExecution._get_exit_limit_price = _patched_exit
    logger.info("patched OrchestratorExecution entry/exit limit → fill_model 0.775")

    if tick_exits_mode != "legacy":
        _ORIG_TICK_EXITS = eex.ExecutionEngineV8._evaluate_second_dynamic_exits

        async def _patched_tick_exits(self, curr_ts: float):
            if tick_exits_mode == "off":
                return
            await evaluate_disaster_tick_exits(self, curr_ts)

        eex.ExecutionEngineV8._evaluate_second_dynamic_exits = _patched_tick_exits
        logger.info("patched ExecutionEngineV8 tick exits → %s", tick_exits_mode)

        global _ORIG_CACHE_EXEC
        _ORIG_CACHE_EXEC = eex.ExecutionEngineV8._cache_execution_market_packet

        def _patched_cache_exec(self, market_packet: dict):
            """秒级 quote 缓存保留;禁止 tick 棘轮 max_roi(与 exit_rails 分钟口径对齐)。"""
            saved: dict = {}
            for sym, st in (getattr(self, "states", {}) or {}).items():
                if st is not None and int(getattr(st, "position", 0) or 0) != 0:
                    saved[sym] = float(getattr(st, "max_roi", -1.0) or -1.0)
            _ORIG_CACHE_EXEC(self, market_packet)
            for sym, prev in saved.items():
                st = self.states.get(sym)
                if st is not None:
                    st.max_roi = prev

        eex.ExecutionEngineV8._cache_execution_market_packet = _patched_cache_exec
        logger.info("patched ExecutionEngineV8 tick max_roi → frozen (minute ALPHA_FRAME only)")

    try:
        from strategy_selector import StrategyCore
        from qqq_btc.live.strategy_exit_bridge import apply_strategy_exit_patch

        apply_strategy_exit_patch(StrategyCore)
        logger.info("patched StrategyCore.check_exit → exit_rails")
    except ImportError as e:
        logger.warning("StrategyCore exit_rails patch skipped: %s", e)

    try:
        from qqq_btc.live.strategy_entry_bridge import apply_strategy_entry_patch

        apply_strategy_entry_patch(StrategyCore)
        logger.info("patched StrategyCore.decide_entry → choose_entry")
    except ImportError as e:
        logger.warning("StrategyCore entry_bridge patch skipped: %s", e)

    try:
        from qqq_btc.live.fill_audit_writer import apply_fill_audit_patch

        apply_fill_audit_patch()
    except ImportError as e:
        logger.warning("fill_audit patch skipped: %s", e)

    _orig_submit = eex.ExecutionEngineV8._submit_strategy_order

    async def _patched_submit_strategy_order(
        self,
        action,
        sym,
        sig,
        stock_price,
        curr_ts,
        batch_idx,
        frame_id=None,
        allow_delay_queue=True,
    ):
        # ALPHA_FRAME 决策后立刻下单;BUY 不走 OMS 延迟队列(与 replay 标签 60s 延迟解耦)
        if str(action).upper() == "BUY":
            allow_delay_queue = False
        return await _orig_submit(
            self,
            action,
            sym,
            sig,
            stock_price,
            curr_ts,
            batch_idx,
            frame_id=frame_id,
            allow_delay_queue=allow_delay_queue,
        )

    eex.ExecutionEngineV8._submit_strategy_order = _patched_submit_strategy_order
    logger.info("patched ExecutionEngineV8._submit_strategy_order → BUY immediate (no delay queue)")

    _PATCHED = True

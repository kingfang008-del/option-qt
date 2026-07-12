#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LiveSignalEngine 输出 → legacy OMS 可消费的 ALPHA_FRAME 载荷。

契约: ExecutionEngineV8._process_alpha_frame 读取 items[].opt_data / alpha / edges。
qqq_btc 路径用 net_edge 绝对值替代 cs_alpha_z; OMS StrategyCore 仍负责最终 BUY/SELL,
但 SE 侧应提供与 strict replay 一致的 edge 字段供日志/门控。
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence


def build_opt_data_from_quotes(quotes: Mapping[str, float], *, leg: str = "CALL") -> dict:
    """从 bar-close quotes 构造 OMS _build_strategy_ctx 所需的 opt_data。"""
    call_bid = float(quotes.get("exec_call_bid", 0.0) or 0.0)
    call_ask = float(quotes.get("exec_call_ask", 0.0) or 0.0)
    put_bid = float(quotes.get("exec_put_bid", 0.0) or 0.0)
    put_ask = float(quotes.get("exec_put_ask", 0.0) or 0.0)
    call_mid = float(
        quotes.get(
            "exec_call_mid",
            (call_bid + call_ask) / 2.0 if call_bid > 0 and call_ask >= call_bid else 0.0,
        )
        or 0.0
    )
    put_mid = float(
        quotes.get(
            "exec_put_mid",
            (put_bid + put_ask) / 2.0 if put_bid > 0 and put_ask >= put_bid else 0.0,
        )
        or 0.0
    )
    leg_u = leg.upper()
    if leg_u == "PUT":
        bid, ask, mid = put_bid, put_ask, put_mid
    else:
        bid, ask, mid = call_bid, call_ask, call_mid
    spread_pct = float(
        quotes.get(
            f"exec_{leg_u.lower()}_spread_pct",
            ((ask - bid) / mid if mid > 0 and ask >= bid else 0.0),
        )
        or 0.0
    )
    call_spread = float(
        quotes.get(
            "exec_call_spread_pct",
            ((call_ask - call_bid) / call_mid if call_mid > 0 and call_ask >= call_bid else 0.0),
        )
        or 0.0
    )
    put_spread = float(
        quotes.get(
            "exec_put_spread_pct",
            ((put_ask - put_bid) / put_mid if put_mid > 0 and put_ask >= put_bid else 0.0),
        )
        or 0.0
    )
    has_feed = bid > 0 and ask >= bid and mid > 0
    return {
        "has_feed": has_feed,
        "bid": bid,
        "ask": ask,
        "price": mid,
        "call_bid": call_bid,
        "call_ask": call_ask,
        "call_price": call_mid,
        "call_spread_pct": call_spread,
        "put_bid": put_bid,
        "put_ask": put_ask,
        "put_price": put_mid,
        "put_spread_pct": put_spread,
        "spread_pct": spread_pct,
    }


def build_alpha_item(
    symbol: str,
    *,
    batch_idx: int,
    preds: Mapping[str, float],
    quotes: Mapping[str, float],
    stock_price: float = 0.0,
    leg: str = "CALL",
) -> dict:
    """单个 symbol 的 ALPHA_FRAME item。"""
    opt_data = build_opt_data_from_quotes(quotes, leg=leg)
    net_edge = float(preds.get("net_edge", 0.0) or 0.0)
    call_edge = float(preds.get("call_net_edge", net_edge) or net_edge)
    put_edge = float(preds.get("put_net_edge", 0.0) or 0.0)
    return {
        "symbol": symbol,
        "batch_idx": batch_idx,
        "stock_price": float(stock_price or 0.0),
        "alpha": net_edge,
        "call_edge": call_edge,
        "put_edge": put_edge,
        "net_edge_raw": float(preds.get("net_edge_raw", net_edge) or net_edge),
        "net_edge_q10": float(preds.get("net_edge_q10", 0.0) or 0.0),
        "straddle_edge": float(preds.get("straddle_net_edge", 0.0) or 0.0),
        "cs_alpha_z": 0.0,
        "execution_cost_pred": float(preds.get("execution_cost", 0.0) or 0.0),
        "vol_z": 0.0,
        "is_ready": True,
        "opt_data": opt_data,
        "qqq_btc": True,
        "chosen_leg": leg.upper(),
    }


def build_alpha_frame(
    *,
    curr_ts: float,
    frame_id: str,
    symbol: str,
    preds: Mapping[str, float],
    quotes: Mapping[str, float],
    stock_price: float = 0.0,
    leg: str = "CALL",
    symbols: Optional[Sequence[str]] = None,
) -> dict:
    """完整 ALPHA_FRAME Redis payload。"""
    syms = list(symbols) if symbols else [symbol]
    item = build_alpha_item(
        symbol,
        batch_idx=0,
        preds=preds,
        quotes=quotes,
        stock_price=stock_price,
        leg=leg,
    )
    return {
        "source": "qqq_btc_live",
        "action": "ALPHA_FRAME",
        "ts": float(curr_ts),
        "frame_id": frame_id,
        "symbols": syms,
        "items": [item],
        "index_trend": 0,
        "spy_roc_5min": [0.0],
        "qqq_roc_5min": [0.0],
        "is_zombie_market": False,
        "global_regime_reversal_cnt": 0,
        "global_is_volatile_regime": False,
        "global_regime_band": "calm",
        "global_regime_score": 0.0,
        "spy_day_roc": 0.0,
        "qqq_day_roc": 0.0,
    }


def action_to_trade_intent(action: dict) -> Optional[dict]:
    """
    LiveSignalEngine.on_bar_close 返回值 → OMS 可消费的显式 BUY/SELL(可选旁路 StrategyCore)。

    默认仍发 ALPHA_FRAME 让 OMS StrategyCore 决策;仅在 ENTER/EXIT 明确时可用于
    shadow 对账或直接下单测试。
    """
    if not action:
        return None
    kind = str(action.get("action", "")).upper()
    if kind == "ENTER":
        return {
            "action": "BUY",
            "leg": action.get("leg", "CALL"),
            "edge": action.get("edge"),
            "limit_price": action.get("limit_price"),
            "source": "qqq_btc_live",
        }
    if kind == "EXIT":
        return {
            "action": "SELL",
            "reason": action.get("reason"),
            "leg": action.get("leg"),
            "price": action.get("price"),
            "source": "qqq_btc_live",
        }
    return None

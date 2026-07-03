#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
OMS 成交模型适配 —— 落单价与 fill 审计接 qqq_btc.common.fill_model。

execution_engine_v8 接线点:
  1. 限价单价格: limit_price_from_quote(bid, ask, side)
  2. 成交审计: audit_fill(bid, ask, fill_px, side) → fill_spread_frac

禁止在 OMS 内硬编码 0.20–0.45 点差位;统一走本模块。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from qqq_btc.common.fill_model import OptionSpreadFillModel, spread_interpolate
from qqq_btc.qqq import config as qcfg


@dataclass(frozen=True)
class FillAuditRecord:
    bid: float
    ask: float
    fill_px: float
    side: str
    spread_pct: float
    fill_spread_frac: float
    model_entry_frac: float


def limit_price_from_quote(
    bid: float,
    ask: float,
    side: str,
    fill_model: Optional[OptionSpreadFillModel] = None,
) -> float:
    """OMS 挂限价:买入用 entry_frac,卖出用 exit_frac(与标签/replay 一致)。"""
    fm = fill_model or qcfg.FILL_MODEL
    side_u = side.upper()
    if side_u in ("BUY", "BOT", "LONG"):
        frac = fm.entry_frac
        px_side = "BUY"
    elif side_u in ("SELL", "SLD", "SHORT"):
        frac = fm.exit_frac
        px_side = "SELL"
    else:
        raise ValueError(f"unknown side: {side}")
    return float(spread_interpolate(bid, ask, frac, px_side))


def audit_fill(
    bid: float,
    ask: float,
    fill_px: float,
    side: str,
    fill_model: Optional[OptionSpreadFillModel] = None,
) -> FillAuditRecord:
    """
    实盘成交审计:反推实际 fill_spread_frac,供 shadow 期与 0.775 假设比对。
    BUY: fill = bid + frac*spread → frac = (fill-bid)/spread
    SELL: fill = ask - frac*spread → frac = (ask-fill)/spread
    """
    fm = fill_model or qcfg.FILL_MODEL
    spread = ask - bid
    spread_pct = float(spread / ((bid + ask) / 2.0)) if bid > 0 and ask > bid else float("nan")
    side_u = side.upper()
    if side_u in ("BUY", "BOT", "LONG"):
        frac = (fill_px - bid) / spread if spread > 0 else float("nan")
        model_frac = fm.entry_frac
    else:
        frac = (ask - fill_px) / spread if spread > 0 else float("nan")
        model_frac = fm.exit_frac
    return FillAuditRecord(
        bid=float(bid),
        ask=float(ask),
        fill_px=float(fill_px),
        side=side_u,
        spread_pct=spread_pct,
        fill_spread_frac=float(frac),
        model_entry_frac=model_frac,
    )


def commission_drag(entry_price: float, fill_model: Optional[OptionSpreadFillModel] = None) -> float:
    fm = fill_model or qcfg.FILL_MODEL
    return float(fm.commission_return_drag(entry_price))


def straddle_limit_prices(
    call_bid: float, call_ask: float, put_bid: float, put_ask: float, side: str = "BUY",
) -> tuple[float, float, float]:
    """跨式两腿限价 + 合成权利金(审计用)。"""
    c = limit_price_from_quote(call_bid, call_ask, side)
    p = limit_price_from_quote(put_bid, put_ask, side)
    return c, p, c + p

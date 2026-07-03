#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
统一成交/成本模型 —— 本路径的 single source of truth。

标签构建 (labels.py)、strict replay (replay_harness.py)、实盘成交审计
必须全部调用本模块,禁止各自实现 fill 假设。这是对上一代
「标签 0.75 / 回测 0.5 / OMS 0.20-0.45 三层互相矛盾」缺陷的直接修复。

插值公式收编自 production/history_replay/mock_ibkr_historical_1s.py
的 _spread_interpolate_fill(已验证正确),默认 frac 从 0.5(mid) 改为
0.775(实盘实测成交 0.75-0.8 点差位的中值)。

所有接口同时支持标量与 numpy/pandas 向量输入。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# 实盘观察:限价单实际成交多落在 0.75-0.8 点差位,取中值作为全路径默认。
DEFAULT_OPTION_FILL_FRAC = 0.775


def spread_interpolate(bid, ask, frac: float, side: str):
    """
    点差插值成交价。BUY: bid + frac*(ask-bid); SELL: ask - frac*(ask-bid)。
    frac=0 → 己方最优价(BUY 成交在 bid),frac=0.5 → mid,frac=1 → 对手价。
    无效盘口 (bid<=0 / ask<=0 / ask<bid) 返回 NaN。
    """
    b = np.asarray(bid, dtype=np.float64)
    a = np.asarray(ask, dtype=np.float64)
    valid = (b > 0.0) & (a > 0.0) & (a >= b)
    spr = a - b
    side_u = str(side).upper()
    if side_u == "BUY":
        px = b + frac * spr
    elif side_u == "SELL":
        px = a - frac * spr
    else:
        raise ValueError(f"side must be BUY/SELL, got {side!r}")
    out = np.where(valid, px, np.nan)
    if np.isscalar(bid) and np.isscalar(ask):
        return float(out)
    return out


@dataclass(frozen=True)
class OptionSpreadFillModel:
    """
    期权点差插值成交模型(QQQ 0DTE 等交易所期权)。

    entry_frac / exit_frac 独立可调:开仓可等价、平仓(尤其止损)往往更差。
    默认两侧同为 0.775。
    """
    entry_frac: float = DEFAULT_OPTION_FILL_FRAC
    exit_frac: float = DEFAULT_OPTION_FILL_FRAC
    commission_per_contract: float = 0.65   # USD/张/边
    contract_multiplier: float = 100.0

    def entry_fill(self, bid, ask):
        """买入开仓成交价(做多期权)。"""
        return spread_interpolate(bid, ask, self.entry_frac, "BUY")

    def exit_fill(self, bid, ask):
        """卖出平仓成交价。"""
        return spread_interpolate(bid, ask, self.exit_frac, "SELL")

    def commission_return_drag(self, entry_price):
        """
        往返佣金折合的收益率拖累(以开仓权利金为基数)。
        entry_price 为单份期权价格(非乘数后名义)。
        """
        px = np.asarray(entry_price, dtype=np.float64)
        notional = px * self.contract_multiplier
        drag = np.where(notional > 0, 2.0 * self.commission_per_contract / notional, np.nan)
        if np.isscalar(entry_price):
            return float(drag)
        return drag

    def round_trip_spread_drag(self, spread_pct):
        """
        点差部分的往返摩擦估算(诊断用,标签本身用精确 fill 价):
        entry 相对 mid 多付 (entry_frac-0.5)*spread,exit 少收 (exit_frac-0.5)*spread。
        """
        s = np.asarray(spread_pct, dtype=np.float64)
        drag = ((self.entry_frac - 0.5) + (self.exit_frac - 0.5)) * s
        drag = np.maximum(drag, 0.0)
        if np.isscalar(spread_pct):
            return float(drag)
        return drag


@dataclass(frozen=True)
class PerpFillModel:
    """
    永续合约(BTC-PERP)成交模型。成本结构与期权完全不同:
    taker 费率 + 冲击滑点 + 持有期 funding,无点差插值语义。
    """
    taker_fee_bps: float = 5.0        # 单边 taker 费率
    slippage_bps: float = 2.0         # 单边冲击滑点(市价单穿越盘口)
    funding_interval_hours: float = 8.0

    def entry_fill(self, price):
        """做多开仓:按 mark/last 上浮费率+滑点。"""
        p = np.asarray(price, dtype=np.float64)
        out = p * (1.0 + (self.taker_fee_bps + self.slippage_bps) / 1e4)
        if np.isscalar(price):
            return float(out)
        return out

    def exit_fill(self, price):
        p = np.asarray(price, dtype=np.float64)
        out = p * (1.0 - (self.taker_fee_bps + self.slippage_bps) / 1e4)
        if np.isscalar(price):
            return float(out)
        return out

    def funding_drag(self, funding_rate_8h, holding_seconds: float):
        """
        持有期 funding 拖累(做多、funding 为正时付出)。
        funding_rate_8h: 每个 funding 周期的费率(如 0.0001 = 1bp/8h)。
        """
        r = np.asarray(funding_rate_8h, dtype=np.float64)
        periods = float(holding_seconds) / (self.funding_interval_hours * 3600.0)
        out = r * periods
        if np.isscalar(funding_rate_8h):
            return float(out)
        return out

    def round_trip_fee_drag(self) -> float:
        """费率+滑点的固定往返拖累(不含 funding)。"""
        return 2.0 * (self.taker_fee_bps + self.slippage_bps) / 1e4

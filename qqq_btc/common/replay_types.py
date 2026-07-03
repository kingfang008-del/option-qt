#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Replay 共享类型 —— 避免 replay_harness / replay_session / event_replay 循环依赖。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ReplayConfig:
    entry_threshold: float = 0.015
    entry_delay_bars: int = 1
    max_spread_pct: float = 0.06
    cooldown_bars: int = 5
    long_only: bool = True
    entry_threshold_schedule: Optional[tuple] = None
    max_trades_per_day: Optional[int] = None
    daily_loss_stop: Optional[float] = None
    loss_streak_n: Optional[int] = None
    loss_streak_cooldown_bars: int = 30
    straddle_entry_threshold: Optional[float] = None
    max_straddles_per_day: Optional[int] = None
    # 会话内 bar 序号(09:30=0)允许新开仓区间;None=不限制
    session_entry_start_bar: Optional[int] = 0
    session_entry_end_bar: Optional[int] = 360

    def threshold_at(self, session_bar: Optional[int]) -> float:
        if self.entry_threshold_schedule is None or session_bar is None:
            return self.entry_threshold
        th = self.entry_threshold
        for start, value in self.entry_threshold_schedule:
            if session_bar >= start:
                th = value
            else:
                break
        return th

    def session_allows_entry(self, session_bar: Optional[int]) -> bool:
        """session_bar 在 [start, end] 内才允许新开仓(None 边界=不限制)。"""
        if session_bar is None:
            return True
        if self.session_entry_start_bar is not None and session_bar < self.session_entry_start_bar:
            return False
        if self.session_entry_end_bar is not None and session_bar > self.session_entry_end_bar:
            return False
        return True


@dataclass
class Trade:
    entry_ts: object
    exit_ts: object
    entry_price: float
    exit_price: float
    net_return: float
    exit_reason: str
    bars_held: int
    signal_edge: float
    leg: str = "CALL"


@dataclass
class ReplayResult:
    trades: List[Trade] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)

    def summary(self) -> dict:
        if not self.trades:
            return {"trades": 0, "total_net_return": 0.0}
        rets = np.array([t.net_return for t in self.trades])
        eq = np.array(self.equity_curve) if self.equity_curve else np.cumprod(1 + rets)
        peak = np.maximum.accumulate(eq)
        mdd = float(((eq - peak) / peak).min()) if len(eq) else 0.0
        wins = rets[rets > 0]
        losses = rets[rets < 0]
        profit_factor = float(wins.sum() / -losses.sum()) if losses.sum() < 0 else float("inf")
        reasons = pd.Series([t.exit_reason for t in self.trades]).value_counts().to_dict()
        legs = pd.Series([t.leg for t in self.trades]).value_counts().to_dict()
        return {
            "trades": int(len(rets)),
            "total_net_return": float(np.prod(1 + rets) - 1),
            "avg_net_return": float(rets.mean()),
            "hit_rate": float((rets > 0).mean()),
            "profit_factor": profit_factor,
            "max_drawdown_mtm": mdd,
            "avg_bars_held": float(np.mean([t.bars_held for t in self.trades])),
            "worst_trade": float(rets.min()),
            "exit_reasons": reasons,
            "trades_by_leg": legs,
        }

    def trades_frame(self) -> pd.DataFrame:
        if not self.trades:
            return pd.DataFrame()
        return pd.DataFrame(
            [
                {
                    "entry_ts": t.entry_ts,
                    "exit_ts": t.exit_ts,
                    "leg": t.leg,
                    "entry_price": t.entry_price,
                    "exit_price": t.exit_price,
                    "net_return": t.net_return,
                    "exit_reason": t.exit_reason,
                    "bars_held": t.bars_held,
                    "signal_edge": t.signal_edge,
                }
                for t in self.trades
            ]
        )

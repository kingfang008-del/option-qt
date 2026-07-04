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
    # 0DTE 权利金 ROI 的经验 p10 约 -10%;高 edge bar 的预测 q10 中位约 -15%。
    # 默认 floor=-0.20:过滤极端悲观分位,但不要求 q10>0。None=不检查 q10。
    edge_q10_floor: Optional[float] = -0.20
    # 账户仓位比例:权益复利用 (1 + f * option_roi)。
    # 0DTE 单笔 ROI 波动大,f=1 远超 Kelly、波动拖累会把正期望复利打成大亏。
    # 默认 1.0 保持旧口径;QQQ 路径用 ~0.25(半 Kelly)。
    position_frac: float = 1.0
    # --- 滚动分位阈值(None=关闭) ---
    # 入场阈值 = max(静态调度阈值, 近 window 根入场窗 bar 的 edge 分位数)。
    # 动机:模型打分分布随月份漂移(2026-04→06 过阈 bar 607→2113 而均值
    # +0.15→+0.04),固定绝对阈值的选择性失控;分位数把"只做头部 x% 机会"
    # 钉住。取 max = 只收紧不放松,绝对阈值仍然是覆盖成本的地板。
    entry_quantile: Optional[float] = None
    entry_quantile_window: int = 1500   # ~5 个交易日的入场窗 bar
    entry_quantile_min_obs: int = 300   # 观测不足时退回静态阈值
    # --- PUT 腿行情开关(None=不门控) ---
    # PUT 只在恐慌/高波动 regime 有正期望:三时期审计显示 vix_level(归一化)
    # 最高四分位贡献了 PUT 几乎全部利润,低 VIX 时 PUT 持续放血且挤占 CALL
    # 额度(2026Q1: 门槛 0.25 把 PUT 腿从 -31% 变 +99%)。
    # 语义:入场 bar 的 put_gate 信号值 >= put_gate_min 才允许开 PUT。
    # 信号值缺失(NaN)时视为不通过——宁可错过,不可误开。
    put_gate_min: Optional[float] = None

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
    position_frac: float = 1.0

    def summary(self, position_frac: Optional[float] = None) -> dict:
        f = float(self.position_frac if position_frac is None else position_frac)
        if not self.trades:
            return {"trades": 0, "total_net_return": 0.0, "position_frac": f}
        rets = np.array([t.net_return for t in self.trades])
        # equity_curve 已按 f 复利;无曲线时回退到 f * ROI
        if self.equity_curve:
            eq = np.array(self.equity_curve, dtype=float)
        else:
            eq = np.cumprod(1.0 + f * rets)
        peak = np.maximum.accumulate(eq)
        mdd = float(((eq - peak) / peak).min()) if len(eq) else 0.0
        wins = rets[rets > 0]
        losses = rets[rets < 0]
        profit_factor = float(wins.sum() / -losses.sum()) if losses.sum() < 0 else float("inf")
        reasons = pd.Series([t.exit_reason for t in self.trades]).value_counts().to_dict()
        legs = pd.Series([t.leg for t in self.trades]).value_counts().to_dict()
        return {
            "trades": int(len(rets)),
            "position_frac": f,
            # 账户复利(含仓位比例)
            "total_net_return": float(eq[-1] - 1.0) if len(eq) else 0.0,
            # 权利金 ROI 口径(信号质量,不受 f 影响)
            "avg_net_return": float(rets.mean()),
            "sum_net_return": float(rets.sum()),
            "full_size_compound": float(np.prod(1.0 + rets) - 1.0),
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

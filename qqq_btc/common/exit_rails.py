#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
退出轨道(exit rails)—— 语义收编自 New_Pro/baseline_qqq/strategy_exit_rails.py,
去除对 legacy StrategyConfig 的耦合,改为独立 dataclass 配置。

检查顺序(与旧实现一致,先风险后利润):
  1. EOD 强平
  2. 硬止损 / 软止损(期权 ROI 口径)
  3. 时间止损(持有超时且未达最低 ROI)
  4. trailing(max_roi 触发后按保留比例回撤离场)
  5. 阶梯保护(max_roi 达档位后跌破 floor 离场)
  6. flash 保护(冲高后回落到保本线)

replay_harness 与实盘策略层共用本模块,保证回放退出分布 = 实盘退出分布。

★ MTM 节奏契约(实盘接线必须遵守):
  `check_exit` 只允许按【分钟收盘 mid】调用 —— 与标签、strict replay 完全同口径。
  0DTE ATM 权利金分钟内振幅可达 15%+,若按 tick 调用 check_exit:
    a) 影线会直接打穿 hard/soft 止损,把趋势中的仓位震掉;
    b) pos.update() 会把 max_roi 棘轮到影线高点,
       导致 trailing/阶梯/flash 全部相对影线提前引爆。
  两者都会使实盘退出分布系统性偏离回放,回放验收随之失效。
  tick 级监控只允许调用 `check_disaster_stop`:宽幅灾难止损(如 -25%),
  无状态、不更新 max_roi,只为跳空/闪崩兜底,正常波动永远不应触发。
  建议实盘 tick MTM 先做 3-5s 中价平滑再喂灾难止损,避免单笔异常报价触发。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass(frozen=True)
class ExitRailsConfig:
    hard_stop_roi: float = -0.12          # 期权权利金 ROI 硬止损
    soft_stop_roi: float = -0.08          # 软止损
    # 早期止损:持有 early_stop_bars 后 ROI 仍 <= early_stop_roi 则离场(None=关闭)
    early_stop_bars: Optional[int] = None
    early_stop_roi: float = -0.05
    time_stop_bars: int = 15              # 持有 N bar 后若 ROI 低于门槛则离场
    time_stop_min_roi: float = 0.05
    max_hold_bars: int = 30               # 无条件最长持有
    trailing_trigger_roi: float = 0.25    # max_roi 达标后启动 trailing
    trailing_keep_ratio: float = 0.60     # 回撤到 max_roi*keep 以下离场
    # (trigger, floor): peak 曾达 trigger 后,累积 ratchet floor = 各档 floor 最大值
    # 0DTE 默认:密档 + 紧 floor(相对 legacy LADDER_TIGHT 前几档),待 calibrate_rails 重标
    ladder: Tuple[Tuple[float, float], ...] = (
        (0.08, 0.05),
        (0.12, 0.08),
        (0.18, 0.12),
        (0.25, 0.18),
        (0.35, 0.26),
    )
    flash_trigger_roi: float = 0.08       # 低于首档 ladder,避免与 8% 档重复抢 exit
    flash_exit_roi: float = 0.03
    eod_close_bar_index: Optional[int] = None  # 会话内强平 bar 序号(None=不启用)
    # tick 级灾难止损(check_disaster_stop 专用,None=不启用)。
    # 必须显著深于 hard_stop:hard_stop 由分钟收盘价触发,
    # 灾难止损只为分钟内跳空/闪崩兜底,正常影线不应碰到。
    disaster_stop_roi: Optional[float] = None


@dataclass
class PositionState:
    entry_price: float
    entry_bar: int
    max_roi: float = 0.0

    def update(self, current_price: float) -> float:
        roi = current_price / self.entry_price - 1.0
        if roi > self.max_roi:
            self.max_roi = roi
        return roi


def ladder_floor(cfg: ExitRailsConfig, max_roi: float) -> float:
    """
    累积 ratchet 阶梯底价:所有 max_roi 已触发的档位取 floor 最大值。

    旧实现只取「最高一档」且 floor 偏低(9DTE 手拍),0DTE 下 peak 12% 却允许
    回吐到 3%。正确语义:peak 12% 时应锁 8%(若存在 (0.12,0.08) 档)。
    """
    floor = float("-inf")
    for trigger, f in sorted(cfg.ladder, key=lambda x: x[0]):
        if max_roi >= trigger:
            floor = max(floor, f)
    return floor


def check_exit(
    cfg: ExitRailsConfig,
    pos: PositionState,
    current_price: float,
    current_bar: int,
    session_bar_index: Optional[int] = None,
) -> Optional[str]:
    """返回退出原因字符串;None 表示继续持有。current_price 为 MTM 估值价。"""
    roi = pos.update(current_price)
    held = current_bar - pos.entry_bar

    if (
        cfg.eod_close_bar_index is not None
        and session_bar_index is not None
        and session_bar_index >= cfg.eod_close_bar_index
    ):
        return "EOD_CLOSE"

    if roi <= cfg.hard_stop_roi:
        return "HARD_STOP"
    if roi <= cfg.soft_stop_roi:
        return "SOFT_STOP"

    if (
        cfg.early_stop_bars is not None
        and held >= cfg.early_stop_bars
        and roi <= cfg.early_stop_roi
    ):
        return "EARLY_STOP"

    if held >= cfg.max_hold_bars:
        return "MAX_HOLD"
    if held >= cfg.time_stop_bars and roi < cfg.time_stop_min_roi:
        return "TIME_STOP"

    if pos.max_roi >= cfg.trailing_trigger_roi:
        if roi < pos.max_roi * cfg.trailing_keep_ratio:
            return "TRAILING"

    lf = ladder_floor(cfg, pos.max_roi)
    if lf > float("-inf") and roi < lf:
        return "STEP_PROTECT"

    if pos.max_roi >= cfg.flash_trigger_roi and roi <= cfg.flash_exit_roi:
        return "FLASH_PROTECT"

    return None


def check_disaster_stop(
    cfg: ExitRailsConfig,
    pos: PositionState,
    tick_price: float,
) -> Optional[str]:
    """
    tick 级灾难止损 —— 实盘高频监控循环里唯一允许调用的 rails 入口。

    刻意无状态:不调用 pos.update(),分钟内影线不污染 max_roi,
    trailing/阶梯/flash 的语义仍严格以分钟收盘为准(与回放一致)。
    """
    if cfg.disaster_stop_roi is None:
        return None
    roi = tick_price / pos.entry_price - 1.0
    if roi <= cfg.disaster_stop_roi:
        return "DISASTER_STOP"
    return None

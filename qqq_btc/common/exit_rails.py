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
  tick 级监控只允许调用 `check_tick_stops`(不更新分钟 max_roi):
    - disaster_stop:跳空/闪崩兜底
    - tick_fast_hard:分钟内快速亏损
    - tick_profit_*:独立 tick_peak 浮盈保护(冲高回落),与分钟棘轮解耦
  禁止在 tick 上调用完整 check_exit(会把影线写入 max_roi)。
  建议实盘 tick MTM 先做中价平滑再喂,避免单笔异常报价触发。
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import List, Optional, Tuple


@dataclass(frozen=True)
class ExitRailsConfig:
    hard_stop_roi: float = -0.12          # 期权权利金 ROI 硬止损
    soft_stop_roi: float = -0.08          # 软止损
    # 孵化期:持有未满 N bar 时只跑 hard_stop(不做 soft/利润保护),给 thesis 时间
    profit_protect_min_bars: Optional[int] = None
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
    # tick 级止损(check_tick_stops 专用,None=不启用)。不污染分钟 max_roi。
    # tick_fast_hard:介于分钟 hard 与 disaster 之间,吃掉「分钟内已亏透、收盘才看见」。
    # disaster:必须显著深于 fast_hard,只为跳空/闪崩兜底。
    tick_fast_hard_roi: Optional[float] = None
    tick_fast_hard_smooth_n: int = 5
    disaster_stop_roi: Optional[float] = None
    disaster_smooth_n: int = 3
    # tick 浮盈保护:独立 tick_peak(与分钟 max_roi 解耦)。
    # peak 达 trigger 后,回落到 peak*keep 以下则平仓 —— 吃「一分钟内翻倍再吐光」。
    tick_profit_trigger_roi: Optional[float] = None
    tick_profit_keep_ratio: float = 0.50
    tick_profit_smooth_n: int = 3
    # 可选:按 tick_peak 的阶梯 floor(与分钟 ladder 同语义,但只读 tick_peak)
    tick_profit_ladder: Tuple[Tuple[float, float], ...] = ()
    # --- 波动自适应缩放(vol_scale_ref=None 关闭) ---
    # 动机:阈值是按常态波动校准的静态数;高波动月(2026-06 尾部 p99 从 2 拉到 10)
    # 会"涨一点就 STEP_PROTECT 出场、跌一段就 HARD_STOP",把右尾剪掉、左尾留下。
    # 做法:入场时用当日过去 vol_scale_window 根 bar 的权利金分钟收益 std,
    # 除以历史参考 vol_scale_ref 得 scale(截断到 [min,max]),
    # 所有 ROI 阈值乘 scale 后对该仓位生效(持仓期不变,比例参数不动)。
    vol_scale_ref: Optional[float] = None
    vol_scale_window: int = 60
    vol_scale_min_obs: int = 20
    vol_scale_min: float = 0.75
    vol_scale_max: float = 2.5
    # True = 只缩放利润保护侧(ladder/trailing/flash/time_stop_min/tick_profit),
    # 止损(hard/soft/early/tick_fast/disaster)保持校准值。
    # 依据:2026-06 复盘,亏损主因是 STEP_PROTECT 剪掉右尾(-1.60),
    # 而放深止损只会加大左尾(全量缩放最差单笔 -0.40 vs -0.36)。
    vol_scale_profit_only: bool = False


def scale_rails(cfg: ExitRailsConfig, scale: float) -> ExitRailsConfig:
    """
    按波动缩放全部 ROI 阈值,返回新配置(bar 数与 keep_ratio 等比例参数不动)。

    scale=1 返回原配置。scale>1 = 高波动日:止损更深、利润档位更高,
    使"阈值 / 当日噪声尺度"保持与校准期一致。
    """
    if not (scale > 0) or scale == 1.0:
        return cfg
    s = float(scale)

    def _opt(v: Optional[float]) -> Optional[float]:
        return None if v is None else v * s

    def _ladder(lad: Tuple[Tuple[float, float], ...]) -> Tuple[Tuple[float, float], ...]:
        return tuple((trig * s, fl * s) for trig, fl in lad)

    profit_fields = dict(
        time_stop_min_roi=cfg.time_stop_min_roi * s,
        trailing_trigger_roi=cfg.trailing_trigger_roi * s,
        ladder=_ladder(cfg.ladder),
        flash_trigger_roi=cfg.flash_trigger_roi * s,
        flash_exit_roi=cfg.flash_exit_roi * s,
        tick_profit_trigger_roi=_opt(cfg.tick_profit_trigger_roi),
        tick_profit_ladder=_ladder(cfg.tick_profit_ladder),
    )
    if cfg.vol_scale_profit_only:
        return replace(cfg, **profit_fields)
    return replace(
        cfg,
        hard_stop_roi=cfg.hard_stop_roi * s,
        soft_stop_roi=cfg.soft_stop_roi * s,
        early_stop_roi=cfg.early_stop_roi * s,
        tick_fast_hard_roi=_opt(cfg.tick_fast_hard_roi),
        disaster_stop_roi=_opt(cfg.disaster_stop_roi),
        **profit_fields,
    )


def vol_scale_from_returns(cfg: ExitRailsConfig, minute_returns: List[float]) -> float:
    """
    入场时的波动缩放因子:当日近端权利金分钟收益 std / 历史参考,截断到配置区间。

    观测不足(开盘初段)返回 1.0 —— 用校准期默认阈值,不猜。
    """
    if cfg.vol_scale_ref is None or cfg.vol_scale_ref <= 0:
        return 1.0
    import numpy as np

    arr = np.asarray(minute_returns, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < int(cfg.vol_scale_min_obs):
        return 1.0
    sd = float(arr[-int(cfg.vol_scale_window):].std())
    if not (sd > 0):
        return 1.0
    return float(min(max(sd / cfg.vol_scale_ref, cfg.vol_scale_min), cfg.vol_scale_max))


@dataclass
class PositionState:
    entry_price: float
    entry_bar: int
    max_roi: float = 0.0
    # 仅 check_tick_stops 更新;分钟 check_exit 绝不读写
    tick_peak_roi: float = 0.0

    def update(self, current_price: float) -> float:
        roi = current_price / self.entry_price - 1.0
        if roi > self.max_roi:
            self.max_roi = roi
        return roi


def ladder_floor_from(
    ladder: Tuple[Tuple[float, float], ...], max_roi: float
) -> float:
    """累积 ratchet 阶梯底价:所有 max_roi 已触发的档位取 floor 最大值。"""
    floor = float("-inf")
    for trigger, f in sorted(ladder, key=lambda x: x[0]):
        if max_roi >= trigger:
            floor = max(floor, f)
    return floor


def ladder_floor(cfg: ExitRailsConfig, max_roi: float) -> float:
    """
    累积 ratchet 阶梯底价:所有 max_roi 已触发的档位取 floor 最大值。

    旧实现只取「最高一档」且 floor 偏低(9DTE 手拍),0DTE 下 peak 12% 却允许
    回吐到 3%。正确语义:peak 12% 时应锁 8%(若存在 (0.12,0.08) 档)。
    """
    return ladder_floor_from(cfg.ladder, max_roi)


def check_forced_time_exit(
    cfg: ExitRailsConfig,
    *,
    entry_bar: int,
    current_bar: int,
    session_bar_index: Optional[int] = None,
) -> Optional[str]:
    """无需 MTM 的强制时间退出；报价缺失时仍必须执行。"""
    if (
        cfg.eod_close_bar_index is not None
        and session_bar_index is not None
        and session_bar_index >= cfg.eod_close_bar_index
    ):
        return "EOD_CLOSE"
    if current_bar - entry_bar >= cfg.max_hold_bars:
        return "MAX_HOLD"
    return None


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
    incubating = (
        cfg.profit_protect_min_bars is not None
        and held < int(cfg.profit_protect_min_bars)
    )

    if (
        cfg.eod_close_bar_index is not None
        and session_bar_index is not None
        and session_bar_index >= cfg.eod_close_bar_index
    ):
        return "EOD_CLOSE"

    # 硬止损始终生效。
    # 软止损也始终生效:孵化期若只留 hard,会拖到 -25%~-28% 才走
    # (Jul1 09:46 PUT: held=6 已 -22% 本该 soft,却拖到 held=13 才 HARD)。
    # 利润保护(阶梯/trailing/flash)仍仅孵化结束后启用,避免刚开仓小反弹被 STEP 剪掉。
    if roi <= cfg.hard_stop_roi:
        return "HARD_STOP"
    if roi <= cfg.soft_stop_roi:
        return "SOFT_STOP"

    if incubating:
        # 孵化期内仍允许 early_stop(持有足够 bar 且持续亏),与 soft 同属止损侧
        if (
            cfg.early_stop_bars is not None
            and held >= int(cfg.early_stop_bars)
            and roi <= cfg.early_stop_roi
        ):
            return "EARLY_STOP"
        return None

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
    """兼容旧调用:仅灾难档。新代码请用 check_tick_stops。"""
    return check_tick_stops(cfg, pos, tick_price, disaster_only=True)


def check_tick_stops(
    cfg: ExitRailsConfig,
    pos: PositionState,
    tick_price: float,
    *,
    disaster_only: bool = False,
) -> Optional[str]:
    """
    tick 级风险/浮盈轨 —— 实盘高频监控循环里唯一允许调用的 rails 入口。

    不调用 pos.update():分钟 max_roi / trailing / 阶梯仍严格以分钟收盘为准。
    浮盈侧只维护 pos.tick_peak_roi,与分钟棘轮解耦。

    判定顺序:disaster → tick_fast_hard → tick 浮盈(trail / step)。
    """
    roi = tick_price / pos.entry_price - 1.0

    if cfg.disaster_stop_roi is not None and roi <= cfg.disaster_stop_roi:
        return "DISASTER_STOP"
    if (
        not disaster_only
        and cfg.tick_fast_hard_roi is not None
        and roi <= cfg.tick_fast_hard_roi
    ):
        return "TICK_FAST_HARD"

    if disaster_only:
        return None

    # 浮盈保护:先更新独立 peak,再判回撤/阶梯
    if roi > pos.tick_peak_roi:
        pos.tick_peak_roi = roi

    if (
        cfg.tick_profit_trigger_roi is not None
        and pos.tick_peak_roi >= cfg.tick_profit_trigger_roi
        and roi < pos.tick_peak_roi * cfg.tick_profit_keep_ratio
    ):
        return "TICK_PROFIT_TRAIL"

    if cfg.tick_profit_ladder:
        lf = ladder_floor_from(cfg.tick_profit_ladder, pos.tick_peak_roi)
        if lf > float("-inf") and roi < lf:
            return "TICK_PROFIT_STEP"

    return None

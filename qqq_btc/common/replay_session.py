#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Replay 状态机 —— strict replay / event replay / live 共用同一决策状态。

设计约束:
  - 单 bar 调用 `on_minute_bar`,禁止向量化批量决策
  - tick 级只允许 `on_tick` → check_tick_stops(MTM 契约: fast_hard + disaster)
  - 成交/佣金口径统一走 FillModel
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import numpy as np

from .entry_decision import EntryDecision, choose_entry
from .exit_rails import (
    ExitRailsConfig,
    PositionState,
    check_exit,
    check_forced_time_exit,
    check_spot_thesis_invalidate,
    check_tick_stops,
    maybe_bounce_cut_rails,
    scale_rails,
    vol_scale_from_returns,
)
from .fill_model import OptionSpreadFillModel, PerpFillModel
from .replay_types import ReplayConfig, ReplayResult, Trade


class BarPhase(str, Enum):
    """分钟 bar 内相位 —— 对齐 S4 ExecutionWindow 契约。"""

    OPEN = "open"    # 分钟首 tick:发信号 / first_tick 入场
    CLOSE = "close"  # 分钟末 tick:rails 退出 / minute_close 入场(L1 默认)


@dataclass(frozen=True)
class SessionQuotes:
    """单时刻各腿盘口(分钟收盘或 tick 快照)。"""

    call_bid: float
    call_ask: float
    call_spread_pct: float = 0.0
    put_bid: Optional[float] = None
    put_ask: Optional[float] = None
    put_spread_pct: Optional[float] = None

    @classmethod
    def from_row(cls, row, prefix: str = "exec_call") -> "SessionQuotes":
        bid = float(row.get(f"{prefix}_bid", row.get("exec_call_bid", np.nan)))
        ask = float(row.get(f"{prefix}_ask", row.get("exec_call_ask", np.nan)))
        sp = row.get(f"{prefix}_spread_pct", row.get("exec_call_spread_pct"))
        sp_f = float(sp) if sp is not None and np.isfinite(sp) else cls._spread_pct(bid, ask)
        put_bid = row.get("exec_put_bid")
        put_ask = row.get("exec_put_ask")
        put_sp = row.get("exec_put_spread_pct")
        return cls(
            call_bid=bid,
            call_ask=ask,
            call_spread_pct=sp_f,
            put_bid=float(put_bid) if put_bid is not None and np.isfinite(put_bid) else None,
            put_ask=float(put_ask) if put_ask is not None and np.isfinite(put_ask) else None,
            put_spread_pct=float(put_sp) if put_sp is not None and np.isfinite(put_sp) else None,
        )

    @classmethod
    def from_perp(cls, mark: float) -> "SessionQuotes":
        return cls(call_bid=mark, call_ask=mark, call_spread_pct=0.0)

    @staticmethod
    def _spread_pct(bid: float, ask: float) -> float:
        if bid > 0 and ask > bid:
            mid = (bid + ask) / 2.0
            return float((ask - bid) / mid)
        return float("inf")

    def has_put(self) -> bool:
        return (
            self.put_bid is not None
            and self.put_ask is not None
            and self.put_bid > 0
            and self.put_ask > 0
        )

    def mid(self, leg: str) -> float:
        if leg == "PUT":
            if not self.has_put():
                return float("nan")
            return (self.put_bid + self.put_ask) / 2.0  # type: ignore[operator]
        if leg == "STRADDLE":
            cm = self.mid("CALL")
            pm = self.mid("PUT")
            return cm + pm if np.isfinite(cm) and np.isfinite(pm) else float("nan")
        if leg == "PERP":
            return self.call_bid
        cb, ca = self.call_bid, self.call_ask
        return (cb + ca) / 2.0 if cb > 0 and ca > 0 else float("nan")

    def entry_fill(
        self,
        leg: str,
        fill_model: Union[OptionSpreadFillModel, PerpFillModel],
    ) -> float:
        if isinstance(fill_model, PerpFillModel):
            return float(fill_model.entry_fill(self.call_bid))
        if leg == "PUT":
            return float(fill_model.entry_fill(self.put_bid, self.put_ask))  # type: ignore[arg-type]
        if leg == "STRADDLE":
            c = fill_model.entry_fill(self.call_bid, self.call_ask)
            p = fill_model.entry_fill(self.put_bid, self.put_ask)  # type: ignore[arg-type]
            return float(c + p)
        return float(fill_model.entry_fill(self.call_bid, self.call_ask))

    def exit_fill(
        self,
        leg: str,
        fill_model: Union[OptionSpreadFillModel, PerpFillModel],
    ) -> float:
        if isinstance(fill_model, PerpFillModel):
            return float(fill_model.exit_fill(self.call_bid))
        if leg == "PUT":
            return float(fill_model.exit_fill(self.put_bid, self.put_ask))  # type: ignore[arg-type]
        if leg == "STRADDLE":
            c = fill_model.exit_fill(self.call_bid, self.call_ask)
            p = fill_model.exit_fill(self.put_bid, self.put_ask)  # type: ignore[arg-type]
            return float(c + p)
        return float(fill_model.exit_fill(self.call_bid, self.call_ask))

    def spread_pct(self, leg: str) -> float:
        if leg == "PUT":
            if self.put_spread_pct is not None and np.isfinite(self.put_spread_pct):
                return float(self.put_spread_pct)
            return self._spread_pct(self.put_bid or 0, self.put_ask or 0)  # type: ignore[arg-type]
        if leg == "STRADDLE":
            return max(self.call_spread_pct, self.put_spread_pct or 0.0)
        return self.call_spread_pct


@dataclass
class SessionSignal:
    edge: Optional[float] = None
    call_edge: Optional[float] = None
    put_edge: Optional[float] = None
    straddle_edge: Optional[float] = None
    edge_q10: Optional[float] = None
    # PUT 腿行情开关信号(如归一化 vix_level);None=缺失,门控开启时视为不通过
    put_gate: Optional[float] = None
    # 日前 lookback 的因果 VIXY z；跨日 PUT quarantine 的低波 regime 条件。
    regime_vix_z: Optional[float] = None
    # 前一已完成 VX 日线的期限结构 VX2/VX1-1。
    vx_curve_slope: Optional[float] = None
    open30_ret: Optional[float] = None
    open30_max_ret: Optional[float] = None
    open30_peak_dd: Optional[float] = None
    spot_ret_5bar: Optional[float] = None
    # 早盘低置信 PUT 的跨资产确认输入。
    spot_ret_15bar: Optional[float] = None
    vix_ret_15bar: Optional[float] = None
    # 30min 拟合趋势收益(trend_fit_ret_30m);PUT 趋势对齐门控输入
    trend_ret_30m: Optional[float] = None
    # 30min 拟合优度(trend_fit_r2_30m);CALL 震荡过滤门控输入
    trend_r2_30m: Optional[float] = None
    # vix_proxy 30min 方向反转次数;V0 regime 弃权门控输入
    vix_reversal_count_30m: Optional[float] = None
    # 当日开盘至当前 bar 现货收益;CALL 追涨洗盘门控输入
    spot_day_ret: Optional[float] = None
    # 30min 现货振幅;CALL 局部尖刺门控输入
    spot_range_30m: Optional[float] = None
    # 当日振幅位置 / BB 宽度;CALL TREND_SPENT 门控输入
    day_range_pos: Optional[float] = None
    bb_width: Optional[float] = None
    # 方向头概率;block_when_side_none / require_leg_*_agree 门控输入
    best_side_put_prob: Optional[float] = None
    best_side_none_prob: Optional[float] = None
    best_side_call_prob: Optional[float] = None
    spot_down_prob: Optional[float] = None
    spot_flat_prob: Optional[float] = None
    spot_up_prob: Optional[float] = None
    # 现货收盘价;SPOT_THESIS 证伪用
    spot_close: Optional[float] = None
    # vwap_log_return;bounce-cut 用 1m jump(与上一根差分在 session 内算)
    vwap_log_return: Optional[float] = None


@dataclass
class ReplayEvent:
    kind: str  # SIGNAL | ENTER | EXIT | DISASTER_EXIT
    ts: Any
    bar_index: int
    leg: str
    price: Optional[float] = None
    edge: Optional[float] = None
    reason: Optional[str] = None
    net_return: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OpenPosition:
    leg: str
    entry_price: float
    entry_bar: int
    entry_ts: Any
    signal_edge: float
    state: PositionState
    commission_mult: float = 1.0
    # 入场时按当日波动缩放后的护栏(vol_scale_ref 未启用时为 None → 用全局配置)
    rails: Optional[ExitRailsConfig] = None
    vol_scale: float = 1.0
    # 强制时间退出遇到空盘口时，用最近一个因果分钟 mid 估值；至少为入场成交价。
    last_valid_mtm: float = 0.0
    # 入场时现货价;SPOT_THESIS 相对入场证伪
    entry_spot: Optional[float] = None
    # 入场时锁定实际账户仓位，避免跨日状态变化影响已开仓。
    position_frac: float = 1.0


class ReplaySession:
    """逐 bar / 逐 tick 事件状态机。"""

    def __init__(
        self,
        replay_cfg: ReplayConfig,
        rails_cfg: ExitRailsConfig,
        fill_model: Union[OptionSpreadFillModel, PerpFillModel],
        *,
        dual_mode: bool = False,
        default_leg: str = "CALL",
        is_option: bool = True,
        signal_only: bool = False,
    ):
        self.replay_cfg = replay_cfg
        self.rails_cfg = rails_cfg
        self.fill_model = fill_model
        self.dual_mode = dual_mode
        self.default_leg = default_leg
        self.is_option = is_option
        self.signal_only = signal_only

        self.result = ReplayResult(
            position_frac=float(getattr(replay_cfg, "position_frac", 1.0) or 1.0)
        )
        self.equity = 1.0
        self.position: Optional[OpenPosition] = None
        self.pending_entry_bar: Optional[int] = None
        self.pending_edge = 0.0
        self.pending_leg = default_leg
        self.cooldown_until = -1
        self.streak_cooldown_until = -1
        self.tick_stop_cooldown_until = -1
        self.tick_stopped_legs: set[str] = set()
        # 当日已亏损的腿(loss_reentry_edge_mult 用)
        self.loss_legs_today: set[str] = set()
        # 跨日腿级隔离：按账户仓位复利累计当日各腿贡献；日切时决定下一日锁腿。
        self.day_leg_equity: Dict[str, float] = {"CALL": 1.0, "PUT": 1.0}
        self.cross_day_quarantined_legs: set[str] = set()
        self.day_account_equity = 1.0
        self.cross_day_all_leg_defense = False
        # SPOT_THESIS 后短期同腿锁:leg -> 禁开至该 bar_index(含)
        self.leg_lock_until: Dict[str, int] = {}
        # 早盘 open30 结构否决延长至该 session_bar(含)
        self.put_structure_veto_until: Optional[int] = None
        # 开盘急跌后低 R² 震荡态；一旦触发保持到日切。
        self.vixy_open_shock_regime_active = False

        self.cur_day = None
        self.trades_today = 0
        self.straddles_today = 0
        self.day_pnl = 0.0
        self.day_halted = False
        self.loss_streak = 0
        self.events: List[ReplayEvent] = []
        # 当日交易腿分钟 mid 序列(波动自适应护栏用,分钟收盘更新)
        self._day_mids: List[float] = []
        # 当日现货收盘序列(SPOT_THESIS 证伪)
        self._spot_closes: List[float] = []
        # 当日 vwap_log_return(bounce-cut 1m jump)
        self._vwap_lrs: List[float] = []
        # 入场窗 bar 的 edge 滚动缓冲(跨日,滚动分位阈值用;call/put 分数尺度不同,分开维护)
        _q_on = getattr(replay_cfg, "entry_quantile", None) is not None
        self._edge_buf: Optional[deque] = (
            deque(maxlen=int(replay_cfg.entry_quantile_window)) if _q_on else None
        )
        self._put_edge_buf: Optional[deque] = (
            deque(maxlen=int(replay_cfg.entry_quantile_window)) if _q_on else None
        )

    def on_minute_bar(
        self,
        bar_index: int,
        ts: Any,
        session_bar: Optional[int],
        quotes: SessionQuotes,
        signal: SessionSignal,
        *,
        day_key: Any = None,
        phase: BarPhase = BarPhase.CLOSE,
        allow_signal: bool = True,
        allow_entry: bool = True,
    ) -> List[ReplayEvent]:
        """
        分钟 bar 事件。

        CLOSE(默认,L1): 退出 / 延迟成交 / 发信号 —— 全在分钟末。
        OPEN(S4 对齐): 仅发信号 + first_tick 入场;rails 退出仍在 CLOSE。
        """
        emitted: List[ReplayEvent] = []
        if day_key is not None and day_key != self.cur_day:
            self._reset_day(
                day_key,
                regime_vix_z=signal.regime_vix_z,
                vx_curve_slope=signal.vx_curve_slope,
            )

        if phase == BarPhase.CLOSE:
            # regime 是会话观察状态，持仓期间也必须持续更新；否则首次入场跨过
            # 检测窗时，平仓后永远无法识别该状态。
            self._maybe_trip_vixy_open_shock_regime(session_bar, signal)
            mid = quotes.mid(self.default_leg)
            if np.isfinite(mid) and mid > 0:
                self._day_mids.append(float(mid))
            if signal.spot_close is not None and np.isfinite(signal.spot_close) and float(signal.spot_close) > 0:
                self._spot_closes.append(float(signal.spot_close))
            if (
                signal.vwap_log_return is not None
                and np.isfinite(signal.vwap_log_return)
            ):
                self._vwap_lrs.append(float(signal.vwap_log_return))
            if (
                self._edge_buf is not None
                and self.replay_cfg.session_allows_entry(session_bar)
                and bar_index > self.tick_stop_cooldown_until
            ):
                # dual 模式 CALL 腿比较的是 call_edge,分位缓冲须跟踪同一分数
                main_edge = (
                    signal.call_edge
                    if self.dual_mode and signal.call_edge is not None
                    else signal.edge
                )
                _call_r2_min = self.replay_cfg.call_trend_r2_min
                _call_r2_blocked = (
                    _call_r2_min is not None
                    and signal.trend_r2_30m is not None
                    and np.isfinite(signal.trend_r2_30m)
                    and signal.trend_r2_30m < float(_call_r2_min)
                )
                _chase_vix = self.replay_cfg.call_chase_vix_rev_min
                _chase_ret = self.replay_cfg.call_chase_spot_day_ret_min
                _call_chase_blocked = (
                    _chase_vix is not None
                    and signal.spot_day_ret is not None
                    and np.isfinite(signal.spot_day_ret)
                    and signal.spot_day_ret > float(_chase_ret)
                    and signal.vix_reversal_count_30m is not None
                    and np.isfinite(signal.vix_reversal_count_30m)
                    and signal.vix_reversal_count_30m >= float(_chase_vix)
                )
                _spike_min = self.replay_cfg.call_spike_range30_min
                _call_spike_blocked = (
                    _spike_min is not None
                    and signal.spot_range_30m is not None
                    and np.isfinite(signal.spot_range_30m)
                    and signal.spot_range_30m >= float(_spike_min)
                )
                _t_spot = self.replay_cfg.call_timing_spot_min
                _t_bar = self.replay_cfg.call_timing_max_bar
                _t_vix = self.replay_cfg.call_timing_vix_min
                _call_timing_blocked = (
                    _t_bar is not None
                    and _t_spot is not None
                    and _t_vix is not None
                    and session_bar is not None
                    and session_bar < int(_t_bar)
                    and signal.spot_day_ret is not None
                    and np.isfinite(signal.spot_day_ret)
                    and signal.spot_day_ret > float(_t_spot)
                    and signal.vix_reversal_count_30m is not None
                    and np.isfinite(signal.vix_reversal_count_30m)
                    and signal.vix_reversal_count_30m >= float(_t_vix)
                )
                _spent_drp = self.replay_cfg.call_spent_day_range_pos_min
                _spent_bb = self.replay_cfg.call_spent_bb_width_max
                _spent_min_bar = self.replay_cfg.call_spent_min_session_bar
                _call_spent_blocked = (
                    _spent_drp is not None
                    and _spent_bb is not None
                    and (
                        _spent_min_bar is None
                        or (session_bar is not None and session_bar >= int(_spent_min_bar))
                    )
                    and signal.day_range_pos is not None
                    and np.isfinite(signal.day_range_pos)
                    and signal.bb_width is not None
                    and np.isfinite(signal.bb_width)
                    and float(signal.day_range_pos) >= float(_spent_drp)
                    and float(signal.bb_width) <= float(_spent_bb)
                )
                _regime_max = self.replay_cfg.regime_vix_reversal_max
                _regime_blocked = (
                    _regime_max is not None
                    and signal.vix_reversal_count_30m is not None
                    and np.isfinite(signal.vix_reversal_count_30m)
                    and signal.vix_reversal_count_30m > float(_regime_max)
                )
                if (
                    main_edge is not None
                    and np.isfinite(main_edge)
                    and not _call_r2_blocked
                    and not _call_chase_blocked
                    and not _call_spike_blocked
                    and not _call_timing_blocked
                    and not _call_spent_blocked
                    and not _regime_blocked
                ):
                    self._edge_buf.append(float(main_edge))
                # 趋势门控拦截的 bar 不进 PUT 分位缓冲:这些 bar 的 put 分数
                # 不代表可交易机会,混入会虚高 p80 动态阈值、误杀顺势 PUT
                _pt_max = self.replay_cfg.put_trend_max_ret
                _put_trend_blocked = (
                    _pt_max is not None
                    and signal.trend_ret_30m is not None
                    and np.isfinite(signal.trend_ret_30m)
                    and signal.trend_ret_30m > float(_pt_max)
                )
                _put_late_bar = self.replay_cfg.put_late_session_bar
                _put_late_blocked = (
                    _put_late_bar is not None
                    and session_bar is not None
                    and session_bar > int(_put_late_bar)
                )
                _put_spot_min = self.replay_cfg.put_spot_day_ret_min
                _put_spot_blocked = (
                    _put_spot_min is not None
                    and signal.spot_day_ret is not None
                    and np.isfinite(signal.spot_day_ret)
                    and signal.spot_day_ret > float(_put_spot_min)
                )
                if (
                    self._put_edge_buf is not None
                    and not _put_trend_blocked
                    and not _put_late_blocked
                    and not _put_spot_blocked
                    and not _regime_blocked
                    and signal.put_edge is not None
                    and np.isfinite(signal.put_edge)
                ):
                    self._put_edge_buf.append(float(signal.put_edge))

        if phase == BarPhase.OPEN:
            if self.position is not None:
                return emitted
            if (
                allow_entry
                and self.pending_entry_bar is not None
                and bar_index >= self.pending_entry_bar
            ):
                ev = self._try_entry(bar_index, ts, quotes)
                if ev:
                    emitted.append(ev)
                return emitted
            if self._blocked_for_entry(bar_index):
                return emitted
            if not allow_signal:
                return emitted
            return emitted + self._try_signal(bar_index, ts, session_bar, quotes, signal)

        # --- CLOSE: rails 退出 ---
        if self.position is not None:
            ev = self._try_minute_exit(bar_index, ts, session_bar, quotes)
            if ev:
                emitted.append(ev)
            return emitted

        if (
            allow_entry
            and self.pending_entry_bar is not None
            and bar_index >= self.pending_entry_bar
        ):
            ev = self._try_entry(bar_index, ts, quotes)
            if ev:
                emitted.append(ev)
            return emitted

        if self._blocked_for_entry(bar_index):
            return emitted

        if not allow_signal:
            return emitted
        return emitted + self._try_signal(bar_index, ts, session_bar, quotes, signal)

    def _try_signal(
        self,
        bar_index: int,
        ts: Any,
        session_bar: Optional[int],
        quotes: SessionQuotes,
        signal: SessionSignal,
    ) -> List[ReplayEvent]:
        if self.cross_day_all_leg_defense:
            defense_start = getattr(
                self.replay_cfg, "next_day_all_leg_defense_entry_start_bar", None
            )
            if (
                defense_start is not None
                and session_bar is not None
                and session_bar < int(defense_start)
            ):
                return []
            defense_q10 = getattr(
                self.replay_cfg, "next_day_all_leg_defense_edge_q10_floor", None
            )
            if defense_q10 is not None:
                q10 = signal.edge_q10
                if (
                    q10 is None
                    or not np.isfinite(q10)
                    or float(q10) < float(defense_q10)
                ):
                    return []
        decision = self._choose_entry(session_bar, quotes, signal)
        if decision is None:
            return []
        if decision.leg in self.tick_stopped_legs:
            return []
        if decision.leg in self.cross_day_quarantined_legs:
            return []
        # STRADDLE 含 PUT 暴露，PUT quarantine 日不可借跨式绕过。
        if (
            decision.leg == "STRADDLE"
            and "PUT" in self.cross_day_quarantined_legs
        ):
            return []
        lock_until = self.leg_lock_until.get(str(decision.leg).upper())
        if lock_until is not None and bar_index <= int(lock_until):
            return []
        sp = quotes.spread_pct(decision.leg)
        if not (np.isfinite(sp) and sp <= self.replay_cfg.max_spread_pct):
            return []
        if self.signal_only:
            ev = ReplayEvent(
                kind="SIGNAL",
                ts=ts,
                bar_index=bar_index,
                leg=decision.leg,
                edge=decision.edge,
                extra={"threshold": decision.threshold, "signal_only": True},
            )
            self.events.append(ev)
            return [ev]
        delay = int(self.replay_cfg.entry_delay_bars or 0)
        if bool(getattr(self.replay_cfg, "immediate_entry", False)):
            self.pending_edge = decision.edge
            self.pending_leg = decision.leg
            ent = self._try_entry(bar_index, ts, quotes)
            if ent:
                return [ent]
            return []
        self.pending_entry_bar = bar_index + delay
        self.pending_edge = decision.edge
        self.pending_leg = decision.leg
        ev = ReplayEvent(
            kind="SIGNAL",
            ts=ts,
            bar_index=bar_index,
            leg=decision.leg,
            edge=decision.edge,
            extra={"pending_bar": self.pending_entry_bar, "threshold": decision.threshold},
        )
        self.events.append(ev)
        return [ev]

    def on_tick(
        self,
        bar_index: int,
        ts: Any,
        quotes: SessionQuotes,
        *,
        smoothed_mtm: Optional[float] = None,
        disaster_only: bool = False,
    ) -> List[ReplayEvent]:
        """秒级 tick:disaster + 可选 tick_fast_hard(均无状态,不污染 max_roi)。"""
        if self.position is None:
            return []
        mtm = smoothed_mtm if smoothed_mtm is not None else quotes.mid(self.position.leg)
        if not (np.isfinite(mtm) and mtm > 0):
            return []
        reason = check_tick_stops(
            self.position.rails or self.rails_cfg,
            self.position.state,
            float(mtm),
            disaster_only=disaster_only,
        )
        if reason is None:
            return []
        return [self._close_position(bar_index, ts, quotes, reason, disaster=True)]

    def _reset_day(
        self,
        day_key: Any,
        *,
        regime_vix_z: Optional[float] = None,
        vx_curve_slope: Optional[float] = None,
    ) -> None:
        put_quarantine_loss = getattr(
            self.replay_cfg, "next_day_put_quarantine_loss", None
        )
        put_quarantine_vix_z_max = getattr(
            self.replay_cfg, "next_day_put_quarantine_vix_z_max", None
        )
        put_quarantine_vx_slope_min = getattr(
            self.replay_cfg, "next_day_put_quarantine_vx_slope_min", None
        )
        prev_put_contribution = self.day_leg_equity.get("PUT", 1.0) - 1.0
        prev_account_contribution = self.day_account_equity - 1.0
        regime_ok = put_quarantine_vix_z_max is None or (
            regime_vix_z is not None
            and np.isfinite(regime_vix_z)
            and float(regime_vix_z) <= float(put_quarantine_vix_z_max)
        )
        vx_curve_ok = put_quarantine_vx_slope_min is None or (
            vx_curve_slope is not None
            and np.isfinite(vx_curve_slope)
            and float(vx_curve_slope) >= float(put_quarantine_vx_slope_min)
        )
        if (
            self.cur_day is not None
            and put_quarantine_loss is not None
            and np.isfinite(prev_put_contribution)
            and prev_put_contribution <= float(put_quarantine_loss)
            and regime_ok
            and vx_curve_ok
        ):
            self.cross_day_quarantined_legs = {"PUT"}
        else:
            self.cross_day_quarantined_legs.clear()

        all_leg_defense_loss = getattr(
            self.replay_cfg, "next_day_all_leg_defense_loss", None
        )
        all_leg_defense_vx_min = getattr(
            self.replay_cfg, "next_day_all_leg_defense_vx_slope_min", None
        )
        all_leg_defense_vx_ok = all_leg_defense_vx_min is None or (
            vx_curve_slope is not None
            and np.isfinite(vx_curve_slope)
            and float(vx_curve_slope) >= float(all_leg_defense_vx_min)
        )
        self.cross_day_all_leg_defense = bool(
            self.cur_day is not None
            and all_leg_defense_loss is not None
            and np.isfinite(prev_account_contribution)
            and prev_account_contribution <= float(all_leg_defense_loss)
            and all_leg_defense_vx_ok
        )

        self.cur_day = day_key
        self.trades_today = 0
        self.straddles_today = 0
        self.day_pnl = 0.0
        self.day_leg_equity = {"CALL": 1.0, "PUT": 1.0}
        self.day_account_equity = 1.0
        self.day_halted = False
        self.pending_entry_bar = None
        self.loss_streak = 0
        self.tick_stop_cooldown_until = -1
        self.tick_stopped_legs.clear()
        self.loss_legs_today.clear()
        self.leg_lock_until.clear()
        self.put_structure_veto_until = None
        self.vixy_open_shock_regime_active = False
        self._day_mids = []
        self._spot_closes = []
        self._vwap_lrs: List[float] = []

    def _entry_rails(
        self,
        *,
        leg: Optional[str] = None,
        signal: Optional[SessionSignal] = None,
    ) -> tuple:
        """入场时刻的 (缩放护栏, scale)。未启用波动自适应时原样返回。

        PUT + bounce onset 时再挂仓位级 SPOT_THESIS(open30 挡不住的减亏层)。
        """
        if self.rails_cfg.vol_scale_ref is None:
            rails, scale = self.rails_cfg, 1.0
        else:
            mids = np.asarray(self._day_mids, dtype=float)
            rets = (mids[1:] / mids[:-1] - 1.0).tolist() if mids.size >= 2 else []
            scale = vol_scale_from_returns(self.rails_cfg, rets)
            rails = scale_rails(self.rails_cfg, scale)
        if leg is not None:
            vwap_jump = None
            if len(self._vwap_lrs) >= 2:
                vwap_jump = float(self._vwap_lrs[-1]) - float(self._vwap_lrs[-2])
            elif (
                signal is not None
                and signal.vwap_log_return is not None
                and len(self._vwap_lrs) == 1
            ):
                # 仅一根时无法算 jump → 不触发 bounce-cut
                vwap_jump = None
            rails = maybe_bounce_cut_rails(
                rails,
                leg=str(leg),
                vwap_jump=vwap_jump,
                spot_closes=self._spot_closes,
            )
        return rails, scale

    def _blocked_for_entry(self, bar_index: int) -> bool:
        rc = self.replay_cfg
        if (
            bar_index <= self.cooldown_until
            or bar_index <= self.streak_cooldown_until
            or bar_index <= self.tick_stop_cooldown_until
        ):
            return True
        if self.day_halted:
            return True
        if rc.max_trades_per_day is not None and self.trades_today >= rc.max_trades_per_day:
            return True
        if self.pending_entry_bar is not None:
            return True
        return False

    def _choose_entry(
        self,
        session_bar: Optional[int],
        quotes: SessionQuotes,
        signal: SessionSignal,
    ) -> Optional[EntryDecision]:
        put_sp = quotes.put_spread_pct
        if put_sp is None and quotes.has_put():
            put_sp = quotes.spread_pct("PUT")
        straddle_sp = max(quotes.call_spread_pct, put_sp or 0.0) if quotes.has_put() else None
        dyn_th = self._quantile_threshold(self._edge_buf)
        put_dyn_th = self._quantile_threshold(self._put_edge_buf)
        if not bool(getattr(self.replay_cfg, "apply_put_entry_quantile", True)):
            put_dyn_th = None

        # 早盘 open30 结构失败且 PUT edge 过静态阈 → 延长否决窗
        self._maybe_trip_put_structure_veto(session_bar, signal)

        mult = getattr(self.replay_cfg, "loss_reentry_edge_mult", None)
        c_mult = 1.0
        p_mult = 1.0
        if mult is not None and float(mult) > 1.0:
            if "CALL" in self.loss_legs_today:
                c_mult = float(mult)
            if "PUT" in self.loss_legs_today:
                p_mult = float(mult)

        return choose_entry(
            self.replay_cfg,
            session_bar=session_bar,
            edge=signal.edge,
            call_edge=signal.call_edge,
            put_edge=signal.put_edge,
            straddle_edge=signal.straddle_edge,
            edge_q10=signal.edge_q10,
            spread_pct=quotes.call_spread_pct,
            put_spread_pct=put_sp,
            straddle_spread_pct=straddle_sp,
            dual_mode=self.dual_mode,
            has_put=quotes.has_put(),
            straddle_enabled=self.dual_mode and not self.replay_cfg.long_only,
            straddles_today=self.straddles_today,
            default_leg=self.default_leg,
            dynamic_threshold=dyn_th,
            put_dynamic_threshold=put_dyn_th,
            put_gate=signal.put_gate,
            open30_max_ret=signal.open30_max_ret,
            open30_peak_dd=signal.open30_peak_dd,
            spot_ret_5bar=signal.spot_ret_5bar,
            spot_ret_15bar=signal.spot_ret_15bar,
            vix_ret_15bar=signal.vix_ret_15bar,
            trend_ret_30m=signal.trend_ret_30m,
            trend_r2_30m=signal.trend_r2_30m,
            vix_reversal_count_30m=signal.vix_reversal_count_30m,
            spot_day_ret=signal.spot_day_ret,
            spot_range_30m=signal.spot_range_30m,
            day_range_pos=signal.day_range_pos,
            bb_width=signal.bb_width,
            best_side_put_prob=signal.best_side_put_prob,
            best_side_none_prob=signal.best_side_none_prob,
            best_side_call_prob=signal.best_side_call_prob,
            spot_down_prob=signal.spot_down_prob,
            spot_flat_prob=signal.spot_flat_prob,
            spot_up_prob=signal.spot_up_prob,
            call_threshold_mult=c_mult,
            put_threshold_mult=p_mult,
            put_structure_veto_until_bar=self.put_structure_veto_until,
            vixy_open_shock_regime_active=self.vixy_open_shock_regime_active,
        )

    def _maybe_trip_vixy_open_shock_regime(
        self, session_bar: Optional[int], signal: SessionSignal
    ) -> None:
        """用已完成分钟识别开盘急跌后的低趋势震荡态，并保持到日切。"""
        rc = self.replay_cfg
        if (
            self.vixy_open_shock_regime_active
            or not bool(getattr(rc, "vixy_open_shock_regime_enabled", False))
            or session_bar is None
        ):
            return
        if not (
            int(rc.vixy_open_shock_detect_start_bar)
            <= int(session_bar)
            <= int(rc.vixy_open_shock_detect_end_bar)
        ):
            return
        vals = (signal.open30_ret, signal.open30_peak_dd, signal.trend_r2_30m)
        if any(v is None or not np.isfinite(v) for v in vals):
            return
        self.vixy_open_shock_regime_active = bool(
            float(signal.open30_ret) < float(rc.vixy_open_shock_open30_ret_max)
            and float(signal.open30_peak_dd)
            <= float(rc.vixy_open_shock_peak_dd_max)
            and float(signal.trend_r2_30m)
            < float(rc.vixy_open_shock_detect_r2_max)
        )

    def _maybe_trip_put_structure_veto(
        self, session_bar: Optional[int], signal: SessionSignal
    ) -> None:
        """早盘 PUT 因 open30 未翻红被挡时,把否决延长到 put_structure_veto_end_bar。"""
        rc = self.replay_cfg
        end = getattr(rc, "put_structure_veto_end_bar", None)
        if end is None or session_bar is None:
            return
        early_bar = rc.put_early_session_bar
        open30_min = rc.put_early_open30_max_min
        if early_bar is None or open30_min is None:
            return
        if session_bar >= int(early_bar):
            return
        th = rc.threshold_at(session_bar)
        pe = signal.put_edge
        if pe is None or not np.isfinite(pe) or float(pe) < float(th):
            return
        omax = signal.open30_max_ret
        if omax is not None and np.isfinite(omax) and float(omax) > float(open30_min):
            return
        # 结构失败(缺失或 <= min)
        cur = self.put_structure_veto_until
        self.put_structure_veto_until = (
            int(end) if cur is None else max(int(cur), int(end))
        )

    def _quantile_threshold(self, buf: Optional[deque]) -> Optional[float]:
        if buf is None or len(buf) < int(self.replay_cfg.entry_quantile_min_obs):
            return None
        return float(np.quantile(np.asarray(buf), float(self.replay_cfg.entry_quantile)))

    def _try_minute_exit(
        self,
        bar_index: int,
        ts: Any,
        session_bar: Optional[int],
        quotes: SessionQuotes,
    ) -> Optional[ReplayEvent]:
        assert self.position is not None
        rails = self.position.rails or self.rails_cfg
        held = bar_index - self.position.entry_bar
        if self.position.entry_spot is not None:
            thesis = check_spot_thesis_invalidate(
                rails,
                leg=self.position.leg,
                spot_closes=self._spot_closes,
                entry_spot=float(self.position.entry_spot),
                held=held,
            )
            if thesis is not None:
                return self._close_position(bar_index, ts, quotes, thesis, disaster=False)
        mtm = quotes.mid(self.position.leg)
        if not (np.isfinite(mtm) and mtm > 0):
            reason = check_forced_time_exit(
                rails,
                entry_bar=self.position.entry_bar,
                current_bar=bar_index,
                session_bar_index=session_bar,
            )
            if reason is None:
                return None
            return self._close_position(bar_index, ts, quotes, reason, disaster=False)
        self.position.last_valid_mtm = float(mtm)
        reason = check_exit(
            rails,
            self.position.state,
            float(mtm),
            bar_index,
            session_bar_index=session_bar,
        )
        if reason is None:
            return None
        return self._close_position(bar_index, ts, quotes, reason, disaster=False)

    def _try_entry(self, bar_index: int, ts: Any, quotes: SessionQuotes) -> Optional[ReplayEvent]:
        leg = self.pending_leg
        fill_px = quotes.entry_fill(leg, self.fill_model)
        sp = quotes.spread_pct(leg)
        gate_ok = np.isfinite(sp) and sp <= self.replay_cfg.max_spread_pct
        self.pending_entry_bar = None
        if not (np.isfinite(fill_px) and fill_px > 0 and gate_ok):
            return None
        comm_mult = 2.0 if leg == "STRADDLE" else 1.0
        rails, vol_scale = self._entry_rails(leg=leg)
        entry_spot = self._spot_closes[-1] if self._spot_closes else None
        bounce_on = (
            rails.spot_thesis_against_entry is not None
            and self.rails_cfg.spot_thesis_against_entry is None
            and bool(getattr(self.rails_cfg, "bounce_cut_enabled", False))
            and str(leg).upper() == "PUT"
        )
        position_frac = float(
            getattr(self.replay_cfg, "position_frac", 1.0) or 1.0
        )
        if self.cross_day_all_leg_defense:
            defense_frac = getattr(
                self.replay_cfg, "next_day_all_leg_defense_position_frac", None
            )
            if defense_frac is not None:
                position_frac = float(defense_frac)
        self.position = OpenPosition(
            leg=leg,
            entry_price=float(fill_px),
            entry_bar=bar_index,
            entry_ts=ts,
            signal_edge=self.pending_edge,
            state=PositionState(entry_price=float(fill_px), entry_bar=bar_index),
            commission_mult=comm_mult,
            rails=rails,
            vol_scale=vol_scale,
            last_valid_mtm=float(fill_px),
            entry_spot=float(entry_spot) if entry_spot is not None else None,
            position_frac=position_frac,
        )
        ev = ReplayEvent(
            kind="ENTER",
            ts=ts,
            bar_index=bar_index,
            leg=leg,
            price=float(fill_px),
            edge=self.pending_edge,
            extra={
                "vol_scale": vol_scale,
                "bounce_cut": bounce_on,
                "position_frac": position_frac,
                "all_leg_defense": self.cross_day_all_leg_defense,
            },
        )
        self.events.append(ev)
        return ev

    def _close_position(
        self,
        bar_index: int,
        ts: Any,
        quotes: SessionQuotes,
        reason: str,
        *,
        disaster: bool,
    ) -> ReplayEvent:
        assert self.position is not None
        pos = self.position
        try:
            exit_px = quotes.exit_fill(pos.leg, self.fill_model)
        except (TypeError, ValueError):
            exit_px = float("nan")
        mtm = quotes.mid(pos.leg)
        if not (np.isfinite(exit_px) and exit_px > 0):
            if np.isfinite(mtm) and mtm > 0:
                exit_px = mtm
            elif np.isfinite(pos.last_valid_mtm) and pos.last_valid_mtm > 0:
                exit_px = pos.last_valid_mtm
            else:
                exit_px = pos.entry_price
            reason = f"{reason}|NO_QUOTE"

        net_ret = float(exit_px) / pos.entry_price - 1.0
        if self.is_option:
            commission = self.fill_model.commission_return_drag(pos.entry_price)
            net_ret -= float(commission) * pos.commission_mult

        self.result.trades.append(
            Trade(
                entry_ts=pos.entry_ts,
                exit_ts=ts,
                entry_price=pos.entry_price,
                exit_price=float(exit_px),
                net_return=net_ret,
                exit_reason=reason,
                bars_held=bar_index - pos.entry_bar,
                signal_edge=pos.signal_edge,
                leg=pos.leg,
                position_frac=pos.position_frac,
            )
        )
        # 账户权益按 position_frac 下注;Trade.net_return 仍记权利金 ROI
        f = float(pos.position_frac)
        self.equity *= 1.0 + f * net_ret
        self.result.equity_curve.append(self.equity)

        closed_leg = pos.leg
        self.position = None
        self.cooldown_until = bar_index + self.replay_cfg.cooldown_bars
        if reason.startswith("TICK_FAST_HARD"):
            tick_bars = self.replay_cfg.tick_stop_cooldown_bars
            if tick_bars is not None:
                self.tick_stop_cooldown_until = bar_index + int(tick_bars)
                self.cooldown_until = max(
                    self.cooldown_until, self.tick_stop_cooldown_until
                )
            if self.replay_cfg.tick_stop_lock_leg_for_day:
                self.tick_stopped_legs.add(closed_leg)

        # bounce-cut / SPOT_THESIS:短期禁同腿再开(不锁全日,保留午后机会)
        reason_u = str(reason).upper()
        if reason_u.startswith("SPOT_THESIS"):
            lock_bars = getattr(self.replay_cfg, "thesis_lock_leg_bars", None)
            if lock_bars is not None and int(lock_bars) > 0:
                until = bar_index + int(lock_bars)
                leg_u = str(closed_leg).upper()
                prev = self.leg_lock_until.get(leg_u)
                self.leg_lock_until[leg_u] = until if prev is None else max(int(prev), until)

        self.trades_today += 1
        if closed_leg == "STRADDLE":
            self.straddles_today += 1
        self.day_pnl += net_ret
        self.day_account_equity *= 1.0 + f * net_ret
        if closed_leg in self.day_leg_equity:
            self.day_leg_equity[closed_leg] *= 1.0 + f * net_ret
        if net_ret < 0:
            self.loss_streak += 1
            self.loss_legs_today.add(closed_leg)
            if bool(getattr(self.replay_cfg, "loss_lock_leg_for_day", False)):
                min_loss = getattr(self.replay_cfg, "loss_lock_leg_min_loss", None)
                if min_loss is None or float(net_ret) <= float(min_loss):
                    self.tick_stopped_legs.add(closed_leg)
            rc = self.replay_cfg
            if rc.loss_streak_n is not None and self.loss_streak >= rc.loss_streak_n:
                self.streak_cooldown_until = bar_index + rc.loss_streak_cooldown_bars
                self.loss_streak = 0
        else:
            self.loss_streak = 0
        if self.replay_cfg.daily_loss_stop is not None and self.day_pnl <= self.replay_cfg.daily_loss_stop:
            self.day_halted = True

        kind = "DISASTER_EXIT" if disaster else "EXIT"
        ev = ReplayEvent(
            kind=kind,
            ts=ts,
            bar_index=bar_index,
            leg=closed_leg,
            price=float(exit_px),
            reason=reason,
            net_return=net_ret,
        )
        self.events.append(ev)
        return ev

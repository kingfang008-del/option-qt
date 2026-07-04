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
    check_tick_stops,
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
    ):
        self.replay_cfg = replay_cfg
        self.rails_cfg = rails_cfg
        self.fill_model = fill_model
        self.dual_mode = dual_mode
        self.default_leg = default_leg
        self.is_option = is_option

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

        self.cur_day = None
        self.trades_today = 0
        self.straddles_today = 0
        self.day_pnl = 0.0
        self.day_halted = False
        self.loss_streak = 0
        self.events: List[ReplayEvent] = []
        # 当日交易腿分钟 mid 序列(波动自适应护栏用,分钟收盘更新)
        self._day_mids: List[float] = []
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
            self._reset_day(day_key)

        if phase == BarPhase.CLOSE:
            mid = quotes.mid(self.default_leg)
            if np.isfinite(mid) and mid > 0:
                self._day_mids.append(float(mid))
            if self._edge_buf is not None and self.replay_cfg.session_allows_entry(session_bar):
                # dual 模式 CALL 腿比较的是 call_edge,分位缓冲须跟踪同一分数
                main_edge = (
                    signal.call_edge
                    if self.dual_mode and signal.call_edge is not None
                    else signal.edge
                )
                if main_edge is not None and np.isfinite(main_edge):
                    self._edge_buf.append(float(main_edge))
                if (
                    self._put_edge_buf is not None
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
        decision = self._choose_entry(session_bar, quotes, signal)
        if decision is None:
            return []
        sp = quotes.spread_pct(decision.leg)
        if not (np.isfinite(sp) and sp <= self.replay_cfg.max_spread_pct):
            return []
        self.pending_entry_bar = bar_index + self.replay_cfg.entry_delay_bars
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

    def _reset_day(self, day_key: Any) -> None:
        self.cur_day = day_key
        self.trades_today = 0
        self.straddles_today = 0
        self.day_pnl = 0.0
        self.day_halted = False
        self.pending_entry_bar = None
        self.loss_streak = 0
        self._day_mids = []

    def _entry_rails(self) -> tuple:
        """入场时刻的 (缩放护栏, scale)。未启用波动自适应时原样返回。"""
        if self.rails_cfg.vol_scale_ref is None:
            return self.rails_cfg, 1.0
        mids = np.asarray(self._day_mids, dtype=float)
        rets = (mids[1:] / mids[:-1] - 1.0).tolist() if mids.size >= 2 else []
        scale = vol_scale_from_returns(self.rails_cfg, rets)
        return scale_rails(self.rails_cfg, scale), scale

    def _blocked_for_entry(self, bar_index: int) -> bool:
        rc = self.replay_cfg
        if bar_index <= self.cooldown_until or bar_index <= self.streak_cooldown_until:
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
        mtm = quotes.mid(self.position.leg)
        if not (np.isfinite(mtm) and mtm > 0):
            return None
        reason = check_exit(
            self.position.rails or self.rails_cfg,
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
        rails, vol_scale = self._entry_rails()
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
        )
        ev = ReplayEvent(
            kind="ENTER",
            ts=ts,
            bar_index=bar_index,
            leg=leg,
            price=float(fill_px),
            edge=self.pending_edge,
            extra={"vol_scale": vol_scale},
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
        exit_px = quotes.exit_fill(pos.leg, self.fill_model)
        mtm = quotes.mid(pos.leg)
        if not (np.isfinite(exit_px) and exit_px > 0):
            exit_px = mtm if np.isfinite(mtm) and mtm > 0 else pos.entry_price
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
            )
        )
        # 账户权益按 position_frac 下注;Trade.net_return 仍记权利金 ROI
        f = float(getattr(self.replay_cfg, "position_frac", 1.0) or 1.0)
        self.equity *= 1.0 + f * net_ret
        self.result.equity_curve.append(self.equity)

        closed_leg = pos.leg
        self.position = None
        self.cooldown_until = bar_index + self.replay_cfg.cooldown_bars

        self.trades_today += 1
        if closed_leg == "STRADDLE":
            self.straddles_today += 1
        self.day_pnl += net_ret
        if net_ret < 0:
            self.loss_streak += 1
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

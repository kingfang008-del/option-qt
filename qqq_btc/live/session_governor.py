#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Live 频率治理 + entry_quantile —— 与 replay_session 日内状态机对齐。

OMS 进程内单例;按 symbol 维护 trades_today / day_pnl / loss_streak / 分位缓冲。
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np
from pytz import timezone

from qqq_btc.common.entry_quantile import maybe_append_edge_buffers, quantile_threshold
from qqq_btc.common.exit_rails import ExitRailsConfig, scale_rails, vol_scale_from_returns
from qqq_btc.common.replay_types import ReplayConfig
from qqq_btc.qqq import config as qcfg

_NY = timezone("America/New_York")
_GOVERNORS: Dict[str, "LiveSessionGovernor"] = {}


@dataclass
class _SymbolDayState:
    cur_day: Optional[str] = None
    trades_today: int = 0
    straddles_today: int = 0
    day_pnl: float = 0.0
    day_halted: bool = False
    loss_streak: int = 0
    streak_cooldown_until_ts: float = 0.0
    edge_buf: Optional[Deque[float]] = field(default=None)
    put_edge_buf: Optional[Deque[float]] = field(default=None)
    day_mids: List[float] = field(default_factory=list)


def _day_key_from_ts(ts: float) -> str:
    dt = datetime.fromtimestamp(float(ts), tz=_NY)
    return dt.strftime("%Y-%m-%d")


class LiveSessionGovernor:
    def __init__(self, replay_cfg: ReplayConfig) -> None:
        self.replay_cfg = replay_cfg
        self._sym: Dict[str, _SymbolDayState] = {}

    def _state(self, symbol: str) -> _SymbolDayState:
        sym = symbol or "QQQ"
        st = self._sym.get(sym)
        if st is None:
            q_on = getattr(self.replay_cfg, "entry_quantile", None) is not None
            win = int(self.replay_cfg.entry_quantile_window)
            st = _SymbolDayState(
                edge_buf=deque(maxlen=win) if q_on else None,
                put_edge_buf=deque(maxlen=win) if q_on else None,
            )
            self._sym[sym] = st
        return st

    def maybe_reset_day(self, symbol: str, ts: float) -> None:
        st = self._state(symbol)
        day = _day_key_from_ts(ts)
        if st.cur_day != day:
            st.cur_day = day
            st.trades_today = 0
            st.straddles_today = 0
            st.day_pnl = 0.0
            st.day_halted = False
            st.loss_streak = 0
            st.streak_cooldown_until_ts = 0.0
            st.day_mids = []

    def record_minute_mid(self, symbol: str, mid: float, ts: float) -> None:
        """记录分钟期权 mid(波动自适应 exit_rails 用,与 replay _day_mids 一致)。"""
        if mid <= 0 or not np.isfinite(mid):
            return
        self.maybe_reset_day(symbol, ts)
        self._state(symbol).day_mids.append(float(mid))

    def scaled_exit_rails(
        self,
        symbol: str,
        base_rails: Optional[ExitRailsConfig] = None,
    ) -> Tuple[ExitRailsConfig, float]:
        """入场时按当日已实现 minute return std 缩放护栏(replay _entry_rails 口径)。"""
        base = base_rails or qcfg.EXIT_RAILS
        if base.vol_scale_ref is None:
            return base, 1.0
        st = self._state(symbol)
        mids = np.asarray(st.day_mids, dtype=float)
        rets = (mids[1:] / mids[:-1] - 1.0).tolist() if mids.size >= 2 else []
        scale = vol_scale_from_returns(base, rets)
        return scale_rails(base, scale), scale

    def record_edges(
        self,
        symbol: str,
        *,
        session_bar: Optional[int],
        call_edge: Optional[float],
        put_edge: Optional[float],
        dual_mode: bool,
        trend_r2_30m: Optional[float] = None,
        spot_day_ret: Optional[float] = None,
        vix_reversal_count_30m: Optional[float] = None,
        spot_range_30m: Optional[float] = None,
        trend_ret_30m: Optional[float] = None,
        day_range_pos: Optional[float] = None,
        bb_width: Optional[float] = None,
    ) -> None:
        st = self._state(symbol)
        maybe_append_edge_buffers(
            self.replay_cfg,
            session_bar=session_bar,
            call_edge=call_edge,
            put_edge=put_edge,
            dual_mode=dual_mode,
            edge_buf=st.edge_buf,
            put_edge_buf=st.put_edge_buf,
            trend_r2_30m=trend_r2_30m,
            spot_day_ret=spot_day_ret,
            vix_reversal_count_30m=vix_reversal_count_30m,
            spot_range_30m=spot_range_30m,
            trend_ret_30m=trend_ret_30m,
            day_range_pos=day_range_pos,
            bb_width=bb_width,
        )

    def dynamic_thresholds(self, symbol: str) -> Tuple[Optional[float], Optional[float]]:
        st = self._state(symbol)
        return (
            quantile_threshold(st.edge_buf, self.replay_cfg),
            quantile_threshold(st.put_edge_buf, self.replay_cfg),
        )

    def blocked_for_entry(
        self,
        symbol: str,
        *,
        curr_ts: float,
        cooldown_until: float = 0.0,
    ) -> Tuple[bool, str]:
        self.maybe_reset_day(symbol, curr_ts)
        st = self._state(symbol)
        rc = self.replay_cfg
        if float(curr_ts) < float(cooldown_until or 0.0):
            return True, "cooldown"
        if float(curr_ts) < float(st.streak_cooldown_until_ts or 0.0):
            return True, "loss_streak_cooldown"
        if st.day_halted:
            return True, "daily_loss_stop"
        if rc.max_trades_per_day is not None and st.trades_today >= int(rc.max_trades_per_day):
            return True, "max_trades_per_day"
        return False, ""

    def record_trade_close(
        self,
        symbol: str,
        *,
        net_ret: float,
        curr_ts: float,
        leg: str = "CALL",
    ) -> float:
        """平仓后更新日内统计;返回冷却截止时间(含 cooldown_bars + 连亏冷却)。"""
        self.maybe_reset_day(symbol, curr_ts)
        st = self._state(symbol)
        rc = self.replay_cfg
        nr = float(net_ret)
        st.trades_today += 1
        if leg == "STRADDLE":
            st.straddles_today += 1
        st.day_pnl += nr
        cool_until = 0.0
        # 与 ReplaySession 一致:每笔平仓后冷却 cooldown_bars 分钟
        bars = int(getattr(rc, "cooldown_bars", 0) or 0)
        if bars > 0 and curr_ts > 0:
            cool_until = float(curr_ts) + bars * 60.0
        if nr < 0:
            st.loss_streak += 1
            if rc.loss_streak_n is not None and st.loss_streak >= int(rc.loss_streak_n):
                streak_bars = int(rc.loss_streak_cooldown_bars)
                streak_until = float(curr_ts) + streak_bars * 60.0
                st.streak_cooldown_until_ts = streak_until
                cool_until = max(cool_until, streak_until)
                st.loss_streak = 0
        else:
            st.loss_streak = 0
        if rc.daily_loss_stop is not None and st.day_pnl <= float(rc.daily_loss_stop):
            st.day_halted = True
        return cool_until

    def straddles_today_for(self, symbol: str) -> int:
        return self._state(symbol).straddles_today

    @property
    def straddles_today(self) -> Dict[str, int]:
        return {sym: st.straddles_today for sym, st in self._sym.items()}


def get_session_governor(replay_cfg: Optional[ReplayConfig] = None) -> LiveSessionGovernor:
    cfg = replay_cfg or qcfg.REPLAY
    key = id(cfg)
    gov = _GOVERNORS.get(str(key))
    if gov is None:
        gov = LiveSessionGovernor(cfg)
        _GOVERNORS[str(key)] = gov
    return gov


def net_return_from_prices(entry_px: float, exit_px: float, *, commission_drag: float = 0.0) -> float:
    if entry_px <= 0 or exit_px <= 0:
        return 0.0
    return float(exit_px / entry_px - 1.0 - commission_drag)

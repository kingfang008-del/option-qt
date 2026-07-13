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
_ATEXIT_REGISTERED = False


def _register_persist_atexit() -> None:
    global _ATEXIT_REGISTERED
    if _ATEXIT_REGISTERED:
        return
    import atexit

    atexit.register(persist_session_governor)
    _ATEXIT_REGISTERED = True


@dataclass
class _SymbolDayState:
    cur_day: Optional[str] = None
    trades_today: int = 0
    straddles_today: int = 0
    day_pnl: float = 0.0
    day_halted: bool = False
    loss_streak: int = 0
    streak_cooldown_until_ts: float = 0.0
    tick_stop_cooldown_until_ts: float = 0.0
    tick_stopped_legs: set[str] = field(default_factory=set)
    loss_legs_today: set[str] = field(default_factory=set)
    # SPOT_THESIS 短期同腿锁:leg -> unix ts 截止(含)
    leg_lock_until_ts: Dict[str, float] = field(default_factory=dict)
    put_structure_veto_until: Optional[int] = None
    edge_buf: Optional[Deque[float]] = field(default=None)
    put_edge_buf: Optional[Deque[float]] = field(default=None)
    last_edge_session_bar: Optional[int] = None
    day_mids: List[float] = field(default_factory=list)
    spot_closes: List[float] = field(default_factory=list)
    vwap_lrs: List[float] = field(default_factory=list)


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
            st.tick_stop_cooldown_until_ts = 0.0
            st.tick_stopped_legs.clear()
            st.loss_legs_today.clear()
            st.leg_lock_until_ts.clear()
            st.put_structure_veto_until = None
            st.last_edge_session_bar = None
            st.day_mids = []
            st.spot_closes = []
            st.vwap_lrs = []

    def record_minute_mid(self, symbol: str, mid: float, ts: float) -> None:
        """记录分钟期权 mid(波动自适应 exit_rails 用,与 replay _day_mids 一致)。"""
        if mid <= 0 or not np.isfinite(mid):
            return
        self.maybe_reset_day(symbol, ts)
        self._state(symbol).day_mids.append(float(mid))

    def record_bounce_inputs(
        self,
        symbol: str,
        *,
        spot_close: Optional[float] = None,
        vwap_log_return: Optional[float] = None,
        ts: float = 0.0,
    ) -> None:
        """记录现货/vwap(bounce-cut 与 replay _spot_closes/_vwap_lrs 对齐)。"""
        if ts > 0:
            self.maybe_reset_day(symbol, ts)
        st = self._state(symbol)
        if spot_close is not None and np.isfinite(spot_close) and float(spot_close) > 0:
            st.spot_closes.append(float(spot_close))
        if vwap_log_return is not None and np.isfinite(vwap_log_return):
            st.vwap_lrs.append(float(vwap_log_return))

    def scaled_exit_rails(
        self,
        symbol: str,
        base_rails: Optional[ExitRailsConfig] = None,
        *,
        leg: Optional[str] = None,
    ) -> Tuple[ExitRailsConfig, float]:
        """入场时按当日已实现 minute return std 缩放护栏(replay _entry_rails 口径)。

        传入 leg=PUT 时叠加 bounce-cut 仓位级证伪。
        """
        from qqq_btc.common.exit_rails import maybe_bounce_cut_rails

        base = base_rails or qcfg.EXIT_RAILS
        if base.vol_scale_ref is None:
            rails, scale = base, 1.0
        else:
            st = self._state(symbol)
            mids = np.asarray(st.day_mids, dtype=float)
            rets = (mids[1:] / mids[:-1] - 1.0).tolist() if mids.size >= 2 else []
            scale = vol_scale_from_returns(base, rets)
            rails = scale_rails(base, scale)
        if leg is not None:
            st = self._state(symbol)
            vwap_jump = None
            if len(st.vwap_lrs) >= 2:
                vwap_jump = float(st.vwap_lrs[-1]) - float(st.vwap_lrs[-2])
            rails = maybe_bounce_cut_rails(
                rails,
                leg=str(leg),
                vwap_jump=vwap_jump,
                spot_closes=st.spot_closes,
            )
        return rails, scale

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
        curr_ts: float = 0.0,
    ) -> None:
        st = self._state(symbol)
        if curr_ts > 0 and curr_ts < st.tick_stop_cooldown_until_ts:
            return
        # 同一分钟 OMS build_ctx + decide_entry 可能各走一次；仅 live(curr_ts>0) 去重。
        if (
            curr_ts > 0
            and session_bar is not None
            and st.last_edge_session_bar == int(session_bar)
        ):
            return
        before = len(st.edge_buf) if st.edge_buf is not None else 0
        before_put = len(st.put_edge_buf) if st.put_edge_buf is not None else 0
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
        after = len(st.edge_buf) if st.edge_buf is not None else 0
        after_put = len(st.put_edge_buf) if st.put_edge_buf is not None else 0
        if curr_ts > 0 and session_bar is not None and (
            after > before
            or after_put > before_put
            or self.replay_cfg.session_allows_entry(session_bar)
        ):
            st.last_edge_session_bar = int(session_bar)
        # 每次更新后落盘：单日有效 edge 可能少于 60，且日终进程可能被强制终止。
        # 文件仅包含有界 deque（默认最多 1500 项），分钟级写入开销可忽略。
        path = _governor_state_path()
        if path and st.edge_buf is not None and len(st.edge_buf) > 0:
            try:
                self.save_quantile_state(path)
            except Exception:
                pass

    def dynamic_thresholds(self, symbol: str) -> Tuple[Optional[float], Optional[float]]:
        st = self._state(symbol)
        put_th = quantile_threshold(st.put_edge_buf, self.replay_cfg)
        if not bool(getattr(self.replay_cfg, "apply_put_entry_quantile", True)):
            put_th = None
        return (
            quantile_threshold(st.edge_buf, self.replay_cfg),
            put_th,
        )

    def save_quantile_state(self, path: str) -> None:
        """持久化各标的 edge/put_edge 分位缓冲（跨日流式对齐 offline 连续周）。"""
        import pickle
        from pathlib import Path

        payload = {}
        for sym, st in self._sym.items():
            payload[sym] = {
                "edge_buf": list(st.edge_buf) if st.edge_buf is not None else None,
                "put_edge_buf": list(st.put_edge_buf) if st.put_edge_buf is not None else None,
            }
        p = Path(path).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(pickle.dumps(payload, protocol=4))

    def load_quantile_state(self, path: str) -> int:
        """恢复分位缓冲；返回主标的 edge_buf 长度。"""
        import pickle
        from pathlib import Path

        p = Path(path).expanduser()
        if not p.is_file():
            return 0
        try:
            payload = pickle.loads(p.read_bytes())
        except Exception:
            return 0
        if not isinstance(payload, dict):
            return 0
        n_main = 0
        win = int(self.replay_cfg.entry_quantile_window)
        for sym, blob in payload.items():
            if not isinstance(blob, dict):
                continue
            st = self._state(str(sym))
            eb = blob.get("edge_buf")
            pb = blob.get("put_edge_buf")
            if eb is not None and st.edge_buf is not None:
                st.edge_buf = deque((float(x) for x in eb if np.isfinite(float(x))), maxlen=win)
            if pb is not None and st.put_edge_buf is not None:
                st.put_edge_buf = deque(
                    (float(x) for x in pb if np.isfinite(float(x))), maxlen=win
                )
            if str(sym).upper() == "QQQ" and st.edge_buf is not None:
                n_main = len(st.edge_buf)
        return n_main

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
        # ReplaySession 用 bar_index <= cooldown_until；时间戳边界也必须封锁，
        # 否则恰好第 N 分钟会比离线早一根重新入场。
        if float(curr_ts) <= float(cooldown_until or 0.0):
            return True, "cooldown"
        if float(curr_ts) <= float(st.streak_cooldown_until_ts or 0.0):
            return True, "loss_streak_cooldown"
        if float(curr_ts) <= float(st.tick_stop_cooldown_until_ts or 0.0):
            return True, "tick_stop_cooldown"
        if st.day_halted:
            return True, "daily_loss_stop"
        if rc.max_trades_per_day is not None and st.trades_today >= int(rc.max_trades_per_day):
            return True, "max_trades_per_day"
        return False, ""

    def leg_blocked_for_entry(
        self, symbol: str, *, leg: str, curr_ts: float
    ) -> bool:
        self.maybe_reset_day(symbol, curr_ts)
        st = self._state(symbol)
        leg_u = str(leg).upper()
        if leg_u in st.tick_stopped_legs:
            return True
        until = st.leg_lock_until_ts.get(leg_u)
        if until is not None and float(curr_ts) <= float(until):
            return True
        return False

    def record_trade_close(
        self,
        symbol: str,
        *,
        net_ret: float,
        curr_ts: float,
        leg: str = "CALL",
        reason: str = "",
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
        reason_u = str(reason or "").upper()
        if reason_u.startswith(("TICK_FAST_HARD", "QQQ_BTC_TICK_FAST_HARD")):
            tick_bars = getattr(rc, "tick_stop_cooldown_bars", None)
            if tick_bars is not None and curr_ts > 0:
                tick_until = float(curr_ts) + int(tick_bars) * 60.0
                st.tick_stop_cooldown_until_ts = tick_until
                cool_until = max(cool_until, tick_until)
            if bool(getattr(rc, "tick_stop_lock_leg_for_day", False)):
                st.tick_stopped_legs.add(str(leg).upper())
        if reason_u.startswith(("SPOT_THESIS", "QQQ_BTC_SPOT_THESIS")):
            lock_bars = getattr(rc, "thesis_lock_leg_bars", None)
            if lock_bars is not None and int(lock_bars) > 0 and curr_ts > 0:
                until = float(curr_ts) + int(lock_bars) * 60.0
                leg_u = str(leg).upper()
                prev = st.leg_lock_until_ts.get(leg_u)
                st.leg_lock_until_ts[leg_u] = until if prev is None else max(float(prev), until)
                cool_until = max(cool_until, until)
        if nr < 0:
            st.loss_streak += 1
            st.loss_legs_today.add(str(leg).upper())
            if bool(getattr(rc, "loss_lock_leg_for_day", False)):
                min_loss = getattr(rc, "loss_lock_leg_min_loss", None)
                if min_loss is None or float(nr) <= float(min_loss):
                    st.tick_stopped_legs.add(str(leg).upper())
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
    def note_put_structure_veto(
        self,
        symbol: str,
        *,
        session_bar: Optional[int],
        put_edge: Optional[float],
        open30_max_ret: Optional[float],
    ) -> None:
        """与 ReplaySession._maybe_trip_put_structure_veto 同口径。"""
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
        if put_edge is None or not np.isfinite(put_edge) or float(put_edge) < float(th):
            return
        if (
            open30_max_ret is not None
            and np.isfinite(open30_max_ret)
            and float(open30_max_ret) > float(open30_min)
        ):
            return
        st = self._state(symbol)
        cur = st.put_structure_veto_until
        st.put_structure_veto_until = int(end) if cur is None else max(int(cur), int(end))

    def entry_threshold_mults(self, symbol: str) -> tuple:
        mult = getattr(self.replay_cfg, "loss_reentry_edge_mult", None)
        c_mult = p_mult = 1.0
        if mult is not None and float(mult) > 1.0:
            st = self._state(symbol)
            if "CALL" in st.loss_legs_today:
                c_mult = float(mult)
            if "PUT" in st.loss_legs_today:
                p_mult = float(mult)
        return c_mult, p_mult

    def put_structure_veto_until_bar(self, symbol: str) -> Optional[int]:
        return self._state(symbol).put_structure_veto_until

    def straddles_today_for(self, symbol: str) -> int:
        return self._state(symbol).straddles_today

    @property
    def straddles_today(self) -> Dict[str, int]:
        return {sym: st.straddles_today for sym, st in self._sym.items()}


def resolve_replay_cfg(replay_cfg: Optional[ReplayConfig] = None) -> ReplayConfig:
    """与 strategy_entry_bridge 同一套 cfg：LIVE_REPLAY env + 阈值 override。

    必须跨 entry / fill 共用同一 governor；禁止用 id(cfg) 做 key
    （USE_LIVE_REPLAY=1 时 entry 拿 LIVE、fill 默认 REPLAY → 双实例，
    max_trades / daily_loss 形同虚设，本周曾出现单日 7 笔）。
    """
    import os
    from dataclasses import replace as _dc_replace

    if replay_cfg is not None:
        cfg = replay_cfg
    else:
        use_live = os.environ.get("QQQ_BTC_USE_LIVE_REPLAY", "").strip().lower() in (
            "1",
            "true",
            "yes",
        )
        cfg = qcfg.LIVE_REPLAY if use_live else qcfg.REPLAY
    _ov: dict = {}
    for env_k, attr, cast in (
        ("QQQ_BTC_PUT_GATE_MIN", "put_gate_min", float),
        ("QQQ_BTC_PUT_EARLY_VIX_MIN", "put_early_vix_min", float),
        ("QQQ_BTC_PUT_EARLY_SESSION_BAR", "put_early_session_bar", int),
        ("QQQ_BTC_PUT_EARLY_VIX_BAN_LO", "put_early_vix_ban_lo", float),
        ("QQQ_BTC_PUT_EARLY_VIX_BAN_HI", "put_early_vix_ban_hi", float),
        ("QQQ_BTC_PUT_EARLY_OPEN30_MAX_MIN", "put_early_open30_max_min", float),
        ("QQQ_BTC_EDGE_Q10_FLOOR", "edge_q10_floor", float),
        ("QQQ_BTC_ENTRY_QUANTILE", "entry_quantile", float),
        ("QQQ_BTC_ENTRY_QUANTILE_MIN_OBS", "entry_quantile_min_obs", int),
        ("QQQ_BTC_ENTRY_QUANTILE_WINDOW", "entry_quantile_window", int),
        ("QQQ_BTC_MORNING_FADE_MIN_RET", "morning_fade_min_ret", float),
        ("QQQ_BTC_MORNING_FADE_MAX_PEAK_DD", "morning_fade_max_peak_dd", float),
        ("QQQ_BTC_MORNING_FADE_SESSION_END_BAR", "morning_fade_session_end_bar", int),
    ):
        raw = os.environ.get(env_k, "").strip()
        if not raw:
            continue
        if raw.lower() in ("none", "null", "off"):
            _ov[attr] = None
        else:
            try:
                _ov[attr] = cast(raw)
            except ValueError:
                pass
    # CALL-only 分位:QQQ_BTC_APPLY_PUT_ENTRY_QUANTILE=0/false/off
    raw_put_q = os.environ.get("QQQ_BTC_APPLY_PUT_ENTRY_QUANTILE", "").strip().lower()
    if raw_put_q in ("0", "false", "no", "off"):
        _ov["apply_put_entry_quantile"] = False
    elif raw_put_q in ("1", "true", "yes", "on"):
        _ov["apply_put_entry_quantile"] = True
    if _ov:
        cfg = _dc_replace(cfg, **_ov)
    return cfg


def _governor_state_path() -> Optional[str]:
    import os

    p = os.environ.get("QQQ_BTC_GOVERNOR_STATE", "").strip()
    return p or None


def get_session_governor(replay_cfg: Optional[ReplayConfig] = None) -> LiveSessionGovernor:
    """进程内唯一 governor；cfg 可热更新，状态不因 cfg 对象身份而分裂。

    若设 QQQ_BTC_GOVERNOR_STATE=path.pkl，则跨日复用 edge 分位缓冲
   （单日流式重启时否则永远凑不满 min_obs=300，dyn_threshold 恒为 None）。
    """
    cfg = resolve_replay_cfg(replay_cfg)
    key = "qqq_btc_default"
    gov = _GOVERNORS.get(key)
    if gov is None:
        gov = LiveSessionGovernor(cfg)
        state_path = _governor_state_path()
        if state_path:
            n = gov.load_quantile_state(state_path)
            if n:
                import logging

                logging.getLogger("qqq_btc.live.session_governor").info(
                    "governor quantile state loaded from %s (edge_buf=%d)",
                    state_path,
                    n,
                )
            _register_persist_atexit()
        _GOVERNORS[key] = gov
    else:
        gov.replay_cfg = cfg
    return gov


def persist_session_governor() -> None:
    """日终把分位缓冲落盘（诚实多日脚本在每天进程退出前调用）。"""
    from pathlib import Path

    gov = _GOVERNORS.get("qqq_btc_default")
    path = _governor_state_path()
    if gov is None or not path:
        return
    # 避免 OMS 空缓冲 atexit 覆盖 Signal 已写入的状态
    n = 0
    for st in gov._sym.values():
        if st.edge_buf is not None:
            n = max(n, len(st.edge_buf))
    if n <= 0:
        return
    try:
        prev = Path(path).expanduser()
        if prev.is_file():
            import pickle

            old = pickle.loads(prev.read_bytes())
            old_n = 0
            if isinstance(old, dict):
                for blob in old.values():
                    if isinstance(blob, dict) and blob.get("edge_buf") is not None:
                        old_n = max(old_n, len(blob["edge_buf"]))
            if n < old_n:
                return
    except Exception:
        pass
    gov.save_quantile_state(path)


def net_return_from_prices(entry_px: float, exit_px: float, *, commission_drag: float = 0.0) -> float:
    if entry_px <= 0 or exit_px <= 0:
        return 0.0
    return float(exit_px / entry_px - 1.0 - commission_drag)

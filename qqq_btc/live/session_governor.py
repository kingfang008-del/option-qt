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
from typing import Any, Deque, Dict, List, Optional, Tuple

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
    vixy_open_shock_regime_active: bool = False
    edge_buf: Optional[Deque[float]] = field(default=None)
    put_edge_buf: Optional[Deque[float]] = field(default=None)
    last_edge_session_bar: Optional[int] = None
    day_mids: List[float] = field(default_factory=list)
    spot_closes: List[float] = field(default_factory=list)
    vwap_lrs: List[float] = field(default_factory=list)
    last_bounce_spot_minute_key: Optional[int] = None
    last_bounce_vwap_minute_key: Optional[int] = None
    # (NY session minute key, QQQ close, VIXY proxy close)，用于因果 15m 确认。
    cross_asset_samples: List[Tuple[int, float, float]] = field(default_factory=list)
    # 跨日账户/腿级状态（与 ReplaySession 对齐）
    day_account_equity: float = 1.0
    day_leg_equity: Dict[str, float] = field(
        default_factory=lambda: {"CALL": 1.0, "PUT": 1.0}
    )
    cross_day_all_leg_defense: bool = False
    cross_day_quarantined_legs: set[str] = field(default_factory=set)
    vx_curve_slope: Optional[float] = None
    # 日频 rule profile（OPEN_DEFENSE / CHOP_NO_TRADE / TREND_PUT_OK）
    active_profile: str = "TREND_PUT_OK"
    profile_meta: Dict[str, Any] = field(default_factory=dict)


def _day_key_from_ts(ts: float) -> str:
    dt = datetime.fromtimestamp(float(ts), tz=_NY)
    return dt.strftime("%Y-%m-%d")


class LiveSessionGovernor:
    def __init__(self, replay_cfg: ReplayConfig) -> None:
        # base=未套 profile 的生产配置；replay_cfg=当日 profile 覆盖后的生效配置。
        self._base_replay_cfg = replay_cfg
        self.replay_cfg = replay_cfg
        self._sym: Dict[str, _SymbolDayState] = {}

    def set_base_cfg(self, replay_cfg: ReplayConfig) -> None:
        """热更新基础配置后，按当前交易日重新套用 profile。"""
        self._base_replay_cfg = replay_cfg
        # 找任一已有当日状态并刷新；没有则直接用 base。
        for st in self._sym.values():
            if st.cur_day:
                from datetime import date as _date

                self._apply_day_profile(_date.fromisoformat(st.cur_day), st.vx_curve_slope)
                return
        self.replay_cfg = replay_cfg

    def _apply_day_profile(
        self, trading_day, vx_slope: Optional[float]
    ) -> Dict[str, Any]:
        from qqq_btc.live.rule_profile_live import apply_live_rule_profile

        cfg, meta = apply_live_rule_profile(
            self._base_replay_cfg,
            trading_day,
            vx_curve_slope=vx_slope,
        )
        self.replay_cfg = cfg
        return meta

    def _state(self, symbol: str) -> _SymbolDayState:
        sym = symbol or "QQQ"
        st = self._sym.get(sym)
        if st is None:
            q_on = getattr(self._base_replay_cfg, "entry_quantile", None) is not None
            win = int(self._base_replay_cfg.entry_quantile_window)
            st = _SymbolDayState(
                edge_buf=deque(maxlen=win) if q_on else None,
                put_edge_buf=deque(maxlen=win) if q_on else None,
            )
            self._sym[sym] = st
        return st

    def maybe_reset_day(self, symbol: str, ts: float) -> None:
        st = self._state(symbol)
        day = _day_key_from_ts(ts)
        if st.cur_day == day:
            return

        from datetime import date as _date

        from qqq_btc.live.vx_term_live import prior_vx_curve_slope

        trading_day = _date.fromisoformat(day)
        vx_slope = prior_vx_curve_slope(trading_day)
        st.vx_curve_slope = vx_slope

        prev_put_contribution = float(st.day_leg_equity.get("PUT", 1.0) - 1.0)
        prev_account_contribution = float(st.day_account_equity - 1.0)
        # 跨日防御阈值读 base，避免被 OPEN_DEFENSE/CHOP 覆盖字段干扰。
        rc = self._base_replay_cfg

        put_quarantine_loss = getattr(rc, "next_day_put_quarantine_loss", None)
        put_quarantine_vx_min = getattr(
            rc, "next_day_put_quarantine_vx_slope_min", None
        )
        put_vx_ok = put_quarantine_vx_min is None or (
            vx_slope is not None
            and np.isfinite(vx_slope)
            and float(vx_slope) >= float(put_quarantine_vx_min)
        )
        if (
            st.cur_day is not None
            and put_quarantine_loss is not None
            and np.isfinite(prev_put_contribution)
            and prev_put_contribution <= float(put_quarantine_loss)
            and put_vx_ok
        ):
            st.cross_day_quarantined_legs = {"PUT"}
        else:
            st.cross_day_quarantined_legs.clear()

        all_leg_loss = getattr(rc, "next_day_all_leg_defense_loss", None)
        all_leg_vx_min = getattr(rc, "next_day_all_leg_defense_vx_slope_min", None)
        all_vx_ok = all_leg_vx_min is None or (
            vx_slope is not None
            and np.isfinite(vx_slope)
            and float(vx_slope) >= float(all_leg_vx_min)
        )
        st.cross_day_all_leg_defense = bool(
            st.cur_day is not None
            and all_leg_loss is not None
            and np.isfinite(prev_account_contribution)
            and prev_account_contribution <= float(all_leg_loss)
            and all_vx_ok
        )

        profile_meta = self._apply_day_profile(trading_day, vx_slope)
        st.active_profile = str(profile_meta.get("profile") or "TREND_PUT_OK")
        st.profile_meta = dict(profile_meta)

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
        st.vixy_open_shock_regime_active = False
        st.last_edge_session_bar = None
        st.day_mids = []
        st.spot_closes = []
        st.vwap_lrs = []
        st.last_bounce_spot_minute_key = None
        st.last_bounce_vwap_minute_key = None
        st.cross_asset_samples = []
        st.day_account_equity = 1.0
        st.day_leg_equity = {"CALL": 1.0, "PUT": 1.0}
        # 日切后立刻落盘，避免开盘后重启丢失半仓/PUT 隔离旗标
        state_path = _governor_state_path()
        if state_path:
            try:
                self.save_quantile_state(state_path)
            except Exception:
                pass

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
        """按分钟记录现货/vwap，避免同一 alpha frame 重复触发 bounce onset。"""
        if ts > 0:
            self.maybe_reset_day(symbol, ts)
        st = self._state(symbol)
        minute_key = int(float(ts) // 60) if ts > 0 else None
        if spot_close is not None and np.isfinite(spot_close) and float(spot_close) > 0:
            if minute_key is not None and st.last_bounce_spot_minute_key == minute_key and st.spot_closes:
                st.spot_closes[-1] = float(spot_close)
            else:
                st.spot_closes.append(float(spot_close))
            if minute_key is not None:
                st.last_bounce_spot_minute_key = minute_key
        if vwap_log_return is not None and np.isfinite(vwap_log_return):
            if minute_key is not None and st.last_bounce_vwap_minute_key == minute_key and st.vwap_lrs:
                st.vwap_lrs[-1] = float(vwap_log_return)
            else:
                st.vwap_lrs.append(float(vwap_log_return))
            if minute_key is not None:
                st.last_bounce_vwap_minute_key = minute_key

    def record_cross_asset_inputs(
        self,
        symbol: str,
        *,
        spot_close: Optional[float],
        vix_proxy_close: Optional[float],
        ts: float,
    ) -> None:
        """按分钟去重记录 QQQ/VIXY 收盘，供早盘 PUT 15m 反向确认。"""
        if ts <= 0:
            return
        self.maybe_reset_day(symbol, ts)
        if (
            spot_close is None
            or vix_proxy_close is None
            or not np.isfinite(spot_close)
            or not np.isfinite(vix_proxy_close)
            or float(spot_close) <= 0
            or float(vix_proxy_close) <= 0
        ):
            return
        st = self._state(symbol)
        minute_key = int(float(ts) // 60)
        sample = (minute_key, float(spot_close), float(vix_proxy_close))
        if st.cross_asset_samples and st.cross_asset_samples[-1][0] == minute_key:
            st.cross_asset_samples[-1] = sample
        else:
            st.cross_asset_samples.append(sample)
            # 只需日内短窗，限制内存与持久化体积。
            if len(st.cross_asset_samples) > 390:
                st.cross_asset_samples = st.cross_asset_samples[-390:]

    def cross_asset_returns(
        self, symbol: str, *, bars: int = 15
    ) -> Tuple[Optional[float], Optional[float]]:
        """返回当前相对 bars 分钟前的 (QQQ, VIXY) 收益；历史不足则 None。"""
        samples = self._state(symbol).cross_asset_samples
        if len(samples) < 2:
            return None, None
        now_key, spot_now, vix_now = samples[-1]
        target = now_key - int(bars)
        prev = next((x for x in reversed(samples[:-1]) if x[0] <= target), None)
        if prev is None:
            return None, None
        _, spot_prev, vix_prev = prev
        if spot_prev <= 0 or vix_prev <= 0:
            return None, None
        return spot_now / spot_prev - 1.0, vix_now / vix_prev - 1.0

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
        """持久化分位缓冲 + 跨日防御状态（夜间重启后仍能触发次日半仓/PUT隔离）。"""
        import pickle
        from pathlib import Path

        payload = {}
        for sym, st in self._sym.items():
            payload[sym] = {
                "edge_buf": list(st.edge_buf) if st.edge_buf is not None else None,
                "put_edge_buf": list(st.put_edge_buf) if st.put_edge_buf is not None else None,
                "cur_day": st.cur_day,
                "day_account_equity": float(st.day_account_equity),
                "day_leg_equity": {
                    "CALL": float(st.day_leg_equity.get("CALL", 1.0)),
                    "PUT": float(st.day_leg_equity.get("PUT", 1.0)),
                },
                "cross_day_all_leg_defense": bool(st.cross_day_all_leg_defense),
                "cross_day_quarantined_legs": sorted(st.cross_day_quarantined_legs),
                "vx_curve_slope": st.vx_curve_slope,
                "active_profile": st.active_profile,
                "profile_meta": dict(st.profile_meta),
                "trades_today": int(st.trades_today),
                "day_pnl": float(st.day_pnl),
                "cross_asset_samples": list(st.cross_asset_samples),
                "vixy_open_shock_regime_active": bool(
                    st.vixy_open_shock_regime_active
                ),
            }
        p = Path(path).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(pickle.dumps(payload, protocol=4))

    def load_quantile_state(self, path: str) -> int:
        """恢复分位缓冲与跨日防御状态；返回主标的 edge_buf 长度。"""
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
            if blob.get("cur_day") is not None:
                st.cur_day = str(blob["cur_day"])
            if blob.get("day_account_equity") is not None:
                st.day_account_equity = float(blob["day_account_equity"])
            leg_eq = blob.get("day_leg_equity")
            if isinstance(leg_eq, dict):
                st.day_leg_equity = {
                    "CALL": float(leg_eq.get("CALL", 1.0)),
                    "PUT": float(leg_eq.get("PUT", 1.0)),
                }
            st.cross_day_all_leg_defense = bool(blob.get("cross_day_all_leg_defense", False))
            legs = blob.get("cross_day_quarantined_legs") or []
            st.cross_day_quarantined_legs = {str(x).upper() for x in legs}
            if blob.get("vx_curve_slope") is not None:
                try:
                    st.vx_curve_slope = float(blob["vx_curve_slope"])
                except (TypeError, ValueError):
                    st.vx_curve_slope = None
            if blob.get("active_profile"):
                st.active_profile = str(blob["active_profile"])
            if isinstance(blob.get("profile_meta"), dict):
                st.profile_meta = dict(blob["profile_meta"])
            if blob.get("trades_today") is not None:
                st.trades_today = int(blob["trades_today"])
            if blob.get("day_pnl") is not None:
                st.day_pnl = float(blob["day_pnl"])
            st.vixy_open_shock_regime_active = bool(
                blob.get("vixy_open_shock_regime_active", False)
            )
            samples = blob.get("cross_asset_samples") or []
            restored: List[Tuple[int, float, float]] = []
            for sample in samples:
                if not isinstance(sample, (list, tuple)) or len(sample) != 3:
                    continue
                try:
                    restored.append(
                        (int(sample[0]), float(sample[1]), float(sample[2]))
                    )
                except (TypeError, ValueError):
                    continue
            st.cross_asset_samples = restored[-390:]
            if str(sym).upper() == "QQQ" and st.edge_buf is not None:
                n_main = len(st.edge_buf)
        # 恢复后按已存当日重新套 profile，使 session_start / CHOP 立即生效。
        for st in self._sym.values():
            if st.cur_day:
                from datetime import date as _date

                meta = self._apply_day_profile(
                    _date.fromisoformat(st.cur_day), st.vx_curve_slope
                )
                st.active_profile = str(meta.get("profile") or st.active_profile)
                st.profile_meta = dict(meta)
                break
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
        position_frac: Optional[float] = None,
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
        f = float(
            position_frac
            if position_frac is not None
            else self.effective_position_frac(symbol)
        )
        st.day_account_equity *= 1.0 + f * nr
        closed_leg = str(leg).upper()
        if closed_leg in st.day_leg_equity:
            st.day_leg_equity[closed_leg] *= 1.0 + f * nr
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
        state_path = _governor_state_path()
        if state_path:
            try:
                self.save_quantile_state(state_path)
            except Exception:
                pass
        return cool_until

    def effective_position_frac(self, symbol: str) -> float:
        """当日实际账户下注比例；防御日可降至 half-size。"""
        base = float(getattr(self.replay_cfg, "position_frac", 1.0) or 1.0)
        st = self._state(symbol)
        if not st.cross_day_all_leg_defense:
            return base
        defense = getattr(
            self.replay_cfg, "next_day_all_leg_defense_position_frac", None
        )
        if defense is None:
            return base
        return float(defense)

    def position_size_mult(self, symbol: str) -> float:
        """相对 REPLAY.position_frac 的 OMS 额度缩放（1.0=正常，0.5=半仓）。"""
        base = float(getattr(self.replay_cfg, "position_frac", 1.0) or 1.0)
        if base <= 0:
            return 1.0
        return float(self.effective_position_frac(symbol) / base)

    def all_leg_defense_active(self, symbol: str) -> bool:
        return bool(self._state(symbol).cross_day_all_leg_defense)

    def cross_day_gates(
        self,
        symbol: str,
        *,
        session_bar: Optional[int],
        edge_q10: Optional[float],
        leg: Optional[str] = None,
    ) -> Tuple[bool, str]:
        """跨日 PUT 隔离 / 全腿防御门控；True=拦截。"""
        st = self._state(symbol)
        rc = self.replay_cfg
        leg_u = str(leg or "").upper()
        if leg_u and leg_u in st.cross_day_quarantined_legs:
            return True, f"put_quarantine:{leg_u}"
        if (
            leg_u == "STRADDLE"
            and "PUT" in st.cross_day_quarantined_legs
        ):
            return True, "put_quarantine:STRADDLE"
        if not st.cross_day_all_leg_defense:
            return False, ""
        start = getattr(rc, "next_day_all_leg_defense_entry_start_bar", None)
        if (
            start is not None
            and session_bar is not None
            and int(session_bar) < int(start)
        ):
            return True, "all_leg_defense_entry_start"
        q10_floor = getattr(rc, "next_day_all_leg_defense_edge_q10_floor", None)
        if q10_floor is not None:
            if (
                edge_q10 is None
                or not np.isfinite(edge_q10)
                or float(edge_q10) < float(q10_floor)
            ):
                return True, "all_leg_defense_q10"
        return False, ""

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

    def note_vixy_open_shock_regime(
        self,
        symbol: str,
        *,
        session_bar: Optional[int],
        open30_ret: Optional[float],
        open30_peak_dd: Optional[float],
        trend_r2_30m: Optional[float],
    ) -> None:
        """与 ReplaySession 的 OPEN_SHOCK_CHOP 日内状态识别保持一致。"""
        rc = self.replay_cfg
        st = self._state(symbol)
        if (
            st.vixy_open_shock_regime_active
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
        vals = (open30_ret, open30_peak_dd, trend_r2_30m)
        if any(v is None or not np.isfinite(v) for v in vals):
            return
        st.vixy_open_shock_regime_active = bool(
            float(open30_ret) < float(rc.vixy_open_shock_open30_ret_max)
            and float(open30_peak_dd) <= float(rc.vixy_open_shock_peak_dd_max)
            and float(trend_r2_30m) < float(rc.vixy_open_shock_detect_r2_max)
        )

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

    def vixy_open_shock_regime_active(self, symbol: str) -> bool:
        return bool(self._state(symbol).vixy_open_shock_regime_active)

    def straddles_today_for(self, symbol: str) -> int:
        return self._state(symbol).straddles_today

    @property
    def straddles_today(self) -> Dict[str, int]:
        return {sym: st.straddles_today for sym, st in self._sym.items()}


def resolve_replay_cfg(replay_cfg: Optional[ReplayConfig] = None) -> ReplayConfig:
    """与 strategy_entry_bridge 同一套 cfg：strategy profile + env override。

    必须跨 entry / fill 共用同一 governor；禁止用 id(cfg) 做 key
    （USE_LIVE_REPLAY=1 时 entry 拿 LIVE、fill 默认 REPLAY → 双实例，
    max_trades / daily_loss 形同虚设，本周曾出现单日 7 笔）。

    优先级：显式 replay_cfg > QQQ_BTC_STRATEGY_PROFILE > config.py 默认；
    最后再应用 env emergency override。
    """
    import os
    from dataclasses import replace as _dc_replace

    if replay_cfg is not None:
        cfg = replay_cfg
    else:
        from qqq_btc.common.strategy_profile import (
            load_active_strategy_profile,
            materialize_replay_cfg,
        )

        profile = load_active_strategy_profile()
        if profile is not None:
            cfg = materialize_replay_cfg(profile)
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
        ("QQQ_BTC_PUT_QUARANTINE_LOSS", "next_day_put_quarantine_loss", float),
        ("QQQ_BTC_PUT_QUARANTINE_VX_SLOPE_MIN", "next_day_put_quarantine_vx_slope_min", float),
        ("QQQ_BTC_ALL_LEG_DEFENSE_LOSS", "next_day_all_leg_defense_loss", float),
        ("QQQ_BTC_ALL_LEG_DEFENSE_POSITION_FRAC", "next_day_all_leg_defense_position_frac", float),
        ("QQQ_BTC_ALL_LEG_DEFENSE_VX_SLOPE_MIN", "next_day_all_leg_defense_vx_slope_min", float),
        ("QQQ_BTC_ALL_LEG_DEFENSE_ENTRY_START_BAR", "next_day_all_leg_defense_entry_start_bar", int),
        ("QQQ_BTC_ALL_LEG_DEFENSE_EDGE_Q10_FLOOR", "next_day_all_leg_defense_edge_q10_floor", float),
        ("QQQ_BTC_MIN_DUAL_LEG_EDGE_GAP", "min_dual_leg_edge_gap", float),
        ("QQQ_BTC_SPOT_DAY_AGREE_EPS", "spot_day_agree_eps", float),
        ("QQQ_BTC_PUT_EARLY_CROSS_CONFIRM_END_BAR", "put_early_cross_confirm_end_bar", int),
        ("QQQ_BTC_PUT_EARLY_CROSS_CONFIRM_GAP_MAX", "put_early_cross_confirm_edge_gap_max", float),
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
    raw_spot_day = os.environ.get("QQQ_BTC_REQUIRE_LEG_SPOT_DAY_AGREE", "").strip().lower()
    if raw_spot_day in ("0", "false", "no", "off"):
        _ov["require_leg_spot_day_agree"] = False
    elif raw_spot_day in ("1", "true", "yes", "on"):
        _ov["require_leg_spot_day_agree"] = True
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
        gov.set_base_cfg(cfg)
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

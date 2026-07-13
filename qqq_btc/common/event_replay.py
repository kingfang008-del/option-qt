#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
事件驱动回放 —— 分钟信号 + 可选 1s tick 流,非向量化。

与 S4 的契约对齐(见 preprocess/backtest/second/s4_run_historical_replay_s2_1s.py):
  1. 数据层:秒级 option/stock + 分钟 alpha 先 merge_asof(+60s alpha 可见性) → 见 replay_io.build_s4_bundle
  2. 时钟层:alpha 左对齐标签 T 在 T+60s 才可交易(无 lookahead)
  3. 编排层:
     - signal_at=minute_open: 分钟首 tick 发信号(S4 ExecutionWindow 首包)
     - signal_at=minute_close: 分钟末发信号(L1 infer parquet 默认)
     - rails 退出:始终在分钟末 mid(CLOSE 相位)
     - 持仓期:秒级 tick 仅风险轨(tick_fast_hard + disaster),不跑完整 check_exit

当前 run_event_replay 若只喂 infer parquet + 独立 tick 文件、未走 build_s4_bundle,
则**决策状态机自洽**,但**不具备 S4 的数据融合与 alpha 延迟语义** —— 需显式选 --from-s4-sqlite
或在 infer 阶段保证 timestamp = alpha 可交易时刻(分钟收盘 + 延迟)。
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Union

import numpy as np
import pandas as pd

from .fill_model import OptionSpreadFillModel, PerpFillModel
from .regime_features import add_vix_regime_features
from .trend_features import add_spot_day_ret, add_trend_features
from .replay_session import BarPhase, ReplaySession, SessionQuotes, SessionSignal
from .replay_types import ReplayConfig, ReplayResult


class FillTiming(str, Enum):
    MINUTE_CLOSE = "minute_close"
    FIRST_TICK = "first_tick"


class SignalTiming(str, Enum):
    """分钟 alpha/edge 触发时刻。"""

    MINUTE_CLOSE = "minute_close"  # infer parquet / L1 默认
    MINUTE_OPEN = "minute_open"    # S4: ExecutionWindow 分钟边界首 tick


@dataclass
class EventReplayConfig:
    fill_timing: FillTiming = FillTiming.MINUTE_CLOSE
    signal_timing: SignalTiming = SignalTiming.MINUTE_CLOSE
    tick_disaster_stop: bool = True
    tick_smooth_n: int = 3
    # L1/infer replay 的分钟入场和普通 rails 仍使用 minute_df 官方盘口；
    # tick_df 只叠加秒级风险轨。S4 minute-open 编排不受此开关影响。
    use_tick_quotes_for_minute_close: bool = False


def _normalize_ts(series: pd.Series) -> pd.Series:
    ts = pd.to_datetime(series, utc=True, errors="coerce")
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize("UTC")
    return ts


def _minute_key(ts: pd.Timestamp) -> int:
    return int(ts.floor("min").timestamp())


def prepare_minute_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values("timestamp").reset_index(drop=True)
    out["timestamp"] = _normalize_ts(out["timestamp"])
    if "session_bar" not in out.columns:
        try:
            from .time_features import session_minute

            out["session_bar"] = session_minute(out["timestamp"]).astype(int)
        except Exception:
            out["session_bar"] = np.arange(len(out), dtype=int)
    out["_day"] = out["timestamp"].dt.date
    out["_minute_key"] = out["timestamp"].map(_minute_key)
    if "close" in out.columns and "spot_ret_5bar" not in out.columns:
        out["spot_ret_5bar"] = out.groupby("_day", sort=False)["close"].pct_change(5)
    if "vix_proxy_close" in out.columns and "vix_reversal_count_30m" not in out.columns:
        out = add_vix_regime_features(out)
    if "close" in out.columns and "spot_day_ret" not in out.columns:
        out = add_spot_day_ret(out)
    if "close" in out.columns and "spot_range_30m" not in out.columns:
        out = add_trend_features(out, price_col="close")
    return out


def prepare_tick_frame(tick_df: pd.DataFrame) -> pd.DataFrame:
    out = tick_df.copy()
    if "timestamp" not in out.columns and "ts" in out.columns:
        ts_raw = pd.to_numeric(out["ts"], errors="coerce")
        out["timestamp"] = pd.to_datetime(ts_raw, unit="s", utc=True)
    else:
        out["timestamp"] = _normalize_ts(out["timestamp"])
    out = out.sort_values("timestamp").reset_index(drop=True)
    out["_minute_key"] = out["timestamp"].map(_minute_key)
    return out


def _signal_from_row(
    row,
    *,
    edge_col: str,
    edge_q10_col: Optional[str],
    call_edge_col: Optional[str],
    put_edge_col: Optional[str],
    straddle_edge_col: Optional[str],
    put_gate_col: Optional[str] = None,
) -> SessionSignal:
    def _f(col: Optional[str]) -> Optional[float]:
        if col is None or col not in row.index:
            return None
        v = row[col]
        return float(v) if v is not None and np.isfinite(v) else None

    return SessionSignal(
        edge=_f(edge_col),
        call_edge=_f(call_edge_col),
        put_edge=_f(put_edge_col),
        straddle_edge=_f(straddle_edge_col),
        edge_q10=_f(edge_q10_col),
        put_gate=_f(put_gate_col),
        open30_max_ret=_f("open30_max_ret"),
        open30_peak_dd=_f("open30_peak_dd"),
        spot_ret_5bar=_f("spot_ret_5bar"),
        trend_ret_30m=_f("trend_fit_ret_30m"),
        trend_r2_30m=_f("trend_fit_r2_30m"),
        vix_reversal_count_30m=_f("vix_reversal_count_30m"),
        spot_day_ret=_f("spot_day_ret"),
        spot_range_30m=_f("spot_range_30m"),
        day_range_pos=_f("day_range_pos"),
        bb_width=_f("bb_width"),
    )


def _tick_mtm_smooth(buf: List[float], mtm: float, n: int) -> float:
    buf.append(mtm)
    if len(buf) > n:
        buf.pop(0)
    # 必须积满窗口才判定；否则所谓 5 秒确认会在第 1 秒退化成单点止损。
    return float(np.mean(buf)) if len(buf) >= n else float("nan")


def _run_disaster_ticks(
    session: ReplaySession,
    bar_index: int,
    ticks: pd.DataFrame,
    *,
    n_smooth: int,
) -> bool:
    """
    秒内风险止损;返回 True 若已平仓。

    disaster 用较短平滑(更快反应闪崩);
    tick_fast_hard 用较长平滑(降影线误杀)。
    """
    rails = session.rails_cfg
    n_disaster = int(getattr(rails, "disaster_smooth_n", None) or n_smooth or 3)
    n_fast = int(getattr(rails, "tick_fast_hard_smooth_n", None) or 5)
    n_profit = int(getattr(rails, "tick_profit_smooth_n", None) or 3)
    # fast_hard / 浮盈共用较长平滑窗(取 max),避免两套缓冲不一致
    n_risk_profit = max(n_fast, n_profit)
    buf_d: List[float] = []
    buf_rp: List[float] = []
    use_risk_profit = (
        rails.tick_fast_hard_roi is not None
        or rails.tick_profit_trigger_roi is not None
        or bool(rails.tick_profit_ladder)
    )

    for _, tick_row in ticks.iterrows():
        tq = SessionQuotes.from_row(tick_row)
        if session.position is None:
            return False
        mid = tq.mid(session.position.leg)
        if not (np.isfinite(mid) and mid > 0):
            continue

        # 1) 短平滑 → 仅 disaster
        smooth_d = _tick_mtm_smooth(buf_d, float(mid), n_disaster)
        evs = session.on_tick(
            bar_index,
            tick_row["timestamp"],
            tq,
            smoothed_mtm=smooth_d,
            disaster_only=True,
        )
        if evs and evs[0].kind == "DISASTER_EXIT":
            return True

        # 2) 风险+浮盈平滑 → fast_hard / tick_peak 回撤 / tick 阶梯
        if use_risk_profit and session.position is not None:
            smooth_rp = _tick_mtm_smooth(buf_rp, float(mid), n_risk_profit)
            evs = session.on_tick(
                bar_index,
                tick_row["timestamp"],
                tq,
                smoothed_mtm=smooth_rp,
                disaster_only=False,
            )
            if evs and evs[0].kind == "DISASTER_EXIT":
                return True
    return False


def run_event_replay(
    minute_df: pd.DataFrame,
    fill_model: Union[OptionSpreadFillModel, PerpFillModel],
    replay_cfg: ReplayConfig = ReplayConfig(),
    rails_cfg=None,
    *,
    tick_df: Optional[pd.DataFrame] = None,
    event_cfg: EventReplayConfig = EventReplayConfig(),
    edge_col: str = "net_edge",
    edge_q10_col: Optional[str] = None,
    call_edge_col: Optional[str] = None,
    put_edge_col: Optional[str] = None,
    straddle_edge_col: Optional[str] = None,
    put_gate_col: Optional[str] = None,
    signal_only: bool = False,
) -> ReplayResult:
    from .exit_rails import ExitRailsConfig

    rails_cfg = rails_cfg or ExitRailsConfig()
    minute_df = prepare_minute_frame(minute_df)
    is_option = isinstance(fill_model, OptionSpreadFillModel)
    default_leg = "CALL" if is_option else "PERP"

    dual_mode = (
        is_option
        and call_edge_col is not None
        and put_edge_col is not None
        and call_edge_col in minute_df.columns
        and put_edge_col in minute_df.columns
    )

    session = ReplaySession(
        replay_cfg,
        rails_cfg,
        fill_model,
        dual_mode=dual_mode,
        default_leg=default_leg,
        is_option=is_option,
        signal_only=signal_only,
    )

    ticks_by_minute: dict = {}
    has_ticks = tick_df is not None and not tick_df.empty
    if has_ticks:
        tick_df = prepare_tick_frame(tick_df)
        for mk, grp in tick_df.groupby("_minute_key", sort=True):
            ticks_by_minute[int(mk)] = grp.reset_index(drop=True)

    use_s4_orchestration = has_ticks and event_cfg.signal_timing == SignalTiming.MINUTE_OPEN
    n_smooth = max(1, event_cfg.tick_smooth_n)

    for bar_index, row in minute_df.iterrows():
        ts = row["timestamp"]
        session_bar = int(row["session_bar"]) if np.isfinite(row["session_bar"]) else None
        day_key = row["_day"]
        minute_key = int(row["_minute_key"])
        signal = _signal_from_row(
            row,
            edge_col=edge_col,
            edge_q10_col=edge_q10_col,
            call_edge_col=call_edge_col,
            put_edge_col=put_edge_col,
            straddle_edge_col=straddle_edge_col,
            put_gate_col=put_gate_col,
        )
        minute_quotes = SessionQuotes.from_row(row)
        ticks = ticks_by_minute.get(minute_key)
        has_minute_ticks = ticks is not None and len(ticks) > 0
        open_quotes = SessionQuotes.from_row(ticks.iloc[0]) if has_minute_ticks else minute_quotes
        close_quotes = SessionQuotes.from_row(ticks.iloc[-1]) if has_minute_ticks else minute_quotes

        if use_s4_orchestration:
            session.on_minute_bar(
                bar_index,
                ts,
                session_bar,
                open_quotes,
                signal,
                day_key=day_key,
                phase=BarPhase.OPEN,
                allow_entry=(event_cfg.fill_timing == FillTiming.FIRST_TICK),
                allow_signal=True,
            )
            if (
                has_minute_ticks
                and event_cfg.tick_disaster_stop
                and session.position is not None
            ):
                if _run_disaster_ticks(session, bar_index, ticks, n_smooth=n_smooth):
                    continue
            session.on_minute_bar(
                bar_index,
                ts,
                session_bar,
                close_quotes,
                SessionSignal(),
                day_key=day_key,
                phase=BarPhase.CLOSE,
                allow_signal=False,
                allow_entry=(event_cfg.fill_timing == FillTiming.MINUTE_CLOSE),
            )
            continue

        # --- L1 / minute_close 信号:单相位(分钟末) ---
        if has_minute_ticks and event_cfg.tick_disaster_stop and session.position is not None:
            if _run_disaster_ticks(session, bar_index, ticks, n_smooth=n_smooth):
                continue

        entry_quotes = minute_quotes
        if (
            session.pending_entry_bar is not None
            and bar_index >= session.pending_entry_bar
            and event_cfg.fill_timing == FillTiming.FIRST_TICK
            and has_minute_ticks
        ):
            entry_quotes = open_quotes

        close_q = (
            close_quotes
            if has_minute_ticks and event_cfg.use_tick_quotes_for_minute_close
            else minute_quotes
        )

        # 注意:持仓 bar 传空 signal → 这些 bar 的 edge 不进分位缓冲。
        # 副作用:出场早晚会通过缓冲改写之后几天的动态阈值(路径依赖,与 live 侧
        # entry bridge 仅在空仓时 record_edges 的语义一致,暂保留)。
        if session.position is not None:
            session.on_minute_bar(
                bar_index, ts, session_bar, close_q, SessionSignal(),
                day_key=day_key, phase=BarPhase.CLOSE, allow_signal=False,
                allow_entry=(event_cfg.fill_timing == FillTiming.MINUTE_CLOSE),
            )
            continue

        if session.pending_entry_bar is not None and bar_index >= session.pending_entry_bar:
            session.on_minute_bar(
                bar_index, ts, session_bar, entry_quotes, SessionSignal(),
                day_key=day_key, phase=BarPhase.CLOSE,
                allow_signal=False,
                allow_entry=True,
            )
            continue

        session.on_minute_bar(
            bar_index,
            ts,
            session_bar,
            close_q,
            signal,
            day_key=day_key,
            phase=BarPhase.CLOSE,
            allow_signal=True,
            allow_entry=(event_cfg.fill_timing == FillTiming.MINUTE_CLOSE),
        )

    result = session.result
    result.events = list(session.events)
    return result


def compare_minute_vs_event(
    minute_df: pd.DataFrame,
    tick_df: pd.DataFrame,
    fill_model: Union[OptionSpreadFillModel, PerpFillModel],
    replay_cfg: ReplayConfig,
    rails_cfg=None,
    **kwargs,
) -> dict:
    r_min = run_event_replay(
        minute_df, fill_model, replay_cfg, rails_cfg, tick_df=None, **kwargs
    )
    r_evt = run_event_replay(
        minute_df,
        fill_model,
        replay_cfg,
        rails_cfg,
        tick_df=tick_df,
        event_cfg=EventReplayConfig(
            fill_timing=FillTiming.FIRST_TICK,
            signal_timing=SignalTiming.MINUTE_OPEN,
            tick_disaster_stop=True,
        ),
        **kwargs,
    )
    s_min, s_evt = r_min.summary(), r_evt.summary()
    return {
        "minute": s_min,
        "event_s4": s_evt,
        "trade_count_delta": s_evt.get("trades", 0) - s_min.get("trades", 0),
        "return_delta": s_evt.get("total_net_return", 0) - s_min.get("total_net_return", 0),
    }


__all__ = [
    "EventReplayConfig",
    "FillTiming",
    "SignalTiming",
    "compare_minute_vs_event",
    "prepare_minute_frame",
    "prepare_tick_frame",
    "run_event_replay",
]

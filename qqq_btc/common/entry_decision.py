#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
入场决策 —— strict replay 与 live 信号层共用同一实现。

设计约束(ARCHITECTURE 2.4):实盘决策不得复制 replay 逻辑,
必须 import 本模块,保证 threshold / 腿竞争 / 门控语义一致。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .replay_types import ReplayConfig


def _morning_fade_ok(
    replay_cfg: ReplayConfig,
    session_bar: Optional[int],
    open30_max_ret: Optional[float],
    open30_peak_dd: Optional[float],
) -> bool:
    mn = replay_cfg.morning_fade_min_ret
    md = replay_cfg.morning_fade_max_peak_dd
    if mn is None or md is None:
        return False
    end = replay_cfg.morning_fade_session_end_bar
    if end is not None and session_bar is not None and session_bar > int(end):
        return False
    if open30_max_ret is None or open30_peak_dd is None:
        return False
    if not (np.isfinite(open30_max_ret) and np.isfinite(open30_peak_dd)):
        return False
    return float(open30_max_ret) >= float(mn) and float(open30_peak_dd) <= float(md)


@dataclass
class EntryDecision:
    leg: str           # CALL / PUT / STRADDLE / PERP
    edge: float
    threshold: float


def choose_entry(
    replay_cfg: ReplayConfig,
    *,
    session_bar: Optional[int],
    edge: Optional[float] = None,
    call_edge: Optional[float] = None,
    put_edge: Optional[float] = None,
    straddle_edge: Optional[float] = None,
    edge_q10: Optional[float] = None,
    spread_pct: float = 0.0,
    put_spread_pct: Optional[float] = None,
    straddle_spread_pct: Optional[float] = None,
    dual_mode: bool = False,
    has_put: bool = False,
    straddle_enabled: bool = False,
    straddles_today: int = 0,
    default_leg: str = "CALL",
    dynamic_threshold: Optional[float] = None,
    put_dynamic_threshold: Optional[float] = None,
    put_gate: Optional[float] = None,
    open30_max_ret: Optional[float] = None,
    open30_peak_dd: Optional[float] = None,
    spot_ret_5bar: Optional[float] = None,
    trend_ret_30m: Optional[float] = None,
    trend_r2_30m: Optional[float] = None,
    vix_reversal_count_30m: Optional[float] = None,
    spot_day_ret: Optional[float] = None,
    spot_range_30m: Optional[float] = None,
) -> Optional[EntryDecision]:
    """
    单 bar 入场决策。spread 门控在调用方二次校验(各腿 spread 不同)。
    dynamic_threshold / put_dynamic_threshold:各腿滚动分位阈值(调用方维护),
    只抬高不降低静态阈值 —— 两个头的打分尺度不同(put 头 std 约为主头一半),
    必须按各自分布收紧。
    put_gate:PUT 腿行情开关信号值(见 ReplayConfig.put_gate_min);
    门控开启且信号缺失/低于门槛时 PUT 不可开仓。返回 None = 不入场。
    """
    if not replay_cfg.session_allows_entry(session_bar):
        return None

    if replay_cfg.regime_vix_reversal_max is not None:
        if (
            vix_reversal_count_30m is not None
            and np.isfinite(vix_reversal_count_30m)
            and vix_reversal_count_30m > float(replay_cfg.regime_vix_reversal_max)
        ):
            return None

    put_gate_ok = True
    fade_ok = False
    if replay_cfg.put_gate_min is not None:
        vix_ok = (
            put_gate is not None
            and np.isfinite(put_gate)
            and put_gate >= float(replay_cfg.put_gate_min)
        )
        fade_ok = _morning_fade_ok(
            replay_cfg, session_bar, open30_max_ret, open30_peak_dd
        )
        put_gate_ok = vix_ok or fade_ok

    # 早盘 PUT 加强:更高 VIX / 开盘结构 / 波动结构(July1 型)
    early_bar = replay_cfg.put_early_session_bar
    in_early = (
        early_bar is not None
        and session_bar is not None
        and session_bar < int(early_bar)
    )
    if in_early and replay_cfg.put_early_vix_min is not None:
        early_vix_ok = (
            put_gate is not None
            and np.isfinite(put_gate)
            and put_gate >= float(replay_cfg.put_early_vix_min)
        )
        put_gate_ok = bool(early_vix_ok or fade_ok)

    call_blocked = False
    if replay_cfg.block_call_on_rapid_drop and replay_cfg.rapid_drop_ret is not None:
        if (
            spot_ret_5bar is not None
            and np.isfinite(spot_ret_5bar)
            and spot_ret_5bar <= float(replay_cfg.rapid_drop_ret)
        ):
            call_blocked = True
    if replay_cfg.call_trend_r2_min is not None:
        if (
            trend_r2_30m is not None
            and np.isfinite(trend_r2_30m)
            and trend_r2_30m < float(replay_cfg.call_trend_r2_min)
        ):
            call_blocked = True
    if replay_cfg.call_chase_vix_rev_min is not None:
        chase_floor = replay_cfg.call_chase_spot_day_ret_min
        if (
            spot_day_ret is not None
            and np.isfinite(spot_day_ret)
            and spot_day_ret > float(chase_floor)
            and vix_reversal_count_30m is not None
            and np.isfinite(vix_reversal_count_30m)
            and vix_reversal_count_30m >= float(replay_cfg.call_chase_vix_rev_min)
        ):
            call_blocked = True
    if replay_cfg.call_spike_range30_min is not None:
        if (
            spot_range_30m is not None
            and np.isfinite(spot_range_30m)
            and spot_range_30m >= float(replay_cfg.call_spike_range30_min)
        ):
            call_blocked = True
    if (
        replay_cfg.call_timing_max_bar is not None
        and replay_cfg.call_timing_spot_min is not None
        and replay_cfg.call_timing_vix_min is not None
    ):
        t_spot = replay_cfg.call_timing_spot_min
        t_bar = int(replay_cfg.call_timing_max_bar)
        t_vix = int(replay_cfg.call_timing_vix_min)
        if (
            session_bar is not None
            and session_bar < t_bar
            and spot_day_ret is not None
            and np.isfinite(spot_day_ret)
            and spot_day_ret > float(t_spot)
            and vix_reversal_count_30m is not None
            and np.isfinite(vix_reversal_count_30m)
            and vix_reversal_count_30m >= float(t_vix)
        ):
            call_blocked = True

    # 趋势对齐:30min 趋势仍向上时禁开 PUT(强于 vix/fade 门控,减法保护)
    put_blocked = False
    if replay_cfg.put_trend_max_ret is not None:
        if (
            trend_ret_30m is not None
            and np.isfinite(trend_ret_30m)
            and trend_ret_30m > float(replay_cfg.put_trend_max_ret)
        ):
            put_blocked = True
    if replay_cfg.put_late_session_bar is not None:
        if (
            session_bar is not None
            and session_bar > int(replay_cfg.put_late_session_bar)
        ):
            put_blocked = True
    if replay_cfg.put_spot_day_ret_min is not None:
        if (
            spot_day_ret is not None
            and np.isfinite(spot_day_ret)
            and spot_day_ret > float(replay_cfg.put_spot_day_ret_min)
        ):
            put_blocked = True
    if in_early and replay_cfg.put_early_open30_max_min is not None:
        if (
            open30_max_ret is None
            or not np.isfinite(open30_max_ret)
            or float(open30_max_ret) < float(replay_cfg.put_early_open30_max_min)
        ):
            put_blocked = True
    if in_early and replay_cfg.put_early_range30_min is not None:
        if (
            spot_range_30m is None
            or not np.isfinite(spot_range_30m)
            or float(spot_range_30m) < float(replay_cfg.put_early_range30_min)
        ):
            put_blocked = True

    th_static = replay_cfg.threshold_at(session_bar)
    th = th_static
    if dynamic_threshold is not None and np.isfinite(dynamic_threshold):
        th = max(th, float(dynamic_threshold))
    th_put = th_static
    if put_dynamic_threshold is not None and np.isfinite(put_dynamic_threshold):
        th_put = max(th_put, float(put_dynamic_threshold))

    if not np.isfinite(spread_pct) or spread_pct > replay_cfg.max_spread_pct:
        return None

    chosen_leg: Optional[str] = None
    chosen_edge = 0.0

    if dual_mode and call_edge is not None and put_edge is not None:
        ec = call_edge if np.isfinite(call_edge) else -np.inf
        ep = put_edge if np.isfinite(put_edge) else -np.inf
        if replay_cfg.long_only or not has_put or not put_gate_ok or put_blocked:
            ep = -np.inf
        candidates = [
            (v, leg)
            for v, leg, t in ((ec, "CALL", th), (ep, "PUT", th_put))
            if v >= t and not (leg == "CALL" and call_blocked)
        ]
        if candidates:
            chosen_edge, chosen_leg = max(candidates)
    elif edge is not None and np.isfinite(edge):
        if edge >= th and not call_blocked:
            chosen_leg, chosen_edge = default_leg, float(edge)
        elif edge <= -th and not replay_cfg.long_only and has_put and put_gate_ok and not put_blocked:
            chosen_leg, chosen_edge = "PUT", float(edge)

    if straddle_enabled and straddle_edge is not None and np.isfinite(straddle_edge):
        th_s = replay_cfg.straddle_entry_threshold if replay_cfg.straddle_entry_threshold is not None else th
        allowed = (
            replay_cfg.max_straddles_per_day is None
            or straddles_today < replay_cfg.max_straddles_per_day
        )
        if allowed and straddle_edge >= th_s and not call_blocked and (chosen_leg is None or straddle_edge > abs(chosen_edge)):
            if straddle_spread_pct is not None and (
                not np.isfinite(straddle_spread_pct) or straddle_spread_pct > replay_cfg.max_spread_pct
            ):
                pass
            else:
                chosen_leg, chosen_edge = "STRADDLE", float(straddle_edge)

    if chosen_leg is None:
        return None

    if chosen_leg == "CALL" and edge_q10 is not None:
        # 0DTE:真实标签 p10 常为负,用 floor 而非强制 q10>0
        floor = replay_cfg.edge_q10_floor
        if floor is not None:
            if not (np.isfinite(edge_q10) and edge_q10 > float(floor)):
                return None

    if chosen_leg == "PUT" and put_spread_pct is not None:
        if not np.isfinite(put_spread_pct) or put_spread_pct > replay_cfg.max_spread_pct:
            return None

    return EntryDecision(leg=chosen_leg, edge=chosen_edge, threshold=th)

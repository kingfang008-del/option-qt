#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
滚动 entry_quantile 缓冲 —— strict replay 与 live 共用门控语义。

replay_session 在 CLOSE 阶段 append;live 在 decide_entry 前 append。
"""
from __future__ import annotations

from collections import deque
from typing import Deque, Optional

import numpy as np

from .replay_types import ReplayConfig


def quantile_threshold(buf: Optional[Deque[float]], replay_cfg: ReplayConfig) -> Optional[float]:
    q = getattr(replay_cfg, "entry_quantile", None)
    if buf is None or q is None or len(buf) < int(replay_cfg.entry_quantile_min_obs):
        return None
    return float(np.quantile(np.asarray(buf, dtype=float), float(q)))


def regime_buffer_blocked(
    replay_cfg: ReplayConfig,
    *,
    vix_reversal_count_30m: Optional[float],
) -> bool:
    mx = replay_cfg.regime_vix_reversal_max
    if mx is None:
        return False
    if vix_reversal_count_30m is None or not np.isfinite(vix_reversal_count_30m):
        return False
    return float(vix_reversal_count_30m) > float(mx)


def call_edge_buffer_blocked(
    replay_cfg: ReplayConfig,
    session_bar: Optional[int],
    *,
    trend_r2_30m: Optional[float],
    spot_day_ret: Optional[float],
    vix_reversal_count_30m: Optional[float],
    spot_range_30m: Optional[float],
) -> bool:
    if replay_cfg.call_trend_r2_min is not None:
        if (
            trend_r2_30m is not None
            and np.isfinite(trend_r2_30m)
            and trend_r2_30m < float(replay_cfg.call_trend_r2_min)
        ):
            return True
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
            return True
    if replay_cfg.call_spike_range30_min is not None:
        if (
            spot_range_30m is not None
            and np.isfinite(spot_range_30m)
            and spot_range_30m >= float(replay_cfg.call_spike_range30_min)
        ):
            return True
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
            return True
    return regime_buffer_blocked(replay_cfg, vix_reversal_count_30m=vix_reversal_count_30m)


def put_edge_buffer_blocked(
    replay_cfg: ReplayConfig,
    session_bar: Optional[int],
    *,
    trend_ret_30m: Optional[float],
    spot_day_ret: Optional[float],
    vix_reversal_count_30m: Optional[float],
) -> bool:
    if replay_cfg.put_trend_max_ret is not None:
        if (
            trend_ret_30m is not None
            and np.isfinite(trend_ret_30m)
            and trend_ret_30m > float(replay_cfg.put_trend_max_ret)
        ):
            return True
    if replay_cfg.put_late_session_bar is not None:
        if session_bar is not None and session_bar > int(replay_cfg.put_late_session_bar):
            return True
    if replay_cfg.put_spot_day_ret_min is not None:
        if (
            spot_day_ret is not None
            and np.isfinite(spot_day_ret)
            and spot_day_ret > float(replay_cfg.put_spot_day_ret_min)
        ):
            return True
    return regime_buffer_blocked(replay_cfg, vix_reversal_count_30m=vix_reversal_count_30m)


def maybe_append_edge_buffers(
    replay_cfg: ReplayConfig,
    *,
    session_bar: Optional[int],
    call_edge: Optional[float],
    put_edge: Optional[float],
    dual_mode: bool,
    edge_buf: Optional[Deque[float]],
    put_edge_buf: Optional[Deque[float]],
    edge: Optional[float] = None,
    trend_r2_30m: Optional[float] = None,
    spot_day_ret: Optional[float] = None,
    vix_reversal_count_30m: Optional[float] = None,
    spot_range_30m: Optional[float] = None,
    trend_ret_30m: Optional[float] = None,
) -> None:
    """在入场窗内、未被门控拦截的 bar 追加 edge 观测(与 replay_session CLOSE 一致)。"""
    if edge_buf is None or not replay_cfg.session_allows_entry(session_bar):
        return

    main_edge = call_edge if dual_mode and call_edge is not None else edge
    if main_edge is None:
        main_edge = call_edge

    if (
        main_edge is not None
        and np.isfinite(main_edge)
        and not call_edge_buffer_blocked(
            replay_cfg,
            session_bar,
            trend_r2_30m=trend_r2_30m,
            spot_day_ret=spot_day_ret,
            vix_reversal_count_30m=vix_reversal_count_30m,
            spot_range_30m=spot_range_30m,
        )
    ):
        edge_buf.append(float(main_edge))

    if (
        put_edge_buf is not None
        and put_edge is not None
        and np.isfinite(put_edge)
        and not put_edge_buffer_blocked(
            replay_cfg,
            session_bar,
            trend_ret_30m=trend_ret_30m,
            spot_day_ret=spot_day_ret,
            vix_reversal_count_30m=vix_reversal_count_30m,
        )
    ):
        put_edge_buf.append(float(put_edge))

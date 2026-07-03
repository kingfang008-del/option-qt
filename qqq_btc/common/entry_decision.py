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
) -> Optional[EntryDecision]:
    """
    单 bar 入场决策。spread 门控在调用方二次校验(各腿 spread 不同)。
    返回 None = 不入场。
    """
    if not replay_cfg.session_allows_entry(session_bar):
        return None

    th = replay_cfg.threshold_at(session_bar)

    if not np.isfinite(spread_pct) or spread_pct > replay_cfg.max_spread_pct:
        return None

    chosen_leg: Optional[str] = None
    chosen_edge = 0.0

    if dual_mode and call_edge is not None and put_edge is not None:
        ec = call_edge if np.isfinite(call_edge) else -np.inf
        ep = put_edge if np.isfinite(put_edge) else -np.inf
        if replay_cfg.long_only:
            ep = -np.inf
        candidates = [(v, leg) for v, leg in ((ec, "CALL"), (ep, "PUT")) if v >= th]
        if candidates:
            chosen_edge, chosen_leg = max(candidates)
    elif edge is not None and np.isfinite(edge):
        if edge >= th:
            chosen_leg, chosen_edge = default_leg, float(edge)
        elif edge <= -th and not replay_cfg.long_only and has_put:
            chosen_leg, chosen_edge = "PUT", float(edge)

    if straddle_enabled and straddle_edge is not None and np.isfinite(straddle_edge):
        th_s = replay_cfg.straddle_entry_threshold if replay_cfg.straddle_entry_threshold is not None else th
        allowed = (
            replay_cfg.max_straddles_per_day is None
            or straddles_today < replay_cfg.max_straddles_per_day
        )
        if allowed and straddle_edge >= th_s and (chosen_leg is None or straddle_edge > abs(chosen_edge)):
            if straddle_spread_pct is not None and (
                not np.isfinite(straddle_spread_pct) or straddle_spread_pct > replay_cfg.max_spread_pct
            ):
                pass
            else:
                chosen_leg, chosen_edge = "STRADDLE", float(straddle_edge)

    if chosen_leg is None:
        return None

    if chosen_leg == "CALL" and edge_q10 is not None:
        if not (np.isfinite(edge_q10) and edge_q10 > 0):
            return None

    if chosen_leg == "PUT" and put_spread_pct is not None:
        if not np.isfinite(put_spread_pct) or put_spread_pct > replay_cfg.max_spread_pct:
            return None

    return EntryDecision(leg=chosen_leg, edge=chosen_edge, threshold=th)

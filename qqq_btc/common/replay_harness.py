#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
strict replay 骨架 —— 模型验收的唯一口径(L1 分钟)。

实现:逐 bar 调用 ReplaySession(与 event replay / live 共用状态机)。
L2 事件回放见 common/event_replay.py、tools/run_event_replay.py。
"""
from __future__ import annotations

from typing import Optional, Union

import pandas as pd

from .event_replay import run_event_replay
from .exit_rails import ExitRailsConfig
from .fill_model import OptionSpreadFillModel, PerpFillModel
from .replay_types import ReplayConfig, ReplayResult, Trade

__all__ = [
    "ReplayConfig",
    "ReplayResult",
    "Trade",
    "run_strict_replay",
]


def run_strict_replay(
    df: pd.DataFrame,
    fill_model: Union[OptionSpreadFillModel, PerpFillModel],
    replay_cfg: ReplayConfig = ReplayConfig(),
    rails_cfg: ExitRailsConfig = ExitRailsConfig(),
    edge_col: str = "net_edge",
    bid_col: str = "exec_call_bid",
    ask_col: str = "exec_call_ask",
    price_col: str = "close",
    spread_col: Optional[str] = "exec_call_spread_pct",
    session_col: Optional[str] = "session_bar",
    edge_q10_col: Optional[str] = None,
    put_bid_col: str = "exec_put_bid",
    put_ask_col: str = "exec_put_ask",
    put_spread_col: Optional[str] = "exec_put_spread_pct",
    call_edge_col: Optional[str] = None,
    put_edge_col: Optional[str] = None,
    straddle_edge_col: Optional[str] = None,
    put_gate_col: Optional[str] = None,
) -> ReplayResult:
    """L1 分钟 strict replay —— 委托 event_replay(无 tick 流)。"""
    work = df.copy()
    if bid_col != "exec_call_bid" and bid_col in work.columns:
        work["exec_call_bid"] = work[bid_col]
    if ask_col != "exec_call_ask" and ask_col in work.columns:
        work["exec_call_ask"] = work[ask_col]
    if spread_col and spread_col in work.columns and "exec_call_spread_pct" not in work.columns:
        work["exec_call_spread_pct"] = work[spread_col]
    if put_bid_col in work.columns:
        work["exec_put_bid"] = work[put_bid_col]
    if put_ask_col in work.columns:
        work["exec_put_ask"] = work[put_ask_col]
    if put_spread_col and put_spread_col in work.columns:
        work["exec_put_spread_pct"] = work[put_spread_col]
    if session_col and session_col in work.columns and "session_bar" not in work.columns:
        work["session_bar"] = work[session_col]

    if isinstance(fill_model, PerpFillModel) and price_col in work.columns:
        work["exec_call_bid"] = pd.to_numeric(work[price_col], errors="coerce")
        work["exec_call_ask"] = work["exec_call_bid"]

    return run_event_replay(
        work,
        fill_model,
        replay_cfg,
        rails_cfg,
        tick_df=None,
        edge_col=edge_col,
        edge_q10_col=edge_q10_col,
        call_edge_col=call_edge_col,
        put_edge_col=put_edge_col,
        straddle_edge_col=straddle_edge_col,
        put_gate_col=put_gate_col,
    )

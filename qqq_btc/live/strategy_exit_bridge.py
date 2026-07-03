#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
StrategyCore.check_exit → qqq_btc exit_rails 桥接(分钟 mid 口径)。

仅在 QQQ_BTC_LIVE bootstrap 时 monkey-patch;legacy 多层 exit 规则(MACD fade/正股止损等)
在此模式下 bypass,与 strict replay 对齐。
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd

from qqq_btc.common.exit_rails import ExitRailsConfig, PositionState, check_exit
from qqq_btc.common.time_features import session_minute
from qqq_btc.qqq import config as qcfg


def _session_bar_from_ctx(ctx: dict) -> int:
    t = ctx.get("time")
    if t is None:
        return 0
    try:
        return int(session_minute(pd.Series([pd.Timestamp(t)])).iloc[0])
    except Exception:
        return 0


def check_exit_via_rails(
    ctx: dict,
    rails: Optional[ExitRailsConfig] = None,
) -> Optional[dict]:
    """
    用 exit_rails.check_exit 替换 StrategyCore 分钟退出链。
    返回 legacy SELL dict 或 None。
    """
    pos = ctx.get("holding")
    if not pos:
        return None

    rails = rails or qcfg.EXIT_RAILS
    entry_price = float(pos.get("entry_price", 0.0) or 0.0)
    curr_price = float(ctx.get("curr_price", 0.0) or 0.0)
    if entry_price <= 0 or curr_price <= 0:
        return None

    held_mins = float(ctx.get("held_mins", 0.0) or 0.0)
    entry_bar = int(pos.get("entry_bar", 0) or 0)
    current_bar = entry_bar + max(0, int(round(held_mins)))

    ps = PositionState(entry_price=entry_price, entry_bar=entry_bar)
    ps.max_roi = float(pos.get("max_roi", 0.0) or 0.0)
    # 同步 ctx 棘轮后的 max_roi
    roi_now = curr_price / entry_price - 1.0
    if roi_now > ps.max_roi:
        ps.max_roi = roi_now
        pos["max_roi"] = ps.max_roi

    reason = check_exit(
        rails,
        ps,
        curr_price,
        current_bar,
        session_bar_index=_session_bar_from_ctx(ctx),
    )
    if reason:
        return {"action": "SELL", "reason": f"QQQ_BTC_{reason}", "dir": pos.get("dir", 1)}
    return None


def apply_strategy_exit_patch(StrategyCore) -> None:
    """Monkey-patch StrategyCore.check_exit → exit_rails(qqq_btc 口径)。"""

    _orig = StrategyCore.check_exit

    def _patched_check_exit(self, ctx: dict):
        sig = check_exit_via_rails(ctx)
        if sig is not None:
            return sig
        return _orig(self, ctx)

    def _rails_only(self, ctx: dict):
        return check_exit_via_rails(ctx)

    # 默认 rails-only;设 QQQ_BTC_EXIT_FALLBACK=1 可回退 legacy 链
    import os

    if os.environ.get("QQQ_BTC_EXIT_FALLBACK", "").strip().lower() in ("1", "true", "yes"):
        StrategyCore.check_exit = _patched_check_exit
    else:
        StrategyCore.check_exit = _rails_only

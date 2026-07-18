#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
StrategyCore.check_exit → qqq_btc exit_rails 桥接(分钟 mid 口径)。

仅在 QQQ_BTC_LIVE bootstrap 时 monkey-patch;legacy 多层 exit 规则(MACD fade/正股止损等)
在此模式下 bypass,与 strict replay 对齐。
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from qqq_btc.common.exit_rails import (
    ExitRailsConfig,
    PositionState,
    check_exit,
    check_forced_time_exit,
    check_spot_thesis_invalidate,
)
from qqq_btc.live.live_clock import live_session_bar
from qqq_btc.qqq import config as qcfg

logger = logging.getLogger("qqq_btc.live.strategy_exit")


def _session_bar_from_ctx(ctx: dict) -> int:
    t = ctx.get("time")
    if t is None:
        return 0
    try:
        return live_session_bar(t)
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
    if pos.get("qqq_btc_exit_rails") is not None:
        rails = pos["qqq_btc_exit_rails"]

    held_mins = float(ctx.get("held_mins", 0.0) or 0.0)
    entry_bar = int(pos.get("entry_bar", 0) or 0)
    session_bar = _session_bar_from_ctx(ctx)
    # 优先用绝对 session_bar(与 offline replay 一致);held_mins 仅作 fallback
    if session_bar > 0 and entry_bar >= 0:
        current_bar = int(session_bar)
    else:
        current_bar = entry_bar + max(0, int(round(held_mins)))

    # MAX_HOLD / EOD 不依赖 MTM；盘口短暂缺失也必须向 OMS 发出强平指令。
    forced_reason = check_forced_time_exit(
        rails,
        entry_bar=entry_bar,
        current_bar=current_bar,
        session_bar_index=session_bar if session_bar > 0 else None,
    )
    entry_price = float(pos.get("entry_price", 0.0) or 0.0)
    curr_price = float(ctx.get("curr_price", 0.0) or 0.0)
    if entry_price <= 0 or curr_price <= 0:
        if forced_reason:
            return {
                "action": "SELL",
                "reason": f"QQQ_BTC_{forced_reason}|NO_QUOTE",
                "dir": pos.get("dir", 1),
            }
        return None

    ps = PositionState(entry_price=entry_price, entry_bar=entry_bar)
    ps.max_roi = float(pos.get("max_roi", 0.0) or 0.0)
    # 同步 ctx 棘轮后的 max_roi
    roi_now = curr_price / entry_price - 1.0
    if roi_now > ps.max_roi:
        ps.max_roi = roi_now
        pos["max_roi"] = ps.max_roi

    # 现货路径证伪(与 offline ReplaySession 同口径)
    spot_px = None
    for k in ("spot_close", "stock_price", "qqq_close", "close"):
        v = ctx.get(k)
        if v is not None:
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            if fv > 0:
                spot_px = fv
                break
    if spot_px is not None:
        buf = pos.setdefault("qqq_btc_spot_closes", [])
        if not buf or abs(float(buf[-1]) - spot_px) > 1e-12:
            buf.append(spot_px)
        if pos.get("entry_spot") is None:
            pos["entry_spot"] = float(spot_px)
        held = max(0, int(current_bar) - int(entry_bar))
        thesis = check_spot_thesis_invalidate(
            rails,
            leg=str(pos.get("leg") or pos.get("chosen_leg") or ("PUT" if int(pos.get("dir", 1) or 1) < 0 else "CALL")),
            spot_closes=buf,
            entry_spot=float(pos["entry_spot"]),
            held=held,
        )
        if rails.spot_thesis_against_entry is not None:
            logger.debug(
                "bounce thesis check leg=%s held=%s spot=%.4f entry_spot=%.4f result=%s",
                pos.get("leg") or pos.get("chosen_leg"),
                held,
                spot_px,
                float(pos["entry_spot"]),
                thesis,
            )
        if thesis:
            return {"action": "SELL", "reason": f"QQQ_BTC_{thesis}", "dir": pos.get("dir", 1)}

    reason = check_exit(
        rails,
        ps,
        curr_price,
        current_bar,
        session_bar_index=session_bar if session_bar > 0 else None,
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

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Shared exit rails: profit ladder / flash / trailing / counter-trend protect,
plus merged option stop floors for TREND + V0.

Single source of truth for LADDER_* config interpretation so TREND (via
super().check_exit) and V0 use the same implementation.
"""

from __future__ import annotations

import logging
from typing import Callable, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from strategy_config0 import StrategyConfig

logger = logging.getLogger("StrategyCore")

TraceFn = Optional[Callable[[str, str, str], None]]


def select_profit_ladder(cfg: "StrategyConfig", pos: dict) -> List[Tuple[float, float]]:
    """TIGHT vs WIDE ladder rows from config (same rules as StrategyCoreV0)."""
    if not getattr(cfg, "DYNAMIC_LADDER_ENABLED", False):
        return list(cfg.LADDER_TIGHT)
    init_alpha = abs(pos.get("init_ctx", {}).get("alpha_z", 0.0))
    if init_alpha >= getattr(cfg, "HIGH_ALPHA_WIDE_THRESHOLD", 2.5):
        return list(cfg.LADDER_WIDE)
    return list(cfg.LADDER_TIGHT)


def evaluate_profit_rails(
    cfg: "StrategyConfig",
    pos: dict,
    current_roi: float,
    trace: TraceFn = None,
) -> Optional[dict]:
    """
    Unified profit locking: counter-trend protect, TRAILING_EPIC, STEP_PROT
    ladder, FLASH L0. Returns SELL dict or None.
    """
    def _t(key: str, status: str, msg: str) -> None:
        if trace is not None:
            trace(key, status, msg)

    max_roi = pos.get("max_roi", 0.0)

    is_counter_trend = False
    if "entry_spy_roc" in pos:
        raw_spy_roc = pos["entry_spy_roc"]
        try:
            spy_roc_val = float(raw_spy_roc) if not isinstance(raw_spy_roc, dict) else 0.0
        except (TypeError, ValueError):
            spy_roc_val = 0.0
        if pos["dir"] == 1 and spy_roc_val < -0.0001:
            is_counter_trend = True
        elif pos["dir"] == -1 and spy_roc_val > 0.0001:
            is_counter_trend = True

    if is_counter_trend:
        if max_roi > cfg.COUNTER_TREND_PROTECT_TRIGGER and current_roi < cfg.COUNTER_TREND_PROTECT_EXIT:
            _t(
                "X10.protect_counter",
                "block",
                f"逆势 max={max_roi:.1%}>trig={cfg.COUNTER_TREND_PROTECT_TRIGGER:.0%} "
                f"cur={current_roi:.1%}<exit={cfg.COUNTER_TREND_PROTECT_EXIT:.0%}",
            )
            return {"action": "SELL", "reason": f"PROTECT_COUNTER({max_roi:.1%}->{current_roi:.1%})"}
        _t("X10.protect_counter", "pass", f"逆势但未触发 max={max_roi:.1%} cur={current_roi:.1%}")
    else:
        _t("X10.protect_counter", "skip", "顺势单")

    if max_roi >= cfg.TRAILING_TRIGGER_ROI:
        trailing_exit = max_roi * cfg.TRAILING_KEEP_RATIO
        if current_roi < trailing_exit:
            _t(
                "X10.trailing_epic",
                "block",
                f"max={max_roi:.1%} cur={current_roi:.1%} < trail={trailing_exit:.1%}",
            )
            return {"action": "SELL", "reason": f"TRAILING_EPIC({max_roi:.1%}->{current_roi:.1%})"}
        _t(
            "X10.trailing_epic",
            "pass",
            f"Epic 就位 max={max_roi:.1%} cur={current_roi:.1%} 高于 trail={trailing_exit:.1%}",
        )
    else:
        _t(
            "X10.trailing_epic",
            "skip",
            f"max_roi={max_roi:.1%} < trigger={cfg.TRAILING_TRIGGER_ROI:.0%}",
        )

    ladder = select_profit_ladder(cfg, pos)
    step_matched = False
    for trigger, floor in reversed(ladder):
        if max_roi >= trigger:
            step_matched = True
            if current_roi < floor:
                _t(
                    "X10.step_protect",
                    "block",
                    f"档位T={trigger:.2f} floor={floor:.2f} max={max_roi:.1%} cur={current_roi:.1%}",
                )
                return {
                    "action": "SELL",
                    "reason": f"STEP_PROT({max_roi:.1%}->{current_roi:.1%})|T:{trigger:.2f}",
                }
            _t(
                "X10.step_protect",
                "pass",
                f"最高档 T={trigger:.2f} floor={floor:.2f} 已守住 cur={current_roi:.1%}",
            )
            break
    if not step_matched:
        _t("X10.step_protect", "skip", f"max_roi={max_roi:.1%} 未触达最低档")

    if max_roi >= cfg.FLASH_PROTECT_TRIGGER:
        if current_roi <= cfg.FLASH_PROTECT_EXIT:
            _t(
                "X10.flash_protect",
                "block",
                f"max={max_roi:.1%}≥trig={cfg.FLASH_PROTECT_TRIGGER:.0%} "
                f"cur={current_roi:.1%}≤exit={cfg.FLASH_PROTECT_EXIT:.0%}",
            )
            return {"action": "SELL", "reason": f"FLASH_PROT_L0({max_roi:.1%}->{current_roi:.1%})"}
        _t("X10.flash_protect", "pass", f"max={max_roi:.1%} cur={current_roi:.1%} 高于保本线")
    else:
        _t("X10.flash_protect", "skip", f"max_roi={max_roi:.1%} < trigger={cfg.FLASH_PROTECT_TRIGGER:.0%}")

    if max_roi > 0.03:
        logger.debug(
            f"⚖️ [Profit Guard Pass] {pos.get('symbol', 'UNK')} | "
            f"Max ROI: {max_roi:.1%} | Curr ROI: {current_roi:.1%}"
        )

    return None


def merged_trend_v0_option_stop_floors(cfg: "StrategyConfig") -> Tuple[float, float]:
    """
    Option ROI stop thresholds (negative fractions). More negative = looser stop.
    Returns (hard_sl, soft_sl) for checks: if roi <= hard_sl -> absolute; if roi <= soft_sl -> soft.
    """
    trend_abs = float(getattr(cfg, "TREND_EXIT_ABSOLUTE_STOP_LOSS", -0.11))
    v0_abs = float(getattr(cfg, "ABSOLUTE_STOP_LOSS", -0.15))
    hard_sl = min(trend_abs, v0_abs)
    trend_soft = float(getattr(cfg, "TREND_EXIT_STOP_LOSS", -0.10))
    v0_soft = float(getattr(cfg, "STOP_LOSS", -0.10))
    soft_sl = min(trend_soft, v0_soft)
    return hard_sl, soft_sl


__all__ = [
    "evaluate_profit_rails",
    "merged_trend_v0_option_stop_floors",
    "select_profit_ladder",
]

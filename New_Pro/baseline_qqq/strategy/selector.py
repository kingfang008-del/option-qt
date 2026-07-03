#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
策略入口 —— 仅 V0（StrategyCoreV0）。

TREND / V1 已归档至 archive/，不再通过 STRATEGY_CORE_VERSION 切换。
qqq_btc 路径下 OMS 会 patch decide_entry / check_exit 对齐 replay。
"""

from strategy.config0 import StrategyConfig
from strategy.core_v0 import StrategyCoreV0 as StrategyCore

ACTIVE_STRATEGY_CORE_VERSION = "V0"


def create_strategy(config: StrategyConfig = None) -> StrategyCore:
    return StrategyCore(config if config else StrategyConfig())


__all__ = [
    "ACTIVE_STRATEGY_CORE_VERSION",
    "StrategyCore",
    "StrategyConfig",
    "create_strategy",
]

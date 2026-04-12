#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Central strategy selector.

Engines should import StrategyCore/StrategyConfig from this module only.
Switch the active strategy via config.STRATEGY_CORE_VERSION or the
STRATEGY_CORE_VERSION environment variable.
"""

from config import STRATEGY_CORE_VERSION


def _normalize_version(version: str) -> str:
    normalized = str(version or "V1").strip().upper()
    if normalized not in {"V0", "V1"}:
        raise ValueError(
            f"Unsupported STRATEGY_CORE_VERSION={version!r}; expected 'V0' or 'V1'."
        )
    return normalized


ACTIVE_STRATEGY_CORE_VERSION = _normalize_version(STRATEGY_CORE_VERSION)

if ACTIVE_STRATEGY_CORE_VERSION == "V0":
    from strategy_core_v0 import StrategyCoreV0 as StrategyCore, StrategyConfig
elif ACTIVE_STRATEGY_CORE_VERSION == "V1":
    from strategy_core_v1 import StrategyCoreV1 as StrategyCore, StrategyConfig


def create_strategy(config: StrategyConfig = None) -> StrategyCore:
    return StrategyCore(config if config else StrategyConfig())


__all__ = [
    "ACTIVE_STRATEGY_CORE_VERSION",
    "StrategyCore",
    "StrategyConfig",
    "create_strategy",
]

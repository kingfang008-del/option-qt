"""Peak-armed profit floor for short-lived satellite option positions.

The arm is causal: once executable option MTM reaches ``arm_ret``, flatten on
the first later quote whose MTM is at or below ``floor_ret``.  TP/SL remain
the outer rails in the OMS caller.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ProfitProtectConfig:
    enabled: bool = False
    arm_ret: float = 0.08
    floor_ret: float = 0.03


def profit_protect_from_raw(raw: Any) -> ProfitProtectConfig:
    if not isinstance(raw, dict):
        return ProfitProtectConfig(enabled=False)
    arm = float(raw.get("arm_ret", 0.08) or 0.08)
    floor = float(raw.get("floor_ret", 0.03) or 0.0)
    valid = 0.0 <= floor < arm
    return ProfitProtectConfig(
        enabled=bool(raw.get("enabled", False)) and valid,
        arm_ret=arm,
        floor_ret=floor,
    )


def profit_protect_on_tick(
    *,
    cfg: ProfitProtectConfig,
    peak_mfe: float,
    opt_mtm: float,
) -> bool:
    """Return True when an armed position has fallen through its profit floor."""
    if not cfg.enabled:
        return False
    if not (peak_mfe == peak_mfe and opt_mtm == opt_mtm):  # NaN-safe
        return False
    return float(peak_mfe) >= float(cfg.arm_ret) and float(opt_mtm) <= float(
        cfg.floor_ret
    )

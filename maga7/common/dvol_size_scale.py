"""Liquidity soft size boost from causal session dollar-vol rank.

Does **not** change TopK seats — only multiplies ``size_frac`` after seating.
Default intent: boost only (min_scale>=1), never cut liquidity laggards.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from maga7.common.signals import _cs_dollar_vol_rank


@dataclass(frozen=True)
class DvolSizeScaleConfig:
    enabled: bool = False
    mode: str = "cs_rank"  # cs_rank only for now
    # 1-based cs dollar-vol rank → size multiplier
    scales: dict[int, float] | None = None
    default_scale: float = 1.0
    min_scale: float = 1.0
    max_scale: float = 1.25


def parse_dvol_size_scale(raw: Any) -> DvolSizeScaleConfig:
    if not isinstance(raw, dict):
        return DvolSizeScaleConfig(enabled=False)
    enabled = bool(raw.get("enabled", False))
    mode = str(raw.get("mode") or "cs_rank").strip().lower()
    scales_in = raw.get("scales") or {"1": 1.25, "2": 1.15}
    scales: dict[int, float] = {}
    if isinstance(scales_in, dict):
        for k, v in scales_in.items():
            try:
                scales[int(k)] = float(v)
            except (TypeError, ValueError):
                continue
    return DvolSizeScaleConfig(
        enabled=enabled,
        mode=mode,
        scales=scales or {1: 1.25, 2: 1.15},
        default_scale=float(raw.get("default_scale", 1.0) or 1.0),
        min_scale=float(raw.get("min_scale", 1.0) or 1.0),
        max_scale=float(raw.get("max_scale", 1.25) or 1.25),
    )


def resolve_dvol_size_scale(
    cfg: DvolSizeScaleConfig,
    *,
    stock_by: dict[str, pd.DataFrame],
    symbol: str,
    date: str,
    asof_ts: pd.Timestamp,
) -> tuple[float, int | None, float | None]:
    """Return ``(scale, cs_rank, session_dvol)``. Scale clamped to [min,max]."""
    if not cfg.enabled:
        return 1.0, None, None
    if cfg.mode not in {"cs_rank", "cs", "rank"}:
        return 1.0, None, None
    rank, dvol = _cs_dollar_vol_rank(
        stock_by, date=str(date), asof_ts=asof_ts, symbol=str(symbol)
    )
    scales = cfg.scales or {}
    if rank is not None and int(rank) in scales:
        scale = float(scales[int(rank)])
    else:
        scale = float(cfg.default_scale)
    scale = max(float(cfg.min_scale), min(float(cfg.max_scale), float(scale)))
    return scale, rank, dvol

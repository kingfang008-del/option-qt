#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""NVDA short-DTE (trading dte∈{0,1,2}) weekday-aware profile."""
from __future__ import annotations

from dataclasses import replace

from stock_options.common.short_dte_config import BASE_REPLAY, make_config

SYMBOL = "NVDA"

# dte0 state-gate was weak for NVDA; keep spreads conservative until per-weekday
# liquidity diagnostics pass.
REPLAY = replace(
    BASE_REPLAY,
    max_spread_pct=0.08,
    position_frac=0.20,
)

CONFIG = make_config(
    SYMBOL,
    stock_id=101,
    sector_id=10,
    replay=REPLAY,
)


def for_symbol() -> dict:
    return CONFIG.as_dict()

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""TSLA short-DTE (trading dte∈{0,1,2}) weekday-aware profile."""
from __future__ import annotations

from dataclasses import replace

from stock_options.common.short_dte_config import BASE_REPLAY, make_config

SYMBOL = "TSLA"

# dte0 state-gate showed stronger TSLA candidates; still calibrate per weekday.
REPLAY = replace(
    BASE_REPLAY,
    max_spread_pct=0.08,
    position_frac=0.25,
)

CONFIG = make_config(
    SYMBOL,
    stock_id=102,
    sector_id=20,
    replay=REPLAY,
)


def for_symbol() -> dict:
    return CONFIG.as_dict()

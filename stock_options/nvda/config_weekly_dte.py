#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""NVDA weekly-DTE option profile."""
from __future__ import annotations

from dataclasses import replace

from stock_options.common.weekly_dte_config import BASE_REPLAY, make_config

SYMBOL = "NVDA"

# NVDA is liquid, but single-stock weekly-DTE quotes still need per-bucket
# spread diagnostics before thresholds are tightened.
REPLAY = replace(
    BASE_REPLAY,
    max_spread_pct=0.10,
    position_frac=0.20,
)

CONFIG = make_config(
    SYMBOL,
    stock_id=101,
    sector_id=10,
    replay=REPLAY,
)


def for_symbol() -> dict:
    """Return a path/parameter bundle matching older qqq_btc config helpers."""
    return CONFIG.as_dict()


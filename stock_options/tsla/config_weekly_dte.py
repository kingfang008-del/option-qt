#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""TSLA weekly-DTE option profile."""
from __future__ import annotations

from dataclasses import replace

from stock_options.common.weekly_dte_config import BASE_REPLAY, make_config

SYMBOL = "TSLA"

# TSLA typically needs wider spread gates and smaller initial sizing than NVDA.
REPLAY = replace(
    BASE_REPLAY,
    max_spread_pct=0.12,
    position_frac=0.15,
)

CONFIG = make_config(
    SYMBOL,
    stock_id=102,
    sector_id=20,
    replay=REPLAY,
)


def for_symbol() -> dict:
    """Return a path/parameter bundle matching older qqq_btc config helpers."""
    return CONFIG.as_dict()


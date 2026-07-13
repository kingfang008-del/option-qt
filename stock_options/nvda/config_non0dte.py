#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""NVDA non-0DTE TFT profile."""
from __future__ import annotations

from dataclasses import replace
from stock_options.common.non0dte_config import BASE_REPLAY, make_config

CONFIG = make_config(
    "NVDA",
    stock_id=101,
    sector_id=10,
    replay=replace(BASE_REPLAY, max_spread_pct=0.10, position_frac=0.20),
)

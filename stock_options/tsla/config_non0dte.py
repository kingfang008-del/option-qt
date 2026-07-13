#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""TSLA non-0DTE TFT profile."""
from __future__ import annotations

from dataclasses import replace
from stock_options.common.non0dte_config import BASE_REPLAY, make_config

CONFIG = make_config(
    "TSLA",
    stock_id=102,
    sector_id=20,
    replay=replace(BASE_REPLAY, max_spread_pct=0.10, position_frac=0.25),
)

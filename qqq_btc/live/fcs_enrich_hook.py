#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FCS 历史 enrich 钩子 —— QQQ_BTC_LIVE 时在 compute 前补 time/trend 列。
"""
from __future__ import annotations

import os
from typing import Dict

import pandas as pd

from qqq_btc.live.fcs_adapter import enrich_fcs_bars


def is_qqq_btc_fcs_enrich_enabled() -> bool:
    return os.environ.get("QQQ_BTC_LIVE", "").strip().lower() in ("1", "true", "yes", "on")


def _frame_with_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    if "timestamp" in df.columns:
        out = df.copy()
    else:
        out = df.reset_index()
        if "index" in out.columns:
            out = out.rename(columns={"index": "timestamp"})
        elif out.columns[0] != "timestamp":
            out = out.rename(columns={out.columns[0]: "timestamp"})
    return out


def enrich_history_map(history_1min: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """对 1m 历史表补算 time/trend(因果,与离线一致)。"""
    if not is_qqq_btc_fcs_enrich_enabled():
        return history_1min
    out: Dict[str, pd.DataFrame] = {}
    for sym, df in history_1min.items():
        if df is None or df.empty:
            out[sym] = df
            continue
        base = _frame_with_timestamp(df)
        if "close" not in base.columns:
            out[sym] = df
            continue
        try:
            enriched = enrich_fcs_bars(base, price_col="close")
            if "timestamp" in enriched.columns:
                enriched = enriched.set_index("timestamp")
            out[sym] = enriched
        except Exception:
            out[sym] = df
    return out

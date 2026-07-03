#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FCS 1min bar → qqq_btc 特征补算(与离线 label_pipeline 同一份 common 代码)。

实盘/影子模式:在 FCS 输出的 1min DataFrame 上调用 enrich_fcs_bar,
保证 time/trend 特征与离线一致。
"""
from __future__ import annotations

import pandas as pd

from qqq_btc.common.time_features import add_time_features
from qqq_btc.common.trend_features import add_trend_features


def enrich_fcs_bars(df: pd.DataFrame, price_col: str = "close") -> pd.DataFrame:
    """对 FCS 聚合后的分钟 bar 表补算 time/trend 特征(因果、与离线同实现)。"""
    if df.empty:
        return df
    out = df.copy().sort_values("timestamp").reset_index(drop=True)
    out = add_time_features(out)
    if price_col not in out.columns:
        for c in ("price", "vwap"):
            if c in out.columns:
                price_col = c
                break
    if price_col in out.columns:
        out = add_trend_features(out, price_col=price_col)
    return out


def enrich_single_bar(history: pd.DataFrame, latest_row: dict, price_col: str = "close") -> pd.DataFrame:
    """
    流式:history 为截至上一 bar 的历史,latest_row 为当前 bar close 事件。
    返回含补算特征的完整 history(含 latest)。
    """
    frame = pd.concat([history, pd.DataFrame([latest_row])], ignore_index=True)
    return enrich_fcs_bars(frame, price_col=price_col)

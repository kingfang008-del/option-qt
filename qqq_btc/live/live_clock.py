#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把 FCS 分钟起点标签转换为离线/交易语义使用的结束标签。"""
from __future__ import annotations

import os
from typing import Any

import pandas as pd

from qqq_btc.common.time_features import session_minute


def live_label_shift_seconds() -> int:
    """FCS 的 alpha_label_ts 默认标记 [T,T+60) 的 T；交易时钟应使用 T+60。"""
    raw = os.environ.get("QQQ_BTC_LIVE_LABEL_SHIFT_SEC", "60")
    try:
        return int(raw)
    except (TypeError, ValueError):
        return 60


def live_end_label_ts(value: Any) -> pd.Timestamp:
    """FCS start-label → 离线 end-label 时间戳。"""
    if isinstance(value, (int, float)):
        ts = pd.Timestamp(float(value), unit="s", tz="UTC")
    else:
        ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts + pd.Timedelta(seconds=live_label_shift_seconds())


def live_session_bar(value: Any) -> int:
    return int(session_minute(pd.Series([live_end_label_ts(value)])).iloc[0])

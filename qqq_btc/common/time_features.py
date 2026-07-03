#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
日内时间特征 —— 0DTE 优化的第一缺口。

0DTE 期权的 theta/gamma 随日内时间剧烈非平稳:同样一段标的波动,
10:00 与 15:00 的期权 payoff 是两个分布。旧模型只有 day-of-week 嵌入,
看不到"现在几点/距到期还有多久",等于在非平稳分布上学平均值。

本模块输出 4 个确定性特征(纯 timestamp 函数,可在任何管线阶段补算):
  time_session_sin / time_session_cos   会话内位置的周期编码(有界 [-1,1])
  time_session_progress                 会话进度 0→1(09:30→16:00)
  time_to_expiry_norm                   距 0DTE 到期(16:00)剩余时间,归一化 0→1

这些特征天然有界,不做 rolling z-score(配置里 calc='raw' 使归一化脚本跳过)。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SESSION_TZ = "America/New_York"
SESSION_OPEN_MINUTE = 9 * 60 + 30    # 09:30
SESSION_CLOSE_MINUTE = 16 * 60       # 16:00 = 0DTE 到期
SESSION_MINUTES = SESSION_CLOSE_MINUTE - SESSION_OPEN_MINUTE  # 390

TIME_FEATURE_NAMES = [
    "time_session_sin",
    "time_session_cos",
    "time_session_progress",
    "time_to_expiry_norm",
]


def session_minute(ts: pd.Series) -> pd.Series:
    """时间戳 → 会话内分钟序号(09:30=0),自动转换纽约时区。"""
    t = pd.to_datetime(ts)
    if t.dt.tz is None:
        t = t.dt.tz_localize(SESSION_TZ, ambiguous="infer")
    else:
        t = t.dt.tz_convert(SESSION_TZ)
    minute_of_day = t.dt.hour * 60 + t.dt.minute
    return (minute_of_day - SESSION_OPEN_MINUTE).clip(lower=0, upper=SESSION_MINUTES)


def add_time_features(df: pd.DataFrame, ts_col: str = "timestamp") -> pd.DataFrame:
    """原地添加 4 个时间特征列并返回 df。要求 df 含时间戳列。"""
    sm = session_minute(df[ts_col]).astype(np.float64)
    progress = sm / SESSION_MINUTES
    angle = 2.0 * np.pi * progress
    df["time_session_sin"] = np.sin(angle)
    df["time_session_cos"] = np.cos(angle)
    df["time_session_progress"] = progress
    df["time_to_expiry_norm"] = 1.0 - progress
    return df

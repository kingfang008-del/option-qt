#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
VIX 波动 regime 特征 —— 对齐 V0 StrategyConfig0 REGIME_* 口径。

V0 用 VIXY 分钟价的 30min 内方向反转次数识别洗盘/chop;
qqq_btc 用特征列 vix_proxy_close(与 vix_level 同源)滚动计算。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

REGIME_FEATURE_NAMES = ("vix_reversal_count_30m",)


def count_reversals(prices: np.ndarray, threshold: float = 0.0015) -> int:
    """过去序列内 ≥threshold 的相邻涨跌方向翻转次数(V0 get_reversal_count 同语义)。"""
    if len(prices) < 2:
        return 0
    rev = 0
    last_dir = 0
    for i in range(1, len(prices)):
        prev, curr = float(prices[i - 1]), float(prices[i])
        if not (np.isfinite(prev) and np.isfinite(curr)) or prev == 0:
            continue
        diff_pct = (curr - prev) / prev
        if abs(diff_pct) < threshold:
            continue
        curr_dir = 1 if diff_pct > 0 else -1
        if last_dir != 0 and curr_dir != last_dir:
            rev += 1
        last_dir = curr_dir
    return rev


def rolling_reversal_count(
    series: pd.Series,
    window: int = 30,
    threshold: float = 0.0015,
) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    n = len(vals)
    out = np.full(n, np.nan)
    for i in range(n):
        lo = max(0, i - window + 1)
        chunk = vals[lo : i + 1]
        finite = chunk[np.isfinite(chunk)]
        if len(finite) >= 2:
            out[i] = float(count_reversals(finite, threshold))
    return pd.Series(out, index=series.index)


def add_vix_regime_features(
    df: pd.DataFrame,
    *,
    vix_col: str = "vix_proxy_close",
    window: int = 30,
    threshold: float = 0.0015,
) -> pd.DataFrame:
    """按交易日因果滚动,写入 vix_reversal_count_30m。"""
    if vix_col not in df.columns:
        return df
    out = df.copy()
    drop_day = False
    if "_day" not in out.columns:
        ts = pd.to_datetime(out["timestamp"])
        if ts.dt.tz is None:
            ts = ts.dt.tz_localize("America/New_York")
        else:
            ts = ts.dt.tz_convert("America/New_York")
        out["_day"] = ts.dt.date
        drop_day = True
    out["vix_reversal_count_30m"] = out.groupby("_day", sort=False)[vix_col].transform(
        lambda s: rolling_reversal_count(s, window=window, threshold=threshold)
    )
    if drop_day:
        out = out.drop(columns=["_day"])
    return out

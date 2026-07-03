#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
日内趋势结构特征 —— 交易员观察的工程化:
QQQ(指数 ETF)聚合掉个股特异噪声后,日内呈现清晰的波峰→波谷→波峰
波段,常常贴着一条"规则的拟合曲线"走。而模型序列窗口只有 30 分钟
(seq_len=30 @1min),小时级的波段结构完全在视野之外,必须以特征
形式显式注入。

特征(全部因果:只用当前 bar 及之前的数据):
  trend_fit_ret_30m    过去 30 bar 对数价格线性拟合的总变化量(拟合收益)
  trend_fit_r2_30m     拟合优度 R²,[0,1] —— "趋势有多像一条规则曲线"
  trend_fit_ret_120m   过去 120 bar 同上(波段级斜率)
  trend_fit_r2_120m    波段级拟合优度
  day_range_pos        当前价在当日已实现高低区间中的位置,[0,1]
  drawdown_from_day_high  距当日(已实现)最高点的回撤,<=0
  drawup_from_day_low     距当日(已实现)最低点的反弹,>=0

R²/range_pos 天然有界(calc='raw',rolling_norm 跳过);
fit_ret 与 drawdown/drawup 为收益量级的小实数,同样不做 z-score,
与 close_log_return 保持同一物理单位。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .time_features import SESSION_TZ

TREND_FEATURE_NAMES = [
    "trend_fit_ret_30m",
    "trend_fit_r2_30m",
    "trend_fit_ret_120m",
    "trend_fit_r2_120m",
    "day_range_pos",
    "drawdown_from_day_high",
    "drawup_from_day_low",
]


def rolling_linear_fit(y: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """
    滚动 OLS:对每个 t,用 [t-window+1, t] 的 y 对 0..window-1 做线性拟合。
    返回 (fit_ret, r2):
      fit_ret = slope * (window-1),拟合线在窗口内的总变化量
      r2      = corr(t, y)^2,拟合优度
    前 window-1 个位置为 NaN。向量化实现(sliding_window_view),因果安全。
    """
    n = len(y)
    fit_ret = np.full(n, np.nan)
    r2 = np.full(n, np.nan)
    if n < window:
        return fit_ret, r2

    from numpy.lib.stride_tricks import sliding_window_view

    win = sliding_window_view(y, window_shape=window)  # (n-window+1, window)
    t = np.arange(window, dtype=np.float64)
    t_mean = t.mean()
    t_var = ((t - t_mean) ** 2).sum()

    y_mean = win.mean(axis=1)
    cov = ((win - y_mean[:, None]) * (t - t_mean)).sum(axis=1)
    slope = cov / t_var

    y_var = ((win - y_mean[:, None]) ** 2).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        r2_win = np.where(y_var > 1e-18, cov**2 / (t_var * y_var), 0.0)

    fit_ret[window - 1:] = slope * (window - 1)
    r2[window - 1:] = np.clip(r2_win, 0.0, 1.0)
    return fit_ret, r2


def add_trend_features(
    df: pd.DataFrame,
    price_col: str = "close",
    ts_col: str = "timestamp",
    short_window: int = 30,
    long_window: int = 120,
) -> pd.DataFrame:
    """原地添加趋势结构特征并返回 df。要求按时间升序。"""
    px = pd.to_numeric(df[price_col], errors="coerce").astype(np.float64)
    log_px = np.log(px.replace(0, np.nan))

    fit_s, r2_s = rolling_linear_fit(log_px.to_numpy(), short_window)
    fit_l, r2_l = rolling_linear_fit(log_px.to_numpy(), long_window)
    df["trend_fit_ret_30m"] = np.nan_to_num(fit_s, nan=0.0)
    df["trend_fit_r2_30m"] = np.nan_to_num(r2_s, nan=0.0)
    df["trend_fit_ret_120m"] = np.nan_to_num(fit_l, nan=0.0)
    df["trend_fit_r2_120m"] = np.nan_to_num(r2_l, nan=0.0)

    # 按交易日分组的已实现高低点(cummax/cummin 只看过去,因果安全)
    t = pd.to_datetime(df[ts_col])
    if t.dt.tz is None:
        t = t.dt.tz_localize(SESSION_TZ, ambiguous="infer")
    else:
        t = t.dt.tz_convert(SESSION_TZ)
    day_key = t.dt.date

    high_col = "high" if "high" in df.columns else price_col
    low_col = "low" if "low" in df.columns else price_col
    day_high = pd.to_numeric(df[high_col], errors="coerce").groupby(day_key).cummax()
    day_low = pd.to_numeric(df[low_col], errors="coerce").groupby(day_key).cummin()

    rng = (day_high - day_low).replace(0, np.nan)
    df["day_range_pos"] = ((px - day_low) / rng).clip(0.0, 1.0).fillna(0.5)
    df["drawdown_from_day_high"] = (px / day_high.replace(0, np.nan) - 1.0).fillna(0.0)
    df["drawup_from_day_low"] = (px / day_low.replace(0, np.nan) - 1.0).fillna(0.0)
    return df

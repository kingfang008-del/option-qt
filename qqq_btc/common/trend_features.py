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
  spot_range_30m       过去 30 bar 现货振幅 (high-low)/close,震荡幅度
  trend_strength_30m   |trend_fit_ret_30m| * trend_fit_r2_30m,方向确信度
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
    "spot_range_30m",
    "trend_strength_30m",
    "day_range_pos",
    "drawdown_from_day_high",
    "drawup_from_day_low",
]

# 开盘 30 分钟(09:30–10:00, bar 0–29)形态特征;10:00 后冻结当日快照(因果安全)
OPEN30_FEATURE_NAMES = [
    "open30_ret",
    "open30_max_ret",
    "open30_peak_dd",
    "open30_reversal",
    "open30_range_pos",
    "bars_since_open30_high_norm",
]

OPEN30_BARS = 30
OPEN30_MID_BAR = 14  # 前 15 bar vs 后 15 bar 分段


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

    high_col = "high" if "high" in df.columns else price_col
    low_col = "low" if "low" in df.columns else price_col
    hi_s = pd.to_numeric(df[high_col], errors="coerce").rolling(short_window).max()
    lo_s = pd.to_numeric(df[low_col], errors="coerce").rolling(short_window).min()
    df["spot_range_30m"] = ((hi_s - lo_s) / px.replace(0, np.nan)).fillna(0.0).clip(lower=0.0)
    df["trend_strength_30m"] = (
        np.abs(df["trend_fit_ret_30m"]) * df["trend_fit_r2_30m"]
    ).astype(np.float64)

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


def add_spot_day_ret(
    df: pd.DataFrame,
    *,
    price_col: str = "close",
    day_col: str = "_day",
) -> pd.DataFrame:
    """当日开盘至当前 bar 的现货收益(因果,按交易日首根 close 为基准)。"""
    if price_col not in df.columns:
        return df
    out = df.copy()
    px = pd.to_numeric(out[price_col], errors="coerce")
    if day_col not in out.columns:
        t = pd.to_datetime(out["timestamp"])
        if t.dt.tz is None:
            t = t.dt.tz_localize(SESSION_TZ, ambiguous="infer")
        else:
            t = t.dt.tz_convert(SESSION_TZ)
        out[day_col] = t.dt.date
    day_open = px.groupby(out[day_col], sort=False).transform("first")
    out["spot_day_ret"] = (px / day_open.replace(0, np.nan) - 1.0).fillna(0.0)
    return out


def _open30_snapshot(px: np.ndarray, open_px: float, end: int) -> dict[str, float]:
    """bar 0..end 的开盘窗快照(end 通常为 29)。"""
    if end < 0 or not np.isfinite(open_px) or open_px <= 0:
        return {
            "open30_ret": 0.0,
            "open30_max_ret": 0.0,
            "open30_peak_dd": 0.0,
            "open30_reversal": 0.0,
            "open30_range_pos": 0.5,
            "bars_since_open30_high_norm": 0.0,
        }
    w = px[: end + 1]
    w = w[np.isfinite(w)]
    if w.size == 0:
        return {
            "open30_ret": 0.0,
            "open30_max_ret": 0.0,
            "open30_peak_dd": 0.0,
            "open30_reversal": 0.0,
            "open30_range_pos": 0.5,
            "bars_since_open30_high_norm": 0.0,
        }
    cur = float(w[-1])
    hi = float(np.max(w))
    lo = float(np.min(w))
    hi_idx = int(np.argmax(w))
    ret = cur / open_px - 1.0
    max_ret = hi / open_px - 1.0
    peak_dd = cur / hi - 1.0 if hi > 0 else 0.0
    mid_i = min(OPEN30_MID_BAR, len(w) - 1)
    mid_px = float(w[mid_i])
    first15 = mid_px / open_px - 1.0
    last15 = cur / mid_px - 1.0 if mid_px > 0 else 0.0
    reversal = first15 - last15
    rng = hi - lo
    range_pos = (cur - lo) / rng if rng > 1e-12 else 0.5
    bars_since = (len(w) - 1 - hi_idx) / float(OPEN30_BARS)
    return {
        "open30_ret": ret,
        "open30_max_ret": max_ret,
        "open30_peak_dd": peak_dd,
        "open30_reversal": reversal,
        "open30_range_pos": float(np.clip(range_pos, 0.0, 1.0)),
        "bars_since_open30_high_norm": float(np.clip(bars_since, 0.0, 1.0)),
    }


def add_open30_features(
    df: pd.DataFrame,
    price_col: str = "close",
    ts_col: str = "timestamp",
    open30_bars: int = OPEN30_BARS,
) -> pd.DataFrame:
    """
    开盘 30 分钟形态特征(因果):
      bar < 30: 用 0..t 的滚动开盘窗;
      bar >= 30: 冻结 bar 29 的快照,全日 carry-forward。
    """
    t = pd.to_datetime(df[ts_col])
    if t.dt.tz is None:
        t = t.dt.tz_localize(SESSION_TZ, ambiguous="infer")
    else:
        t = t.dt.tz_convert(SESSION_TZ)
    day_key = t.dt.date
    px = pd.to_numeric(df[price_col], errors="coerce").astype(np.float64)

    out = {k: np.zeros(len(df), dtype=np.float64) for k in OPEN30_FEATURE_NAMES}
    for _, g in df.groupby(day_key, sort=False):
        idx = g.index.to_numpy()
        loc = df.index.get_indexer(idx)
        p = px.loc[idx].to_numpy()
        if p.size == 0:
            continue
        open_px = float(p[0]) if np.isfinite(p[0]) and p[0] > 0 else np.nan
        snap_end = min(open30_bars, len(p)) - 1
        frozen = _open30_snapshot(p, open_px, snap_end)
        for i in range(len(p)):
            live = _open30_snapshot(p, open_px, i) if i < open30_bars else frozen
            for k, v in live.items():
                out[k][loc[i]] = v

    for k, v in out.items():
        df[k] = np.nan_to_num(v, nan=0.0)
    return df

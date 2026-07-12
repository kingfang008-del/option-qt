#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
price pandas 特征的唯一实现 —— 实时 FCS 与离线回放/对拍共用。

原则(honest sim):
- 公式以离线训练管线 preprocess/ask_bid/feature_merge_option_raw.py 为黄金基准;
- 实时 FCS (RealTimeFeatureEngine) 与 parity 工具都 import 这里,不允许两边各自维护公式;
- 输入只依赖因果可得的 OHLCV(+bar vwap),不读任何离线特征列。

vwap 口径说明(与 feature_merge 一致):
- vwap_log_return / return_divergence 用 bar 自带分钟 vwap(交易所口径, live 对应 IBKR wap);
- vwap_diff 用日内累计 VWAP cum(close*vol)/cum(vol)。
"""
from __future__ import annotations

import math
from typing import List

import numpy as np
import pandas as pd
import ta

EPSILON = 1e-6

PRICE_PANDAS_FEATURES = (
    'close_log_return',
    'open_log_return',
    'vwap_log_return',
    'return_divergence',
    'vwap_diff',
    'volume_ratio',
    'rsi',
    'k',
    'cci',
    'adx_smooth_10',
    'bb_width',
    'garman_klass_vol',
    'poc_deviation',
    'price_slope_norm_by_atr',
    'price_dist_from_ma_atr',
)

try:
    import numba

    @numba.njit(cache=True)
    def _numba_poc_loop(close_windows, volume_windows, bins):
        n_windows = len(close_windows)
        poc_values = np.full(n_windows, np.nan, dtype=np.float64)
        for i in range(n_windows):
            close_win = close_windows[i]
            volume_win = volume_windows[i]
            min_p, max_p = np.min(close_win), np.max(close_win)
            if min_p == max_p:
                poc_values[i] = min_p
                continue
            bin_edges = np.linspace(min_p, max_p, bins + 1)
            digitized = np.digitize(close_win, bin_edges) - 1
            digitized[digitized < 0] = 0
            digitized[digitized >= bins] = bins - 1
            volume_by_bin = np.bincount(digitized, weights=volume_win, minlength=bins)
            max_volume_bin_idx = np.argmax(volume_by_bin)
            poc_values[i] = (bin_edges[max_volume_bin_idx] + bin_edges[max_volume_bin_idx + 1]) / 2.0
        return poc_values

except ImportError:  # numba 缺失时退化为纯 numpy(慢但等价)
    def _numba_poc_loop(close_windows, volume_windows, bins):
        n_windows = len(close_windows)
        poc_values = np.full(n_windows, np.nan, dtype=np.float64)
        for i in range(n_windows):
            close_win = close_windows[i]
            volume_win = volume_windows[i]
            min_p, max_p = float(np.min(close_win)), float(np.max(close_win))
            if min_p == max_p:
                poc_values[i] = min_p
                continue
            bin_edges = np.linspace(min_p, max_p, bins + 1)
            digitized = np.clip(np.digitize(close_win, bin_edges) - 1, 0, bins - 1)
            volume_by_bin = np.bincount(digitized, weights=volume_win, minlength=bins)
            j = int(np.argmax(volume_by_bin))
            poc_values[i] = (bin_edges[j] + bin_edges[j + 1]) / 2.0
        return poc_values


def calculate_poc_deviation(df: pd.DataFrame, window: int = 50, bins: int = 50) -> pd.Series:
    """POC 偏离(对齐离线 50 窗口 + 50 价格桶)。"""
    if len(df) < window:
        return pd.Series(0.0, index=df.index)
    close_prices = df['close'].to_numpy().astype(np.float64)
    volumes = df['volume'].to_numpy().astype(np.float64)
    close_windows = np.lib.stride_tricks.sliding_window_view(close_prices, window_shape=window)
    volume_windows = np.lib.stride_tricks.sliding_window_view(volumes, window_shape=window)
    poc_values = _numba_poc_loop(close_windows, volume_windows, bins)
    result_series = pd.Series(np.nan, index=df.index, dtype=float)
    result_series.iloc[window - 1:] = poc_values
    result_series.ffill(inplace=True)
    result_series.fillna(0.0, inplace=True)
    return (df['close'] - result_series) / (result_series + 1e-9)


def compute_slope(series: pd.Series, window: int) -> pd.Series:
    """滚动 OLS 斜率(与离线一致)。"""
    if window >= len(series):
        return pd.Series(np.nan, index=series.index)
    x = np.arange(window)
    y_matrix = np.lib.stride_tricks.sliding_window_view(series.values, window)
    A = np.vstack([x, np.ones(len(x))]).T
    slopes = np.linalg.lstsq(A, y_matrix.T, rcond=None)[0][0]
    result = np.full(len(series), np.nan)
    result[window - 1:] = slopes
    return pd.Series(result, index=series.index)


def compute_price_pandas_features(df_in: pd.DataFrame, active_feats: List[str]) -> pd.DataFrame:
    """
    在分钟频率上按离线 feature_merge_option_raw.py 的公式计算 price 特征。
    输入 df 需含 open/high/low/close/volume,可选 vwap(bar 分钟 vwap)。
    """
    if df_in.empty:
        return df_in

    df = df_in.copy()

    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype(np.float64)

    # [对齐离线训练] 遇 0 视为缺失并 ffill
    close_ff = df['close'].replace(0, np.nan).ffill()
    open_ff = df['open'].replace(0, np.nan).ffill()

    prev_close = close_ff.shift(1).replace(0, np.nan)

    if 'close_log_return' in active_feats:
        df['close_log_return'] = np.log(close_ff / prev_close).fillna(0.0)

    if 'open_log_return' in active_feats:
        df['open_log_return'] = np.log(open_ff / prev_close).fillna(0.0)

    needs_vwap = any(f in active_feats for f in ['vwap_diff', 'vwap_log_return', 'return_divergence', 'poc_deviation'])
    if needs_vwap:
        vol_nonneg = df['volume'].clip(lower=0).fillna(0.0)
        if isinstance(df.index, pd.DatetimeIndex):
            session_key = pd.Series(df.index.date, index=df.index)
            cum_value = (close_ff * vol_nonneg).groupby(session_key).cumsum()
            cum_volume = vol_nonneg.groupby(session_key).cumsum()
        else:
            cum_value = (close_ff * vol_nonneg).cumsum()
            cum_volume = vol_nonneg.cumsum()
        vwap = cum_value / (cum_volume + 1e-9)
        vwap = vwap.replace([np.inf, -np.inf], np.nan).ffill().fillna(close_ff)

        if 'vwap_diff' in active_feats:
            df['vwap_diff'] = (df['close'] - vwap) / (vwap + 1e-9)

        # 对齐 feature_merge: vwap_log_return 用 bar 自带分钟 vwap;缺失时退回累计 vwap
        bar_vwap = None
        if 'vwap' in df.columns:
            bv = pd.to_numeric(df['vwap'], errors='coerce').replace(0, np.nan)
            if bv.notna().any():
                bar_vwap = bv.ffill().fillna(close_ff)
        vwap_for_ret = bar_vwap if bar_vwap is not None else vwap

        if 'vwap_log_return' in active_feats:
            df['vwap_log_return'] = np.log(vwap_for_ret / prev_close).fillna(0.0)

        if 'return_divergence' in active_feats:
            c_log_ret = df['close_log_return'] if 'close_log_return' in df.columns else np.log(close_ff / prev_close).fillna(0.0)
            v_log_ret = df['vwap_log_return'] if 'vwap_log_return' in df.columns else np.log(vwap_for_ret / prev_close).fillna(0.0)
            df['return_divergence'] = c_log_ret - v_log_ret

        if 'poc_deviation' in active_feats:
            df['poc_deviation'] = calculate_poc_deviation(df)

    if 'volume_ratio' in active_feats:
        sma20_vol = df['volume'].rolling(20, min_periods=1).mean()
        df['volume_ratio'] = (df['volume'] / (sma20_vol + EPSILON)).fillna(1.0)

    if 'rsi' in active_feats:
        df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi().fillna(50.0)

    if 'k' in active_feats:
        df['k'] = ta.momentum.StochasticOscillator(
            high=df['high'], low=df['low'], close=df['close'], window=9, smooth_window=3
        ).stoch().fillna(50.0)

    if 'cci' in active_feats:
        if len(df) >= 20:
            cci_raw = ta.trend.CCIIndicator(
                high=df['high'], low=df['low'], close=df['close'], window=20
            ).cci()
            cci_raw.iloc[:19] = np.nan
            df['cci'] = cci_raw.ffill().fillna(0.0)
        else:
            df['cci'] = 0.0

    if 'adx_smooth_10' in active_feats:
        try:
            raw_adx = ta.trend.ADXIndicator(
                high=df['high'], low=df['low'], close=df['close'], window=14
            ).adx()
            # 对齐 feature_merge: adx ffill 后再 ewm(span=10),不用 fillna(20)
            adx_ff = raw_adx.ffill().fillna(0.0)
            df['adx_smooth_10'] = adx_ff.ewm(span=10).mean().fillna(0.0)
        except Exception:
            df['adx_smooth_10'] = 0.0

    if 'bb_width' in active_feats:
        bb = ta.volatility.BollingerBands(close=df['close'], window=20, window_dev=2)
        # 对齐 feature_merge: bollinger_wband() / close
        df['bb_width'] = (bb.bollinger_wband() / df['close']).ffill().fillna(0.0)

    if 'garman_klass_vol' in active_feats:
        log_hl = np.log((df['high'] + EPSILON) / (df['low'] + EPSILON))
        log_co = np.log((df['close'] + EPSILON) / (df['open'] + EPSILON))
        gk = 0.5 * log_hl ** 2 - (2 * math.log(2) - 1) * log_co ** 2
        df['garman_klass_vol'] = np.sqrt(gk.clip(lower=0)).rolling(20).mean().fillna(0.0)

    if 'price_slope_norm_by_atr' in active_feats or 'price_dist_from_ma_atr' in active_feats:
        if len(df) < 14:
            atr = pd.Series(0.0, index=df.index)
        else:
            try:
                atr = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()
            except Exception:
                atr = pd.Series(0.0, index=df.index)

        if 'price_dist_from_ma_atr' in active_feats:
            sma200 = df['close'].rolling(window=200, min_periods=50).mean()
            df['price_dist_from_ma_atr'] = ((df['close'] - sma200) / (atr + EPSILON)).fillna(0.0)

        if 'price_slope_norm_by_atr' in active_feats:
            slope = compute_slope(df['close'], window=10)
            df['price_slope_norm_by_atr'] = (slope / (atr + EPSILON)).fillna(0.0)

    return df

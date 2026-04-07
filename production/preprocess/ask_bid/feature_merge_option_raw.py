


import pandas as pd
import numpy as np
import logging
from ta.trend import MACD, ADXIndicator, CCIIndicator
from ta.volatility import BollingerBands
from ta.momentum import RSIIndicator, StochasticOscillator
import arch
from pathlib import Path
import json
import os
from functools import partial
from tqdm import tqdm
import concurrent.futures
import pytz
from datetime import time as dt_time  # 重命名避免冲突
from typing import Dict, List, Tuple, Callable
import numba
import sqlite3
import traceback
import holidays  # 新增：确保 holidays 库可用
import math
import pyarrow.parquet as pq
from pandas.tseries.offsets import WeekOfMonth
from pandas.tseries.offsets import BMonthEnd, BDay, MonthBegin
import ta
from numpy.lib.stride_tricks import sliding_window_view
 
# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# --- 全局配置 ---
# 输入目录：包含重采样后的数据
STOCK_RESAMPLED_DIR = Path.home() / "train_data/spnq_train_resampled"
# 输出目录：用于存放计算好的特征文件. 
OPTION_MONTHLY_DIR = Path.home() / "train_data/quote_options_monthly_iv"
AGG_OPTION_MONTHLY_DIR = Path.home() / "train_data/quote_options_bucketed_v7"

OUTPUT_FEATURES_DIR = Path.home() / "train_data/quote_features_raw"
# VIX数据路径模板 (基于 STOCK_RESAMPLED_DIR)
VIX_BASE_DIR = STOCK_RESAMPLED_DIR / "VIXY/regular/09:30-16:00"
VIX_PATH_TEMPLATE = str(VIX_BASE_DIR / "{res}" / "{year_month}.parquet")
# 配置文件路径
CONFIG_FILE =  Path.home() /"notebook/train/feature_all.json"  
 
 
 
# 是否覆盖已存在的特征文件
OVERWRITE_EXISTING = True
# 使用的CPU核心数
MAX_WORKERS = 30
DB_PATH = Path.home() / "notebook/stocks.db"



@numba.njit
def _numba_poc_loop(close_windows, volume_windows, bins):
    """
    A Numba-jitted function to compute POC for all windows efficiently.
    This function contains the performance-critical loop.
    """
    n_windows = len(close_windows)
    poc_values = np.full(n_windows, np.nan, dtype=np.float64)

    for i in range(n_windows):
        close_win = close_windows[i]
        volume_win = volume_windows[i]

        min_p, max_p = np.min(close_win), np.max(close_win)

        # Handle edge case where all prices in window are the same
        if min_p == max_p:
            poc_values[i] = min_p
            continue

        # Create bin edges for the current window
        bin_edges = np.linspace(min_p, max_p, bins + 1)
        
        # Digitize prices (assign each price to a bin index)
        # Bins are 1-based, so subtract 1 for 0-based indexing
        digitized = np.digitize(close_win, bin_edges) - 1
        
        # Ensure indices are within bounds [0, bins-1]
        digitized[digitized < 0] = 0
        digitized[digitized >= bins] = bins - 1
        
        # Use np.bincount for a highly optimized groupby-sum operation
        # It calculates the sum of 'volume_win' for each bin in 'digitized'
        volume_by_bin = np.bincount(digitized, weights=volume_win, minlength=bins)
        
        # Find the bin with the maximum volume
        max_volume_bin_idx = np.argmax(volume_by_bin)
        
        # Calculate the midpoint of the winning bin to get the POC
        poc = (bin_edges[max_volume_bin_idx] + bin_edges[max_volume_bin_idx + 1]) / 2.0
        poc_values[i] = poc
        
    return poc_values



class FeatureEngineer:
    """
    根据配置文件，为单一分辨率的金融时序数据高效、稳健地计算特征。
    采用混合标准化策略（Robust -> Clip -> MinMax）以获得最佳性能。
    """
    def __init__(self, config ):
        self.config = config
        
         # 预先定义session的边界和标签，避免在函数内重复创建
        self.ny_tz = pytz.timezone('America/New_York')
        self.session_bins = [
            dt_time(0, 0, 0),          # 边界 1: 一天开始
            dt_time(4, 0, 0),          # 边界 2: 盘前时段1开始
            dt_time(7, 0, 0),          # 边界 3: 盘前时段2开始
            dt_time(9, 30, 0),         # 边界 4: 常规交易开始
            dt_time(16, 0, 0),         # 边界 5: 盘后时段1开始
            dt_time(18, 0, 0),         # 边界 6: 盘后时段2开始
            dt_time(20, 0, 0),         # 边界 7: 盘后时段2结束
            dt_time(23, 59, 59, 999999) # 边界 8: 一天结束
        ]

        # 定义7个标签，分别对应由8个边界点创建的7个区间
        self.session_labels = [
            -1,  # 区间1: [00:00, 04:00) -> regular (默认)
            0,  # 区间2: [04:00, 07:00) -> pre_market_0400_0700
            1,  # 区间3: [07:00, 09:30) -> pre_market_0700_0930
            2,  # 区间4: [09:30, 16:00) -> regular
            3,  # 区间5: [16:00, 18:00) -> after_hours_1600_1800
            4,  # 区间6: [18:00, 20:00] -> after_hours_1800_2000
            -1   # 区间7: (20:00, 23:59:59] -> regular (默认)
        ]
        self.session_label_map = {"pre_market_0400_0700": 0, "pre_market_0700_0930": 1, "regular": 2, 
                                  "after_hours_1600_1800": 3, "after_hours_1800_2000": 4}
 
    def _robust_scaler(self, series: pd.Series):
        # --- 加固点 ---
        if series.isnull().all():
            return pd.Series(np.zeros(len(series)), index=series.index)
        median = series.median()
        q25 = series.quantile(0.25)
        q75 = series.quantile(0.75)
        iqr = q75 - q25
        if pd.notna(iqr) and iqr > 1e-9:
            return (series - median) / iqr
        return series - median

    def _min_max_scaler_duplicate(self, series: pd.Series):
        # --- 加固点 ---
        if series.isnull().all():
            return pd.Series(np.zeros(len(series)), index=series.index)
        min_val = series.min()
        max_val = series.max()
        # 确保min_val和max_val都不是NaN
        if pd.notna(min_val) and pd.notna(max_val) and (max_val - min_val > 1e-9):
            return (series - min_val) / (max_val - min_val)
        return pd.Series(np.zeros(len(series)), index=series.index)
    
    def _min_max_scaler(self, series: pd.Series):
        """
        【优化版】Min-Max Scaler，能稳健处理低方差或常数序列。
        """
        if series.isnull().all():
            return pd.Series(np.zeros(len(series)), index=series.index)
        
        min_val = series.min()
        max_val = series.max()
        
        # 检查min和max是否有效
        if not pd.notna(min_val) or not pd.notna(max_val):
            # 如果无法确定边界，填充0是合理的备用方案
            return pd.Series(np.zeros(len(series)), index=series.index)
    
        range_val = max_val - min_val
        if range_val > 1e-9:
            # 正常情况：进行缩放
            return (series - min_val) / range_val
        else:
            # 方差极低或为0的情况：
            # 由于 robust_scaler 已经将数据中心化到0附近，
            # 此时返回一个全0序列是逻辑正确的，代表该特征在此期间无变化。
            return pd.Series(np.zeros(len(series)), index=series.index)

    def _compute_slope(self, series: pd.Series, window: int):
        if window >= len(series):
            return pd.Series(np.nan, index=series.index)
        x = np.arange(window)
        y_matrix = np.lib.stride_tricks.sliding_window_view(series.values, window)
        A = np.vstack([x, np.ones(len(x))]).T
        slopes = np.linalg.lstsq(A, y_matrix.T, rcond=None)[0][0]
        result = np.full(len(series), np.nan)
        result[window - 1:] = slopes
        return pd.Series(result, index=series.index)
        
    def _add_vix_level(self, df: pd.DataFrame):
        """
        【日内优化版】计算VIX Level特征，调整参数以更好地捕捉日内波动。
        short_span=9, long_span=21, rolling_std window=21, scale=2.0, min_periods优化。
        """
        # 1. 输入检查
        if df.empty or 'vix_proxy_close' not in df.columns or df['vix_proxy_close'].isnull().all():
            df['vix_level'] = 0.0
            return df
    
        try:
            # 2. 确保时间戳索引并预热
            df_combined = df.copy()
            if 'timestamp' in df_combined.columns:
                df_combined['timestamp'] = pd.to_datetime(df_combined['timestamp'])
                df_combined.set_index('timestamp', inplace=True)
                
                # 加载上个月数据以预热
                current_min_date = df_combined.index.min()
                prev_month = (current_min_date - pd.DateOffset(months=1)).strftime('%Y-%m')
                prev_vix_path = Path(VIX_PATH_TEMPLATE.format(year_month=prev_month))
                
                if prev_vix_path.exists():
                    df_prev = pd.read_parquet(prev_vix_path)
                    if not df_prev.empty:
                        df_prev['timestamp'] = pd.to_datetime(df_prev['timestamp'])
                        df_prev.set_index('timestamp', inplace=True)
                        df_prev.rename(columns={'close': 'vix_proxy_close'}, inplace=True)
                        # 取足够预热行（long_span * 2 + buffer）
                        warmup_rows = 50  # 调整为较小值，因为long_span=21
                        df_warmup = df_prev.tail(warmup_rows)
                        df_combined = pd.concat([df_warmup, df_combined])
    
            # 3. Contango校正（VIXY-specific，2025-08-08: VIX≈16.77, VIXY≈42.18）
            contango_factor = 2.52  # VIXY / VIX ≈ 42.18 / 16.77，含短期期货溢价
            df_combined['adjusted_proxy'] = df_combined['vix_proxy_close'] / contango_factor
    
            # 4. 计算EMA（基于价格，优化为日内敏感参数）
            short_span = 9   # ~9分钟，快速捕捉日内spikes
            long_span = 21   # ~21分钟，短期基线
            short_ma = df_combined['adjusted_proxy'].ewm(span=short_span, adjust=True, min_periods=1).mean()
            long_ma = df_combined['adjusted_proxy'].ewm(span=long_span, adjust=True, min_periods=5).mean()  # min_periods=5加速启动
    
            # 5. 计算Z-score（基于EMA差值的滚动标准差，window=21匹配long_span）
            ema_diff = short_ma - long_ma
            rolling_std = ema_diff.rolling(window=long_span, min_periods=short_span).std()
            epsilon = 1e-9
            scale = 2.0  # 放大信号，增强恐慌期差异
            vix_level_feature = scale * (ema_diff / (rolling_std + epsilon))
    
            # 6. 切回当前月份数据
            df['vix_level'] = vix_level_feature.loc[df.index]
    
            # 7. 插值和填充（确保无NaN，仅允许前向填充，严禁未来函数）
            df['vix_level'] = df['vix_level'].ffill()
            df['vix_level'].fillna(0.0, inplace=True)
    
        except Exception as e:
            logging.error(f"计算 VIX level 时出错: {e}")
            df['vix_level'] = 0.0
    
        return df
    
    def calculate_poc_vectorized(self, df: pd.DataFrame, window: int = 50, bins: int = 50) -> pd.Series:
        """
        Calculates Point of Control (POC) using a highly optimized and 
        vectorized approach with NumPy and Numba.
        """
        # Ensure we have enough data
        if len(df) < window:
            return pd.Series(np.nan, index=df.index)

        # Get the underlying NumPy arrays for performance
        close_prices = df['close'].to_numpy()
        volumes = df['volume'].to_numpy()

        # 1. Create rolling window views without copying data
        close_windows = np.lib.stride_tricks.sliding_window_view(close_prices, window_shape=window)
        volume_windows = np.lib.stride_tricks.sliding_window_view(volumes, window_shape=window)

        # 2. Call the fast, JIT-compiled Numba function
        poc_values = _numba_poc_loop(close_windows, volume_windows, bins)

        # 3. Format the result as a pandas Series
        # The result corresponds to the *end* of each window, so we pad the start with NaNs
        result_series = pd.Series(np.nan, index=df.index, dtype=float)
        result_series.iloc[window - 1:] = poc_values
        
        # Forward-fill to handle initial NaNs, then back-fill if any remain at the start
        result_series.ffill(inplace=True)
        result_series.fillna(0.0, inplace=True)
        
        # Final calculation of deviation from POC
        return (df['close'] - result_series) / (result_series + 1e-9)

    def calculate_vol_acceleration(self, df: pd.DataFrame, feature_col: str = 'garch_vol') -> Tuple[pd.Series, pd.Series]:
        """计算特征的1阶和2阶导数（速度和加速度）"""
        try:

           velocity = df[feature_col].diff()
           acceleration = velocity.diff()
        except Exception as e:
            logging.error(f"calculate_vol_acceleration数据 出错: {e}")
        return velocity, acceleration
    
    def calculate_divergence(self, df: pd.DataFrame, price_col: str = 'high', indicator_col: str = 'rsi', window: int = 14) -> pd.Series:
        """计算看跌背离 (Bearish Divergence)"""
        # --- FIX START ---
        # Initialize to a default value to prevent UnboundLocalError
        divergence = pd.Series(0, index=df.index, dtype=int)
        # --- FIX END ---
        try:
            # 计算价格和指标的滚动最大值
            price_highs = df[price_col].rolling(window).max()
            indicator_highs = df[indicator_col].rolling(window).max()
            
            # 当价格创出新高，但指标没有创出新高时，标记为背离
            divergence_condition = (price_highs > price_highs.shift(1)) & (indicator_highs < indicator_highs.shift(1))
            divergence = divergence_condition.astype(int)

        except Exception as e:
            logging.error(f"calculate_divergence failed: {e}")
            # On error, the pre-defined default series of zeros will be returned.
             
        return divergence
    
    def calculate_vol_contraction(self, df: pd.DataFrame, window: int = 20, long_window: int = 100) -> pd.Series:
        """计算波动率收缩 (Squeeze)"""
        # --- FIX START ---
        # Initialize to a neutral default value (1.0 indicates no contraction)
        result = pd.Series(1.0, index=df.index, dtype=float)
        # --- FIX END ---
        try:
           # 使用布林带宽度作为波动率指标
           bb = BollingerBands(close=df['close'], window=window)
           bb_width = bb.bollinger_wband()
           # 计算长期窗口内的最小宽度
           min_width = bb_width.rolling(long_window).min()
           
           # --- FIX START ---
           # Add epsilon to prevent division by zero
           result = bb_width / (min_width + 1e-9)
           # --- FIX END ---

        except Exception as e:
            logging.error(f"calculate_vol_contraction failed: {e}")
            # On error, the pre-defined neutral series will be returned.

        return result
    
    def get_session_label(self, timestamp, config):
        ny_tz = pytz.timezone('America/New_York')
        
        # 1. 转换时间戳（不立即附加时区）
        if isinstance(timestamp, (int, float)):
            dt = pd.Timestamp.fromtimestamp(timestamp)
        else:
            dt = pd.Timestamp(timestamp)
        
        # 2. 转换为纽约时区（如果尚未设置时区）
        if dt.tz is None:
            dt = dt.tz_localize('UTC').tz_convert(ny_tz)
        else:
            dt = dt.tz_convert(ny_tz)
        
        # 3. 获取时间并分类
        t = dt.time()
        
        if dt_time(4, 0) <= t < dt_time(7, 0):
            return 0  # pre_market_0400_0700
        elif dt_time(7, 0) <= t < dt_time(9, 30):
            return 1  # pre_market_0700_0930
        elif dt_time(9, 30) <= t < dt_time(16, 0):
            return 2  # regular
        elif dt_time(16, 0) <= t < dt_time(18, 0):
            return 3  # after_hours_1600_1800
        elif dt_time(18, 0) <= t <= dt_time(20, 0):
            return 4  # after_hours_1800_2000
        return 2  # 默认 regular
    
    

    # @numba.jit(nopython=True)
    # def _compute_fixed_labels(self, close_np: np.ndarray, horizon: int, entry_threshold: float) -> np.ndarray:
    #     future_close_np = np.roll(close_np, -horizon)
    #     future_close_np[-horizon:] = np.nan
    #     returns_np = (future_close_np - close_np) / (close_np + 1e-9)
    #     labels_np = np.zeros(len(close_np), dtype=np.int32)
    #     labels_np[returns_np > entry_threshold] = 1
    #     labels_np[returns_np < -entry_threshold] = 2
    #     labels_np[np.isnan(returns_np)] = 0
    #     return labels_np
    
    # def _add_fixed_horizon_labels_vec(self, df: pd.DataFrame, horizon: int = 10, entry_threshold: float = 0.001) -> pd.DataFrame:
    #     close_np = df['close'].ffill().replace(0, np.nan).ffill().to_numpy()  # 预处理在 Python
    #     labels_np = self._compute_fixed_labels(close_np, horizon, entry_threshold)  # 只传入 NumPy 到 Numba
    #     df['label'] = labels_np
    #     return df


    def _add_fixed_horizon_labels(self, df: pd.DataFrame, horizon: int = 10, entry_threshold: float = 0.001) -> pd.DataFrame:
        # 使用 shift(-horizon) 来获取未来第 N 个时间点的收盘价
       future_close = df['close'].shift(-horizon)
       # 计算未来回报率
       returns = (future_close - df['close']) / df['close']
        # 创建 label 列，默认为 0 (持有)
       df['label'] = 0
        # 根据回报率阈值设置买入(1)和卖空(2)标签
       df.loc[returns > entry_threshold, 'label'] = 1
       df.loc[returns < -entry_threshold, 'label'] = 2
       df['label'] = df['label'].fillna(0)
       df['label'] = df['label'].astype(int)
       return df
    # 此方法应添加在 new_feature.py 的 FeatureEngineer 类里_add_fixed_horizon_labels
    # @numba.jit(nopython=True, optional=True)  # 可选：加速 2-5x
    # def _add_fixed_horizon_labels(self, df: pd.DataFrame, horizon: int = 10, entry_threshold: float = 0.001) -> pd.DataFrame:
    #     close_np = df['close'].ffill().replace(0, np.nan).ffill().to_numpy()
    #     future_close_np = np.roll(close_np, -horizon)  # NumPy shift 等效
    #     future_close_np[-horizon:] = np.nan  # 最后 horizon 为 NaN
    #     returns_np = (future_close_np - close_np) / (close_np + 1e-9)  # 避免除零
        
    #     labels_np = np.zeros(len(close_np), dtype=int)
    #     labels_np[returns_np > entry_threshold] = 1
    #     labels_np[returns_np < -entry_threshold] = 2
    #     labels_np[np.isnan(returns_np)] = 0  # 填充 NaN 为 0
        
    #     df['label'] = labels_np
    #     return df

    def _add_triple_barrier_labels(self, df: pd.DataFrame, horizon: int = 10, 
                               upper_threshold: float = 0.001, lower_threshold: float = -0.001,
                               vol_multiplier: float = 1.5) -> pd.DataFrame:
        if 'close' not in df.columns or df.empty:
            df['label'] = 0
            return df
    
        # 数据准备（向量）
        close = df['close'].ffill().replace(0, np.nan).ffill()
        returns = close.pct_change().fillna(0)
        returns_np = returns.to_numpy()
    
        if 'garch_vol' in df.columns:
            vol = df['garch_vol']
        else:
            vol = returns.ewm(span=20).std().fillna(0)
        vol_np = vol.to_numpy()
    
        # 障碍计算（向量）
        upper_np = np.full(len(df), upper_threshold) if upper_threshold != 0 else vol_np * vol_multiplier
        lower_np = np.full(len(df), lower_threshold) if lower_threshold != 0 else -vol_np * vol_multiplier
    
        # 滑动窗口：(N - horizon, horizon + 1) for cum_returns
        ret_windows = sliding_window_view(returns_np, window_shape=horizon + 1)
        cum_ret_windows = np.cumsum(ret_windows, axis=1)
    
        # 击中检查：广播 upper/lower 到窗口
        hit_upper = cum_ret_windows >= upper_np[:-horizon, np.newaxis]
        hit_lower = cum_ret_windows <= lower_np[:-horizon, np.newaxis]
    
        # argmax for first hit (if no hit, set to horizon + 1)
        hit_upper_idx = np.argmax(hit_upper, axis=1)
        hit_upper_idx[~np.any(hit_upper, axis=1)] = horizon + 1
        hit_lower_idx = np.argmax(hit_lower, axis=1)
        hit_lower_idx[~np.any(hit_lower, axis=1)] = horizon + 1
    
        # 决定 labels
        labels_np = np.zeros(len(df), dtype=int)
        window_labels = np.where(hit_upper_idx < hit_lower_idx, 1,
                                 np.where(hit_lower_idx < hit_upper_idx, 2, 0))
        labels_np[:-horizon] = window_labels  # 最后 horizon 为 0
    
        df['label'] = labels_np
        #logging.info(f"成功生成 triple barrier 标签。Horizon: {horizon}, Upper: {upper_threshold}, Lower: {lower_threshold}, Vol Multiplier: {vol_multiplier}")
        return df
        
    def _add_meta_labels(self, df: pd.DataFrame, horizon: int = 10, entry_threshold: float = 0.001) -> pd.DataFrame:
        # 生成 primary（复用向量版本）
        df = self._add_fixed_horizon_labels(df, horizon=horizon, entry_threshold=entry_threshold)
        primary_np = df['label'].to_numpy()  # 临时用 'label' 作为 primary
        df.drop(columns=['label'], inplace=True)  # 立即清理
    
        # 实际回报（向量）
        close_np = df['close'].to_numpy()
        future_close_np = np.roll(close_np, -horizon)
        future_close_np[-horizon:] = np.nan
        actual_returns_np = (future_close_np - close_np) / (close_np + 1e-9)
        actual_returns_np[np.isnan(actual_returns_np)] = 0
    
        # Meta 计算（向量）
        buy_correct = (primary_np == 1) & (actual_returns_np > 0)
        sell_correct = (primary_np == 2) & (actual_returns_np < 0)
        labels_meta_np = np.zeros(len(df), dtype=int)
        labels_meta_np[buy_correct | sell_correct] = 1
    
        df['label_meta'] = labels_meta_np
        #logging.info(f"成功生成 meta 标签。Horizon: {horizon}, Entry Threshold: {entry_threshold}")
        return df

 
    
 
    
    def compute_features_raw(self, df: pd.DataFrame, resolution: str):
        df = df.copy()

        # 🚀 [Surgery 15] 强制列名标准化，防止 'VWAP' 或 'vwap ' 导致判断失败
        df.columns = [str(c).lower().strip() for c in df.columns]

        # --- 保证 VWAP 存在 ---
        if 'vwap' in df.columns:
            df['vwap'] = df['vwap'].ffill()
        else:
            # 只有当原始数据真没有 vwap 时，才迫不得已用 close 估算
            # 打印当前列名以便诊断到底是谁冲突了或确实缺失
            logging.warning(f"数据缺少真实 'vwap' 列 (分辨率: {resolution})，将使用 close 估算。现有列: {df.columns.tolist()}")
            daily_groups = df.groupby(df['timestamp'].dt.date)
            df['vwap'] = (df['close'] * df['volume']).groupby(daily_groups.grouper).cumsum() / \
                         (df['volume'].groupby(daily_groups.grouper).cumsum() + 1e-9)
            df['vwap'] = df['vwap'].fillna(df['close']) # 防止除零开局

        # --- 关键修复：在一开始就将 timestamp 保存起来 ---
        # 这样可以确保它不参与任何数值计算，并在最后加回来
        if 'timestamp' not in df.columns:
            # 这是一个应急措施，正常情况下 df 应该总是有 timestamp 列
            logging.error(f"输入的数据帧缺少 'timestamp' 列！")
            return pd.DataFrame() # 返回空DF，避免后续错误
        
        timestamps = df['timestamp']

        df['resolution'] = resolution # 确保分辨率列存在
        
        # --- 【核心修正】: 极度稳健的数据准备阶段 ---
    
        # 1. 定义最基础的列，这些是模型的生命线
        base_price_cols = ['open', 'high', 'low', 'close', 'vwap']
        
        # 2. 保证 'close' 列存在且基本有效
        #    'close' 是最重要的回退值（fallback），我们优先处理它
        if 'close' in df.columns:
            # 使用 ffill() 确保 'close' 列没有 NaN，后续可以用它来填充其他列
            df['close'] = df['close'].ffill()

                       
            
        else:
            # 如果连 'close' 列都没有，这是严重的数据问题。
            # 我们创建一个全零的列来防止程序崩溃，并打印错误日志。
            df['close'] = 0.0
            logging.error(f"数据文件为 {resolution} 分辨率缺少 'close' 列。这是一个严重的数据质量问题。已用 0.0 填充。")
    
        # 3. 检查并创建所有缺失的 OHLC 列
        #    最合理的策略是用可用的 'close' 价格来填充缺失的 'open', 'high', 'low'
        for col in base_price_cols:
            if col not in df.columns:
                logging.warning(f"数据文件中缺少 '{col}' 列（分辨率: {resolution}）。将使用 'close' 列的值进行创建。")
                df[col] = df['close'] # 使用 'close' 作为最可靠的回退值
        
        # 4. 对价格和成交量进行区别填充 (现在更安全了)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
        # 再次填充，确保所有价格列都完整
        df[base_price_cols] = df[base_price_cols].ffill() 

        option_features = [f['name'] for f in self.config['features'] if f.get('calc') == 'options' or f.get('calc') == 'options_chain']
        existing_option_features = [f for f in option_features if f in df.columns]
        if existing_option_features:
            df[existing_option_features] = df[existing_option_features].ffill() 
        pd.set_option('future.no_silent_downcasting', True)
        df = df.fillna(0.0)

        # 对成交量用 0 填充
        if 'volume' not in df.columns:
            df['volume'] = 0.0 # 如果 volume 也没有，就创建它
        df['volume'] = df['volume'].fillna(0)
    
        # 对于其他可能全为NaN的列（如bid_volume），也用0填充
        pd.set_option('future.no_silent_downcasting', True)
        df = df.fillna(0.0)

       
        
        # # 阶段1: 数据准备
        # base_cols = ['open', 'high', 'low', 'close', 'volume']
        # df[base_cols] = df[base_cols].interpolate(method='linear', limit_direction='both').ffill()

        # 阶段2: 预计算基础指标
        #df = self._add_vix_level(df)
        macd_indicator = MACD(close=df['close'])
        df['macd'] = macd_indicator.macd()/ df['close']
        df['macd_signal'] = macd_indicator.macd_signal()/ df['close']
        df['macd_diff'] = macd_indicator.macd_diff()/ df['close']

        macd_ind = MACD(close=df['close'], window_slow=26, window_fast=12, window_sign=9)
        df['macd_ratio'] = macd_ind.macd() / df['close']         # ✅ 这是对的
        df['macd_diff_ratio'] = macd_ind.macd_diff() / df['close'] # ✅ 这是对的


        bb_indicator = BollingerBands(close=df['close'], window=20, window_dev=2)
        df['bb_upper'] = bb_indicator.bollinger_hband()
        df['bb_lower'] = bb_indicator.bollinger_lband()
        df['bb_width'] = bb_indicator.bollinger_wband()/ df['close']

        stoch_indicator = StochasticOscillator(high=df['high'], low=df['low'], close=df['close'], window=9, smooth_window=3)
        df['k'] = stoch_indicator.stoch()
        df['d'] = stoch_indicator.stoch_signal()
        
        adx_indicator = ADXIndicator(high=df['high'], low=df['low'], close=df['close'], window=14)
        df['adx'] = adx_indicator.adx()


        # 2. 【新增】整合计算ATR归一化的原始特征 (不再需要 add_and_normalize_atr_features)
        atr = self._calculate_atr(df, period=14)
        previous_close = df['close'].shift(1)
        df['gap_over_atr'] = (df['open'] - previous_close) / atr
        sma_200_feat = df['close'].rolling(window=200, min_periods=50).mean() 
        df['price_dist_from_ma_atr'] = (df['close'] - sma_200_feat) / atr
        bb_feat = BollingerBands(close=df['close'], window=20, window_dev=2)
        df['bb_width_over_atr'] = bb_feat.bollinger_wband() / atr
        slope_feat = self._compute_slope(df['close'], window=10)
        df['price_slope_norm_by_atr'] = slope_feat / atr
        df['body_size_over_atr'] = (df['close'] - df['open']).abs() / atr
        df['wick_size_over_atr'] = (df['high'] - df['low']) / atr
        df['upper_wick_over_atr'] = (df['high'] - df[['open', 'close']].max(axis=1)) / atr
        df['lower_wick_over_atr'] = (df[['open', 'close']].min(axis=1) - df['low']) / atr

        # --- 1. 计算对数回报率和对数成交量 (逻辑不变) ---
        prev_close = df['close'].shift(1).replace(0, np.nan)
        for col in ['open', 'high', 'low', 'close', 'vwap']:
            df[f'{col}_log_return'] = np.log(df[col] / prev_close)
        df['volume_log'] = np.log(df['volume'] + 1)
        
         
        # 3. 核心变异：微观动量背离
        df['return_divergence'] = df['close_log_return'] - df['vwap_log_return']
        # --- 【核心修正 1】: 计算完基础指标后，立即进行一次填充 ---
        # 这可以确保后续依赖这些指标的计算（如背离）有干净的输入
        cols_to_fill = ['macd',   'macd_diff', 'bb_upper', 'bb_lower', 'bb_width', 'k', 'd', 'adx' ]
        df[cols_to_fill] = df[cols_to_fill].ffill().fillna(0.0)
        # --- 修正结束 ---


        # 阶段3: 循环计算
        for feature in self.config['features']: 
            name = feature['name']
            calc = feature['calc']


            # --- [插入点] 处理自定义特征 ---
            
            # 1. 处理 trend_macro (宏观趋势偏离)
            if calc == 'custom_trend_macro':
                # 计算 30 周期均线
                sma_30 = df['close'].rolling(window=30).mean()
                # 计算偏离度: (Price - SMA) / Price
                # 归一化处理，反映当前价格相对于宏观趋势的位置
                df[name] = (df['close'] - sma_30) / (df['close'] + 1e-9)
                
            # 2. 处理 slope_long (长期斜率)
            elif calc == 'custom_slope_long':
                # 计算 30 周期线性回归斜率
                # 为了速度，我们可以用简化的动量代替，或者用 numpy 的 polyfit
                # 这里提供一个高性能的 rolling_slope 实现
                
                # 方法 A: 简单动量 (速度极快) -> (Price_t - Price_t-30) / Price_t-30
                # df[name] = df['close'].pct_change(30) 
                
                # 方法 B: 真实线性回归斜率 (更准确，推荐)
                # 使用 normalized price 来计算，使其具有普适性
                def calc_slope(y):
                    x = np.arange(len(y))
                    # 简单的线性回归斜率公式: Cov(x,y) / Var(x)
                    # 因为 x 是固定的 [0, 1, ... 29]，Var(x) 是常数
                    return np.polyfit(x, y, 1)[0]
                
                # 对归一化后的价格做回归 (除以均值，消除股价绝对值影响)
                norm_close = df['close'] / df['close'].rolling(30).mean()
                df[name] = norm_close.rolling(window=30).apply(calc_slope, raw=True)



            if name in df.columns or feature['calc'] == 'raw':
                continue
            try:
                if name == 'price_slope':
                    resolution = df['resolution'].iloc[0] if 'resolution' in df else '1min'
                    window_map = self.config.get('parameters', {}).get(resolution, {}).get('slope_window', 10) # 示例从配置获取
                    if not isinstance(window_map, int): # Fallback if config is different
                        window_map = {'5s': 12, '10s': 10, '15s': 8, '30s': 6, '1min': 4, '5min': 3}
                        window = window_map.get(resolution, 4)
                    else:
                        window = window_map
                    df['price_slope'] = self._compute_slope(df['close'], window)

                
                
                elif name == 'rsi':
                    df['rsi'] = RSIIndicator(close=df['close'], window=14).rsi()
                
                elif name == 'cci':
                    df['cci'] = CCIIndicator(high=df['high'], low=df['low'], close=df['close'], window=20).cci()

                elif name == 'vwap_diff':
                    # 使用 groupby().cumsum() 进行高效的每日累加
                    df['price_x_vol'] = df['close'] * df['volume']
                    daily_groups = df.groupby(df['timestamp'].dt.date)
                    cum_value = daily_groups['price_x_vol'].cumsum()
                    cum_volume = daily_groups['volume'].cumsum()
                    df['vwap'] = cum_value / (cum_volume + 1e-9)
                    
                    df['vwap_diff'] = (df['close'] - df['vwap'])/ (df['vwap'] + 1e-9)
                    df.drop(columns=['price_x_vol'], inplace=True) # 清理辅助列
                
               
                elif name == 'volume_ratio':
                     

                    df['volume_ratio'] = df['volume'] / (df['volume'].rolling(window=20).mean() + 1e-9)

                elif name == 'garch_vol':
                    # --- 性能优化点：用EWMA波动率替换GARCH拟合 ---
                    
                    # 1. 准备回报率数据 (与之前相同)
                    close_prices = df['close'].replace(0, np.nan).ffill()
                    returns = close_prices.pct_change().fillna(0.0)
                    returns.replace([np.inf, -np.inf], 0.0, inplace=True)
                    
                    # 2. 直接计算EWMA波动率 (速度极快)
                    # span=20 与 GARCH回退方案中的 window=20 概念上对应
                    span_size = 20 
                    df['garch_vol'] = returns.ewm(span=span_size).std()

                elif name == 'vol_roc':
                    # ... 在计算完 garch_vol 之后 ...
                    df['vol_roc'] = df['garch_vol'].pct_change(periods=5).fillna(0.0)
                     

                elif name == 'vix_absolute_regime':
                    vix_bins = [0, 20, 30, 1000] # VIXY价格通常比VIX指数高，边界可适当放宽
                    vix_labels = [0, 1, 2] # 0=低, 1=中, 2=高
                    df['vix_absolute_regime'] = pd.cut(df['vix_proxy_close'], bins=vix_bins, labels=vix_labels)
                
                # ... 其他自定义计算逻辑可以继续添加 ...
                # (sma_ratio_30, vol_band_20, adx_smooth_10, vp_corr_15)
                elif name == 'sma_200':
                    df['sma_200'] = df['close'].rolling(window=200).mean()
                elif name == 'sma_ratio_30':
                      df['sma_30'] = df['close'].rolling(window=30).mean()
                      df['sma_ratio_30'] = df['close'] /  (df['sma_30'] + 1e-9)
                elif name == 'vol_band_20':
                    df['vol_band_20'] = df['volume'].rolling(window=20).std()
                elif name == 'adx_smooth_10':
                    df['adx_smooth_10'] = df['adx'] if 'adx' in df else ta.trend.ADXIndicator(
                        high=df['high'], low=df['low'], close=df['close'], window=14
                    ).adx()
                    df['adx_smooth_10'] = df['adx_smooth_10'].ewm(span=10).mean()

                elif name == 'vp_corr_15':
                    # 假设需要额外的量价数据，占位为收盘价和成交量的相关性
                    df['vp_corr_15'] = df['close'].rolling(window=15).corr(df['volume'])
                # elif name == 'pre_auction_imbalance':
                #     if 'bid_volume' in df and 'ask_volume' in df:
                #         df['pre_auction_imbalance'] = (df['bid_volume'] - df['ask_volume']) / (df['bid_volume'] + df['ask_volume'] + 1e-6)
                #     else:
                #         df['pre_auction_imbalance'] = 0

                # overnight_gap 在日内交易中有用，但需谨慎使用， 优势：快速反映隔夜市场情绪，提供明确的支撑/阻力位。
                # elif name == 'overnight_gap':
                #     if 'previous_close' in df:
                #         df['overnight_gap'] = (df['open'] - df['previous_close']) / df['previous_close']
                #     else:
                #         df['overnight_gap'] = 0
                elif name == 'pre_relative_volume':  #相对成交量比
                    df['pre_relative_volume'] = df['volume'] / df['volume'].rolling(window=20).mean()
                # elif name == 'event_intensity':
                #     df['event_intensity'] = 0  # 占位，需外部事件数据

                elif name == 'poc_deviation': 
                    df['poc_deviation'] = self.calculate_poc_vectorized(df)
                elif name == 'vol_acceleration':
                    df['vol_velocity'], df['vol_acceleration'] = self.calculate_vol_acceleration(df)
                elif name == 'rsi_divergence': 
                    # 计算看跌背离
                    #df['rsi_divergence'] = self.calculate_divergence(df, price_col='high', indicator_col='rsi')
                    df['rsi_divergence'] = self.calculate_divergence(df, indicator_col='rsi')
                elif name == 'vol_contraction_ratio':
                    df['vol_contraction_ratio'] = self.calculate_vol_contraction(df)

                elif name == 'session':
                    # 向量化 session 计算，避免 apply
                    # 1. 一次性转换整个索引到纽约时区
                    #    假设原始索引是UTC或无时区，需先本地化
                    ts_series = pd.to_datetime(df['timestamp'])
                    if ts_series.dt.tz is None:
                        ny_timestamps = ts_series.dt.tz_localize('America/New_York', ambiguous='infer')
                    else:
                        ny_timestamps = ts_series.dt.tz_convert(self.ny_tz)
                    
                    # 2. 提取时间部分
                    ny_time_series = pd.Series(ny_timestamps.dt.time, index=df.index)
                    
                    # 3. 使用 pd.cut 高效地进行区间划分
                    df['session'] = pd.cut(ny_time_series, bins=self.session_bins, labels=self.session_labels, 
                                           right=False, ordered=False).astype(int)
                    
                # 新添加的领先指标
                elif name == 'rvi':
                    # Relative Volatility Index (RVI)
                    # 计算向上和向下的标准差
                    diff = df['close'].diff()
                    up_std = diff.where(diff > 0, 0).rolling(window=10).std()
                    down_std = diff.where(diff < 0, 0).abs().rolling(window=10).std()
                    # SMA 平滑
                    up_sma = up_std.rolling(window=14).mean()
                    down_sma = down_std.rolling(window=14).mean()
                    df['rvi'] = 100 * up_sma / (up_sma + down_sma + 1e-9)

                elif name == 'chaikin_vol':
                    # Chaikin Volatility
                    hl_range = df['high'] - df['low']
                    ema_hl = hl_range.ewm(span=10, adjust=False).mean()
                    ema_hl_shift = ema_hl.shift(10)
                    df['chaikin_vol'] = (ema_hl - ema_hl_shift) / (ema_hl_shift + 1e-9) * 100

                elif name == 'garman_klass_vol':
                    # Garman-Klass Volatility (滚动计算)
                    def gk_vol(row):
                        if row['open'] == 0:
                            return 0.0
                        term1 = 0.5 * np.log(row['high'] / row['low']) ** 2
                        term2 = (2 * np.log(2) - 1) * np.log(row['close'] / row['open']) ** 2
                        return np.sqrt(np.maximum(0, term1 - term2))
                    df['garman_klass_vol'] = df.apply(gk_vol, axis=1).rolling(window=20).mean()  # 滚动平均作为估计

                elif name == 'price_vol_ratio':
                    # Price Volatility Ratio: 当前范围 / 历史平均范围
                    current_range = df['high'] - df['low']
                    hist_avg_range = current_range.rolling(window=14).mean()
                    df['price_vol_ratio'] = current_range / (hist_avg_range + 1e-9)

                # elif name == 'order_imbalance':
                #     # Order Imbalance (假设有 bid_volume 和 ask_volume 列，否则默认0)
                #     if 'bid_volume' in df and 'ask_volume' in df:
                #         df['order_imbalance'] = (df['bid_volume'] - df['ask_volume']) / (df['bid_volume'] + df['ask_volume'] + 1e-9)
                #     else:
                #         df['order_imbalance'] = 0.0

                elif name == 'intraday_vol_forecast':
                    # Simplified Intraday Volume Forecast: 使用 EWMA 作为未来volume的代理预测
                    df['intraday_vol_forecast'] = df['volume'].ewm(span=20, adjust=False).mean()


            except Exception as e:
                logging.warning(f"特征 '{name}' 计算失败: {e}")
                df[name] = 0.0

        



        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
        df = df.infer_objects(copy=False)
        df.ffill(inplace=True)
        # df.bfill(inplace=True) # 严禁未来填充
        df.fillna(0.0, inplace=True) # 最终防线


        # --- 【核心修正 2】: 在标准化之前，进行一次全局的 NaN 和 inf 清理 ---
        # 这是在所有特征计算完毕后的最后一道防线
        final_feature_cols = [f['name'] for f in self.config['features']]
        existing_cols = [col for col in final_feature_cols if col in df.columns]
        
        # df[existing_cols] = df[existing_cols].replace([np.inf, -np.inf], np.nan)
        # df[existing_cols] = df[existing_cols].ffill().fillna(0.0)
        for col in existing_cols:
            if col in df.columns:
                df[col] = df[col].replace([np.inf, -np.inf], np.nan)
                df[col] = df[col].ffill().fillna(0.0)
        
         
         
       
        
        # --- 4. 最终列选择 ---
        final_feature_cols = [f['name'] for f in self.config['features']]
        
        # 2. 从 df 中只选择实际存在的列，避免因计算失败而报错
        existing_cols = [col for col in final_feature_cols  if col in df.columns]
        final_df = df[existing_cols]
        
        # 3. 将之前保存的 timestamp 作为第一列加回来
        final_df.insert(0, 'timestamp', timestamps)
         
        return final_df
    
    

    # feature_merge_option.py 中 FeatureEngineer 类的方法
    def _calculate_vix_level_logic(self, price_series: pd.Series) -> pd.Series:
        """
        【新增】纯计算逻辑函数：只负责数学计算，不进行任何文件操作。
        输入一个价格序列，输出vix_level序列。
        """
        # 1. Contango 校正
        contango_factor = 2.40  # 可根据需要调整
        adjusted_proxy = price_series / contango_factor
    
        # 2. 计算EMA
        short_span = 20
        long_span = 200
        short_ma = adjusted_proxy.ewm(span=short_span, adjust=True, min_periods=short_span).mean()
        long_ma = adjusted_proxy.ewm(span=long_span, adjust=True, min_periods=long_span).mean()
    
        # 3. Z-score 计算
        ema_diff = short_ma - long_ma
        rolling_std_of_diff = ema_diff.rolling(window=long_span, min_periods=short_span).std()
        epsilon = 1e-9
        vix_level_feature = ema_diff / (rolling_std_of_diff + epsilon)
        
        # 4. 填充并返回
        return vix_level_feature.fillna(0.0)
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """健壮的ATR计算"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        # 使用 amax 替代 max，兼容 pandas 2.x
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.amax(ranges.to_numpy(), axis=1)
        
        # 使用 ewm (指数移动平均) 代替 rolling mean，对近期波动更敏感
        atr = pd.Series(true_range).ewm(span=period, adjust=False, min_periods=period).mean()
        
        # 填充开头的NaN，并确保最小值不为0
        atr = atr.fillna(1e-9)
        atr[atr < 1e-9] = 1e-9
        return atr.set_axis(df.index)

    def add_and_normalize_atr_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        【独立更新版】为一个已有的DataFrame计算、添加并独立进行混合标准化的ATR特征。
        此方法是自洽的，不依赖外部的标准化流程。
        """
        #logging.info("Starting: Add and normalize ATR features...")
        df = df.copy() # 创建副本以避免修改原始传入的DataFrame
    
        # --- 阶段1: 计算原始的ATR归一化特征 ---
        
        # a. 计算基准ATR
        # ATR周期可以从config中读取，这里使用通用默认值14
        atr_period = self.config.get('parameters', {}).get('labeling', {}).get('atr_period', 14)
        atr = self._calculate_atr(df, period=atr_period)
    
        # b. 计算各个归一化特征
        previous_close = df['close'].shift(1)
        df['gap_over_atr'] = (df['open'] - previous_close) / atr
    
        sma_200 = df['close'].rolling(window=200, min_periods=50).mean()
        df['price_dist_from_ma_atr'] = (df['close'] - sma_200) / atr
    
        bb = BollingerBands(close=df['close'], window=20, window_dev=2)
        df['bb_width_over_atr'] = bb.bollinger_wband() / atr
    
        slope = self._compute_slope(df['close'], window=10)
        df['price_slope_norm_by_atr'] = slope / atr
        
        df['body_size_over_atr'] = (df['close'] - df['open']).abs() / atr
        df['wick_size_over_atr'] = (df['high'] - df['low']) / atr
        df['upper_wick_over_atr'] = (df['high'] - df[['open', 'close']].max(axis=1)) / atr
        df['lower_wick_over_atr'] = (df[['open', 'close']].min(axis=1) - df['low']) / atr
    
        # --- 阶段2: 仅对新增的ATR特征进行混合标准化 ---
        
        atr_norm_cols = [
            'gap_over_atr', 'price_dist_from_ma_atr', 'bb_width_over_atr',
            'price_slope_norm_by_atr', 'body_size_over_atr', 'wick_size_over_atr',
            'upper_wick_over_atr', 'lower_wick_over_atr'
        ]
        print(f"缺失值统计: {df[atr_norm_cols].isna().sum()}")
        df[atr_norm_cols] = df[atr_norm_cols].ffill().fillna(0)
        print("原始 / ATR describe:\n", df[atr_norm_cols].describe())
        #logging.info(f"Applying mixed scaling to {len(atr_norm_cols)} new features...")
        for col in atr_norm_cols:
            # 确保列存在
            if col not in df.columns:
                logging.warning(f"Column '{col}' not found for scaling, skipping.")
                continue
            
            # 步骤 i: Robust Scaler (处理异常值)
            df[col] = self._robust_scaler(df[col])
            
            # 步骤 ii: Clip (去除极端值)
            # 使用更稳健的分位数，例如0.5%和99.5%
            p005, p995 = df[col].quantile(0.005), df[col].quantile(0.995)
            # 只有在分位数有效时才进行裁剪
            if pd.notna(p005) and pd.notna(p995):
                df[col] = df[col].clip(p005, p995)
    
            # 步骤 iii: Min-Max Scaler (归一化到 [0, 1] 区间)
            df[col] = self._min_max_scaler(df[col])
    
        # --- 阶段3: 最终清理 ---
        
        # 填充在计算和标准化过程中可能产生的任何NaN值
        df[atr_norm_cols] = df[atr_norm_cols].fillna(0.0)
    
        #logging.info("Finished: Add and normalize ATR features." )
        return df
 
# NFP (非农) 发布日期列表 (通常是每月第一个星期五)
# 数据来源: U.S. Bureau of Labor Statistics (BLS)
nfp_release_dates = pd.to_datetime([
    "2021-01-08", "2021-02-05", "2021-03-05", "2021-04-02", "2021-05-07", "2021-06-04", 
    "2021-07-02", "2021-08-06", "2021-09-03", "2021-10-08", "2021-11-05", "2021-12-03",
    "2022-01-07", "2022-02-04", "2022-03-04", "2022-04-01", "2022-05-06", "2022-06-03", 
    "2022-07-08", "2022-08-05", "2022-09-02", "2022-10-07", "2022-11-04", "2022-12-02",
    "2023-01-06", "2023-02-03", "2023-03-10", "2023-04-07", "2023-05-05", "2023-06-02", 
    "2023-07-07", "2023-08-04", "2023-09-01", "2023-10-06", "2023-11-03", "2023-12-08",
    "2024-01-05", "2024-02-02", "2024-03-08", "2024-04-05", "2024-05-03", "2024-06-07", 
    "2024-07-05", "2024-08-02", "2024-09-06", "2024-10-04", "2024-11-01", "2024-12-06",
    "2025-01-03", "2025-02-07", "2025-03-07", "2025-04-04", "2025-05-02", "2025-06-06",
    "2025-07-03", "2025-08-01", "2025-09-05", "2025-10-03", "2025-11-07", "2025-12-05"
]).tz_localize('America/New_York')

# CPI (通胀) 发布日期列表 (通常在月中)
# 数据来源: U.S. Bureau of Labor Statistics (BLS)
cpi_release_dates = pd.to_datetime([
    "2021-01-13", "2021-02-10", "2021-03-10", "2021-04-13", "2021-05-12", "2021-06-10", 
    "2021-07-13", "2021-08-11", "2021-09-14", "2021-10-13", "2021-11-10", "2021-12-10",
    "2022-01-12", "2022-02-10", "2022-03-10", "2022-04-12", "2022-05-11", "2022-06-10", 
    "2022-07-13", "2022-08-10", "2022-09-13", "2022-10-13", "2022-11-10", "2022-12-13",
    "2023-01-12", "2023-02-14", "2023-03-14", "2023-04-12", "2023-05-10", "2023-06-13", 
    "2023-07-12", "2023-08-10", "2023-09-13", "2023-10-12", "2023-11-14", "2023-12-12",
    "2024-01-11", "2024-02-13", "2024-03-12", "2024-04-10", "2024-05-15", "2024-06-12", 
    "2024-07-11", "2024-08-14", "2024-09-11", "2024-10-10", "2024-11-13", "2024-12-11",
     "2025-01-14", "2025-02-13", "2025-03-13", "2025-04-10", "2025-05-13", "2025-06-11",
    "2025-07-11", "2025-08-13", "2025-09-11", "2025-10-10", "2025-11-13", "2025-12-11"
]).tz_localize('America/New_York')

fomc_dates_naive = [
    # --- 2021年 ---
    "2021-01-27", "2021-03-17", "2021-04-28", "2021-06-16", "2021-07-28", "2021-09-22", "2021-11-03", "2021-12-15",
    # --- 2022年 ---
    "2022-01-26", "2022-03-16", "2022-05-04", "2022-06-15", "2022-07-27", "2022-09-21", "2022-11-02", "2022-12-14",
    # --- 2023年 ---
    "2023-02-01", "2023-03-22", "2023-05-03", "2023-06-14", "2023-07-26", "2023-09-20", "2023-11-01", "2023-12-13",
    # --- 2024年 ---
    "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12", "2024-07-31", "2024-09-18", "2024-11-07", "2024-12-18",
    # --- 2025年 ---
    "2025-01-29", "2025-03-19", "2025-04-30", "2025-06-11", "2025-07-30", "2025-09-17", "2025-11-05", "2025-12-17"
]


# PPI (生产者价格指数) 发布日期
# 数据来源: U.S. Bureau of Labor Statistics (BLS)
ppi_release_dates = pd.to_datetime([
    "2021-01-15", "2021-02-12", "2021-03-12", "2021-04-19", "2021-05-13", "2021-06-15",
    "2021-07-14", "2021-08-12", "2021-09-10", "2021-10-14", "2021-11-12", "2021-12-14",
    "2022-01-13", "2022-02-15", "2022-03-15", "2022-04-13", "2022-05-13", "2022-06-14",
    "2022-07-14", "2022-08-11", "2022-09-14", "2022-10-12", "2022-11-15", "2022-12-19",
    "2023-01-18", "2023-02-15", "2023-03-15", "2023-04-13", "2023-05-12", "2023-06-14",
    "2023-07-13", "2023-08-11", "2023-09-14", "2023-10-13", "2023-11-15", "2023-12-14",
    "2024-01-12", "2024-02-16", "2024-03-14", "2024-04-11", "2024-05-14", "2024-06-13",
    "2024-07-12", "2024-08-15", "2024-09-13", "2024-10-15", "2024-11-14", "2024-12-13",
    "2025-01-15", "2025-02-14", "2025-03-14", "2025-04-15", "2025-05-14", "2025-06-13",
    "2025-07-15", "2025-08-14", "2025-09-12", "2025-10-15", "2025-11-14", "2025-12-12"
]).tz_localize('America/New_York')

# --- 【核心修正】: 为FOMC日期列表本地化纽约时区 ---
fomc_dates = pd.to_datetime(fomc_dates_naive).tz_localize('America/New_York')


def calculate_is_event_day(dates: pd.Series, event_dates: pd.DatetimeIndex, n_days_before=1, n_days_after=0) -> pd.Series:
    """
    通用函数：判断日期是否在某个事件窗口内。
    例如，发布日前一天和当天。
    """
    is_event = pd.Series(False, index=dates.index)
    # 为了性能，我们只比较日期部分
    normalized_dates = dates.dt.normalize()
    for event_date in event_dates:
        start_date = event_date.normalize() - pd.DateOffset(days=n_days_before)
        end_date = event_date.normalize() + pd.DateOffset(days=n_days_after)
        is_event |= (normalized_dates >= start_date) & (normalized_dates <= end_date)
    return is_event.astype(int)

def calculate_is_fomc_week(dates: pd.Series, fomc_dates: pd.DatetimeIndex, n_days_before=3, n_days_after=1) -> pd.Series:
    """
    【V2 - 已修复TypeError】判断日期是否在FOMC会议窗口内。
    """
    is_fomc = pd.Series(False, index=dates.index)
    # 两个操作数现在都有相同的时区，无需再对 dates 进行 normalize
    for fomc_date in fomc_dates:
        start_date = fomc_date - pd.DateOffset(days=n_days_before)
        end_date = fomc_date + pd.DateOffset(days=n_days_after)
        # 现在 dates 和 start/end_date 都是纽约时区，可以安全比较
        is_fomc |= (dates >= start_date) & (dates <= end_date)
    return is_fomc.astype(int)

def calculate_is_opex_week(dates: pd.Series) -> pd.Series:
    """判断日期是否在月度期权交割周（第三周的周一至周五）"""
    # WeekOfMonth(week=3, weekday=4) 表示第三个星期五 (Monday=0, Friday=4)
    third_fridays = dates.apply(lambda d: WeekOfMonth(week=3, weekday=4).rollforward(d))

    # 交割周的开始（周一）
    start_of_opex_week = third_fridays - pd.DateOffset(days=4)

    return dates.between(start_of_opex_week, third_fridays).astype(int)


def calculate_is_month_end(dates: pd.Series, n_days_before=2, n_days_after=2) -> pd.Series:
    """
    【V2 - 已修复TypeError】
    使用向量化操作，高效地判断日期是否在月末/月初的交易窗口内。
    """
    # 确保dates是datetime类型，并且为了比较，我们只关心日期部分
    normalized_dates = dates.dt.normalize()

    # --- 1. 计算月末窗口 ---
    # a. 找到每个日期对应的月末最后一天
    end_of_month = dates + pd.offsets.MonthEnd(0)
    # b. 计算窗口的开始日期（月末前n个工作日）
    end_window_start = end_of_month - BDay(n_days_before)
    
    # c. 判断日期是否落在月末窗口 [窗口开始日, 月末日]
    is_near_end = (normalized_dates >= end_window_start.dt.normalize()) & \
                  (normalized_dates <= end_of_month.dt.normalize())

    # --- 2. 计算月初窗口 ---
    # a. 找到每个日期对应的月初第一天
    start_of_month = dates + pd.offsets.MonthBegin(0) - pd.offsets.Day(1)
    # b. 计算窗口的结束日期（月初后n个工作日）
    #    注意：BDay(2)是从基准日期前进2个工作日，所以窗口是[月初, 月初+2BD]，共3天
    start_window_end = start_of_month + BDay(n_days_after)
    
    # c. 判断日期是否落在月初窗口 [月初日, 窗口结束日]
    is_near_start = (normalized_dates >= start_of_month.dt.normalize()) & \
                    (normalized_dates <= start_window_end.dt.normalize())
    
    # --- 3. 合并两个判断结果 ---
    return (is_near_end | is_near_start).astype(int)

def process_atr_features_file(file_path: Path, config: dict):
    """
    工作函数：加载单个文件，计算并添加ATR归一化特征，然后保存。
    """
    try:
        df = pd.read_parquet(file_path)
        if df.empty or 'close' not in df.columns:
            return f"[跳过] {file_path} 为空或缺少'close'列。"

        # 实例化 FeatureEngineer 来调用其计算方法
        feature_engineer = FeatureEngineer(config)
        
        # 调用新的特征计算方法
        df_updated = feature_engineer.add_and_normalize_atr_features(df)
        
        # 覆写保存文件
        df_updated.to_parquet(file_path, index=False, compression='zstd', compression_level=9)
        logging.info(f"Finished: Add and normalize ATR features. {file_path}" )
        return f"[成功] 更新 {file_path}"
    except Exception as e:
        import traceback
        return f"[错误] 更新 {file_path} 失败: {e}\n{traceback.format_exc()}"
 

def process_stock_month(symbol: str, year_month: str, config: dict ):
    """
    【核心重构】处理单一股票在某个月的所有数据，并生成所有分辨率的特征。
    【调试优化版】增加完整的错误堆栈跟踪，以便定位问题。
    """
    processed_count = 0
    skipped_count = 0
    try:
        # 1. 加载月度期权数据
        option_root = OPTION_MONTHLY_DIR / symbol
        agg_option_root = AGG_OPTION_MONTHLY_DIR / symbol
        std_path = agg_option_root / f"{year_month}.parquet"
        high_path = option_root / 'high_features' / f"{year_month}.parquet"
        
        option_df = None
        if std_path.exists():
            option_df = pd.read_parquet(std_path)
             
        vix_df = None

         #   处理股票和期权重名的数据 (逻辑不变)
        if option_df is not None and not option_df.empty:
            option_df['timestamp'] = pd.to_datetime(option_df['timestamp'])
            
            # --- 【核心修正】---
            # 在 merge_asof 之前，必须对右侧的 DataFrame 排序，并去除重复的时间戳。
            # 这可以保证 'timestamp' 键是单调递增的。
            option_df.sort_values('timestamp', inplace=True)
            option_df.drop_duplicates(subset=['timestamp'], keep='last', inplace=True)
            # --- 【修正结束】---

            # --- 【核心修正】---
            # 🚀 [Surgery 14] 扩展重命名列表，彻底防止 option 数据中的 OHLV/VWAP 污染 stock 主表
            # 在合并前，主动、明确地重命名 option_df 中与 stock_df 冲突的列
            # 这样可以完全避免 suffixes 参数可能带来的歧义和冲突
            rename_dict = {
                'open': 'open_option',
                'high': 'high_option',
                'low': 'low_option',
                'close': 'close_option',
                'vwap': 'vwap_option',
                'volume': 'volume_option',
                'transactions': 'transactions_option'
            }

            # 只重命名实际存在的列，避免因列不存在而报错
            cols_to_rename = {k: v for k, v in rename_dict.items() if k in option_df.columns}
            if cols_to_rename:
                option_df.rename(columns=cols_to_rename, inplace=True)
            # --- 【修正结束】---

         
        # 遍历所有时段和分辨率
        for session, time_ranges in config['resample_freq'].items():
          # 中间循环遍历 time_range 和对应的 resolutions 列表
            # e.g., time_range="09:30-16:00", resolutions=["1min", "5min"]
            for time_range, resolutions in time_ranges.items():
                # 内部循环遍历该时间范围下的每一个 resolution
                for res in resolutions:
                    # 新增: 加载月度 VIX 数据 (根据当前分辨率 res)
                    vix_df = None
                    vix_file_path = Path(VIX_PATH_TEMPLATE.format(res=res, year_month=year_month))
                    if vix_file_path.exists():
                        try:
                            vix_df = pd.read_parquet(vix_file_path)
                            if vix_df.empty:
                                vix_df = None
                        except Exception as e:
                            logging.warning(f"加载VIXY文件失败: {vix_file_path}, 错误: {e}")
                            vix_df = None

                    # 使用 session, time_range, res 构建新的文件路径
                    stock_file = STOCK_RESAMPLED_DIR / symbol / session / time_range / res / f"{year_month}.parquet"
                    
                    if not stock_file.exists():
                        # 不再静默跳过，增加计数
                        skipped_count += 1
                        continue

                    processed_count += 1
                    stock_df = pd.read_parquet(stock_file)
                    # 🚀 [Diagnostic] 检查 spnq_train_resampled 原始文件的列名
                    #logging.info(f"Loaded stock_df from {stock_file}. Columns: {stock_df.columns.tolist()}")
                    stock_df['timestamp'] = pd.to_datetime(stock_df['timestamp'])


                    # 合并股票和期权数据
                    if option_df is not None and not option_df.empty:
                        # 确保 timestamp 类型一致，并排序去重
                        option_df['timestamp'] = pd.to_datetime(option_df['timestamp'])
                        option_df.sort_values('timestamp', inplace=True)
                        option_df.drop_duplicates(subset=['timestamp'], keep='last', inplace=True)
                        
                        merged_df = pd.merge_asof(
                            stock_df.sort_values('timestamp'),
                            option_df,
                            on='timestamp',
                            direction='backward',
                        )
                    else:
                        merged_df = stock_df
    
                    
    
                    # 计算特征和标签
                    feature_engineer = FeatureEngineer(config)
                    merged_df = feature_engineer.compute_features_raw(merged_df, resolution=res)

                    # 合并 VIX 数据
                    if vix_df is not None:
                        vix_df['timestamp'] = pd.to_datetime(vix_df['timestamp'])
                        if vix_df['timestamp'].dt.tz is None:
                             vix_df['timestamp'] = vix_df['timestamp'].dt.tz_localize('America/New_York', ambiguous='infer')
                        
                        vix_df.sort_values('timestamp', inplace=True)
                        vix_df.drop_duplicates(subset=['timestamp'], keep='last', inplace=True)
                    
                        # 简洁版修复
                        vix_cols = ['vix_level', 'vix_proxy_close']
                        available_vix_cols = [col for col in vix_cols if col in vix_df.columns]
                        
                        # 删除可能存在的重复列
                        for col in available_vix_cols:
                            if col in merged_df.columns:
                                merged_df.drop(columns=[col], inplace=True)
                        
                        merged_df.sort_values('timestamp')
                        merged_df = pd.merge_asof(
                            merged_df,
                            vix_df[['timestamp'] + available_vix_cols],
                            on='timestamp',
                            direction='backward'
                        )
                        
                        # 填充NaN值
                        for col in available_vix_cols:
                            merged_df[col] = merged_df[col].ffill().fillna(0.0)
                    else:
                        merged_df['vix_level'] = 0.0
                        merged_df['vix_proxy_close'] = 0.0
    
                    # 保存
                    output_dir = OUTPUT_FEATURES_DIR / symbol / session / time_range / res

                    output_dir.mkdir(parents=True, exist_ok=True)
                    output_path = output_dir / f"{year_month}.parquet"
                    
                    if not output_path.exists() or OVERWRITE_EXISTING:
                        merged_df.to_parquet(output_path, index=False, compression='zstd', compression_level=9)
        
        # --- 3. 根据处理结果返回详细信息 ---
        if processed_count == 0:
            return f"[警告] {symbol} - {year_month}: 任务成功运行，但未找到任何匹配的源文件进行处理 (跳过了 {skipped_count} 个预设路径)。"
        

        return f"[成功] {symbol} - {year_month}: 成功处理了 {processed_count} 个文件 (跳过了 {skipped_count} 个预设路径)。"

    except Exception as e:
        # 【关键修改】捕获完整的错误堆栈信息并返回
        tb_str = traceback.format_exc()
        return f"[错误] {symbol} - {year_month}: {e}\n{tb_str}"
    


# 先单独计算到VIXY_resampled_1min，然后再合并
def generate_vix_level_global(config: dict):
    """
    【混合标准化恢复版】对不同分辨率(1min, 5min等)分别进行计算。
    合并所有VIXY月份数据，一次性计算raw vix_level，然后全局混合标准化，按月切分回写。
    """
    # 动态获取所有分辨率
    resolutions = []
    for session, time_ranges in config['resample_freq'].items():
        for time_range, res_list in time_ranges.items():
            resolutions.extend(res_list)
    resolutions = sorted(list(set(resolutions)))

    for res in resolutions:
        vix_dir = VIX_BASE_DIR / res
        if not vix_dir.exists():
            logging.info(f"跳过分辨率 {res}: 目录不存在 {vix_dir}")
            continue

        logging.info(f"开始计算 VIX 指标, 分辨率: {res}")
        all_files = sorted(vix_dir.glob('*.parquet'))
        if not all_files:
            logging.warning(f"分辨率 {res} 下没有找到VIXY Parquet文件。")
            continue

        df_all = pd.concat([pd.read_parquet(f) for f in all_files], ignore_index=True)
        if df_all.empty:
            logging.warning(f"分辨率 {res} 下所有VIXY文件为空。")
            continue
            
        df_all['timestamp'] = pd.to_datetime(df_all['timestamp'])
        df_all.set_index('timestamp', inplace=True)
        df_all.sort_index(inplace=True)
         
        df_all['vix_proxy_close'] = df_all['close']
        df_all['high'] = df_all['close']
        df_all['vix_proxy_close'] = df_all['vix_proxy_close'].ffill()

        # --- 窗口大小自适应 ---
        res_factor = 1
        if 'min' in res:
            try: res_factor = int(res.replace('min', ''))
            except: pass
        
        adj_long_term = int(63 * 390 / res_factor)
        long_term_ma = df_all['vix_proxy_close'].rolling(window=adj_long_term, min_periods=int(adj_long_term/3)).mean()
        df_all['vixy_detrended_level'] = df_all['vix_proxy_close'] / (long_term_ma + 1e-9)
        
        adj_macro = int(63 * 390 / res_factor)
        adj_macro_min = int(21 * 390 / res_factor)
        vixy_mean_macro = df_all['vix_proxy_close'].rolling(window=adj_macro, min_periods=adj_macro_min).mean()
        vixy_std_macro = df_all['vix_proxy_close'].rolling(window=adj_macro, min_periods=adj_macro_min).std()
        df_all['vix_z'] = (df_all['vix_proxy_close'] - vixy_mean_macro) / (vixy_std_macro + 1e-9)

        adj_intraday = int(60 / res_factor) if res_factor < 60 else 3
        adj_intraday_min = max(2, int(adj_intraday / 3))
        vixy_mean_intraday = df_all['vix_proxy_close'].rolling(window=adj_intraday, min_periods=adj_intraday_min).mean()
        vixy_std_intraday = df_all['vix_proxy_close'].rolling(window=adj_intraday, min_periods=adj_intraday_min).std()
        df_all['vix_level'] = (df_all['vix_proxy_close'] - vixy_mean_intraday) / (vixy_std_intraday + 1e-9)

        vixy_returns = df_all['vix_proxy_close'].pct_change()
        vixy_rolling_std_jump = vixy_returns.rolling(window=21).std()
        df_all['is_vix_jump'] = (vixy_returns.abs() > 4 * vixy_rolling_std_jump).astype(int)

        cols_to_fill = ['vixy_detrended_level', 'vix_z', 'vix_level']
        for col in cols_to_fill:
            df_all[col] = df_all[col].ffill().fillna(0.0)
        
        # 严禁未来插值：df_all['vix_level'].interpolate(method='linear', limit_direction='both')
        df_all['vix_level'] = df_all['vix_level'].replace(0, np.nan).ffill().fillna(0.0)
        
        df_all['close'] = df_all['vix_proxy_close']
        df_all['year_month'] = df_all.index.strftime('%Y-%m')
        
        for year_month, df_month in tqdm(df_all.groupby('year_month'), desc=f"Saving VIX {res}"):
            file_path = vix_dir / f"{year_month}.parquet"
            # 🚀 [Surgery 16] 修复列丢失漏洞：补全 vwap 和 transactions
            cols_to_keep = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'vwap', 'transactions', 'vix_proxy_close', 'vix_z', 'vix_level', 'is_vix_jump', 'vixy_detrended_level']
            df_to_save = df_month.reset_index()
            final_cols = [col for col in cols_to_keep if col in df_to_save.columns]
            df_to_save[final_cols].to_parquet(file_path, index=False, compression='zstd', compression_level=9)
            
        logging.info(f"分辨率 {res} 计算完成。")
        
        zero_count = (df_month['vix_level'] == 0).sum()
        if zero_count > 0:
            logging.warning(f"告警: {year_month} 有 {zero_count} 行 vix_level=0。")

        # --- 【核心修改】: 在保存的列中，增加 vwap, transactions, is_vix_jump ---
        cols_to_keep = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'vwap', 'transactions', 'vix_proxy_close', 'vix_z', 'vix_level', 'is_vix_jump', 'vixy_detrended_level']
        df_to_save = df_month.reset_index()
        
        final_cols = [col for col in cols_to_keep if col in df_to_save.columns]
        df_to_save[final_cols].to_parquet(file_path, index=False, compression='zstd', compression_level=9)
        # --- 【修改结束】 ---
    logging.info(f"vixy_detrended_level, vix_level, is_vix_jump 计算完成，处理 {len(all_files)} 个月份数据。")

# 在 if __name__ == "__main__": 下添加调用（可选，根据需要运行）
# generate_vix_level_to_uvxy()
# feature_merge_option.py 中的全局函数
 
def process_session_file(file_path: Path, config: dict):
    """
    【新增】工作函数：加载单个特征文件，使用高效的向量化方法更新 'session' 列。
    """
    try:
        df = pd.read_parquet(file_path)
        if df.empty or 'timestamp' not in df.columns:
            return f"[跳过] {file_path} 为空或缺少 'timestamp' 列。"

        # 实例化 FeatureEngineer 以复用其 session 边界和标签定义
        feature_engineer = FeatureEngineer(config)
        
        # --- 核心：高效的向量化 session 计算 ---
        
        # 1. 确保时间戳列是 datetime 类型
        ts_series = pd.to_datetime(df['timestamp'])
        
        # 2. 统一转换为纽约时区
        #    如果原始数据是 naive（无时区），先本地化为UTC，再转换为纽约时区
        if ts_series.dt.tz is None:
            ny_timestamps = ts_series.dt.tz_localize('America/New_York', ambiguous='infer')
        else:
            ny_timestamps = ts_series.dt.tz_convert(feature_engineer.ny_tz)
        
        # 3. 提取所有行的时间部分
        ny_time_series = pd.Series(ny_timestamps.dt.time, index=df.index)
        
        # 4. 使用 pandas.cut 进行高性能的区间划分
        #    这是 .apply() 的高性能替代方案，它一次性处理整个序列
        df['session'] = pd.cut(
            ny_time_series, 
            bins=feature_engineer.session_bins, 
            labels=feature_engineer.session_labels, 
            right=False,  # 区间为左闭右开 [start, end)
            ordered=False
        ).astype(int)
        
        # 5. 保存更新后的文件
        df.to_parquet(file_path, index=False, compression='zstd', compression_level=9)
        
        return f"[成功] 更新 session 列于 {file_path}"
    except Exception as e:
        import traceback
        return f"[错误] 更新 {file_path} 失败: {e}\n{traceback.format_exc()}"
    
def update_session_level_in_files(config: dict):
    """
    【新增】遍历所有已生成的特征文件，并批量更新 'session' 特征列。
    结构与 update_vix_level_in_files 类似。
    """
    tasks = list(OUTPUT_FEATURES_DIR.glob('*/*/*/*.parquet'))
    if not tasks:
        logging.warning("在输出目录中没有找到任何特征文件。")
        return

    logging.info(f"找到 {len(tasks)} 个特征文件，开始批量更新 'session' 特征列...")

    # 使用多进程并行更新
    worker_func = partial(process_session_file, config=config)
    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        results = list(tqdm(executor.map(worker_func, tasks), total=len(tasks), desc="Updating Session Feature"))

    # 打印统计和错误日志
    failures = [res for res in results if res.startswith("[错误]")]
    if failures:
        log_file = "session_feature_update_failures.log"
        with open(log_file, "w") as f:
            for failure_log in failures:
                f.write(f"{failure_log}\n")
        print(f"\n处理完成，有 {len(failures)} 个任务失败。详情请见 {log_file}。")
    else:
        print("\n处理完成，所有任务均成功！")

    success_count = len(results) - len(failures)
    logging.info(f"更新完成: 成功 {success_count}, 错误 {len(failures)}")

def process_cat_features_file(file_path: Path):
    """
    工作函数：加载单个特征文件，并从 timestamp 列重新、正确地计算所有
    时间相关的分类特征 (session, day_of_week, hour, is_holiday)，
    以覆盖任何可能因插值等操作产生的错误值。
    """
    try:
        df = pd.read_parquet(file_path)
        if df.empty or 'timestamp' not in df.columns:
            return f"[跳过] {file_path} 为空或缺少 'timestamp' 列。"

        # 1. 准备时间序列和 holidays 对象
        ts_series = pd.to_datetime(df['timestamp'])
        us_holidays = holidays.US()
        ny_tz = pytz.timezone('America/New_York')

        # 2. 统一转换为纽约时区
        if ts_series.dt.tz is None:
            ny_timestamps = ts_series.dt.tz_localize('America/New_York', ambiguous='infer')
        else:
            ny_timestamps = ts_series.dt.tz_convert(ny_tz)
        
        # 3. 重新计算并覆盖所有时间相关的分类特征
        # a) session
        session_bins = [
            dt_time(0, 0, 0), dt_time(4, 0, 0), dt_time(7, 0, 0), 
            dt_time(9, 30, 0), dt_time(16, 0, 0), dt_time(18, 0, 0), 
            dt_time(20, 0, 0), dt_time(23, 59, 59, 999999)
        ]
        session_labels = [-1, 0, 1, 2, 3, 4, -1]
        ny_time_series = pd.Series(ny_timestamps.dt.time, index=df.index)
        df['session'] = pd.cut(
            ny_time_series, bins=session_bins, labels=session_labels, 
            right=False, ordered=False
        ).astype(int)

        # b) day_of_week (范围 0-6)
        df['day_of_week'] = ny_timestamps.dt.dayofweek

        # c) hour (范围 0-23)
        df['hour'] = ny_timestamps.dt.hour

        # d) is_holiday (范围 0-1)
        df['month'] = ny_timestamps.dt.month
        df['is_holiday'] = pd.Series(ny_timestamps.dt.date).isin(us_holidays).astype(int)
       # c) 新增的周期性特征
        df['is_month_end'] = calculate_is_month_end(ny_timestamps)
        df['day_of_month'] = ny_timestamps.dt.day
        df['is_opex_week'] = calculate_is_opex_week(ny_timestamps)
        df['is_fomc_week'] = calculate_is_fomc_week(ny_timestamps, fomc_dates, n_days_before=3, n_days_after=2) # 比如扩大到会后两天

        

        # --- 【新增】计算 NFP 和 CPI 特征 ---
        df['is_nfp_day'] = calculate_is_event_day(ny_timestamps, nfp_release_dates, n_days_before=1, n_days_after=1)
        df['is_cpi_day'] = calculate_is_event_day(ny_timestamps, cpi_release_dates, n_days_before=1, n_days_after=1)
        df['is_ppi_day'] = calculate_is_event_day(ny_timestamps, ppi_release_dates, n_days_before=1, n_days_after=1)

         # --- 【核心修改 1】: 增加写前验证步骤 ---
        new_features_to_check = [
            'session', 'day_of_week', 'hour', 'is_holiday', 'is_month_end', 
            'day_of_month', 'is_opex_week', 'is_fomc_week', 'month','is_nfp_day', 'is_cpi_day','is_ppi_day'
        ]
        
        missing_cols = [col for col in new_features_to_check if col not in df.columns]

        if missing_cols:
            # 如果有任何一个新特征列没有成功生成，则不保存文件，并返回错误
            error_msg = f"未能成功生成以下字段: {', '.join(missing_cols)}"
            return {'status': 'error', 'path': str(file_path), 'message': error_msg}
        # --- 【修改结束】 ---


        # 4. 保存更新后的文件
        df.to_parquet(file_path, index=False, compression='zstd', compression_level=9)
        
        return {'status': 'success', 'path': str(file_path)}
    except Exception as e:
        return f"[错误] 校准 {file_path} 失败: {e}\n{traceback.format_exc()}"
    
def process_tnx_fnv_file(file_path: Path) -> dict:
    """
    【新】工作函数：为单个特征文件添加 vol_roc 和 vix_absolute_regime 特征。
    """
    try:
        df = pd.read_parquet(file_path)
        # 基础检查
        if df.empty or 'timestamp' not in df.columns:
            return {'status': 'skipped', 'path': str(file_path), 'message': "文件为空或缺少 'timestamp' 列。"}
        if 'garch_vol' not in df.columns:
             return {'status': 'skipped', 'path': str(file_path), 'message': "缺少 'garch_vol' 列，无法计算 vol_roc。"}

        
        # --- 2. 合并 VIXY 数据以计算 vix_absolute_regime ---
        # a. 确保主DataFrame的时间戳是带时区的
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        if df['timestamp'].dt.tz is None:
            df['timestamp'] = df['timestamp'].dt.tz_localize('America/New_York', ambiguous='infer')

        yield_df = pd.read_parquet('~/data/yield_curve_daily.parquet')
        yield_df['timestamp'] = pd.to_datetime(yield_df['timestamp']).dt.tz_localize('America/New_York')


        df.sort_values('timestamp', inplace=True)
        df = pd.merge_asof(
            df,
            yield_df[['timestamp', 'yield_spread_10y2y']],
            on='timestamp',
            direction='backward'
        )

        # 4. 保存更新后的文件
        df.to_parquet(file_path, index=False, compression='zstd', compression_level=9)
        
        return {'status': 'success', 'path': str(file_path)}
    except Exception as e:
        import traceback
        return {'status': 'error', 'path': str(file_path), 'message': f"发生意外异常: {e}\n{traceback.format_exc()}"}


def process_vol_vix_file(file_path: Path) -> dict:
    """
    【新】工作函数：为单个特征文件添加 vol_roc 和 vix_absolute_regime 特征。
    """
    try:
        df = pd.read_parquet(file_path)
        # 基础检查
        if df.empty or 'timestamp' not in df.columns:
            return {'status': 'skipped', 'path': str(file_path), 'message': "文件为空或缺少 'timestamp' 列。"}
        if 'garch_vol' not in df.columns:
             return {'status': 'skipped', 'path': str(file_path), 'message': "缺少 'garch_vol' 列，无法计算 vol_roc。"}

        # --- 1. 计算 vol_roc ---
        # garch_vol 是个股自身的波动率，可以直接计算
        df['vol_roc'] = df['garch_vol'].pct_change(periods=5).ffill().fillna(0.0)

        # --- 2. 合并 VIXY 数据以计算 vix_absolute_regime ---
        # a. 确保主DataFrame的时间戳是带时区的
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        if df['timestamp'].dt.tz is None:
            df['timestamp'] = df['timestamp'].dt.tz_localize('America/New_York', ambiguous='infer')

        # b. 从文件路径推断分辨率和年月
        # 路径结构: .../Resolution/YearMonth.parquet
        res = file_path.parent.name
        year_month = df['timestamp'].iloc[0].strftime('%Y-%m')
        vix_file_path = Path(VIX_PATH_TEMPLATE.format(res=res, year_month=year_month))

        if vix_file_path.exists():
            vix_df = pd.read_parquet(vix_file_path)
            if not vix_df.empty and 'vixy_detrended_level' in vix_df.columns:
                vix_df['timestamp'] = pd.to_datetime(vix_df['timestamp'])
                if vix_df['timestamp'].dt.tz is None:
                    vix_df['timestamp'] = vix_df['timestamp'].dt.tz_localize('America/New_York', ambiguous='infer')
                
                vix_df.sort_values('timestamp', inplace=True)
                vix_df.drop_duplicates(subset=['timestamp'], keep='last', inplace=True)

                df.drop(columns=['vix_proxy_close', 'vix_level'], inplace=True, errors='ignore')

                # 使用 merge_asof 合并 vix_proxy_close
                df.sort_values('timestamp', inplace=True)
                df = pd.merge_asof(
                    df,
                    vix_df[['timestamp',  'vix_z', 'is_vix_jump', 'vix_level', 'vixy_detrended_level']],
                    on='timestamp',
                    direction='backward'
                )
                df['vixy_detrended_level'] = df['vixy_detrended_level'].ffill()
                df['vix_z'] = df['vix_z'].ffill()
                df['is_vix_jump'] = df['is_vix_jump'].ffill().fillna(0).astype(int)
                df['vix_level'] = df['vix_level'].ffill()
            else:
                df['vixy_detrended_level'] = 0.0 # VIXY文件存在但为空或无数据
        else:
            df['vixy_detrended_level'] = 0.0 # VIXY文件不存在

         

        # 4. 保存更新后的文件
        df.to_parquet(file_path, index=False, compression='zstd', compression_level=9)
        
        return {'status': 'success', 'path': str(file_path)}
    except Exception as e:
        import traceback
        return {'status': 'error', 'path': str(file_path), 'message': f"发生意外异常: {e}\n{traceback.format_exc()}"}


def update_vol_vix_abs(config: dict):
    """
    【新】管理者函数：并行地为所有特征文件添加 vol_roc 和 vix_absolute_regime 特征。
    """
    tasks = list(OUTPUT_FEATURES_DIR.glob('*/*/*/*/*.parquet'))
    if not tasks:
        logging.warning("在输出目录中没有找到任何特征文件可供更新。")
        return

    logging.info(f"找到 {len(tasks)} 个特征文件，开始批量添加 vol_roc 和 vix_Z等 特征...")

    results = []
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # 我们这里不需要传递 config，所以可以直接 map
        results_iterator = executor.map(process_vol_vix_file, tasks)
        
        pbar = tqdm(results_iterator, total=len(tasks), desc="更新波动率特征")
        # 直接将迭代器转换为列表以等待所有任务完成
        results = list(pbar)

    # 统计成功与失败
    failures = [res for res in results if res['status'] == 'error']
    success_count = sum(1 for res in results if res['status'] == 'success')
    skipped_count = sum(1 for res in results if res['status'] == 'skipped')

    if failures:
        log_file = "vol_vix_update_failures.log"
        with open(log_file, "w", encoding='utf-8') as f:
            for failure_dict in failures:
                f.write(f"文件路径: {failure_dict['path']}\n")
                f.write(f"  - 错误信息: {failure_dict['message']}\n")
                f.write("-" * 50 + "\n")
        print(f"\n处理完成，有 {len(failures)} 个任务失败。详情请见 {log_file}。")
    else:
        print("\n处理完成，所有新特征均已成功添加！")

    logging.info(f"更新摘要: 成功 {success_count} 个文件, 跳过 {skipped_count} 个文件, 错误 {len(failures)} 个文件。")


def update_cat_features_in_files(config: dict):
    """
    【V3 - 增强版】主函数：遍历所有特征文件，批量校准时间分类特征。
    能处理结构化的成功/失败信息，并提供更清晰的日志。
    """
    tasks = list(OUTPUT_FEATURES_DIR.glob('*/*/*/*/*.parquet'))
    if not tasks:
        logging.warning("在输出目录中没有找到任何特征文件可供校准。")
        return

    logging.info(f"找到 {len(tasks)} 个特征文件，开始批量校准时间分类特征...")

    results = []
    
    import concurrent.futures

    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_path = {executor.submit(process_cat_features_file, path): path for path in tasks}
        
        pbar = tqdm(concurrent.futures.as_completed(future_to_path), total=len(tasks), desc="校准分类特征")
        for future in pbar:
            path = future_to_path[future]
            
            try:
                # 解析股票代码用于实时显示
                symbol = path.parts[-5]
                pbar.set_postfix_str(f"完成: {symbol}")
                
                # 获取工作函数返回的结构化字典
                result = future.result()
                results.append(result)
                
            except Exception as e:
                # 捕获子进程崩溃等严重错误
                error_dict = {'status': 'error', 'path': str(path), 'message': f"子进程崩溃: {e}"}
                results.append(error_dict)
    
    # --- 【核心修改 2】: 根据结构化的结果进行统计 ---
    failures = [res for res in results if res['status'] == 'error']
    success_count = sum(1 for res in results if res['status'] == 'success')
    skipped_count = sum(1 for res in results if res['status'] == 'skipped')

    if failures:
        log_file = "cat_feature_update_failures.log"
        with open(log_file, "w", encoding='utf-8') as f:
            for failure_dict in failures:
                f.write(f"文件路径: {failure_dict['path']}\n")
                f.write(f"  - 错误信息: {failure_dict['message']}\n")
                f.write("-" * 50 + "\n")
        print(f"\n处理完成，有 {len(failures)} 个任务失败。详情请见 {log_file}。")
    else:
        print("\n处理完成，所有分类特征均已成功校准！")

    logging.info(f"校准摘要: 成功 {success_count} 个文件, 跳过 {skipped_count} 个文件, 错误 {len(failures)} 个文件。")

# feature_merge_option.py -> 添加这两个新函数


def update_vix_level_in_files(config: dict):
    """
    遍历已生成 Parquet 文件，只更新 vix_level 列。
    """
    tasks = list(OUTPUT_FEATURES_DIR.glob('*/*/*/*/*.parquet'))
    if not tasks:
        logging.warning("没有找到已生成的文件。")
        return

    

    # 多进程并行更新
    worker_func = partial(process_file, config=config)
    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        results = list(tqdm(executor.map(worker_func, tasks), total=len(tasks), desc="Updating VIX Level"))

    failures = []
    # 遍历所有结果，筛选出失败的日志
    for i, res in enumerate(results):
        if res.startswith("[错误]"):
            # 将任务参数和详细错误信息都记录下来
            failed_task = tasks[i]
            failures.append(f"任务: {failed_task}\n错误信息: {res}\n" + "="*50 + "\n")

    # 将所有失败日志写入文件
    if failures:
        with open("feature_generation_failures.log", "w") as f:
            for failure_log in failures:
                f.write(failure_log)
        print(f"\n处理完成，发现 {len(failures)} 个失败任务。详细错误已写入 feature_generation_failures.log 文件。")
    else:
        print("\n处理完成，所有任务均成功！")


    # 打印统计
    success_count = sum(1 for r in results if r.startswith("[成功]"))
    error_count = sum(1 for r in results if r.startswith("[错误]"))
    logging.info(f"更新完成: 成功 {success_count}, 错误 {error_count}")

# new_feature.py 中的全局函数
def process_file(file_path: Path, config: dict):
    """
    【简化版】加载单个已生成的特征文件，合并VIXY数据（含预计算的vix_level），并直接更新 'vix_level' 列。
    无需重新计算或标准化。
    """
    try:
        df = pd.read_parquet(file_path)
        if df.empty or 'timestamp' not in df.columns:
            return f"[跳过] {file_path} 为空或无 timestamp 列。"

        # 确保主DataFrame的时间戳是带时区的
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        if df['timestamp'].dt.tz is None:
            df['timestamp'] = df['timestamp'].dt.tz_localize('America/New_York', ambiguous='infer')

        # 获取年月
        parts = file_path.parts
        year_month = Path(parts[-1]).stem

        # 加载VIXY数据（已含vix_level）
        vix_df = None
        vix_file_path = Path(VIX_PATH_TEMPLATE.format(year_month=year_month))
        if vix_file_path.exists():
            try:
                vix_df = pd.read_parquet(vix_file_path)
                if vix_df.empty:
                    vix_df = None
            except Exception as e:
                logging.warning(f"加载VIXY文件失败: {vix_file_path}, 错误: {e}")
                vix_df = None

        if vix_df is not None:
            # 合并VIXY数据，直接取vix_level
            vix_df['timestamp'] = pd.to_datetime(vix_df['timestamp'])
            if vix_df['timestamp'].dt.tz is None:
                vix_df['timestamp'] = vix_df['timestamp'].dt.tz_localize('America/New_York', ambiguous='infer')

            vix_df.sort_values('timestamp', inplace=True)
            vix_df.drop_duplicates(subset=['timestamp'], keep='last', inplace=True)

            df.sort_values('timestamp', inplace=True)

            # 先删除df中现有的vix_level（如果存在），以避免冲突
            if 'vix_level' in df.columns:
                df.drop(columns=['vix_level'], inplace=True)

            df = pd.merge_asof(
                df,
                vix_df[['timestamp', 'vix_level']],
                on='timestamp',
                direction='backward'
            )
            df['vix_level'] = df['vix_level'].ffill()  # 填充任何缺失
        else:
            # VIXY不可用，填充默认值
            logging.warning(f"文件 {file_path} 对应的VIXY数据 ({year_month}) 不存在或为空。'vix_level' 将被填充为0。")
            df['vix_level'] = 0.0

        # 保存更新后的文件
        df.to_parquet(file_path, index=False, compression='zstd', compression_level=9)
        
        return f"[成功] 更新 {file_path}"
    except Exception as e:
        import traceback
        return f"[错误] 更新 {file_path} 失败: {e}\n{traceback.format_exc()}"


def update_new_labels_in_files(config: dict):
    """
    【v4.1 - 修复版】增加对缺失统计数据的兼容性处理，确保计数正确。
    """
    tasks = list(OUTPUT_FEATURES_DIR.glob('*/*/*/*/*.parquet'))
    if not tasks:
        logging.warning("没有找到已生成的文件。")
        return

    logging.info(f"找到 {len(tasks)} 个文件，开始批量更新标签 (Triple Barrier)...")

    worker_func = partial(process_labels_file, config=config)
    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        results = list(tqdm(executor.map(worker_func, tasks), total=len(tasks), desc="Updating Labels"))

    # --- 数据收集 ---
    failures = []
    success_stats = [] 
    skipped_no_volume_count = 0

    for res in results:
        status = res.get("status") if isinstance(res, dict) else "unknown"
        
        if status == "success":
            # [核心修复] 使用 .get() 提供默认值，防止因 key 缺失导致不计入成功
            # 如果 process_labels_file 忘记返回 ratio，默认设为 100% 盘整，0 volume
            ratio = res.get("consol_ratio", 100.0)
            vol_ratio = res.get("volume_ratio", 0.0)
            
            success_stats.append((ratio, vol_ratio))
            
        elif status == "skipped_no_volume":
            skipped_no_volume_count += 1
        elif status == "error":
             failures.append(f"任务: {res.get('path')}\n错误信息: {res.get('message')}\n" + "="*50 + "\n")

    # --- 分布摘要计算 ---
    total_success = len(success_stats)
    
    # 定义辅助函数
    def avg_vol_ratio(bucket):
        if not bucket: return 0.0
        return np.mean([s[1] for s in bucket]) * 100

    if total_success > 0:
        bucket_100 = [s for s in success_stats if s[0] >= 100]
        bucket_95_100 = [s for s in success_stats if 95 <= s[0] < 100]
        bucket_90_95 = [s for s in success_stats if 90 <= s[0] < 95]
        bucket_80_90 = [s for s in success_stats if 80 <= s[0] < 90]
        bucket_lt_80 = [s for s in success_stats if s[0] < 80]
        
        print("\n" + "="*80)
        print("--- 全局盘整率分布摘要 (包含数据质量诊断) ---")
        print(f"总计成功文件数: {total_success}")
        print(f"  - 等于 100%   : {len(bucket_100):<5} ({len(bucket_100)/total_success:6.1%}) | Vol: {avg_vol_ratio(bucket_100):5.1f}%")
        print(f"  - [95%, 100%) : {len(bucket_95_100):<5} ({len(bucket_95_100)/total_success:6.1%}) | Vol: {avg_vol_ratio(bucket_95_100):5.1f}%")
        print(f"  - [90%, 95%)  : {len(bucket_90_95):<5} ({len(bucket_90_95)/total_success:6.1%}) | Vol: {avg_vol_ratio(bucket_90_95):5.1f}%")
        print(f"  - [80%, 90%)  : {len(bucket_80_90):<5} ({len(bucket_80_90)/total_success:6.1%}) | Vol: {avg_vol_ratio(bucket_80_90):5.1f}%")
        print(f"  - 小于 80%    : {len(bucket_lt_80):<5} ({len(bucket_lt_80)/total_success:6.1%}) | Vol: {avg_vol_ratio(bucket_lt_80):5.1f}%")
        print("="*80 + "\n")

    # --- 最终报告 ---
    total_processed = len(results)
    print("--- 总体处理情况 ---")
    print(f"总任务数: {total_processed}")
    print(f"  - 成功: {total_success}")
    print(f"  - 失败: {len(failures)}")
    
    if failures:
        with open("label_update_failures.log", "w", encoding='utf-8') as f:
            for log in failures: f.write(log)
        print(f"  失败详情已写入 label_update_failures.log")

 
def process_labels_file(file_path: Path, config: dict = None) -> dict:
    """
    [New Delta 架构 - 30min Horizon 适配版]
    修改点:
    1. k_slow: 5 -> 30 (30分钟预测)
    2. vol_window: 60 -> 120 (更稳定的基准)
    3. Multipliers: 根据 sqrt(T) 放大阈值，防止标签噪声化。
    """
    try:
        # 1. [I/O] 读取数据
        if not file_path.exists():
            return {'status': 'error', 'path': str(file_path), 'message': 'File not found'}
            
        df = pd.read_parquet(file_path)
        if 'timestamp' in df.columns:
            df = df.sort_values('timestamp').reset_index(drop=True)
        
        price_col = 'close' if 'close' in df.columns else 'price'
        if price_col not in df.columns:
             return {'status': 'skipped', 'path': str(file_path), 'message': 'No price column'}

        # 2. [基础计算]
        df['log_ret'] = np.log(df[price_col] / df[price_col].shift(1).replace(0, np.nan)).fillna(0.0)
        
        # [修改点 A] 波动率基准窗口放大
        # 预测未来 30 分钟，建议参考过去 2 小时 (120 min) 的波动率环境
        vol_window = 120 
        current_vol = df['log_ret'].rolling(window=vol_window).std()
        
        # [修改点 B] 核心预测窗口
        k_slow = 30
        
        # 准备数据视图
        from numpy.lib.stride_tricks import sliding_window_view
        high_arr = df['high'].to_numpy()
        low_arr = df['low'].to_numpy()
        close_arr = df['close'].to_numpy()
        
        f_high = np.append(high_arr[1:], [np.nan]) 
        f_low = np.append(low_arr[1:], [np.nan])   
        
        f_high = np.nan_to_num(f_high, nan=close_arr[-1])
        f_low = np.nan_to_num(f_low, nan=close_arr[-1])
        
        n = len(df)
        valid_len = max(0, n - k_slow)
        
        # ==============================================================================
        # 3. [Label Generation]
        # ==============================================================================
        
        if valid_len > 0:
            win_high = sliding_window_view(f_high[:n-1], window_shape=k_slow)[:valid_len]
            win_low = sliding_window_view(f_low[:n-1], window_shape=k_slow)[:valid_len]
            curr_close = close_arr[:valid_len]
            curr_vol = current_vol.to_numpy()[:valid_len]
            
            # --- [修改点 C] Label: Direction (适配 30min) ---
            # curr_vol 是 1min 级别的 std。
            # 30min 的波动率约为 curr_vol * sqrt(30) ≈ 5.4 * curr_vol
            # 我们希望 Barrier 设在约 0.5 倍的 30min 波动率处，即 2.5 ~ 3.0 * curr_vol
            
            # 系数: 0.5 -> 2.5
            # Clip Min: 0.0002 -> 0.002 (0.2% 起步，太小的波动不算趋势)
            # Clip Max: 0.002 -> 0.015 (1.5% 封顶，30分钟涨跌 1.5% 算特大行情了)
            dir_thresh = (curr_vol * 2.5).clip(min=0.002, max=0.015)
            
            upper_barrier = curr_close * (1.0 + dir_thresh)
            lower_barrier = curr_close * (1.0 - dir_thresh)
            
            hit_up = win_high > upper_barrier[:, None]
            hit_down = win_low < lower_barrier[:, None]
            
            first_up_idx = np.argmax(hit_up, axis=1)
            first_down_idx = np.argmax(hit_down, axis=1)
            any_up = np.any(hit_up, axis=1)
            any_down = np.any(hit_down, axis=1)
            
            is_up = any_up & (~any_down | (first_up_idx < first_down_idx))
            is_down = any_down & (~any_up | (first_down_idx < first_up_idx))
            
            dir_labels = np.ones(valid_len, dtype=int)
            dir_labels[is_up] = 2
            dir_labels[is_down] = 0
            
            # --- [修改点 D] Label: Event (适配 30min) ---
            # Event 代表极端行情。
            # 30min 内发生极端行情的阈值应该更高。
            # 系数: 2.0 -> 6.0 (约等于 30min 波动率的 1 倍标准差以上才算 Event)
            evt_thresh = (curr_vol * 6.0).clip(min=0.005) # 最小波动 0.5% 才开始算 Event
            
            max_h = np.max(win_high, axis=1)
            min_l = np.min(win_low, axis=1)


            
            # --- 【修复开始】数值稳定性保护 ---
            # 1. 识别有效价格 (价格必须 > 0)
            valid_price_mask = curr_close > 1e-9
            
            # 2. 创建安全分母：将 0 替换为 inf (或非常大的数)，这样除法结果为 0，不会触发阈值
            safe_close = np.where(valid_price_mask, curr_close, np.inf)
            
            # 3. 计算比率 (现在除以 inf 会得到 0，这是安全的)
            ratio_up = max_h / safe_close - 1.0
            ratio_down = 1.0 - min_l / safe_close
            
            # 4. 判定 (自动过滤掉了 safe_close 为 inf 的情况)
            is_huge_up = ratio_up > evt_thresh
            is_huge_down = ratio_down > evt_thresh
            # --- 【修复结束】 ---
            evt_labels = (is_huge_up | is_huge_down).astype(int)
            
            # 赋值
            df['label_direction'] = 1
            df['label_event'] = 0
            df.iloc[:valid_len, df.columns.get_loc('label_direction')] = dir_labels
            df.iloc[:valid_len, df.columns.get_loc('label_event')] = evt_labels
            
            # 其他标签
            vol_series = df['log_ret'].rolling(window=k_slow).std().shift(-k_slow)
            df['label_volatility'] = vol_series.fillna(0.0)
            df['label_return_fwd'] = (df[price_col].shift(-k_slow) / df[price_col] - 1.0).fillna(0.0)

        else:
            df['label_direction'] = 1
            df['label_event'] = 0
            df['label_volatility'] = 0.0
            df['label_return_fwd'] = 0.0

        # 4. [I/O] 保存
        cols_to_drop = ['ret_5m_fwd', 'log_ret', 'return_fwd']
        old_cols = ['direction', 'volatility', 'event']
        df.drop(columns=cols_to_drop + [c for c in old_cols if c in df.columns], inplace=True, errors='ignore')
        
        df.to_parquet(file_path, index=False, engine='pyarrow')
        
        # 5. [Stats]
        consol_ratio = 100.0
        if valid_len > 0:
            valid_labels = df['label_direction'].iloc[:valid_len]
            consol_ratio = (valid_labels == 1).mean() * 100
            
        volume_ratio = 0.0
        if 'volume' in df.columns:
            volume_ratio = (df['volume'] > 0).mean()

        return {
            'status': 'success', 
            'path': str(file_path), 
            'consol_ratio': consol_ratio,
            'volume_ratio': volume_ratio
        }

    except Exception as e:
        return {'status': 'error', 'path': str(file_path), 'message': str(e) + f"\n{traceback.format_exc()}"}


def process_labels_file_fast(file_path: Path, config: dict = None) -> dict:
    """
    [New Delta 架构 - Triple Barrier 终极版]
    修复内容:
    1. Event: 基于 High/Low 触达，放宽阈值 (2.0x)。
    2. Direction: 基于 Triple Barrier (先碰上轨还是下轨)。
    3. Stats: 增加返回 consol_ratio 和 volume_ratio，修复统计显示问题。
    """
    try:
        # 1. [I/O] 读取数据
        if not file_path.exists():
            return {'status': 'error', 'path': str(file_path), 'message': 'File not found'}
            
        df = pd.read_parquet(file_path)
        if 'timestamp' in df.columns:
            df = df.sort_values('timestamp').reset_index(drop=True)
        
        price_col = 'close' if 'close' in df.columns else 'price'
        if price_col not in df.columns:
             return {'status': 'skipped', 'path': str(file_path), 'message': 'No price column'}

        # 2. [基础计算]
        df['log_ret'] = np.log(df[price_col] / df[price_col].shift(1).replace(0, np.nan)).fillna(0.0)
        
        # 动态波动率基准 (60分钟窗口)
        vol_window = 60 
        current_vol = df['log_ret'].rolling(window=vol_window).std()
        
        k_slow = 5 
        
        # 准备数据视图
        from numpy.lib.stride_tricks import sliding_window_view
        high_arr = df['high'].to_numpy()
        low_arr = df['low'].to_numpy()
        close_arr = df['close'].to_numpy()
        
        f_high = np.append(high_arr[1:], [np.nan]) 
        f_low = np.append(low_arr[1:], [np.nan])   
        
        f_high = np.nan_to_num(f_high, nan=close_arr[-1])
        f_low = np.nan_to_num(f_low, nan=close_arr[-1])
        
        n = len(df)
        valid_len = max(0, n - k_slow)
        
        # ==============================================================================
        # 3. [Label Generation]
        # ==============================================================================
        
        if valid_len > 0:
            win_high = sliding_window_view(f_high[:n-1], window_shape=k_slow)[:valid_len]
            win_low = sliding_window_view(f_low[:n-1], window_shape=k_slow)[:valid_len]
            curr_close = close_arr[:valid_len]
            curr_vol = current_vol.to_numpy()[:valid_len]
            
            # --- Label: Direction (Triple Barrier) ---
            dir_thresh = (curr_vol * 0.5).clip(min=0.0002, max=0.002)
            upper_barrier = curr_close * (1.0 + dir_thresh)
            lower_barrier = curr_close * (1.0 - dir_thresh)
            
            hit_up = win_high > upper_barrier[:, None]
            hit_down = win_low < lower_barrier[:, None]
            
            first_up_idx = np.argmax(hit_up, axis=1)
            first_down_idx = np.argmax(hit_down, axis=1)
            any_up = np.any(hit_up, axis=1)
            any_down = np.any(hit_down, axis=1)
            
            is_up = any_up & (~any_down | (first_up_idx < first_down_idx))
            is_down = any_down & (~any_up | (first_down_idx < first_up_idx))
            
            dir_labels = np.ones(valid_len, dtype=int)
            dir_labels[is_up] = 2
            dir_labels[is_down] = 0
            
            # --- Label: Event ---
            evt_thresh = (curr_vol * 2.0).clip(min=0.001)
            max_h = np.max(win_high, axis=1)
            min_l = np.min(win_low, axis=1)

            
            # --- 【修复开始】数值稳定性保护 ---
            # 1. 识别有效价格
            valid_price_mask = curr_close > 1e-9
            
            # 2. 安全分母 (用 np.inf 避免除零，且让结果趋近于0)
            safe_close = np.where(valid_price_mask, curr_close, np.inf)
            
            # 3. 计算
            ratio_up = max_h / safe_close - 1.0
            ratio_down = 1.0 - min_l / safe_close
            
            is_huge_up = ratio_up > evt_thresh
            is_huge_down = ratio_down > evt_thresh
            # --- 【修复结束】 ---
            evt_labels = (is_huge_up | is_huge_down).astype(int)
            
            # 赋值
            # 🚀 [修正：移除抢跑式标注以防止泄露]
            df['label_direction'] = dir_labels
            df['label_event'] = evt_labels
            
            # 其他标签
            vol_series = df['log_ret'].rolling(window=k_slow).std().shift(-k_slow)
            df['label_volatility'] = vol_series.fillna(0.0)
            df['label_return_fwd'] = (df[price_col].shift(-k_slow) / df[price_col] - 1.0).fillna(0.0)

        else:
            df['label_direction'] = 1
            df['label_event'] = 0
            df['label_volatility'] = 0.0
            df['label_return_fwd'] = 0.0

        # 4. [I/O] 保存
        cols_to_drop = ['ret_5m_fwd', 'log_ret', 'return_fwd']
        old_cols = ['direction', 'volatility', 'event']
        df.drop(columns=cols_to_drop + [c for c in old_cols if c in df.columns], inplace=True, errors='ignore')
        
        df.to_parquet(file_path, index=False, engine='pyarrow')
        
        # 5. [Stats] 计算统计信息 (修复点)
        consol_ratio = 100.0
        if valid_len > 0:
            valid_labels = df['label_direction'].iloc[:valid_len]
            consol_ratio = (valid_labels == 1).mean() * 100
            
        volume_ratio = 0.0
        if 'volume' in df.columns:
            volume_ratio = (df['volume'] > 0).mean()

        return {
            'status': 'success', 
            'path': str(file_path), 
            'consol_ratio': consol_ratio,  # 返回盘整率
            'volume_ratio': volume_ratio   # 返回成交量占比
        }

    except Exception as e:
        return {'status': 'error', 'path': str(file_path), 'message': str(e) + f"\n{traceback.format_exc()}"}

def process_labels_file_back(file_path: Path, config: dict):
    """
    【V10 - 纯动态 Triple Barrier 版】
    专为日内高频策略设计：
    1. 移除 Quantile Fallback：拒绝在低波动环境强行打标签。
    2. 引入 Triple Barrier：路径依赖，检测未来 K 周期内 High/Low 是否触及动态阈值。
    3. 硬性底线：阈值永远不会低于交易成本覆盖线。
    """
    try:
        # --- 1. 读取数据与基础检查 ---
        df = pd.read_parquet(file_path)
        required_cols = ['close', 'volume', 'high', 'low']
        if df.empty or not all(col in df.columns for col in required_cols):
            return {"status": "skipped_empty", "file_path": str(file_path)}

        # --- 2. 获取配置参数 ---
        labeling_params = config.get('parameters', {}).get('labeling', {})
        k = labeling_params.get('k', 10)  # 预测窗口，例如未来10分钟
        
        # 波动率计算窗口：日内分钟线建议用 60 (1小时)
        vol_window = 60 
        
        # [关键参数] 绝对最小阈值 (Hard Floor)
        # 必须大于: 2 * (Bid-Ask Spread + Commission + Slippage)
        # 0.0005 = 5bps (万分之五)，对于 NQ/个股是合理的日内底线
        abs_min_thresh = 0.0005 

        # --- 3. 预处理：无量过滤 ---
        has_volume_mask = df['volume'] > 1e-9 
        volume_ratio = has_volume_mask.mean()

        # 如果全天无量，直接返回全 Flat 标签
        if not has_volume_mask.any():
            df['label_direction'] = 1
            df['label_volatility'] = 0.0
            df['label_event'] = 0
            df.to_parquet(file_path, index=False, compression='zstd', compression_level=9)
            return {"status": "success", "file_path": str(file_path), "consol_ratio": 100.0, "volume_ratio": 0.0}

        # ==============================================================================
        # [核心模块 A] 计算动态波动率阈值 (Barrier Width)
        # ==============================================================================
        
        # 1. 计算对数收益率
        df['log_ret'] = np.log(df['close'] / df['close'].shift(1)).fillna(0)
        
        # 2. 计算滚动波动率 (Std Dev)
        rolling_vol = df['log_ret'].rolling(window=vol_window, min_periods=20).std()
        
        # 3. 确定自适应乘数 (Multiplier)
        # 日内数据噪音大，需要较小的乘数来捕捉突破，或者较大的乘数来过滤噪音
        # Triple Barrier 逻辑下，我们希望捕捉"能碰到边界"的行情
        # 建议：0.5 ~ 0.8 倍的 sqrt(k) * vol
        adaptive_multiplier = 0.5 
        
        if 'vix_level' in df.columns:
            # 如果有 VIX 数据，可以微调乘数
            # VIX 高 -> 波动大 -> 乘数可以保持 0.5 (阈值自然变大)
            # VIX 低 -> 波动小 -> 乘数可以提高到 0.7 (过滤死水)
            vix_series = df['vix_level'].fillna(0)
            adaptive_multiplier = np.where(vix_series < -0.5, 0.7, 0.5)
        
        # 4. 计算半宽 (Half-Width) 阈值
        # 公式: Vol * sqrt(k) * multiplier
        dynamic_width = (rolling_vol * np.sqrt(k) * adaptive_multiplier)
        
        # 5. [关键] 应用硬性底线
        # 即使波动率极低，阈值也不能小于交易成本
        dynamic_width = dynamic_width.fillna(abs_min_thresh).clip(lower=abs_min_thresh)

        # ==============================================================================
        # [核心模块 B] Triple Barrier (三重障碍) 路径扫描
        # ==============================================================================
        
        # 准备数据：我们需要看未来 k 步的 High 和 Low
        # shift(-1) 是因为站在 t 时刻，我们看的是 t+1 到 t+k
        future_highs = df['high'].shift(-1).to_numpy()
        future_lows = df['low'].shift(-1).to_numpy()
        current_closes = df['close'].to_numpy()
        
        # 计算有效长度 (最后 k 行无法向后看 k 步)
        n_samples = len(df)
        valid_idx_end = n_samples - k
        
        # 初始化标签为 1 (Hold/Flat)
        labels = np.ones(n_samples, dtype=int)
        
        if valid_idx_end > 0:
            # 1. 构建滑动窗口矩阵 (Shape: [N-k, k])
            # 这比循环快 100 倍以上
            # highs_window[i] 包含: high[i+1], high[i+2] ... high[i+k]
            highs_window = sliding_window_view(future_highs[:n_samples-1], window_shape=k)
            lows_window = sliding_window_view(future_lows[:n_samples-1], window_shape=k)
            
            # 截取对应长度的当前价格和阈值
            curr_close_valid = current_closes[:valid_idx_end]
            thresholds_valid = dynamic_width.to_numpy()[:valid_idx_end]
            
            # 2. 计算上轨和下轨价格 (Barrier Prices)
            upper_barriers = curr_close_valid * (1 + thresholds_valid)
            lower_barriers = curr_close_valid * (1 - thresholds_valid)
            
            # 3. 广播比较：检查是否触及
            # hit_upper: 布尔矩阵，如果某一步 High >= Upper Barrier 则为 True
            # hit_lower: 布尔矩阵，如果某一步 Low <= Lower Barrier 则为 True
            hit_upper = highs_window[:valid_idx_end] >= upper_barriers[:, None]
            hit_lower = lows_window[:valid_idx_end] <= lower_barriers[:, None]
            
            # 4. 寻找"第一次"触及的时间
            # argmax 会返回第一个 True 的索引；如果全 False 也会返回 0
            # 所以我们需要配合 any() 使用
            first_upper_idx = hit_upper.argmax(axis=1)
            first_lower_idx = hit_lower.argmax(axis=1)
            
            did_hit_upper = hit_upper.any(axis=1)
            did_hit_lower = hit_lower.any(axis=1)
            
            # 5. 判定逻辑
            # 情况 A: 这是一个做多信号 (Label 2)
            # 条件: (碰到上轨) 且 (没碰到下轨 或者 碰到下轨但在碰到上轨之后)
            # 注意：如果同一分钟同时碰到上下轨 (idx相等)，通常视为噪音或震荡，这里偏向于不操作或根据 Close 决定
            # 这里采用保守策略：如果同时碰到，视为 Hold (1)，不产生信号
            mask_buy = did_hit_upper & (~did_hit_lower | (first_upper_idx < first_lower_idx))
            
            # 情况 B: 这是一个做空信号 (Label 0)
            # 条件: (碰到下轨) 且 (没碰到上轨 或者 碰到上轨但在碰到下轨之后)
            mask_sell = did_hit_lower & (~did_hit_upper | (first_lower_idx < first_upper_idx))
            
            # 6. 赋值
            # 临时标签数组
            temp_labels = np.ones(valid_idx_end, dtype=int)
            temp_labels[mask_buy] = 2
            temp_labels[mask_sell] = 0
            
            # 将计算好的标签填回主数组
            labels[:valid_idx_end] = temp_labels
            
            # 确保无量时段强制为 1
            labels[:valid_idx_end] = np.where(has_volume_mask[:valid_idx_end], labels[:valid_idx_end], 1)

        df['label_direction'] = labels

        # ==============================================================================
        # [辅助模块] 其他标签计算 (保持原逻辑，优化实现)
        # ==============================================================================
        
        # 1. label_volatility (未来 K 周期已实现波动率)
        if valid_idx_end > 0:
            log_rets_future = np.log(df['close'] / df['close'].shift(1)).shift(-1).to_numpy()
            log_rets_window = sliding_window_view(log_rets_future[:n_samples-1], window_shape=k)
            
            # 计算窗口内的标准差并年化
            raw_vol = np.std(log_rets_window[:valid_idx_end], axis=1, ddof=1)
            annual_factor = np.sqrt(252 * 390)
            df['label_volatility'] = 0.0
            df.iloc[:valid_idx_end, df.columns.get_loc('label_volatility')] = raw_vol * annual_factor
        else:
             df['label_volatility'] = 0.0

        # 2. label_event (大波动事件检测)
        # 沿用之前的逻辑，检测 Range 是否异常大
        D_event = labeling_params.get('D', 2.0)
        
        if valid_idx_end > 0:
            # 这里的 high/low window 已经在上面 Triple Barrier 计算过了，直接复用逻辑
            # 计算区间幅度: (Max(High) - Min(Low)) / Current Close
            # 注意：windowview 是 shape (N, k)
            range_max = np.max(highs_window[:valid_idx_end], axis=1)
            range_min = np.min(lows_window[:valid_idx_end], axis=1)
            price_range_ratio = (range_max - range_min) / (current_closes[:valid_idx_end] + 1e-9)
            
            # 事件阈值
            event_thresh = (dynamic_width[:valid_idx_end] * D_event).to_numpy()
            
            # 事件日特殊处理
            is_event_day = (df.get('is_fomc_week', 0) == 1) | \
                           (df.get('is_cpi_day', 0) == 1) | \
                           (df.get('is_nfp_day', 0) == 1)
            
            # 如果是事件日，稍微放宽一点点阈值，或者保持不变
            # 这里简单处理：直接比较
            is_event = price_range_ratio > event_thresh
            
            df['label_event'] = 0
            df.iloc[:valid_idx_end, df.columns.get_loc('label_event')] = is_event.astype(int)
        else:
            df['label_event'] = 0

        # ==============================================================================
        # 5. 清理与保存
        # ==============================================================================
        if 'log_ret' in df.columns: 
            df.drop(columns=['log_ret'], inplace=True)
        
        df.to_parquet(file_path, index=False, compression='zstd', compression_level=9)
        
        # 计算盘整率 (Label 1 的占比)
        valid_labels = df['label_direction'][:valid_idx_end]
        consol_ratio = (valid_labels == 1).sum() / len(valid_labels) * 100 if len(valid_labels) > 0 else 100.0
        
        return {"status": "success", "file_path": str(file_path), 
                "consol_ratio": consol_ratio, "volume_ratio": volume_ratio}

    except Exception as e:
        return {"status": "error", "file_path": str(file_path), "message": f"{e}\n{traceback.format_exc()}"}

 
# === 新增模块：期权定制化标签生成 ===
# 您可以将这部分代码添加到 feature_merge_option_raw.py 的末尾
#

def find_benchmark_contract_relaxed(option_chain_df: pd.DataFrame, stock_price: float, target_dte: int = 30):
    """
    [宽松版 v3 - 兼容无 bid/ask 的期权链]
    改进自 find_benchmark_contract:
      ✅ 放宽 DTE 范围: 7~60天
      ✅ DTE 不足则自动 fallback
      ✅ 允许低成交量 (volume>0 或 OI>10)
      ✅ 若无 volume 数据则按最接近平价 strike 选取
      ✅ 优先取与目标 DTE 接近的合约
      ✅ 增加详细日志，便于诊断匹配率
    """
    if option_chain_df.empty:
        logging.debug("find_benchmark_contract_relaxed: 输入期权链为空。")
        return None, None

    ny_tz = "America/New_York"
    try:
        exp_dt = pd.to_datetime(option_chain_df["expiration"], errors="coerce")
        if exp_dt.dt.tz is None:
            exp_dt = exp_dt.dt.tz_localize(ny_tz, ambiguous="infer")
        else:
            exp_dt = exp_dt.dt.tz_convert(ny_tz)

        ts_dt = pd.to_datetime(option_chain_df["timestamp"], errors="coerce")
        if ts_dt.dt.tz is None:
            ts_dt = ts_dt.dt.tz_localize(ny_tz, ambiguous="infer")
        else:
            ts_dt = ts_dt.dt.tz_convert(ny_tz)

        option_chain_df["dte"] = (exp_dt - ts_dt).dt.days
    except Exception as e:
        logging.warning(f"[find_benchmark_contract_relaxed] DTE计算失败: {e}")
        return None, None

    # --- Step 1. 放宽 DTE 范围 ---
    dte_min, dte_max = 7, 60
    chain_in_dte_range = option_chain_df[option_chain_df["dte"].between(dte_min, dte_max)].copy()
    if chain_in_dte_range.empty:
        logging.debug("DTE 7~60 天范围内无合约，放宽至全范围。")
        chain_in_dte_range = option_chain_df.copy()

    # --- Step 2. 计算与现价差距 ---
    chain_in_dte_range["strike_diff"] = (chain_in_dte_range["strike"] - stock_price).abs()
    chain_in_dte_range["dte_diff"] = (chain_in_dte_range["dte"] - target_dte).abs()

    # --- Step 3. 放宽流动性条件 ---
    if "volume" in chain_in_dte_range.columns:
        liquidity_mask = (chain_in_dte_range["volume"].fillna(0) > 0) | (
            chain_in_dte_range.get("open_interest", 0) > 10
        )
        if liquidity_mask.any():
            chain_in_dte_range = chain_in_dte_range[liquidity_mask]
        else:
            logging.debug("流动性过滤后无合约，保留全部。")

    # --- Step 4. 计算 rank 评分 (综合 DTE 接近度 + 平价程度) ---
    chain_in_dte_range["rank"] = (
        chain_in_dte_range["dte_diff"] * 0.5
        + (chain_in_dte_range["strike_diff"] / stock_price * 100)
    )

    if chain_in_dte_range.empty:
        return None, None

    # --- Step 5. Call / Put 各自挑选 ---
    atm_call, atm_put = None, None
    calls = chain_in_dte_range[
        chain_in_dte_range["option_type"].astype(str).str.upper() == "C"
    ]
    if not calls.empty:
        atm_call = calls.loc[calls["rank"].idxmin()]

    puts = chain_in_dte_range[
        chain_in_dte_range["option_type"].astype(str).str.upper() == "P"
    ]
    if not puts.empty:
        atm_put = puts.loc[puts["rank"].idxmin()]

    # --- Step 6. fallback：若仍为空，跨所有 expiration 选最近平价 ---
    if atm_call is None and not option_chain_df.empty:
        logging.debug("未找到 call，跨所有 expiration 寻找最近平价 call。")
        calls_all = option_chain_df[
            option_chain_df["option_type"].astype(str).str.upper() == "C"
        ]
        if not calls_all.empty:
            calls_all["strike_diff"] = (calls_all["strike"] - stock_price).abs()
            atm_call = calls_all.loc[calls_all["strike_diff"].idxmin()]

    if atm_put is None and not option_chain_df.empty:
        puts_all = option_chain_df[
            option_chain_df["option_type"].astype(str).str.upper() == "P"
        ]
        if not puts_all.empty:
            puts_all["strike_diff"] = (puts_all["strike"] - stock_price).abs()
            atm_put = puts_all.loc[puts_all["strike_diff"].idxmin()]

    if atm_call is not None or atm_put is not None:
        logging.debug(
            f"[find_benchmark_contract_relaxed] 成功找到合约 "
            f"(Call={atm_call['expiration'] if atm_call is not None else None}, "
            f"Put={atm_put['expiration'] if atm_put is not None else None})"
        )
    else:
        logging.debug("[find_benchmark_contract_relaxed] 未找到任何平价合约。")

    return atm_call, atm_put



def find_benchmark_contract(option_chain_df: pd.DataFrame, stock_price: float, target_dte: int = 30):
    """
    [最终稳健版 v4 - 修复时区错误]
    在计算DTE之前，强制将 'expiration' 和 'timestamp' 列统一为纽约时区。
    """
    # 诊断点 1: 检查输入的期权链是否为空
    if option_chain_df.empty:
        return None, None

    # --- 1. [核心修正] DTE (到期日) 计算，包含时区统一 ---
    ny_tz = 'America/New_York'
    
    try:
        # a. 确保 expiration 列是 tz-aware
        #    期权到期日通常不带时间，我们假设它代表纽约时区的午夜
        exp_dt = pd.to_datetime(option_chain_df['expiration'], errors='coerce')
        if exp_dt.dt.tz is None:
            exp_dt = exp_dt.dt.tz_localize(ny_tz, ambiguous='infer')
        else:
            exp_dt = exp_dt.dt.tz_convert(ny_tz)

        # b. 确保 timestamp 列是 tz-aware (增加健壮性)
        ts_dt = pd.to_datetime(option_chain_df['timestamp'], errors='coerce')
        if ts_dt.dt.tz is None:
            ts_dt = ts_dt.dt.tz_localize(ny_tz, ambiguous='infer')
        else:
            ts_dt = ts_dt.dt.tz_convert(ny_tz)

        # c. 现在两个都是 tz-aware，可以安全地相减
        option_chain_df['dte'] = (exp_dt - ts_dt).dt.days

    except Exception as e:
        logging.warning(f"find_benchmark_contract: DTE计算失败，时区处理异常: {e}")
        return None, None

    # --- 2. 使用 DTE 范围进行筛选 ---
    dte_min = 7
    dte_max = 60
    chain_in_dte_range = option_chain_df[option_chain_df['dte'].between(dte_min, dte_max)].copy()

    if chain_in_dte_range.empty:
        return None, None

    # --- 3. 在DTE范围内的合约中，找到流动性最好的到期日 ---
    if 'volume' not in chain_in_dte_range.columns or chain_in_dte_range['volume'].sum() < 1:
        # 如果没有成交量数据，则选择最近的到期日
        best_expiration = chain_in_dte_range['expiration'].min()
    else:
        # 否则，选择成交量最大的到期日
        best_expiration = chain_in_dte_range.groupby('expiration')['volume'].sum().idxmax()
    
    chain_for_best_exp = chain_in_dte_range[chain_in_dte_range['expiration'] == best_expiration].copy()

    # --- 4. Strike (行权价) 筛选 ---
    chain_for_best_exp['strike_diff'] = (chain_for_best_exp['strike'] - stock_price).abs()
    
    # --- 5. 分离 Call 和 Put 并找到最近平价的合约 ---
    atm_call, atm_put = None, None
    calls = chain_for_best_exp[chain_for_best_exp['option_type'] == 'C']
    if not calls.empty:
        atm_call = calls.loc[calls['strike_diff'].idxmin()]

    puts = chain_for_best_exp[chain_for_best_exp['option_type'] == 'P']
    if not puts.empty:
        atm_put = puts.loc[puts['strike_diff'].idxmin()]

    return atm_call, atm_put


def _calculate_gamma_pnl_standard(
    S0: float, 
    S1: float, 
    gamma0: float, 
    delta0: float, 
    T0_days: float, 
    T1_days: float
) -> float:
    """
    计算 Delta 对冲的期权 Gamma P&L (忽略时间价值和利率变化)。
    该公式基于 Delta-Gamma 逼近，用于估计对冲后的损益。

    公式简化版 (P&L ≈ Gamma * (dS)^2 / 2 - Theta * dT):
    这里我们专注于 Delta 和 Gamma 对冲的损益，忽略 Theta 项（Theta P&L 通常是另一个标签）。

    Delta P&L: delta0 * (S1 - S0)
    Gamma P&L (二次项): 0.5 * gamma0 * (S1 - S0)^2
    
    我们计算的是期权在 t0 持有并对冲后，在 t1 时的理论损益。
    
    参数:
        S0 (float): t0 时的标的资产价格 (Stock Close)。
        S1 (float): t1 时的标的资产价格 (Future Stock Close)。
        gamma0 (float): t0 时的期权 Gamma 值。
        delta0 (float): t0 时的期权 Delta 值。
        T0_days (float): t0 时的剩余天数 (DTE)。
        T1_days (float): t1 时的剩余天数 (DTE)。
        
    返回:
        float: Gamma P&L (二次项)
    """
    
    # 标的资产价格变化
    dS = S1 - S0
    
    # --- Gamma P&L (二次项) ---
    # Gamma 捕捉了 Delta 对冲后的剩余二次收益/损失
    # 核心公式: 0.5 * Gamma * (dS)^2
    gamma_pnl = 0.5 * gamma0 * (dS ** 2)
    
    # 注意:
    # 1. 完整的 P&L 还应包含 Theta 损益: - Theta0 * (T0_days - T1_days) / 365
    # 2. 我们这里只返回 Gamma P&L 部分，因为它通常是 Gamma 对冲标签的目标。
    
    return float(gamma_pnl)

import concurrent.futures # 确保这个 import 在文件顶部

def add_option_labels_to_all_files(config: dict, symbols_to_process: list):
    """
    [主协调函数 v4 - 数据库驱动版]
    根据提供的股票列表，查找并并行处理所有对应的特征文件以添加期权标签。

    Args:
        config (dict): 特征配置文件。
        symbols_to_process (list): 从数据库查询到的需要处理的股票代码列表。
    """
    if not symbols_to_process:
        logging.warning("需要处理的股票列表为空，任务终止。")
        return

    # --- 1. 根据股票列表，动态构建文件任务列表 ---
    logging.info(f"正在为数据库中查询到的 {len(symbols_to_process)} 个股票搜寻特征文件...")
    all_feature_files = []
    for symbol in tqdm(symbols_to_process, desc="搜寻文件"):
        symbol_dir = OUTPUT_FEATURES_DIR / symbol
        if symbol_dir.is_dir():
            # 使用 rglob (**) 来递归查找所有子目录中的 .parquet 文件
            all_feature_files.extend(list(symbol_dir.rglob('*.parquet')))
    
    if not all_feature_files:
        logging.warning(f"未能为指定的股票列表在目录 {OUTPUT_FEATURES_DIR} 中找到任何特征文件。")
        return

    logging.info(f"--- 共找到 {len(all_feature_files)} 个文件，开始添加期权定制标签 (向量化版) ---")
    
    # --- 2. 并行处理找到的文件 (核心逻辑不变) ---
    worker_func = partial(process_option_labels_file_vectorized, config=config)
    results = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_path = {executor.submit(worker_func, path): path for path in all_feature_files}
        
        pbar = tqdm(concurrent.futures.as_completed(future_to_path), total=len(all_feature_files), desc="添加期权标签")
        
        for future in pbar:
            path = future_to_path[future]
            try:
                symbol = path.parts[path.parts.index(OUTPUT_FEATURES_DIR.name) + 1]
                pbar.set_postfix_str(f"已完成: {symbol}")
                result = future.result()
                results.append(result)
            except Exception as exc:
                logging.error(f"处理 '{path}' 时子进程产生异常: {exc}")
                results.append({"status": "error", "file_path": str(path), "message": str(exc)})

    # --- 3. 结果分析与统计 (核心逻辑不变) ---
    failures = [res for res in results if res['status'] == 'error']
    successes = [res for res in results if res['status'] == 'success']
    skipped_count = len([res for res in results if res['status'] == 'skipped'])
    
    success_percentages = [res['success_percentage'] for res in successes if 'success_percentage' in res]
    
    print("\n" + "="*80)
    print("--- 期权标签添加处理完成 ---")
    print(f"总任务数 (文件数): {len(all_feature_files)}")
    print(f"  - 成功处理文件数: {len(successes)}")
    print(f"  - 失败文件数: {len(failures)}")
    print(f"  - 跳过文件数 (无数据或RTH不足): {skipped_count}")
    
    if success_percentages:
        avg_perc = np.mean(success_percentages)
        print(f"\n--- '成功找到未来合约' 的平均成功率: {avg_perc:.2f}% ---")
    
    print("="*80)

    if failures:
        log_file = "option_label_failures.log"
        with open(log_file, "w", encoding='utf-8') as f:
            for failure in failures:
                f.write(f"文件: {failure['file_path']}\n错误: {failure['message']}\n---\n")
        print(f"\n失败详情已写入日志文件: {log_file}")

def _calculate_total_hedged_pnl(
    S0: float, 
    S1: float, 
    gamma0: float, 
    delta0: float, 
    theta0: float,  # 新增: t0 时的期权 Theta 值
    T0_days: float, 
    T1_days: float
) -> float:
    """
    计算 Delta-Gamma 对冲后的总 P&L (Gamma P&L + Theta P&L)。
    """
    
    # 标的资产价格变化
    dS = S1 - S0
    
    # 时间变化 (天数)
    dT_days = T0_days - T1_days
    
    # 1. Gamma P&L (二次项)
    gamma_pnl = 0.5 * gamma0 * (dS ** 2)
    
    # 2. Theta P&L (时间衰减项)
    # Theta 通常是基于年或日计算的。假设 theta0 是日变动：
    theta_pnl = theta0 * dT_days  # 如果 theta0 是日变动率
    
    # 如果 theta0 是年变动率 (常见):
    # theta_pnl = theta0 * (dT_days / 365.0) 
    
    # 假设您的 theta0 是每秒/每分钟的，请根据您的数据源调整单位。
    # 根据标准惯例，这里我们假设 theta0 是日收益率。
    
    total_hedged_pnl = gamma_pnl + theta_pnl
    
    return float(total_hedged_pnl)
#
# === 请用这个新版本完整替换旧的 process_option_labels_file 函数 ===
#

def _calculate_gamma_pnl_ratio(gamma_pnl: float, mid_price_0: float) -> float:
    """
    计算 Gamma P&L 与期权初始价格的比率 (label_gamma_pnl_std)。
    """
    if mid_price_0 > 1e-9:
        return gamma_pnl / mid_price_0
    return 0.0


def process_option_labels_file_vectorized(feature_file_path: Path, config: dict):
    """
    [最终合并版 V20 - 严谨向量化版]
    核心改进：
    1. Tolerance 收紧：T+k 匹配容忍度改为 5 分钟，防止使用陈旧数据。
    2. 数值维稳：Gamma PnL 标准化时过滤低价合约 (<$0.10)，防止梯度爆炸。
    3. 时区对齐：强制统一 NY 时区。
    """
    ny_tz = 'America/New_York'
    try:
        # --- 1. 加载数据 ---
        stock_symbol = feature_file_path.parts[-5]
        year_month = feature_file_path.stem
        
        feature_df = pd.read_parquet(feature_file_path)
        if feature_df.empty: 
            return {"status": "skipped", "message": "股票文件为空"}
        
        # 构造期权文件路径
        option_path = OPTION_MONTHLY_DIR / stock_symbol / 'standard' / f"{year_month}.parquet"
        if not option_path.exists(): 
            return {"status": "skipped", "message": "期权文件不存在"}
        
        option_df = pd.read_parquet(option_path)
        if option_df.empty: 
            return {"status": "skipped", "message": "期权文件为空"}

        # --- 2. 预处理列名与格式 ---
        rename_map = {
            'ticker': 'contract_symbol',
            'expiration_date': 'expiration',
            'contract_type': 'option_type',
            'strike_price': 'strike'
        }
        option_df.rename(columns={k: v for k, v in rename_map.items() if k in option_df.columns}, inplace=True)

        # 强制时间戳统一处理 (Stock & Option)
        for df in [feature_df, option_df]:
            if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            if df['timestamp'].dt.tz is None:
                df['timestamp'] = df['timestamp'].dt.tz_localize(ny_tz, ambiguous='infer')
            else:
                df['timestamp'] = df['timestamp'].dt.tz_convert(ny_tz)
            
            df.sort_values('timestamp', inplace=True)

        # 准备 ATM IV (用于后续 Spread 计算)
        if 'options_struc_atm_iv' in feature_df.columns:
            feature_df['robust_atm_iv'] = feature_df['options_struc_atm_iv'].ffill()
        else:
            feature_df['robust_atm_iv'] = np.nan

        # 处理期权链数据 (DTE 计算)
        if 'expiration' in option_df.columns:
             if not pd.api.types.is_datetime64_any_dtype(option_df['expiration']):
                 option_df['expiration'] = pd.to_datetime(option_df['expiration'])
             
             if option_df['expiration'].dt.tz is None:
                 option_df['expiration'] = option_df['expiration'].dt.tz_localize(ny_tz)
             else:
                 option_df['expiration'] = option_df['expiration'].dt.tz_convert(ny_tz)
             
             # 计算 DTE (天)
             option_df['dte'] = (option_df['expiration'] - option_df['timestamp']).dt.total_seconds() / 86400.0
        else:
             return {"status": "error", "message": "期权数据缺少 expiration 列"}

        # --- 3. 筛选 Call 且 7 < DTE < 60 ---
        if 'option_type' in option_df.columns:
            # 兼容 'call'/'put' 或 'C'/'P'
            calls_mask = option_df['option_type'].astype(str).str.upper().str.startswith('C')
            dte_mask = option_df['dte'].between(7, 60)
            calls_df = option_df[calls_mask & dte_mask].copy()
        else:
            return {"status": "error", "message": "期权数据缺少 option_type 列"}

        if calls_df.empty: 
            return {"status": "skipped", "message": "无符合条件(DTE 7-60)的 Call 合约"}

        # 确保有 mid_price
        if 'mid_price' not in calls_df.columns:
             if 'ask' in calls_df.columns and 'bid' in calls_df.columns:
                 calls_df['mid_price'] = (calls_df['ask'] + calls_df['bid']) / 2.0
             else:
                 calls_df['mid_price'] = calls_df.get('close', 0.0)

        # --- 4. 向量化寻找 ATM 合约 (T0) ---
        
        # 防止 merge 后出现 stock_close_x, stock_close_y 冲突
        if 'stock_close' in calls_df.columns:
            calls_df.drop(columns=['stock_close'], inplace=True)

        # 4.1 将股票价格 Merge 进期权表 (Backward search)
        calls_df = pd.merge_asof(
            calls_df, 
            feature_df[['timestamp', 'close']].rename(columns={'close': 'stock_close'}), 
            on='timestamp', 
            direction='backward'
        )
        
        # 4.2 计算与平价的距离 (Strike Diff)
        calls_df['diff'] = (calls_df['strike'] - calls_df['stock_close']).abs()
        
        # 4.3 排序: 优先同一时间点，其次 Diff 最小 (ATM)，再次 Volume 最大 (流动性)
        calls_df.sort_values(['timestamp', 'diff', 'volume'], ascending=[True, True, False], inplace=True)
        
        # 4.4 去重: 每个时间点只留第一条 (即最佳 ATM)
        atm_chain = calls_df.drop_duplicates(subset=['timestamp'], keep='first')
        
        # 4.5 准备合并回主表的数据
        # 选择需要的列并重命名为 t0_ 前缀
        target_cols = ['contract_symbol', 'iv', 'gamma', 'delta', 'theta', 'mid_price', 'dte']
        valid_target_cols = [c for c in target_cols if c in atm_chain.columns]
        
        atm_chain_renamed = atm_chain[['timestamp'] + valid_target_cols].rename(
            columns={c: f"t0_{c}" for c in valid_target_cols}
        )
        
        # 4.6 合并 T0 合约信息回 feature_df
        feature_df = pd.merge_asof(feature_df, atm_chain_renamed, on='timestamp', direction='backward')

        # --- 5. 向量化查找未来合约状态 (T+k) ---
        k = config.get('parameters', {}).get('labeling', {}).get('k_option', 10)
        
        # 计算未来的时间戳和价格
        feature_df['target_ts'] = feature_df['timestamp'].shift(-k)
        feature_df['future_stock_close'] = feature_df['close'].shift(-k)
        
        # 只有当 T0 找到了合约，且 T+k 时间有效时，才需要查找
        if 't0_contract_symbol' not in feature_df.columns:
             return {"status": "skipped", "message": "未能匹配到任何 ATM 合约"}

        valid_mask = feature_df['t0_contract_symbol'].notna() & feature_df['target_ts'].notna()
        
        # 构造查询表
        query_df = feature_df.loc[valid_mask, ['target_ts', 't0_contract_symbol']].copy()
        query_df = query_df.sort_values('target_ts')
        query_df['orig_index'] = query_df.index
        
        # 构造查找库 (Lookup Table)
        option_lookup_cols = ['timestamp', 'contract_symbol', 'iv', 'mid_price']
        valid_lookup_cols = [c for c in option_lookup_cols if c in option_df.columns]
        option_lookup = option_df[valid_lookup_cols].sort_values('timestamp')
        
        # [核心优化] 执行 Merge AsOf 匹配未来合约
        # Tolerance 设为 5 分钟，防止使用 30 分钟前的陈旧报价
        res_df = pd.merge_asof(
            query_df,
            option_lookup,
            left_on='target_ts',
            right_on='timestamp',
            left_by='t0_contract_symbol',
            right_by='contract_symbol',
            direction='backward',
            tolerance=pd.Timedelta(minutes=5)  # 严格限制回溯时间
        )
        
        # 将结果赋值回 feature_df
        res_df.set_index('orig_index', inplace=True)
        if 'iv' in res_df.columns: feature_df.loc[res_df.index, 'tk_iv'] = res_df['iv']
        if 'mid_price' in res_df.columns: feature_df.loc[res_df.index, 'tk_mid_price'] = res_df['mid_price']
        
        # --- 6. 向量化计算所有标签 ---
        
        # 6.1 Realized Volatility (k steps)
        # 计算 log return
        log_ret = np.log(feature_df['close'] / feature_df['close'].shift(1))
        # 滚动计算 StdDev，并 shift(-k) 对齐到当前时刻
        # 年化因子: sqrt(252 * 390) (假设分钟线)
        annual_factor = np.sqrt(252 * 390)
        feature_df['label_rv_k_steps'] = log_ret.rolling(window=k).std().shift(-k) * annual_factor
        
        # 6.2 Delta IV (Implied Volatility Change)
        if 'tk_iv' in feature_df.columns and 't0_iv' in feature_df.columns:
            feature_df['label_delta_iv'] = feature_df['tk_iv'] - feature_df['t0_iv']
        
        # 6.3 Option Return (Price Change)
        if 'tk_mid_price' in feature_df.columns and 't0_mid_price' in feature_df.columns:
            feature_df['label_option_return'] = (feature_df['tk_mid_price'] - feature_df['t0_mid_price']) / (feature_df['t0_mid_price'] + 1e-9)
        
        # [已回滚] 恢复原始的 IV-RV Spread 逻辑
        base_iv = feature_df.get('robust_atm_iv', pd.Series(np.nan)).fillna(feature_df.get('t0_iv', 0))
        rv_safe = feature_df['label_rv_k_steps'].replace(0, np.nan)
        
        # 🚀 [终极去泄露：切断特征自相关]
        # 标签不再定义为 IV - RV，因为特征里包含 IV，会导致模型直接“复述”当前 IV。
        # 现在的标签直接指向未来的已实现波动率 (RV)。如果模型能预测出这个，才是真 Alpha。
        feature_df['label_iv_rv_spread'] = feature_df['label_rv_k_steps'].clip(0, 5.0).fillna(0.0)
        feature_df['label_iv_rv_ratio'] = np.log1p(feature_df['label_iv_rv_spread'])


        
        # 6.5 Vol Direction (方向性标签)
        # -----------------------------------------------------------
        # [核心修复] 6.5 Vol Direction (方向性标签)
        # 目标: 0 (Down), 1 (Flat), 2 (Up)
        # -----------------------------------------------------------
        threshold = 0.01
        
        # 默认设为 1 (Flat)
        feature_df['label_vol_direction'] = 1.0
        
        # Spread > 0.01 -> 认为波动率溢价高，未来波动率可能下跌回归? 
        # 或者 这里的定义是：IV 比 RV 高 -> 看涨波动率溢价?
        # 通常：
        # Spread > 0 (IV > RV) -> Market prices high volatility (Fear/Event) -> Up (2)
        # Spread < 0 (IV < RV) -> Market prices low volatility (Complacency) -> Down (0)
        
        # 逻辑修改：映射到 2.0 (Up) 和 0.0 (Down)
        feature_df.loc[feature_df['label_iv_rv_spread'] > threshold, 'label_vol_direction'] = 2.0
        feature_df.loc[feature_df['label_iv_rv_spread'] < -threshold, 'label_vol_direction'] = 0.0
        
        # 确保转为整数 (虽然这里存float没问题，但训练时要用LongTensor)
        # -----------------------------------------------------------
        
        # 6.6 Gamma & Theta PnL (Theoretical)
        if 't0_gamma' in feature_df.columns and 't0_theta' in feature_df.columns:
            # 价格变动
            dS = feature_df['future_stock_close'] - feature_df['close']
            
            # Gamma PnL (0.5 * Gamma * dS^2)
            gamma_pnl = 0.5 * feature_df['t0_gamma'] * (dS ** 2)
            
            # Theta PnL (Theta * dt)
            # dt 单位转换为天 (假设 Theta 为每日损耗)
            dt_days = (feature_df['target_ts'] - feature_df['timestamp']).dt.total_seconds() / 86400.0
            theta_pnl = feature_df['t0_theta'] * dt_days
            
            total_pnl = gamma_pnl + theta_pnl
            feature_df['label_gamma_pnl'] = total_pnl
            feature_df['label_theta_pnl'] = theta_pnl
            
            # [核心优化] Standardized Gamma PnL (归一化)
            # [已回滚] 恢复原始的 Gamma PnL 归一化逻辑
            t0_mid = feature_df.get('t0_mid_price', 0.0)
            safe_mid = t0_mid.where(t0_mid > 0.10, np.nan) 
            feature_df['label_gamma_pnl_std'] = total_pnl / safe_mid
            feature_df['label_gamma_pnl_std'] = feature_df['label_gamma_pnl_std'].clip(-5.0, 5.0).fillna(0.0)
        
        # 6.7 Future Stock Return
        feature_df['label_future_return'] = (feature_df['future_stock_close'] - feature_df['close']) / feature_df['close']

        # --- 7. 清理与保存 ---
        # 目标标签列
        label_cols = [
            'label_delta_iv', 'label_option_return', 'label_iv_rv_spread', 'label_iv_rv_ratio',
            'label_gamma_pnl', 'label_theta_pnl', 'label_vol_direction', 
            'label_gamma_pnl_std', 'label_future_return', 'label_rv_k_steps'
        ]
        
        # 仅保留存在的列，并填充 NaN
        existing_labels = [c for c in label_cols if c in feature_df.columns]
        feature_df[existing_labels] = feature_df[existing_labels].fillna(0.0)
        
        # 删除中间计算产生的临时列
        temp_cols = [c for c in feature_df.columns if c.startswith('t0_') or c.startswith('tk_') or c in ['target_ts', 'future_stock_close', 'robust_atm_iv', 'orig_index']]
        feature_df.drop(columns=temp_cols, inplace=True, errors='ignore')
        
        feature_df.to_parquet(feature_file_path, index=False, compression='zstd', compression_level=9)
        
        return {"status": "success", "file_path": str(feature_file_path), "count": len(feature_df)}

    except Exception as e:
        import traceback
        return {"status": "error", "file_path": str(feature_file_path), "message": f"{e}\n{traceback.format_exc()}"}
 

 


def process_ohlc_file(feature_file_path: Path) -> str:
    """
    Worker function: Loads a feature file and its corresponding raw resampled file,
    merges the OHLC columns into the feature file, and saves it.
    """
    try:
        # 1. Derive the path to the original resampled data file
        raw_file_path = Path(str(feature_file_path).replace(str(OUTPUT_FEATURES_DIR), str(STOCK_RESAMPLED_DIR)))

        if not raw_file_path.exists():
            return f"[Skipped] Raw data file not found for {feature_file_path.name}"

        # 2. Load both the feature data and the raw data
        feature_df = pd.read_parquet(feature_file_path)
        raw_df = pd.read_parquet(raw_file_path)

        if 'timestamp' not in feature_df.columns or 'timestamp' not in raw_df.columns:
            return f"[Error] Timestamp column missing in {feature_file_path.name} or its raw counterpart."

        # Ensure timestamp is in a consistent format for merging
        feature_df['timestamp'] = pd.to_datetime(feature_df['timestamp'])
        raw_df['timestamp'] = pd.to_datetime(raw_df['timestamp'])

        # 3. Prepare the OHLC data for merging
        ohlc_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        
        # Check if raw data has all necessary columns
        if not all(col in raw_df.columns for col in ohlc_columns):
             return f"[Error] Raw data file {raw_file_path.name} is missing one or more OHLC columns."

        ohlc_df = raw_df[ohlc_columns]

        # 4. Remove any old/stale OHLC columns from the feature file to prevent merge conflicts
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in feature_df.columns:
                feature_df = feature_df.drop(columns=[col])

        # 5. Merge the OHLC data into the feature dataframe
        # Using a left merge ensures we keep all rows from the feature file
        updated_df = pd.merge(feature_df, ohlc_df, on='timestamp', how='left')

        # 6. Fill any potential NaN values that might result from the merge
        # This is a safety measure, though a left merge on a shared key should not create NaNs
        updated_df[['open', 'high', 'low', 'close', 'volume']] = updated_df[['open', 'high', 'low', 'close', 'volume']].ffill() 

        # 7. Overwrite the original feature file with the updated data
        updated_df.to_parquet(feature_file_path, index=False, compression='zstd', compression_level=9)

        return f"[Success] Added OHLC to {feature_file_path.name}"

    except Exception as e:
        import traceback
        return f"[Error] Failed to process {feature_file_path.name}: {e}\n{traceback.format_exc()}"


def update_OHLC_in_files():
    """
    Main orchestrator function: Finds all generated feature files and uses a
    process pool to add back the essential OHLC columns from the raw resampled data.
    """
    tasks = list(OUTPUT_FEATURES_DIR.glob('*/*/*/*/*.parquet'))
    if not tasks:
        logging.warning("In update_OHLC_in_files: No feature files found in the output directory.")
        return

    logging.info(f"Found {len(tasks)} feature files to update with OHLC data...")

    # Use a process pool for parallel execution
    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        results = list(tqdm(executor.map(process_ohlc_file, tasks), total=len(tasks), desc="Updating OHLC Data"))

    # Log the results
    failures = [res for res in results if res.startswith("[Error]")]
    success_count = len(results) - len(failures)

    print("\n" + "="*80)
    print("OHLC Update Complete. Summary:")
    print(f"  - Successfully updated files: {success_count}")
    print(f"  - Failed files: {len(failures)}")
    print("="*80 + "\n")

    if failures:
        log_file = "ohlc_update_failures.log"
        with open(log_file, "w", encoding='utf-8') as f:
            for failure_log in failures:
                f.write(f"{failure_log}\n")
        print(f"Details for the {len(failures)} failed updates have been written to {log_file}")

def main():
    # --- 1. 加载配置和任务 (与之前相同) ---
    with open(CONFIG_FILE, 'r') as f:
        config = json.load(f)

    db_path = "/home/kingfang007/notebook/stocks.db"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor() #level IN ('sp500', 'nq100', 'spnq', 'nq')
    from config import TARGET_SYMBOLS
         # 动态生成占位符并执行查询
    placeholders = ','.join(['?'] * len(TARGET_SYMBOLS))
    query = f"SELECT  distinct  symbol  FROM stocks_us WHERE symbol IN ({placeholders})"
    cursor.execute(query, TARGET_SYMBOLS)
    #cursor.execute("SELECT DISTINCT symbol FROM stocks_us WHERE  level IN ( 'spnq' )  ")
    symbols = [row[0] for row in cursor.fetchall()]
    conn.close()
    
    date_range = pd.date_range(start="2022-03-01", end="2026-03-17", freq='MS')
    tasks = [(symbol, date.strftime('%Y-%m')) for symbol in symbols for date in date_range]
    
    # --- 2. 【新增】启动前诊断检查 ---
    if not tasks:
        logging.error("未能根据数据库和日期范围创建任何任务。")
        return
    logging.info(f"共对 {len(symbols)} 个股票创建任务。")
    logging.info(f"共创建 {len(tasks)} 个 (股票, 年月) 任务。")
    logging.info("--- 正在执行启动前诊断 ---")
    
    first_symbol, first_ym = tasks[0]
    found_any_files = False
    
    # 根据配置文件，检查第一个任务是否能找到任何一个源文件
    for session, time_ranges in config.get('resample_freq', {}).items():
        for time_range, resolutions in time_ranges.items():
            for res in resolutions:
                expected_file = STOCK_RESAMPLED_DIR / first_symbol / session / time_range / res / f"{first_ym}.parquet"
                if expected_file.exists():
                    logging.info(f"  ✅ 诊断成功: 找到一个示例源文件 -> {expected_file}")
                    found_any_files = True
                    break
            if found_any_files: break
        if found_any_files: break

    if not found_any_files:
        logging.error("  ❌ 诊断失败: 未能为第一个任务找到任何源文件。这意味着很可能所有任务都会失败。")
        logging.error("  请仔细检查以下几点:")
        logging.error(f"    1. 您的基础重采样数据目录是否正确: '{STOCK_RESAMPLED_DIR}'")
        logging.error(f"    2. 您的 `feature.json` 中的 `resample_freq` 配置，是否与实际的目录结构 ({first_symbol}/<session>/<time_range>/<resolution>) 匹配。")
        logging.error("    程序将退出，不进行后续处理。")
        return
    
    logging.info("--- 诊断检查通过，至少部分源文件路径配置正确。开始并行处理... ---")
    
    # --- 3. 并行处理 (与之前相同) ---
    worker_func = partial(process_stock_month, config=config)
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        results = list(tqdm(executor.map(worker_func, *zip(*tasks)), total=len(tasks), desc="Generating Features"))

    # --- 4. 【新增】更详细的结果分类统计 ---
    failures = []
    successes = []
    warnings = [] # 用于捕获那些成功运行但未处理任何文件的任务

    for res in results:
        if isinstance(res, str) and res.startswith("[错误]"):
            failures.append(res)
        elif isinstance(res, str) and res.startswith("[警告]"):
            warnings.append(res)
        elif isinstance(res, str) and res.startswith("[成功]"):
            successes.append(res)

    print("\n" + "="*80)
    print("处理完成！最终报告：")
    print(f"  - [成功] 完成并处理了文件的任务数: {len(successes)}")
    print(f"  - [警告] 成功运行但未找到文件的任务数: {len(warnings)}")
    print(f"  - [错误] 发生严重错误的任务数: {len(failures)}")
    print("="*80 + "\n")

    if warnings:
        # 只打印少量警告示例，避免刷屏
        print("--- '未找到文件'警告示例 (表示配置与部分数据不匹配) ---\n")
        for warning_log in warnings[:5]:
            print(f"{warning_log}\n")
        if len(warnings) > 5:
            print(f"...以及其他 {len(warnings) - 5} 条警告...\n")
    
    if failures:
        log_file = "feature_generation_failures.log"
        print(f"发现 {len(failures)} 个失败任务。详细错误信息已写入日志文件: {log_file}")
        with open(log_file, "w", encoding='utf-8') as f:
            for i, failure_log in enumerate(failures):
                f.write(f"--- 失败任务 #{i+1} ---\n")
                f.write(f"{failure_log}\n\n")

def add_option_labels_data():
    """
    程序主入口：连接数据库获取股票列表，然后调用标签生成函数。
    """
    # --- 1. 加载配置 ---
    with open(CONFIG_FILE, 'r') as f:
        config = json.load(f)

    # --- 2. 从 SQLite 数据库获取股票列表 ---
    logging.info(f"正在从数据库 {DB_PATH} 查询股票列表...")
    try:
        conn = sqlite3.connect(DB_PATH)
        # *** 在这里修改你的 SQL 查询来选择特定的股票 ***
        # 示例: 选择所有在 'nq100' 或 'sp500' 列表中的股票
        cursor = conn.cursor() #level IN ('sp500', 'nq100', 'spnq', 'nq')
        from config import TARGET_SYMBOLS
             # 动态生成占位符并执行查询
        placeholders = ','.join(['?'] * len(TARGET_SYMBOLS))
        query = f"SELECT distinct symbol  FROM stocks_us WHERE symbol IN ({placeholders})"
        cursor.execute(query, TARGET_SYMBOLS)
        
        symbols = [row[0] for row in cursor.fetchall()]
        conn.close()
        if not symbols:
            logging.warning("数据库查询未返回任何股票代码，程序将退出。")
            return
            
        logging.info(f"从数据库成功查询到 {len(symbols)} 个股票代码。")
        add_option_labels_to_all_files(config=config, symbols_to_process=symbols)

    except Exception as e:
        logging.error(f"连接或查询数据库时发生错误: {e}")
        return

    # --- 3. 调用核心处理函数 ---
    

def process_log_transform_label(file_path: Path) -> dict:
    """
    【新】工作函数：
    加载单个特征文件，读取 'label_iv_rv_ratio' 列，
    将其转换为对数形式 (log(1 + x))，并覆盖保存。
    """
    try:
        df = pd.read_parquet(file_path)
        
        target_col = 'label_iv_rv_ratio'

        if df.empty or target_col not in df.columns:
            return {"status": "skipped", "file_path": str(file_path), "message": f"文件为空或缺少 '{target_col}' 列。"}

        # --- 核心对数转换逻辑 ---
        
        # 1. 我们的标签 (IV/RV) - 1，范围可以从 -1 (即 IV=0) 到正无穷。
        #    我们想计算 log(IV/RV)。
        #    使用 np.log1p(x) = log(1 + x)
        #    log(1 + [(IV/RV) - 1]) = log(IV/RV)
        
        # 2. 计算 log(1 + x)
        transformed_values = np.log1p(df[target_col])
        # 使用 errstate 告诉 Numpy："这行代码产生的除零警告我知道，别打印出来"
        with np.errstate(divide='ignore'):
            transformed_values = np.log1p(df[target_col])
        
        # 3. 处理边界情况：
        #    - 如果原始值是 -1 (IV=0), log1p(-1) 会得到 -inf。
        #    - 如果原始值是 inf, log1p(inf) 会得到 inf。
        #    我们将这些无穷值替换为 0.0，这是一个中性的值。
        transformed_values.replace([np.inf, -np.inf], 0.0, inplace=True)
        
        # 4. 确保没有 NaN (尽管 log1p 不应该产生 NaN)
        transformed_values.fillna(0.0, inplace=True)
        
        # 5. 将变换后的值写回 DataFrame
        df[target_col] = transformed_values

        # 6. 覆盖保存文件
        df.to_parquet(file_path, index=False, compression='zstd', compression_level=9)
        
        return {"status": "success", "file_path": str(file_path)}

    except Exception as e:
        import traceback
        return {"status": "error", "file_path": str(file_path), "message": f"发生意外异常: {e}\n{traceback.format_exc()}"}


def update_iv_rv_ratio_in_files(config: dict):
    """
    【新】管理者函数（模仿 update_cat_features_in_files）：
    遍历所有特征文件，并并行调用 'process_log_transform_label'
    来将 'label_iv_rv_ratio' 列就地转换为其对数形式。
    """
    
    # 1. 查找所有特征文件
    tasks = list(OUTPUT_FEATURES_DIR.glob('*/*/*/*/*.parquet'))
    if not tasks:
        logging.warning("在输出目录中没有找到任何特征文件可供更新。")
        return

    logging.info(f"找到 {len(tasks)} 个特征文件，开始批量将 'label_iv_rv_ratio' 转换为对数...")

    results = []
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # 工作函数 process_log_transform_label 不需要 config，所以可以直接 map
        results_iterator = executor.map(process_log_transform_label, tasks)
        
        pbar = tqdm(results_iterator, total=len(tasks), desc="转换IV/RV标签为对数")
        results = list(pbar) # 等待所有任务完成

    # 3. 结果统计 (与 update_cat_features_in_files 相同)
    failures = [res for res in results if res['status'] == 'error']
    success_count = sum(1 for res in results if res['status'] == 'success')
    skipped_count = sum(1 for res in results if res['status'] == 'skipped')

    if failures:
        log_file = "label_log_transform_failures.log"
        with open(log_file, "w", encoding='utf-8') as f:
            for failure_dict in failures:
                f.write(f"文件路径: {failure_dict['path']}\n")
                f.write(f"  - 错误信息: {failure_dict['message']}\n")
                f.write("-" * 50 + "\n")
        print(f"\n处理完成，有 {len(failures)} 个任务失败。详情请见 {log_file}。")
    else:
        print("\n处理完成，所有标签均已成功转换为对数！")

    logging.info(f"对数转换摘要: "
        f"成功 {success_count} 个文件, "
        f"跳过 {skipped_count} 个文件, "
        f"错误 {len(failures)} 个文件。")


# 添加到 feature_merge_option_raw.py

def process_iv_rv_spread_file(file_path: Path, config: dict) -> dict:
    """
    【新】工作函数：计算 iv_rv_spread_feature 并更新文件。
    逻辑：Spread = atm_iv - Annualized(garman_klass_vol)
    """
    try:
        df = pd.read_parquet(file_path)
        
        # 1. 检查必要列
        required_cols = ['atm_iv', 'garman_klass_vol']
        if df.empty or not all(col in df.columns for col in required_cols):
            return {"status": "skipped", "file_path": str(file_path), "message": "缺少 atm_iv 或 garman_klass_vol 列"}

        # 2. 推断分辨率以确定年化因子
        # 尝试从路径解析分辨率 (假设路径结构包含分辨率文件夹，如 /1min/, /5min/)
        # 如果路径中找不到，默认按 1min 处理 (最保守)
        annual_factor = np.sqrt(252 * 390) # 默认 1min: sqrt(252天 * 390分钟)
        
        path_str = str(file_path)
        if '/5min/' in path_str:
            annual_factor = np.sqrt(252 * 78) # 5min: 390/5 = 78
        elif '/10s/' in path_str:
            annual_factor = np.sqrt(252 * 2340)
        elif '/30s/' in path_str:
            annual_factor = np.sqrt(252 * 780)
        elif '/5s/' in path_str:
            annual_factor = np.sqrt(252 * 4680)
            
        # 3. 计算 Spread
        # 注意：garman_klass_vol 在之前的 compute_features 中是滚动平均后的 per-step volatility
        # 我们将其年化以便与 atm_iv (通常是年化值) 进行比较
        
        rv_annualized = df['garman_klass_vol'] * annual_factor
        
        # Spread: 如果 IV 远大于 RV，通常意味着期权溢价高（恐慌或预期大波动）
        # 对于 UOA 策略，我们寻找 Spread 突然扩大的时刻
        df['iv_rv_spread_feature'] = df['atm_iv'] - rv_annualized
        
        # 填充 NaN (通常在开头)
        df['iv_rv_spread_feature'] = df['iv_rv_spread_feature'].ffill().fillna(0.0)

        # 4. 覆盖保存
        df.to_parquet(file_path, index=False, compression='zstd', compression_level=9)
        
        return {"status": "success", "file_path": str(file_path)}

    except Exception as e:
        import traceback
        return {"status": "error", "file_path": str(file_path), "message": f"{e}\n{traceback.format_exc()}"}


def update_iv_rv_spread_in_files(config: dict):
    """
    【新】管理者函数：批量更新 iv_rv_spread_feature。
    """
    tasks = list(OUTPUT_FEATURES_DIR.glob('*/*/*/*/*.parquet'))
    if not tasks:
        logging.warning("在输出目录中没有找到任何特征文件。")
        return

    logging.info(f"找到 {len(tasks)} 个文件，开始计算 IV-RV Spread (基于 Garman-Klass)...")

    worker_func = partial(process_iv_rv_spread_file, config=config)
    
    # 统计计数器
    success_count = 0
    skipped_count = 0
    failures = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # 使用 tqdm 显示进度
        results = list(tqdm(executor.map(worker_func, tasks), total=len(tasks), desc="Updating IV-RV Spread"))
    
    for res in results:
        if res['status'] == 'success':
            success_count += 1
        elif res['status'] == 'skipped':
            skipped_count += 1
        elif res['status'] == 'error':
            failures.append(res)

    # 打印摘要
    if failures:
        log_file = "iv_rv_update_failures.log"
        with open(log_file, "w", encoding='utf-8') as f:
            for fail in failures:
                f.write(f"File: {fail['file_path']}\nError: {fail['message']}\n{'-'*50}\n")
        print(f"\n⚠️ 有 {len(failures)} 个文件处理失败，详情见 {log_file}")
    
    logging.info(f"更新完成: 成功 {success_count}, 跳过 {skipped_count}, 失败 {len(failures)}")
    logging.info(f"新特征 'iv_rv_spread_feature' 已添加。")

# 添加到 feature_merge_option_raw.py
 

def calculate_slope_vectorized(series, window=30):
    """
    高效计算滚动线性回归斜率 (Vectorized Rolling Slope)
    逻辑：Slope = Cov(x, y) / Var(x)
    x 是固定的序列 [0, 1, ..., window-1]
    """
    y = series.values
    n = len(y)
    
    if n < window:
        return pd.Series(np.zeros(n), index=series.index)
    
    # 构造 x (固定)
    x = np.arange(window)
    x_mean = x.mean()
    x_var = ((x - x_mean)**2).sum() # 这是一个常数
    
    # 我们需要计算滚动协方差: Sum((x - x_mean) * (y - y_mean))
    # 但更快的公式是: (Sum(x*y) - n*x_mean*y_mean) / ...
    # 为了利用 pandas 的 rolling，我们拆解公式：
    # Slope ~ Rolling_Cov(x, y)
    
    # 由于 x 是固定的线性增长序列，我们可以用卷积或者简单的加权移动平均来近似
    # 或者直接用 stride_tricks (最高效但复杂)，这里为了稳健使用 pandas rolling apply 的优化版
    
    # 方法 C: 使用预计算的系数 (最快且准确)
    # Slope = Sum(w_i * y_i)
    # w_i = (i - x_mean) / x_var
    weights = (x - x_mean) / x_var
    
    # 使用卷积计算加权和
    # mode='valid' 会导致前 window-1 个变为 NaN，符合预期
    # flip weights 因为卷积是反向的
    slope_values = np.convolve(y, weights[::-1], mode='valid')
    
    # 补齐前面的 NaN
    pad = np.full(window - 1, np.nan)
    full_slope = np.concatenate([pad, slope_values])
    
    return pd.Series(full_slope, index=series.index)

  
  
 
def process_dist_from_vwap_file(file_path: Path):
    """
    【修复版】计算 dist_from_vwap，增加数值稳定性保护，防止 float32 溢出。
    """
    try:
        df = pd.read_parquet(file_path)
        
        # 1. 检查必要的基础列
        required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        if df.empty or not all(col in df.columns for col in required_cols):
            return {"status": "skipped", "path": str(file_path), "message": "文件为空或缺少 OHLCV 基础列"}

        # 确保时间戳格式正确并排序 (计算时序指标必须排序)
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.sort_values('timestamp', inplace=True)

        # 2. 计算或获取 VWAP (日内 VWAP)
        if 'vwap' not in df.columns:
            # 使用 float64 进行中间计算以保证精度
            df['price_x_vol'] = (df['close'] * df['volume']).astype('float64')
            df['vol_f64'] = df['volume'].astype('float64')
            
            # 按日期分组计算累积值
            daily_groups = df.groupby(df['timestamp'].dt.date)
            cum_value = daily_groups['price_x_vol'].cumsum()
            cum_volume = daily_groups['vol_f64'].cumsum()
            
            # 防止除零 (使用较大的 epsilon 1e-6)
            df['vwap'] = cum_value / (cum_volume + 1e-6)
            
            # 填补可能的 NaN (例如开盘第一笔成交前) 为 Close
            df['vwap'] = df['vwap'].fillna(df['close'])

        # 3. 计算 ATR (Period=14)
        # 计算 True Range
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.amax(ranges.to_numpy(), axis=1)
        
        # 计算 EMA
        atr = pd.Series(true_range, index=df.index).ewm(span=14, adjust=False, min_periods=1).mean()
        
        # 【关键修复 1】: 防止 ATR 过小导致除法爆炸
        # 即使对于低价股，ATR 也不应小于 1e-6
        atr = atr.fillna(0).replace(0, 1e-6)
        atr = np.maximum(atr, 1e-6) 

        # 4. 计算 dist_from_vwap (归一化距离)
        # 使用 float64 计算
        dist = (df['close'].astype('float64') - df['vwap'].astype('float64')) / atr

        # 【关键修复 2】: 清理 Inf 和 NaN
        dist = dist.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        # 【关键修复 3】: 截断极端值 (Clipping)
        # 逻辑：偏离 VWAP 超过 100 倍 ATR 已经是天文数字，直接截断即可，
        # 这样可以保证数值永远在 float32 的安全范围内。
        # float32 max 是 3.4e38，但我们限制在 +/- 1000 就足够了。
        dist = dist.clip(lower=-1000.0, upper=1000.0)

        # 5. 安全转换为 float32 并保存
        df['dist_from_vwap'] = dist.astype('float32')

        # 保存
        df.to_parquet(file_path, index=False, compression='zstd', compression_level=9)
        
        return {"status": "success", "path": str(file_path)}

    except Exception as e:
        import traceback
        return {"status": "error", "path": str(file_path), "message": f"{e}\n{traceback.format_exc()}"}

def update_dist_from_vwap():
    """
    【新】管理者函数：批量更新所有特征文件中的 'dist_from_vwap' 特征。
    """
    # 扫描所有特征文件
    tasks = list(OUTPUT_FEATURES_DIR.glob('*/*/*/*/*.parquet'))
    if not tasks:
        logging.warning("在输出目录中没有找到任何特征文件。")
        return

    logging.info(f"找到 {len(tasks)} 个文件，开始批量更新 'dist_from_vwap'...")

    success_count = 0
    skipped_count = 0
    failures = []

    # 使用多进程并行处理
    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # map 返回有序结果迭代器
        results = list(tqdm(executor.map(process_dist_from_vwap_file, tasks), total=len(tasks), desc="Updating VWAP Dist"))

    # 统计结果
    for res in results:
        if res['status'] == 'success':
            success_count += 1
        elif res['status'] == 'skipped':
            skipped_count += 1
        elif res['status'] == 'error':
            failures.append(res)

    # 记录失败日志
    if failures:
        log_file = "dist_vwap_update_failures.log"
        with open(log_file, "w", encoding='utf-8') as f:
            for fail in failures:
                f.write(f"File: {fail['path']}\nError: {fail['message']}\n{'-'*50}\n")
        print(f"\n⚠️ 有 {len(failures)} 个文件处理失败，详情见 {log_file}")
    
    logging.info(f"VWAP Dist 更新完成: 成功 {success_count}, 跳过 {skipped_count}, 失败 {len(failures)}")

def process_ker_file(file_path: Path):
    """
    【新】工作函数：为单个文件计算并更新 Kaufman Efficiency Ratio (KER)。
    公式: KER = |Change| / Volatility
          Change = |Close_t - Close_{t-n}|
          Volatility = Sum(|Close_i - Close_{i-1}|) for i in range(n)
    含义: 
        1.0 = 完美的直线趋势
        0.0 = 纯噪音/震荡
    """
    try:
        df = pd.read_parquet(file_path)
        
        # 1. 检查必要列
        if df.empty or 'close' not in df.columns:
            return {"status": "skipped", "path": str(file_path), "message": "文件为空或缺少 'close' 列"}

        # 2. 设置窗口参数
        # 建议使用 30 作为标准窗口，既能捕捉日内趋势，又不会太迟钝
        ker_window = 30

        # 3. 计算 KER
        # 分子: 价格的位移 (Directional Movement)
        change = df['close'].diff(ker_window).abs()
        
        # 分母: 价格走过的路程 (Total Path Length / Volatility)
        # 先计算每一步的变动，然后滚动求和
        volatility = df['close'].diff().abs().rolling(window=ker_window).sum()
        
        # 计算比率 (防止除零)
        ker = change / (volatility + 1e-9)
        
        # 4. 清洗与边界处理
        # 理论上 KER 在 [0, 1] 之间
        ker = ker.replace([np.inf, -np.inf], 0.0).fillna(0.0)
        ker = ker.clip(lower=0.0, upper=1.0)

        # 5. 保存
        df['ker'] = ker.astype('float32')
        
        df.to_parquet(file_path, index=False, compression='zstd', compression_level=9)
        
        return {"status": "success", "path": str(file_path)}

    except Exception as e:
        import traceback
        return {"status": "error", "path": str(file_path), "message": f"{e}\n{traceback.format_exc()}"}

def process_cross_sectional_month(res: str, year_month: str, base_dir: Path) -> str:
    """
    【新】工作函数：按月和分辨率，加载所有股票，计算截面动量 Z-Score 并写回。
    """
    try:
        # 1. 找到所有股票在该月、该分辨率下的文件 (主要针对 regular 时段)
        file_paths = list(base_dir.rglob(f"*/regular/*/{res}/{year_month}.parquet"))
        if not file_paths:
            return f"[跳过] {res} {year_month} 无文件"
        
        # 2. 读取并拼装该月所有股票的收益率特征
        dfs = []
        for p in file_paths:
            try:
                df = pd.read_parquet(p, columns=['timestamp', 'close_log_return'])
                if not df.empty:
                    df['_filepath'] = str(p)  # 记录来源文件路径
                    dfs.append(df)
            except Exception:
                continue
        
        if not dfs:
            return f"[跳过] {res} {year_month} 数据读取失败"
            
        merged_df = pd.concat(dfs, ignore_index=True)
        
        # 3. 计算截面 Z-Score (Cross-Sectional Z-Score)
        # 按 timestamp 分组，计算横截面上的 close_log_return 的 Z-Score
        # clip(-5, 5) 防止单只股票极端暴涨暴跌拉爆均值
        merged_df['cross_sect_mom_z'] = merged_df.groupby('timestamp')['close_log_return'].transform(
            lambda x: (x - x.mean()) / (x.std() + 1e-9)
        ).clip(lower=-5.0, upper=5.0).astype('float32')
        
        # 4. 将计算结果精确写回各个原始文件
        success_count = 0
        for filepath_str, group in merged_df.groupby('_filepath'):
            try:
                original_df = pd.read_parquet(filepath_str)
                # 按照 timestamp 对齐
                group_to_merge = group[['timestamp', 'cross_sect_mom_z']].drop_duplicates('timestamp')
                
                # 删除旧列（防止重复运行时列冲突）
                if 'cross_sect_mom_z' in original_df.columns:
                    original_df.drop(columns=['cross_sect_mom_z'], inplace=True)
                    
                # 左连接合并回原始特征表
                updated_df = pd.merge(original_df, group_to_merge, on='timestamp', how='left')
                # 填充可能因交易时间不完全对齐产生的 NaN
                updated_df['cross_sect_mom_z'] = updated_df['cross_sect_mom_z'].ffill().fillna(0.0)
                
                # 保存覆盖
                updated_df.to_parquet(filepath_str, index=False, compression='zstd', compression_level=9)
                success_count += 1
            except Exception as e:
                logging.error(f"写入截面特征失败 {filepath_str}: {e}")
                
        return f"[成功] {res} {year_month}: 处理了 {success_count} 只股票的截面特征"
        
    except Exception as e:
        import traceback
        return f"[错误] 截面计算 {res} {year_month} 失败: {e}\n{traceback.format_exc()}"

def update_cross_sectional_features():
    """
    【新】管理者函数：批量计算截面动量特征 (Cross-Sectional Momentum Z-Scores)。
    """
    # 扫描目录推断需要处理的 (分辨率, 年月) 组合
    res_ym_set = set()
    for p in OUTPUT_FEATURES_DIR.rglob("*.parquet"):
        res = p.parent.name
        year_month = p.stem
        res_ym_set.add((res, year_month))
        
    tasks = list(res_ym_set)
    if not tasks:
        logging.warning("没有找到特征文件进行截面计算。")
        return

    logging.info(f"找到 {len(tasks)} 个 (分辨率, 年月) 批次，开始跨股票计算截面动量...")

    failures = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # 提交所有年月批次
        futures = [executor.submit(process_cross_sectional_month, res, ym, OUTPUT_FEATURES_DIR) for res, ym in tasks]
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(tasks), desc="Cross-Sectional Z-Score"):
            res = future.result()
            if '[错误]' in res:
                failures.append(res)

    if failures:
        log_file = "cross_sectional_failures.log"
        with open(log_file, "w", encoding='utf-8') as f:
            for fail in failures:
                f.write(f"{fail}\n{'-'*50}\n")
        print(f"\n⚠️ 截面特征处理有失败批次，详情见 {log_file}")
    else:
        logging.info("截面动量特征 (Cross-Sectional Z-Scores) 全部计算完成并成功写入！")

if __name__ == "__main__":
    with open(CONFIG_FILE, 'r') as f:
        config = json.load(f)

    generate_vix_level_global(config)
    #0. 基本指标包括ohlv
    main()

    #1. 更新vix
    update_vol_vix_abs(config)
   
    #2. 更新分类标签，包括月份，小时，美联储会议等 cat features
    update_cat_features_in_files(config)
    
    # #3.三个 label,因为需要ohlv和部分分类标签，所以在后面
    update_new_labels_in_files(config)

    # #4.2  最后生成期权标签
    add_option_labels_data( )

    # 5. 【新增】计算截面动量特征 (最后执行，因为它依赖前置的 log_return)
    update_cross_sectional_features()

    
   
   
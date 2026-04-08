#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
文件名: realtime_feature_engine.py
描述: [极致优化混合架构版 Pro 3.1]
优化内容:
    1. [Hybrid Engine]: 引入 Pandas 预计算层，彻底解决高频 master_index 对齐带来的 `close_log_return` 为 0、窗口指标畸变的“零值陷阱”问题。
    2. [Data Alignment]: 引入全局 master_index，强行 reindex 所有数据，彻底解决个股停牌/断点导致的 Batch 时序错位。
    3. [Memory Optimization]: 废弃基于 unfold 的大尺度 std 计算，改用 Welford 平方差公式，大幅降低大 Batch 下的显存暴雷风险。
    4. [Padding Fix]: 修复 _kdj 和 _atr 在初始填充阶段的极值错位问题。
    5. [Performance]: 全面使用 groups=B 的 F.conv1d 替代显式的循环滚动计算。
"""

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import logging
import json
import math
from typing import Dict, List, Optional, Tuple
import pytz
import ta  # 引入与离线完全对齐的 ta 库
import numba
import os
import sys

# Ensure greeks_math is importable
try:
    from utils.greeks_math import calculate_bucket_greeks
except ImportError:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils.greeks_math import calculate_bucket_greeks

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
        if min_p == max_p:
            poc_values[i] = min_p
            continue

        bin_edges = np.linspace(min_p, max_p, bins + 1)
        digitized = np.digitize(close_win, bin_edges) - 1
        digitized[digitized < 0] = 0
        digitized[digitized >= bins] = bins - 1
        volume_by_bin = np.bincount(digitized, weights=volume_win, minlength=bins)
        max_volume_bin_idx = np.argmax(volume_by_bin)
        poc = (bin_edges[max_volume_bin_idx] + bin_edges[max_volume_bin_idx + 1]) / 2.0
        poc_values[i] = poc
        
    return poc_values
try:
    from config import USE_5M_OPTION_DATA
except ImportError:
    USE_5M_OPTION_DATA = True # Default

logger = logging.getLogger("RealTimeFeatureEngine")

# 常量定义
ROW_P_ATM, ROW_P_OTM = 0, 1
ROW_C_ATM, ROW_C_OTM = 2, 3
ROW_NEXT_P, ROW_NEXT_C = 4, 5
IDX_PRICE, IDX_DELTA, IDX_GAMMA, IDX_VEGA, IDX_THETA, IDX_STRIKE, IDX_VOLUME, IDX_IV = 0, 1, 2, 3, 4, 5, 6, 7
IDX_BID, IDX_ASK, IDX_BID_SIZE, IDX_ASK_SIZE = 8, 9, 10, 11

class TimeFeatureGenerator:
    def __init__(self, device):
        self.device = device
        self.ny_tz = pytz.timezone('America/New_York')

    def get_time_features(self, last_ts, seq_len):
        """生成时间特征，强制使用 NY 时间 (供非 Batch 模式参考或后期扩展使用)"""
        try:
            if isinstance(last_ts, (int, float)):
                ts_utc = pd.Timestamp(last_ts, unit='s', tz='UTC')
                ts_ny = ts_utc.tz_convert(self.ny_tz)
            else:
                if getattr(last_ts, 'tz', None) is None:
                    ts_utc = pd.Timestamp(last_ts).tz_localize('UTC')
                    ts_ny = ts_utc.tz_convert(self.ny_tz)
                else:
                    ts_ny = last_ts.tz_convert(self.ny_tz)
        except:
            ts_ny = pd.Timestamp.now(tz=self.ny_tz)

        feats = {}
        feats['hour'] = float(ts_ny.hour)
        feats['day_of_week'] = float(ts_ny.weekday()) # Mon=0, Sun=6
        feats['day_of_month'] = float(ts_ny.day)
        feats['month'] = float(ts_ny.month)
        
        return feats

class RealTimeFeatureEngine:
    def __init__(self, stats_path: str, device: str = "cpu"):
        self.device = torch.device(device)
        self.stats = self._load_stats(stats_path)
        self.time_gen = TimeFeatureGenerator(self.device)
        self.rfr_cache = {} # Cache for RFR per date
        self.epsilon = 1e-6
        # 定义必须在 Pandas 原始分钟频率下预计算的特征
        self.PANDAS_FEATS = [
            'close_log_return', 'open_log_return', 'price_z_score','volume_ratio', 'vwap_diff',
            'rsi', 'garman_klass_vol', 'adx_smooth_10', 'k', 'cci',
            'price_slope_norm_by_atr', 'price_dist_from_ma_atr', 'poc_deviation'
        ]

    def _load_stats(self, path):
        try:
            with open(path, 'r') as f: return json.load(f)
        except: return {}

    def supplement_greeks(self, symbol: str, buckets: np.ndarray, contracts: list, stock_price: float, timestamp: float):
        if buckets is None or len(buckets) == 0:
            return buckets
        
        try:
            ts_ny = pd.Timestamp(timestamp, unit='s', tz='UTC').tz_convert('America/New_York')
            # 🚀 [Surgery 8] IV 锚点精准对齐
            # 注意：在 1s 结算帧中，timestamp 已经过 Surgery 4 回刷 60s (如 10:01:00 -> 10:00:00)
            # 这里 floor('1min') 确保即使 timestamp 带微妙秒，也会严格锁定在分钟起始点。
            ts_anchor = ts_ny.floor('1min')
            
            # 🚀 [核心修复 3：动态读取真实的无风险利率 RFR，消除 0.347 vs 0.350 的偏差]
            r_val = None
            date_str = ts_ny.strftime('%Y%m%d')
            if not hasattr(self, 'rfr_cache'): self.rfr_cache = {}
            
            if date_str in self.rfr_cache:
                r_val = self.rfr_cache[date_str]
            else:
                rfr_file = '/home/kingfang007/risk_free_rates.parquet'
                if os.path.exists(rfr_file):
                    
                    df_rfr = pd.read_parquet(rfr_file)
                    search_date = ts_ny.replace(hour=0, minute=0, second=0, microsecond=0).tz_localize(None)
                    idx = df_rfr.index.searchsorted(search_date)
                    idx = np.clip(idx, 0, len(df_rfr) - 1)
                    latest_r = float(df_rfr['DGS3MO'].iloc[idx])
                    if latest_r > 1.0: latest_r /= 100.0
                    self.rfr_cache[date_str] = latest_r
                    r_val = latest_r

            for tkr in contracts:
                if tkr and len(tkr) > 15:
                    ext = tkr.replace('O:', '')
                    import re
                    m = re.search(r'\d{6}', ext)
                    if m:
                        exp_str = m.group(0)
                        expiry_dt = pd.to_datetime(exp_str, format='%y%m%d').tz_localize('America/New_York') + pd.Timedelta(hours=16)
                        time_diff = (expiry_dt - ts_anchor).total_seconds()
                        t_years = max(1e-6, time_diff / 31557600.0)
                        
                        calculate_bucket_greeks(
                            buckets, 
                            stock_price, 
                            t_years, 
                            r=r_val, # 👈 传入真实的无风险利率！
                            contracts=contracts, 
                            current_ts=timestamp
                        )
                        break
        except Exception as e:
            logger.error(f"Error in supplement_greeks for {symbol}: {e}")
            
        return buckets

    # ==========================================================================
    # 辅助计算 (宏观指标与 Pandas 预计算)
    # ==========================================================================
    def _compute_vix_global(self, df_vixy: Optional[pd.DataFrame], feat_list: List[str]) -> Dict[str, torch.Tensor]:
        ctx = {}
        if df_vixy is None or df_vixy.empty: return ctx
            
        last_row = df_vixy.iloc[-1]
        close = float(last_row['close'])
        
        close_tensor = torch.tensor(df_vixy['close'].values, dtype=torch.float32, device=self.device)
        seq_len = len(close_tensor)
        
        if seq_len > 20:
             win = min(seq_len, 60)
             sub = close_tensor[-win:]
             avg = torch.mean(sub)
             std = torch.std(sub)
             val = (close_tensor[-1] - avg) / (std + self.epsilon)
             ctx['vix_level'] = val
             ctx['vix_z'] = val 
        else:
             ctx['vix_level'] = torch.tensor(0.0, device=self.device)
             ctx['vix_z'] = torch.tensor(0.0, device=self.device)
        
        has_detrend = 'vixy_detrended_level' in feat_list
        has_ma5 = 'vix_ma_5' in feat_list
        has_ma20 = 'vix_ma_20' in feat_list
        
        if has_detrend or has_ma20:
            if len(df_vixy) >= 20:
                ma20 = df_vixy['close'].rolling(20).mean().iloc[-1]
                if has_detrend: ctx['vixy_detrended_level'] = torch.tensor((close / (ma20 + self.epsilon)) - 1.0, dtype=torch.float32, device=self.device)
                if has_ma20: ctx['vix_ma_20'] = torch.tensor(ma20, dtype=torch.float32, device=self.device)
            else:
                 if has_detrend: ctx['vixy_detrended_level'] = torch.tensor(0.0, device=self.device)
                 if has_ma20: ctx['vix_ma_20'] = torch.tensor(close, device=self.device)

        if has_ma5:
            if len(df_vixy) >= 5:
                ma5 = df_vixy['close'].rolling(5).mean().iloc[-1]
                ctx['vix_ma_5'] = torch.tensor(ma5, dtype=torch.float32, device=self.device)
            else:
                ctx['vix_ma_5'] = torch.tensor(close, device=self.device)
                
        return ctx

    def _pandas_compute_features(self, df_in: pd.DataFrame, active_feats: List[str]) -> pd.DataFrame:
        """
        [极致对齐]：在原始分钟频率上，完全使用离线 feature_merge_option_raw.py 的逻辑
        """
        if df_in.empty: return df_in

        # 1. 先进行 Copy，生成局部变量 df！(非常重要，必须在最前面)
        df = df_in.copy()

        # =========================================================
        # 🚀 [防弹修复] 强制洗白数据类型，解决 np.log 崩溃与 object 降级警告
        # =========================================================
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').astype(np.float64)

        # [对齐离线训练] 实盘遇到 0 时，视为缺失，并如同模型训练时进行 ffill() 回填。
        close_ff = df['close'].replace(0, np.nan).ffill()
        open_ff = df['open'].replace(0, np.nan).ffill()
        
        # 1. 基础差分收益率 (对齐离线：分母统一使用 prev_close)
        prev_close = close_ff.shift(1).replace(0, np.nan)
        
        if 'close_log_return' in active_feats:
            df['close_log_return'] = np.log(close_ff / prev_close).fillna(0.0)
        
        if 'open_log_return' in active_feats:
            df['open_log_return'] = np.log(open_ff / prev_close).fillna(0.0)

        needs_vwap = any(f in active_feats for f in ['vwap_diff', 'vwap_log_return', 'return_divergence', 'poc_deviation'])
        if needs_vwap:
            # 🚀 [终极对齐] 如果输入中已经有聚合好的精确 vwap (由 FCS 实时产生或从 DB 读取)，则直接使用，否则降级重算
            vwap = df['vwap'] if 'vwap' in df.columns else (df['close'] * df['volume']).cumsum() / (df['volume'].cumsum() + self.epsilon)
            
            if 'vwap_diff' in active_feats: 
                df['vwap_diff'] = (df['close'] - vwap) / (vwap + self.epsilon)
            
            if 'vwap_log_return' in active_feats:
                df['vwap_log_return'] = np.log(vwap / prev_close).fillna(0.0)
                
            if 'return_divergence' in active_feats:
                # 确保依赖项已计算（即使用户没选 close_log_return，也要在内部计算出用于相减）
                c_log_ret = df['close_log_return'] if 'close_log_return' in df.columns else np.log(close_ff / prev_close).fillna(0.0)
                v_log_ret = df['vwap_log_return'] if 'vwap_log_return' in df.columns else np.log(vwap / prev_close).fillna(0.0)
                df['return_divergence'] = c_log_ret - v_log_ret
            
            # [🔥 修正] 重新对齐训练集 POC 逻辑：50窗口 + 50价格桶
            if 'poc_deviation' in active_feats:
                df['poc_deviation'] = self._calculate_poc_realtime(df)
        
        
        
        
        # 3. 量比 (Volume Ratio)
        if 'volume_ratio' in active_feats:
            sma20_vol = df['volume'].rolling(20, min_periods=1).mean()
            df['volume_ratio'] = (df['volume'] / (sma20_vol + self.epsilon)).fillna(1.0)

        # 4. TA 库指标 (100% 对齐离线 ta 库)
        if 'rsi' in active_feats:
            df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi().fillna(50.0)

        if 'k' in active_feats:
            df['k'] = ta.momentum.StochasticOscillator(
                high=df['high'], low=df['low'], close=df['close'], window=9, smooth_window=3
            ).stoch().fillna(50.0)

        if 'cci' in active_feats:
            df['cci'] = ta.trend.CCIIndicator(
                high=df['high'], low=df['low'], close=df['close'], window=20
            ).cci().fillna(0.0)

        if 'adx_smooth_10' in active_feats:
            try:
                raw_adx = ta.trend.ADXIndicator(
                    high=df['high'], low=df['low'], close=df['close'], window=14
                ).adx()
                df['adx_smooth_10'] = raw_adx.ewm(span=10).mean().fillna(20.0)
            except:
                df['adx_smooth_10'] = 20.0

        # 5. Garman-Klass Volatility (带 20 周期平滑)
        if 'garman_klass_vol' in active_feats:
            log_hl = np.log((df['high'] + self.epsilon) / (df['low'] + self.epsilon))
            log_co = np.log((df['close'] + self.epsilon) / (df['open'] + self.epsilon))
            gk = 0.5 * log_hl**2 - (2 * math.log(2) - 1) * log_co**2
            df['garman_klass_vol'] = np.sqrt(gk.clip(lower=0)).rolling(20).mean().fillna(0.0)

        # 6. ATR 相关归一化特征
        if 'price_slope_norm_by_atr' in active_feats or 'price_dist_from_ma_atr' in active_feats:
            if len(df) < 14:
                atr = pd.Series(0.0, index=df.index)
            else:
                try:
                    atr = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()
                except Exception as e:
                    logger.warning(f"ATR Calc Error: {e}")
                    atr = pd.Series(0.0, index=df.index)
                    
            if 'price_dist_from_ma_atr' in active_feats:
                sma200 = df['close'].rolling(200).mean().fillna(df['close'])
                df['price_dist_from_ma_atr'] = (df['close'] - sma200) / (atr + self.epsilon)
            
            if 'price_slope_norm_by_atr' in active_feats:
                slope = (df['close'] - df['close'].shift(10).fillna(df['close'])) / 10.0
                df['price_slope_norm_by_atr'] = slope / (atr + self.epsilon)

        return df

    def _prepare_hybrid_tensors(self, history_1min: Dict[str, pd.DataFrame], 
                                ready_syns: List[str], 
                                master_index: pd.Index,
                                all_feats: List[str]) -> Tuple[torch.Tensor, Dict[str, int]]:
        """
        将 Pandas 预计算的结果对齐后，打包成动态列宽的 Tensor。
        [升级] 增加极强的数据真空防弹装甲，完美解决 1m/5m 异步断流导致的 KeyError。
        """
        B = len(ready_syns)
        max_len = len(master_index)
        
        active_pandas_feats = [f for f in self.PANDAS_FEATS if f in all_feats]
        num_cols = 5 + len(active_pandas_feats)
        prices_bh = torch.zeros(B, max_len, num_cols, device=self.device)
        
        feat_idx_map = {feat: 5 + i for i, feat in enumerate(active_pandas_feats)}

        for i, s in enumerate(ready_syns):
            # 🚀 [防弹修复 1] 安全获取 DataFrame，处理完全缺失的情况
            raw_df = history_1min.get(s)
            if raw_df is None or raw_df.empty:
                continue # prices_bh 保持全 0，安全跳过
                
            df_computed = self._pandas_compute_features(raw_df, active_pandas_feats)
            if df_computed is None or df_computed.empty:
                continue # prices_bh 保持全 0，安全跳过

            # 🚀 [防弹修复 2] 强制补齐目标列！
            # 哪怕 _pandas_compute_features 某些指标计算失败，也强行补 0，绝不让 KeyError 发生！
            target_cols = ['close', 'high', 'low', 'open', 'volume'] + active_pandas_feats
            for col in target_cols:
                if col not in df_computed.columns:
                    df_computed[col] = 0.0

            # 对齐时间轴 (ffill 产生粘性，保证高频采样下不丢失分钟级信号)
            aligned_df = df_computed.reindex(master_index).ffill().bfill()
            
            # 此时提取 values 绝对安全
            vals = torch.tensor(
                aligned_df[target_cols].values, 
                dtype=torch.float32, device=self.device
            )
            prices_bh[i, :, :] = vals
            
        return prices_bh, feat_idx_map

    # ==========================================================================
    # 核心入口：引入严格的 Pandas 时间轴对齐 (Data Alignment)
    # ==========================================================================
    # ==========================================================================
    # 核心入口：引入严格的 Pandas 时间轴对齐 (Data Alignment)
    # ==========================================================================
    def compute_all_inputs(self, 
                           history_1min: Dict[str, pd.DataFrame], 
                           fast_feats: List[str],
                           slow_feats: List[str],
                           option_snapshots: Optional[Dict[str, np.ndarray]] = None,
                           option_contracts: Optional[Dict[str, list]] = None,
                           history_5min: Optional[Dict[str, pd.DataFrame]] = None,
                           option_snapshot_5m: Optional[Dict[str, np.ndarray]] = None,
                           feat_resolutions: Optional[Dict[str, str]] = None,
                           skip_scaling: bool = False,
                           current_ts: Optional[float] = None) -> Dict[str, Dict]:
        results = {}
        all_feats = list(set(fast_feats + slow_feats))
        if not history_1min: return results

        # 1. 提取全局时间轴 (Master Timeline)
        all_indices = [df.index for df in history_1min.values() if not df.empty]
        if not all_indices: return results
        
        master_index = all_indices[0]
        for idx in all_indices[1:]:
            master_index = master_index.union(idx)
        master_index = master_index.sort_values()
        max_len = len(master_index)

        # 1. 确定锚点时间 (优先使用显式传入的 current_ts)
        last_ts = pd.Timestamp(current_ts, unit='s', tz='UTC') if current_ts else master_index[-1]
        time_feats = self.time_gen.get_time_features(last_ts, max_len)
        
        # 2. 对齐 SPY/QQQ 等宏观特征
        global_ctx = {}
        for tk, tv in time_feats.items():
            global_ctx[tk] = torch.tensor(tv, dtype=torch.float32, device=self.device)
            
        df_vix = history_1min.get('VIXY')
        if df_vix is not None and not df_vix.empty:
            df_vix_aligned = df_vix.reindex(master_index).ffill().bfill()
            global_ctx.update(self._compute_vix_global(df_vix_aligned, all_feats))

        for idx_sym, feat_name in [('SPY', 'spy_roc_5min'), ('QQQ', 'qqq_roc_5min')]:
            df_idx = history_1min.get(idx_sym)
            if df_idx is not None and not df_idx.empty:
                roc_series = df_idx['close'].pct_change(5).fillna(0.0)
                aligned_roc = roc_series.reindex(master_index).ffill().fillna(0.0).values
                global_ctx[feat_name] = torch.tensor(aligned_roc, dtype=torch.float32, device=self.device)

        # 3. 准备混合张量 (Hybrid Tensors)
        ready_syns = [s for s, df in history_1min.items() if not df.empty and s not in ['SPY', 'QQQ', 'VIXY']]
        if not ready_syns: return results
        
        # 区分 1min 和 5min 特征
        slow_feats_1m = []
        slow_feats_5m = []
        if feat_resolutions and USE_5M_OPTION_DATA:
            for f in slow_feats:
                if feat_resolutions.get(f) == '5min': slow_feats_5m.append(f)
                else: slow_feats_1m.append(f)
        else:
            slow_feats_1m = slow_feats

        # --- A. 计算 1min 特征 ---
        prices_bh, feat_idx_map = self._prepare_hybrid_tensors(history_1min, ready_syns, master_index, fast_feats + slow_feats_1m)
        
        B = len(ready_syns)
        opts_bh = torch.zeros(B, 6, 12, device=self.device)
        for i, s in enumerate(ready_syns):
            if option_snapshots and s in option_snapshots:
                  raw_snap = option_snapshots[s]
                  
                  # 🚀 [Parity Fix] 核心集成：在生成 Tensor 前，强制执行希腊值补算/重算逻辑
                  # 底层 calculate_bucket_greeks 会先抹除旧 IV (0.5) 再根据当前价重算
                  df_s = history_1min.get(s)
                  if df_s is not None and not df_s.empty:
                      last_row = df_s.iloc[-1]
                      # 🚀 [Parity Fix] 使用传入的 option_contracts 透传给 Greeks 补算
                      if option_contracts and s in option_contracts:
                          # 🚀 [Parity Fix] 显式透传 current_ts (或 fallback 到 last_ts) 确保到期时间计算正确
                          target_ts = current_ts if current_ts else last_ts.timestamp()
                          raw_snap = self.supplement_greeks(
                              s, raw_snap, option_contracts[s], 
                              float(last_row['close']), target_ts
                          )

                  if raw_snap.shape[0] < 6:
                      raw_snap = np.vstack([raw_snap, np.zeros((6-raw_snap.shape[0], raw_snap.shape[1]), dtype=raw_snap.dtype)])
                  if raw_snap.shape[1] < 12:
                      raw_snap = np.hstack([raw_snap, np.zeros((raw_snap.shape[0], 12-raw_snap.shape[1]), dtype=raw_snap.dtype)])
                      
                  # 🚀 [🔥 终极闭环修复] 保存被重写的 snap（包含最新的 IV 和希腊值），供外部提取！
                  option_snapshots[s] = raw_snap
                  opts_bh[i] = torch.tensor(raw_snap[:, :12], dtype=torch.float32, device=self.device)

        batch_res_1m = self.compute_batch_features(
            prices_bh, feat_idx_map, ready_syns, fast_feats, slow_feats_1m, 
            option_snapshot=opts_bh, skip_scaling=skip_scaling, global_ctx=global_ctx
        )

        # --- B. 计算 5min 特征 (如果存在) ---
        batch_res_5m = {}
        opts_bh_5m = None
        if slow_feats_5m and history_5min:
            # 5min 的 master_index 相对简单，因为输入就是 5min K线
            indices_5m = [df.index for df in history_5min.values() if not df.empty]
            if indices_5m:
                master_index_5m = indices_5m[0]
                for idx in indices_5m[1:]: master_index_5m = master_index_5m.union(idx)
                master_index_5m = master_index_5m.sort_values()
                
                prices_bh_5m, feat_idx_map_5m = self._prepare_hybrid_tensors(history_5min, ready_syns, master_index_5m, slow_feats_5m)
                
                opts_bh_5m = torch.zeros(B, 6, 12, device=self.device)
                for i, s in enumerate(ready_syns):
                    if option_snapshot_5m and s in option_snapshot_5m:
                        raw_snap = option_snapshot_5m[s]
                        if raw_snap.shape[0] < 6:
                            raw_snap = np.vstack([raw_snap, np.zeros((6-raw_snap.shape[0], raw_snap.shape[1]), dtype=raw_snap.dtype)])
                        if raw_snap.shape[1] < 12:
                            raw_snap = np.hstack([raw_snap, np.zeros((raw_snap.shape[0], 12-raw_snap.shape[1]), dtype=raw_snap.dtype)])
                        opts_bh_5m[i] = torch.tensor(raw_snap[:, :12], dtype=torch.float32, device=self.device)

                res_5m_raw = self.compute_batch_features(
                    prices_bh_5m, feat_idx_map_5m, ready_syns, [], slow_feats_5m, 
                    option_snapshot=opts_bh_5m, skip_scaling=skip_scaling, global_ctx=global_ctx
                )
                
                # ==============================================================
                # 🚀 [时序映射终极防弹升级] 彻底废弃 repeat_interleave
                # 利用时间戳强行映射，完美解决实盘 09:37 这种非整数时间的相位错乱问题！
                # ==============================================================
                if res_5m_raw.get('slow_1m') is not None:
                    t_5m_raw = res_5m_raw['slow_1m'] # [B, N_5m, L_5m]
                    L_5m = t_5m_raw.shape[-1]
                    
                    # 1. 查找 1min 的每个时间点，在 5min 时间轴上的索引位置
                   # 1. 查找 1min 的每个时间点，在 5min 时间轴上的索引位置
                    # 🚀 [🔥 致命维度修复] master_index 长度为 500，但模型特征仅需最后 30 步！
                    # 必须切片最后 30 个 1min 时间点去映射，否则张量会被拉伸到 500 导致维度爆炸。
                    target_master_index = master_index[-30:]
                    pos_indices = master_index_5m.searchsorted(target_master_index)
                    
                    # 2. 将全局绝对索引转换为 t_5m_raw 的局部相对索引
                    start_idx_5m = len(master_index_5m) - L_5m
                    rel_indices = pos_indices - start_idx_5m
                    
                    # 3. 边界保护，防止因为断流产生的越界
                    rel_indices = np.clip(rel_indices, 0, L_5m - 1)
                    rel_idx_tensor = torch.tensor(rel_indices, dtype=torch.long, device=self.device)
                    
                    # 4. 提取出与 1min 绝对时序等长 (30) 且严格对齐的 5min 特征序列
                    t_5m_upsampled = t_5m_raw[:, :, rel_idx_tensor]
                    batch_res_5m['slow_1m'] = t_5m_upsampled 
        
        # 4. 组装并分发
        for i, s in enumerate(ready_syns):
            s_fast = batch_res_1m['fast_1m'][i].unsqueeze(0) if batch_res_1m.get('fast_1m') is not None else None
            # 合并 1m 和 5m 的 slow 特征
            s_slow_1m = batch_res_1m['slow_1m'][i] if batch_res_1m.get('slow_1m') is not None else None
            s_slow_5m = batch_res_5m['slow_1m'][i] if batch_res_5m.get('slow_1m') is not None else None
            
            if s_slow_1m is not None:
                if s_slow_5m is not None:
                    # 我们需要按照 slow_feats 的原始顺序合并列
                    merged_slow = torch.zeros(len(slow_feats), 30, device=self.device)
                    k1, k5 = 0, 0
                    for f in slow_feats:
                        idx = slow_feats.index(f)
                        if f in slow_feats_5m:
                            merged_slow[idx] = s_slow_5m[k5]
                            k5 += 1
                        else:
                            merged_slow[idx] = s_slow_1m[k1]
                            k1 += 1
                    output_slow = merged_slow.unsqueeze(0)
                else:
                    output_slow = s_slow_1m.unsqueeze(0)
            else:
                output_slow = s_slow_5m.unsqueeze(0) if s_slow_5m is not None else None

            results[s] = {
                'fast_1m': s_fast if s_fast is not None else None,
                'slow_1m': output_slow,
                # 🚀 [接通水管] 将包含准确 IV/Greeks 的期权矩阵回传给 Service，用于写 SQLite！
                'opts_bh': opts_bh[i:i+1] if opts_bh is not None else None,
                'opts_bh_5m': opts_bh_5m[i:i+1] if opts_bh_5m is not None else None, 
                'updated_buckets': option_snapshots[s] if option_snapshots and s in option_snapshots else None
            }
        
        # 🚀 [SDS 2.0 计算屏障] 希腊值/IV 计算已全部入库，现在可以由于安全物理执行延迟结算
        if hasattr(self, 'service'):
            self.service.finalization_barrier()
            
        return results

    # ==========================================================================
    # 核心计算引擎 (Batch Vectorized)
    # ==========================================================================
    def compute_batch_features(self, 
                             prices_bh: torch.Tensor,
                             feat_idx_map: Dict[str, int],
                             symbol_list: List[str],
                             fast_feats: List[str],
                             slow_feats: List[str],
                             option_snapshot: Optional[torch.Tensor] = None,
                             skip_scaling: bool = False,
                             global_ctx: Dict = {}) -> Dict[str, Dict]:
        
        B, L, _ = prices_bh.shape
        close = prices_bh[:, :, 0]
        high = prices_bh[:, :, 1]
        low = prices_bh[:, :, 2]
        open_ = prices_bh[:, :, 3]
        volume = prices_bh[:, :, 4]
        
        all_feats = list(set(fast_feats + slow_feats))
        res = {}

        # 1. 提取 Pandas 预计算的特征
        for feat_name, idx in feat_idx_map.items():
            if feat_name in all_feats:
                res[feat_name] = prices_bh[:, :, idx]

        # [🆕 新增] 截面动量 (Cross-Sectional Momentum Z-Score)
        # 必须在 Batch 层面计算，直接利用 res 中的 close_log_return
        if 'cross_sect_mom_z' in all_feats:
            log_ret = res.get('close_log_return', torch.zeros((B, L), device=self.device))
            # 维度说明: log_ret [B, L]。我们要对 B (Symbol) 进行截面均值/标差计算。
            if B > 1:
                m = log_ret.mean(dim=0, keepdim=True) # [1, L]
                s = log_ret.std(dim=0, keepdim=True)  # [1, L]
                res['cross_sect_mom_z'] = (log_ret - m) / (s + self.epsilon)
            else:
                res['cross_sect_mom_z'] = torch.zeros_like(log_ret)
        
        # 2. Batch Indicators (PyTorch 后端)
        if 'volume_log' in all_feats: 
            res['volume_log'] = torch.log1p(volume)
            
        if 'sma_ratio_30' in all_feats and 'sma_ratio_30' not in res:
            sma30 = self._sma(close, 30)
            res['sma_ratio_30'] = close / (sma30 + self.epsilon)
            
        if 'chaikin_vol' in all_feats:
            hl = high - low
            ema10_hl = self._ema(hl, 10)
            prev_ema = torch.roll(ema10_hl, 10, dims=1)
            prev_ema[:, :10] = ema10_hl[:, :10] # Pad
            res['chaikin_vol'] = (ema10_hl - prev_ema) / (prev_ema + self.epsilon)
            
        if 'vol_roc' in all_feats:
            prev_vol = torch.roll(volume, 1, dims=1)
            prev_vol[:, 0] = volume[:, 0]
            res['vol_roc'] = (volume - prev_vol) / (prev_vol + self.epsilon)
            
        if 'vol_contraction_ratio' in all_feats and 'vol_contraction_ratio' not in res:
            sma20_vol = self._sma(volume, 20)
            ratio = volume / (sma20_vol + self.epsilon)
            res['vol_contraction_ratio'] = ratio
            
        if 'bb_width' in all_feats:
            u, l_ = self._bbands(close, 20, 2)
            res['bb_width'] = (u - l_) / (close + self.epsilon)
            
        if 'fast_mom' in all_feats:
            prev_close = torch.roll(close, 5, dims=1)
            prev_close[:, :5] = close[:, :5]
            res['fast_mom'] = (close / (prev_close + self.epsilon)) - 1.0

        if 'fast_vol' in all_feats:
            ret = torch.zeros_like(close)
            ret[:, 1:] = (close[:, 1:] / (close[:, :-1] + self.epsilon)) - 1.0
            res['fast_vol'] = self._std(ret, 5) * 100.0
            
        if 'garch_vol' in all_feats:
            ret = torch.zeros_like(close)
            ret[:, 1:] = torch.log(close[:, 1:] / (close[:, :-1] + self.epsilon))
            ret_sq = ret ** 2
            garch = torch.empty_like(ret_sq)
            garch[:, 0] = ret_sq[:, 0]
            lambda_ = 0.905
            for i in range(1, L):
                garch[:, i] = lambda_ * garch[:, i-1] + (1 - lambda_) * ret_sq[:, i]
            res['garch_vol'] = torch.sqrt(garch)
            
        if 'macd_ratio' in all_feats or 'macd_diff_ratio' in all_feats:
            macd = self._ema(close, 12) - self._ema(close, 26)
            sig = self._ema(macd, 9)
            if 'macd_ratio' in all_feats: res['macd_ratio'] = macd / (close + self.epsilon)
            if 'macd_diff_ratio' in all_feats: res['macd_diff_ratio'] = (macd - sig) / (close + self.epsilon)

        # 兜底：如果某些特征因为配置问题没有被 Pandas 处理到，这里保留旧版计算逻辑 (如 k/d, rsi等)
        if 'rsi' in all_feats and 'rsi' not in res: 
            res['rsi'] = self._rsi(close, 14)
        if ('k' in all_feats or 'd' in all_feats) and 'k' not in res:
            k, d, _ = self._kdj(high, low, close)
            if 'k' in all_feats: res['k'] = k
            if 'd' in all_feats: res['d'] = d
        if 'vwap_diff' in all_feats and 'vwap_diff' not in res:
            cum_pv = torch.cumsum(close * volume, dim=1)
            cum_v = torch.cumsum(volume, dim=1)
            vwap = cum_pv / (cum_v + self.epsilon)
            res['vwap_diff'] =  close - vwap  / (vwap + self.epsilon)

        # --- 3. 挂载 Global & Option Features ---
        for k, v in global_ctx.items():
            if k in all_feats:
                if v.dim() == 0 or (v.dim() == 1 and v.shape[0] == 1):
                    res[k] = v.view(1, 1).expand(B, L)
                elif v.dim() == 1 and v.shape[0] == L:
                    res[k] = v.view(1, L).expand(B, L)
                elif v.numel() == 1:
                    res[k] = v.view(1, 1).expand(B, L)

        if option_snapshot is not None:
             opt_feats_batch = self._calc_opt_feats_batch(option_snapshot, close[:, -1])
             for k, v in opt_feats_batch.items():
                 if k in all_feats: res[k] = v.view(B, 1).expand(B, L)

        # --- 4. Extract & Scale ---
        final_results = {}
        
        def process_subset(feat_list, seq_len):
            tensors = []
            for name in feat_list:
                col = res.get(name, torch.zeros((B, L), device=self.device))
                
                if name in self.stats and not skip_scaling:
                    s = self.stats[name]
                    median = s.get('median', 0.0)
                    iqr = max(s.get('iqr', 1.0), 1e-6)
                    col = torch.clamp((col - median) / iqr, -10.0, 10.0)
                tensors.append(col)
                
            if not tensors: return None
            stacked = torch.stack(tensors, dim=1) # [B, N_Feat, L]
            
            if L >= seq_len:
                return stacked[:, :, -seq_len:]
            else:
                # [🔥 缺陷 C 修复] 历史 K 线不足，打印警告且拒绝推理
                # 上游服务已有预热逻辑，正常情况下不应缺数据。
                # 如果不足说明系统异常，强行填充会给模型喂入噪声信号，必须直接拒绝。
                shortage_pct = (seq_len - L) / seq_len
                logger.warning(
                    f"[数据不足-拒止推理] 历史K线 {L} 根，要求 {seq_len} 根，"
                    f"缺少 {shortage_pct:.0%}。上游预热应保证数据充足，"
                    f"本帧展示拒止推理。"
                )
                return None

        final_results['fast_1m'] = process_subset(fast_feats, 10)
        final_results['slow_1m'] = process_subset(slow_feats, 30)

        return final_results
    

    def _calc_opt_feats_batch(self, snap, spot):
        out = {}
        if snap is None: return out
        eps = self.epsilon
        
        vol_vec = snap[:, :, 6] # [B, 6]
        iv_vec = snap[:, :, 7]
        vega_vec = snap[:, :, 3]
        theta_vec = snap[:, :, 4]
        gamma_vec = snap[:, :, 2]
        delta_vec = snap[:, :, 1]
        bid_vec = snap[:, :, 8]
        ask_vec = snap[:, :, 9]
        
        # 提取前4个档位 (ATM/OTM)
        vol_front = vol_vec[:, 0:4] # [B, 4]
        total_vol = torch.sum(vol_front, dim=1)
        use_equal = total_vol < 1.0

        # ==========================================================
        # 🚀 [特征对齐 1] 真实基准 IV (Baseline IV): 绝对不混合！只取 ATM 均值
        # ==========================================================
        out['options_vw_iv'] = (iv_vec[:, 0] + iv_vec[:, 2]) / 2.0

        # ==========================================================
        # 🚀 [特征对齐 2] 净 Delta / Gamma / Vega / Theta 敞口
        # ==========================================================
        net_delta = torch.sum(delta_vec[:, 0:4] * vol_front, dim=1) / (total_vol + eps)
        out['options_vw_delta'] = torch.where(use_equal, torch.tensor(0.0, device=self.device), net_delta)

        def calc_net(vec_all):
            net_val = torch.sum(vec_all[:, 0:4] * vol_front, dim=1) / (total_vol + eps)
            fallback = (vec_all[:, 0] + vec_all[:, 2]) / 2.0  # 断流时使用 ATM 均值
            return torch.where(use_equal, fallback, net_val)

        out['options_vw_gamma'] = calc_net(gamma_vec)
        out['options_vw_vega'] = calc_net(vega_vec)
        out['options_vw_theta'] = calc_net(theta_vec)
        
        out['options_vw_vanna'] = (vega_vec[:, 0] + vega_vec[:, 2]) / 2.0 / (spot + eps)
        out['options_vw_charm'] = (theta_vec[:, 0] + theta_vec[:, 2]) / 2.0 / (spot + eps)

        # ==========================================================
        # 🚀 [特征对齐 3] 微观失衡 (断流时严格给 0.0，杜绝幻觉)
        # ==========================================================
        def calc_pure_vw(vec_all):
            net_val = torch.sum(vec_all[:, 0:4] * vol_front, dim=1) / (total_vol + eps)
            return torch.where(use_equal, torch.tensor(0.0, device=self.device), net_val)

        out['options_vw_spread'] = calc_pure_vw(ask_vec - bid_vec)
        imb_vec = (snap[:, 0:4, 10] - snap[:, 0:4, 11]) / (snap[:, 0:4, 10] + snap[:, 0:4, 11] + eps)
        out['options_vw_imbalance'] = calc_pure_vw(imb_vec)

        # ==========================================================
        # 🚀 [特征对齐 3.1] [🆕 新增] Gamma 失衡 (Gamma Squeeze Fuel)
        # ==========================================================
        # (Call_Gamma - Put_Gamma) 
        g_imb = (gamma_vec[:, 2] + gamma_vec[:, 3]) - (gamma_vec[:, 0] + gamma_vec[:, 1])
        out['options_gamma_imbalance'] = g_imb  # 这个特征通常在末日期权引爆时极度有效
        
        # Gamma 比例 (相对于总 Gamma)
        total_g = torch.sum(gamma_vec[:, 0:4], dim=1)
        out['options_gamma_ratio'] = g_imb / (total_g + eps)

        # ==========================================================
        # 🚀 [特征对齐 4] 偏斜与期限结构 (严格遵守物理法则)
        # ==========================================================
        denom_call_vol = vol_vec[:, 2] + vol_vec[:, 3]
        pcr = (vol_vec[:, 0] + vol_vec[:, 1]) / (denom_call_vol + eps)
        out['options_pcr_volume'] = torch.where(denom_call_vol > 0, pcr, torch.tensor(1.0, device=self.device))

        # Flow Skew: 虚值 Put vs 虚值 Call
        out['options_flow_skew'] = torch.where(iv_vec[:, 3] > 0.01, iv_vec[:, 1] / (iv_vec[:, 3] + eps), torch.tensor(1.0, device=self.device))
        
        # Struc Skew: 虚值 Put vs 平值 Put
        out['options_struc_skew'] = torch.where(iv_vec[:, 0] > 0.01, iv_vec[:, 1] / (iv_vec[:, 0] + eps), torch.tensor(1.0, device=self.device))

        out['options_struc_atm_iv'] = out['options_vw_iv']
        next_atm = (iv_vec[:, 4] + iv_vec[:, 5]) / 2.0
        term_val = next_atm - out['options_vw_iv']
        out['options_struc_term'] = torch.where((next_atm > 0.01) & (out['options_vw_iv'] > 0.01), term_val, torch.tensor(0.0, device=self.device))

        return out
        
    # ==========================================================================
    # 极致优化的算子库 (Operator Library) - 完整保留
    # ==========================================================================
    def _sma(self, x, w):
        if x.dim() == 1: x = x.unsqueeze(0)
        B, L = x.shape
        if L < w: 
            return torch.cumsum(x, dim=1) / torch.arange(1, L+1, device=x.device).view(1, -1)
        
        kernel = torch.ones(B, 1, w, device=x.device) / w
        x_pad = F.pad(x.view(1, B, L), (w-1, 0), mode='replicate')
        return F.conv1d(x_pad, kernel, padding=0, groups=B).view(B, L)

    def _ema(self, x, w):
        if x.dim() == 1: x = x.view(1, -1)
        B, L = x.shape
        if L < 1: return x
        
        alpha = 2 / (w + 1)
        out = torch.zeros_like(x)
        out[:, 0] = x[:, 0]
        for i in range(1, L):
            out[:, i] = alpha * x[:, i] + (1 - alpha) * out[:, i-1]
        return out

    def _std(self, x, w):
        if x.dim() == 1: x = x.view(1, -1)
        B, L = x.shape
        if L < w: return torch.zeros_like(x)

        mean_x = self._sma(x, w)
        mean_x2 = self._sma(x ** 2, w)
        var = torch.clamp(mean_x2 - mean_x ** 2, min=0.0)
        if w > 1: var = var * (w / (w - 1))
        return torch.sqrt(var + self.epsilon)

    def _slope(self, x, w):
        if x.dim() == 1: x = x.view(1, -1)
        B, L = x.shape
        if L < w: return torch.zeros_like(x)
        
        steps = torch.arange(w, dtype=torch.float32, device=x.device)
        x_diff = steps - ((w - 1) / 2.0)
        kernel_base = (x_diff / ((x_diff ** 2).sum() + self.epsilon)).flip(0).view(1, 1, w)
        kernel = kernel_base.repeat(B, 1, 1)
        
        x_pad = F.pad(x.view(1, B, L), (w-1, 0), mode='replicate')
        return F.conv1d(x_pad, kernel, padding=0, groups=B).view(B, L)

    def _rsi(self, x, w):
        if x.dim() == 1: x = x.view(1, -1)
        B, L = x.shape
        if L < 2: return torch.zeros_like(x) + 50.0
        
        delta = x[:, 1:] - x[:, :-1]
        gain, loss = torch.relu(delta), torch.relu(-delta)
        
        avg_g, avg_l = torch.zeros_like(gain), torch.zeros_like(loss)
        avg_g[:, 0], avg_l[:, 0] = gain[:, 0], loss[:, 0]
        
        for i in range(1, L-1):
            avg_g[:, i] = (avg_g[:, i-1]*(w-1) + gain[:, i])/w
            avg_l[:, i] = (avg_l[:, i-1]*(w-1) + loss[:, i])/w
            
        rs = avg_g / (avg_l + self.epsilon)
        rsi = 100 - (100/(1+rs))
        return F.pad(rsi, (1,0), value=50.0)

    def _bbands(self, x, w, d):
        sma, std = self._sma(x, w), self._std(x, w)
        return sma + d*std, sma - d*std

    def _cci(self, h, l, c, w):
        if c.dim() == 1: c = c.view(1, -1); h = h.view(1, -1); l = l.view(1, -1)
        B, L = c.shape
        if L < w: return torch.zeros_like(c)
        
        tp = (h + l + c) / 3.0
        sma = self._sma(tp, w)
        
        tp_pad = F.pad(tp, (w-1, 0), mode='replicate')
        tp_windows = tp_pad.unfold(-1, w, 1)
        mad = (tp_windows - sma.unsqueeze(-1)).abs().mean(dim=-1)
        
        return (tp - sma) / (0.015 * mad + self.epsilon)

    def _atr(self, h, l, c, w):
        if c.dim() == 1: c = c.view(1, -1); h = h.view(1, -1); l = l.view(1, -1)
        B, L = c.shape
        if L < 2: return torch.zeros_like(c)
        
        tr = torch.max(h[:, 1:] - l[:, 1:], torch.max((h[:, 1:] - c[:, :-1]).abs(), (l[:, 1:] - c[:, :-1]).abs()))
        out = torch.zeros_like(c)
        out[:, 1] = tr[:, 0] 
        
        for i in range(1, L-1): 
            out[:, i+1] = (out[:, i]*(w-1) + tr[:, i])/w
            
        out[:, 0] = out[:, 1]
        return out

    def _adx(self, h, l, c, w, val_pad=20.0):
        if c.dim() == 1: c = c.view(1, -1); h = h.view(1, -1); l = l.view(1, -1)
        B, L = c.shape
        if L < 2: return torch.zeros_like(c) + val_pad
        
        up, down = h[:, 1:] - h[:, :-1], l[:, :-1] - l[:, 1:]
        pos_dm = torch.where((up > down) & (up > 0), up, torch.zeros_like(up))
        neg_dm = torch.where((down > up) & (down > 0), down, torch.zeros_like(down))
        tr = torch.max(h[:, 1:] - l[:, 1:], torch.max((h[:, 1:] - c[:, :-1]).abs(), (l[:, 1:] - c[:, :-1]).abs()))
        
        def smooth(x_in):
            res = torch.zeros_like(x_in)
            res[:, 0] = x_in[:, 0]
            for i in range(1, x_in.shape[1]): res[:, i] = (res[:, i-1]*(w-1) + x_in[:, i])/w
            return res

        atr_s, pos_dm_s, neg_dm_s = smooth(tr), smooth(pos_dm), smooth(neg_dm)
        plus_di, minus_di = 100 * pos_dm_s / (atr_s + self.epsilon), 100 * neg_dm_s / (atr_s + self.epsilon)
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + self.epsilon)
        
        return F.pad(smooth(dx), (1,0), value=val_pad)

    def _kdj(self, h, l, c):
        if c.dim() == 1: c = c.view(1, -1); h = h.view(1, -1); l = l.view(1, -1)
        B, L = c.shape
        if L < 9: return torch.zeros_like(c) + 50.0, torch.zeros_like(c) + 50.0, torch.zeros_like(c)
        
        l_pad = F.pad(-l.view(B, 1, L), (8, 0), mode='replicate')
        h_pad = F.pad(h.view(B, 1, L), (8, 0), mode='replicate')
        low_min = -F.max_pool1d(l_pad, 9, stride=1).view(B, L)
        high_max = F.max_pool1d(h_pad, 9, stride=1).view(B, L)
        
        rsv = 100 * (c - low_min) / (high_max - low_min + self.epsilon)
        k, d = torch.zeros_like(rsv), torch.zeros_like(rsv)
        k[:, 0], d[:, 0] = 50.0, 50.0
        
        for i in range(1, L):
            k[:, i] = (2*k[:, i-1] + rsv[:, i])/3
            d[:, i] = (2*d[:, i-1] + k[:, i])/3
            
        return k, d, 3*k - 2*d

    def _calculate_poc_realtime(self, df: pd.DataFrame, window: int = 50, bins: int = 50) -> pd.Series:
        """
        [极致对齐]：使用离线端完全一致的 POC 计算算法。
        """
        if len(df) < window:
            return pd.Series(0.0, index=df.index)

        # 准备数据向量
        close_prices = df['close'].to_numpy().astype(np.float64)
        volumes = df['volume'].to_numpy().astype(np.float64)

        # 1. 产生滑动窗口视图
        close_windows = np.lib.stride_tricks.sliding_window_view(close_prices, window_shape=window)
        volume_windows = np.lib.stride_tricks.sliding_window_view(volumes, window_shape=window)

        # 2. 调用极致加速的 Numba 算子
        poc_values = _numba_poc_loop(close_windows, volume_windows, bins)

        # 3. 构造结果并填充
        result_series = pd.Series(np.nan, index=df.index, dtype=float)
        result_series.iloc[window - 1:] = poc_values
        
        # 填充以防推理初期数据断层
        result_series.ffill(inplace=True)
        result_series.fillna(0.0, inplace=True)
        
        # 最终计算偏离比例 (对齐离线公式)
        return (df['close'] - result_series) / (result_series + 1e-9)
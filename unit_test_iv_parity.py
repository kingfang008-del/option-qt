import pandas as pd
import numpy as np
import os
import sys

# 假设我们在 repo 根目录
# 模拟 1m 引擎的逻辑 (Simplified from option_cac_day_vectorized_day.py)
def mock_1m_engine_factors(date_str, ts_str):
    # 加载利率
    rfr_path = "data/fred/DGS3MO.parquet" # 假设在 data 目录下
    if not os.path.exists(rfr_path):
        rfr_path = "production/preprocess/backtest/data/fred/DGS3MO.parquet" # 尝试备用路径
        
    rfr_df = pd.read_parquet(rfr_path)
    rfr_df.index = pd.to_datetime(rfr_df.index)
    rfr_series = rfr_df['DGS3MO'] / 100.0

    current_ts = pd.to_datetime(ts_str).tz_localize('America/New_York')
    # 模拟 1m 的 r 匹配逻辑
    search_key = current_ts.normalize().tz_localize(None)
    r_idx = rfr_series.index.searchsorted(search_key)
    r_idx = np.clip(r_idx, 0, len(rfr_series) - 1)
    r_val = rfr_series.values[r_idx]

    # 模拟 1m 的 T 计算逻辑
    # 1m 通常在 process_symbol_task_entry 中处理
    expiry_date = pd.to_datetime(date_str).tz_localize('America/New_York')
    expiry_ts = expiry_date + pd.Timedelta(hours=16)
    time_diff = expiry_ts - current_ts
    T_years = time_diff.total_seconds() / 31557600.0
    
    return r_val, T_years

# 模拟 1s 引擎目前的逻辑 (calc_offline_1s_greeks.py)
def mock_1s_engine_factors(date_str, ts_str):
    # 加载利率 (当前 1s 引擎的路径版本)
    rfr_path = "data/fred/DGS3MO.parquet"
    rfr_df = pd.read_parquet(rfr_path)
    rfr_df.index = pd.to_datetime(rfr_df.index)
    rfr_series = rfr_df['DGS3MO'] / 100.0

    current_ts = pd.to_datetime(ts_str).tz_localize('America/New_York')
    # 模拟 1s 的匹配逻辑
    search_keys = current_ts.normalize().tz_localize(None)
    r_idx = rfr_series.index.searchsorted(search_keys)
    r_idx = np.clip(r_idx, 0, len(rfr_series) - 1)
    r_val = rfr_series.values[r_idx]

    expiry_date = pd.to_datetime(date_str).tz_localize('America/New_York')
    expiry_ts = expiry_date + pd.Timedelta(hours=16)
    time_diff = expiry_ts - current_ts
    T_years = time_diff.total_seconds() / 31557600.0
    
    return r_val, T_years

if __name__ == "__main__":
    date_expiry = "2026-01-09" # 示例到期日
    ts_now = "2026-01-02 10:00:00"
    
    r1m, t1m = mock_1m_engine_factors(date_expiry, ts_now)
    r1s, t1s = mock_1s_engine_factors(date_expiry, ts_now)
    
    print(f"Comparison for {ts_now} (Expiry {date_expiry}):")
    print(f"1m Engine -> r: {r1m:.6f}, T: {t1m:.10f}")
    print(f"1s Engine -> r: {r1s:.6f}, T: {t1s:.10f}")
    
    if r1m == r1s and t1m == t1s:
        print("\n✅ Factors are identical in theory!")
    else:
        print("\n❌ Discrepancy found in calculation factors!")

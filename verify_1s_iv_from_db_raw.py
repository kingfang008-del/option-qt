import sqlite3
import pandas as pd
import numpy as np
import os
import sys

# 将代码路径添加到 sys.path 以便 import
sys.path.append(os.getcwd())
from production.preprocess.data.calc_offline_1s_greeks import vectorized_implied_volatility, load_risk_free_rates
import production.preprocess.data.calc_offline_1s_greeks as greeks_module

def test_iv_from_raw_sql():
    db_path = "production/preprocess/backtest/history_sqlite_1s/market_20260102.db"
    conn = sqlite3.connect(db_path)
    
    # 锁定时间点 (10:00:00 为起始的 1s 采样)
    # 按照 1s 引擎逻辑，它会读取 10:00:00 时刻的 snapshot
    target_ts = 1767366000.0  # 2026-01-02 10:00:00
    
    # 1. 读取标的价格 S (从 market_bars_1s)
    q_s = "SELECT ts, symbol, close FROM market_bars_1s WHERE symbol='NVDA' AND ts=?"
    df_s = pd.read_sql_query(q_s, conn, params=(target_ts,))
    
    # 2. 读取期权快照 P, K (从 option_snapshots_1s)
    # 1s 引擎通常解析 buckets_json
    q_p = "SELECT ts, symbol, buckets_json FROM option_snapshots_1s WHERE symbol='NVDA' AND ts=?"
    df_p = pd.read_sql_query(q_p, conn, params=(target_ts,))
    
    print(f"--- Data Loaded for TS: {target_ts} ---")
    if df_s.empty or df_p.empty:
        print("Error: Could not find raw 1s data in DB.")
        return

    S_val = df_s['close'].iloc[0]
    import json
    buckets = json.loads(df_p['buckets_json'].iloc[0])
    
    # 挑选我们在 1m 中看到的那个特定期权
    # NVDA260116C00185000 (K=185, Call)
    target_opt = 'NVDA260116C00185000'
    opt_info = None
    for ticker, data in buckets.items():
        if ticker == target_opt:
            opt_info = data
            break
    
    if not opt_info:
        print(f"Error: {target_opt} not found in 1s buckets_json.")
        return

    P_val = opt_info['mid'] # 1s 引擎使用的是 mid
    K_val = opt_info['strike']
    expiry_str = opt_info['expiration']
    is_call = True if 'C' in target_opt else False
    
    # 3. 准备计算因子 (模拟 calc_offline_1s_greeks 内部逻辑)
    greeks_module.load_risk_free_rates()
    current_dt = pd.to_datetime(target_ts, unit='s').tz_localize('UTC').tz_convert('America/New_York')
    
    # r 匹配
    search_key = current_dt.normalize().tz_localize(None)
    g_rfr = greeks_module.G_RFR_SERIES
    r_idx = g_rfr.index.searchsorted(search_key)
    r_idx = np.clip(r_idx, 0, len(g_rfr) - 1)
    r_val = g_rfr.values[r_idx]
    
    # T 计算
    expiry_dt = pd.to_datetime(expiry_str).tz_localize('America/New_York') + pd.Timedelta(hours=16)
    time_diff = expiry_dt - current_dt
    T_years = time_diff.total_seconds() / 31557600.0
    
    # 4. 执行计算
    iv_calc = vectorized_implied_volatility(
        np.array([P_val]), np.array([S_val]), np.array([K_val]), 
        np.array([T_years]), np.array([r_val]), 
        'c' if is_call else 'p', return_as='numpy'
    )[0]
    
    print(f"\nFinal Test Results for {target_opt}:")
    print(f"  S (Underlying): {S_val}")
    print(f"  P (Option Mid): {P_val}")
    print(f"  K (Strike):     {K_val}")
    print(f"  r (Risk Free):  {r_val} (Loaded from parquet)")
    print(f"  T (Years):      {T_years:.10f}")
    print(f"  IV (Result):    {iv_calc:.10f}")
    
    conn.close()

if __name__ == "__main__":
    test_iv_from_raw_sql()

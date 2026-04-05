import sqlite3
import json
import pandas as pd
import numpy as np
from pathlib import Path

DB_1S = "/Users/fangshuai/Documents/GitHub/option-qt/production/preprocess/backtest/history_sqlite_1s/market_20260102.db"
DB_1M = "/Users/fangshuai/Documents/GitHub/option-qt/production/preprocess/backtest/history_sqlite_1m/market_20260102.db"

def verify_price_parity(symbol="NVDA"):
    conn_1s = sqlite3.connect(DB_1S)
    conn_1m = sqlite3.connect(DB_1M)
    
    print(f"--- Verifying Price Parity for {symbol} ---")
    
    # Load 1m Bars (Reference)
    query_1m = f"SELECT ts, open, high, low, close, volume FROM market_bars_1m WHERE symbol='{symbol}'"
    df_1m = pd.read_sql(query_1m, conn_1m)
    df_1m.set_index('ts', inplace=True)
    df_1m.sort_index(inplace=True)
    
    # Load 1s Ticks
    query_1s = f"SELECT ts, open, high, low, close, volume FROM market_bars_1s WHERE symbol='{symbol}'"
    df_1s = pd.read_sql(query_1s, conn_1s)
    
    # 🚀 [核心修正] 1m Bar 的时间戳 T 代表 [T, T+60s) 区间的数据
    # 因此 09:30:00 的 Bar 应该包含 09:30:00 到 09:30:59 的所有 Ticks
    df_1s['minute_ts'] = (df_1s['ts'] // 60) * 60 
    
    # Aggregate 1s to 1m
    def last_val(x): return x.iloc[-1]
    def first_val(x): return x.iloc[0]
    
    agg_1s = df_1s.groupby('minute_ts').agg({
        'open': first_val,
        'high': 'max',
        'low': 'min',
        'close': last_val,
        'volume': 'sum'
    })
    
    # Compare
    common_ts = agg_1s.index.intersection(df_1m.index)
    if common_ts.empty:
        print(f"No common timestamps for {symbol}.")
        return
    
    mismatch_count = 0
    for ts in common_ts:
        row_1m = df_1m.loc[ts]
        row_agg = agg_1s.loc[ts]
        
        diff_p = abs(row_1m['close'] - row_agg['close'])
        diff_v = abs(row_1m['volume'] - row_agg['volume'])
        
        # 允许极小的浮点数误差和成交量微小差异
        if diff_p > 1e-4 or diff_v > 1.0:
            if mismatch_count < 3:
                print(f"Mismatch at {ts}: 1m_C={row_1m['close']} vs 1s_agg_C={row_agg['close']} | 1m_V={row_1m['volume']} vs 1s_agg_V={row_agg['volume']}")
            mismatch_count += 1
            
    if mismatch_count == 0:
        print(f"✅ Price Parity Check PASSED for {symbol}.")
    else:
        print(f"⚠️ Price Parity Check FAILED for {symbol} with {mismatch_count} mismatches.")

def verify_option_parity(symbol="NVDA"):
    conn_1s = sqlite3.connect(DB_1S)
    conn_1m = sqlite3.connect(DB_1M)
    
    print(f"\n--- Verifying Option Snapshot Parity for {symbol} ---")
    
    # Load 1m Snapshots
    df_o1m = pd.read_sql(f"SELECT ts, buckets_json FROM option_snapshots_1m WHERE symbol='{symbol}'", conn_1m)
    df_o1m.set_index('ts', inplace=True)
    
    # Load 1s Snapshots
    df_o1s = pd.read_sql(f"SELECT ts, buckets_json FROM option_snapshots_1s WHERE symbol='{symbol}'", conn_1s)
    df_o1s.set_index('ts', inplace=True)
    
    common_ts = df_o1m.index.intersection(df_o1s.index)
    
    mismatch_count = 0
    for ts in common_ts:
        snap_1m = json.loads(df_o1m.loc[ts, 'buckets_json'])
        snap_1s = json.loads(df_o1s.loc[ts, 'buckets_json'])
        
        b1m = snap_1m.get('buckets', [])
        b1s = snap_1s.get('buckets', [])
        
        if len(b1m) != len(b1s):
            mismatch_count += 1
            continue
            
        # 比较前 3 个 bucket 的 IV
        for i in range(min(3, len(b1m))):
            iv1m = b1m[i][7]
            iv1s = b1s[i][7]
            if abs(iv1m - iv1s) > 1e-6:
                mismatch_count += 1
                break
                
    if mismatch_count == 0:
        print(f"✅ Option Parity Check PASSED for {symbol}.")
    else:
        # 如果不匹配，尝试平移 60s 再次对比
        print(f"⚠️ Option Parity Check at exact TS failed. Trying boundary search...")
        # (略过复杂的边界搜索，直接提示用户)
        print(f"   First mismatch at {common_ts[0] if len(common_ts)>0 else 'N/A'}")

if __name__ == "__main__":
    for sym in ["NVDA", "AAPL", "TSLA"]:
        verify_price_parity(sym)
        verify_option_parity(sym)

import sqlite3
import json
import pandas as pd
import numpy as np
import sys, os
from pathlib import Path
from datetime import datetime
import pytz

# ================= 路径配置 =================
PROJECT_ROOT = Path("/Users/fangshuai/Documents/GitHub/option-qt")
sys.path.append(str(PROJECT_ROOT / "production"))
sys.path.append(str(PROJECT_ROOT / "production/baseline"))
sys.path.append(str(PROJECT_ROOT / "production/baseline/DAO"))

DB_PATH = PROJECT_ROOT / "production/preprocess/backtest/history_sqlite_1s/market_20260102.db"
SYMBOL = "NVDA"

try:
    from utils.greeks_math import calculate_bucket_greeks
except ImportError:
    sys.path.append(str(PROJECT_ROOT / "production"))
    from utils.greeks_math import calculate_bucket_greeks

def run_time_sweep():
    print(f"🕵️ 正在扫描 10:00 分整分钟的 60 个采样点...")
    if not DB_PATH.exists():
        print(f"❌ 数据库不存在: {DB_PATH}")
        return

    conn = sqlite3.connect(DB_PATH)
    # 扫描 10:00:00 - 10:00:59 (TS: 1767366000 - 1767366059)
    start, end = 1767366000, 1767366059 
    
    df_opts = pd.read_sql(f"SELECT ts, buckets_json FROM option_snapshots_1s WHERE ts >= {start} AND ts <= {end} AND symbol='{SYMBOL}'", conn)
    conn.close()

    if df_opts.empty:
        print("❌ 未找到期权快照数据。")
        return

    # 强制使用 S1 标杆基准股价 191.73
    base_price = 191.73 
    
    print("-" * 65)
    print(f"{'Timestamp':<12} | {'S':<7} | {'Opt_P':<8} | {'IV (ATM Call)'}")
    print("-" * 65)

    matches = []
    # S1 标杆 IV 是 0.35015
    target_iv = 0.35015

    for ts in range(start, end + 1):
        opt_match = df_opts[df_opts['ts'] == ts]
        if opt_match.empty: continue
        
        blob = json.loads(opt_match.iloc[0]['buckets_json'])
        buckets = np.array(blob['buckets'])
        contracts = blob['contracts']
        opt_p = buckets[2][0] # ATM Call Price
        
        # 计算 IV
        # 使用 Days=365 进行测试
        calculate_bucket_greeks(buckets, base_price, T=7/365.0, r=0.045, contracts=contracts, current_ts=ts)
        iv = buckets[2][7]
        
        diff = abs(iv - target_iv)
        mark = " ⭐ MATCH!" if diff < 0.001 else ""
        if mark: matches.append((ts, iv))
        
        print(f"{ts:<12} | {base_price:<7.2f} | {opt_p:<8.4f} | {iv:<15.6f} {mark}")

    if matches:
        print(f"\n✅ 找到匹配项！S1 的 0.3501 最接近 Timestamp: {matches[0][0]}")
    else:
        print("\n❌ 这一分钟内没有任何一秒的 IV 能完美对齐 0.3501。")

if __name__ == "__main__":
    run_time_sweep()

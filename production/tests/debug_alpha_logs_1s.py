#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sqlite3
import pandas as pd
from pathlib import Path

DB_1S = Path("/Users/fangshuai/Documents/GitHub/option-qt/production/preprocess/backtest/backtest_data_1s_20260102.db")

def debug_logs():
    if not DB_1S.exists():
        print(f"❌ DB NOT FOUND: {DB_1S}")
        return
        
    print(f"📡 Analyzing: {DB_1S.name}")
    conn = sqlite3.connect(DB_1S)
    
    # 0. 检查总行数
    count = conn.execute("SELECT count(*) FROM alpha_logs").fetchone()[0]
    print(f"📊 Total Rows in alpha_logs: {count}")

    # 1. 查找 vol_z 异常的行
    print("\n🚩 Rows with extreme vol_z (> 100):")
    df_huge = pd.read_sql("SELECT * FROM alpha_logs WHERE abs(vol_z) > 100 LIMIT 10", conn)
    print(df_huge.to_string())
    
    # 2. 查找 Price 为 0 的行
    print("\n🚩 Rows with Price = 0:")
    df_zero = pd.read_sql("SELECT * FROM alpha_logs WHERE price = 0 LIMIT 10", conn)
    print(df_zero.to_string())
    
    # 3. 统计各标的的异常情况
    print("\n🚩 Anomalies by Symbol:")
    df_stats = pd.read_sql("""
        SELECT symbol, 
               COUNT(*) as total,
               SUM(CASE WHEN price = 0 THEN 1 ELSE 0 END) as zero_price,
               SUM(CASE WHEN abs(vol_z) > 100 THEN 1 ELSE 0 END) as crazy_vol_z
        FROM alpha_logs 
        GROUP BY symbol
    """, conn)
    print(df_stats.sort_values('crazy_vol_z', ascending=False).head(20).to_string())
    
    conn.close()

if __name__ == "__main__":
    debug_logs()

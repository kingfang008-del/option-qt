import sqlite3
import pandas as pd
import numpy as np

def check_parity():
    # 路径匹配 ls 结果
    db_1m = "production/preprocess/backtest/minute/history_sqlite_1m/market_20260102.db"
    db_1s = "production/preprocess/backtest/second/history_sqlite_1s/market_20260102.db"
    
    symbol = 'NVDA'
    ts_sec = 1767366060.0  # 10:01:00
    ts_min = '2026-01-02 10:01:00'

    print(f"=== Parity Check: {symbol} @ {ts_min} ===")

    try:
        # 1. 查 1m 库
        with sqlite3.connect(db_1m) as c1m:
            df1m = pd.read_sql_query(f"SELECT * FROM alpha_logs WHERE symbol='{symbol}' AND timestamp='{ts_min}'", c1m)
        
        # 2. 查 1s 库
        with sqlite3.connect(db_1s) as c1s:
            df1s = pd.read_sql_query(f"SELECT * FROM alpha_logs WHERE symbol='{symbol}' AND timestamp={ts_sec}", c1s)

        if df1m.empty :
            print("❌ 1m database has no data for this timestamp.")
            return
        if df1s.empty:
            print("❌ 1s database has no data for this timestamp.")
            return

        # 对比数据
        iv_1m = df1m.iloc[0]['iv']
        iv_1s = df1s.iloc[0]['iv']
        p_1m = df1m.iloc[0]['price']
        p_1s = df1s.iloc[0]['price']
        
        print(f"\n[Price Check]")
        print(f"1m Price: {p_1m:.4f}")
        print(f"1s Price: {p_1s:.4f}")
        print(f"Price Diff: {p_1m - p_1s:.8f}")

        print(f"\n[IV Check]")
        print(f"1m IV: {iv_1m:.6f}")
        print(f"1s IV: {iv_1s:.6f}")
        print(f"IV Diff: {(iv_1m - iv_1s)*10000:.2f} bps")

        # 尝试反推 T
        # 假设到期日是当天 16:00:00
        expiry = pd.to_datetime("2026-01-02 16:00:00")
        now = pd.to_datetime(ts_min)
        t_years = (expiry - now).total_seconds() / 31557600.0
        print(f"\n[Reference T]")
        print(f"T (0DTE at 10:01): {t_years:.10f}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_parity()

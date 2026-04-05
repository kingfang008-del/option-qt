import sqlite3
import pandas as pd
from datetime import datetime

def check_raw_factors():
    # 路径使用您纠正后的正确路径
    db_1m = "production/preprocess/backtest/history_sqlite_1m/market_20260102.db"
    db_1s = "production/preprocess/backtest/history_sqlite_1s/market_20260102.db"
    
    symbol = 'NVDA'
    ts_sec = 1767366000.0  # 10:00:00
    
    print("--- 1m Factors (Physical Read) ---")
    conn1m = sqlite3.connect(db_1m)
    res1m = conn1m.execute("SELECT price, iv, alpha FROM alpha_logs WHERE symbol=? AND timestamp='2026-01-02 10:00:00'", (symbol,)).fetchone()
    print(f"1m: Price={res1m[0]}, IV={res1m[1]}, Alpha={res1m[2]}")
    conn1m.close()
    
    print("\n--- 1s Factors (Physical Read) ---")
    conn1s = sqlite3.connect(db_1s)
    # 使用您确认的字段名 'ts'
    res1s = conn1s.execute("SELECT price, iv, alpha FROM alpha_logs WHERE symbol=? AND ts=?", (symbol, ts_sec)).fetchone()
    if res1s:
        print(f"1s: Price={res1s[0]}, IV={res1s[1]}, Alpha={res1s[2]}")
    else:
        print("1s data NOT found at this TS!")
    conn1s.close()

if __name__ == "__main__":
    check_raw_factors()

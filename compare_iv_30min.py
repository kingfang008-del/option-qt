import sqlite3
import pandas as pd
from datetime import datetime, timedelta

def compare_nvada_30min():
    db_1m = "/Users/fangshuai/Documents/GitHub/option-qt/production/preprocess/backtest/minute/history_sqlite_1m/market_20260102.db"
    db_1s = "/Users/fangshuai/Documents/GitHub/option-qt/production/preprocess/backtest/second/history_sqlite_1s/market_20260102.db"
    symbol = 'NVDA'
    
    # 修正时间段：从 10:00:00 开始对比 30 分钟
    start_dt = datetime.strptime("2026-01-02 10:00:00", "%Y-%m-%d %H:%M:%S")
    time_points = [start_dt + timedelta(minutes=i) for i in range(31)]
    
    results = []
    
    try:
        conn1m = sqlite3.connect(db_1m)
        conn1s = sqlite3.connect(db_1s)
        
        for dt in time_points:
            ts_str = dt.strftime("%Y-%m-%d %H:%M:%S")
            ts_sec = float(dt.timestamp())
            
            # 1m 查询
            q1m = f"SELECT iv, alpha, price FROM alpha_logs WHERE symbol='{symbol}' AND timestamp='{ts_str}'"
            df1m = pd.read_sql(q1m, conn1m)
            
            # 1s 查询
            q1s = f"SELECT iv, alpha, price FROM alpha_logs WHERE symbol='{symbol}' AND timestamp={ts_sec}"
            df1s = pd.read_sql(q1s, conn1s)
            
            if not df1m.empty and not df1s.empty:
                row = {
                    'time': ts_str,
                    'iv_1m': df1m.iloc[0]['iv'],
                    'iv_1s': df1s.iloc[0]['iv'],
                    'p_1m': df1m.iloc[0]['price'],
                    'p_1s': df1s.iloc[0]['price'],
                    'alpha_1m': df1m.iloc[0]['alpha'],
                    'alpha_1s': df1s.iloc[0]['alpha']
                }
                results.append(row)
            
        conn1m.close()
        conn1s.close()
            
    except Exception as e:
        print(f"Error: {e}")
        return

    if not results:
        print("No intersecting data found between 10:00 and 10:30.")
        return

    df_res = pd.DataFrame(results)
    df_res['iv_diff_bps'] = (df_res['iv_1m'] - df_res['iv_1s']) * 10000
    df_res['alpha_diff'] = df_res['alpha_1m'] - df_res['alpha_1s']
    
    print(f"\nComparing NVDA IV and Alpha (10:00 - 10:30)")
    print(f"{'Time':<20} | {'IV 1m':<10} | {'IV 1s':<10} | {'Alpha 1m':<8} | {'Alpha 1s':<8} | {'Diff-BPS':<10}")
    print("-" * 95)
    for _, row in df_res.iterrows():
        print(f"{row['time']:<20} | {row['iv_1m']:.6f} | {row['iv_1s']:.6f} | {row['alpha_1m']:.4f} | {row['alpha_1s']:.4f} | {row['iv_diff_bps']:.4f}")

if __name__ == "__main__":
    compare_nvada_30min()

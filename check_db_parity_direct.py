import sqlite3
import pandas as pd

def check_parity_in_single_db():
    # 路径使用您指定的 1s 数据库
    db_path = "production/preprocess/backtest/history_sqlite_1s/market_20260102.db"
    conn = sqlite3.connect(db_path)
    
    symbol = 'NVDA'
    # 2026-01-02 10:00:00 的秒级时间戳
    ts_val = 1767366000.0
    
    print(f"Checking Parity in DB: {db_path} for {symbol} at {ts_val}")
    
    # 1. 查询 alpha_logs (这是 1s 引擎计算后的产物)
    query_log = "SELECT ts, alpha, iv, price FROM alpha_logs WHERE symbol=? AND ts=?"
    log_data = conn.execute(query_log, (symbol, ts_val)).fetchone()
    
    # 2. 查询 option_snapshots_1m (这是 1m 引擎提供的输入/原始基准)
    # 注意：表名和字段可能略有不同，根据实际 schema 调整
    try:
        # 尝试查询该时刻对应的分钟级快照数据
        query_snap = "SELECT ticker, close, iv FROM option_snapshots_1m WHERE ticker LIKE ? AND timestamp='2026-01-02 10:00:00' LIMIT 5"
        snap_data = pd.read_sql_query(query_snap, conn, params=(f"{symbol}%",))
        
        print("\n--- 1s Engine Result (from alpha_logs) ---")
        if log_data:
            print(f"TS: {log_data[0]}, Alpha: {log_data[1]}, IV: {log_data[2]}, Price: {log_data[3]}")
        else:
            print("No 1s log found.")

        print("\n--- 1m Input Reference (from option_snapshots_1m) ---")
        if not snap_data.empty:
            print(snap_data)
        else:
            print("No 1m snapshot found in this DB.")
            
    except Exception as e:
        print(f"Error querying snapshots: {e}")
        # 如果表名不对，列出所有表
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table';")
        print("Available tables:", [r[0] for r in cursor.fetchall()])

    conn.close()

if __name__ == "__main__":
    check_parity_in_single_db()

import sqlite3, pandas as pd
import numpy as np

def check_giants_on_0106():
    db_path = "history_sqlite_1m/market_20260106.db"
    conn = sqlite3.connect(db_path)
    
    giants = ['AAPL', 'AMZN', 'GOOGL', 'META', 'MSFT', 'NVDA', 'TSLA', 'AVGO', 'ADBE']
    ts_0935 = 1767710100.0
    
    print(f"--- Mag 7 + Giants Status at 09:35 (Jan 06) ---")
    print(f"{'Symbol':<8} | {'Alpha':>8} | {'Price':>8} | {'Stab(5m)':>8}")
    print("-" * 45)
    
    for sym in giants:
        df_a = pd.read_sql(f"SELECT ts, alpha FROM alpha_logs WHERE symbol = '{sym}' AND ts <= {ts_0935} ORDER BY ts DESC LIMIT 5", conn)
        if not df_a.empty:
            alpha_now = df_a.iloc[0]['alpha']
            stab = sum((df_a['alpha'] * np.sign(alpha_now) >= 1.45)) if abs(alpha_now) >= 1.45 else 0
            df_p = pd.read_sql(f"SELECT close FROM market_bars_1m WHERE symbol = '{sym}' AND ts = {ts_0935}", conn)
            price = df_p.iloc[0]['close'] if not df_p.empty else 0
            print(f"{sym:<8} | {alpha_now:>8.2f} | {price:>8.2f} | {int(stab):>8}")
            
    conn.close()

if __name__ == "__main__":
    check_giants_on_0106()

import sqlite3, pandas as pd
from datetime import datetime
import pytz

def analyze_nvda_dynamics():
    db_path = "history_sqlite_1m/market_20260106.db"
    conn = sqlite3.connect(db_path)
    df_a = pd.read_sql("SELECT ts, alpha FROM alpha_logs WHERE symbol = 'NVDA'", conn)
    df_p = pd.read_sql("SELECT ts, close FROM market_bars_1m WHERE symbol = 'NVDA'", conn)
    df = pd.merge(df_a, df_p, on='ts', how='inner').sort_values('ts')
    ny_tz = pytz.timezone('America/New_York')
    df['time'] = df['ts'].apply(lambda t: datetime.fromtimestamp(t, tz=pytz.utc).astimezone(ny_tz).strftime('%H:%M:%S'))
    df['roc_1m'] = df['close'].pct_change(1)
    df = df[df['time'] >= '09:30:00']
    
    print(f"--- NVDA Dynamics on Jan 06 ---")
    print(f"{'Time':<10} | {'Alpha':>8} | {'Price':>8} | {'1m ROC':>8}")
    print("-" * 45)
    for idx, row in df.head(15).iterrows():
        print(f"{row['time']:<10} | {row['alpha']:>8.2f} | {row['close']:>8.2f} | {row['roc_1m']:>8.2%}")
    conn.close()

if __name__ == "__main__":
    analyze_nvda_dynamics()

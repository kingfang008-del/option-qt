import sqlite3, pandas as pd
from datetime import datetime
import pytz

def analyze_amzn_dynamics():
    db_path = "history_sqlite_1m/market_20260106.db"
    conn = sqlite3.connect(db_path)
    
    # 1. Fetch Alpha & Price
    df_a = pd.read_sql("SELECT ts, alpha FROM alpha_logs WHERE symbol = 'AMZN'", conn)
    df_p = pd.read_sql("SELECT ts, close FROM market_bars_1m WHERE symbol = 'AMZN'", conn)
    
    df = pd.merge(df_a, df_p, on='ts', how='inner').sort_values('ts')
    
    ny_tz = pytz.timezone('America/New_York')
    def ts_to_ny(ts):
        return datetime.fromtimestamp(ts, tz=pytz.utc).astimezone(ny_tz).strftime('%H:%M:%S')

    df['time'] = df['ts'].apply(ts_to_ny)
    
    # Calculate ROC (1 minute and 5 minute)
    df['roc_1m'] = df['close'].pct_change(1)
    df['roc_5m'] = df['close'].pct_change(5)
    
    # Filter for session
    df = df[df['time'] >= '09:30:00']
    
    print(f"--- AMZN Dynamics on Jan 06 ---")
    print(f"{'Time':<10} | {'Alpha':>8} | {'Price':>8} | {'1m ROC':>8} | {'5m ROC':>8}")
    print("-" * 55)
    
    # Show entries around 09:35 and the trend thereafter
    for idx, row in df.head(30).iterrows(): # First 30 mins
        print(f"{row['time']:<10} | {row['alpha']:>8.2f} | {row['close']:>8.2f} | {row['roc_1m']:>8.2%} | {row['roc_5m']:>8.2%}")

    # Find peak price
    peak = df.loc[df['close'].idxmax()]
    print("-" * 55)
    print(f"PEAK REACHED AT {peak['time']} | Price: {peak['close']:.2f} | Alpha: {peak['alpha']:.2f}")

    conn.close()

if __name__ == "__main__":
    analyze_amzn_dynamics()

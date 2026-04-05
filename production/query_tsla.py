import psycopg2
import pandas as pd
from datetime import datetime
import sys
sys.path.append('/Users/fangshuai/Documents/GitHub/option-qt/production/baseline')
from config import PG_DB_URL, NY_TZ

def check_tsla():
    conn = psycopg2.connect(PG_DB_URL)
    
    now = datetime.now(NY_TZ)
    start_of_day = now.replace(hour=0, minute=0, second=0, microsecond=0)
    start_ts = int(start_of_day.timestamp())
    
    query = f"SELECT * FROM alpha_logs WHERE symbol='TSLA' AND ts >= {start_ts} ORDER BY ts ASC"
    df = pd.read_sql(query, conn)
    
    if not df.empty:
        print(f"Total TSLA alpha logs today: {len(df)}")
        print(f"Max alpha: {df['alpha'].max():.4f}, Min alpha: {df['alpha'].min():.4f}")
        df['dt'] = pd.to_datetime(df['ts'], unit='s').dt.tz_localize('UTC').dt.tz_convert('America/New_York')
        
        buy_signals = df[df['alpha'] > 0.85]
        print(f"Logs crossing ALPHA_ENTRY_THRESHOLD (0.85): {len(buy_signals)}")
        if not buy_signals.empty:
            print("--- Alpha > 0.85 Candidates ---")
            print(buy_signals[['dt', 'alpha', 'iv', 'price', 'vol_z']].head(20))
            
        print("\n--- Top 10 Highest Alpha Snapshots ---")
        print(df[['dt', 'alpha', 'iv', 'price', 'vol_z']].sort_values(by='alpha', ascending=False).head(10))
    else:
        print("No TSLA alpha logs found for today.")
    conn.close()

if __name__ == '__main__':
    check_tsla()

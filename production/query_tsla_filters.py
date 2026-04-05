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
        df['dt'] = pd.to_datetime(df['ts'], unit='s').dt.tz_localize('UTC').dt.tz_convert('America/New_York')
        buy_signals = df[(df['dt'].dt.hour == 9) & (df['dt'].dt.minute >= 50) & (df['dt'].dt.minute <= 58)]
        if not buy_signals.empty:
            print("--- TSLA 09:50 to 09:58 Log Details ---")
            
            # Print available columns to see what features are saved
            print("Available columns:", list(df.columns))
            
            # Print full rows for this duration to see why check_channel_a_momentum returned None
            pd.set_option('display.max_columns', None)
            pd.set_option('display.width', 1000)
            print(buy_signals)
    else:
        print("No TSLA alpha logs found for today.")
    conn.close()

if __name__ == '__main__':
    check_tsla()

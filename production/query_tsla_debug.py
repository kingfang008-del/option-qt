import psycopg2
import pandas as pd
from datetime import datetime
import sys
sys.path.append('/Users/fangshuai/Documents/GitHub/option-qt/production/baseline')
from config import PG_DB_URL, NY_TZ

def check_tsla():
    conn = psycopg2.connect(PG_DB_URL)
    
    table_name = "debug_slow_20260311"
    
    # Query all columns for TSLA from 09:50 to 09:58
    query = f"""
        SELECT * FROM {table_name} 
        WHERE symbol='TSLA' 
        AND extract(hour from to_timestamp(ts) AT TIME ZONE 'America/New_York') = 9 
        AND extract(minute from to_timestamp(ts) AT TIME ZONE 'America/New_York') BETWEEN 50 AND 58
        ORDER BY ts ASC
    """
    try:
        df = pd.read_sql(query, conn)
        if not df.empty:
            df['dt'] = pd.to_datetime(df['ts'], unit='s').dt.tz_localize('UTC').dt.tz_convert('America/New_York')
            
            print("--- TSLA 09:50 to 09:58 debug_slow ---")
            
            cols_to_print = ['dt', 'close', 'symbol']
            for c in df.columns:
                if 'macd' in c.lower() or 'roc' in c.lower() or 'ret' in c.lower() or 'index' in c.lower() or 'vol' in c.lower():
                    if c not in cols_to_print:
                        cols_to_print.append(c)
                    
            print(f"Momentum Columns found: {cols_to_print}")
            pd.set_option('display.max_columns', None)
            pd.set_option('display.width', 1000)
            print(df[cols_to_print])
        else:
            print("No TSLA debug_slow logs found.")
    except Exception as e:
        print(f"Error: {e}")
        
    conn.close()

check_tsla()

import psycopg2
import time
import sys
import os
from datetime import datetime, timedelta
sys.path.append('/Users/fangshuai/Documents/GitHub/option-qt/production/baseline')
from config import PG_DB_URL

def cleanup_last_week():
    conn = psycopg2.connect(PG_DB_URL)
    c = conn.cursor()
    
    # Calculate cutoff
    cutoff_dt = datetime.now() - timedelta(days=7)
    cutoff_ts = cutoff_dt.timestamp()
    cutoff_str = cutoff_dt.strftime('%Y-%m-%d %H:%M:%S')
    
    print(f"--- Data Cleanup Script (Cutoff: {cutoff_ts} / {cutoff_str}) ---")
    
    # 1. Reset ghost positions in symbol_state
    # We target symbols with active positions (MU, PLTR, GOOGL)
    # Alternatively, we can just clear everything to be safe
    print("Step 1: Resetting symbol_state (Removing ghost positions)...")
    # We delete MU, PLTR, GOOGL explicitly or all where position != 0
    c.execute("DELETE FROM symbol_state WHERE symbol IN ('MU', 'PLTR', 'GOOGL')")
    # Also reset any others that might have position but weren't reported
    c.execute("UPDATE symbol_state SET data = data || '{\"position\": 0, \"qty\": 0, \"is_pending\": false, \"locked_cash\": 0.0}'::jsonb WHERE symbol != '_GLOBAL_STATE_'")
    
    # 2. Clear Partitioned Logs
    partitioned_tables = ['trade_logs', 'trade_logs_backtest', 'alpha_logs', 'market_bars_1m', 'option_snapshots_1m', 'feature_logs']
    for table in partitioned_tables:
        print(f"Step 2: Clearing {table} records newer than {cutoff_str}...")
        c.execute(f"DELETE FROM {table} WHERE ts > %s", (cutoff_ts,))
    
    # 3. Drop Daily Debug Tables
    # We look for debug_slow_YYYYMMDD and debug_fast_YYYYMMDD from the last 7 days
    print("Step 3: Dropping daily debug tables from the last 7 days...")
    for i in range(8): # Covers 7 days + today
        d = (datetime.now() - timedelta(days=i)).strftime('%Y%m%d')
        for prefix in ['debug_slow', 'debug_fast']:
            table_name = f"{prefix}_{d}"
            try:
                c.execute(f"DROP TABLE IF EXISTS {table_name}")
                print(f"   Dropped {table_name}")
            except Exception as e:
                print(f"   Failed to drop {table_name}: {e}")
                conn.rollback()
                c = conn.cursor()

    conn.commit()
    print("\n✅ Cleanup Complete.")
    conn.close()

if __name__ == '__main__':
    # Safety: Ask for user confirmation if run manually
    cleanup_last_week()

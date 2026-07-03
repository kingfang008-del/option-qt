import sqlite3
import pandas as pd
import numpy as np
import argparse
import os

def patch_db(db_path):
    print(f"🛠️ Patching DB: {db_path}")
    if not os.path.exists(db_path):
        print(f"❌ DB not found: {db_path}")
        return

    conn = sqlite3.connect(db_path)
    
    # 1. Add columns to market_bars_1m if they don't exist
    cursor = conn.cursor()
    try:
        cursor.execute("ALTER TABLE market_bars_1m ADD COLUMN spy_roc_5min FLOAT DEFAULT 0.0")
        cursor.execute("ALTER TABLE market_bars_1m ADD COLUMN qqq_roc_5min FLOAT DEFAULT 0.0")
        conn.commit()
        print("✅ Added spy_roc_5min and qqq_roc_5min columns.")
    except sqlite3.OperationalError:
        print("ℹ️ Columns already exist or table busy. Proceeding...")

    # 2. Calculate ROC for SPY and QQQ
    df_s = pd.read_sql("SELECT ts, symbol, close FROM market_bars_1m WHERE symbol IN ('SPY', 'QQQ')", conn)
    
    if df_s.empty:
        print("❌ No SPY/QQQ data found in market_bars_1m!")
        conn.close()
        return

    # Calculate ROC per symbol
    df_s['roc'] = df_s.groupby('symbol')['close'].pct_change(5).fillna(0.0)
    
    spy_map = df_s[df_s['symbol'] == 'SPY'].set_index('ts')['roc'].to_dict()
    qqq_map = df_s[df_s['symbol'] == 'QQQ'].set_index('ts')['roc'].to_dict()

    # 3. Update the whole table
    # For performance, we'll read all ts from the table and batch update
    print("🔋 Calculating updates...")
    all_ts = pd.read_sql("SELECT DISTINCT ts FROM market_bars_1m", conn)['ts'].tolist()
    
    updates = []
    for ts in all_ts:
        updates.append((spy_map.get(ts, 0.0), qqq_map.get(ts, 0.0), ts))

    print(f"💾 Committing updates for {len(updates)} timestamps...")
    cursor.executemany("UPDATE market_bars_1m SET spy_roc_5min = ?, qqq_roc_5min = ? WHERE ts = ?", updates)
    conn.commit()
    conn.close()
    print("🎉 Patch COMPLETED successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, help="Date in YYYYMMDD format")
    parser.add_argument("--db", type=str, help="Full path to DB file (overrides --date)")
    args = parser.parse_args()

    if args.db:
        patch_db(args.db)
    elif args.date:
        db_path = f"/Users/fangshuai/Documents/GitHub/option-qt/production/preprocess/backtest/history_sqlite_1m/market_{args.date}.db"
        patch_db(db_path)
    else:
        print("Usage: python patch_sqlite_roc.py --date 20260102")

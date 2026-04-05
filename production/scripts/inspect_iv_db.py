import sqlite3
import json
import os
import sys

def check_db_iv(db_path, symbol):
    if not os.path.exists(db_path):
        print(f"❌ DB not found at {db_path}")
        return

    print(f"📊 Checking IV in DB: {db_path} for {symbol}")
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    # 1. Check alpha_logs (What was written)
    cur.execute(f"SELECT ts, alpha, iv, price FROM alpha_logs WHERE symbol='{symbol}' LIMIT 10")
    rows = cur.fetchall()
    print("\n--- alpha_logs (Recorded Alphas) ---")
    if not rows:
        print("Empty.")
    for r in rows:
        print(f" TS: {r[0]} | Alpha: {r[1]:.4f} | IV: {r[2]:.4f} | Price: {r[3]}")

    # 2. Check option_snapshots_1s (Source Raw Data)
    cur.execute(f"SELECT ts, buckets_json FROM option_snapshots_1s WHERE symbol='{symbol}' LIMIT 5")
    rows = cur.fetchall()
    print("\n--- option_snapshots_1s (Raw Buckets) ---")
    if not rows:
        print("Empty.")
    for r in rows:
        ts = r[0]
        buckets = json.loads(r[1]).get('buckets', [])
        print(f" TS: {ts}")
        for i, b in enumerate(buckets):
            # b[7] is IV
            iv = b[7] if len(b) > 7 else "N/A"
            vol = b[6] if len(b) > 6 else "N/A"
            print(f"   Bucket {i}: IV={iv} | Vol={vol}")

    conn.close()

if __name__ == "__main__":
    # Update these paths as needed on the server
    DB_PATH = "/home/kingfang007/quant_project/data/history_sqlite_1s/market_20260102.db"
    check_db_iv(DB_PATH, "NVDA")

import sqlite3
import os
import json
import time
from datetime import datetime
from pathlib import Path
import sys

# Setup Path
base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir)) # Add 'production' root
sys.path.append(str(base_dir / "DB"))
sys.path.append(str(base_dir / "baseline")) # Added for config

from DB.data_persistence_service_v8_sqlite import DataPersistenceServiceSQLite

def test_alpha_persistence():
    test_db_dir = "/tmp/test_persistence_v8"
    if not os.path.exists(test_db_dir):
        os.makedirs(test_db_dir)
    
    test_date = "20260327"
    db_file = os.path.join(test_db_dir, f"market_{test_date}.db")
    if os.path.exists(db_file):
        os.remove(db_file)

    print(f"🧪 Testing DataPersistenceServiceSQLite with DB: {db_file}")
    
    # 1. Initialize Service
    service = DataPersistenceServiceSQLite(start_date=test_date, db_dir=test_db_dir)
    
    # 2. Add Dummy Alpha Logs
    dummy_alphas = [
        # (ts, symbol, alpha, iv, price, vol_z)
        (1767366720.0, "NVDA", 0.5, 0.45, 191.5, 1.2),
        (1767366721.0, "TSLA", -0.2, 0.65, 245.0, -0.5),
        (1767366722.0, "AAPL", 0.1, 0.25, 180.2, 0.8)
    ]
    
    service.alpha_buffer.extend(dummy_alphas)
    
    print(f"📡 Buffer size before flush: {len(service.alpha_buffer)}")
    
    # 3. Flush
    service.flush()
    
    print(f"💾 Flush completed. Buffer size after flush: {len(service.alpha_buffer)}")
    
    # 4. Verify DB
    conn = sqlite3.connect(db_file)
    cur = conn.cursor()
    
    # Check table exists
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='alpha_logs'")
    if not cur.fetchone():
        print("❌ FAILED: Table 'alpha_logs' was not created!")
        return
    
    # Check data content
    cur.execute("SELECT * FROM alpha_logs")
    rows = cur.fetchall()
    
    print(f"📊 Rows found in alpha_logs: {len(rows)}")
    for r in rows:
        print(f"   Row: {r}")
        
    if len(rows) == len(dummy_alphas):
        print("✅ SUCCESS: Alpha logs persisted correctly!")
    else:
        print(f"❌ FAILED: Expected {len(dummy_alphas)} rows, found {len(rows)}")
    
    conn.close()

if __name__ == "__main__":
    test_alpha_persistence()

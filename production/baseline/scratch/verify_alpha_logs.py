import os
import sqlite3
import sys
from pathlib import Path

# Add project root to sys.path
# project root is /Users/fangshuai/Documents/GitHub/option-qt
project_root = str(Path(__file__).parent.parent.parent.parent)
sys.path.append(project_root)

from production.baseline.DAO.data_persistence_service_v8_sqlite import DataPersistenceServiceSQLite
from production.baseline.config import NY_TZ
from datetime import datetime

def verify_sqlite():
    current_date = datetime.now(NY_TZ).strftime('%Y%m%d')
    # Use a temporary DB for verification
    temp_db_dir = "/Users/fangshuai/Documents/GitHub/option-qt/production/baseline/scratch"
    os.makedirs(temp_db_dir, exist_ok=True)
    
    print(f"Initializing DataPersistenceServiceSQLite for date {current_date}...")
    service = DataPersistenceServiceSQLite(start_date=current_date, db_dir=temp_db_dir)
    
    db_path = os.path.join(temp_db_dir, f"market_{current_date}.db")
    print(f"Checking DB at {db_path}...")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("PRAGMA table_info(alpha_logs)")
    columns = {row[1] for row in cursor.fetchall()}
    
    required_cols = {'event_prob', 'spy_roc_5min', 'qqq_roc_5min'}
    missing = required_cols - columns
    
    if not missing:
        print("✅ SQLite: All required columns found in alpha_logs.")
    else:
        print(f"❌ SQLite: Missing columns in alpha_logs: {missing}")
    
    conn.close()

if __name__ == "__main__":
    verify_sqlite()

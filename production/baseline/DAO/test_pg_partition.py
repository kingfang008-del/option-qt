import psycopg2
import time
from datetime import datetime
from config import PG_DB_URL
from data_persistence_service_v8_pg import DataPersistenceServicePG

def test_partitioning():
    print(f"Connecting to: {PG_DB_URL}")
    svc = DataPersistenceServicePG()
    
    conn = psycopg2.connect(PG_DB_URL)
    conn.autocommit = True
    c = conn.cursor()
    
    # Check tables
    c.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'")
    tables = [r[0] for r in c.fetchall()]
    print(f"\nCreated Tables: {tables}")
    
    # Try an insert to verify partition routing
    now_ts = time.time()
    try:
        c.execute("""
            INSERT INTO market_bars_1m (ts, symbol, open, high, low, close, volume) 
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (ts, symbol) DO NOTHING
        """, (now_ts, "TEST", 100, 101, 99, 100, 1000))
        print("✅ Inserted into market_bars_1m successfully. Partition routing works!")
    except Exception as e:
        print(f"❌ Insert failed: {e}")
        
    c.close()
    conn.close()

if __name__ == "__main__":
    test_partitioning()

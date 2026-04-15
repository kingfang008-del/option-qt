import psycopg2
import sys
import os

# Hardcode PG_DB_URL to avoid dependency issues with config.py imports (like pytz)
PG_DB_URL = "dbname=quant_trade user=postgres password=postgres host=192.168.50.116 port=5432"

def cleanup_debug_tables():
    conn = None
    try:
        print(f"Connecting to database: {PG_DB_URL.split('password=')[0]}...")
        conn = psycopg2.connect(PG_DB_URL)
        conn.autocommit = True
        cur = conn.cursor()

        # 1. 查找所有匹配的前缀表
        # 包括基础表和所有分区表
        cur.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
              AND (table_name LIKE 'debug_slow%' OR table_name LIKE 'debug_fast%')
        """)
        
        tables = [row[0] for row in cur.fetchall()]
        
        if not tables:
            print("No debug_slow* or debug_fast* tables found.")
            return

        print(f"Found {len(tables)} tables to delete: {tables}")
        
        # 2. 依次删除
        for table in tables:
            try:
                print(f"Dropping table: {table}...")
                cur.execute(f"DROP TABLE IF EXISTS {table} CASCADE")
                print(f"✅ Successfully dropped {table}")
            except Exception as e:
                print(f"❌ Failed to drop {table}: {e}")

        print("\n✨ Cleanup finished.")
        
    except Exception as e:
        print(f"FATAL: Database connection error: {e}")
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    cleanup_debug_tables()

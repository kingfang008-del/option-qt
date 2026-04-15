import psycopg2
import os
from config import PG_DB_URL

def check_schema():
    try:
        conn = psycopg2.connect(PG_DB_URL)
        c = conn.cursor()
        c.execute("SELECT column_name FROM information_schema.columns WHERE table_name = 'market_bars_1m'")
        cols = [row[0] for row in c.fetchall()]
        print(f"Columns in market_bars_1m: {cols}")
        
        c.execute("SELECT column_name FROM information_schema.columns WHERE table_name = 'debug_slow'")
        cols_slow = [row[0] for row in c.fetchall()]
        print(f"Columns in debug_slow: {cols_slow}")
        
        conn.close()
    except Exception as e:
        print(f"Error checking schema: {e}")

if __name__ == "__main__":
    check_schema()

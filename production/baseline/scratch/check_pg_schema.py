import psycopg2
import os
from config import PG_DB_URL

def check_schema():
    conn = psycopg2.connect(PG_DB_URL)
    cur = conn.cursor()
    cur.execute("SELECT column_name FROM information_schema.columns WHERE table_name = 'market_bars_1m'")
    cols = [r[0] for r in cur.fetchall()]
    print(f"Columns in market_bars_1m: {cols}")
    
    cur.execute("SELECT * FROM market_bars_1m LIMIT 1")
    row = cur.fetchone()
    print(f"Sample row: {row}")
    
    conn.close()

if __name__ == "__main__":
    check_schema()

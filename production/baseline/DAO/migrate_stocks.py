import sqlite3
import psycopg2
import sys
from pathlib import Path

# Add parent directory to path to import config
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import PG_DB_URL

def migrate():
    sqlite_db_path = Path("/home/kingfang007/notebook/stocks.db")
    if not sqlite_db_path.exists():
        print(f"Error: {sqlite_db_path} does not exist.")
        sys.exit(1)

    print(f"Connecting to SQLite: {sqlite_db_path}")
    sl_conn = sqlite3.connect(sqlite_db_path)
    sl_cursor = sl_conn.cursor()

    print(f"Connecting to PostgreSQL...")
    pg_conn = psycopg2.connect(PG_DB_URL)
    pg_cursor = pg_conn.cursor()

    # 1. Create table in PostgreSQL
    print("Creating table stocks_us in PostgreSQL...")
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS stocks_us (
        id SERIAL PRIMARY KEY,
        symbol TEXT NOT NULL UNIQUE,
        company_name TEXT,
        Industry TEXT,
        level TEXT,
        sector TEXT,
        duplicate INTEGER
    );
    """
    pg_cursor.execute(create_table_sql)
    pg_conn.commit()

    # 2. Fetch data from SQLite
    print("Fetching data from SQLite `stocks_us`...")
    sl_cursor.execute("SELECT id, symbol, company_name, Industry, level, sector, duplicate FROM stocks_us")
    rows = sl_cursor.fetchall()
    print(f"Found {len(rows)} records in SQLite.")

    if len(rows) > 0:
        # 3. Insert into PostgreSQL
        print("Inserting records into PostgreSQL...")
        from psycopg2.extras import execute_batch
        
        insert_sql = """
        INSERT INTO stocks_us (id, symbol, company_name, Industry, level, sector, duplicate)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (id) DO UPDATE SET
            symbol = EXCLUDED.symbol,
            company_name = EXCLUDED.company_name,
            Industry = EXCLUDED.Industry,
            level = EXCLUDED.level,
            sector = EXCLUDED.sector,
            duplicate = EXCLUDED.duplicate;
        """
        execute_batch(pg_cursor, insert_sql, rows)
        pg_conn.commit()
        
        # 4. Sync the sequence so next insert gets correct ID
        sync_seq_sql = """
        SELECT setval(pg_get_serial_sequence('stocks_us', 'id'), coalesce(max(id),0) + 1, false) FROM stocks_us;
        """
        pg_cursor.execute(sync_seq_sql)
        pg_conn.commit()
        print(f"Migration completed successfully! Inserted {len(rows)} rows.")
    else:
        print("No records found in SQLite stocks_us table.")

    sl_cursor.close()
    sl_conn.close()
    pg_cursor.close()
    pg_conn.close()

if __name__ == "__main__":
    migrate()

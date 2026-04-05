import sqlite3
import psycopg2
import sys
import os
from pathlib import Path
from psycopg2.extras import execute_batch

# Add baseline to path to import config
sys.path.append(str(Path(__file__).resolve().parent.parent / "baseline"))
try:
    from config import PG_DB_URL, TARGET_SYMBOLS
except ImportError:
    PG_DB_URL = "dbname=quant_trade user=postgres password=postgres host=192.168.50.116 port=5432"
    TARGET_SYMBOLS = []

# Approved Sector Mapping
SECTOR_MAP = {
    # 1: Crypto
    'MSTR': 1, 'COIN': 1,
    # 2: Index/ETF
    'VIXY': 2, 'SPY': 2, 'QQQ': 2, 'IWM': 2, 'GLD': 2,
    # 3: Semiconductor
    'NVDA': 3, 'AMD': 3, 'MU': 3, 'INTC': 3, 'SMCI': 3, 'AVGO': 3,
    # 4: Big Tech
    'AAPL': 4, 'META': 4, 'AMZN': 4, 'MSFT': 4, 'GOOGL': 4, 'ADBE': 4, 'CRM': 4, 'ORCL': 4, 'NFLX': 4,
    # 5: EV
    'TSLA': 5,
    # 6: Healthcare
    'UNH': 6,
    # 7: Financial
    'GS': 7,
    # 8: Consumer
    'WMT': 8, 'NKE': 8,
    # 9: Tech Growth
    'PLTR': 9, 'APP': 9, 'HOOD': 9, 'CRWV': 9, 'DELL': 9,
    # 10: Energy
    'XOM': 10
}

def migrate():
    current_dir = Path(__file__).resolve().parent
    sqlite_db_path = current_dir / "stocks.db"
    
    if not sqlite_db_path.exists():
        print(f"❌ Error: {sqlite_db_path} does not exist.")
        sys.exit(1)

    print(f"🔍 Connecting to SQLite: {sqlite_db_path}")
    sl_conn = sqlite3.connect(sqlite_db_path)
    sl_cursor = sl_conn.cursor()

    print(f"🐘 Connecting to PostgreSQL...")
    try:
        pg_conn = psycopg2.connect(PG_DB_URL)
        pg_cursor = pg_conn.cursor()
    except Exception as e:
        print(f"❌ PG Connection Error: {e}")
        sys.exit(1)

    # 1. Create table in PostgreSQL with sector_id
    print("🔨 Creating table stocks_us in PostgreSQL with sector_id...")
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS stocks_us (
        id SERIAL PRIMARY KEY,
        symbol TEXT NOT NULL UNIQUE,
        company_name TEXT,
        Industry TEXT,
        level TEXT,
        sector TEXT,
        sector_id INTEGER DEFAULT 0,
        duplicate INTEGER
    );
    """
    pg_cursor.execute(create_table_sql)
    pg_conn.commit()

    # 2. Fetch data from SQLite
    print("📥 Fetching data from SQLite `stocks_us`...")
    sl_cursor.execute("SELECT id, symbol, company_name, Industry, level, sector, duplicate FROM stocks_us")
    rows = sl_cursor.fetchall()
    print(f"✅ Found {len(rows)} records in SQLite.")

    if len(rows) > 0:
        # 3. Process rows with sector_id
        processed_rows = []
        for r in rows:
            # id, symbol, company_name, Industry, level, sector, duplicate
            row_list = list(r)
            sym = row_list[1]
            sid = SECTOR_MAP.get(sym, 0)
            
            # Map Industry if it's one of our target sectors
            if sym in SECTOR_MAP:
                mapping = {
                    1: 'Crypto', 2: 'Index', 3: 'Semis', 4: 'BigTech', 5: 'EV',
                    6: 'Health', 7: 'Finance', 8: 'Consumer', 9: 'Growth', 10: 'Energy'
                }
                row_list[3] = mapping.get(sid, row_list[3]) # Update Industry
                row_list[5] = mapping.get(sid, row_list[5]) # Update Sector string
            
            # Add sector_id at the end (for matching INSERT columns)
            # id, symbol, company_name, Industry, level, sector, duplicate, sector_id
            row_list.append(sid)
            processed_rows.append(tuple(row_list))

        # 4. Insert into PostgreSQL
        print("📤 Inserting/Updating records into PostgreSQL...")
        
        insert_sql = """
        INSERT INTO stocks_us (id, symbol, company_name, Industry, level, sector, duplicate, sector_id)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (symbol) DO UPDATE SET
            id = EXCLUDED.id,
            company_name = EXCLUDED.company_name,
            Industry = EXCLUDED.Industry,
            level = EXCLUDED.level,
            sector = EXCLUDED.sector,
            sector_id = EXCLUDED.sector_id,
            duplicate = EXCLUDED.duplicate;
        """
        execute_batch(pg_cursor, insert_sql, processed_rows)
        pg_conn.commit()
        
        # 5. Sync the sequence
        sync_seq_sql = """
        SELECT setval(pg_get_serial_sequence('stocks_us', 'id'), coalesce(max(id),0) + 1, false) FROM stocks_us;
        """
        pg_cursor.execute(sync_seq_sql)
        pg_conn.commit()
        print(f"🎉 Migration completed successfully! Synchronized {len(processed_rows)} rows.")
        
        # 6. Verify specific symbols
        print("\n🧪 Verification:")
        test_syms = ['MSTR', 'COIN', 'NVDA', 'AAPL', 'TSLA']
        pg_cursor.execute("SELECT symbol, sector_id, Industry FROM stocks_us WHERE symbol = ANY(%s)", (test_syms,))
        for sym, s_id, ind in pg_cursor.fetchall():
            print(f"   - {sym}: sector_id={s_id}, Industry={ind}")
            
    else:
        print("⚠️ No records found in SQLite stocks_us table.")

    sl_cursor.close()
    sl_conn.close()
    pg_cursor.close()
    pg_conn.close()

if __name__ == "__main__":
    migrate()

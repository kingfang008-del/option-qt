import sqlite3
import psycopg2
import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from production.config import PG_DB_URL, DB_DIR

def verify_alpha(date_str, symbols=None):
    sqlite_path = DB_DIR / f"market_{date_str}.db"
    if not sqlite_path.exists():
        print(f"❌ SQLite DB not found: {sqlite_path}")
        return

    print(f"🔍 Comparing Alpha for {date_str}...")
    
    # 1. Load from SQLite
    conn_sl = sqlite3.connect(str(sqlite_path))
    query_sl = "SELECT symbol, ts, alpha, iv, price FROM alpha_logs"
    if symbols:
        sym_str = "','".join(symbols)
        query_sl += f" WHERE symbol IN ('{sym_str}')"
    df_sl = pd.read_sql_query(query_sl, conn_sl)
    conn_sl.close()
    
    if df_sl.empty:
        print("❌ No Alpha logs found in SQLite.")
        return

    # 2. Load from Postgres
    conn_pg = psycopg2.connect(PG_DB_URL)
    query_pg = "SELECT symbol, ts, alpha, iv, price FROM alpha_logs"
    # Convert ts from PG (usually timestamp) to float if needed, but our PG schema uses numeric/float for ts
    if symbols:
        sym_str = "','".join(symbols)
        query_pg += f" WHERE symbol IN ('{sym_str}')"
    df_pg = pd.read_sql_query(query_pg, conn_pg)
    conn_pg.close()

    if df_pg.empty:
        print("❌ No Alpha logs found in Postgres.")
        return

    # 3. Merge and Compare
    # Ensure types match
    df_sl['ts'] = df_sl['ts'].astype(float).round(0)
    df_pg['ts'] = df_pg['ts'].astype(float).round(0)
    
    merged = pd.merge(df_sl, df_pg, on=['symbol', 'ts'], suffixes=('_sl', '_pg'))
    
    if merged.empty:
        print("❌ No overlapping timestamps found between SQLite and Postgres.")
        return

    merged['alpha_diff'] = (merged['alpha_sl'] - merged['alpha_pg']).abs()
    
    mismatches = merged[merged['alpha_diff'] > 1e-6]
    
    print(f"📊 Total Overlapping Samples: {len(merged)}")
    print(f"✅ Exact Matches: {len(merged) - len(mismatches)}")
    print(f"❌ Mismatches: {len(mismatches)}")
    
    if not mismatches.empty:
        print("\nTop Mismatches:")
        print(mismatches.sort_values('alpha_diff', ascending=False).head(10))
        
        avg_diff = merged['alpha_diff'].mean()
        max_diff = merged['alpha_diff'].max()
        print(f"\nMean Absolute Difference: {avg_diff:.8f}")
        print(f"Max Absolute Difference: {max_diff:.8f}")
    else:
        print("\n✨ Perfect Consistency Achieved!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, required=True, help="Date in YYYYMMDD format")
    parser.add_argument("--symbols", type=str, help="Comma separated symbols")
    args = parser.parse_args()
    
    syms = args.symbols.split(',') if args.symbols else None
    verify_alpha(args.date, syms)

import sqlite3
import pandas as pd
import numpy as np
import os
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("TradeVerify")

def load_trades_from_sqlite(db_path):
    if not os.path.exists(db_path):
        logger.error(f"Database not found: {db_path}")
        return None
    
    conn = sqlite3.connect(db_path)
    # 提取成交记录
    query = """
    SELECT ts as entry_ts, symbol, direction, entry_price, exit_price, qty, pnl, reason
    FROM trade_logs
    WHERE action IN ('SELL', 'BUY_TO_CLOSE', 'SELL_TO_CLOSE')
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def main():
    parser = argparse.ArgumentParser(description="Compare Online vs Offline Trade Consistency")
    parser.add_argument("--ref", required=True, help="Path to reference trades (CSV or SQLite)")
    parser.add_argument("--sha", required=True, help="Path to shadow LIVEREPLAY trades (SQLite DB)")
    
    args = parser.parse_args()
    
    logger.info(f"Comparing Reference Trades vs Shadow trades...")
    
    # Load Shadow
    df_sha = load_trades_from_sqlite(args.sha)
    
    # Load Reference
    if args.ref.endswith(".csv"):
        df_ref = pd.read_csv(args.ref)
    elif args.ref.endswith(".db") or args.ref.endswith(".sqlite"):
        df_ref = load_trades_from_sqlite(args.ref)
    else:
        logger.error("Unsupported reference format. Use .csv or .db")
        return

    if df_ref is None or df_sha is None:
        logger.error("Failed to load trade data.")
        return

    # Align by symbol and rough entry time (allowing small drift)
    # For simplicity in this tool, we assume timestamps match closely
    df_ref = df_ref.sort_values(['entry_ts', 'symbol'])
    df_sha = df_sha.sort_values(['entry_ts', 'symbol'])
    
    logger.info(f"Ref Trades: {len(df_ref)} | Sha Trades: {len(df_sha)}")
    
    # Merge and compare
    # Using a merge with a tolerance on entry_ts if needed, but here let's try exact first
    merged = pd.merge(df_ref, df_sha, on=['symbol'], suffixes=('_ref', '_sha'))
    
    # Filter by time proximity (e.g., within 5 minutes)
    merged = merged[abs(merged['entry_ts_ref'] - merged['entry_ts_sha']) < 300]
    
    if merged.empty:
        logger.warning("No matching trades found by symbol and time proximity!")
        return

    # Analyze deviations
    merged['pnl_diff'] = merged['pnl_sha'] - merged['pnl_ref']
    merged['price_diff'] = merged['exit_price_sha'] - merged['exit_price_ref']
    
    avg_pnl_err = merged['pnl_diff'].abs().mean()
    logger.info(f"Matched {len(merged)} trades.")
    logger.info(f"Average PnL absolute deviation: ${avg_pnl_err:.2f}")
    
    for _, row in merged.iterrows():
        pnl_err = abs(row['pnl_diff'])
        if pnl_err > 10.0: # Deviation threshold
             logger.warning(f"❌ [Mismatch] {row['symbol']} @ {row['entry_ts_ref']}: Ref PnL={row['pnl_ref']:.2f}, Sha PnL={row['pnl_sha']:.2f} (Diff: {row['pnl_diff']:.2f})")
        else:
             logger.info(f"✅ [Match] {row['symbol']} @ {row['entry_ts_ref']} matches within threshold.")

if __name__ == "__main__":
    main()

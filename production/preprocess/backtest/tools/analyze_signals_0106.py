import sqlite3, pandas as pd, json

def analyze_0106_signals():
    db_path = "history_sqlite_1m/market_20260106.db"
    conn = sqlite3.connect(db_path)
    # Get 09:35 signals
    df = pd.read_sql("SELECT ts, symbol, alpha, vol_z FROM alpha_logs WHERE ts = 1767690300", conn)
    # Get prices for ROI calculation (EOD or subsequent peaks)
    # For now, just look at the signals
    
    # Let's also look at the return of these signals by the end of the day or a fixed window
    # To see which one was the true winner
    
    df_m = pd.read_sql("SELECT ts, symbol, close FROM market_bars_1m", conn)
    
    # Function to get ROI for a symbol starting from 09:35
    def get_roi(sym, start_ts):
        start_price = df_m[(df_m['ts'] == start_ts) & (df_m['symbol'] == sym)]['close'].values
        if len(start_price) == 0: return 0.0
        sp = start_price[0]
        # Max price for the rest of the day
        rest = df_m[(df_m['ts'] > start_ts) & (df_m['symbol'] == sym)]['close']
        if len(rest) == 0: return 0.0
        # Return at 16:00
        eod_p = rest.iloc[-1]
        return (eod_p - sp) / sp

    df['final_roi'] = df['symbol'].apply(lambda s: get_roi(s, 1767690300))
    df['abs_alpha'] = df['alpha'].abs()
    
    # Sort by absolute alpha
    print("--- Sorted by ABS Alpha (Current Logic) ---")
    print(df.sort_values('abs_alpha', ascending=False)[['symbol', 'alpha', 'final_roi']])
    
    # Sort by ROM (Return on Alpha?)
    print("\n--- Sorted by Alphabetical (Prev 'Lucky' Logic) ---")
    print(df.sort_values('symbol')[['symbol', 'alpha', 'final_roi']])
    
    conn.close()

if __name__ == "__main__":
    analyze_0106_signals()

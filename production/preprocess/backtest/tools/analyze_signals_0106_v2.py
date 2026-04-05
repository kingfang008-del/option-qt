import sqlite3, pandas as pd, json

def analyze_0106_signals():
    db_path = "history_sqlite_1m/market_20260106.db"
    conn = sqlite3.connect(db_path)
    
    # Get 09:35 signals
    start_ts = 1767710100
    df_raw = pd.read_sql(f"SELECT ts, symbol, alpha, vol_z FROM alpha_logs WHERE ts = {start_ts}", conn)
    
    df_m = pd.read_sql("SELECT ts, symbol, close FROM market_bars_1m", conn)
    
    def get_roi(sym, start_ts):
        start_price = df_m[(df_m['ts'] == start_ts) & (df_m['symbol'] == sym)]['close'].values
        if len(start_price) == 0: return 0.0
        sp = start_price[0]
        # Return at 16:00
        rest = df_m[(df_m['ts'] > start_ts) & (df_m['symbol'] == sym)]['close']
        if len(rest) == 0: return 0.0
        eod_p = rest.iloc[-1]
        return (eod_p - sp) / sp

    df_raw['final_roi'] = df_raw['symbol'].apply(lambda s: get_roi(s, start_ts))
    df_raw['abs_alpha'] = df_raw['alpha'].abs()
    
    print(f"--- All Signals at 09:35 (Jan 06) ---")
    print(df_raw.sort_values('abs_alpha', ascending=False)[['symbol', 'alpha', 'final_roi']])
    
    conn.close()

if __name__ == "__main__":
    analyze_0106_signals()

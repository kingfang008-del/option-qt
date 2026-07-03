import sqlite3
import pandas as pd
import numpy as np
import os
from datetime import datetime

# Configuration
DB_DIR = "production/preprocess/backtest/history_sqlite_1m"
# Auto-detect all dates from market_*.db files
import glob
files = glob.glob(os.path.join(DB_DIR, "market_*.db"))
TARGET_DATES = sorted([os.path.basename(f).replace("market_", "").replace(".db", "") for f in files])
PROXY_SYMBOL = "VIXY"

def load_data(date_str, table_name):
    db_path = os.path.join(DB_DIR, f"market_{date_str}.db")
    if not os.path.exists(db_path):
        print(f"Warning: {db_path} not found.")
        return None
    
    conn = sqlite3.connect(db_path)
    query = f"SELECT * FROM {table_name}"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def analyze_date(date_str):
    print(f"--- Analyzing {date_str} ---")
    
    # 1. Volatility Data (VIXY)
    bars_df = load_data(date_str, "market_bars_1m")
    if bars_df is None or bars_df.empty:
        return None
    
    vixy_df = bars_df[bars_df['symbol'] == PROXY_SYMBOL].sort_values('ts')
    if vixy_df.empty:
        print(f"No {PROXY_SYMBOL} data for {date_str}")
        return None

    # VIXY specific metrics
    vixy_df['ret_1m'] = vixy_df['close'].pct_change()
    open_price = vixy_df.iloc[0]['open']
    vixy_df['chg_from_open'] = (vixy_df['close'] - open_price) / open_price
    
    max_chg_from_open = vixy_df['chg_from_open'].max()
    avg_vol_1m = vixy_df['ret_1m'].abs().mean()
    
    # 5m Vol Spike (Rolling 5m absolute returns sum)
    vixy_df['vol_spike_5m'] = vixy_df['ret_1m'].abs().rolling(5).sum()
    max_vol_spike_5m = vixy_df['vol_spike_5m'].max()

    # 2. Market Choppiness (Reversals)
    # Define reversal: Up > 0.1% then Down > 0.1% within 5 mins (or vice versa)
    # Simpler version: count number of times sign of return changes and both returns > thresh
    thresh = 0.001 # 0.1%
    vixy_df['prev_ret'] = vixy_df['ret_1m'].shift(1)
    reversals = vixy_df[
        (vixy_df['ret_1m'] * vixy_df['prev_ret'] < 0) & 
        (vixy_df['ret_1m'].abs() > thresh) & 
        (vixy_df['prev_ret'].abs() > thresh)
    ]
    num_reversals = len(reversals)

    # 3. Alpha Score Divergence
    alpha_df = load_data(date_str, "alpha_logs")
    if alpha_df is not None and not alpha_df.empty:
        # Match alpha with future returns (15m)
        # We need the price bars again for all symbols if we want full correlation, 
        # but let's stick to the symbol matched in alpha_logs.
        
        # Merge alpha with price to get forward returns
        # Sort and merge_asof is best for TS data
        alpha_df = alpha_df.sort_values('ts')
        
        # Calculate forward returns for all symbols in bars_df
        bars_df = bars_df.sort_values(['symbol', 'ts'])
        bars_df['fwd_ret_15m'] = bars_df.groupby('symbol')['close'].shift(-15).pct_change(15).shift(15) 
        # Wait, pct_change(15).shift(-15) is better for forward return
        bars_df['fwd_ret_15m'] = bars_df.groupby('symbol')['close'].pct_change(periods=-15).mul(-1) # approximate
        # Let's do it properly:
        bars_df['price_15m_later'] = bars_df.groupby('symbol')['close'].shift(-15)
        bars_df['fwd_ret_15m'] = (bars_df['price_15m_later'] - bars_df['close']) / bars_df['close']
        
        # Merge
        merged = pd.merge(alpha_df, bars_df[['symbol', 'ts', 'fwd_ret_15m']], on=['symbol', 'ts'], how='inner')
        corr = merged['alpha'].corr(merged['fwd_ret_15m'])
        alpha_std = merged['alpha'].std()
    else:
        corr = np.nan
        alpha_std = np.nan

    return {
        'date': date_str,
        'max_chg_open': max_chg_from_open,
        'avg_vol_1m': avg_vol_1m,
        'max_vol_spike_5m': max_vol_spike_5m,
        'num_reversals': num_reversals,
        'alpha_corr_15m': corr,
        'alpha_std': alpha_std
    }

if __name__ == "__main__":
    results = []
    for d in TARGET_DATES:
        res = analyze_date(d)
        if res:
            results.append(res)
    
    summary_df = pd.DataFrame(results)
    print("\n=== Market Regime Comparison Summary ===")
    print(summary_df.to_string(index=False))
    
    # Save to CSV for reference
    summary_df.to_csv("production/baseline/comparison_results.csv", index=False)

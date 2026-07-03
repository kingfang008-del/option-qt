import sqlite3
import pandas as pd
import numpy as np
import json
import sys
import os
from pathlib import Path
from datetime import datetime
import pytz

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / "production" / "baseline"))

# Mock heavy dependencies before imports
import unittest.mock
from unittest.mock import MagicMock

# Create mock structure
mock_scipy = MagicMock()
mock_stats = MagicMock()
mock_scipy.stats = mock_stats
sys.modules['scipy'] = mock_scipy
sys.modules['scipy.stats'] = mock_stats

mock_torch = MagicMock()
mock_device = MagicMock()
mock_torch.device.return_value = mock_device
mock_torch.cuda.is_available.return_value = False
sys.modules['torch'] = mock_torch
sys.modules['torch.nn'] = MagicMock()
sys.modules['torch.nn.functional'] = MagicMock()
sys.modules['torch.optim'] = MagicMock()

sys.modules['py_vollib_vectorized'] = MagicMock()
sys.modules['psycopg2'] = MagicMock()
sys.modules['redis'] = MagicMock()
sys.modules['trading_tft_stock_embed'] = MagicMock()
sys.modules['AdvancedAlphaNet'] = MagicMock()

from signal_engine_v8 import SignalEngineV8, SymbolState

def audit_nvda_features():
    # Corrected path to verified local DB
    db_path = PROJECT_ROOT / "production" / "preprocess" / "backtest" / "history_sqlite_1s" / "market_20260102.db"
    parquet_path = PROJECT_ROOT / "production" / "preprocess" / "backtest" / "rl_feed_parquet_batch" / "NVDA.parquet"

    if not db_path.exists():
        print(f"❌ DB not found: {db_path}")
        return
    if not parquet_path.exists():
        print(f"❌ Parquet not found: {parquet_path}")
        return

    print(f"📂 [Audit] Loading SQLite DB: {db_path}")
    print(f"📂 [Audit] Loading Parquet Baseline: {parquet_path}")

    # 1. Load Parquet Baseline
    df_parquet = pd.read_parquet(parquet_path)
    df_parquet = df_parquet[df_parquet['symbol'] == 'NVDA']
    # Filter for first 5 mins for display
    # 09:30 is 1767364200
    
    # 2. Extract 1s Data for NVDA
    start_ts = 1767364200 # 09:30:00
    end_ts = 1767366300   # 10:05:00
    
    db_path_1s = db_path
    db_path_1m = PROJECT_ROOT / "production" / "preprocess" / "backtest" / "history_sqlite_1m" / "market_20260102.db"
    
    print(f"📂 [Audit] 1s DB: {db_path_1s}")
    print(f"📂 [Audit] 1m DB: {db_path_1m}")

    conn_1s = sqlite3.connect(db_path_1s)
    conn_1m = sqlite3.connect(db_path_1m)
    
    # 2. Extract Data
    print(f"🔍 [Query] Extracting 1s prices and 1m options...")
    df_b1 = pd.read_sql(f"SELECT * FROM market_bars_1s WHERE symbol='NVDA' AND ts >= {start_ts} AND ts <= {end_ts} ORDER BY ts ASC", conn_1s)
    df_o1 = pd.read_sql(f"SELECT * FROM option_snapshots_1m WHERE symbol='NVDA' AND ts >= {start_ts} AND ts <= {end_ts} ORDER BY ts ASC", conn_1m)
    
    # Also get 1m price for drift debug
    df_b_1m = pd.read_sql(f"SELECT ts, close as close_1m FROM market_bars_1m WHERE symbol='NVDA' AND ts >= {start_ts} AND ts <= {end_ts} ORDER BY ts ASC", conn_1m)
    
    conn_1s.close()
    conn_1m.close()

    print(f"📊 [Data] Loaded {len(df_b1)} 1s bars and {len(df_o1)} 1s option snapshots.")

    # 3. Initialize Minimal Signal Engine
    engine = SignalEngineV8(symbols=['NVDA'], mode='backtest')
    engine.dynamic_alpha_mean = 0.0 
    engine.dynamic_alpha_std = 1.0
    st = engine.states['NVDA']

    results_1s = []
    # Ensure keys are integers to avoid float precision issues
    opt_map = {int(row['ts']): row['buckets_json'] for _, row in df_o1.iterrows()}
    print(f"✅ [Debug] Built Opt Map with {len(opt_map)} keys. Sample Key: {next(iter(opt_map.keys()))}")

    # 4. Simulation Loop (Tick-by-Tick)
    print("🚀 [Sim] Running feature simulation (Minute boundaries)...")
    for _, row in df_b1.iterrows():
        ts = int(row['ts'])
        price = float(row['close'])
        
        st.update_tick_state(price, price, price)
        
        if ts % 60 == 0:
            metrics_batch = {'alpha_mean': 0.0, 'alpha_std': 1.0, 'vol_z_dict': {}, 'curr_ts': ts}
            st.update_indicators(price, 0.0, ts=ts, use_precalc_feed=True)
            metrics = engine._prep_symbol_metrics(0, 'NVDA', [price], [0.0], [0.1], True, metrics_batch)
            
            buckets_json = opt_map.get(ts)
            buckets_payload = json.loads(buckets_json) if buckets_json else {}
            if buckets_payload and 'buckets' in buckets_payload:
                buckets = buckets_payload['buckets']
                if ts == 1767366000: # 10:00:00 Debug
                   print(f"🔍 [Match] Found buckets for {ts}. Call IV at idx 2, slt 7: {buckets[2][7]}")
                call_iv = float(buckets[2][7]) if len(buckets) > 2 else 0.0
                put_iv = float(buckets[0][7]) if len(buckets) > 0 else 0.0
                
                # Apply Mean IV Logic
                if call_iv > 0.01 and put_iv > 0.01:
                    actual_iv = (call_iv + put_iv) / 2.0
                elif call_iv > 0.01:
                    actual_iv = call_iv
                else:
                    actual_iv = put_iv
                    
                st.last_valid_iv = actual_iv 
                
                results_1s.append({
                    'ts': ts,
                    'close_1s': price,
                    'call_iv_1s': call_iv,
                    'put_iv_1s': put_iv,
                    'mean_iv_1s': actual_iv,
                    'macd_1s': metrics['macd'],
                    'roc_1s': metrics['roc_5m']
                })

    df_1s = pd.DataFrame(results_1s)
    df_compare = pd.merge(df_1s, df_parquet, left_on='ts', right_on='ts', how='inner')
    df_compare = pd.merge(df_compare, df_b_1m, on='ts', how='inner')

    print("\n" + "="*80)
    print(f"📊 [Result] Parity Comparison for NVDA (Dual-DB Aligned)")
    print("="*80)
    
    df_compare['parquet_mean_iv'] = (df_compare['feed_call_iv'] + df_compare['feed_put_iv']) / 2.0
    pd.options.display.max_columns = None
    pd.options.display.width = 160
    # Add NY Time for readability
    NY_TZ = pytz.timezone('America/New_York')
    df_compare['time_ny'] = df_compare['ts'].apply(lambda x: datetime.fromtimestamp(x, NY_TZ).strftime('%H:%M:%S'))

    print(df_compare[['time_ny', 'close_1s', 'close_1m', 'close', 'mean_iv_1s', 'parquet_mean_iv', 'alpha_score']])
    
    iv_diff = (df_compare['mean_iv_1s'] - df_compare['parquet_mean_iv']).abs().mean()
    price_diff = (df_compare['close_1s'] - df_compare['close']).abs().mean()
    price_1m_diff = (df_compare['close_1m'] - df_compare['close']).abs().mean()
    
    print("\n📈 [Audit Summary]")
    print(f"🔹 Price (1s -> Baseline) MAE: {price_diff:.4f}")
    print(f"🔹 Price (1m -> Baseline) MAE: {price_1m_diff:.4f}")
    print(f"🔹 Mean IV MAE: {iv_diff:.6f}")
    
    if iv_diff < 0.0001:
        print("✅ [SUCCESS] IV Parity bit-perfect!")
    else:
        print("❌ [FAILURE] IV Drift detected.")

if __name__ == "__main__":
    audit_nvda_features()

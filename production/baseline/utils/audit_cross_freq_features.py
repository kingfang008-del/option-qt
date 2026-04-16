import sqlite3
import pandas as pd
import numpy as np
import json
import sys
import os
from pathlib import Path
from datetime import datetime
import pytz
import unittest.mock
from unittest.mock import MagicMock

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / "production" / "baseline"))

# Mock heavy dependencies before imports
sys.modules['scipy'] = MagicMock()
sys.modules['scipy.stats'] = MagicMock()
mock_torch = MagicMock()
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

def audit_cross_frequency():
    # We use the SAME physical DB file for both sources to isolate logic drift
    db_path = PROJECT_ROOT / "production" / "preprocess" / "backtest" / "history_sqlite_1s" / "market_20260102.db"
    
    if not db_path.exists():
        print(f"❌ DB not found: {db_path}")
        return

    print("="*80)
    print(f"🚀 [Cross-Freq Audit] Source: {db_path.name}")
    print("="*80)

    # 1. Load Data
    start_ts = 1767364200 # 09:30:00
    end_ts = 1767364800   # 09:40:00
    
    conn = sqlite3.connect(db_path)
    df_1s = pd.read_sql(f"SELECT * FROM market_bars_1s WHERE symbol='NVDA' AND ts >= {start_ts} AND ts <= {end_ts} ORDER BY ts ASC", conn)
    df_1m = pd.read_sql(f"SELECT * FROM market_bars_1m WHERE symbol='NVDA' AND ts >= {start_ts} AND ts <= {end_ts} ORDER BY ts ASC", conn)
    df_opt = pd.read_sql(f"SELECT * FROM option_snapshots_1m WHERE symbol='NVDA' AND ts >= {start_ts} AND ts <= {end_ts} ORDER BY ts ASC", conn)
    conn.close()

    print(f"📊 [Data] Loaded {len(df_1s)} ticks and {len(df_1m)} bars for NVDA.")

    # 2. Setup Engines
    engine_1s = SignalEngineV8(symbols=['NVDA'], mode='backtest')
    engine_s1 = SignalEngineV8(symbols=['NVDA'], mode='backtest')
    
    st_1s = engine_1s.states['NVDA']
    st_s1 = engine_s1.states['NVDA']
    
    opt_map = {int(row['ts']): row['buckets_json'] for _, row in df_opt.iterrows()}
    bar_1m_map = {int(row['ts']): row for _, row in df_1m.iterrows()}

    audit_results = []

    # 3. Simulation Loop
    current_ticks = []
    
    print("⏳ Running Dual-Track Simulation...")
    for _, tick in df_1s.iterrows():
        ts = int(tick['ts'])
        price = float(tick['close'])
        
        # Track 1s Engine State (Tick-by-tick accumulation)
        st_1s.update_tick_state(price, price, price)
        
        # At Minute Boundary
        if ts % 60 == 0 and ts in bar_1m_map:
            # --- Pathway B (s1 Driver Logic: Correctly Aligned Minute Comparison) ---
            # When we are at 10:01:00, we have just finished the 10:00 minute (10:00:00-10:00:59).
            # We compare with the 1m bar labeled 10:00:00.
            target_1m_ts = ts - 60 
            
            # 🚀 [Crucial Parity Fix] Only start comparing when both have valid precursors
            if target_1m_ts in bar_1m_map and st_1s.last_tick_price is not None:
                # Mock option data for audit (since we focus on stock-based MACD/ROC logic here)
                mock_opt = {'has_feed': False} 

                # --- Pathway A (1s Driver Logic: High-Freq Accumulation) ---
                metrics_1s = engine_1s._prep_symbol_metrics(0, 'NVDA', [price], [0.0], mock_opt, 
                                                            {'alpha_mean': 0.0, 'alpha_std': 1.0, 'vol_z_dict': {}, 'curr_ts': ts})
                
                # --- Pathway B (s1 Driver Logic: Minute-Bar Jump) ---
                bar_1m = bar_1m_map[target_1m_ts]
                close_1m = float(bar_1m['close'])
                metrics_s1 = engine_s1._prep_symbol_metrics(0, 'NVDA', [close_1m], [0.0], mock_opt, 
                                                            {'alpha_mean': 0.0, 'alpha_std': 1.0, 'vol_z_dict': {}, 'curr_ts': target_1m_ts})
                
                # Record results
                audit_results.append({
                    'ts': ts,
                    'time': datetime.fromtimestamp(ts, pytz.timezone('America/New_York')).strftime('%H:%M:%S'),
                    'p_1s_actual': price,
                    'p_1s_used': st_1s.last_tick_price,
                    'p_s1': close_1m,
                    'macd_1s': metrics_1s['macd'],
                    'macd_s1': metrics_s1['macd'],
                    'roc_1s': metrics_1s['roc_5m'],
                    'roc_s1': metrics_s1['roc_5m'],
                    'volz_1s': metrics_1s['vol_z'],
                    'volz_s1': metrics_s1['vol_z']
                })

        # MUST update last_tick_price AT THE END of the tick processing to match SignalEngineV8 flow
        st_1s.last_tick_price = price

    # 4. Display Results
    df_audit = pd.DataFrame(audit_results)
    
    # Calculate Drifts
    df_audit['macd_diff'] = (df_audit['macd_1s'] - df_audit['macd_s1']).abs()
    df_audit['roc_diff'] = (df_audit['roc_1s'] - df_audit['roc_s1']).abs()
    
    print("\n" + "="*120)
    print(f"{'Time':<10} | {'P_1s_Act':<8} | {'P_1s_Used':<8} | {'P_s1':<8} | {'MACD_1s':<10} | {'MACD_s1':<10} | {'ROC_1s':<10} | {'ROC_s1':<10}")
    print("-"*120)
    for _, r in df_audit.iterrows():
        print(f"{r['time']:<10} | {r['p_1s_actual']:<8.4f} | {r['p_1s_used']:<8.4f} | {r['p_s1']:<8.4f} | {r['macd_1s']:<10.6f} | {r['macd_s1']:<10.6f} | {r['roc_1s']:<10.6f} | {r['roc_s1']:<10.6f}")

    print("\n📈 [Parity Summary]")
    print(f"🔹 Price MAE (Used p_1s vs Actual p_s1): { (df_audit['p_1s_used'] - df_audit['p_s1']).abs().mean():.6f}")
    print(f"🔹 MACD MAE:  { df_audit['macd_diff'].mean():.8f}")
    print(f"🔹 ROC MAE:   { df_audit['roc_diff'].mean():.8f}")
    
    if df_audit['macd_diff'].mean() < 1e-6:
        print("\n✅ [SUCCESS] Bit-perfect parity between 1s and s1 logic streams!")
    else:
        print("\n❌ [FAILURE] Drift detected between Tick accumulation and Bar sampling.")

if __name__ == "__main__":
    audit_cross_frequency()

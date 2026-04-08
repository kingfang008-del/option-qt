import asyncio
import sqlite3
import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path

# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / "production"))
sys.path.append(str(PROJECT_ROOT / "production" / "baseline"))
sys.path.append(str(PROJECT_ROOT / "production" / "baseline" / "DAO"))

# Mock Heavy Drivers to avoid Torch/Redis on Mac if possible, but we need FCS
import unittest.mock
sys.modules['redis'] = unittest.mock.MagicMock()

# Mock torch if not in python3.10 environment
try:
    import torch
except ImportError:
    mock_torch = unittest.mock.MagicMock()
    sys.modules['torch'] = mock_torch
    sys.modules['torch.nn'] = unittest.mock.MagicMock()
    sys.modules['torch.nn.functional'] = unittest.mock.MagicMock()

from feature_compute_service_v8 import FeatureComputeService

async def main():
    db_path = PROJECT_ROOT / "production" / "preprocess" / "backtest" / "history_sqlite_1s" / "market_20260102.db"
    if not db_path.exists():
        print(f"❌ DB NOT FOUND: {db_path}")
        return

    print("="*80)
    print("🚀 [FCS OHLCV Aggregation Audit] Starts")
    print("="*80)

    # 1. Load Data (NVDA test case)
    # Start: 09:30:00 (1767364200) to 09:45:00 (1767365100)
    conn = sqlite3.connect(db_path)
    df_1s = pd.read_sql("SELECT * FROM market_bars_1s WHERE symbol='NVDA' AND ts >= 1767364200 AND ts <= 1767365100 ORDER BY ts ASC", conn)
    df_1m = pd.read_sql("SELECT * FROM market_bars_1m WHERE symbol='NVDA' AND ts >= 1767364200 AND ts <= 1767365100 ORDER BY ts ASC", conn)
    conn.close()

    if df_1s.empty or df_1m.empty:
        print("❌ No data loaded. Check DB.")
        return

    print(f"📊 Loaded {len(df_1s)} ticks (1s) and {len(df_1m)} bars (1m) baseline.")

    # 2. Init FCS
    mock_redis_cfg = {'input_stream': 'test', 'output_stream': 'test', 'group': 'test', 'consumer': 'test'}
    mock_config_paths = {'fast': str(PROJECT_ROOT / "production" / "CONFIG" / "fast_feature.json"),
                         'slow': str(PROJECT_ROOT / "production" / "CONFIG" / "slow_feature.json")}
    
    # Force mock engine since we only care about aggregation parts
    with unittest.mock.patch('feature_compute_service_v8.RealTimeFeatureEngine'):
        fcs = FeatureComputeService(mock_redis_cfg, ['NVDA'], mock_config_paths)
    
    # 3. Simulation Loop
    print("⏳ Simulating 1s Ticks -> FCS Aggregation...")
    
    for _, tick in df_1s.iterrows():
        ts_val = int(tick['ts'])
        stock_data = {
            'open': tick['open'], 'high': tick['high'], 
            'low': tick['low'], 'close': tick['close'], 'volume': tick['volume']
        }
        
        payload = {
            'ts': ts_val,
            'symbol': 'NVDA',
            'stock': stock_data,
        }
        
        # Call the logic being audited
        await fcs.process_market_data([payload])
        
    print("✅ Simulation complete. Comparing history_1min data frames...")
    
    # 4. Parity Check
    fcs_hist = fcs.history_1min['NVDA']
    # Convert index (DatetimeIndex) to Unix TS for easy comparison
    fcs_hist['ts_unix'] = [int(x.timestamp()) for x in fcs_hist.index]
    
    comparison = []
    for _, b_row in df_1m.iterrows():
        b_ts = int(b_row['ts'])
        f_match = fcs_hist[fcs_hist['ts_unix'] == b_ts]
        
        if f_match.empty: continue
        f_row = f_match.iloc[0]
        
        comparison.append({
            'ts': b_ts,
            'b_close': b_row['close'], 'f_close': f_row['close'],
            'b_vol': b_row['volume'], 'f_vol': f_row['volume'],
            'b_high': b_row['high'], 'f_high': f_row['high']
        })
        
    df_comp = pd.DataFrame(comparison)
    if df_comp.empty:
        print("❌ FAILED: No overlapping timestamps found between generated and baseline.")
        return

    print("\n" + "-"*120)
    print(f"{'Time':<12} | {'Close(Base)':<12} | {'Close(FCS)':<12} | {'Vol(Base)':<12} | {'Vol(FCS)':<12} | {'Result'}")
    print("-" * 120)
    
    success_count = 0
    for _, r in df_comp.head(30).iterrows():
        c_match = abs(r['b_close'] - r['f_close']) < 1e-4
        v_match = abs(r['b_vol'] - r['f_vol']) < 1e-4
        match_str = "✅ PASS" if (c_match and v_match) else "❌ FAIL"
        if c_match and v_match: success_count += 1
        
        time_str = pd.Timestamp(r['ts'], unit='s', tz='UTC').tz_convert('America/New_York').strftime('%H:%M:%S')
        print(f"{time_str:<12} | {r['b_close']:<12.4f} | {r['f_close']:<12.4f} | {r['b_vol']:<12.2f} | {r['f_vol']:<12.2f} | {match_str}")

    close_mae = (df_comp['b_close'] - df_comp['f_close']).abs().mean()
    vol_mae = (df_comp['b_vol'] - df_comp['f_vol']).abs().mean()
    
    print("\n" + "="*80)
    print("📈 FINAL PARITY SUMMARY")
    print(f"🔹 Close MAE:  {close_mae:.8f}")
    print(f"🔹 Volume MAE: {vol_mae:.2f}")
    print(f"🔹 Global Match Rate: {success_count/len(df_comp.head(30))*100:.1f}%")
    
    if close_mae < 1e-6 and vol_mae < 1e-6:
        print("\n✅ [SUCCESS] Bit-perfect parity achieved between 1s aggregation and 1m baseline database!")
    else:
        print("\n❌ [FAILURE] Aggregation discrepancy detected. Check cleanup logic.")
    print("="*80)

if __name__ == "__main__":
    asyncio.run(main())

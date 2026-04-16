import asyncio
import sqlite3
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / "production"))
sys.path.append(str(PROJECT_ROOT / "production" / "baseline"))
sys.path.append(str(PROJECT_ROOT / "production" / "baseline" / "DAO"))

# SUPER-QUICK AGGREGATION AUDIT (NO TORCH, NO TA)
# We test the raw OHLCV buffer logic in FeatureComputeService

from feature_compute_service_v8 import FeatureComputeService

# Mock EVERYTHING heavy
import unittest.mock
sys.modules['torch'] = unittest.mock.MagicMock()
sys.modules['redis'] = unittest.mock.MagicMock()
sys.modules['ta'] = unittest.mock.MagicMock()

# Mock the engine module and its class
mock_engine_module = unittest.mock.MagicMock()
sys.modules['DAO.realtime_feature_engine'] = mock_engine_module
sys.modules['realtime_feature_engine'] = mock_engine_module
mock_engine_module.RealTimeFeatureEngine = unittest.mock.MagicMock()

async def main():
    db_path = PROJECT_ROOT / "production" / "preprocess" / "backtest" / "history_sqlite_1s" / "market_20260102.db"
    
    print("🚀 [LITE Audit] Starting raw OHLCV aggregation check...")
    
    conn = sqlite3.connect(db_path)
    # Just 10 minutes of data for instant result
    df_1s = pd.read_sql("SELECT * FROM market_bars_1s WHERE symbol='NVDA' AND ts >= 1767364200 AND ts <= 1767364800 ORDER BY ts ASC", conn)
    df_1m = pd.read_sql("SELECT * FROM market_bars_1m WHERE symbol='NVDA' AND ts >= 1767364200 AND ts <= 1767364800 ORDER BY ts ASC", conn)
    conn.close()

    mock_config_paths = {
        'fast': str(PROJECT_ROOT / "production" / "CONFIG" / "fast_feature.json"),
        'slow': str(PROJECT_ROOT / "production" / "CONFIG" / "slow_feature.json")
    }
    fcs = FeatureComputeService({'input_stream': 'test'}, ['NVDA'], mock_config_paths)
    
    for _, tick in df_1s.iterrows():
        payload = {
            'ts': int(tick['ts']),
            'symbol': 'NVDA',
            'stock': {'open': tick['open'], 'high': tick['high'], 'low': tick['low'], 'close': tick['close'], 'volume': tick['volume']}
        }
        await fcs.process_market_data([payload])
    
    fcs_hist = fcs.history_1min['NVDA']
    fcs_hist['ts_unix'] = [int(x.timestamp()) for x in fcs_hist.index]
    
    print(f"\n{'Time':<10} | {'BaseClose':<10} | {'FCS_Close':<10} | {'Status'}")
    print("-" * 50)
    for _, b in df_1m.iterrows():
        f_match = fcs_hist[fcs_hist['ts_unix'] == b['ts']]
        if f_match.empty: continue
        f = f_match.iloc[0]
        match = abs(b['close'] - f['close']) < 1e-4
        print(f"{int(b['ts']):<10} | {b['close']:<10.2f} | {f['close']:<10.2f} | {'✅' if match else '❌'}")

if __name__ == "__main__":
    asyncio.run(main())

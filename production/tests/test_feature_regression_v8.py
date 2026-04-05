import numpy as np
import pandas as pd
import sys
import os
from pathlib import Path
import asyncio
from datetime import datetime
import unittest.mock as mock

# Setup Path
curr_file = Path(__file__).resolve()
proj_root = curr_file.parent.parent # production
sys.path.append(str(proj_root))
sys.path.append(str(proj_root / "baseline"))
sys.path.append(str(proj_root / "DB"))

print(f"DEBUG: sys.path includes: {sys.path[-3:]}")
print(f"DEBUG: proj_root: {proj_root}")

# Mock dependencies
import unittest.mock as mock
mock_torch = mock.MagicMock()
sys.modules['torch'] = mock_torch
sys.modules['psycopg2'] = mock.MagicMock()
sys.modules['redis'] = mock.MagicMock()
sys.modules['msgpack'] = mock.MagicMock()
sys.modules['zstandard'] = mock.MagicMock()
sys.modules['realtime_feature_engine'] = mock.MagicMock()
from unittest.mock import MagicMock
mock_engine_cls = MagicMock()
sys.modules['realtime_feature_engine'].RealTimeFeatureEngine = mock_engine_cls

from DB.feature_compute_service_v8 import FeatureComputeService

async def test_reproduce_nonetype():
    print("🧪 Testing for NoneType regression in FeatureComputeService...")
    symbols = ['NVDA']
    
    # Mocking _load_json_info to avoid file access
    with mock.patch.object(FeatureComputeService, '_load_json_info') as mock_load:
        mock_load.side_effect = [
            [{'name': 'price_z', 'resolution': '1min'}], # fast
            [{'name': 'roc_5m', 'resolution': '5min'}]  # slow
        ]
        
        config_paths = {'fast': 'ignored', 'slow': 'ignored'}
        feat_svc = FeatureComputeService(
            symbols=symbols, 
            redis_cfg={}, 
            config_paths=config_paths
        )
    
    # Force initialize state
    feat_svc.all_feat_names = ['price_z', 'roc_5m']
    feat_svc.feat_name_to_idx = {'price_z': 0, 'roc_5m': 1}
    feat_svc.fast_indices = [0]
    feat_svc.slow_indices = [1]
    feat_svc.fast_feat_names = ['price_z']
    feat_svc.slow_feat_names = ['roc_5m']
    feat_svc.symbols = symbols
    feat_svc.cached_batch_raw = None # Ensure it starts at None
    for s in symbols:
        feat_svc.history_1min[s] = pd.DataFrame([{
            'open': 190.0, 'high': 191.0, 'low': 189.0, 'close': 190.0, 'volume': 1000.0
        }], index=[pd.Timestamp('2026-01-02 09:59:00')])
        feat_svc.latest_prices[s] = 190.0
        feat_svc.warmup_needed[s] = False

    # Simulate start at 09:59:55 (Not a boundary)
    ts_start = 1767367195.0 # 2026-01-02 09:59:55
    print(f"🚀 Calling run_compute_cycle at {datetime.fromtimestamp(ts_start)}")
    
    try:
        # Mock engine to return something
        feat_svc.engine = mock.MagicMock()
        feat_svc.engine.compute_all_inputs.return_value = {
            'NVDA': {
                'fast_1m': np.zeros((1, 10, 1)),
                'slow_1m': np.zeros((1, 10, 1)),
                'is_valid': True
            }
        }
        feat_svc.feat_name_to_idx = {name: i for i, name in enumerate(feat_svc.all_feat_names)}
        feat_svc.fast_indices = []
        feat_svc.slow_indices = []
        
        # This is where it should crash if logic is wrong
        payload = await feat_svc.run_compute_cycle(ts_from_payload=ts_start, return_payload=True)
        print("✅ Call 1 (09:59:55) SUCCESS")
        
        # Call again at 10:00:00 (Boundary)
        ts_boundary = 1767367200.0
        print(f"🚀 Calling run_compute_cycle at {datetime.fromtimestamp(ts_boundary)}")
        payload2 = await feat_svc.run_compute_cycle(ts_from_payload=ts_boundary, return_payload=True)
        print("✅ Call 2 (10:00:00) SUCCESS")

    except Exception as e:
        print(f"❌ REPRODUCED ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_reproduce_nonetype())

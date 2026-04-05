import numpy as np
import pandas as pd
import json
import os
import sys
from pathlib import Path
import asyncio

# Setup Path
base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))
sys.path.append(str(base_dir / "DB"))
sys.path.append(str(base_dir / "baseline"))

# Mock torch before imports
import unittest.mock as mock
mock_torch = mock.MagicMock()
sys.modules['torch'] = mock_torch
sys.modules['psycopg2'] = mock.MagicMock()

from DB.feature_compute_service_v8 import FeatureComputeService
from baseline.signal_engine_v8 import SignalEngineV8

async def test_iv_flow():
    symbols = ['NVDA']
    print("🧪 Testing IV Flow from Feature Engine to Signal Engine...")
    
    # 1. Initialize Feature Service
    feat_svc = FeatureComputeService(symbols=symbols)
    
    # Mock some history to avoid warmup issues
    for s in symbols:
        feat_svc.history_1min[s] = pd.DataFrame([{
            'open': 190.0, 'high': 191.0, 'low': 189.0, 'close': 190.0, 'volume': 1000.0
        }], index=[pd.Timestamp('2026-01-02 10:11:00')])
        feat_svc.latest_prices[s] = 190.0
        feat_svc.warmup_needed[s] = False

    # 2. Mock Market Data with IV = 0.45
    # Column 7 is IV according to s1_seed_option_snapshots_sqlite_1s.py
    # Column 6 is Volume
    buckets = [
        [1.5, 190.0, 1.0, 5.0, 1.4, 1.6, 0.0, 0.45, 1.4, 1.6, 0.0, 0.0], # Row 0: Put?
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],    # Row 1
        [1.8, 190.0, 1.0, 5.0, 1.7, 1.9, 0.0, 0.45, 1.7, 1.9, 0.0, 0.0], # Row 2: Call
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ]
    
    batch = [{
        'ts': 1767366720.0,
        'symbol': 'NVDA',
        'stock': {'close': 190.0},
        'option_buckets': buckets
    }]
    
    # 3. Process Market Data
    print("📡 Processing Market Data (Volume is 0)...")
    await feat_svc.process_market_data(batch)
    
    # Verify internal snapshot
    snap = feat_svc.option_snapshot['NVDA']
    print(f"📊 Internal Snapshot IV (Row 2, Col 7): {snap[2, 7]}")
    if abs(snap[2, 7] - 0.45) < 1e-6:
        print("✅ Feature Engine Internal Buffer Update: SUCCESS")
    else:
        print(f"❌ Feature Engine Internal Buffer Update: FAILED (Found {snap[2, 7]})")

    # 4. Run Compute Cycle
    print("⚙️ Running Compute Cycle...")
    payload = await feat_svc.run_compute_cycle(ts_from_payload=1767366720.0, return_payload=True)
    
    # Verify Payload
    p_iv = payload['cheat_call_iv'][0]
    print(f"📦 Payload cheat_call_iv: {p_iv}")
    if abs(p_iv - 0.45) < 1e-6:
        print("✅ Feature Engine Payload Extraction: SUCCESS")
    else:
        print(f"❌ Feature Engine Payload Extraction: FAILED (Found {p_iv})")

    # 5. Signal Engine Verification
    print("📡 Signal Engine Processing...")
    signal_svc = SignalEngineV8(symbols=symbols)
    signal_svc.only_log_alpha = True
    
    await signal_svc.process_batch(payload)
    
    # Verify Alpha Buffer
    if signal_svc.alpha_buffer:
        res_iv = signal_svc.alpha_buffer[0]['iv']
        print(f"🚀 Final Logged IV: {res_iv}")
        if abs(res_iv - 0.45) < 1e-6:
            print("✅ Signal Engine IV Persistence: SUCCESS")
        else:
            print(f"❌ Signal Engine IV Persistence: FAILED (Expected 0.45, Found {res_iv})")
    else:
        print("❌ Signal Engine: No Alpha Generated!")

if __name__ == "__main__":
    asyncio.run(test_iv_flow())

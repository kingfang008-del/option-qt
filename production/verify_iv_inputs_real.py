import sqlite3
import json
import numpy as np
import pandas as pd
import sys
import os
from datetime import datetime
import pytz

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from production.baseline.DAO.realtime_feature_engine import RealTimeFeatureEngine
from production.utils.greeks_math import calculate_bucket_greeks

def verify_iv_calculation():
    db_path = "/Users/fangshuai/Documents/GitHub/option-qt/production/preprocess/backtest/history_sqlite_1s/market_20260102.db"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    symbol = 'NVDA'
    print(f"\n🔍 [START] Verifying IV Calculation for {symbol}")

    # 1. Get a sample snapshot
    cursor.execute("SELECT * FROM option_snapshots_1s WHERE symbol=? LIMIT 1", (symbol,))
    row = cursor.fetchone()
    if not row:
        print("❌ Error: No option snapshots found for NVDA")
        return

    ts = row['ts']
    datetime_ny = row['datetime_ny']
    buckets = np.array(json.loads(row['buckets_json']), dtype=np.float32)
    
    print(f"✅ Found Snapshot at {datetime_ny} (TS: {ts})")
    print(f"📊 Buckets Shape: {buckets.shape}")

    # 2. Get corresponding stock price
    # We look for the last bar before or at this timestamp
    cursor.execute("SELECT close FROM market_bars_1s WHERE symbol=? AND ts <= ? ORDER BY ts DESC LIMIT 1", (symbol, ts))
    bar_row = cursor.fetchone()
    if not bar_row:
        print(f"❌ Error: No stock price found for {symbol} at/before {ts}")
        return
    
    stock_price = float(bar_row['close'])
    print(f"📈 Stock Price: {stock_price}")

    # 3. Get contracts for this snapshot (Mocking what Service would provide)
    # Since sqlite doesn't store the contracts list in option_snapshots_1s (it's implicit in buckets),
    # we need to simulate the contracts list.
    # In production, contracts are passed from the Service.
    # Here we mock them based on NVDA example format.
    # Standard format: O:NVDA260116C00150000
    # The RealTimeFeatureEngine.supplement_greeks needs these to fetch expiration.
    
    # Let's try to find the contracts from another table if available, 
    # or just use a dummy one with correct timestamp for testing.
    # Actually, we can check how s1_seed_option_snapshots_sqlite_1s.py seeds them.
    
    # For now, let's look at the buckets[i, 5] (Strike) to build a mock contract
    contracts = []
    for i in range(len(buckets)):
        strike = buckets[i, 5]
        # Mocking a Jan 2026 contract for NVDA
        suffix = "C" if i % 2 == 0 else "P"
        mock_tkr = f"O:NVDA260102{suffix}{int(strike*1000):08d}"
        contracts.append(mock_tkr)

    print(f"📜 Mocked Contracts Sample: {contracts[:2]}")

    # 4. Initialize Engine
    engine = RealTimeFeatureEngine(stats_path="production/CONFIG/slow_feature.json")
    
    # 5. Execute supplement_greeks
    print("\n🚀 Calling supplement_greeks...")
    updated_buckets = engine.supplement_greeks(symbol, buckets.copy(), contracts, stock_price, float(ts))

    # 6. Analyze Results
    print("\n📊 [RESULTS] Bucket IVs:")
    for i in range(len(updated_buckets)):
        iv = updated_buckets[i, 7]
        strike = updated_buckets[i, 5]
        price = updated_buckets[i, 0]
        print(f"Row {i} | Strike: {strike:7.2f} | MktPrice: {price:6.2f} | Calculated IV: {iv:.6f}")

    iv_max = updated_buckets[:, 7].max()
    if iv_max < 0.0001:
        print("\n❌ FAILED: All IVs are zero!")
    elif abs(iv_max - 0.5) < 1e-6:
        print("\n⚠️ WARNING: IV matched default 0.5 (leaked?)")
    else:
        print(f"\n✅ SUCCESS: Calculated IV Max: {iv_max:.6f}")

    conn.close()

if __name__ == "__main__":
    verify_iv_calculation()

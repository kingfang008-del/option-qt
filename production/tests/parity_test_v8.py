import os
import sys
import asyncio
import numpy as np
from pathlib import Path

# Mock heavy dependencies before importing SignalEngineV8
import unittest.mock
sys.modules['torch'] = unittest.mock.MagicMock()
sys.modules['trading_tft_stock_embed'] = unittest.mock.MagicMock()
sys.modules['lmdb'] = unittest.mock.MagicMock()
sys.modules['zstandard'] = unittest.mock.MagicMock()
sys.modules['msgpack_numpy'] = unittest.mock.MagicMock()
sys.modules['psycopg2'] = unittest.mock.MagicMock()
sys.modules['redis'] = unittest.mock.MagicMock()

# Add project root, baseline, and model to path
root = Path(__file__).parent.parent
sys.path.append(str(root))
sys.path.append(str(root / "baseline"))
sys.path.append(str(root / "model"))

from signal_engine_v8 import SignalEngineV8

def mock_batch(ts, symbols, prices, alpha=None):
    return {
        'ts': ts,
        'symbols': symbols,
        'stock_price': prices,
        'fast_vol': [0.0] * len(symbols),
        'spy_roc_5min': [0.0] * len(symbols),
        'qqq_roc_5min': [0.0] * len(symbols),
        'alpha_score': [alpha] * len(symbols) if alpha is not None else [0.0] * len(symbols)
    }

async def run_parity_test():
    symbols = ['NVDA']
    config_paths = {'fast': 'dummy', 'slow': 'dummy'}
    
    # Initialize two engines
    engine_1s = SignalEngineV8(symbols=symbols, mode='backtest', config_paths=config_paths)
    engine_1m = SignalEngineV8(symbols=symbols, mode='backtest', config_paths=config_paths)
    
    # Set only_log_alpha to True to skip Redis sync loop
    engine_1s.only_log_alpha = True
    engine_1m.only_log_alpha = True
    
    # Disable actual model inference to use provided alpha
    async def dummy_inference(*args):
        return [args[1][0] if isinstance(args[1], list) else 0.0]
    
    engine_1s._run_model_inference = dummy_inference
    engine_1m._run_model_inference = dummy_inference
    
    # Override _run_model_inference to return the alpha we passed in the batch
    async def mock_inference(batch, symbols, prices, ny_now):
        return np.array(batch.get('alpha_score', [0.0] * len(symbols)))
    
    engine_1s._run_model_inference = mock_inference
    engine_1m._run_model_inference = mock_inference

    # Patch process_batch to print gating info
    org_process = engine_1s.process_batch
    async def debug_process(batch):
        # We need to compute the same logic as inside the engine
        curr_ts = batch['ts']
        last_t = getattr(engine_1s, 'last_process_ts_for_gating', 0.0)
        is_new_min_crossing = (int(curr_ts / 60) > int(last_t / 60)) or (last_t == 0)
        is_high_freq = os.environ.get('FORCE_HIGH_FREQ') == '1'
        should_update_full = is_new_min_crossing if is_high_freq else True
        if is_new_min_crossing:
            print(f"DEBUG: T={curr_ts} | NewMin={is_new_min_crossing} | FullUpdate={should_update_full}")
        return await org_process(batch)
    
    engine_1s.process_batch = debug_process

    base_ts = 1704205800.0 # 09:30:00 ET (2024-01-02)
    prices = [100.0 + i * 0.1 for i in range(61)] # 0 to 60 seconds
    alpha = 1.5
    
    print("\n🚀 Starting Parity Test: 1s vs 1m...")

    # --- Case 1: 1s Processing ---
    # Force high frequency mode for 1s engine
    os.environ['FORCE_HIGH_FREQ'] = '1'
    
    # Feed 60 ticks (0 to 59s)
    for i in range(60):
        ts = base_ts + i
        batch = mock_batch(ts, symbols, [prices[i]], alpha=alpha)
        await engine_1s.process_batch(batch)
    
    # Check 1s state at 59s
    st_1s = engine_1s.states['NVDA']
    print(f"1s Engine @ 59s: Prices Len = {len(st_1s.prices)}")
    if len(st_1s.prices) > 0:
        print(f"Last MACD = {st_1s.cached_macd_hist}")

    # Feed the 60th tick (10:31:00) to trigger minute crossing
    ts_60 = base_ts + 60
    batch_60 = mock_batch(ts_60, symbols, [prices[60]], alpha=alpha)
    await engine_1s.process_batch(batch_60)
    
    print(f"1s Engine @ 60s: Prices Len = {len(st_1s.prices)}")
    if len(st_1s.prices) > 0:
        print(f"Last Price = {st_1s.prices[-1]}, ROC = {st_1s.cached_min_roc}")
    else:
        print("❌ ERROR: 1s Engine failed to update minute indicators!")

    # --- Case 2: 1m Processing ---
    os.environ['FORCE_HIGH_FREQ'] = '0' # Minute mode
    
    # Feed Tick 0
    batch_0 = mock_batch(base_ts, symbols, [prices[0]], alpha=alpha)
    await engine_1m.process_batch(batch_0)
    
    # Feed Tick 60
    batch_60_m = mock_batch(base_ts + 60, symbols, [prices[60]], alpha=alpha)
    await engine_1m.process_batch(batch_60_m)
    
    st_1m = engine_1m.states['NVDA']
    print(f"1m Engine @ 60s: Prices Len = {len(st_1m.prices)}")
    if len(st_1m.prices) > 0:
        print(f"Last Price = {st_1m.prices[-1]}, ROC = {st_1m.cached_min_roc}")
    else:
        print("❌ ERROR: 1m Engine failed to update minute indicators!")

    # --- Comparison ---
    if len(st_1s.prices) == 0 or len(st_1m.prices) == 0:
        print("\n❌ FAIL: One or both engines failed to process data.")
        sys.exit(1)
        
    print("\n🔍 Final Comparison:")
    diff_price = abs(st_1s.prices[-1] - st_1m.prices[-1])
    diff_roc = abs(st_1s.cached_min_roc - st_1m.cached_min_roc)
    diff_macd = abs(st_1s.cached_macd_hist - st_1m.cached_macd_hist)
    
    print(f"Price Diff: {diff_price}")
    print(f"ROC Diff: {diff_roc}")
    print(f"MACD Diff: {diff_macd}")

    if diff_price < 1e-7 and diff_roc < 1e-7 and diff_macd < 1e-7:
        print("\n✅ PASS: 1s and 1m engines are identical at the minute boundary!")
    else:
        print("\n❌ FAIL: Discrepancy detected between 1s and 1m engines.")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(run_parity_test())

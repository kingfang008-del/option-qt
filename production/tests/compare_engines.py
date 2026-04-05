
import sys
import os
import torch
import numpy as np
import pandas as pd
import importlib.util
from pathlib import Path

# Add project root
sys.path.append(str(Path(__file__).parent.parent))

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def compare_engines():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if str(device) == 'cpu' and torch.backends.mps.is_available():
        device = torch.device('mps')
    
    print(f"Using Device: {device}")
    
    # Load Engines
    base_path = Path(__file__).parent.parent / "script"
    mod_raw = load_module("engine_raw", base_path / "realtime_feature_engine_raw.py")
    mod_vec = load_module("engine_vec", base_path / "realtime_feature_engine.py")
    
    engine_raw = mod_raw.RealTimeFeatureEngine(stats_path="dummy.json", device=device)
    engine_vec = mod_vec.RealTimeFeatureEngine(stats_path="dummy.json", device=device)
    
    # Mock Data
    L = 200 # Sufficient history
    dates = pd.date_range(end=pd.Timestamp.now(), periods=L, freq='1min')
    
    # Create deterministic random data
    np.random.seed(42)
    close = np.random.uniform(100, 200, L)
    # Add trend to verify moving averages
    close = close + np.linspace(0, 10, L)
    high = close + np.random.uniform(0, 5, L)
    low = close - np.random.uniform(0, 5, L)
    open_ = (high + low) / 2.0
    volume = np.random.uniform(1000, 5000, L)
    
    df = pd.DataFrame({
        'close': close, 'high': high, 'low': low, 'open': open_, 'volume': volume
    }, index=dates)
    
    history = {'TEST_SYM': df}
    history_vix = {'VIXY': df.copy()} # Dummy VIXY
    
    # Features to test
    feats = [
        'cci', 'bb_width', 'rsi', 'sma_ratio_30', 'adx_smooth_10', 
        'garman_klass_vol', 'atr', 'price_dist_from_ma_atr',
        'vix_level', 'vixy_detrended_level' # Global
    ]
    
    # Run Raw
    print("\n--- Running Raw Engine ---")
    # Raw engine modifies history_1min? No, returns results dict
    # Raw engine compute_all_inputs signature: 
    # (history_1min, fast, slow, option_snapshots, skip_scaling)
    # But wait, raw engine logic needs VIXY in history_1min for global?
    # Yes, it pulls VIXY from history_1min.
    
    full_history = history.copy()
    full_history.update(history_vix)
    
    # NOTE: Raw engine compute_all_inputs might expect 'VIXY' in keys for global comp, 
    # but iterates keys for symbol comp.
    
    try:
        res_raw = engine_raw.compute_all_inputs(
            full_history, feats, feats, option_snapshots=None, skip_scaling=True
        )
        # Result is Dict[str, DataFrame]
        df_raw = res_raw['TEST_SYM']
    except Exception as e:
        print(f"Raw Engine Failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # Run Vectorized
    print("\n--- Running Vectorized Engine ---")
    # Vectorized compute_all_inputs refactored to use batch engine
    # But it returns Dict[str, Dict['fast_1m': Tensor...]]
    
    try:
        res_vec = engine_vec.compute_all_inputs(
            full_history, feats, feats, option_snapshots=None, skip_scaling=True
        )
        # Extract 'slow_1m' which has length 30.
        # We need the LAST value (time=-1) to match Raw engine's last row (presumably).
        # Wait, Raw engine returns DataFrame with same length as input? 
        # Checking implementation... Raw engine compute_all_inputs returns pd.DataFrame(res).
        # `res` keys are features, values are Tensors (expanded to seq_len).
        # So Raw DataFrame has length `seq_len` (200).
        
        # Vectorized `compute_all_inputs` returns 'slow_1m' with seq_len=30 (hardcoded slice).
        # So we compare the LAST element (-1) of Vectorized with LAST element (-1) of Raw.
        
        vec_dict = res_vec['TEST_SYM']
        # slow_1m: [1, N_Feats, 30]
        # We need to map feature list to indices?
        # The Vectorized engine `compute_all_inputs` currently returns 'slow_1m' which is ALREADY FILTERED/STACKED?
        # Let's look at `compute_all_inputs` in `realtime_feature_engine.py`.
        # It calls `compute_batch_features`.
        # `compute_batch_features` uses `process_subset` which stacks chosen features.
        # But `compute_batch_features` returns `{'fast_1m': [B, N, 10], 'slow_1m': [B, N, 30]}`
        # AND check `compute_all_inputs` again.. 
        # It calls `results[s] = {'fast_1m': b_fast[i].unsqueeze(0) ...}`
        
        # KEY ISSUE: The returned tensor doesn't have feature names attached!
        # `process_subset` stacks them in order of `feat_list`.
        # `slow_feats` in my call is `feats` list.
        # So I can map by index.
        
        t_slow = vec_dict['slow_1m'][0] # [N, 30]
        
    except Exception as e:
        print(f"Vectorized Engine Failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # Comparison
    print("\n--- comparison Results (Last Step) ---")
    print(f"{'Feature':<25} | {'Raw':<10} | {'Vectorized':<10} | {'Diff':<10} | {'Status'}")
    print("-" * 75)
    
    # We passed `feats` as `slow_feats` to both.
    
    # Check VIX Level separately from ctx?
    # VIX level is in global ctx, potentially added to result?
    # In vectorized `compute_batch_features`: 
    # `for k, v in global_ctx.items(): if k in all_feats: res[k] = v...`
    # So it IS in the result tensor if requested.
    
    for i, name in enumerate(feats):
        # Raw Value
        if name in df_raw.columns:
            val_raw = df_raw.iloc[-1][name]
            # If tensor item
            if hasattr(val_raw, 'item'): val_raw = val_raw.item()
        else:
            val_raw = float('nan')
            
        # Vec Value
        # t_slow is [N, 30]. Last step is t_slow[:, -1]
        # feature index i corresponds to `feats[i]`?
        # `process_subset` iterates `feat_list`.
        # So yes, order should be preserved.
        val_vec = t_slow[i, -1].item()
        
        diff = abs(val_raw - val_vec)
        status = "✅" if diff < 1e-4 else "❌"
        
        print(f"{name:<25} | {val_raw:<10.4f} | {val_vec:<10.4f} | {diff:<10.4f} | {status}")
        
    print("\nDone.")

if __name__ == "__main__":
    compare_engines()

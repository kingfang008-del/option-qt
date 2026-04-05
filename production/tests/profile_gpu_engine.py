
import sys
import os
import torch
import numpy as np
import pandas as pd
import time
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from realtime_feature_engine import RealTimeFeatureEngine

def profile_batch_compute():
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if str(device) == 'cpu' and torch.backends.mps.is_available():
        device = torch.device('mps')
        
    print(f"Using Device: {device}")
    
    # Pass dummy stats path (handled gracefully by _load_stats)
    engine = RealTimeFeatureEngine(stats_path="dummy_stats.json", device=device)
    
    # Mock Data
    batch_size = 500
    seq_len = 500
    n_feats = 5 # OHLCV
    
    # Create Batch Tensor [B, L, 5]
    # C, H, L, O, V
    data = torch.randn(batch_size, seq_len, 5, device=device).abs() + 100.0
    
    # Create Option Snapshot [B, 6, 8]
    opt_snap = torch.randn(batch_size, 6, 8, device=device).abs()
    
    # Feature Config
    fast_feats = ['close_log_return', 'volume_log', 'fast_vol', 'fast_mom']
    slow_feats = ['sma_ratio_30', 'garman_klass_vol', 'chaikin_vol', 'vol_roc', 
                  'vol_contraction_ratio', 'cci', 'adx_smooth_10', 'price_dist_from_ma_atr',
                  'poc_deviation', 'vwap_diff', 'bb_width', 'rsi', 'k', 'garch_vol', 'macd_ratio']
                  
    sym_list = [f"SYM_{i}" for i in range(batch_size)]
    
    # Global CTX
    global_ctx = {
        'spy_roc_5min': torch.randn(seq_len, device=device),
        'qqq_roc_5min': torch.randn(seq_len, device=device),
        'vix_level': torch.zeros(1, device=device), # scalar ctx
        'vixy_detrended_level': torch.zeros(1, device=device)
    }

    # Warmup
    print("Warmup...")
    _ = engine.compute_batch_features(data, sym_list, fast_feats, slow_feats, option_snapshot=opt_snap, global_ctx=global_ctx)
    
    # Benchmark
    print("Benchmarking...")
    t0 = time.time()
    n_iter = 10
    for _ in range(n_iter):
        _ = engine.compute_batch_features(data, sym_list, fast_feats, slow_feats, option_snapshot=opt_snap, global_ctx=global_ctx)
    
    dt = time.time() - t0
    avg_dt = dt / n_iter
    print(f"Batch Compute (B={batch_size}, L={seq_len}): {avg_dt*1000:.2f} ms")
    print(f"Throughput: {batch_size * n_iter / dt:.1f} symbols/sec")
    
    # Verification
    # Check output shape
    res = engine.compute_batch_features(data, sym_list, fast_feats, slow_feats, option_snapshot=opt_snap, global_ctx=global_ctx)
    
    f1 = res.get('fast_1m')
    s1 = res.get('slow_1m')
    
    print("Output Shapes:")
    if f1 is not None: print(f"Fast: {f1.shape}")
    if s1 is not None: print(f"Slow: {s1.shape}")
    
    if f1 is not None and f1.shape[0] != batch_size:
        print("❌ Fast Batch Size Mismatch")
    elif s1 is not None and s1.shape[0] != batch_size:
        print("❌ Slow Batch Size Mismatch")
    else:
        print("✅ Output Shapes Correct")
        
    
if __name__ == "__main__":
    profile_batch_compute()

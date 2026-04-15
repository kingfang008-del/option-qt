import os
import sys
import copy
import json
import numpy as np
import pandas as pd
from pathlib import Path

# Force rigorous environment settings for the test script
os.environ['RUN_MODE'] = 'LIVEREPLAY'
os.environ['STRICT_LIQUIDITY_MODE'] = '2'
os.environ['DUAL_CONVERGE_TO_SINGLE'] = '1'
os.environ['PURE_ALPHA_REPLAY'] = '1'
os.environ['STRATEGY_CORE_VERSION'] = 'V0'
os.environ['REPLAY_1S_PARITY_MODE'] = '1'

# Add production dirs to PYTHONPATH automatically
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT / "baseline"))
sys.path.append(str(PROJECT_ROOT / "utils"))
sys.path.append(str(PROJECT_ROOT / "preprocess" / "backtest" / "PGSQL"))
sys.path.append(str(PROJECT_ROOT / "preprocess" / "backtest" / "second"))
sys.path.append(str(PROJECT_ROOT / "history_replay"))
sys.path.append(str(PROJECT_ROOT / "baseline" / "DAO"))

try:
    from s4_run_historical_replay_pg_1s import build_option_arrays, safe_col
    # S4 import overwrites os.environ, restore them immediately
    os.environ['RUN_MODE'] = 'LIVEREPLAY'
    os.environ['STRICT_LIQUIDITY_MODE'] = '2'
    os.environ['DUAL_CONVERGE_TO_SINGLE'] = '1'
    os.environ['PURE_ALPHA_REPLAY'] = '1'
    os.environ['STRATEGY_CORE_VERSION'] = 'V0'
    os.environ['REPLAY_1S_PARITY_MODE'] = '1'
    
    import serialization_utils as ser
    from signal_engine_v8 import SignalEngineV8
    from standalone_oms_replay_pitcher import BatchPostgresDriver1s, _load_alpha_map, _compute_index_roc_map, label_ts_for_frame, build_inference_payload
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

def test_single_frame_parity(date_str="20260305"):
    print(f"🧪 [UNIT TEST] Starting Strict Parity Isolation for {date_str}...")

    # 1. Fetch RAW data directly from PG (Identical for both logic paths)
    print("⏳ Fetching raw PG data...")
    driver = BatchPostgresDriver1s([date_str], run_id="TEST", parity_mode=True)
    df_b1, df_o1, df_b5, df_o5 = driver._fetch_day_frames(date_str)
    if df_b1 is None or df_b1.empty:
        print("❌ No PG data found.")
        return

    # Select exactly ONE tick for deep comparison (e.g., 09:45:00)
    target_dt = pd.to_datetime(f"{date_str} 09:45:00", format="%Y%m%d %H:%M:%S")
    target_ts = target_dt.tz_localize('America/New_York').tz_convert('UTC').timestamp()
    
    # 2. Setup S4 Merged DataFrame to emulate S4's precise row
    df_b1['ts'] = df_b1['ts'].astype(float)
    df_o1['ts'] = df_o1['ts'].astype(float)
    df_market = pd.merge_asof(df_b1.sort_values('ts'), df_o1.sort_values('ts'), on='ts', by='symbol', direction='backward', tolerance=2)
    
    alpha_map = _load_alpha_map(date_str)
    # Synthesize df_a from alpha_map
    alpha_records = []
    for lbl_ts, sym_dict in alpha_map.items():
        for sym, row in sym_dict.items():
            row['symbol'] = sym
            row['ts'] = float(lbl_ts) + 2.0 # ALPHA_AVAILABLE_DELAY_SECONDS = 2
            row['alpha_label_ts'] = float(lbl_ts)
            alpha_records.append(row)
    df_a = pd.DataFrame(alpha_records).sort_values('ts')
    
    df = pd.merge_asof(df_market, df_a, on='ts', by='symbol', direction='backward', tolerance=120)
    
    # Extract SPY/QQQ ROC
    df_b1_idx = df_b1.copy()
    df_b1_idx['ts'] = df_b1_idx['ts'].astype(int)
    df_idx = df_b1_idx[df_b1_idx['symbol'].isin(['SPY', 'QQQ'])].pivot(index='ts', columns='symbol', values='close')
    if 'SPY' in df_idx.columns:
        df['spy_roc_5min'] = df['ts'].astype(int).map(df_idx['SPY'].pct_change(periods=300).fillna(0.0))
    if 'QQQ' in df_idx.columns:
        df['qqq_roc_5min'] = df['ts'].astype(int).map(df_idx['QQQ'].pct_change(periods=300).fillna(0.0))

    # Get the exact group for our target_ts
    closest_ts = df[df['ts'] <= target_ts]['ts'].max()
    s4_group = df[df['ts'] == closest_ts].copy()
    
    # 3. Build S4 Packet
    symbols_list = s4_group['symbol'].tolist()
    opt_s4 = build_option_arrays(s4_group)
    
    s4_packet = {
        'symbols': symbols_list,
        'ts': float(closest_ts),
        'stock_price': s4_group['close'].values.astype(np.float32),
        'fast_vol': safe_col(s4_group, 'vol_z', 0.0),
        'precalc_alpha': safe_col(s4_group, 'alpha', 0.0),
        'alpha_score': safe_col(s4_group, 'alpha_score', 0.0),
        'alpha_label_ts': safe_col(s4_group, 'alpha_label_ts', 0.0, dtype=np.float64),
        'alpha_available_ts': np.full(len(s4_group), float(closest_ts), dtype=np.float64),
        'spy_roc_5min': safe_col(s4_group, 'spy_roc_5min', 0.0),
        'qqq_roc_5min': safe_col(s4_group, 'qqq_roc_5min', 0.0),
        'is_new_minute': int(closest_ts) % 60 == 0,
        'symbols_with_data': opt_s4['symbols_with_data'],
        'feed_put_price': opt_s4['put_prices'],
        'feed_call_price': opt_s4['call_prices'],
        'feed_put_k': opt_s4['put_ks'],
        'feed_call_k': opt_s4['call_ks'],
        'feed_put_iv': opt_s4['put_ivs'],
        'feed_call_iv': opt_s4['call_ivs'],
        'feed_put_bid': opt_s4['put_bids'],
        'feed_put_ask': opt_s4['put_asks'],
        'feed_call_bid': opt_s4['call_bids'],
        'feed_call_ask': opt_s4['call_asks'],
        'feed_call_bid_size': np.full(len(s4_group), 100.0, dtype=np.float32),
        'feed_call_ask_size': np.full(len(s4_group), 100.0, dtype=np.float32),
        'feed_put_bid_size': np.full(len(s4_group), 100.0, dtype=np.float32),
        'feed_put_ask_size': np.full(len(s4_group), 100.0, dtype=np.float32),
    }

    # 4. Build Live Pitcher Packet
    pitcher_frames = driver._build_day_stream(date_str, df_b1, df_o1, df_b5, df_o5)
    target_frame = next(f for f in pitcher_frames if abs(f[0] - closest_ts) < 1.0)
    batch_payloads = target_frame[1]
    
    alpha_label_ts = label_ts_for_frame(closest_ts)
    roc_map = _compute_index_roc_map(df_b1)
    pitcher_payload = build_inference_payload(
        closest_ts, batch_payloads, alpha_map.get(alpha_label_ts, {}), roc_map.get(int(closest_ts), {"spy_roc_5min": 0.0, "qqq_roc_5min": 0.0}), symbols_list
    )
    pitcher_payload['is_new_minute'] = int(closest_ts) % 60 == 0
    pitcher_payload['symbols_with_data'] = opt_s4['symbols_with_data'] # Hack for unit testing mock state
    
    # 5. Simulate Serialization Roundtrip for Pitcher
    pitcher_payload = ser.unpack(ser.pack(pitcher_payload))

    # 6. Deep Comparison of Context Fields
    print(f"\n🔍 [PHASE 1] Payload Attribute Audit (TS={closest_ts}):")
    # Find a valid symbol that has option data
    sym_idx = -1
    for i, s in enumerate(symbols_list):
        if s in opt_s4['symbols_with_data']:
            sym_idx = i
            break
            
    if sym_idx == -1: sym_idx = 0
    sym = symbols_list[sym_idx]
    
    diff_found = False
    compare_keys = [('stock_price', float), ('precalc_alpha', float), ('fast_vol', float), ('spy_roc_5min', float)]
    for key, ctype in compare_keys:
        s4_val = ctype(s4_packet[key][sym_idx]) if s4_packet[key] is not None and len(s4_packet[key]) > 0 else "MISSING"
        live_val = ctype(pitcher_payload[key][sym_idx]) if pitcher_payload.get(key) is not None and len(pitcher_payload[key]) > 0 else "MISSING"
        if abs(s4_val - live_val) > 1e-5:
            print(f"❌ DIFF in {key} for {sym}: S4={s4_val} vs LIVE={live_val}")
            diff_found = True
            
    # 7. Compare Option Extraction inside SE
    print(f"\n🔍 [PHASE 2] Signal Engine Option Extraction Parity for symbol {sym}:")
    se_s4 = SignalEngineV8(symbols=symbols_list, mode='backtest')
    se_live = SignalEngineV8(symbols=symbols_list, mode='realtime')
    
    # Init blank dummy states
    class MockState:
        def __init__(self):
            self.position = 0
            self.latest_call_id = ""
            self.latest_put_id = ""
            self.last_spread_pct = 0.0
            
    st_mock = MockState()

    opt_s4_extracted = se_s4._get_opt_data_backtest(s4_packet, sym_idx, sym, st_mock)
    opt_live_extracted = se_live._get_opt_data_realtime(sym, st_mock, pd.to_datetime(f"{date_str} 09:45:00").time(), s4_packet['stock_price'][sym_idx], pitcher_payload)

    opt_keys = ['call_price', 'put_price', 'call_bid', 'call_ask', 'call_bid_size', 'call_k', 'call_iv']
    for key in opt_keys:
        s4_v = opt_s4_extracted.get(key, 'N/A')
        live_v = opt_live_extracted.get(key, 'N/A')
        
        # Avoid float precision mismatch warnings
        if isinstance(s4_v, float) and isinstance(live_v, float):
             if abs(s4_v - live_v) > 1e-4:
                 print(f"❌ OPTION DIFF in {key}: S4={s4_v} vs LIVE={live_v}")
                 diff_found = True
             else:
                 print(f"✅ OPTION MATCH {key}: {s4_v}")
        elif s4_v != live_v:
             print(f"❌ OPTION DIFF in {key}: S4={s4_v} vs LIVE={live_v}")
             diff_found = True
             
    if not diff_found:
         print("\n✅ [SUCCESS] Payload & Extraction is exactly identical.")
    else:
         print("\n⚠️ Differences found. These disparities cause the PnL gap.")

if __name__ == "__main__":
    test_single_frame_parity("20260305")

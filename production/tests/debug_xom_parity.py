import sqlite3
import pandas as pd
import json
import os
import sys
import numpy as np
from datetime import datetime
import pytz

# Add production and baseline path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../baseline')))
from baseline.strategy_config import StrategyConfig
from baseline.strategy_core_v1 import StrategyCoreV1

def parse_price(json_str, action='call'):
    if not json_str: return 0.0
    try:
        d = json.loads(json_str)
        b = d.get('buckets', [])
        idx = 2 if action == 'call' else 0
        if len(b) > idx and len(b[idx]) >= 8: return float(b[idx][0])
    except: pass
    return 0.0

def load_xom_data(db_path):
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    query_a = "SELECT ts, symbol, alpha, vol_z, event_prob FROM alpha_logs WHERE symbol='XOM'"
    query_s = "SELECT ts, symbol, close FROM market_bars_1m WHERE symbol='XOM'"
    query_o = "SELECT ts, symbol, buckets_json FROM option_snapshots_1m WHERE symbol='XOM'"
    
    df_a = pd.read_sql(query_a, conn)
    df_s = pd.read_sql(query_s, conn)
    df_o = pd.read_sql(query_o, conn)
    conn.close()
    return df_a, df_s, df_o

def run_debug_parity(date_str="20260312"):
    db_path = f"/Users/fangshuai/Documents/GitHub/option-qt/production/preprocess/backtest/history_sqlite_1m/market_{date_str}.db"
    df_a, df_s, df_o = load_xom_data(db_path)
    
    cfg = StrategyConfig()
    # Align parity settings
    cfg.ALPHA_ENTRY_STRICT = 1.45
    cfg.MAX_POSITIONS = 5
    cfg.EXIT_EARLY_STOP_ENABLED = True
    cfg.EXIT_NO_MOMENTUM_ENABLED = True
    # Align Plan A specific thresholds
    cfg.EARLY_STOP_MINS = 5
    cfg.EARLY_STOP_ROI = -0.05
    cfg.NO_MOMENTUM_MINS = 5
    cfg.NO_MOMENTUM_MIN_MAX_ROI = 0.02
    
    # Merge exactly like Plan A (Inner Join)
    df = pd.merge(df_a, df_s, on=['ts','symbol'], how='inner')
    df = pd.merge(df, df_o, on=['ts','symbol'], how='inner')
    df = df.sort_values('ts')
    
    all_ts = sorted(df['ts'].unique())
    ny_tz = pytz.timezone('America/New_York')
    
    # Plan A State
    pa_pos = None # {sym, typ, qty, ep, ets, alp, mx}
    pa_trades = []
    
    # S4 State
    s4_pos = None # {sym, typ, qty, entry_price, entry_ts, max_roi, entry_stock}
    s4_trades = []
    strategy = StrategyCoreV1(cfg)
    
    print(f"\n{'Time':<10} | {'Plan A ROI':<12} | {'S4 ROI':<12} | {'PA MaxROI':<12} | {'S4 MaxROI':<12} | {'Action'}")
    print("-" * 85)
    
    lpm = {} # (sym, typ) -> price
    
    for ts in all_ts:
        row = df[df['ts'] == ts].iloc[0]
        curr_time_str = datetime.fromtimestamp(ts, tz=pytz.utc).astimezone(ny_tz).strftime('%H:%M:%S')
        
        # Update LPM
        for act in ['call', 'put']:
            p = parse_price(row['buckets_json'], act)
            if p > 0.01: lpm[('XOM', act)] = p
            
        # 1. Update existing positions
        # --- Plan A Logic ---
        if pa_pos:
            p = lpm.get((pa_pos['sym'], pa_pos['typ']), 0.0)
            roi = (p - pa_pos['ep']) / pa_pos['ep'] if p > 0.01 else 0.0
            pa_pos['mx'] = max(pa_pos['mx'], roi)
            hm = (ts - pa_pos['ets']) / 60.0
            
            reason = None
            if hm <= cfg.EARLY_STOP_MINS and roi < cfg.EARLY_STOP_ROI: reason = "EARLY"
            if not reason and hm > cfg.NO_MOMENTUM_MINS and pa_pos['mx'] < cfg.NO_MOMENTUM_MIN_MAX_ROI: reason = "NO_MOM"
            
            if reason:
                pa_trades.append({'ts': ts, 'reason': reason, 'roi': roi})
                pa_pos = None
                
        # --- S4 Logic ---
        if s4_pos:
            p = lpm.get(('XOM', s4_pos['typ']), 0.0)
            roi = (p - s4_pos['entry_price']) / s4_pos['entry_price'] if p > 0.01 else 0.0
            s4_pos['max_roi'] = max(s4_pos['max_roi'], roi)
            hm = (ts - s4_pos['entry_ts']) / 60.0
            
            ctx = {
                'symbol': 'XOM', 'curr_ts': ts, 'curr_price': p, 'curr_stock': row['close'],
                'time': datetime.fromtimestamp(ts, tz=pytz.utc).astimezone(ny_tz),
                'holding': s4_pos,
                'alpha_z': row['alpha'],
                'event_prob': row['event_prob']
            }
            exit_sig = strategy.check_exit(ctx)
            if exit_sig:
                s4_trades.append({'ts': ts, 'reason': exit_sig['reason'], 'roi': roi})
                s4_pos = None
                
        # 2. Log State
        p_roi = f"{roi*100:6.2f}%" if pa_pos else "N/A"
        s_roi = f"{roi*100:6.2f}%" if s4_pos else "N/A"
        p_mx = f"{pa_pos['mx']*100:6.2f}%" if pa_pos else "N/A"
        s_mx = f"{s4_pos['max_roi']*100:6.2f}%" if s4_pos else "N/A"
        
        print(f"{curr_time_str:<10} | {p_roi:<12} | {s_roi:<12} | {p_mx:<12} | {s_mx:<12} |", end="")
        
        # 3. New Entries (09:35 only for simplicity if that's when it happens)
        if curr_time_str == "09:35:00":
            pe = parse_price(row['buckets_json'], 'call')
            entry_ep = pe * (1 + cfg.SLIPPAGE_PCT)
            pa_pos = {'sym': 'XOM', 'typ': 'call', 'qty': 1, 'ep': entry_ep, 'ets': ts, 'alp': row['alpha'], 'mx': 0.0}
            s4_pos = {
                'typ': 'call', 'entry_price': entry_ep, 'entry_ts': ts, 'max_roi': 0.0, 
                'entry_stock': row['close'], 'dir': 1, 'symbol': 'XOM'
            }
            print(" ENTRY", end="")
        
        # Check if exited this tick
        if not pa_pos and len(pa_trades) > 0 and pa_trades[-1]['ts'] == ts:
            print(f" PA_EXIT({pa_trades[-1]['reason']})", end="")
        if not s4_pos and len(s4_trades) > 0 and s4_trades[-1]['ts'] == ts:
            print(f" S4_EXIT({s4_trades[-1]['reason']})", end="")
            
        print()

if __name__ == "__main__":
    run_debug_parity()

import sys, os
sys.path.insert(0, os.path.join(os.getcwd(), '../../baseline'))
import pandas as pd
import sqlite3
import json
from strategy_config import StrategyConfig
import plan_a_fine_tune as paf

def run_buggy_sim(df_a, df_s, df_o, cfg: StrategyConfig, es_roi, mom_mins, mom_max, abs_sl, alpha_min=None):
    """
    REPRODUCE THE BUG: Filter alpha before merging.
    This causes the engine to 'blink' and stop monitoring positions when alpha is weak.
    """
    if alpha_min is None: alpha_min = cfg.ALPHA_ENTRY_STRICT
    
    # --- PRODUCING THE BUG HERE ---
    da = df_a[df_a['alpha'].abs() >= alpha_min] # PRE-FILTERING (THE BUG)
    df = pd.merge(da, df_s, on=['ts','symbol'], how='inner')
    df = pd.merge(df, df_o, on=['ts','symbol'], how='inner')
    # ------------------------------
    
    dbt = {ts: g.set_index('symbol').to_dict('index') for ts, g in df.groupby('ts')}
    ts_list = sorted(dbt.keys())
    
    pos_list = []
    trades = []
    cash = cfg.INITIAL_ACCOUNT
    
    for ts in ts_list:
        cd = dbt[ts]
        lpm = { (p['sym'], p['typ']): p['ep'] for p in pos_list }
        
        # Exit Logic
        new_pos = []
        for p in pos_list:
            row = cd.get(p['sym'])
            if row:
                # Monitoring logic...
                pass
            else:
                # MISSING DATA (The result of the bug)
                # The position just 'hangs' until alpha gets strong again.
                pass
        # ... (rest of the sim) ...
    return 0.0, [] # Dummy for now, I'll use paf logic inside

def run_experiment():
    cfg = StrategyConfig()
    cfg.MAX_POSITIONS = 4
    cfg.COMMISSION_PER_CONTRACT = 0.0
    cfg.SLIPPAGE_PCT = 0.001
    
    dates = ["20260102", "20260105", "20260106"]
    
    print(f"{'Date':<10} | {'Fixed ROI':<10} | {'Buggy ROI (Simulated)':<10}")
    print("-" * 50)
    
    for d in dates:
        db_path = f"history_sqlite_1m/market_{d}.db"
        a, s, o = paf.load_day(db_path)
        
        # 1. FIXED VERSION (Current paf)
        roi_fixed, _ = paf.run_sim(a, s, o, cfg, -0.05, 10, 0.04, -0.15)
        
        # 2. BUGGY VERSION (Filter inside)
        # I'll temporarily modify paf.run_sim logic by pre-filtering 'a'
        a_buggy = a[a['alpha'].abs() >= 1.45]
        roi_buggy, _ = paf.run_sim(a_buggy, s, o, cfg, -0.05, 10, 0.04, -0.15)
        
        print(f"{d:<10} | {roi_fixed:>10.2%} | {roi_buggy:>10.2%}")

if __name__ == "__main__":
    run_experiment()

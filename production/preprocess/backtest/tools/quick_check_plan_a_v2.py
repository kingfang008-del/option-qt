import sys, os
sys.path.insert(0, os.path.join(os.getcwd(), '../../baseline'))
import pandas as pd
import sqlite3
import json
from strategy_config import StrategyConfig
# Import directly from the file
import plan_a_fine_tune as paf

def run_parity_check():
    cfg = StrategyConfig()
    cfg.MAX_POSITIONS = 4
    cfg.COMMISSION_PER_CONTRACT = 0.0
    cfg.SLIPPAGE_PCT = 0.001
    
    dates = ["20260102", "20260105", "20260106"]
    
    print(f"{'Date':<10} | {'Trades':<6} | {'ROI':<10}")
    print("-" * 30)
    
    for d in dates:
        db_path = f"history_sqlite_1m/market_{d}.db"
        if not os.path.exists(db_path):
            continue
            
        a, s, o = paf.load_day(db_path)
        # Match standard Plan A thresholds
        trades, final_cash = paf.run_sim(a, s, o, cfg, 
                                        es_roi=-0.05, 
                                        mom_mins=10, 
                                        mom_max=0.04, 
                                        abs_sl=-0.15)
        
        roi = (final_cash - cfg.INITIAL_ACCOUNT) / cfg.INITIAL_ACCOUNT
        print(f"{d:<10} | {len(trades):<6} | {roi:>10.2%}")

if __name__ == "__main__":
    run_parity_check()

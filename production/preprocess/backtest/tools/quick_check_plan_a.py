import sys, os
sys.path.insert(0, os.path.join(os.getcwd(), '../../baseline'))
from plan_a_fine_tune import load_day, run_sim
from strategy_config import StrategyConfig
import pandas as pd

def quick_check():
    cfg = StrategyConfig()
    # Align with S4 parity settings
    cfg.MAX_POSITIONS = 4
    cfg.COMMISSION_PER_CONTRACT = 0.0
    cfg.SLIPPAGE_PCT = 0.001
    
    dates = ["20260102", "20260105", "20260106"]
    results = []
    
    print(f"{'Date':<10} | {'Trades':<6} | {'ROI':<10} | {'Exit Reasons'}")
    print("-" * 50)
    
    for d in dates:
        db_path = f"history_sqlite_1m/market_{d}.db"
        if not os.path.exists(db_path):
            print(f"❌ {d} not found")
            continue
            
        a, s, o = load_day(db_path)
        # Default Plan A parameters used in parity test
        trades, final_cash = run_sim(a, s, o, cfg, 
                                    es_roi=-0.05, 
                                    mom_mins=10, 
                                    mom_max=0.04, 
                                    abs_sl=-0.15)
        
        total_roi = (final_cash - cfg.INITIAL_ACCOUNT) / cfg.INITIAL_ACCOUNT
        reasons = pd.Series([t['reason'] for t in trades]).value_counts().to_dict()
        results.append({'date': d, 'roi': total_roi, 'trades': len(trades)})
        
        print(f"{d:<10} | {len(trades):<6} | {total_roi:>10.2%} | {reasons}")

if __name__ == "__main__":
    quick_check()

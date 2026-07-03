import sys, os
sys.path.insert(0, os.path.join(os.getcwd(), '../../baseline'))
import plan_a_fine_tune as paf
from strategy_config import StrategyConfig

def final_parity():
    cfg = StrategyConfig()
    cfg.MAX_POSITIONS = 4
    cfg.COMMISSION_PER_CONTRACT = 0.0
    cfg.SLIPPAGE_PCT = 0.001
    
    # EXACT S4 PARITY SETTINGS
    es_roi = -0.05
    mom_mins = 10
    mom_max = 0.02 # MATCHING StrategyConfig
    abs_sl = -0.20 # MATCHING StrategyConfig
    
    dates = ["20260102", "20260105", "20260106"]
    
    print(f"{'Date':<10} | {'Trades':<6} | {'ROI':<10}")
    print("-" * 30)
    
    for d in dates:
        db_path = f"history_sqlite_1m/market_{d}.db"
        a, s, o = paf.load_day(db_path)
        roi, trades = paf.run_sim(a, s, o, cfg, es_roi, mom_mins, mom_max, abs_sl)
        print(f"{d:<10} | {len(trades):<6} | {roi:>10.2%}")

if __name__ == "__main__":
    final_parity()

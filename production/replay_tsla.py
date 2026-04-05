import psycopg2
import pandas as pd
from datetime import datetime
from pytz import timezone
import sys

sys.path.append('/Users/fangshuai/Documents/GitHub/option-qt/production/baseline')
from config import PG_DB_URL, NY_TZ, GLOBAL_STATS, ROLLING_WINDOW
from strategy_core_v1 import StrategyCoreV1, StrategyConfig
from collections import deque
import numpy as np

class SymbolState:
    def __init__(self, symbol):
        self.symbol = symbol
        self.prices = deque(maxlen=60)
        self.alpha_history = deque(maxlen=ROLLING_WINDOW + 10)
        self.pct_history = deque(maxlen=ROLLING_WINDOW + 10)
        self.correction_mode = "NORMAL"
        self.ema_fast_val = None; self.ema_slow_val = None; self.dea_val = None
        self.k_fast = 2 / (8 + 1); self.k_slow = 2 / (21 + 1); self.k_sig  = 2 / (5 + 1)
        self.position = 0
        self.prev_macd_hist = 0.0
        self.last_alpha_z = 0.0
        self.prev_alpha_z = 0.0
        self.last_snap_roc = 0.0

    def update_indicators(self, price, raw_alpha_score):
        prev_price = self.prices[-1] if len(self.prices) > 0 else price
        self.prices.append(price)
        roc_5m = 0.0
        if len(self.prices) >= 6:
            prev_5m = self.prices[-6]
            if prev_5m > 0: roc_5m = (price - prev_5m) / prev_5m
        pct_change = 0.0
        if prev_price > 0: pct_change = (price - prev_price) / prev_price

        self.alpha_history.append(raw_alpha_score)
        self.pct_history.append(pct_change)

        if self.ema_fast_val is None:
            self.ema_fast_val = price; self.ema_slow_val = price; self.dea_val = 0.0
        else:
            self.ema_fast_val = price * self.k_fast + self.ema_fast_val * (1 - self.k_fast)
            self.ema_slow_val = price * self.k_slow + self.ema_slow_val * (1 - self.k_slow)
            dif = self.ema_fast_val - self.ema_slow_val
            self.dea_val = dif * self.k_sig + self.dea_val * (1 - self.k_sig)
        macd_hist = (self.ema_fast_val - self.ema_slow_val) - self.dea_val
        macd_hist_slope = macd_hist - self.prev_macd_hist
        self.prev_macd_hist = macd_hist
        return roc_5m, macd_hist, macd_hist_slope, pct_change

def replay_tsla():
    conn = psycopg2.connect(PG_DB_URL)
    
    # Load TSLA market bars and alpha logs for today
    now = datetime.now(NY_TZ)
    start_of_day = now.replace(hour=0, minute=0, second=0, microsecond=0)
    start_ts = int(start_of_day.timestamp())
    
    # Get alpha logs
    alpha_query = f"""
    SELECT ts, symbol, alpha, iv, price, vol_z 
    FROM alpha_logs WHERE symbol IN ('TSLA', 'SPY', 'QQQ') AND ts >= {start_ts} ORDER BY ts ASC
    """
    df_alpha = pd.read_sql(alpha_query, conn)
    conn.close()
    
    if df_alpha.empty:
        print("No alpha logs found.")
        return
        
    df_alpha['dt'] = pd.to_datetime(df_alpha['ts'], unit='s').dt.tz_localize('UTC').dt.tz_convert('America/New_York')
    
    # Sort into grouped timestamps
    timestamps = sorted(df_alpha['ts'].unique())
    
    # Init Strategy and States
    strategy = StrategyCoreV1()
    states = {'TSLA': SymbolState('TSLA'), 'SPY': SymbolState('SPY'), 'QQQ': SymbolState('QQQ')}
    
    index_opening_prices = {}
    spy_ema_roc = 0.0
    qqq_ema_roc = 0.0
    
    for t in timestamps:
        batch_df = df_alpha[df_alpha['ts'] == t]
        
        # Prepare prices and alphas
        prices = {}
        alphas = {}
        vols = {}
        
        for _, row in batch_df.iterrows():
            sym = row['symbol']
            prices[sym] = row['price']
            alphas[sym] = row['alpha']
            vols[sym] = row['vol_z']
            
        dt_ny = datetime.fromtimestamp(t, NY_TZ)
        time_str = dt_ny.strftime('%H:%M:%S')
        
        # Update index opening prices
        spy_day_roc, qqq_day_roc = 0.0, 0.0
        for sym in ['SPY', 'QQQ']:
            if sym in prices:
                p = prices[sym]
                if sym not in index_opening_prices and p > 1.0:
                    index_opening_prices[sym] = p
                if sym in index_opening_prices:
                    open_p = index_opening_prices[sym]
                    if sym == 'SPY': spy_day_roc = (p - open_p) / open_p
                    else: qqq_day_roc = (p - open_p) / open_p
                    
        # Update metrics for all
        metrics = {}
        spy_roc_5min = 0.0
        qqq_roc_5min = 0.0
        for sym in ['SPY', 'QQQ', 'TSLA']:
            if sym in prices:
                st = states[sym]
                roc_5m, macd, macd_slope, snap_roc = st.update_indicators(prices[sym], alphas[sym])
                
                if sym == 'SPY': spy_roc_5min = roc_5m
                elif sym == 'QQQ': qqq_roc_5min = roc_5m
                
                # Replicate exactly alpha_z logic
                # system_orchestrator_v8 logs alpha directly as passed (it already is alpha_z in alpha_logs? No, in DB it logs final_alpha)
                # Wait: data in alpha_logs `alpha` IS `final_alpha`. `vol_z` IS already scaled.
                # Let's assume the DB `alpha` is already the `final_alpha`.
                
                metrics[sym] = {
                    'price': prices[sym],
                    'alpha_z': alphas[sym],
                    'vol_z': vols[sym],
                    'roc_5m': roc_5m,
                    'macd': macd,
                    'macd_slope': macd_slope,
                    'snap_roc': snap_roc,
                    'st': st
                }
                st.last_snap_roc = snap_roc
        
        # Index Trend Calculation
        s_5m = spy_roc_5min
        q_5m = qqq_roc_5min
        alpha_ema = 0.4
        spy_ema_roc = alpha_ema * s_5m + (1 - alpha_ema) * spy_ema_roc
        qqq_ema_roc = alpha_ema * q_5m + (1 - alpha_ema) * qqq_ema_roc
        
        index_trend = 0
        is_bull_day = (spy_day_roc > 0.0002 or qqq_day_roc > 0.0002)
        is_bear_day = (spy_day_roc < -0.0002 or qqq_day_roc < -0.0002)
        
        if is_bull_day and (spy_ema_roc > -0.0001 or qqq_ema_roc > -0.0001): index_trend = 1
        elif is_bear_day and (spy_ema_roc < 0.0001 or qqq_ema_roc < 0.0001): index_trend = -1
        
        # Evaluate TSLA
        if 'TSLA' in metrics and dt_ny.hour == 9 and 50 <= dt_ny.minute <= 58:
            sym_metrics = metrics['TSLA']
            
            ctx = {
                'symbol': 'TSLA', 'time': dt_ny, 'curr_ts': t, 'price': sym_metrics['price'],
                'alpha_z': sym_metrics['alpha_z'], 'vol_z': sym_metrics['vol_z'], 
                'stock_roc': sym_metrics['roc_5m'],
                'macd_hist': sym_metrics['macd'], 'macd_hist_slope': sym_metrics['macd_slope'],
                'spy_roc': spy_roc_5min, 'qqq_roc': qqq_roc_5min,
                'index_trend': index_trend,
                'position': 0, 'cooldown_until': 0.0,
                'is_ready': True, # force true
                'is_banned': False,
                'held_mins': 0.0,
                'stock_iv': 0.5,
                'holding': None,
                'curr_price': 0.0, 'curr_stock': sym_metrics['price'],
                'bid': 1.0, 'ask': 1.1, 'spread_divergence': 0.0,
                'snap_roc': sym_metrics['snap_roc']
            }
            
            sig = strategy.decide_entry(ctx)
            print(f"[{time_str}] TSLA Alpha: {sym_metrics['alpha_z']:.2f} | MACD: {sym_metrics['macd']:.4f} | TSLA 5m ROC: {sym_metrics['roc_5m']*100:.3f}% | Snap ROC: {sym_metrics['snap_roc']*100:.3f}% | SPY 5m ROC: {spy_roc_5min*100:.3f}% | Trend: {index_trend}")
            if sig:
                print(f"   => SIGNAL FIRED: {sig}")
            else:
                s = strategy.cfg
                # Manual checks to see what failed
                if sym_metrics['alpha_z'] < s.ALPHA_ENTRY_THRESHOLD: print("   => Blocked: Alpha Threshold")
                elif not (s.VOL_MIN_Z < sym_metrics['vol_z'] < s.VOL_MAX_Z): print(f"   => Blocked: Vol_z out of bounds ({sym_metrics['vol_z']:.2f})")
                elif sym_metrics['roc_5m'] < -s.STOCK_MOMENTUM_TOLERANCE: print(f"   => Blocked: Stock ROC < {-s.STOCK_MOMENTUM_TOLERANCE}")
                elif sym_metrics['snap_roc'] < s.MIN_LAST_SNAP_ROC: print(f"   => Blocked: Snap ROC {sym_metrics['snap_roc']} < {s.MIN_LAST_SNAP_ROC}")
                elif sym_metrics['macd'] <= s.MACD_HIST_THRESHOLD: print(f"   => Blocked: MACD {sym_metrics['macd']:.4f} <= {s.MACD_HIST_THRESHOLD}")
                elif spy_roc_5min < s.INDEX_ROC_THRESHOLD: print(f"   => Blocked: SPY ROC {spy_roc_5min} < {s.INDEX_ROC_THRESHOLD}")

if __name__ == '__main__':
    replay_tsla()

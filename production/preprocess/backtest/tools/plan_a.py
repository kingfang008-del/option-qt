import sqlite3
import pandas as pd
import json
import numpy as np
import os
from datetime import datetime, timedelta
from dataclasses import dataclass

# ================= Production Configuration =================
@dataclass
class ProdConfig:
    DB_PATH = "/home/kingfang007/quant_project/data/history_sqlite_1m/market_20260313.db"
    INITIAL_ACCOUNT = 50000.0
    MAX_POSITIONS = 4
    POSITION_RATIO = 1.0 / 4.0
    COMMISSION = 0.65
    SLIPPAGE = 0.001 
    
    START_HOUR = 9
    START_MINUTE = 40
    MIN_ENTRY_PRICE = 1.0
    
    # --- Strategy Entry ---
    EVENT_PROB_THRESHOLD = 0.3
    
    # --- Strategy Exit ---
    STOP_LOSS = -0.15
    TIME_STOP_MINS = 120
    TIME_STOP_ROI = 0.05
    
    # [TIGHT] For Low Alpha
    TIGHT_LADDER = [(0.12, 0.10), (0.25, 0.20), (0.35, 0.30), (0.50, 0.40)]
    # [WIDE] For High Alpha (e.g. >= 2.0)
    WIDE_LADDER = [(0.30, 0.20), (0.50, 0.40), (0.85, 0.70), (1.50, 1.25)]

def get_ny_time(ts):
    dt = datetime.fromtimestamp(ts)
    return (dt - timedelta(hours=13)).strftime('%H:%M')

def parse_price(json_str, action):
    if not json_str: return 0.0, 0.0
    try:
        data = json.loads(json_str)
        buckets = data.get('buckets', [])
        idx = 2 if action == 'call' else 0
        if len(buckets) > idx and len(buckets[idx]) >= 8:
            b = buckets[idx]
            return float(b[0]), float(b[5])
    except: pass
    return 0.0, 0.0

def check_ladder_exit(roi, max_roi, alpha, high_alpha_th=2.0):
    cfg = ProdConfig()
    ladder = cfg.WIDE_LADDER if abs(alpha) >= high_alpha_th else cfg.TIGHT_LADDER
    for trigger, floor in reversed(ladder):
        if max_roi >= trigger:
            if roi < floor: return f"LADD_{int(trigger*100)}%"
            break
    return None

def run_sim(mode="DYNAMIC", alpha_min=2, high_alpha_th=2.0):
    cfg = ProdConfig()
    conn = sqlite3.connect(f"file:{cfg.DB_PATH}?mode=ro", uri=True)
    df_alpha = pd.read_sql(f"SELECT ts, symbol, alpha, event_prob FROM alpha_logs WHERE abs(alpha) >= {alpha_min}", conn)
    df_stock = pd.read_sql("SELECT ts, symbol, close FROM market_bars_1m", conn)
    df_opt = pd.read_sql("SELECT ts, symbol, buckets_json FROM option_snapshots_1m", conn)
    conn.close()
    
    df = pd.merge(df_alpha, df_stock, on=['ts', 'symbol'], how='inner')
    df = pd.merge(df, df_opt, on=['ts', 'symbol'], how='inner')
    df = df.sort_values(['symbol', 'ts'])
    
    data_by_ts = {ts: group.set_index('symbol').to_dict('index') for ts, group in df.groupby('ts')}
    all_ts = sorted(list(data_by_ts.keys()))
    
    cash, positions, trade_history, last_price_map = cfg.INITIAL_ACCOUNT, [], [], {}
    cur_slippage, cur_comm = cfg.SLIPPAGE, cfg.COMMISSION

    for ts in all_ts:
        curr_data = data_by_ts[ts]
        active = []
        for sym, row in curr_data.items():
            for action in ['call', 'put']:
                p, _ = parse_price(row['buckets_json'], action)
                if p > 0.01: last_price_map[(sym, action)] = p

        for pos in positions:
            p = last_price_map.get((pos['symbol'], pos['type']), 0.0)
            if p <= 0.01: active.append(pos); continue
            
            roi = (p - pos['entry_price']) / pos['entry_price']
            pos['max_roi'] = max(pos['max_roi'], roi)
            held_mins = (ts - pos['entry_ts'])/60
            
            reason = None
            if mode != "NONE":
                if roi < cfg.STOP_LOSS: reason = "STOP_LOSS"
                elif held_mins > cfg.TIME_STOP_MINS and roi < cfg.TIME_STOP_ROI: reason = "TIME_INACTIVE"
                elif mode == "DYNAMIC": reason = check_ladder_exit(roi, pos['max_roi'], pos['alpha'], high_alpha_th)
                elif mode == "TIGHT": reason = check_ladder_exit(roi, pos['max_roi'], 0.0, high_alpha_th) 
            
            if reason:
                exit_p = p * (1 - cur_slippage)
                cash += (exit_p * pos['qty'] * 100) - (pos['qty'] * cur_comm)
                trade_history.append({
                    'symbol': pos['symbol'], 
                    'alpha': pos['alpha'], 
                    'entry_t': get_ny_time(pos['entry_ts']), 
                    'exit_t': get_ny_time(ts),
                    'entry_p': pos['entry_price'],
                    'exit_p': exit_p,
                    'roi': roi, 
                    'max': pos['max_roi'], 
                    'reason': reason
                })
            else: active.append(pos)
        positions = active
        
        hour_local = datetime.fromtimestamp(ts).hour
        min_local = datetime.fromtimestamp(ts).minute
        is_ready = (hour_local > 22) or (hour_local == 22 and min_local >= 40)
        
        if is_ready and len(positions) < cfg.MAX_POSITIONS:
            for sym, row in curr_data.items():
                if any(p['symbol'] == sym for p in positions): continue
                if row['event_prob'] < cfg.EVENT_PROB_THRESHOLD: continue
                action = 'call' if row['alpha'] > 0 else 'put'
                p_entry = last_price_map.get((sym, action), 0.0)
                if p_entry >= cfg.MIN_ENTRY_PRICE:
                    qty = int((cash * cfg.POSITION_RATIO) // (p_entry * 100))
                    if qty >= 1:
                        entry_p = p_entry * (1 + cur_slippage)
                        cash -= (qty * entry_p * 100) + (qty * cur_comm)
                        positions.append({
                            'symbol': sym, 'type': action, 'qty': qty, 
                            'entry_price': entry_p, 'entry_ts': ts, 
                            'alpha': row['alpha'], 'max_roi': 0.0
                        })
        
    for pos in positions:
        p = last_price_map.get((pos['symbol'], pos['type']), pos['entry_price'])
        roi = (p - pos['entry_price']) / pos['entry_price']
        trade_history.append({
            'symbol': pos['symbol'], 'alpha': pos['alpha'], 
            'entry_t': get_ny_time(pos['entry_ts']), 'exit_t': '16:00',
            'entry_p': pos['entry_price'], 'exit_p': p,
            'roi': roi, 'max': pos['max_roi'], 'reason': 'EOD'
        })
        
    return (cash + sum([p['qty'] * last_price_map.get((p['symbol'], p['type']), p['entry_price']) * 100 for p in positions])), trade_history

if __name__ == "__main__":
    print(f"🧩 PROJECT ODYSSEY: Audit Suite V3.1 - Enhanced Price Telemetry")
    print("-" * 60)
    
    mode = "DYNAMIC" if os.getenv("SIM_MODE") == "DYNAMIC" else "TIGHT"
    eq, trades = run_sim(mode=mode, alpha_min=2, high_alpha_th=2)
    df = pd.DataFrame(trades)
    
    print(f"FINAL RESULT ({mode}): ${(eq/50000.0-1):.2%} | Trades: {len(df)}")
    if not df.empty:
        # Reorder columns for better visibility
        cols = ['symbol', 'alpha', 'entry_t', 'exit_t', 'entry_p', 'exit_p', 'roi', 'max', 'reason']
        print(df[cols].sort_values('roi', ascending=False).head(20))

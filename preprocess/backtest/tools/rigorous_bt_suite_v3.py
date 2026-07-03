#!/usr/bin/env python3
"""
rigorous_bt_suite_v3.py - 高保真审计套件 (V4 - 统一逻辑版)
与 strategy_core_v1.py 100% 对齐的简化审计工具
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../baseline'))

import sqlite3, pandas as pd, json, numpy as np
from datetime import datetime, timedelta
from strategy_config import StrategyConfig

DB_PATH = os.path.join(os.path.dirname(__file__), "market_20260102.db")

def get_ny_time(ts):
    return (datetime.fromtimestamp(ts) - timedelta(hours=13)).strftime('%H:%M')

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

def run_sim(cfg: StrategyConfig, alpha_min=1.45):
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    df_alpha = pd.read_sql(f"SELECT ts, symbol, alpha, event_prob FROM alpha_logs WHERE abs(alpha) >= {alpha_min}", conn)
    df_stock = pd.read_sql("SELECT ts, symbol, close FROM market_bars_1m", conn)
    df_opt = pd.read_sql("SELECT ts, symbol, buckets_json FROM option_snapshots_1m", conn)
    conn.close()
    
    df = pd.merge(df_alpha, df_stock, on=['ts', 'symbol'], how='inner')
    df = pd.merge(df, df_opt, on=['ts', 'symbol'], how='inner')
    df = df.sort_values(['symbol', 'ts'])
    
    data_by_ts = {ts: group.set_index('symbol').to_dict('index') for ts, group in df.groupby('ts')}
    all_ts = sorted(list(data_by_ts.keys()))
    
    cash, positions, trade_history, lpm = cfg.INITIAL_ACCOUNT, [], [], {}

    for ts in all_ts:
        curr_data = data_by_ts[ts]
        active = []
        for sym, row in curr_data.items():
            for action in ['call', 'put']:
                p, _ = parse_price(row['buckets_json'], action)
                if p > 0.01: lpm[(sym, action)] = p

        for pos in positions:
            p = lpm.get((pos['symbol'], pos['type']), 0.0)
            if p <= 0.01: active.append(pos); continue
            
            roi = (p - pos['entry_price']) / pos['entry_price']
            pos['max_roi'] = max(pos['max_roi'], roi)
            held_mins = (ts - pos['entry_ts'])/60
            
            reason = None
            # Plan A: Early Stop
            if held_mins <= cfg.EARLY_STOP_MINS and roi < cfg.EARLY_STOP_ROI: reason = "EARLY_STOP"
            # Plan A: No Momentum
            elif held_mins > cfg.NO_MOMENTUM_MINS and pos['max_roi'] < cfg.NO_MOMENTUM_MIN_MAX_ROI: reason = "NO_MOM"
            # Hard Stop
            elif roi < cfg.ABSOLUTE_STOP_LOSS: reason = "HARD_STOP"
            # Time Stop
            elif held_mins > cfg.TIME_STOP_MINS and roi < 0.0: reason = "TIME_INACTIVE"
            # Ladder
            else:
                ladder = cfg.LADDER_WIDE if abs(pos['alpha']) >= cfg.HIGH_ALPHA_WIDE_THRESHOLD else cfg.LADDER_TIGHT
                for trigger, floor in reversed(ladder):
                    if pos['max_roi'] >= trigger:
                        if roi < floor: reason = f"LADD_{int(trigger*100)}%"; break
            
            if reason:
                exit_p = p * (1 - cfg.SLIPPAGE_PCT)
                cash += (exit_p * pos['qty'] * 100) - (pos['qty'] * cfg.COMMISSION_PER_CONTRACT)
                trade_history.append({
                    'symbol': pos['symbol'], 'alpha': pos['alpha'],
                    'entry_t': get_ny_time(pos['entry_ts']), 'exit_t': get_ny_time(ts),
                    'entry_p': pos['entry_price'], 'exit_p': exit_p,
                    'roi': roi, 'max': pos['max_roi'], 'reason': reason
                })
            else: active.append(pos)
        positions = active
        
        curr_time_str = (datetime.fromtimestamp(ts) - timedelta(hours=13)).strftime('%H:%M:%S')
        if curr_time_str >= cfg.START_TIME and len(positions) < cfg.MAX_POSITIONS:
            for sym, row in curr_data.items():
                if any(p['symbol'] == sym for p in positions): continue
                if row['event_prob'] < cfg.EVENT_PROB_THRESHOLD: continue
                action = 'call' if row['alpha'] > 0 else 'put'
                p_entry = lpm.get((sym, action), 0.0)
                if p_entry >= cfg.MIN_OPTION_PRICE:
                    qty = int((cash * cfg.POSITION_RATIO) // (p_entry * 100))
                    if qty >= 1:
                        entry_p = p_entry * (1 + cfg.SLIPPAGE_PCT)
                        cash -= (qty * entry_p * 100) + (qty * cfg.COMMISSION_PER_CONTRACT)
                        positions.append({
                            'symbol': sym, 'type': action, 'qty': qty,
                            'entry_price': entry_p, 'entry_ts': ts,
                            'alpha': row['alpha'], 'max_roi': 0.0
                        })
    
    for pos in positions:
        p = lpm.get((pos['symbol'], pos['type']), pos['entry_price'])
        roi = (p - pos['entry_price']) / pos['entry_price']
        trade_history.append({
            'symbol': pos['symbol'], 'alpha': pos['alpha'],
            'entry_t': get_ny_time(pos['entry_ts']), 'exit_t': '16:00',
            'entry_p': pos['entry_price'], 'exit_p': p,
            'roi': roi, 'max': pos['max_roi'], 'reason': 'EOD'
        })
    
    eq = cash + sum([p['qty'] * lpm.get((p['symbol'], p['type']), p['entry_price']) * 100 for p in positions])
    return eq, trade_history

if __name__ == "__main__":
    cfg = StrategyConfig()
    print("=" * 60)
    print(f"🧩 Rigorous BT Suite V4 - Synced with Plan A Strategy")
    print(f"   START_TIME: {cfg.START_TIME}")
    print(f"   LADDER_WIDE: {cfg.LADDER_WIDE[0][0]*100}% trigger")
    print(f"   EARLY_STOP: {cfg.EARLY_STOP_ROI*100}% at {cfg.EARLY_STOP_MINS}m")
    print("=" * 60)
    
    eq, trades = run_sim(cfg, alpha_min=1.45)
    df = pd.DataFrame(trades)
    ret = (eq / cfg.INITIAL_ACCOUNT - 1)
    print(f"\nFINAL ACCOUNT EQUITY: ${eq:,.2f} | RETURN: {ret:+.2%} | Trades: {len(df)}")
    if not df.empty:
        cols = ['symbol', 'alpha', 'entry_t', 'exit_t', 'entry_p', 'exit_p', 'roi', 'max', 'reason']
        print(df[cols].sort_values('roi', ascending=False).to_string(index=False))

#!/usr/bin/env python3
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../baseline'))

import sqlite3, pandas as pd, json, numpy as np
from datetime import datetime, timedelta
from strategy_config import StrategyConfig

BT_DIR = os.path.dirname(__file__)
DB_FILES = [
    os.path.join(BT_DIR, "market_20260102.db"),
    os.path.join(BT_DIR, "market_20260105.db"),
    os.path.join(BT_DIR, "market_20260106.db"),
]

def parse_price(json_str, action)
    if not json_str: return 0.0
    try:
        d = json.loads(json_str)
        b = d.get('buckets', [])
        idx = 2 if action == 'call' else 0
        if len(b) > idx and len(b[idx]) >= 8: return float(b[idx][0])
    except: pass
    return 0.0

def load_day(db_path):
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    a = pd.read_sql("SELECT ts, symbol, alpha, event_prob FROM alpha_logs", conn)
    s = pd.read_sql("SELECT ts, symbol, close FROM market_bars_1m", conn)
    o = pd.read_sql("SELECT ts, symbol, buckets_json FROM option_snapshots_1m", conn)
    conn.close()
    return a, s, o

def run_sim(df_a, df_s, df_o, cfg, es_roi, no_mom_mins, no_mom_max, abs_sl, min_p=1.0):
    cash = 50000.0
    pos_list, trades, lpm = [], [], {}
    
    da = df_a[df_a['alpha'].abs() >= 1.45]
    df = pd.merge(da, df_s, on=['ts','symbol'], how='inner')
    df = pd.merge(df, df_o, on=['ts','symbol'], how='inner')
    df = df.sort_values(['symbol','ts'])
    dbt = {ts: g.set_index('symbol').to_dict('index') for ts, g in df.groupby('ts')}
    all_ts = sorted(dbt.keys())

    for ts in all_ts:
        cd = dbt[ts]
        active = []
        for sym, row in cd.items():
            for act in ['call','put']:
                p = parse_price(row['buckets_json'], act)
                if p > 0.01: lpm[(sym,act)] = p

        for pos in pos_list:
            p = lpm.get((pos['sym'], pos['typ']), 0.0)
            if p <= 0.01: active.append(pos); continue
            roi = (p - pos['ep']) / pos['ep']
            pos['mx'] = max(pos['mx'], roi)
            hm = (ts - pos['ets']) / 60.0
            
            reason = None
            if hm <= 5 and roi < es_roi: reason = "EARLY"
            if not reason and hm > no_mom_mins and pos['mx'] < no_mom_max: reason = "NO_MOM"
            if not reason and roi < abs_sl: reason = "HARD"
            if not reason and hm > 240 and roi < 0.0: reason = "TIME"
            if not reason:
                ladder = cfg.LADDER_WIDE if abs(pos['alp']) >= 2.5 else cfg.LADDER_TIGHT
                for tr, fl in reversed(ladder):
                    if pos['mx'] >= tr:
                        if roi < fl: reason = "LADD"; break
            
            if reason:
                exit_ep = p * (1 - 0.001)
                cash += (exit_ep * pos['qty'] * 100) - (pos['qty'] * 0.65)
                trades.append({'roi': roi})
            else: active.append(pos)
        pos_list = active
        
        curr_time = datetime.fromtimestamp(ts) - timedelta(hours=13)
        if curr_time.hour == 9 and curr_time.minute >= 30 and len(pos_list) < 4:
            for sym, row in cd.items():
                if any(p['sym'] == sym for p in pos_list): continue
                if row['event_prob'] < 0.7: continue
                act = 'call' if row['alpha'] > 0 else 'put'
                pe = lpm.get((sym, act), 0.0)
                if pe >= 0.1:
                    qty = int((cash * 0.25) // (pe * 100))
                    if qty >= 1:
                        entry_ep = pe * (1 + 0.001)
                        cash -= (qty * entry_ep * 100) + (qty * 0.65)
                        pos_list.append({'sym': sym, 'typ': act, 'qty': qty, 'ep': entry_ep, 'ets': ts, 'alp': row['alpha'], 'mx': 0.0})
    
    for pos in pos_list:
        p = lpm.get((pos['sym'], pos['typ']), pos['ep'])
        roi = (p - pos['ep']) / pos['ep']
        cash += (p * pos['qty'] * 100) - (pos['qty'] * 0.65)
    
    return (cash / 50000.0 - 1)

if __name__ == "__main__":
    cfg = StrategyConfig()
    print("Verifying previous benchmark (+15.1%)...")
    day_rets = []
    for db in DB_FILES:
        a, s, o = load_day(db)
        # 使用用户提到的参数: -5%, 5m, 2%, -20%
        ret = run_sim(a, s, o, cfg, -0.05, 5, 0.02, -0.20, min_p=1.0)
        day_rets.append(ret)
    
    print(f"Results: {[f'{r:+.1%}' for r in day_rets]}")
    print(f"Average: {np.mean(day_rets):+.1%}")

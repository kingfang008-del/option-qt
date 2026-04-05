import sys, os
sys.path.insert(0, os.path.join(os.getcwd(), '../../baseline'))
import plan_a_fine_tune as paf
from strategy_config import StrategyConfig
import pandas as pd
import json

def parse_price(json_str, typ):
    if not json_str: return 0.0
    try:
        d = json.loads(json_str)
        b = d.get('buckets', [])
        idx = 2 if typ == 'call' else 0
        if len(b) > idx and len(b[idx]) >= 8: return float(b[idx][0])
    except: pass
    return 0.0

def parse_bid(json_str, typ):
    if not json_str: return 0.0
    try:
        d = json.loads(json_str)
        b = d.get('buckets', [])
        idx = 2 if typ == 'call' else 0
        if len(b) > idx and len(b[idx]) >= 9: return float(b[idx][8])
    except: pass
    return 0.0

def parse_ask(json_str, typ):
    if not json_str: return 0.0
    try:
        d = json.loads(json_str)
        b = d.get('buckets', [])
        idx = 2 if typ == 'call' else 0
        if len(b) > idx and len(b[idx]) >= 10: return float(b[idx][9])
    except: pass
    return 0.0

def run_strict_4_sim(df_a, df_s, df_o, cfg: StrategyConfig):
    df = pd.merge(df_a, df_s, on=['ts','symbol'], how='inner')
    df = pd.merge(df, df_o, on=['ts','symbol'], how='inner')
    dbt = {ts: g.set_index('symbol').to_dict('index') for ts, g in df.groupby('ts')}
    ts_list = sorted(dbt.keys())
    
    pos_list = []
    trades = []
    cash = cfg.INITIAL_ACCOUNT
    es_roi = -0.05
    mom_mins = 10
    mom_max = 0.02
    abs_sl = -0.20
    
    final_row_map = dbt[ts_list[-1]]

    for ts in ts_list:
        cd = dbt[ts]
        new_pos = []
        for p in pos_list:
            row = cd.get(p['sym'])
            if not row:
                new_pos.append(p)
                continue
            bj = row['buckets_json']
            p_fair = parse_price(bj, p['typ'])
            if p_fair <= 0:
                new_pos.append(p)
                continue
            p_bid = parse_bid(bj, p['typ'])
            p_exit = p_bid * (1 - cfg.SLIPPAGE_PCT) if p_bid > 0 else p_fair * (1 - cfg.SLIPPAGE_PCT)
            p_roi = (p_fair - p['ep']) / p['ep'] if p['ep'] > 0 else 0
            p['hm'] += 1
            p['mx'] = max(p['mx'], p_roi)
            if p_roi < abs_sl:
                cash += p['qty'] * p_exit * 100
                trades.append({**p, 'exit': ts, 'er': p_roi, 'reason': 'ABS_SL'})
                continue
            if p['hm'] <= 5 and p_roi < es_roi:
                cash += p['qty'] * p_exit * 100
                trades.append({**p, 'exit': ts, 'er': p_roi, 'reason': 'EARLY_STOP'})
                continue
            if p['hm'] >= mom_mins and p['mx'] < mom_max:
                cash += p['qty'] * p_exit * 100
                trades.append({**p, 'exit': ts, 'er': p_roi, 'reason': 'NO_MOM'})
                continue
            new_pos.append(p)
        pos_list = new_pos
        
        curr_time = pd.to_datetime(ts, unit='s').tz_localize('UTC').tz_convert('America/New_York')
        if curr_time.strftime("%H:%M:%S") >= cfg.START_TIME and curr_time.strftime("%H:%M:%S") < cfg.NO_ENTRY_TIME:
            for sym, row in sorted(cd.items()):
                if len(pos_list) >= 4: break
                if any(p['sym'] == sym for p in pos_list): continue
                alpha = row['alpha']
                if abs(alpha) >= 1.45:
                    typ = 'call' if alpha > 0 else 'put'
                    bj = row['buckets_json']
                    p_entry_fair = parse_price(bj, typ)
                    p_ask_entry = parse_ask(bj, typ)
                    if p_entry_fair <= 0: continue
                    p_exec = p_ask_entry * (1 + cfg.SLIPPAGE_PCT) if p_ask_entry > 0 else p_entry_fair * (1 + cfg.SLIPPAGE_PCT)
                    qty = int((cash * cfg.POSITION_RATIO) / (p_exec * 100))
                    if qty > 0:
                        cash -= qty * p_exec * 100
                        pos_list.append({
                            'sym': sym, 'typ': typ, 'ep': p_entry_fair, 'qty': qty,
                            'hm': 0, 'mx': -1.0, 'entry': ts, 'entry_alpha': alpha
                        })

    # Liquidate at EOD (Actual price, not max_roi)
    for p in pos_list:
        row = final_row_map.get(p['sym'])
        if row:
            p_fair = parse_price(row['buckets_json'], p['typ'])
            p_roi = (p_fair - p['ep']) / p['ep'] if p['ep'] > 0 else 0
            cash += p['qty'] * p_fair * 100
            trades.append({**p, 'exit': ts_list[-1], 'er': p_roi, 'reason': 'EOD'})
        else:
            cash += p['qty'] * p['ep'] * 100 # Fallback
            trades.append({**p, 'exit': ts_list[-1], 'er': 0.0, 'reason': 'EOD_NO_DATA'})

    return (cash / cfg.INITIAL_ACCOUNT - 1), trades

def run_experiment():
    cfg = StrategyConfig()
    cfg.MAX_POSITIONS = 4
    cfg.POSITION_RATIO = 0.25
    cfg.INITIAL_ACCOUNT = 50000.0
    cfg.SLIPPAGE_PCT = 0.001
    dates = ["20260106"]
    for d in dates:
        db_path = f"history_sqlite_1m/market_{d}.db"
        a, s, o = paf.load_day(db_path)
        roi, trades = run_strict_4_sim(a, s, o, cfg)
        print(f"{d:<10} | {roi:>15.2%} | {len(trades):<6}")
        for t in sorted(trades, key=lambda x: x['entry']):
            print(f"  TR: {t['sym']:<5} {t['typ']:<4} In:{t['entry']} Alp:{t['entry_alpha']:>6.2f} ROI:{t['er']:>7.2%} ({t['reason']})")

if __name__ == "__main__":
    run_experiment()

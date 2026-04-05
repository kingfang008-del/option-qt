import os, sqlite3, pandas as pd, numpy as np, json, pytz
from datetime import datetime

class SimConfig:
    BET_SIZE_PCT = 0.25
    MAX_POSITIONS = 4
    SLIPPAGE_PCT = 0.001
    START_TIME = '09:30:00'
    NO_ENTRY_TIME = '15:30:00'
    ALPHA_STRICT = 1.45
    
    HARD_STOP = -0.20
    TIME_STOP_MINS = 300
    LADDER_TIGHT = [(0.12, 0.10), (0.25, 0.20), (0.50, 0.40)]
    LADDER_WIDE = [(0.30, 0.25), (0.50, 0.40), (0.80, 0.65)]
    HIGH_ALPHA_WIDE_THRESHOLD = 2.5

def parse_price(json_str, action):
    if not json_str: return 0.0
    try:
        d = json.loads(json_str)
        b = d.get('buckets', [])
        idx = 2 if action == 'call' else 0
        if len(b) > idx and len(b[idx]) >= 8: return float(b[idx][0])
    except: pass
    return 0.0

def run_sim_one_day(date, cfg):
    db_path = f"history_sqlite_1m/market_{date}.db"
    if not os.path.exists(db_path): return 0.0
    conn = sqlite3.connect(db_path)
    df_a = pd.read_sql("SELECT ts, symbol, alpha FROM alpha_logs", conn)
    df_s = pd.read_sql("SELECT ts, symbol, close FROM market_bars_1m", conn)
    df_o = pd.read_sql("SELECT ts, symbol, buckets_json FROM option_snapshots_1m", conn)
    conn.close()
    
    # 【核心回滚逻辑】方向性稳定性：统计过去 5 分钟内数值达标且方向一致的次数 (1-5)
    df_a = df_a.sort_values(['symbol','ts'])
    df_a['is_p'] = (df_a['alpha'] >= cfg.ALPHA_STRICT).astype(int)
    df_a['is_n'] = (df_a['alpha'] <= -cfg.ALPHA_STRICT).astype(int)
    def rsum(x): return x.rolling(5, min_periods=1).sum()
    df_a['ps5'] = df_a.groupby('symbol')['is_p'].transform(rsum)
    df_a['ns5'] = df_a.groupby('symbol')['is_n'].transform(rsum)
    df_a['stability'] = np.where(df_a['alpha'] > 0, df_a['ps5'], df_a['ns5'])
    
    df = pd.merge(df_a, df_s, on=['ts','symbol'], how='inner')
    df = pd.merge(df, df_o, on=['ts','symbol'], how='left')
    df = df.sort_values('ts')
    
    all_ts = sorted(df['ts'].unique())
    ny_tz = pytz.timezone('America/New_York')
    dbt = {ts: g.set_index('symbol').to_dict('index') for ts, g in df.groupby('ts')}
    
    total_roi, pos_list, lpm = 0.0, [], {}
    
    for ts in all_ts:
        cd = dbt[ts]
        for s, r in cd.items():
            for act in ['call','put']:
                p = parse_price(r['buckets_json'], act)
                if p > 0.01: lpm[(s,act)] = p
        
        # 1. 出场业务
        new_pos_list = []
        curr_time = datetime.fromtimestamp(ts, tz=pytz.utc).astimezone(ny_tz).strftime('%H:%M:%S')
        for pos in pos_list:
            p = lpm.get((pos['sym'], pos['typ']), 0.0)
            if p <= 0.01: new_pos_list.append(pos); continue
            roi = (p - pos['ep']) / pos['ep']
            pos['mx'] = max(pos['mx'], roi)
            hm = (ts - pos['ets']) / 60.0
            reason = None
            if roi <= cfg.HARD_STOP: reason = "HARD"
            elif hm >= cfg.TIME_STOP_MINS: reason = "TIME"
            else:
                ladder = cfg.LADDER_WIDE if abs(pos['alp']) >= cfg.HIGH_ALPHA_WIDE_THRESHOLD else cfg.LADDER_TIGHT
                for tr, fl in reversed(ladder):
                    if pos['mx'] >= tr and roi < fl: reason = "LADD"; break
            if reason:
                total_roi += (roi - cfg.SLIPPAGE_PCT) * cfg.BET_SIZE_PCT
            else:
                new_pos_list.append(pos)
        pos_list = new_pos_list
        
        # 2. 入场业务
        if cfg.START_TIME <= curr_time < cfg.NO_ENTRY_TIME:
            potential = []
            for s, r in cd.items():
                if any(p['sym'] == s for p in pos_list): continue
                if abs(r['alpha']) >= cfg.ALPHA_STRICT:
                    # Score = abs(Alpha) * Stability_Count
                    score = abs(r['alpha']) * r['stability']
                    potential.append((s, r, score))
            potential.sort(key=lambda x: x[2], reverse=True)
            for s, r, score in potential[:4]:
                if len(pos_list) >= cfg.MAX_POSITIONS: break
                typ = 'call' if r['alpha'] > 0 else 'put'
                p = parse_price(r['buckets_json'], typ)
                if p > 0.1:
                    pos_list.append({'sym':s,'typ':typ,'ep':p,'ets':ts,'alp':r['alpha'],'mx':0.0})
    
    # 强制收盘 (EOD)
    for pos in pos_list:
        p = lpm.get((pos['sym'], pos['typ']), 0.0)
        if p > 0.01:
            roi = (p - pos['ep']) / pos['ep']
            total_roi += (roi - cfg.SLIPPAGE_PCT) * cfg.BET_SIZE_PCT
            
    return total_roi

def main():

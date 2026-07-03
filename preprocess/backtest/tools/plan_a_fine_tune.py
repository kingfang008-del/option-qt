#!/usr/bin/env python3
"""
plan_a_fine_tune.py - Plan A 止损参数精细网格搜索 (V3 - 动态多日版)
功能: 自动扫描当前目录下所有 market_*.db 文件并进行网格搜索分析
"""
import sys, os, glob
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../baseline'))

import sqlite3, pandas as pd, json, numpy as np, pytz
from datetime import datetime, timedelta
from strategy_config import StrategyConfig
from itertools import product

BT_DIR = os.path.dirname(__file__)

def get_db_files():
    """动态获取 history_sqlite_1m 目录下所有的 market_*.db 文件"""
    hist_dir = os.path.join(BT_DIR, "history_sqlite_1m")
    if os.path.exists(hist_dir):
        files = glob.glob(os.path.join(hist_dir, "market_*.db"))
    else:
        files = glob.glob(os.path.join(BT_DIR, "market_*.db"))
    files.sort()
    return files

def get_ny_time(ts):
    return (datetime.fromtimestamp(ts) - timedelta(hours=13)).strftime('%H:%M')

def parse_price(json_str, action):
    if not json_str: return 0.0
    try:
        d = json.loads(json_str)
        b = d.get('buckets', [])
        idx = 2 if action == 'call' else 0
        if len(b) > idx and len(b[idx]) >= 8: return float(b[idx][0])
    except: pass
    return 0.0

def parse_iv(json_str, action):
    if not json_str: return 0.4 # Default IV fallback
    try:
        d = json.loads(json_str)
        b = d.get('buckets', [])
        idx = 2 if action == 'call' else 0
        if len(b) > idx and len(b[idx]) >= 8: return float(b[idx][7])
    except: pass
    return 0.4

def load_day(db_path, use_event=False):
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    
    # 🕵️ 动态检测 event_prob 列是否存在
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(alpha_logs)")
    columns = [row[1] for row in cursor.fetchall()]
    
    a_cols = ["ts", "symbol", "alpha", "vol_z"]
    if use_event:
        if "event_prob" in columns:
            a_cols.append("event_prob")
        else:
            print(f"⚠️ Warning: --use-event is True but 'event_prob' missing in {os.path.basename(db_path)}. Defaulting to 1.0")
            
    a_query = f"SELECT {', '.join(a_cols)} FROM alpha_logs"
    a = pd.read_sql(a_query, conn)
    
    # 如果开启了 event 但列不存在，补齐为 1.0 (全放行)
    if use_event and "event_prob" not in a.columns:
        a['event_prob'] = 1.0
        
    s = pd.read_sql("SELECT ts, symbol, close FROM market_bars_1m", conn)
    o = pd.read_sql("SELECT ts, symbol, buckets_json FROM option_snapshots_1m", conn)
    conn.close()
    return a, s, o

def run_sim(df_a, df_s, df_o, cfg: StrategyConfig, es_roi, mom_mins, mom_max, abs_sl, alpha_min=None, use_vol_z=False, use_event=False, event_thresh=0.7):
    if alpha_min is None: alpha_min = cfg.ALPHA_ENTRY_STRICT
    
    # 🧠 [Event Filter] 如果开启了事件过滤，则在入场前先过滤 Alpha
    if use_event and 'event_prob' in df_a.columns:
        df_a = df_a[df_a['event_prob'] >= event_thresh].copy()
    # 🚨 [关键修复] Plan A 以前先过滤 alpha < 1.45 再 Merge，导致持仓期间如果 alpha 弱了，就不看价格了！
    # 这会导致错过最高点 (max_roi) 和止损触发。现在改为 Left Merge 或先全量 Merge 再过滤 Entry。
    df_s['roc_5m'] = df_s.groupby('symbol')['close'].pct_change(5).fillna(0.0)
    df_s['roc_1m'] = df_s.groupby('symbol')['close'].pct_change(1).fillna(0.0)
        
    # 🚀 [Directional Stability Optimization]
    # 计算过去 5 分钟内，方向一致且 Alpha 达标的次数
    df_a = df_a.sort_values(['symbol','ts'])
    df_a['is_pos_sig'] = ((df_a['alpha'] >= 1.45)).astype(int)
    df_a['is_neg_sig'] = ((df_a['alpha'] <= -1.45)).astype(int)
    
    # 分别计算正向和负向的累积稳定性
    df_a['pos_stab'] = df_a.groupby('symbol')['is_pos_sig'].rolling(5).sum().reset_index(0, drop=True)
    df_a['neg_stab'] = df_a.groupby('symbol')['is_neg_sig'].rolling(5).sum().reset_index(0, drop=True)
    
    # 根据当前 Alpha 方向选取对应的稳定性分
    df_a['stability_score'] = np.where(df_a['alpha'] > 0, df_a['pos_stab'], df_a['neg_stab'])
    df_a['stability_score'] = df_a['stability_score'].fillna(1.0).replace(0, 1.0) # 默认为 1
    
    df = pd.merge(df_a, df_s, on=['ts','symbol'], how='inner')
    df = pd.merge(df, df_o, on=['ts','symbol'], how='inner')
    df = df.sort_values(['symbol','ts'])
    
    # 标注哪些行满足入场条件
    df['is_entry_signal'] = df['alpha'].abs() >= alpha_min
    if use_vol_z:
        df['is_entry_signal'] &= (df['vol_z'] >= cfg.VOL_MIN_Z) & (df['vol_z'] <= cfg.VOL_MAX_Z)
    
    dbt = {ts: g.set_index('symbol').to_dict('index') for ts, g in df.groupby('ts')}
    all_ts = sorted(dbt.keys())
    
    # [Debug]
    debug_sym = 'DELL'
    print(f"DEBUG: Simulating {len(all_ts)} points. alpha_min={alpha_min}")
    
    cash = cfg.INITIAL_ACCOUNT
    pos_list, trades, lpm = [], [], {}

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
            if hm <= cfg.EARLY_STOP_MINS and roi < es_roi: reason = "EARLY"
            if not reason and hm > mom_mins and pos['mx'] < mom_max: reason = "NO_MOM"
            if not reason and roi < abs_sl: reason = "HARD"
            if not reason and hm > cfg.TIME_STOP_MINS and roi < 0.0: reason = "TIME"
            
            if not reason:
                ladder = cfg.LADDER_WIDE if abs(pos['alp']) >= cfg.HIGH_ALPHA_WIDE_THRESHOLD else cfg.LADDER_TIGHT
                for tr, fl in reversed(ladder):
                    if pos['mx'] >= tr:
                        if pos['sym'] in ['CRM', 'XOM']:
                            print(f"🔬 [PlanA-LADD] {pos['sym']} | hm={hm:.1f} | roi={roi:.2%} | mx={pos['mx']:.2%} | TR={tr:.2%} | FL={fl:.2%}")
                        if roi < fl: reason = "LADD"; break
            
            if reason:
                exit_ep = p * (1 - cfg.SLIPPAGE_PCT)
                cash += (exit_ep * pos['qty'] * 100) - (pos['qty'] * cfg.COMMISSION_PER_CONTRACT)
                trades.append({
                    'sym': pos['sym'], 'typ': pos['typ'], 
                    'ets': pos['ets'], 'ots': ts, 
                    'ep': pos['ep'], 'xp': exit_ep,
                    'roi': roi, 'reason': reason
                })
            else: active.append(pos)
        
        pos_list = active
        
        ny_tz = pytz.timezone('America/New_York')
        dt_ny = datetime.fromtimestamp(ts, tz=pytz.utc).astimezone(ny_tz)
        curr_time_str = dt_ny.strftime('%H:%M:%S')
        # 2. 开仓信号评估
        if curr_time_str >= cfg.START_TIME and curr_time_str < cfg.NO_ENTRY_TIME:
            # 🚀 [关键对齐] 将当前分钟的所有潜在信号按 Alpha 绝对值降序排列，取前 4 个入场
            potential_entries = []
            whitelist = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'SPY', 'QQQ']
            for sym, row in cd.items():
                if sym not in whitelist: continue # 🚀 只做 Mag 7 + SPY/QQQ
                if any(p['sym'] == sym for p in pos_list): continue
                if row['is_entry_signal']:
                    alpha = row['alpha']
                    
                    typ = 'call' if alpha > 0 else 'put'
                    iv = parse_iv(row['buckets_json'], typ)
                    roc_1 = row['roc_1m']
                    
                    # 🚀 [Final Combined Optimization]
                    # 每一个 Sym 只有一次计算得分并入队的机会，杜绝重复开仓 Bug
                    sync_bonus = np.sign(alpha) * np.tanh(roc_1 * 500)
                    stab = row['stability_score']
                    
                    # 综合分数 = 绝对 Alpha * 稳定性 * 方向一致性
                    w_score = abs(alpha) * stab * (1.0 + sync_bonus)
                    potential_entries.append((sym, row, w_score))
            
            # 按稳定性加权分由大到小排序
            potential_entries.sort(key=lambda x: x[2], reverse=True)
            
            # 简单选取前 4 个即可（不再需要强制 Balance，稳定性会自动选出高质量信号）
            for sym, row, w_score in potential_entries[:4]:
                if len(pos_list) >= cfg.MAX_POSITIONS: 
                    break 
                
                alpha = row['alpha']
                typ = 'call' if alpha > 0 else 'put'
                p_entry = parse_price(row['buckets_json'], typ)
                if p_entry <= 0: continue
                
                qty = int((cash * cfg.POSITION_RATIO) / (p_entry * 100))
                if qty > 0:
                    cash -= qty * p_entry * 100
                    pos_list.append({
                        'sym': sym, 'typ': typ, 'ep': p_entry, 'qty': qty,
                        'ets': ts, 'alp': row['alpha'], 'mx': 0.0
                    })
                    print(f"ENTRY! {curr_time_str} {sym} {typ} @ {p_entry:.2f} (alpha={row['alpha']:.2f})")
    
    for pos in pos_list:
        p = lpm.get((pos['sym'], pos['typ']), pos['ep'])
        exit_ep = p # EOD exit usually at mark
        roi = (p - pos['ep']) / pos['ep']
        trades.append({
            'sym': pos['sym'], 'typ': pos['typ'], 
            'ets': pos['ets'], 'ots': all_ts[-1], 
            'ep': pos['ep'], 'xp': exit_ep,
            'roi': roi, 'reason': 'EOD'
        })
    
    eq = cash + sum([p['qty'] * lpm.get((p['sym'], p['typ']), p['ep']) * 100 for p in pos_list])
    return (eq / cfg.INITIAL_ACCOUNT - 1), trades

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, default=None, help="Specific date (YYYYMMDD) to test")
    parser.add_argument("--use-event", action="store_true", help="Enable event_prob filtering")
    parser.add_argument("--event-thresh", type=float, default=0.7, help="Event probability threshold")
    args_cli = parser.parse_args()

    cfg = StrategyConfig()
    print(f"DEBUG: GLOBAL CFG INIT: ALPHA_STRICT={cfg.ALPHA_ENTRY_STRICT}")
    db_files = get_db_files()
    if args_cli.date:
        db_files = [f for f in db_files if args_cli.date in f]
        
    if not db_files:
        print(f"Error: No market_*.db files found matching {args_cli.date if args_cli.date else 'any date'}")
        sys.exit(1)
        
    print(f"Loading data for {len(db_files)} days: {[os.path.basename(f) for f in db_files]}")
    days = {os.path.basename(db)[-11:-3]: load_day(db, use_event=args_cli.use_event) for db in db_files}
    day_names = sorted(days.keys())
    
    # Grid Search Params
    # 为了演示动态效果并减少运行时间，我们保留了核心的 Plan A 调优维度
    early_stops = [-0.05] if args_cli.date else [-0.03, -0.05, -0.08]
    mom_mins = [5] if args_cli.date else [3, 5, 8]
    mom_maxs = [0.02] if args_cli.date else [0.01, 0.02, 0.03]
    abs_sls = [-0.20] if args_cli.date else [-0.15, -0.20]
    
    results = []
    total = len(early_stops) * len(mom_mins) * len(mom_maxs) * len(abs_sls)
    print(f"Running {total} combinations across {len(day_names)} days...")
    
    for i, (es, mm, mx, asl) in enumerate(product(early_stops, mom_mins, mom_maxs, abs_sls)):
        day_rets, day_trades = [], []
        for d in day_names:
            a, s, o = days[d]
            ret, tlist = run_sim(a, s, o, cfg, es, mm, mx, asl, use_vol_z=False, use_event=args_cli.use_event, event_thresh=args_cli.event_thresh)
            day_rets.append(ret)
            day_trades.append(tlist)
        
        avg = np.mean(day_rets)
        std = np.std(day_rets) if len(day_rets) > 1 else 0.0
        sharpe = avg / std if std > 0 else (10.0 if avg > 0 else -10.0)
        
        res = {
            'ES': es, 'MomM': mm, 'MomX': mx, 'SL': asl,
            'avg': avg, 'sharpe': sharpe, 'min': min(day_rets),
            'trades': day_trades
        }
        for d, r in zip(day_names, day_rets):
            res[f"d_{d}"] = r
        results.append(res)
    
    df = pd.DataFrame(results)
    
    # 按照用户要求识别 最高收益 和 最稳定 配置
    best_avg = df.loc[df['avg'].idxmax()]
    best_sharpe = df.loc[df['sharpe'].idxmax()]
    
    def get_shorthand(full_date):
        # 20260102 -> 1/2
        try:
            m = int(full_date[4:6])
            d = int(full_date[6:8])
            return f"{m}/{d}"
        except: return full_date

    print("\n" + "=" * 120)
    print(f"📊 Plan A Optimization Report ({len(day_names)} Days)")
    print("=" * 120)
    
    # 表头
    header = f"{'配置':<10} {'极端止损':<12} {'无动力窗口':<16} {'硬止损':<10} |"
    for d in day_names:
        header += f" {get_shorthand(d):>8}"
    header += f" {'平均':>8} {'Sharpe':>8}"
    print(header)
    print("-" * len(header))
    
    def print_row(label, row):
        es_str = f"{row['ES']:+.0%} @5m"
        mom_str = f"{row['MomM']:.0f}m, {row['MomX']:.0%}阈值"
        sl_str = f"{row['SL']:+.0%}"
        
        line = f"{label:<10} {es_str:<12} {mom_str:<16} {sl_str:<10} |"
        for d in day_names:
            line += f" {row[f'd_{d}']:>8.1%}"
        line += f" {row['avg']:>8.1%} {row['sharpe']:>8.1f}"
        print(line)

    print_row("最高收益", best_avg)
    print_row("最稳定", best_sharpe)
    print("=" * len(header))

    # --- [新增] 每日盈亏与资金曲线汇总 ---
    print("\n📈 DAILY PERFORMANCE & EQUITY CURVE (For: 最稳定 配置)")
    print("-" * 60)
    print(f"{'Date':<10} {'Return %':>10} {'Cumulative Equity':>20}")
    print("-" * 60)
    
    current_equity = cfg.INITIAL_ACCOUNT
    daily_summary = []
    for d_name in day_names:
        daily_ret = best_sharpe[f"d_{d_name}"]
        current_equity *= (1 + daily_ret)
        daily_summary.append({
            'date': get_shorthand(d_name),
            'ret': daily_ret,
            'equity': current_equity
        })
        print(f"{get_shorthand(d_name):<10} {daily_ret:>9.1%}  ${current_equity:>18,.2f}")
    
    print("-" * 60)
    total_net = current_equity - cfg.INITIAL_ACCOUNT
    total_ret = (current_equity / cfg.INITIAL_ACCOUNT) - 1
    print(f"Final Equity: ${current_equity:,.2f} | Total Net: ${total_net:+,.2f} ({total_ret:+.1%})")

    # --- [新增] 详细交易流水展示 ---
    print("\n🔍 DETAILED TRADE EXPLORER (For: 最稳定 配置)")
    print("-" * 120)
    print(f"{'Date':<8} {'Sym':<8} {'Type':<5} {'In Time':<8} {'Out Time':<8} | {'Entry':>7} {'Exit':>7} | {'ROI':>8} {'Reason'}")
    print("-" * 120)
    
    for d_idx, d_name in enumerate(day_names):
        d_trades = best_sharpe['trades'][d_idx]
        for t in d_trades:
            in_t = get_ny_time(t['ets'])
            out_t = get_ny_time(t['ots'])
            line = f"{get_shorthand(d_name):<8} {t['sym']:<8} {t['typ']:<5} {in_t:<8} {out_t:<8} | {t['ep']:>7.2f} {t['xp']:>7.2f} | {t['roi']:>8.1%} {t['reason']}"
            print(line)
    print("-" * 120)

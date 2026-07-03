import sqlite3
import pandas as pd
import json
import numpy as np
import argparse
import os
from tqdm import tqdm

def parse_option_by_strike(json_exit, target_strike, side='call'):
    """从快照中精准定位同执行价期权价格"""
    if not json_exit: return None
    try:
        d = json.loads(json_exit)
        buckets = d.get('buckets', [])
        for b in buckets:
            if len(b) > 9:
                strike = float(b[5])
                if abs(strike - target_strike) < 0.01:
                    return {
                        'bid': float(b[8]),
                        'ask': float(b[9])
                    }
    except: pass
    return None

def get_atm_option(json_str, side='call'):
    """提取 ATM 期权及其执行价"""
    if not json_str: return None
    try:
        d = json.loads(json_str)
        buckets = d.get('buckets', [])
        idx = 2 if side == 'call' else 0
        if len(buckets) > idx:
            b = buckets[idx]
            return {
                'price': float(b[0]),
                'delta': float(b[1]),
                'strike': float(b[5]),
                'bid': float(b[8]),
                'ask': float(b[9])
            }
    except: pass
    return None

def analyze_date(db_path, hold_mins=15):
    print(f"\n🚀 [Lab v3] 攻坚: {os.path.basename(db_path)} (Mode: {hold_mins}m Hold)")
    conn = sqlite3.connect(db_path)
    
    # 1. 极速全量加载
    df_a = pd.read_sql("SELECT ts, symbol, alpha, vol_z FROM alpha_logs", conn)
    df_s = pd.read_sql("SELECT ts, symbol, close, spy_roc_5min FROM market_bars_1m", conn)
    df_o = pd.read_sql("SELECT ts, symbol, buckets_json FROM option_snapshots_1m", conn)
    conn.close()

    # 2. 预研: 效率比 (Smoothness) 与 5min 动量 (Speed)
    df_s = df_s.sort_values(['symbol', 'ts'])
    
    def calc_er(x):
        if len(x) < 30: return 0.5
        net = abs(x.iloc[-1] - x.iloc[0])
        gross = np.sum(np.abs(np.diff(x)))
        return net / (gross + 1e-6)

    # 计算 30min 效率比 (Kaufman ER)
    df_s['er_30m'] = df_s.groupby('symbol')['close'].transform(lambda x: x.rolling(30).apply(calc_er))
    # 提取当前 5-min ROC
    df_s['roc_5m'] = df_s.groupby('symbol')['close'].pct_change(5)
    
    # 3. 数据全域对撞
    df = pd.merge(df_a, df_s[['ts', 'symbol', 'close', 'er_30m', 'roc_5m', 'spy_roc_5min']], on=['ts', 'symbol'])
    df = pd.merge(df, df_o, on=['ts', 'symbol'])
    
    # 构建出场查询映射
    opt_map = {(row.symbol, row.ts): row.buckets_json for row in df_o.itertuples()}
    
    results = []
    sample_ts = sorted(df['ts'].unique()) # 全量分钟级审计，不再跳步
    
    for ts in tqdm(sample_ts, desc="Simulating Trades"):
        snap = df[df['ts'] == ts].copy()
        if snap.empty: continue
        
        candidates = []
        for _, row in snap.iterrows():
            side = 'call' if row['alpha'] > 0 else 'put'
            # 1. 入场合约锁定
            opt_entry = get_atm_option(row['buckets_json'], side)
            if not opt_entry or opt_entry['price'] <= 0.05: continue
            
            # 2. 15分钟后同执行价出场
            ts_exit = ts + (hold_mins * 60)
            json_exit = opt_map.get((row['symbol'], ts_exit))
            
            real_net_pnl = -1.0 # 找不到视为爆仓/下架
            if json_exit:
                opt_exit = parse_option_by_strike(json_exit, opt_entry['strike'], side)
                if opt_exit:
                    real_net_pnl = (opt_exit['bid'] - opt_entry['ask']) / opt_entry['ask']
            
            if real_net_pnl < -0.99: continue

            # --- Logic C: Profit-Lock (Baseline) ---
            spread = opt_entry['ask'] - opt_entry['bid']
            score_c = (abs(row['alpha']) * abs(opt_entry['delta'])) / max(0.01, spread)
            
            # --- 🔥 [Optimizer] Logic E: Resonance Filter ---
            # 惩罚项 1: 相对点差 (Spread %) - 淘汰低价高损耗合约
            spread_pct = spread / opt_entry['ask'] if opt_entry['ask'] > 0 else 1.0
            
            # 共振项 2: (动量速度 * 平滑度平方)
            er = row['er_30m'] if not np.isnan(row['er_30m']) else 0.5
            resonance = abs(row['roc_5m']) * (er ** 2)
            
            # 综合评分: (期权效能 / 点差损耗) * 趋势共振
            score_e = (abs(row['alpha']) * abs(opt_entry['delta']) / (spread_pct + 1e-4)) * (resonance * 1000)

            candidates.append({
                'symbol': row['symbol'],
                'side': side,
                'er': er,
                'roc': row['roc_5m'],
                'score_c': score_c,
                'score_e': score_e,
                'net_pnl': real_net_pnl
            })
            
        if not candidates: continue
        df_c = pd.DataFrame(candidates)
        
        # 记录不同逻辑下的最优选择 (Top 1 Pick)
        best_c = df_c.sort_values('score_c', ascending=False).iloc[0]
        best_e = df_c.sort_values('score_e', ascending=False).iloc[0]
        
        results.append({'logic': 'C:ProfitLock', 'net_pnl': best_c['net_pnl'], 'score': best_c['score_c']})
        results.append({'logic': 'E:Resonance', 'net_pnl': best_e['net_pnl'], 'score': best_e['score_e']})

    res_df = pd.DataFrame(results)
    if res_df.empty: return

    # 4. 回报展示
    summary = res_df.groupby('logic')['net_pnl'].agg(['mean', 'sum', 'count']).reset_index()
    summary['win_rate'] = res_df.groupby('logic').apply(lambda x: (x['net_pnl'] > 0).mean()).values
    
    print("\n" + "="*70)
    print("🏆 ALPHA OPTIMIZATION TOURNAMENT v3 (Logic E vs Logic C)")
    print("="*70)
    print(summary.to_string(index=False))
    print("-" * 70)
    
    # 5. [门槛扫描] 寻找转正甜点区
    for log_name in ['C:ProfitLock', 'E:Resonance']:
        print(f"\n🧐 Examining {log_name} Thresholds...")
        data_sub = res_df[res_df['logic'] == log_name]
        for q in [0.0, 0.5, 0.7, 0.8, 0.9]:
            thresh = data_sub['score'].quantile(q)
            subset = data_sub[data_sub['score'] >= thresh]
            if not subset.empty:
                print(f"Top {100-q*100:>2.0f}% Signal (Score >= {thresh:>8.2f}) | PnL: {subset['net_pnl'].mean()*100:>6.2f}% | WR: { (subset['net_pnl']>0).mean()*100:>5.1f}% | Count: {len(subset)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, required=True)
    parser.add_argument("--hold", type=int, default=15)
    args = parser.parse_args()
    
    db_path = f"/Users/fangshuai/Documents/GitHub/option-qt/production/preprocess/backtest/history_sqlite_1m/market_{args.date}.db"
    analyze_date(db_path, args.hold)

import sqlite3
import pandas as pd
import json
import numpy as np
import argparse
import os

def parse_option_stats(json_str, side='call'):
    """从 buckets_json 中提取价格、Delta 和价差"""
    if not json_str: return None
    try:
        d = json.loads(json_str)
        buckets = d.get('buckets', [])
        # 索引约定: 2=Call ATM, 0=Put ATM (根据 plan_a_fine_tune.py)
        idx = 2 if side == 'call' else 0
        if len(buckets) > idx:
            b = buckets[idx]
            # [Price, Delta, Gamma, Theta, Vega, OI, Vol, IV, Bid, Ask, ...]
            return {
                'price': float(b[0]),
                'delta': float(b[1]),
                'bid': float(b[8]),
                'ask': float(b[9]),
                'spread': float(b[9]) - float(b[8])
            }
    except: pass
    return None

def analyze_date(db_path):
    print(f"\n🔍 Analyzing: {os.path.basename(db_path)}")
    conn = sqlite3.connect(db_path)
    
    # 1. 加载基础数据
    df_a = pd.read_sql("SELECT ts, symbol, alpha, vol_z FROM alpha_logs", conn)
    df_s = pd.read_sql("SELECT ts, symbol, close FROM market_bars_1m", conn)
    df_o = pd.read_sql("SELECT ts, symbol, buckets_json FROM option_snapshots_1m", conn)
    conn.close()

    # 2. 计算 15 分钟后的真实涨幅 (Forward Return)
    df_s = df_s.sort_values(['symbol', 'ts'])
    df_s['fwd_ret_15m'] = df_s.groupby('symbol')['close'].shift(-15) / df_s['close'] - 1
    
    # 3. 合并数据
    df = pd.merge(df_a, df_s[['ts', 'symbol', 'close', 'fwd_ret_15m']], on=['ts', 'symbol'])
    df = pd.merge(df, df_o, on=['ts', 'symbol'])
    
    results = []
    # 抽样分析: 每 30 分钟取一次全市场的信号进行“选优”比赛
    sample_ts = sorted(df['ts'].unique())[::30]
    
    for ts in sample_ts:
        snap = df[df['ts'] == ts].copy()
        if snap.empty: continue
        
        candidates = []
        for _, row in snap.iterrows():
            side = 'call' if row['alpha'] > 0 else 'put'
            opt = parse_option_stats(row['buckets_json'], side)
            if not opt or opt['price'] <= 0.05: continue
            
            # --- Logic A: Raw Alpha ---
            score_a = abs(row['alpha'])
            
            # --- Logic B: Vol-Adjusted (Assume 14d ATR proxy via vol_z) ---
            # 简单的归一化方式
            score_b = abs(row['alpha']) / (row['vol_z'] if row['vol_z'] > 0 else 1.0)
            
            # --- Logic C: Profit-Lock Score (User Request) ---
            # Score = (Alpha * Delta) / (Ask - Bid)
            spread = opt['spread'] if opt['spread'] > 0 else 0.01
            score_c = (abs(row['alpha']) * abs(opt['delta'])) / spread
            
            # --- 模拟深层 PnL (扣除价差损耗) ---
            # 15分钟后的预估 Bid 出场价
            # 假设 IV 不变，使用 Delta 估算期权价格变动 (保守估计)
            opt_move = (row['close'] * row['fwd_ret_15m']) * opt['delta']
            est_exit_bid = opt['bid'] + opt_move
            # 实盘 PnL = (出场 Bid - 入场 Ask) / 入场 Ask
            real_net_pnl = (est_exit_bid - opt['ask']) / opt['ask']
            
            candidates.append({
                'symbol': row['symbol'],
                'side': side,
                'alpha': row['alpha'],
                'net_pnl_15m': real_net_pnl,
                'score_a': score_a,
                'score_b': score_b,
                'score_c': score_c,
                'ts': ts
            })
            
        if not candidates: continue
        df_c = pd.DataFrame(candidates)
        
        # 记录每种逻辑选出的 Top 1
        pick_a = df_c.sort_values('score_a', ascending=False).iloc[0]
        pick_c = df_c.sort_values('score_c', ascending=False).iloc[0]
        
        results.append({'ts': ts, 'logic': 'Raw_Alpha', 'sym': pick_a['symbol'], 'net_pnl': pick_a['net_pnl_15m']})
        results.append({'ts': ts, 'logic': 'Profit_Lock', 'sym': pick_c['symbol'], 'net_pnl': pick_c['net_pnl_15m']})

    # 4. 汇总报告
    final_df = pd.DataFrame(results)
    summary = final_df.groupby('logic')['net_pnl'].agg(['mean', 'sum', 'count']).reset_index()
    print("\n" + "="*50)
    print("🏆 PROFIT-LOCK TOURNAMENT REPORT (Net PnL After Slippage)")
    print("="*50)
    print(summary.to_string(index=False))
    print("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, required=True)
    args = parser.parse_args()
    
    db_path = f"/Users/fangshuai/Documents/GitHub/option-qt/production/preprocess/backtest/history_sqlite_1m/market_{args.date}.db"
    analyze_date(db_path)

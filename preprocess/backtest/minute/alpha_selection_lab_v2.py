import sqlite3
import pandas as pd
import json
import numpy as np
import argparse
import os
from tqdm import tqdm

def parse_option_by_strike(json_str, target_strike, side='call'):
    """从 buckets_json 中寻找特定执行价的期权数据"""
    if not json_str: return None
    try:
        d = json.loads(json_str)
        buckets = d.get('buckets', [])
        # 搜索逻辑: 遍历所有 bucket，寻找执行价匹配的
        # [Price, Delta, Gamma, Theta, Vega, Strike, OI, Vol, IV, Bid, Ask, ...]
        for b in buckets:
            if len(b) > 9:
                strike = float(b[5])
                # 检查执行价是否匹配 (浮点数容错)
                if abs(strike - target_strike) < 0.01:
                    return {
                        'price': float(b[0]),
                        'delta': float(b[1]),
                        'strike': strike,
                        'bid': float(b[8]),
                        'ask': float(b[9]),
                        'spread': float(b[9]) - float(b[8])
                    }
    except: pass
    return None

def get_atm_option(json_str, side='call'):
    """初始入场时获取 ATM 期权"""
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
                'ask': float(b[9]),
                'spread': float(b[9]) - float(b[8])
            }
    except: pass
    return None

def analyze_date(db_path):
    print(f"\n🧪 [Lab v2] Analyzing: {os.path.basename(db_path)}")
    conn = sqlite3.connect(db_path)
    
    # 1. 加载数据
    print("📥 Loading logs...")
    df_a = pd.read_sql("SELECT ts, symbol, alpha, vol_z FROM alpha_logs", conn)
    df_s = pd.read_sql("SELECT ts, symbol, close FROM market_bars_1m", conn)
    df_o = pd.read_sql("SELECT ts, symbol, buckets_json FROM option_snapshots_1m", conn)
    conn.close()

    # 2. 计算 30 分钟效率比 (ER) - 判定平滑度
    df_s = df_s.sort_values(['symbol', 'ts'])
    print("📈 Calculating Smoothness Metrics (ER)...")
    
    # 使用 rolling 计算 ER
    def calc_er_rolling(x):
        if len(x) < 30: return 0.5
        net = abs(x.iloc[-1] - x.iloc[0])
        gross = np.sum(np.abs(np.diff(x)))
        return net / (gross + 1e-6)

    df_s['er_30m'] = df_s.groupby('symbol')['close'].transform(lambda x: x.rolling(30).apply(calc_er_rolling))
    
    # 3. 合并基础数据
    df = pd.merge(df_a, df_s[['ts', 'symbol', 'close', 'er_30m']], on=['ts', 'symbol'])
    df = pd.merge(df, df_o, on=['ts', 'symbol'])
    
    # 创建 T+15 查找表，用于真期权价格对撞
    df_target_15m = df_o[['ts', 'symbol', 'buckets_json']].copy()
    df_target_15m['ts_key'] = df_target_15m['ts'] - 900 # 查找 15 分钟前对应的入场点
    
    # 这里我们采用字典映射加速查找
    opt_map = {}
    for row in df_o.itertuples():
        opt_map[(row.symbol, row.ts)] = row.buckets_json

    results = []
    # 采样分析: 每 30 分钟一个切片进行全市场比赛
    sample_ts = sorted(df['ts'].unique())[::30]
    
    print("⚔️ Running Tournament...")
    for ts in tqdm(sample_ts):
        snap = df[df['ts'] == ts].copy()
        if snap.empty: continue
        
        candidates = []
        for _, row in snap.iterrows():
            side = 'call' if row['alpha'] > 0 else 'put'
            # 入场时选择 ATM
            opt_entry = get_atm_option(row['buckets_json'], side)
            if not opt_entry or opt_entry['price'] <= 0.05: continue
            
            # --- 真期权价格对撞 (T+15m) ---
            ts_exit = ts + 900
            json_exit = opt_map.get((row['symbol'], ts_exit))
            
            real_net_pnl = -1.0 # 默认全亏 (如果找不到对应合约)
            if json_exit:
                # 在 15 分钟后的快照中，寻找【同执行价】的同一个合约
                opt_exit = parse_option_by_strike(json_exit, opt_entry['strike'], side)
                if opt_exit:
                    # 真实 PnL = (出场 Bid - 入场 Ask) / 入场 Ask
                    # 考虑到回放延迟和滑点，入场用 Ask，出场用 Bid 是最公平的
                    real_net_pnl = (opt_exit['bid'] - opt_entry['ask']) / opt_entry['ask']
            
            # 如果依然找不到 (例如合约下架/行权价消失), 使用 Alpha 保护
            if real_net_pnl < -0.99: continue 

            # --- Logic C: Profit-Lock (Alpha * Delta / Spread) ---
            spread = opt_entry['spread'] if opt_entry['spread'] > 0 else 0.01
            score_c = (abs(row['alpha']) * abs(opt_entry['delta'])) / spread
            
            # --- Logic D: Smoothness-Aware Profit-Lock (C * ER^2) ---
            er = row['er_30m'] if not np.isnan(row['er_30m']) else 0.5
            # 使用 ER 的平方来增强平滑度的权重，严惩“织布机”行情
            score_d = score_c * (er ** 2)
            
            candidates.append({
                'symbol': row['symbol'],
                'side': side,
                'er': er,
                'alpha': row['alpha'],
                'net_pnl_15m': real_net_pnl,
                'score_a': abs(row['alpha']),
                'score_c': score_c,
                'score_d': score_d,
                'ts': ts
            })
            
        if not candidates: continue
        df_c = pd.DataFrame(candidates)
        
        # 筛选每种逻辑的最佳选择
        best_a = df_c.sort_values('score_a', ascending=False).iloc[0]
        best_c = df_c.sort_values('score_c', ascending=False).iloc[0]
        best_d = df_c.sort_values('score_d', ascending=False).iloc[0]
        
        results.append({'ts': ts, 'logic': 'A:RawAlpha',   'sym': best_a['symbol'], 'net_pnl': best_a['net_pnl_15m'], 'er': best_a['er']})
        results.append({'ts': ts, 'logic': 'C:ProfitLock', 'sym': best_c['symbol'], 'net_pnl': best_c['net_pnl_15m'], 'er': best_c['er']})
        results.append({'ts': ts, 'logic': 'D:Smoothness', 'sym': best_d['symbol'], 'net_pnl': best_d['net_pnl_15m'], 'er': best_d['er']})
    
    if not results:
        print("❌ No valid samples found.")
        return

    # 4. 汇总分析
    final_df = pd.DataFrame(results)
    summary = final_df.groupby('logic').agg({
        'net_pnl': ['mean', 'sum', 'count'],
        'er': 'mean'
    })
    
    # [Fix] 扁平化多级索引
    summary.columns = ['pnl_mean', 'pnl_sum', 'count', 'er_mean']
    summary = summary.reset_index()
    
    # 计算胜率 (PnL > 0)
    win_rates = final_df.groupby('logic').apply(lambda x: (x['net_pnl'] > 0).mean()).reset_index(name='win_rate')
    summary = pd.merge(summary, win_rates, on='logic')

    print("\n" + "="*70)
    print("🏆 ALPHA SELECTION TOURNAMENT v2 (True Option PnL + Smoothness Filter)")
    print("="*70)
    print(summary.to_string())
    print("-" * 70)
    print("💡 Logic D: Selective on ER (Efficiency Ratio). Values close to 1.0 are smoother.")
    print("💡 Logic C: Optimized for Profit Expectancy (Alpha * Delta / Spread).")
    print("="*70)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, required=True)
    args = parser.parse_args()
    
    db_path = f"/Users/fangshuai/Documents/GitHub/option-qt/production/preprocess/backtest/history_sqlite_1m/market_{args.date}.db"
    analyze_date(db_path)

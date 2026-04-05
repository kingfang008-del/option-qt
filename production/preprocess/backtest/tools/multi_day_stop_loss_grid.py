#!/usr/bin/env python3
"""
multi_day_stop_loss_grid.py
3日网格搜索：Plan A (5 分钟观察窗) + Plan B (正股止损) + Ladder
找到让 Winners 跑起来 & Losers 快速认错 的最优配置
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../baseline'))

import sqlite3
import pandas as pd
import json
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List, Tuple
from strategy_config import StrategyConfig
from itertools import product

BT_DIR = os.path.dirname(__file__)
DB_FILES = [
    os.path.join(BT_DIR, "market_20260102.db"),
    os.path.join(BT_DIR, "market_20260105.db"),
    os.path.join(BT_DIR, "market_20260106.db"),
]
INITIAL_ACCOUNT = 50000.0
MAX_POSITIONS = 4
POSITION_RATIO = 1.0 / 4.0
COMMISSION = 0.65
SLIPPAGE = 0.001
MIN_ENTRY_PRICE = 1.0

def get_ny_time(ts):
    dt = datetime.fromtimestamp(ts)
    return (dt - timedelta(hours=13)).strftime('%H:%M')

def parse_price(json_str, action):
    if not json_str: return 0.0
    try:
        data = json.loads(json_str)
        buckets = data.get('buckets', [])
        idx = 2 if action == 'call' else 0
        if len(buckets) > idx and len(buckets[idx]) >= 8:
            return float(buckets[idx][0])
    except: pass
    return 0.0

@dataclass
class StopConfig:
    """快速止损参数"""
    # Plan A: 观察窗
    early_stop_mins: int = 5        # 前 N 分钟极端止损
    early_stop_roi: float = -0.05   # 前 N 分钟极端止损线
    no_momentum_mins: int = 5       # N 分钟后检查是否曾有正向动力
    no_momentum_min_max_roi: float = 0.03  # 如果 max_roi 从未超过此值 → 平仓
    
    # Plan B: 正股联动止损
    stock_stop_enabled: bool = True
    stock_stop_threshold: float = 0.003  # 正股反向 0.3% 就砍
    stock_stop_buffer_mins: int = 3      # 前 N 分钟给缓冲
    
    # 基础止损
    absolute_stop_loss: float = -0.15
    time_stop_mins: int = 240       # 加长到 4 小时
    time_stop_roi: float = 0.0      # 时间止损时 ROI 门槛
    
    tag: str = ""

def load_day_data(db_path):
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    df_alpha = pd.read_sql("SELECT ts, symbol, alpha, event_prob FROM alpha_logs", conn)
    df_stock = pd.read_sql("SELECT ts, symbol, close FROM market_bars_1m", conn)
    df_opt = pd.read_sql("SELECT ts, symbol, buckets_json FROM option_snapshots_1m", conn)
    conn.close()
    return df_alpha, df_stock, df_opt

def run_sim(df_alpha, df_stock, df_opt, cfg: StrategyConfig, stop: StopConfig, alpha_min=1.45):
    df_a = df_alpha[df_alpha['alpha'].abs() >= alpha_min].copy()
    df = pd.merge(df_a, df_stock, on=['ts', 'symbol'], how='inner')
    df = pd.merge(df, df_opt, on=['ts', 'symbol'], how='inner')
    df = df.sort_values(['symbol', 'ts'])
    
    data_by_ts = {}
    for ts, group in df.groupby('ts'):
        data_by_ts[ts] = group.set_index('symbol').to_dict('index')
    all_ts = sorted(data_by_ts.keys())
    
    cash = INITIAL_ACCOUNT
    positions = []
    trade_history = []
    last_price_map = {}
    last_stock_map = {}

    for ts in all_ts:
        curr_data = data_by_ts[ts]
        active = []
        
        # Update price maps
        for sym, row in curr_data.items():
            last_stock_map[sym] = row['close']
            for action in ['call', 'put']:
                p = parse_price(row['buckets_json'], action)
                if p > 0.01: last_price_map[(sym, action)] = p

        # Check exits
        for pos in positions:
            p = last_price_map.get((pos['symbol'], pos['type']), 0.0)
            if p <= 0.01: active.append(pos); continue
            
            roi = (p - pos['entry_price']) / pos['entry_price']
            pos['max_roi'] = max(pos['max_roi'], roi)
            held_mins = (ts - pos['entry_ts']) / 60.0
            
            # Current stock price
            curr_stock = last_stock_map.get(pos['symbol'], 0)
            stock_roi = 0.0
            if pos['entry_stock'] > 0 and curr_stock > 0:
                stock_roi = (curr_stock - pos['entry_stock']) / pos['entry_stock']
                if pos['dir'] == -1: stock_roi = -stock_roi  # Put direction
            
            reason = None
            
            # === Plan A: 观察窗止损 ===
            # A1: 前 N 分钟极端止损
            if held_mins <= stop.early_stop_mins and roi < stop.early_stop_roi:
                reason = f"EARLY_STOP({roi:.1%})"
            
            # A2: N 分钟后检查是否有动力
            if not reason and held_mins > stop.no_momentum_mins:
                if pos['max_roi'] < stop.no_momentum_min_max_roi:
                    reason = f"NO_MOMENTUM(max:{pos['max_roi']:.1%})"
            
            # === Plan B: 正股联动止损 ===
            if not reason and stop.stock_stop_enabled:
                if held_mins > stop.stock_stop_buffer_mins:
                    adverse = (pos['dir'] == 1 and stock_roi < -stop.stock_stop_threshold) or \
                              (pos['dir'] == -1 and stock_roi > stop.stock_stop_threshold)
                    if adverse:
                        reason = f"STOCK_STOP(s:{stock_roi:.2%})"
            
            # === 基础止损 ===
            if not reason and roi < stop.absolute_stop_loss:
                reason = "HARD_STOP"
            
            # === 时间止损 ===
            if not reason and held_mins > stop.time_stop_mins and roi < stop.time_stop_roi:
                reason = "TIME_STOP"
            
            # === 阶梯止盈 ===
            if not reason:
                if cfg.DYNAMIC_LADDER_ENABLED and abs(pos['alpha']) >= cfg.HIGH_ALPHA_WIDE_THRESHOLD:
                    ladder = cfg.LADDER_WIDE
                else:
                    ladder = cfg.LADDER_TIGHT
                for trigger, floor in reversed(ladder):
                    if pos['max_roi'] >= trigger:
                        if roi < floor:
                            reason = f"LADD_{int(trigger*100)}%"
                        break
            
            if reason:
                exit_p = p * (1 - SLIPPAGE)
                cash += (exit_p * pos['qty'] * 100) - (pos['qty'] * COMMISSION)
                trade_history.append({
                    'symbol': pos['symbol'], 'alpha': pos['alpha'],
                    'entry_t': get_ny_time(pos['entry_ts']),
                    'entry_p': pos['entry_price'], 'exit_p': exit_p,
                    'roi': roi, 'max': pos['max_roi'], 'reason': reason
                })
            else:
                active.append(pos)
        positions = active
        
        # Entry logic
        hour_local = datetime.fromtimestamp(ts).hour
        min_local = datetime.fromtimestamp(ts).minute
        is_ready = (hour_local > 22) or (hour_local == 22 and min_local >= cfg.START_MINUTE)
        
        if is_ready and len(positions) < MAX_POSITIONS:
            for sym, row in curr_data.items():
                if any(p['symbol'] == sym for p in positions): continue
                if row['event_prob'] < cfg.EVENT_PROB_THRESHOLD: continue
                action = 'call' if row['alpha'] > 0 else 'put'
                direction = 1 if row['alpha'] > 0 else -1
                p_entry = last_price_map.get((sym, action), 0.0)
                if p_entry >= MIN_ENTRY_PRICE:
                    qty = int((cash * POSITION_RATIO) // (p_entry * 100))
                    if qty >= 1:
                        entry_p = p_entry * (1 + SLIPPAGE)
                        cash -= (qty * entry_p * 100) + (qty * COMMISSION)
                        positions.append({
                            'symbol': sym, 'type': action, 'qty': qty,
                            'entry_price': entry_p, 'entry_ts': ts,
                            'alpha': row['alpha'], 'max_roi': 0.0,
                            'dir': direction,
                            'entry_stock': last_stock_map.get(sym, 0),
                        })
    
    # Close remaining at EOD
    for pos in positions:
        p = last_price_map.get((pos['symbol'], pos['type']), pos['entry_price'])
        roi = (p - pos['entry_price']) / pos['entry_price']
        trade_history.append({
            'symbol': pos['symbol'], 'alpha': pos['alpha'],
            'entry_t': get_ny_time(pos['entry_ts']),
            'entry_p': pos['entry_price'], 'exit_p': p,
            'roi': roi, 'max': pos['max_roi'], 'reason': 'EOD'
        })
    
    eq = cash + sum([p['qty'] * last_price_map.get((p['symbol'], p['type']), p['entry_price']) * 100 for p in positions])
    return (eq / INITIAL_ACCOUNT - 1), trade_history

# ============= GRID SEARCH =============
if __name__ == "__main__":
    cfg = StrategyConfig()
    
    # Define strategy variants to test
    strategies = {
        "BASELINE (current)": StopConfig(
            early_stop_mins=999, early_stop_roi=-1.0,    # Disabled
            no_momentum_mins=999, no_momentum_min_max_roi=-1.0,  # Disabled
            stock_stop_enabled=False,
            absolute_stop_loss=-0.15,
            time_stop_mins=15, time_stop_roi=0.05,
        ),
        "Plan_A (5m window)": StopConfig(
            early_stop_mins=5, early_stop_roi=-0.05,
            no_momentum_mins=5, no_momentum_min_max_roi=0.03,
            stock_stop_enabled=False,
            absolute_stop_loss=-0.15,
            time_stop_mins=240, time_stop_roi=0.0,
        ),
        "Plan_B (stock stop)": StopConfig(
            early_stop_mins=999, early_stop_roi=-1.0,
            no_momentum_mins=999, no_momentum_min_max_roi=-1.0,
            stock_stop_enabled=True, stock_stop_threshold=0.003, stock_stop_buffer_mins=3,
            absolute_stop_loss=-0.15,
            time_stop_mins=240, time_stop_roi=0.0,
        ),
        "A+B Combined": StopConfig(
            early_stop_mins=5, early_stop_roi=-0.05,
            no_momentum_mins=5, no_momentum_min_max_roi=0.03,
            stock_stop_enabled=True, stock_stop_threshold=0.003, stock_stop_buffer_mins=3,
            absolute_stop_loss=-0.15,
            time_stop_mins=240, time_stop_roi=0.0,
        ),
        "A+B Tight": StopConfig(
            early_stop_mins=3, early_stop_roi=-0.03,
            no_momentum_mins=5, no_momentum_min_max_roi=0.02,
            stock_stop_enabled=True, stock_stop_threshold=0.002, stock_stop_buffer_mins=2,
            absolute_stop_loss=-0.10,
            time_stop_mins=240, time_stop_roi=0.0,
        ),
        "A+B Relaxed": StopConfig(
            early_stop_mins=10, early_stop_roi=-0.08,
            no_momentum_mins=10, no_momentum_min_max_roi=0.05,
            stock_stop_enabled=True, stock_stop_threshold=0.005, stock_stop_buffer_mins=5,
            absolute_stop_loss=-0.20,
            time_stop_mins=240, time_stop_roi=0.0,
        ),
    }
    
    # Load all days
    print("Loading data...")
    day_data = {}
    for db_path in DB_FILES:
        day = os.path.basename(db_path).replace('market_', '').replace('.db', '')
        day_data[day] = load_day_data(db_path)
        print(f"  {day}: {len(day_data[day][0])} alpha signals")
    
    print(f"\n{'Strategy':<22} ", end='')
    for day in day_data: print(f"{'D'+day[-2:]:>8}", end='')
    print(f"{'AVG':>8} {'MaxDD':>8} {'Trades':>8}")
    print("-" * 80)
    
    best_avg = -999
    best_name = ""
    
    for name, stop in strategies.items():
        rets = []
        total_trades = 0
        details = {}
        for day, (df_a, df_s, df_o) in day_data.items():
            ret, trades = run_sim(df_a, df_s, df_o, cfg, stop, alpha_min=1.45)
            rets.append(ret)
            total_trades += len(trades)
            details[day] = trades
        
        avg_ret = np.mean(rets)
        max_dd = min(rets)
        
        print(f"{name:<22} ", end='')
        for r in rets: print(f"{r:>+7.1%}", end=' ')
        print(f"{avg_ret:>+7.1%} {max_dd:>+7.1%} {total_trades:>7}")
        
        if avg_ret > best_avg:
            best_avg = avg_ret
            best_name = name
            best_details = details
    
    print(f"\n🏆 Best Strategy: {best_name} (Avg: {best_avg:+.2%})")
    
    # Show detailed trades for best strategy
    print(f"\n📋 Detailed Trades ({best_name}):")
    for day, trades in best_details.items():
        df = pd.DataFrame(trades)
        if df.empty: continue
        print(f"\n--- Day {day} ---")
        cols = ['symbol', 'alpha', 'entry_t', 'entry_p', 'exit_p', 'roi', 'max', 'reason']
        print(df[cols].sort_values('roi', ascending=False).to_string(index=False))

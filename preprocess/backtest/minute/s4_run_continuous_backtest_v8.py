#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
文件名: s4_run_continuous_backtest_v8.py
描述: [旗舰级] 跨天连续 Parquet 回测引擎。
核心功能:
    1. 内存级高速回放 (无 Redis 延迟)。
    2. 资金与指标状态单向滚动 (Rolling State)。
    3. 支持多日累计收益率计算。
    4. 强制 V0 策略一致性。
"""

import os
os.environ['RUN_MODE'] = 'BACKTEST'

import asyncio
import time
import logging
import json
from pathlib import Path
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import pytz
from tqdm import tqdm

# Add project root to sys.path
MY_DIR = Path(__file__).resolve().parent
ROOT_DIR = MY_DIR.parent.parent
sys.path.insert(0, str(MY_DIR))
sys.path.insert(0, str(ROOT_DIR / "production" / "history_replay"))
sys.path.insert(0, str(ROOT_DIR / "production" / "model"))
sys.path.insert(0, str(ROOT_DIR / "production" / "preprocess"))

from system_orchestrator_v8 import V8Orchestrator
from mock_ibkr_historical import MockIBKRHistorical

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [CONTINUOUS_BT] - %(levelname)s - %(message)s')
logger = logging.getLogger("ContinuousBT")

def load_all_parquets(parquet_dir, target_symbols):
    """从 Parquet 目录加载所有标的数据并按时间戳交织"""
    p = Path(parquet_dir)
    files = list(p.glob("*.parquet"))
    dfs = []
    
    logger.info(f"📂 Loading {len(files)} Parquet files from {parquet_dir}...")
    for f in tqdm(files, desc="Loading Data"):
        sym = f.stem
        if target_symbols and sym not in target_symbols and sym not in {'VIXY', 'SPY', 'QQQ'}:
            continue
            
        df = pd.read_parquet(f)
        df['symbol'] = sym
        dfs.append(df)
        
    if not dfs:
        raise ValueError("❌ No data loaded! Please check parquet_dir.")
        
    full_df = pd.concat(dfs, ignore_index=True)
    logger.info(f"🔗 Sorting {len(full_df)} rows by timestamp...")
    full_df.sort_values('ts', inplace=True)
    return full_df

def build_packet(ts_val, group, is_new_min):
    """复刻 S5 Driver 的 Packet 组装逻辑"""
    symbols_list = group['symbol'].tolist()
    
    packet = {
        'symbols': symbols_list,
        'ts': float(ts_val),
        'stock_price': group['close'].values.astype(np.float32),
        'fast_vol': group['fast_vol'].values.astype(np.float32) if 'fast_vol' in group.columns else np.zeros(len(group), dtype=np.float32),
        'precalc_alpha': group['alpha_score'].values.astype(np.float32) if 'alpha_score' in group.columns else np.zeros(len(group), dtype=np.float32),
        'is_new_minute': is_new_min,
        'spy_roc_5min': group['spy_roc_5min'].values.astype(np.float32) if 'spy_roc_5min' in group.columns else np.zeros(len(group), dtype=np.float32),
        'qqq_roc_5min': group['qqq_roc_5min'].values.astype(np.float32) if 'qqq_roc_5min' in group.columns else np.zeros(len(group), dtype=np.float32),
        
        # 核心期权注入
        'feed_put_price': group['opt_0'].values.astype(np.float32),
        'feed_call_price': group['opt_8'].values.astype(np.float32),
        'feed_put_k': group['opt_5'].values.astype(np.float32),
        'feed_call_k': group['opt_13'].values.astype(np.float32),
        'feed_put_iv': group['opt_7'].values.astype(np.float32),
        'feed_call_iv': group['opt_15'].values.astype(np.float32),
        'feed_put_id': group['opt_0_id'].fillna("").astype(str).tolist(),
        'feed_call_id': group['opt_8_id'].fillna("").astype(str).tolist(),
        
        # 报价相关
        'feed_put_bid': group['feed_put_bid'].values.astype(np.float32) if 'feed_put_bid' in group.columns else group['opt_0'].values.astype(np.float32),
        'feed_put_ask': group['feed_put_ask'].values.astype(np.float32) if 'feed_put_ask' in group.columns else group['opt_0'].values.astype(np.float32),
        'feed_call_bid': group['feed_call_bid'].values.astype(np.float32) if 'feed_call_bid' in group.columns else group['opt_8'].values.astype(np.float32),
        'feed_call_ask': group['feed_call_ask'].values.astype(np.float32) if 'feed_call_ask' in group.columns else group['opt_8'].values.astype(np.float32),
        
        'feed_put_vol': group['feed_put_vol'].values.astype(np.float32) if 'feed_put_vol' in group.columns else np.ones(len(group)),
        'feed_call_vol': group['feed_call_vol'].values.astype(np.float32) if 'feed_call_vol' in group.columns else np.ones(len(group)),
        
        'slow_1m': np.zeros((len(symbols_list), 30, 1), dtype=np.float32)
    }
    # 标注哪些标的在当前 Tick 真正有数据存在
    packet['symbols_with_data'] = set(group[group['opt_0'] > 0.01]['symbol'].tolist()) | set(group[group['opt_8'] > 0.01]['symbol'].tolist())
    
    return packet

async def run_backtest(parquet_dir):
    from config import TARGET_SYMBOLS
    indices = {'VIXY', 'SPY', 'QQQ'}
    engine_symbols = list(set(TARGET_SYMBOLS) | indices)
    
    # 1. 唯一实例化 Orchestrator (开启 Rolling 状态)
    logger.info("🚀 Initializing Stateful V8 Orchestrator (StrategyV0)...")
    orch = V8Orchestrator(symbols=engine_symbols, mode='backtest', config_paths={}, model_paths={})
    
    # 2. 准备 Mock账户
    mock_ib = MockIBKRHistorical()
    await mock_ib.connect()
    orch.ibkr = mock_ib
    orch.mock_cash = 50000.0  # 初始资金
    
    # 3. 加载数据
    full_df = load_all_parquets(parquet_dir, TARGET_SYMBOLS)
    
    # 4. 执行回放循环
    last_min = -1
    daily_reports = []
    current_day_pnl_start = 0.0
    last_date = None
    
    grouped = full_df.groupby('ts')
    logger.info(f"🏃 Starting Replay across {full_df['ts'].nunique()} intervals...")
    
    for ts_val, group in tqdm(grouped, desc="Processing Ticks"):
        dt_ny = datetime.fromtimestamp(ts_val, tz=pytz.utc).astimezone(pytz.timezone('America/New_York'))
        date_str = dt_ny.strftime('%Y-%m-%d')
        time_str = dt_ny.strftime('%H:%M:%S')
        
        # 每一天的开盘结算
        if last_date != date_str:
            if last_date is not None:
                day_net = orch.realized_pnl - current_day_pnl_start
                daily_reports.append({'date': last_date, 'day_pnl': day_net, 'total_pnl': orch.realized_pnl})
                logger.info(f"📅 Day End: {last_date} | Realized PnL: ${day_net:,.2f} | Acc PnL: ${orch.realized_pnl:,.2f}")
            
            last_date = date_str
            current_day_pnl_start = orch.realized_pnl

        # 过滤交易时间 (09:35 - 16:00)
        if time_str < "09:35:00" or time_str > "16:00:00":
            continue
            
        current_min = int(ts_val // 60)
        is_new_min = (current_min != last_min)
        last_min = current_min
        
        packet = build_packet(ts_val, group, is_new_min)
        
        # 核心驱动：执行 V0 逻辑
        await orch.process_batch(packet)
        
        # 记录行情以便 MockIBKR 进行盘中止损判定 (如果需要)
        mock_ib.record_market_data(packet, alphas=packet['precalc_alpha'])
        
    # 最后一天的报告
    if last_date:
        day_net = orch.realized_pnl - current_day_pnl_start
        daily_reports.append({'date': last_date, 'day_pnl': day_net, 'total_pnl': orch.realized_pnl})
        
    # 5. 输出汇总
    report_df = pd.DataFrame(daily_reports)
    print("\n" + "="*60)
    print("📈 CONTINUOUS BACKTEST FINAL REPORT (V0)")
    print("="*60)
    print(report_df.to_string(index=False))
    print("-" * 60)
    print(f"💰 Final Acc PnL:   ${orch.realized_pnl:+,.2f}")
    print(f"📊 Total Trades:     {orch.trade_count}")
    print(f"🎯 Win Rate:         {orch.get_win_rate() if hasattr(orch, 'get_win_rate') else 'N/A'}")
    print("="*60 + "\n")
    
    return report_df

if __name__ == "__main__":
    P_DIR = "/Users/fangshuai/Documents/GitHub/option-qt/production/preprocess/backtest/rl_feed_parquet_batch"
    asyncio.run(run_backtest(P_DIR))

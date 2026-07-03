#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
# 🚨 绝对第一优先级！按倒在 BACKTEST 模式
os.environ['RUN_MODE'] = 'BACKTEST'

import asyncio
import time
import logging
import json
import pandas as pd
import numpy as np
from datetime import datetime, time as dt_time
from pathlib import Path
import sys
import copy
import pytz

 
 

from signal_engine_v8 import SignalEngineV8
from execution_engine_v8 import ExecutionEngineV8
from mock_ibkr_historical import MockIBKRHistorical
import config
from tqdm import tqdm

# ================= 配置区域 =================
PARQUET_DIR = Path.home() / "quant_project/data/rl_feed_parquet"
NY_TZ = pytz.timezone('America/New_York')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [REPLAY] - %(message)s')
logger = logging.getLogger("MultiDayReplay")

def load_multi_symbol_parquet(start_date, end_date, symbols):
    """从 Parquet 目录加载并合并多日、多标的数据"""
    all_dfs = []
    logger.info(f"📂 Loading Parquet data from {start_date} to {end_date}...")
    
    for sym in symbols:
        p = PARQUET_DIR / f"{sym}.parquet"
        if not p.exists():
            logger.warning(f"⚠️ Parquet not found for {sym}")
            continue
        
        df = pd.read_parquet(p)
        df['symbol'] = sym
        # 转换日期进行过滤
        df['date_str'] = pd.to_datetime(df['ts'], unit='s', utc=True).dt.tz_convert('America/New_York').dt.strftime('%Y%m%d')
        df = df[(df['date_str'] >= start_date) & (df['date_str'] <= end_date)]
        
        if not df.empty:
            all_dfs.append(df)
            
    if not all_dfs:
        return pd.DataFrame()
        
    full_df = pd.concat(all_dfs, ignore_index=True)
    full_df = full_df.sort_values('ts')
    return full_df

async def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=str, default="20260102")
    parser.add_argument("--end", type=str, default="20260109")
    args = parser.parse_args()

    symbols = config.TARGET_SYMBOLS
    df = load_multi_symbol_parquet(args.start, args.end, symbols)
    if df.empty:
        logger.error("❌ No data loaded. Check Parquet directory and dates.")
        return

    # 1. 注入 Mock IBKR (仅在总任务开始时 Flush 一次)
    mock_ibkr = MockIBKRHistorical()
    await mock_ibkr.connect()
    logger.info("💥 Nuking Redis Backtest Database (DB 1) for Fresh Start...")
    mock_ibkr.r.flushdb()
    
    # 2. 初始化双引擎 (共享内存确定性模式)
    V8_ROOT = Path(__file__).parent.parent
    config_paths = {
        'fast': str(V8_ROOT/"daily_backtest/fast_feature.json"), 
        'slow': str(V8_ROOT/"daily_backtest/slow_feature.json")
    }

    signal_engine = SignalEngineV8(symbols=symbols, mode='backtest', config_paths=config_paths)
    shared_signal_queue = asyncio.Queue()
    signal_engine.signal_queue = shared_signal_queue
    signal_engine.use_shared_mem = True
    
    exec_engine = ExecutionEngineV8(
        symbols=symbols, mode='backtest',
        shared_states=signal_engine.states,
        signal_queue=shared_signal_queue
    )
    exec_engine.ibkr = mock_ibkr
    signal_engine.ibkr = mock_ibkr
    exec_engine.use_shared_mem = True
    
    # 将 accounting 设为持久模式 (跨日不重置)
    exec_engine.accounting.persistence_mode = True 

    # 3. 准备回放主循环
    grouped = df.groupby('ts')
    unique_ts = sorted(df['ts'].unique())
    last_day = None
    
    logger.info(f"🚀 Starting Multi-Day Deterministic Bus ({len(unique_ts)} ticks)...")

    for ts_val in tqdm(unique_ts, desc="Continuous Replay"):
        group = grouped.get_group(ts_val)
        dt_ny = datetime.fromtimestamp(ts_val, pytz.utc).astimezone(NY_TZ)
        curr_day = dt_ny.strftime('%Y%m%d')
        curr_time = dt_ny.time()
        
        # --- 每天 09:00 强制热重置指标队列 ---
        if curr_day != last_day:
            logger.info(f"📆 --- New Trading Day: {curr_day} ---")
            for st in signal_engine.states.values():
                st.prices.clear() # 清空 30min 滚动窗口，防止隔夜跳空干扰 ER 计算
            last_day = curr_day
            
        # --- 判定 Warmup 阶段 (09:00 - 09:45) ---
        is_warmup = (curr_time < dt_time(9, 45))
        # 强制让 ExecEngine 知道现在是否允许开仓
        exec_engine.strategy.cfg.WARMUP_MODE = is_warmup 

        symbols_list = group['symbol'].tolist()
        # 组装 Packet (结构适配 S3 Parquet 输出)
        packet = {
            'symbols': symbols_list,
            'ts': float(ts_val),
            'stock_price': group['close'].values.astype(np.float32),
            'precalc_alpha': group['alpha_score'].values.astype(np.float32),
            'fast_vol': group['fast_vol'].values.astype(np.float32),
            'event_prob': group['event_prob'].values.astype(np.float32),
            'spy_roc_5min': group['spy_roc_5min'].values.astype(np.float32),
            'qqq_roc_5min': group['qqq_roc_5min'].values.astype(np.float32),
            'is_new_minute': True, # Parquet 通常已是分钟级对齐
            'symbols_with_data': set(symbols_list),
            'feed_put_price': group['feed_put_price'].values.astype(np.float32),
            'feed_call_price': group['feed_call_price'].values.astype(np.float32),
            'feed_put_bid': group['feed_put_bid'].values.astype(np.float32),
            'feed_put_ask': group['feed_put_ask'].values.astype(np.float32),
            'feed_call_bid': group['feed_call_bid'].values.astype(np.float32),
            'feed_call_ask': group['feed_call_ask'].values.astype(np.float32),
            'feed_put_id': group['feed_put_id'].tolist(),
            'feed_call_id': group['feed_call_id'].tolist(),
        }

        # 1. 信号引擎步进
        await signal_engine.process_batch(packet)
        # 2. 模拟柜台更新行情 (确保成交价格有据可查)
        mock_ibkr.record_market_data(packet, alphas=packet['precalc_alpha'])
        
        # 3. 同步排空 OMS 信号队列
        while not shared_signal_queue.empty():
            try:
                sig_payload = shared_signal_queue.get_nowait()
                # 如果是 Warmup 模式且是 BUY 信号，直接拦截
                if is_warmup and sig_payload.get('action') == 'BUY':
                    shared_signal_queue.task_done()
                    continue
                await exec_engine.process_trade_signal(sig_payload)
                shared_signal_queue.task_done()
            except Exception: break
            
        # 4. 注入 SYNC 信号触发 V8 排队打分逻辑
        await exec_engine.process_trade_signal({
            'action': 'SYNC', 
            'ts': float(ts_val), 
            'payload': {}
        })

    # 4. 最终结算
    logger.info("🏁 Multi-day Replay Finished. Generating Final Report...")
    await exec_engine.force_close_all()
    mock_ibkr.save_trades(filename=f"continuous_trades_{args.start}_{args.end}.csv")
    
    print("\n" + "="*50)
    print(f"📊 CUMULATIVE PERFORMANCE ({args.start} - {args.end})")
    print("-" * 50)
    exec_engine.accounting.print_backtest_summary()
    print("="*50)

if __name__ == "__main__":
    asyncio.run(main())
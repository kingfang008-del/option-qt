#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
文件名: s6_fast_memory_backtest.py
描述: [V8 终极极速回测架构] 纯内存计算 + Fused 视图
特点: 完全摒弃 Redis 与网络 I/O 开销，采用 asyncio.Queue 在同进程内实现双引擎直接对话。绝对不会卡死！
"""

import os
import sys

# 🚀 强制环境注入，确保双端逻辑匹配回测场景
os.environ['RUN_MODE'] = 'BACKTEST'
os.environ['DUAL_CONVERGE_TO_SINGLE'] = '1'
os.environ['PURE_ALPHA_REPLAY'] = '1'
os.environ['DISABLE_ICEBERG'] = '1'

import asyncio
import time
import json
import logging
import pandas as pd
import psycopg2
import numpy as np
from datetime import datetime
from pytz import timezone
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='pandas')

# 引入配置与业务模块
from config import PG_DB_URL, TARGET_SYMBOLS
from signal_engine_v8 import SignalEngineV8
from execution_engine_v8 import ExecutionEngineV8
from mock_ibkr_historical_1s import MockIBKRHistorical

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("FastBacktest")

def fetch_fused_data(date_str: str) -> pd.DataFrame:
    """直接从 PG 数据库读取准备好的 行情+信号 终态视图"""
    ny_tz = timezone('America/New_York')
    dt = datetime.strptime(date_str, '%Y%m%d')
    start_dt = ny_tz.localize(dt.replace(hour=9, minute=30, second=0))
    end_dt = ny_tz.localize(dt.replace(hour=16, minute=0, second=0))
    
    conn = psycopg2.connect(PG_DB_URL)
    query = f"""
        SELECT * FROM fused_market_alpha_1m 
        WHERE ts >= {start_dt.timestamp()} AND ts <= {end_dt.timestamp()}
        ORDER BY ts ASC
    """
    df = pd.read_sql(query, conn)
    conn.close()
    return df

async def run_pure_memory_backtest(date_str: str):
    start_time = time.time()
    logger.info(f"📥 Loading fused data for {date_str}...")
    df_fused = fetch_fused_data(date_str)
    
    if df_fused.empty:
        logger.error("❌ No data found for the given date.")
        return

    # 1. 建立内存底座 (🚀 [严格对齐 S5] 修复参数)
    mock_ibkr = MockIBKRHistorical()
    mock_ibkr.initial_capital = 50000.0  # 手动设置初始资金
    
    shared_signal_queue = asyncio.Queue()
    
    # 2. 实例化双引擎 (🚀 [严格对齐 S5] 注入 TARGET_SYMBOLS 与 ibkr kwargs)
    exec_engine = ExecutionEngineV8(TARGET_SYMBOLS, ibkr=mock_ibkr)
    # 注意: 若你的 SignalEngineV8 初始化参数略有不同，请自行微调，通常与 ExecEngine 保持一致
    signal_engine = SignalEngineV8(TARGET_SYMBOLS, ibkr=mock_ibkr, shared_signal_queue=shared_signal_queue)
    
    grouped = df_fused.groupby('ts')
    total_frames = len(grouped)
    logger.info(f"🚀 Starting Pure Memory Backtest: {total_frames} frames to process.")

    frame_id = 0
    for ts_val, group_df in grouped:
        frame_id += 1
        rows = group_df.to_dict('records')
        
        # --- 组装标准引擎载荷 ---
        put_prices, call_prices = [], []
        put_ivs, call_ivs = [], []
        put_bids, put_asks = [], []
        call_bids, call_asks = [], []
        put_ks, call_ks = [], []

        for row in rows:
            symbols.append(row['symbol'])
            stock_price.append(float(row['close']))
            stock_volume.append(float(row['volume']))
            precalc_alpha.append(float(row.get('precalc_alpha', 0.0)))
            fast_vol.append(float(row.get('fast_vol', 0.0)))
            spy_rocs.append(float(row.get('spy_roc_5min', 0.0)))
            qqq_rocs.append(float(row.get('qqq_roc_5min', 0.0)))

            # 🚀 [Parity Fix] 解析真实的期权快照 (S4 同款索引)
            try:
                bk_raw = row.get('buckets_json')
                if bk_raw:
                    bk = json.loads(bk_raw) if isinstance(bk_raw, str) else bk_raw
                    buckets = bk.get('buckets', [])
                else:
                    buckets = []
                
                # ATM Put (index 0) / ATM Call (index 2)
                p_p = float(buckets[0][0]) if len(buckets) > 0 and len(buckets[0]) > 0 else 0.0
                c_p = float(buckets[2][0]) if len(buckets) > 2 and len(buckets[2]) > 0 else 0.0
                
                put_prices.append(p_p)
                put_ks.append(float(buckets[0][5]) if len(buckets) > 0 and len(buckets[0]) > 5 else 0.0)
                put_ivs.append(float(buckets[0][7]) if len(buckets) > 0 and len(buckets[0]) > 7 else 0.0)
                put_bids.append(float(buckets[0][8]) if len(buckets) > 0 and len(buckets[0]) > 8 else 0.0)
                put_asks.append(float(buckets[0][9]) if len(buckets) > 0 and len(buckets[0]) > 9 else 0.0)

                call_prices.append(c_p)
                call_ks.append(float(buckets[2][5]) if len(buckets) > 2 and len(buckets[2]) > 5 else 0.0)
                call_ivs.append(float(buckets[2][7]) if len(buckets) > 2 and len(buckets[2]) > 7 else 0.0)
                call_bids.append(float(buckets[2][8]) if len(buckets) > 2 and len(buckets[2]) > 8 else 0.0)
                call_asks.append(float(buckets[2][9]) if len(buckets) > 2 and len(buckets[2]) > 9 else 0.0)
            except Exception as e:
                # 异常容错处理
                for arr in [put_prices, call_prices, put_ivs, call_ivs, put_bids, put_asks, call_bids, call_asks, put_ks, call_ks]:
                    arr.append(0.0)

        if not symbols: continue

        packet = {
            "ts": float(ts_val),
            "frame_id": frame_id,
            "symbols": symbols,
            "stock_price": np.asarray(stock_price, dtype=np.float32),
            "stock_volume": np.asarray(stock_volume, dtype=np.float32),
            "precalc_alpha": np.asarray(precalc_alpha, dtype=np.float32),
            "fast_vol": np.asarray(fast_vol, dtype=np.float32),
            "spy_roc_5min": np.asarray(spy_rocs, dtype=np.float32),
            "qqq_roc_5min": np.asarray(qqq_rocs, dtype=np.float32),
            "alpha_label_ts": np.full(len(symbols), float(ts_val), dtype=np.float64),
            
            # --- 🚀 [CRITICAL] 注入真实期权字段，对齐 S4 ---
            "feed_call_price": np.array(call_prices, dtype=np.float32),
            "feed_put_price": np.array(put_prices, dtype=np.float32),
            "feed_call_bid": np.array(call_bids, dtype=np.float32),
            "feed_call_ask": np.array(call_asks, dtype=np.float32),
            "feed_put_bid": np.array(put_bids, dtype=np.float32),
            "feed_put_ask": np.array(put_asks, dtype=np.float32),
            "feed_call_iv": np.array(call_ivs, dtype=np.float32),
            "feed_put_iv": np.array(put_ivs, dtype=np.float32),
            "feed_call_k": np.array(call_ks, dtype=np.float32),
            "feed_put_k": np.array(put_ks, dtype=np.float32),
            
            "source": "fused_memory_v8"
        }
        
        # 强制缩短暖机期，确保留第一分钟就能看到信号输出
        if hasattr(signal_engine, 'cfg'):
            signal_engine.cfg.WARMUP_PERIOD = 0
            # 同时也让每个状态实例立即准备好
            for st in signal_engine.states.values():
                st.warmup_complete = True
        
        # --- 3. 内存直接推演 (告别 Redis) ---
        
        # 3.1 灌入底层基准价格
        if hasattr(mock_ibkr, 'record_market_data'):
            mock_ibkr.record_market_data(packet, alphas=packet['precalc_alpha'])
        
        # 3.2 信号引擎接收宽表，计算 Kelly 并生成买卖指令放入 shared_signal_queue
        await signal_engine.process_batch(packet)
        
        # 3.3 执行引擎消费 Queue 中的全部交易指令
        while not shared_signal_queue.empty():
            sig_payload = shared_signal_queue.get_nowait()
            await exec_engine.process_trade_signal(sig_payload)
            
        # 3.4 核心心跳对齐 (促使 OMS 更新当前持仓对账状态)
        await exec_engine.process_trade_signal({'action': 'SYNC', 'ts': float(ts_val), 'payload': {}})
        
        if hasattr(signal_engine, 'mock_cash') and hasattr(exec_engine, 'mock_cash'):
            signal_engine.mock_cash = exec_engine.mock_cash
        
        if frame_id % 60 == 0:
            print(f"✅ Processed {frame_id}/{total_frames} minutes...", end='\r')

    print(f"\n🏁 Finished processing. Wrapping up...")
    await exec_engine.force_close_all()
    await asyncio.sleep(0.5)
    
    elapsed = time.time() - start_time
    print(f"✅ Fast Backtest Completed in {elapsed:.1f}s.")
    
    # 4. 生成报告
    report_name = f"replay_trades_s6_{date_str}.csv"
    mock_ibkr.save_trades(filename=report_name)
    
    print("\n" + "=" * 50)
    print(f"📊 FINAL PURE-MEMORY BACKTEST SUMMARY")
    print("=" * 50)
    print(f"💰 Final Cash Balance: ${mock_ibkr.initial_capital:.2f}")
    if hasattr(mock_ibkr, 'get_total_pnl'):
        print(f"💵 Total Realized PnL: ${mock_ibkr.get_total_pnl():.2f}")
    print(f"📄 Report saved to: {report_name}")
    print("=" * 50 + "\n")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python s6_fast_memory_backtest.py YYYYMMDD")
        sys.exit(1)
        
    try:
        asyncio.run(run_pure_memory_backtest(sys.argv[1]))
    except KeyboardInterrupt:
        print("\n⏹️ Terminated by User.")
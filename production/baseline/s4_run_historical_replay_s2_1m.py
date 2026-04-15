#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
# 🚨 绝对第一优先级！在所有 import 之前，强行把整个进程及其子模块按倒在 BACKTEST 模式！
os.environ['RUN_MODE'] = 'BACKTEST'
os.environ['DUAL_CONVERGE_TO_SINGLE'] = '1'
os.environ['PURE_ALPHA_REPLAY'] = '1'

import asyncio
import threading
import time
import logging
from pathlib import Path
import argparse 
import sys
import pandas as pd
import sqlite3
import json
import numpy as np
from datetime import datetime
import pytz
from tqdm import tqdm

# 🚨 动态添加路径，确保 baseline, history_replay 和 model 目录下的模块能被识别
MY_DIR = Path(__file__).resolve().parent
ROOT_DIR = MY_DIR.parent.parent # 🚀 [Refined Fix] 精准指向 option-qt 根目录
# 强行插入到开头，确保 production 目录下的文件优先于系统同名模块
sys.path.insert(0, str(ROOT_DIR / "production" / "baseline"))
sys.path.insert(0, str(ROOT_DIR / "production" / "history_replay"))
sys.path.insert(0, str(ROOT_DIR / "production" / "model"))

# 导入双引擎
from signal_engine_v8 import SignalEngineV8
from execution_engine_v8 import ExecutionEngineV8
# 导入 Driver 和 Mock IBKR 
import mock_ibkr_historical
from mock_ibkr_historical import MockIBKRHistorical
print(f"🔍 [Import Trace] Using MockIBKR from: {mock_ibkr_historical.__file__}")

# 🚨 导入 config 并确认 Redis 流名称 (虽然不再作为主要 Bus，但维持一致性)
import config
import signal_engine_v8
import execution_engine_v8

BACKTEST_STREAM = "backtest_stream"
BACKTEST_GROUP  = "backtest_group"

# 统一所有可能用到的配置源
config.REDIS_CFG['input_stream'] = BACKTEST_STREAM
config.REDIS_CFG['orch_group'] = BACKTEST_GROUP
signal_engine_v8.REDIS_CFG['input_stream'] = BACKTEST_STREAM
signal_engine_v8.REDIS_CFG['orch_group'] = BACKTEST_GROUP
execution_engine_v8.REDIS_CFG['input_stream'] = BACKTEST_STREAM
execution_engine_v8.REDIS_CFG['orch_group'] = BACKTEST_GROUP

# ================= 配置区域 =================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [REPLAY_S2_1M] - %(message)s')
logger = logging.getLogger("ReplayS2_1M")

def resolve_db_path(args):
    """[Critical Fix] 严格锁定并寻找 history_sqlite_1m 目录下的权威数据库"""
    CURRENT_DIR = Path(__file__).resolve().parent
    # 🚀 既然确定了是在 history_sqlite_1m 下，我们直接锁定这个路径，不做任何模糊搜索
    #HIST_DIR = CURRENT_DIR.parent / "preprocess" / "backtest" / "history_sqlite_1m"
    HIST_DIR = Path("/home/kingfang007/quant_project/data/history_sqlite_1m")
    if args.date:
        db_name = f"market_{args.date}.db"
        p = HIST_DIR / db_name
        if p.exists(): 
            return p
        else:
            logger.error(f"❌ 关键数据库未找到: {p}")
            return None
    
    all_dbs = sorted(HIST_DIR.glob("market_*.db"))
    return all_dbs[-1] if all_dbs else None

def safe_col(group, col, default_val, dtype=np.float32):
    """安全提取 DataFrame 列，缺失时返回默认值数组"""
    if col in group.columns:
        return group[col].fillna(default_val).values.astype(dtype)
    return np.full(len(group), default_val, dtype=dtype)

async def main():
    print(f"!!! EXECUTING UPDATED DETERMINISTIC V8 SCRIPT: {__file__} !!!")
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, default="20260102")
    parser.add_argument("--symbol", type=str, default=None) 
    args = parser.parse_args()
    
    db_path = resolve_db_path(args)
    if db_path is None:
        logger.error(f"❌ Target Database not found. (Date: {args.date})")
        return
    
    logger.info(f"📂 Loading data directly from SQLite: {db_path}...")

    V8_ROOT = Path(__file__).parent.parent
    config_paths = {
        'fast': str(V8_ROOT/"daily_backtest/fast_feature.json"), 
        'slow': str(V8_ROOT/"daily_backtest/slow_feature.json")
    }
    from config import TARGET_SYMBOLS
    symbols = [args.symbol] if args.symbol else TARGET_SYMBOLS

    logger.info("🛠️ Building V8 Dual Engine...") 
    signal_engine = SignalEngineV8(
        symbols=symbols,
        mode='backtest',
        config_paths=config_paths,
        model_paths={}
    )
    shared_signal_queue = asyncio.Queue()
    signal_engine.signal_queue = shared_signal_queue
    signal_engine.use_shared_mem = True

    exec_engine = ExecutionEngineV8(
        symbols=symbols,
        mode='backtest',
        shared_states=signal_engine.states,
        signal_queue=shared_signal_queue
    )
    exec_engine.strategy.cfg = signal_engine.strategy.cfg
    exec_engine.cfg = signal_engine.strategy.cfg
    exec_engine.use_shared_mem = True
    
    logger.info(f"🔌 Injecting Mock IBKR...")
    mock_ibkr = MockIBKRHistorical()
    await mock_ibkr.connect()
    
    # 彻底清理 Redis 遗留状态
    mock_ibkr.r.flushdb()
    
    signal_engine.ibkr = mock_ibkr
    exec_engine.ibkr = mock_ibkr
    signal_engine.mock_cash = mock_ibkr.initial_capital
    exec_engine.mock_cash = mock_ibkr.initial_capital
    
    # ==========================================
    # 数据加载与直接对齐 (Robust AsOf Join)
    # ==========================================
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    df_a = pd.read_sql(f"SELECT ts, symbol, alpha as alpha_score, vol_z as fast_vol FROM alpha_logs", conn)
    df_s = pd.read_sql("SELECT ts, symbol, close, open, high, low, volume FROM market_bars_1m", conn)
    df_o = pd.read_sql("SELECT ts, symbol, buckets_json FROM option_snapshots_1m", conn)
    conn.close()

    # 🚀 [NEW] Calculate Index ROCs from the full dataset before filtering
    # Ensure ts is float for all calculations
    df_s['ts'] = df_s['ts'].astype(float)
    df_idx = df_s[df_s['symbol'].isin(['SPY', 'QQQ'])].pivot(index='ts', columns='symbol', values='close')
    
    # Prepare a ROC map for all unique timestamps
    unique_ts = sorted(df_s['ts'].unique())
    df_roc_map = pd.DataFrame(index=unique_ts)
    for col in ['SPY', 'QQQ']:
        if col in df_idx.columns:
            # 5-minute ROC (1m bars = 5 periods)
            df_roc_map[f'{col.lower()}_roc_5min'] = df_idx[col].pct_change(periods=5).fillna(0.0)
        else:
            df_roc_map[f'{col.lower()}_roc_5min'] = 0.0
    
    # Merge ROCs into df_s so they carry over after symbol filtering
    df_s = df_s.merge(df_roc_map.reset_index().rename(columns={'index': 'ts'}), on='ts', how='left')

    if symbols:
        df_a = df_a[df_a['symbol'].isin(symbols)]
        df_s = df_s[df_s['symbol'].isin(symbols)]
        df_o = df_o[df_o['symbol'].isin(symbols)]

    # 🚀 [巅峰复原关键] 统一数据类型并使用 merge_asof 保证数据流不断
    for df_tmp in [df_a, df_s, df_o]:
        df_tmp['ts'] = df_tmp['ts'].astype(float)
        df_tmp['symbol'] = df_tmp['symbol'].astype(str)
        
    df_a = df_a.sort_values('ts')
    df_s = df_s.sort_values('ts')
    df_o = df_o.sort_values('ts')
    
    # 先对齐行情与期权
    df_market = pd.merge_asof(df_s, df_o, on='ts', by='symbol', direction='backward', tolerance=120)
    # 再对齐 Alpha (向后对齐 120s)
    df = pd.merge_asof(df_market, df_a, on='ts', by='symbol', direction='backward', tolerance=120)
    
    if df.empty:
        logger.error("❌ Merged dataset is EMPTY! Check DB integrity.")
        return

    grouped = df.sort_values('ts', ascending=True).groupby('ts')

    unique_groups = df['ts'].nunique()
    
    logger.info(f"🚀 Starting Deterministic Replay Bus ({unique_groups} ticks)...")
    start_time = time.time()
    
    # ==========================================
    # 确定性回放主循环
    # ==========================================
    ny_tz = pytz.timezone('America/New_York')
    last_minute = -1
    
    for ts_val, group in tqdm(grouped, desc="Replaying", total=unique_groups):
        dt_ny = datetime.fromtimestamp(ts_val, tz=pytz.utc).astimezone(ny_tz)
        curr_time_str = dt_ny.strftime('%H:%M:%S')
        
        # 限制交易时段
        if curr_time_str < "09:35:00" or curr_time_str > "16:00:00":
            continue

        current_minute = int(ts_val // 60)
        is_new_minute = (last_minute != current_minute)
        last_minute = current_minute
        
        symbols_list = group['symbol'].tolist()
        
        # 组装期权 Bucket 数据
        put_prices, call_prices = [], []
        put_ks, call_ks = [], []
        put_ivs, call_ivs = [], []
        put_bids, put_asks = [], []
        call_bids, call_asks = [], []
        symbols_with_data = set()
        
        for idx_row, row in enumerate(group.itertuples()):
            try:
                bk = json.loads(row.buckets_json).get('buckets', [])
                p_p = float(bk[0][0]) if len(bk) > 0 and len(bk[0]) > 0 else 0.0
                c_p = float(bk[2][0]) if len(bk) > 2 and len(bk[2]) > 0 else 0.0
                
                put_prices.append(p_p)
                put_ks.append(float(bk[0][5]) if len(bk) > 0 and len(bk[0]) > 5 else 0.0)
                put_ivs.append(float(bk[0][7]) if len(bk) > 0 and len(bk[0]) > 7 else 0.0)
                put_bids.append(float(bk[0][8]) if len(bk) > 0 and len(bk[0]) > 8 else 0.0)
                put_asks.append(float(bk[0][9]) if len(bk) > 0 and len(bk[0]) > 9 else 0.0)

                call_prices.append(c_p)
                call_ks.append(float(bk[2][5]) if len(bk) > 2 and len(bk[2]) > 5 else 0.0)
                call_ivs.append(float(bk[2][7]) if len(bk) > 2 and len(bk[2]) > 7 else 0.0)
                call_bids.append(float(bk[2][8]) if len(bk) > 2 and len(bk[2]) > 8 else 0.0)
                call_asks.append(float(bk[2][9]) if len(bk) > 2 and len(bk[2]) > 9 else 0.0)
                
                if p_p > 0.01 or c_p > 0.01:
                    symbols_with_data.add(row.symbol)
            except:
                for arr in [put_prices, call_prices, put_ks, call_ks, put_ivs, call_ivs, put_bids, put_asks, call_bids, call_asks]:
                    arr.append(0.0)

        # 组装 V8 兼容 Packet
        packet = {
            'symbols': symbols_list,
            'ts': float(ts_val),
            'stock_price': group['close'].values.astype(np.float32),
            'fast_vol': group['fast_vol'].values.astype(np.float32),
            'precalc_alpha': group['alpha_score'].values.astype(np.float32),
            'spy_roc_5min': safe_col(group, 'spy_roc_5min', 0.0),
            'qqq_roc_5min': safe_col(group, 'qqq_roc_5min', 0.0),
            'is_new_minute': is_new_minute,
            'symbols_with_data': symbols_with_data,
            'feed_put_price': np.array(put_prices, dtype=np.float32),
            'feed_call_price': np.array(call_prices, dtype=np.float32),
            'feed_put_k': np.array(put_ks, dtype=np.float32),
            'feed_call_k': np.array(call_ks, dtype=np.float32),
            'feed_put_iv': np.array(put_ivs, dtype=np.float32),
            'feed_call_iv': np.array(call_ivs, dtype=np.float32),
            'feed_put_bid': np.array(put_bids, dtype=np.float32),
            'feed_put_ask': np.array(put_asks, dtype=np.float32),
            'feed_call_bid': np.array(call_bids, dtype=np.float32),
            'feed_call_ask': np.array(call_asks, dtype=np.float32),
            'feed_put_vol': np.ones(len(group), dtype=np.float32),
            'feed_call_vol': np.ones(len(group), dtype=np.float32),
            'feed_call_bid_size': np.full(len(group), 100.0, dtype=np.float32),
            'feed_call_ask_size': np.full(len(group), 100.0, dtype=np.float32),
            'feed_put_bid_size': np.full(len(group), 100.0, dtype=np.float32),
            'feed_put_ask_size': np.full(len(group), 100.0, dtype=np.float32),
            'slow_1m': np.zeros((len(symbols_list), 30, 1), dtype=np.float32), 
            'feed_put_id': [""] * len(group),
            'feed_call_id': [""] * len(group),
        }

        if is_new_minute:
            sig_count = int((packet['precalc_alpha'] != 0).sum())
            if sig_count > 0:
                logger.info(f"📡 [REPLAY_S2_1M] Minute {current_minute} | Active Signals: {sig_count}")

        await signal_engine.process_batch(packet)
        mock_ibkr.record_market_data(packet, alphas=packet['precalc_alpha'])

        while not shared_signal_queue.empty():
            try:
                sig_payload = shared_signal_queue.get_nowait()
                await exec_engine.process_trade_signal(sig_payload)
                shared_signal_queue.task_done()
            except Exception:
                break

        await exec_engine.process_trade_signal({'action': 'SYNC', 'ts': float(ts_val), 'payload': {}})
        signal_engine.mock_cash = exec_engine.mock_cash

    # ==========================================
    # 汇总
    # ==========================================
    elapsed = time.time() - start_time
    print(f"\n✅ Stable Replay Finished in {elapsed:.1f}s.")
    await exec_engine.force_close_all()
    await asyncio.sleep(0.5)
    mock_ibkr.save_trades(filename="replay_trades_s2_1m.csv") 
    
    print("\n" + "="*50)
    print("📊 FINAL BACKTEST PERFORMANCE SUMMARY (S2 1M DUAL)")
    print("="*50)
    exec_engine.accounting.print_backtest_summary()
    exec_engine.accounting.print_counter_trend_summary()
    print("="*50 + "\n")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass

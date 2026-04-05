#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
文件名: verify_dual_engine_1s.py
描述: 1秒级数据对标验证工具。
      直接读取 history_sqlite_1s 下预生成的 alpha_logs，通过 Dual Engine (SE+EE) 逻辑跑出交易记录，
      用于验证秒级 Alpha 的“实战可用性”及与 1m 基准的偏离程度。
"""

import os
os.environ['RUN_MODE'] = 'BACKTEST'

import asyncio
import time
import logging
import sqlite3
import json
from pathlib import Path
import argparse 
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import pytz

# Add project root to sys.path
MY_DIR = Path(__file__).resolve().parent
ROOT_DIR = MY_DIR.parent.parent
sys.path.insert(0, str(ROOT_DIR / "production" / "baseline"))
sys.path.insert(0, str(ROOT_DIR / "production" / "history_replay"))
sys.path.insert(0, str(ROOT_DIR / "production" / "model"))

from system_orchestrator_v8 import V8Orchestrator
from signal_engine_v8 import SignalEngineV8
from execution_engine_v8 import ExecutionEngineV8
from mock_ibkr_historical import MockIBKRHistorical

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [VERIFY_1S] - %(levelname)s - %(message)s')
logger = logging.getLogger("Verify1S")

def safe_col(group, col, default_val, dtype=np.float32):
    if col in group.columns:
        return group[col].fillna(default_val).values.astype(dtype)
    return np.full(len(group), default_val, dtype=dtype)

def load_official_stable_data(db_path, target_symbols):
    """
    针对 1s 数据库的结构化加载：
    秒级库中通常包含 market_bars_1m 和 alpha_logs，能直接被复用。
    """
    logger.info(f"📂 Loading data from {db_path}...")
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    
    # 1s 库中依然存有同步后的 1m K线，方便对标
    df_a = pd.read_sql(f"SELECT ts, symbol, alpha as alpha_score, vol_z as fast_vol FROM alpha_logs", conn)
    df_s = pd.read_sql("SELECT ts, symbol, close, open, high, low, volume FROM market_bars_1m", conn)
    df_o = pd.read_sql("SELECT ts, symbol, buckets_json FROM option_snapshots_1m", conn)
    conn.close()
    
    mandatory_indices = {'VIXY', 'SPY', 'QQQ'}
    all_symbols = set(target_symbols) | mandatory_indices
    df_a = df_a[df_a['symbol'].isin(all_symbols)]
    df_s = df_s[df_s['symbol'].isin(all_symbols)]
    df_o = df_o[df_o['symbol'].isin(all_symbols)]
    
    for df_tmp in [df_a, df_s, df_o]:
        df_tmp['ts'] = df_tmp['ts'].astype(float)
        df_tmp['symbol'] = df_tmp['symbol'].astype(str)
        df_tmp.sort_values('ts', inplace=True)
        
    df_market = pd.merge_asof(df_s, df_o, on='ts', by='symbol', direction='backward', tolerance=120)
    df = pd.merge_asof(df_market, df_a, on='ts', by='symbol', direction='backward', tolerance=120)

    # 🚀 [Parity Fix] 重新执行 100% 截面归一化 (Cross-Sectional Z-Score)
    # 既然 1s 库里的 Alpha 是之前混合归一化生成的 (偏低)，我们在验证阶段将其“还原”回纯截面模式
    def normalize_slice(group):
        # 排除指数标的 (NaN) 进行统计计算
        raw_a = group['alpha_score'].values
        valid_a = raw_a[~np.isnan(raw_a)]
        if len(valid_a) > 1:
            m = np.mean(valid_a)
            s = np.std(valid_a)
            group['alpha_score'] = (raw_a - m) / (s + 1e-6)
        else:
            # 兜底：如果本秒没有有效 Alpha，全部归零
            group['alpha_score'] = 0.0
        return group

    logger.info("🧪 Re-normalizing Alphas to 100% Cross-Sectional Mode...")
    df = df.groupby('ts', group_keys=False).apply(normalize_slice)
    
    return df.sort_values('ts')

def build_stable_packet(ts_val, group, is_new_min):
    symbols_list = group['symbol'].tolist()
    put_prices, call_prices = [], []
    put_ks, call_ks = [], []
    put_ivs, call_ivs = [], []
    put_bids, put_asks = [], []
    call_bids, call_asks = [], []
    symbols_with_data = set()
    
    for row in group.itertuples():
        try:
            buckets_json = getattr(row, 'buckets_json', None)
            if not buckets_json or buckets_json == "{}": 
                for arr in [put_prices, call_prices, put_ks, call_ks, put_ivs, call_ivs, put_bids, put_asks, call_bids, call_asks]: arr.append(0.0)
                continue
            bk = json.loads(buckets_json).get('buckets', [])
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
            if p_p > 0.01 or c_p > 0.01: symbols_with_data.add(row.symbol)
        except:
            for arr in [put_prices, call_prices, put_ks, call_ks, put_ivs, call_ivs, put_bids, put_asks, call_bids, call_asks]: arr.append(0.0)
            
    packet = {
        'symbols': symbols_list, 'ts': float(ts_val), 
        'stock_price': group['close'].values.astype(np.float32), 
        'fast_vol': safe_col(group, 'fast_vol', 0.0).astype(np.float32), 
        'precalc_alpha': safe_col(group, 'alpha_score', 0.0).astype(np.float32), 
        'is_new_minute': is_new_min, 
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
        'spy_roc_5min': safe_col(group, 'spy_roc_5min', 0.0), 
        'qqq_roc_5min': safe_col(group, 'qqq_roc_5min', 0.0)
    }
    return packet

async def run_dual_engine(date_str, db_path, target_symbols):
    engine_symbols = list(set(target_symbols) | {'VIXY', 'SPY', 'QQQ'})
    # 注意：SE 这里初始化不需要模型路径，因为我们强制使用 DB 里的 precalc_alpha
    se = SignalEngineV8(symbols=engine_symbols, mode='backtest', config_paths={}, model_paths={})
    ee = ExecutionEngineV8(symbols=engine_symbols, mode='backtest')
    
    # 同步初始状态
    for sym in engine_symbols:
        if sym in se.states and sym in ee.states: ee.states[sym] = se.states[sym]
        
    signal_queue = asyncio.Queue()
    se.use_shared_mem = True
    se.signal_queue = signal_queue
    ee.use_shared_mem = True
    ee.signal_queue = signal_queue
    
    mock_ib = MockIBKRHistorical(db_path)
    await mock_ib.connect()
    ee.ibkr = mock_ib
    
    df = load_official_stable_data(db_path, target_symbols)
    if df.empty:
        logger.warning("⚠️ No data loaded for simulation.")
        return 0, 0
        
    # 🚀 [Parity Fix] 进场前全局行情对齐: 解决 1s 库中 option_snapshots_1m 偶尔断流导致“零元离场”的问题
    logger.info("🔧 Re-aligning option quotes (Forward Fill)...")
    df.loc[df['buckets_json'] == "{}", 'buckets_json'] = None
    df.loc[df['buckets_json'] == "None", 'buckets_json'] = None
    df['buckets_json'] = df.groupby('symbol')['buckets_json'].ffill()
    
    last_min = -1
    logger.info(f"🚀 Starting Dual-Engine Replay for {date_str}...")
    
    for ts_val, group in df.groupby('ts'):
        dt_ny = datetime.fromtimestamp(ts_val, tz=pytz.utc).astimezone(pytz.timezone('America/New_York'))
        time_str = dt_ny.strftime('%H:%M:%S')
        if time_str < "09:35:00" or time_str > "16:00:00": continue
        
        current_min = int(ts_val // 60)
        is_new_min = (current_min != last_min)
        last_min = current_min
        
        packet = build_stable_packet(ts_val, group, is_new_min)
        
        # 🚀 [Parity Fix] 必须在处理信号前更新 Mock Broker 行情，否则执行时会因为“暂无行情”而以 $0.01 结账！
        mock_ib.record_market_data(packet, alphas=packet['precalc_alpha'])
        
        # 🚀 [Parity Fix] 必须全量发送 Batch，否则 SE 内部的截面归一化 (Z-Score) 会因为样本数为 1 而退化归零！
        await se.process_batch(packet)
        
        # 处理信号队列
        while not signal_queue.empty():
            payload = signal_queue.get_nowait()
            await ee.process_trade_signal(payload)
            signal_queue.task_done()
        
    # 🚀 [New] 打印所有交易明细以供审计
    if ee.trade_count > 0:
        logger.info(f"📜 ALL {ee.trade_count} TRADE DETAILS:")
        trades = getattr(ee.accounting.orch, 'daily_trades', [])
        for t in trades:
            logger.info(f"👉 {t['symbol']} | Entry: {t['entry_ts']}@${t['entry_price']:.2f} | Exit: {t['exit_ts']}@${t['exit_price']:.2f} | Qty: {t['qty']} | PnL: ${t['pnl']:.2f} | Reason: {t['reason']}")
        
    return ee.realized_pnl, ee.trade_count

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dates", type=str, default="20260102")
    args = parser.parse_args()
    
    date_list = args.dates.split(",")
    # 🚀 强制指向 1s 数据库目录
    HIST_DIR = ROOT_DIR / "production" / "preprocess" / "backtest" / "history_sqlite_1s"
    
    from config import TARGET_SYMBOLS
    tradeable_symbols = [s for s in TARGET_SYMBOLS if s not in {'VIXY', 'SPY', 'QQQ'}]
    
    results = []
    for date_str in date_list:
        db_path = HIST_DIR / f"market_{date_str}.db"
        if not db_path.exists():
            logger.error(f"❌ Database not found: {db_path}")
            continue
            
        pnl_d, trades_d = await run_dual_engine(date_str, str(db_path), tradeable_symbols)
        results.append({
            'date': date_str, 
            'dual_pnl': pnl_d, 
            'dual_trades': trades_d
        })
        
    logger.info("="*80)
    logger.info(f"📊 1S DUAL-ENGINE VERIFICATION REPORT")
    logger.info("="*80)
    logger.info(f"{'Date':<10} | {'Dual PnL':<12} | {'Trd Count':<6}")
    logger.info("-" * 80)
    for r in results:
        logger.info(f"{r['date']:<10} | ${r['dual_pnl']:>10.2f} | {r['dual_trades']:>6}")
    logger.info("="*80)
    
if __name__ == "__main__":
    asyncio.run(main())

#!/usr/bin/env python
# -*- coding: utf-8 -*-

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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [VERIFY_ULTIMATE] - %(levelname)s - %(message)s')
logger = logging.getLogger("VerifySync")

def safe_col(group, col, default_val, dtype=np.float32):
    if col in group.columns:
        return group[col].fillna(default_val).values.astype(dtype)
    return np.full(len(group), default_val, dtype=dtype)

def load_official_stable_data(db_path, target_symbols):
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
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
        df_tmp['ts'] = df_tmp['ts'].astype(float); df_tmp['symbol'] = df_tmp['symbol'].astype(str); df_tmp.sort_values('ts', inplace=True)
    df_market = pd.merge_asof(df_s, df_o, on='ts', by='symbol', direction='backward', tolerance=120)
    df = pd.merge_asof(df_market, df_a, on='ts', by='symbol', direction='backward', tolerance=120)
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
            if not getattr(row, 'buckets_json', None): 
                for arr in [put_prices, call_prices, put_ks, call_ks, put_ivs, call_ivs, put_bids, put_asks, call_bids, call_asks]: arr.append(0.0)
                continue
            bk = json.loads(row.buckets_json).get('buckets', [])
            p_p = float(bk[0][0]) if len(bk) > 0 and len(bk[0]) > 0 else 0.0
            c_p = float(bk[2][0]) if len(bk) > 2 and len(bk[2]) > 0 else 0.0
            put_prices.append(p_p); put_ks.append(float(bk[0][5]) if len(bk) > 0 and len(bk[0]) > 5 else 0.0); put_ivs.append(float(bk[0][7]) if len(bk) > 0 and len(bk[0]) > 7 else 0.0); put_bids.append(float(bk[0][8]) if len(bk) > 0 and len(bk[0]) > 8 else 0.0); put_asks.append(float(bk[0][9]) if len(bk) > 0 and len(bk[0]) > 9 else 0.0)
            call_prices.append(c_p); call_ks.append(float(bk[2][5]) if len(bk) > 2 and len(bk[2]) > 5 else 0.0); call_ivs.append(float(bk[2][7]) if len(bk) > 2 and len(bk[2]) > 7 else 0.0); call_bids.append(float(bk[2][8]) if len(bk) > 2 and len(bk[2]) > 8 else 0.0); call_asks.append(float(bk[2][9]) if len(bk) > 2 and len(bk[2]) > 9 else 0.0)
            if p_p > 0.01 or c_p > 0.01: symbols_with_data.add(row.symbol)
        except:
            for arr in [put_prices, call_prices, put_ks, call_ks, put_ivs, call_ivs, put_bids, put_asks, call_bids, call_asks]: arr.append(0.0)
    packet = {
        'symbols': symbols_list, 'ts': float(ts_val), 'stock_price': group['close'].values.astype(np.float32), 'fast_vol': safe_col(group, 'fast_vol', 0.0).astype(np.float32), 'precalc_alpha': safe_col(group, 'alpha_score', 0.0).astype(np.float32), 'is_new_minute': is_new_min, 'symbols_with_data': symbols_with_data, 'feed_put_price': np.array(put_prices, dtype=np.float32), 'feed_call_price': np.array(call_prices, dtype=np.float32), 'feed_put_k': np.array(put_ks, dtype=np.float32), 'feed_call_k': np.array(call_ks, dtype=np.float32), 'feed_put_iv': np.array(put_ivs, dtype=np.float32), 'feed_call_iv': np.array(call_ivs, dtype=np.float32), 'feed_put_bid': np.array(put_bids, dtype=np.float32), 'feed_put_ask': np.array(put_asks, dtype=np.float32), 'feed_call_bid': np.array(call_bids, dtype=np.float32), 'feed_call_ask': np.array(call_asks, dtype=np.float32), 'feed_put_vol': np.ones(len(group), dtype=np.float32), 'feed_call_vol': np.ones(len(group), dtype=np.float32), 'feed_call_bid_size': np.full(len(group), 100.0, dtype=np.float32), 'feed_call_ask_size': np.full(len(group), 100.0, dtype=np.float32), 'feed_put_bid_size': np.full(len(group), 100.0, dtype=np.float32), 'feed_put_ask_size': np.full(len(group), 100.0, dtype=np.float32), 'slow_1m': np.zeros((len(symbols_list), 30, 1), dtype=np.float32), 'feed_put_id': [""] * len(group), 'feed_call_id': [""] * len(group), 'spy_roc_5min': safe_col(group, 'spy_roc_5min', 0.0), 'qqq_roc_5min': safe_col(group, 'qqq_roc_5min', 0.0)
    }
    return packet

async def run_single_engine(date_str, db_path, target_symbols):
    engine_symbols = list(set(target_symbols) | {'VIXY', 'SPY', 'QQQ'})
    orch = V8Orchestrator(symbols=engine_symbols, mode='backtest', config_paths={}, model_paths={})
    mock_ib = MockIBKRHistorical(db_path); await mock_ib.connect(); orch.ibkr = mock_ib
    df = load_official_stable_data(db_path, target_symbols); last_min = -1
    for ts_val, group in df.groupby('ts'):
        dt_ny = datetime.fromtimestamp(ts_val, tz=pytz.utc).astimezone(pytz.timezone('America/New_York'))
        time_str = dt_ny.strftime('%H:%M:%S')
        if time_str < "09:35:00" or time_str > "16:00:00": continue
        current_min = int(ts_val // 60); is_new_min = (current_min != last_min); last_min = current_min
        packet = build_stable_packet(ts_val, group, is_new_min)
        await orch.process_batch(packet)
        mock_ib.record_market_data(packet, alphas=packet['precalc_alpha'])
    return orch.realized_pnl, orch.trade_count

async def run_dual_engine(date_str, db_path, target_symbols):
    engine_symbols = list(set(target_symbols) | {'VIXY', 'SPY', 'QQQ'})
    se = SignalEngineV8(symbols=engine_symbols, mode='backtest', config_paths={}, model_paths={})
    ee = ExecutionEngineV8(symbols=engine_symbols, mode='backtest')
    for sym in engine_symbols:
        if sym in se.states and sym in ee.states: ee.states[sym] = se.states[sym]
    signal_queue = asyncio.Queue(); se.use_shared_mem = True; se.signal_queue = signal_queue; ee.use_shared_mem = True; ee.signal_queue = signal_queue
    mock_ib = MockIBKRHistorical(db_path); await mock_ib.connect(); ee.ibkr = mock_ib
    df = load_official_stable_data(db_path, target_symbols); last_min = -1
    for ts_val, group in df.groupby('ts'):
        dt_ny = datetime.fromtimestamp(ts_val, tz=pytz.utc).astimezone(pytz.timezone('America/New_York'))
        time_str = dt_ny.strftime('%H:%M:%S')
        if time_str < "09:35:00" or time_str > "16:00:00": continue
        current_min = int(ts_val // 60); is_new_min = (current_min != last_min); last_min = current_min
        packet = build_stable_packet(ts_val, group, is_new_min)
        symbols = packet['symbols']
        # 🛡️ 原子级循环：确保状态实时同步
        for i, sym in enumerate(symbols):
            single_packet = {k: v[i:i+1] if isinstance(v, (np.ndarray, list)) else v for k, v in packet.items()}
            single_packet['symbols'] = [sym]
            single_packet['symbols_with_data'] = {sym} if sym in packet['symbols_with_data'] else set()
            await se.process_batch(single_packet)
            while not signal_queue.empty():
                payload = signal_queue.get_nowait()
                await ee.process_trade_signal(payload)
                signal_queue.task_done()
        mock_ib.record_market_data(packet, alphas=packet['precalc_alpha'])
    return ee.realized_pnl, ee.trade_count

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dates", type=str, default="20260102")
    args = parser.parse_args(); date_list = args.dates.split(","); HIST_DIR = ROOT_DIR / "production" / "preprocess" / "backtest" / "history_sqlite_1m"
    from config import TARGET_SYMBOLS
    tradeable_symbols = [s for s in TARGET_SYMBOLS if s not in {'VIXY', 'SPY', 'QQQ'}]
    results = []
    for date_str in date_list:
        db_path = HIST_DIR / f"market_{date_str}.db"
        if not db_path.exists(): continue
        pnl_s, trades_s = await run_single_engine(date_str, str(db_path), tradeable_symbols)
        pnl_d, trades_d = await run_dual_engine(date_str, str(db_path), tradeable_symbols)
        diff = abs(pnl_s - pnl_d)
        is_match = (diff < 1.0 or trades_s == trades_d)
        results.append({'date': date_str, 'single_pnl': pnl_s, 'single_trades': trades_s, 'dual_pnl': pnl_d, 'dual_trades': trades_d, 'match': is_match})
    logger.info("="*80); logger.info(f"📊 FINAL ULTIMATE V0 PARITY REPORT"); logger.info("="*80); logger.info(f"{'Date':<10} | {'Single PnL':<12} | {'Dual PnL':<12} | {'S-Trd':<6} | {'D-Trd':<6} | {'Match'}"); logger.info("-" * 80)
    for r in results:
        m_str = "✅" if r['match'] else "❌"
        logger.info(f"{r['date']:<10} | ${r['single_pnl']:>10.2f} | ${r['dual_pnl']:>10.2f} | {r['single_trades']:>6} | {r['dual_trades']:>6} | {m_str}")
    logger.info("="*80)
    
if __name__ == "__main__":
    asyncio.run(main())

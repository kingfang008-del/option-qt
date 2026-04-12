#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

os.environ['RUN_MODE'] = 'BACKTEST'
os.environ['DUAL_CONVERGE_TO_SINGLE'] = '1'
os.environ['PURE_ALPHA_REPLAY'] = '1'

import asyncio
import time
import logging
from pathlib import Path
import argparse
import sys
import pandas as pd
import psycopg2
import json
import numpy as np
from datetime import datetime
import pytz
from tqdm import tqdm

 

from signal_engine_v8 import SignalEngineV8
from execution_engine_v8 import ExecutionEngineV8
import mock_ibkr_historical_1s
from mock_ibkr_historical_1s import MockIBKRHistorical
import config
import signal_engine_v8
import execution_engine_v8
from config import PG_DB_URL

print(f"🔍 [Import Trace] Using MockIBKR from: {mock_ibkr_historical_1s.__file__}")

BACKTEST_STREAM = "backtest_stream"
BACKTEST_GROUP = "backtest_group"

config.REDIS_CFG['input_stream'] = BACKTEST_STREAM
config.REDIS_CFG['orch_group'] = BACKTEST_GROUP
signal_engine_v8.REDIS_CFG['input_stream'] = BACKTEST_STREAM
signal_engine_v8.REDIS_CFG['orch_group'] = BACKTEST_GROUP
execution_engine_v8.REDIS_CFG['input_stream'] = BACKTEST_STREAM
execution_engine_v8.REDIS_CFG['orch_group'] = BACKTEST_GROUP

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [REPLAY_S2_PG_1S] - %(message)s')
logger = logging.getLogger("ReplayS2PG1S")
ALPHA_AVAILABLE_DELAY_SECONDS = 60.0


def partition_exists(cursor, table_name):
    cursor.execute(
        "SELECT EXISTS(SELECT 1 FROM information_schema.tables WHERE table_name=%s)",
        (table_name,),
    )
    return bool(cursor.fetchone()[0])


def safe_col(group, col, default_val, dtype=np.float32):
    if col in group.columns:
        return group[col].fillna(default_val).values.astype(dtype)
    return np.full(len(group), default_val, dtype=dtype)


def build_option_arrays(group):
    put_prices, call_prices = [], []
    put_ks, call_ks = [], []
    put_ivs, call_ivs = [], []
    put_bids, put_asks = [], []
    call_bids, call_asks = [], []
    symbols_with_data = set()

    for row in group.itertuples():
        try:
            bk = row.buckets_json
            if isinstance(bk, str):
                bk = json.loads(bk)
            if isinstance(bk, dict):
                bk = bk.get('buckets', [])

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
        except Exception:
            for arr in [put_prices, call_prices, put_ks, call_ks, put_ivs, call_ivs, put_bids, put_asks, call_bids, call_asks]:
                arr.append(0.0)

    return {
        'put_prices': np.array(put_prices, dtype=np.float32),
        'call_prices': np.array(call_prices, dtype=np.float32),
        'put_ks': np.array(put_ks, dtype=np.float32),
        'call_ks': np.array(call_ks, dtype=np.float32),
        'put_ivs': np.array(put_ivs, dtype=np.float32),
        'call_ivs': np.array(call_ivs, dtype=np.float32),
        'put_bids': np.array(put_bids, dtype=np.float32),
        'put_asks': np.array(put_asks, dtype=np.float32),
        'call_bids': np.array(call_bids, dtype=np.float32),
        'call_asks': np.array(call_asks, dtype=np.float32),
        'symbols_with_data': symbols_with_data,
    }


async def main():
    print(f"!!! EXECUTING S2 PG 1S REPLAY SCRIPT: {__file__} !!!")
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, default="20260102")
    parser.add_argument("--symbol", type=str, default=None)
    args = parser.parse_args()

    date_str = args.date
    part_alpha = f"alpha_logs_{date_str}"
    part_bars = f"market_bars_1s_{date_str}"
    part_opts = f"option_snapshots_1s_{date_str}"

    conn = psycopg2.connect(PG_DB_URL)
    cur = conn.cursor()
    for table_name in [part_alpha, part_bars, part_opts]:
        if not partition_exists(cur, table_name):
            logger.error(f"❌ PostgreSQL partition missing: {table_name}")
            cur.close()
            conn.close()
            return

    logger.info(f"📂 Loading data directly from PostgreSQL partitions: {part_alpha}, {part_bars}, {part_opts}")

     
    config_paths = {
            'fast': str("/home/kingfang007/notebook/train/fast_feature.json"),
            'slow': str("/home/kingfang007/notebook/train/slow_feature.json")
        }
    from config import TARGET_SYMBOLS
    symbols = [args.symbol] if args.symbol else TARGET_SYMBOLS

    logger.info("🛠️ Building S2 dual engine (PG 1s execution path)...")
    signal_engine = SignalEngineV8(
        symbols=symbols,
        mode='backtest',
        config_paths=config_paths,
        model_paths={}
    )
    logger.info(
        "🧭 [Strategy Audit] core=%s cfg=%s INDEX_GUARD_ENABLED=%s INDEX_REVERSAL_EXIT_ENABLED=%s",
        signal_engine.strategy.__class__.__name__,
        signal_engine.strategy.cfg.__class__.__module__,
        getattr(signal_engine.strategy.cfg, 'INDEX_GUARD_ENABLED', None),
        getattr(signal_engine.strategy.cfg, 'INDEX_REVERSAL_EXIT_ENABLED', None),
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

    logger.info("🔌 Injecting Mock IBKR 1s...")
    mock_ibkr = MockIBKRHistorical()
    await mock_ibkr.connect()
    mock_ibkr.r.flushdb()

    signal_engine.ibkr = mock_ibkr
    exec_engine.ibkr = mock_ibkr
    signal_engine.mock_cash = mock_ibkr.initial_capital
    exec_engine.mock_cash = mock_ibkr.initial_capital

    df_a = pd.read_sql(f"SELECT ts, symbol, alpha as alpha_score, vol_z as fast_vol FROM {part_alpha}", conn)
    df_s = pd.read_sql(f"SELECT ts, symbol, close, open, high, low, volume FROM {part_bars}", conn)
    df_o = pd.read_sql(f"SELECT ts, symbol, buckets_json FROM {part_opts}", conn)
    cur.close()
    conn.close()

    df_s['ts'] = df_s['ts'].astype(float)
    df_idx = df_s[df_s['symbol'].isin(['SPY', 'QQQ'])].pivot(index='ts', columns='symbol', values='close')

    unique_ts = sorted(df_s['ts'].unique())
    df_roc_map = pd.DataFrame(index=unique_ts)
    for col in ['SPY', 'QQQ']:
        if col in df_idx.columns:
            df_roc_map[f'{col.lower()}_roc_5min'] = df_idx[col].pct_change(periods=300).fillna(0.0)
        else:
            df_roc_map[f'{col.lower()}_roc_5min'] = 0.0
    df_s = df_s.merge(df_roc_map.reset_index().rename(columns={'index': 'ts'}), on='ts', how='left')

    if symbols:
        df_a = df_a[df_a['symbol'].isin(symbols)]
        df_s = df_s[df_s['symbol'].isin(symbols)]
        df_o = df_o[df_o['symbol'].isin(symbols)]

    for df_tmp in [df_a, df_s, df_o]:
        df_tmp['ts'] = df_tmp['ts'].astype(float)
        df_tmp['symbol'] = df_tmp['symbol'].astype(str)

    df_a['alpha_label_ts'] = df_a['ts']
    df_a['ts'] = df_a['ts'] + ALPHA_AVAILABLE_DELAY_SECONDS
    df_a = df_a.sort_values('ts')
    df_s = df_s.sort_values('ts')
    df_o = df_o.sort_values('ts')

    df_market = pd.merge_asof(df_s, df_o, on='ts', by='symbol', direction='backward', tolerance=2)
    df = pd.merge_asof(df_market, df_a, on='ts', by='symbol', direction='backward', tolerance=120)
    if df.empty:
        logger.error("❌ Merged dataset is EMPTY! Check PG partitions.")
        return

    grouped = df.sort_values('ts', ascending=True).groupby('ts')
    unique_groups = df['ts'].nunique()

    logger.info(f"🚀 Starting S2 PG 1s replay bus ({unique_groups} ticks)...")
    start_time = time.time()
    ny_tz = pytz.timezone('America/New_York')
    last_minute = -1

    for ts_val, group in tqdm(grouped, desc="Replaying", total=unique_groups):
        dt_ny = datetime.fromtimestamp(ts_val, tz=pytz.utc).astimezone(ny_tz)
        curr_time_str = dt_ny.strftime('%H:%M:%S')
        if curr_time_str < "09:35:00" or curr_time_str > "16:00:00":
            continue

        current_minute = int(ts_val // 60)
        is_new_minute = (last_minute != current_minute)
        last_minute = current_minute

        symbols_list = group['symbol'].tolist()
        opt = build_option_arrays(group)
        packet = {
            'symbols': symbols_list,
            'ts': float(ts_val),
            'stock_price': group['close'].values.astype(np.float32),
            'fast_vol': safe_col(group, 'fast_vol', 0.0),
            'precalc_alpha': safe_col(group, 'alpha_score', 0.0),
            'alpha_label_ts': safe_col(group, 'alpha_label_ts', 0.0, dtype=np.float64),
            'alpha_available_ts': np.full(len(group), float(ts_val), dtype=np.float64),
            'spy_roc_5min': safe_col(group, 'spy_roc_5min', 0.0),
            'qqq_roc_5min': safe_col(group, 'qqq_roc_5min', 0.0),
            'is_new_minute': is_new_minute,
            'symbols_with_data': opt['symbols_with_data'],
            'feed_put_price': opt['put_prices'],
            'feed_call_price': opt['call_prices'],
            'feed_put_k': opt['put_ks'],
            'feed_call_k': opt['call_ks'],
            'feed_put_iv': opt['put_ivs'],
            'feed_call_iv': opt['call_ivs'],
            'feed_put_bid': opt['put_bids'],
            'feed_put_ask': opt['put_asks'],
            'feed_call_bid': opt['call_bids'],
            'feed_call_ask': opt['call_asks'],
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
                logger.info(f"📡 [REPLAY_S2_PG_1S] Minute {current_minute} | Active Signals: {sig_count}")

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

    elapsed = time.time() - start_time
    print(f"\n✅ S2 PG 1S Replay Finished in {elapsed:.1f}s.")
    await exec_engine.force_close_all()
    await asyncio.sleep(0.5)
    mock_ibkr.save_trades(filename="replay_trades_s2_pg_1s.csv")

    print("\n" + "=" * 50)
    print("📊 FINAL BACKTEST PERFORMANCE SUMMARY (S2 PG 1S DUAL)")
    print("=" * 50)
    print("ℹ️ 最终收益统计以 MockIBKR 生成的成交账本为准（已包含 execution delay / FIFO / 手续费 / chunk 撮合）。")
    print("ℹ️ ExecutionEngine 内部 realized_pnl 口径与 Mock 重撮合价不同，这里不再重复打印，避免双账本混淆。")
    exec_engine.accounting.print_counter_trend_summary()
    print("=" * 50 + "\n")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass

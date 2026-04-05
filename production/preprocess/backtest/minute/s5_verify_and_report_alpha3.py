import os
# 🚨 绝对第一优先级！强行把整个进程及其子模块按倒在 BACKTEST 模式！
os.environ['RUN_MODE'] = 'BACKTEST'

import asyncio
import time
import logging
import json
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import sys
import copy

# 设置项目根目录
MY_DIR = Path(__file__).resolve().parent
ROOT_DIR = MY_DIR.parent.parent.parent.parent 
sys.path.insert(0, str(ROOT_DIR / "production" / "baseline"))
sys.path.insert(0, str(ROOT_DIR / "production" / "history_replay"))

from signal_engine_v8 import SignalEngineV8
from execution_engine_v8 import ExecutionEngineV8
from mock_ibkr_historical import MockIBKRHistorical
import config
from tqdm import tqdm

def resolve_db_path(date):
    """根据日期寻找对应的 market_*.db 文件"""
    hist_dir = MY_DIR.parent / "history_sqlite_1m"
    db_name = f"market_{date}.db"
    p = hist_dir / db_name
    return p if p.exists() else None

def safe_col(group, col, default_val, dtype=np.float32):
    if col in group.columns:
        return group[col].fillna(default_val).values.astype(dtype)
    return np.full(len(group), default_val, dtype=dtype)

async def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, default="20260106")
    args = parser.parse_args()

    db_path = resolve_db_path(args.date)
    if not db_path:
        print(f"❌ Target Database not found for {args.date}")
        return

    print(f"🚀 [Alpha 3.0 对账系统] 启动 | 目标日期: {args.date}")
    print(f"📂 数据源: {db_path}")

    # 1. 注入 Mock IBKR (强制保留 0.65 手续费，关闭时延)
    mock_ibkr = MockIBKRHistorical()
    mock_ibkr.execution_delay_seconds = 0 
    mock_ibkr.fill_delay = 0
    # 强制同步配置
    from config import COMMISSION_PER_CONTRACT, TARGET_SYMBOLS
    print(f"💰 对账手续费设定: ${COMMISSION_PER_CONTRACT}/手")

    # 2. 初始化双引擎 (共享内存确定性模式)
    V8_ROOT = Path(__file__).parent.parent
    config_paths = {
        'fast': str(V8_ROOT/"daily_backtest/fast_feature.json"), 
        'slow': str(V8_ROOT/"daily_backtest/slow_feature.json")
    }

    signal_engine = SignalEngineV8(symbols=TARGET_SYMBOLS, mode='backtest', config_paths=config_paths)
    shared_signal_queue = asyncio.Queue()
    signal_engine.signal_queue = shared_signal_queue
    signal_engine.use_shared_mem = True
    
    exec_engine = ExecutionEngineV8(
        symbols=TARGET_SYMBOLS, mode='backtest',
        shared_states=signal_engine.states,
        signal_queue=shared_signal_queue
    )
    exec_engine.ibkr = mock_ibkr
    signal_engine.ibkr = mock_ibkr
    exec_engine.use_shared_mem = True

    # 3. 加载并对齐数据
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    df_a = pd.read_sql("SELECT ts, symbol, alpha as alpha_score, vol_z as fast_vol, event_prob FROM alpha_logs", conn)
    df_s = pd.read_sql("SELECT ts, symbol, close, open, high, low, volume, spy_roc_5min, qqq_roc_5min FROM market_bars_1m", conn)
    df_o = pd.read_sql("SELECT ts, symbol, buckets_json FROM option_snapshots_1m", conn)
    conn.close()

    for df_tmp in [df_a, df_s, df_o]:
        df_tmp['ts'] = df_tmp['ts'].astype(float)
        df_tmp['symbol'] = df_tmp['symbol'].astype(str)
        
    df_a = df_a.sort_values('ts')
    df_s = df_s.sort_values('ts')
    df_o = df_o.sort_values('ts')

    df_market = pd.merge_asof(df_s, df_o, on='ts', by='symbol', direction='backward', tolerance=120)
    df = pd.merge_asof(df_market, df_a, on='ts', by='symbol', direction='backward', tolerance=120)
    
    grouped = df.sort_values('ts', ascending=True).groupby('ts')
    unique_groups = df['ts'].nunique()

    # 4. 回放主循环
    import pytz
    ny_tz = pytz.timezone('America/New_York')
    last_minute = -1

    for ts_val, group in tqdm(grouped, desc="Verification Replay", total=unique_groups):
        dt_ny = datetime.fromtimestamp(ts_val, pytz.utc).astimezone(ny_tz)
        curr_time_str = dt_ny.strftime('%H:%M:%S')
        if curr_time_str < "09:35:00" or curr_time_str > "16:00:00": continue

        current_minute = int(ts_val // 60)
        is_new_minute = (last_minute != current_minute)
        last_minute = current_minute
        
        symbols_list = group['symbol'].tolist()
        put_prices, call_prices = [], []
        put_bids, put_asks, call_bids, call_asks = [], [], [], []
        symbols_with_data = set()
        
        for idx_row, row in enumerate(group.itertuples()):
            try:
                bk = json.loads(row.buckets_json).get('buckets', [])
                p_p = float(bk[0][0]) if len(bk) > 0 and len(bk[0]) > 0 else 0.0
                c_p = float(bk[2][0]) if len(bk) > 2 and len(bk[2]) > 0 else 0.0
                put_prices.append(p_p)
                call_prices.append(c_p)
                put_bids.append(float(bk[0][8]) if len(bk) > 0 and len(bk[0]) > 8 else 0.0)
                put_asks.append(float(bk[0][9]) if len(bk) > 0 and len(bk[0]) > 9 else 0.0)
                call_bids.append(float(bk[2][8]) if len(bk) > 2 and len(bk[2]) > 8 else 0.0)
                call_asks.append(float(bk[2][9]) if len(bk) > 2 and len(bk[2]) > 9 else 0.0)
                if p_p > 0.01 or c_p > 0.01: symbols_with_data.add(row.symbol)
            except:
                for arr in [put_prices, call_prices, put_bids, put_asks, call_bids, call_asks]: arr.append(0.0)

        packet = {
            'symbols': symbols_list,
            'ts': float(ts_val),
            'stock_price': group['close'].values.astype(np.float32),
            'fast_vol': group['fast_vol'].values.astype(np.float32),
            'precalc_alpha': group['alpha_score'].values.astype(np.float32),
            'event_prob': safe_col(group, 'event_prob', 0.0),
            'spy_roc_5min': safe_col(group, 'spy_roc_5min', 0.0),
            'qqq_roc_5min': safe_col(group, 'qqq_roc_5min', 0.0),
            'is_new_minute': is_new_minute,
            'symbols_with_data': symbols_with_data,
            'feed_put_price': np.array(put_prices, dtype=np.float32),
            'feed_call_price': np.array(call_prices, dtype=np.float32),
            'feed_put_bid': np.array(put_bids, dtype=np.float32),
            'feed_put_ask': np.array(put_asks, dtype=np.float32),
            'feed_call_bid': np.array(call_bids, dtype=np.float32),
            'feed_call_ask_size': np.full(len(group), 100.0, dtype=np.float32),
            'feed_call_ask': np.array(call_asks, dtype=np.float32),
        }

        # 信号引擎
        await signal_engine.process_batch(packet)
        mock_ibkr.record_market_data(packet, alphas=packet['precalc_alpha'])
        
        # 排空 OMS 队列 (同步模式保证结果确定性)
        while not shared_signal_queue.empty():
            try:
                sig_payload = shared_signal_queue.get_nowait()
                await exec_engine.process_trade_signal(sig_payload)
                shared_signal_queue.task_done()
            except Exception: break
            
        # 结尾 SYNC
        await exec_engine.process_trade_signal({'action': 'SYNC', 'ts': ts_val, 'payload': {}})

    # 5. 生成对账报表
    print(f"\n📊 对账结束，正在落地物理对账单...")
    await exec_engine.force_close_all()
    mock_ibkr.save_trades(filename=f"verification_trades_{args.date}.csv")
    
    print("\n" + "="*50)
    print("📈 FINAL VERIFICATION SUMMARY (0 LATENCY)")
    print("-" * 50)
    print(f"累计实现盈亏 (Realized): ${exec_engine.realized_pnl:,.2f}")
    print(f"总手续费支出: ${exec_engine.total_commission:,.2f}")
    print(f"总成交笔数: {exec_engine.trade_count}")
    print("="*50)

if __name__ == "__main__":
    asyncio.run(main())

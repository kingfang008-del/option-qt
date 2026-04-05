#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
🚀 确定性总线回测 (Deterministic Bus Replay)
核心思想: 不改 SE/OMS 内部任何代码，仅靠启动脚本强制串行步进。
利用 asyncio 单线程的特性，让 SE 的 `signal_queue.join()` 自然阻塞，
由后台 consumer task 同步消费信号，实现 SE → OMS 的确定性交替执行。
"""

import os
# 🚨 绝对第一优先级！强行把整个进程及其子模块按倒在 BACKTEST 模式！
os.environ['RUN_MODE'] = 'BACKTEST'

import asyncio
import time
import logging
from datetime import datetime
from pathlib import Path
import argparse 
import sys
import copy
import numpy as np
import pandas as pd
import sqlite3

# 🚨 动态添加路径，确保 baseline 目录下的模块 (SignalEngineV8 等) 能被识别
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT_DIR / "baseline"))
sys.path.append(str(ROOT_DIR / "history_replay"))

from signal_engine_v8 import SignalEngineV8
from execution_engine_v8 import ExecutionEngineV8
from mock_ibkr_historical import MockIBKRHistorical
from tqdm import tqdm

import config
BACKTEST_STREAM = "backtest_stream"
BACKTEST_GROUP  = "backtest_group"
config.REDIS_CFG['orch_group'] = BACKTEST_GROUP

import signal_engine_v8
import execution_engine_v8

# 断开 Python 字典的内存引用共享
signal_engine_v8.REDIS_CFG = copy.deepcopy(config.REDIS_CFG)
signal_engine_v8.REDIS_CFG['input_stream'] = BACKTEST_STREAM
execution_engine_v8.REDIS_CFG = copy.deepcopy(config.REDIS_CFG)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ReplayBus")


def resolve_db_path(args):
    """根据日期寻找对应的 market_*.db 文件"""
    BT_DIR = Path(__file__).resolve().parent
    hist_dir = BT_DIR / "history_sqlite_1m"
    
    if args.date:
        db_name = f"market_{args.date}.db"
        p1 = hist_dir / db_name
        if p1.exists(): return p1
        p2 = BT_DIR / db_name
        if p2.exists(): return p2
        return None
    
    all_dbs = sorted(hist_dir.glob("market_*.db"))
    return all_dbs[-1] if all_dbs else None


def safe_col(group, col, default_val, dtype=np.float32):
    """安全提取 DataFrame 列，缺失时返回默认值数组"""
    if col in group.columns:
        return group[col].fillna(default_val).values.astype(dtype)
    return np.full(len(group), default_val, dtype=dtype)


async def main():
    print(f"!!! EXECUTING DETERMINISTIC BUS SCRIPT: {__file__} !!!")
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, default=None)
    parser.add_argument("--parquet-dir", type=str, default=None) # Keep for compatibility but unused
    parser.add_argument("--symbol", type=str, default=None) # Allow single symbol debug
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
    if args.symbol:
        symbols = [args.symbol]
    else:
        symbols = TARGET_SYMBOLS    
    logger.info("🛠️ Building Deterministic Backtest Bus...") 
    
    # ==========================================
    # 1. 初始化双引擎 (共享内存模式)
    # ==========================================
    signal_engine = SignalEngineV8(
        symbols=symbols, mode='backtest', 
        config_paths=config_paths, model_paths={}
    )
    shared_signal_queue = asyncio.Queue()
    signal_engine.signal_queue = shared_signal_queue
    signal_engine.use_shared_mem = True
    
    # ==========================================
    # 1.5 [核心] PARITY_MODE: 注入 plan_a 对齐配置
    # ==========================================
    if os.environ.get('PARITY_MODE') == 'PLAN_A':
        logger.warning("🔧 [PARITY_MODE] 启用 plan_a 对齐模式 — 关闭所有 s4 独有 Guard!")
        pcfg = signal_engine.strategy.cfg
        # 关闭 s4 独有的入场 Guard
        pcfg.ENTRY_MOMENTUM_GUARD_ENABLED = False
        pcfg.ENTRY_LIQUIDITY_GUARD_ENABLED = False
        pcfg.MACD_HIST_CONFIRM_ENABLED = False
        pcfg.INDEX_GUARD_ENABLED = False
        # 关闭 s4 独有的平仓 Guard
        pcfg.EXIT_COUNTER_TREND_ENABLED = False
        pcfg.EXIT_INDEX_REVERSAL_ENABLED = False
        pcfg.EXIT_STOCK_HARD_STOP_ENABLED = False
        pcfg.EXIT_ZOMBIE_STOP_ENABLED = False
        pcfg.EXIT_MACD_FADE_ENABLED = False
        pcfg.EXIT_SIGNAL_FLIP_ENABLED = False
        pcfg.EXIT_LIQUIDITY_GUARD_ENABLED = False
        pcfg.EXIT_COND_STOP_ENABLED = False
        pcfg.EXIT_SMALL_GAIN_ENABLED = False
        pcfg.INDEX_REVERSAL_EXIT_ENABLED = False
        # 开启 plan_a 独有的平仓规则
        pcfg.EXIT_EARLY_STOP_ENABLED = True
        pcfg.EXIT_NO_MOMENTUM_ENABLED = True
        pcfg.PARITY_STRICT_MODE = True
        # 🚀 [核心对齐] 关闭所有 entry guard，因为 Plan A 没有任何过滤
        pcfg.INDEX_GUARD_ENABLED = False
        pcfg.ENTRY_MOMENTUM_GUARD_ENABLED = False
        pcfg.ENTRY_LIQUIDITY_GUARD_ENABLED = False
        pcfg.MACD_HIST_CONFIRM_ENABLED = False
        # 强制对齐阈值
        pcfg.ALPHA_ENTRY_STRICT = 1.45
        pcfg.ALPHA_ENTRY_THRESHOLD = 1.45
        # 🚀 [核心对齐] Plan A 采用 "一分钟入门后再查票" 模式。
        # 这里设为 4，SignalEngine 会在 Batch 开始前检查 < 4，
        # 如果通过，则该分钟（如 09:35）哪怕有 5 个信号也会全部发出。
        pcfg.MAX_POSITIONS = 4
        pcfg.COMMISSION_PER_CONTRACT = 0.0 # Plan A 不计手续费
        logger.warning(f"🔧 [PARITY_MODE] Alpha阈值: {pcfg.ALPHA_ENTRY_STRICT}, MAX_POSITIONS: {pcfg.MAX_POSITIONS}")

    exec_engine = ExecutionEngineV8(
        symbols=symbols, mode='backtest',
        shared_states=signal_engine.states,
        signal_queue=shared_signal_queue
    )
    # 🚀 [核心修复] 强制同步配置对象，确保 exec_engine 也能看到 PARITY_MODE 和修正后的 MAX_POSITIONS
    exec_engine.strategy.cfg = signal_engine.strategy.cfg
    exec_engine.cfg = signal_engine.strategy.cfg
    
    exec_engine.use_shared_mem = True

    # ==========================================
    # 2. 注入 Mock IBKR & 清理 Redis
    # ==========================================
    logger.info(f"🔌 Injecting Mock IBKR...")
    mock_ibkr = MockIBKRHistorical()
    await mock_ibkr.connect()
    
    # 清空回测 DB，消除幽灵状态
    logger.info("💥 Nuking Redis Backtest Database (DB 1)...")
    mock_ibkr.r.flushdb()
    mock_ibkr.r.delete("orch_trade_signals")
    mock_ibkr.r.delete("sync:orch_done")
    
    signal_engine.ibkr = mock_ibkr
    exec_engine.ibkr = mock_ibkr
    signal_engine.mock_cash = mock_ibkr.initial_capital
    exec_engine.mock_cash = mock_ibkr.initial_capital
    
    # ==========================================
    # 3. [PARITY FIX] 关闭异步 Consumer，改用同步排空模式
    # ==========================================
    # 在 Parity 模式下，我们通过在主循环中同步调用 exec_engine.process_trade_signal 来保证 100% 确定性。
    # 这一步不需要启动任何后台任务。
    pass

    # ==========================================
    # 4. 直接从 SQLite 加载并对齐数据 (对标 plan_a_fine_tune.py)
    # ==========================================
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    df_a = pd.read_sql(f"SELECT ts, symbol, alpha as alpha_score, vol_z as fast_vol FROM alpha_logs", conn)
    df_s = pd.read_sql("SELECT ts, symbol, close, open, high, low, volume FROM market_bars_1m", conn)
    df_o = pd.read_sql("SELECT ts, symbol, buckets_json FROM option_snapshots_1m", conn)
    conn.close()

    if symbols:
        df_a = df_a[df_a['symbol'].isin(symbols)]
        df_s = df_s[df_s['symbol'].isin(symbols)]
        df_o = df_o[df_o['symbol'].isin(symbols)]

    is_parity = (os.environ.get('PARITY_MODE') == 'PLAN_A')
    
    if is_parity:
        # 🚀 [PARITY] 100% 对齐 plan_a_fine_tune.py 的数据对齐逻辑：
        # 必须是 Alpha, Market, Option 三者同时存在的 Tick 才会参与计算。
        # 严禁使用 reindex 或 ffill 填补空洞，否则会造成“虚假平仓”和“虚假入场”。
        df = pd.merge(df_a, df_s, on=['ts', 'symbol'], how='inner')
        df = pd.merge(df, df_o, on=['ts', 'symbol'], how='inner')
        logger.info(f"📊 Parity Inner Join Complete: {len(df)} rows across {df['ts'].nunique()} ticks")
    else:
        # 🚀 生产/高保真模式 (仍使用 Inner Join，但基础集不同)
        df = pd.merge(df_a, df_s, on=['ts', 'symbol'], how='inner')
        df = pd.merge(df, df_o, on=['ts', 'symbol'], how='inner')
    
    if df.empty:
        logger.error("❌ Merged dataset is EMPTY! Check DB integrity.")
        consumer_running = False
        consumer_task.cancel()
        return

    grouped = df.sort_values('ts').groupby('ts')
    unique_groups = df['ts'].nunique()
    
    logger.info(f"🚀 Starting Deterministic Bus ({unique_groups} ticks)...")
    start_time = time.time()
    
    # ==========================================
    # 5. 确定性总线主循环
    # ==========================================
    import pytz
    ny_tz = pytz.timezone('America/New_York')
    
    last_minute = -1
    parity_mode = (os.environ.get('PARITY_MODE') == 'PLAN_A')
    for ts_val, group in tqdm(grouped, desc="Replaying", total=unique_groups):
        dt_ny = datetime.fromtimestamp(ts_val, tz=pytz.utc).astimezone(ny_tz)
        curr_time_str = dt_ny.strftime('%H:%M:%S')
        
        # 🚀 [核心限制] 与 plan_a 保持一致，仅在 09:35 - 16:00 交易
        if curr_time_str < "09:35:00" or curr_time_str > "16:00:00":
            continue

        current_minute = int(ts_val // 60)
        is_new_minute = (last_minute != current_minute)
        last_minute = current_minute
        
        symbols_list = group['symbol'].tolist()
        
        # 🧪 [Parity Fix] 实时从 buckets_json 提取必要的列，并确定有效数据集合
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

        # 组装 Packet
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
            sig_count = (packet['precalc_alpha'] != 0).sum()
            if sig_count > 0:
                logger.info(f"📡 [REPLAY_BUS] Minute {current_minute} | Active Signals: {sig_count}")

        await signal_engine.process_batch(packet)
        # 🧪 [Fix] 显式调用 record_market_data，确保 MockIBKR 能在 EOD 匹配到离散价格
        mock_ibkr.record_market_data(packet, alphas=packet['precalc_alpha'])
        
        # 🚀 [核心修复] 同步排空OMS队列，确保 st.position 在下一tick前被更新
        # plan_a 是同步循环，开仓/平仓立即生效。s4 的异步OMS必须在此处同步等待。
        while not shared_signal_queue.empty():
            try:
                sig_payload = shared_signal_queue.get_nowait()
                await exec_engine.process_trade_signal(sig_payload)
                shared_signal_queue.task_done()
            except Exception:
                break
        
        # 🚀 [PARITY_MODE 关键修复] 直接对持仓标的进行退出扫描
        # 问题: 当 re-indexed 空行 price=0 时，_prep_symbol_metrics 返回 None，
        #        跳过了持仓标的的 exit 评估，导致所有持仓持有到 EOD。
        # 修复: 在 replay loop 层直接扫描所有持仓，调用 check_exit，
        #        使用上一次成功更新的 last_opt_price 作为当前价。
        # 🚀 [PARITY_MODE] 信号引擎已经内置了 _process_exits 的秒级扫描，且现在已经具备了 gap-skip 能力。
        # 此处不再需要手动扫描 states，防止 ctx 参数不全导致平仓逻辑失效或偏差。
        pass
        
        signal_engine.mock_cash = exec_engine.mock_cash

    # ==========================================
    # 6. 回放结束 & 汇总
    # ==========================================
    elapsed = time.time() - start_time
    print(f"\n✅ Deterministic Replay Finished in {elapsed:.1f}s.")
    
    await exec_engine.force_close_all()
    await asyncio.sleep(0.5)
    
    print("\n" + "="*50)
    print("📊 FINAL BACKTEST PERFORMANCE SUMMARY (V8 BUS-MODE)")
    print("="*50)
    exec_engine.accounting.print_backtest_summary()
    exec_engine.accounting.print_counter_trend_summary()
    print("="*50 + "\n")
    
    mock_ibkr.save_trades() 

if __name__ == "__main__":
    import json
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
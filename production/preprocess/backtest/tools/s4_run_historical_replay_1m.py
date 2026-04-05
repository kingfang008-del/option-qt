#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
# 🚨 绝对第一优先级！在所有 import 之前，强行把整个进程及其子模块按倒在 BACKTEST 模式！
os.environ['RUN_MODE'] = 'BACKTEST'

import asyncio
import threading
import time
import logging
from pathlib import Path
import argparse 
import sys
import copy  # [🔥 引入深拷贝]

# 导入 V8 引擎的双分支组件
from signal_engine_v8 import SignalEngineV8
from execution_engine_v8 import ExecutionEngineV8
# 导入 Driver 和 Mock IBKR
from replay_driver_s5_parquet import S5ParquetDriver
from mock_ibkr_historical import MockIBKRHistorical

import config

BACKTEST_STREAM = "backtest_stream"
BACKTEST_GROUP  = "backtest_group"

# 统一所有可能用到的配置源
config.REDIS_CFG['orch_group'] = BACKTEST_GROUP

import signal_engine_v8
import execution_engine_v8

# 断开 Python 字典的内存引用共享
signal_engine_v8.REDIS_CFG = copy.deepcopy(config.REDIS_CFG)
signal_engine_v8.REDIS_CFG['input_stream'] = BACKTEST_STREAM
signal_engine_v8.STREAM = BACKTEST_STREAM            # 👈 强行覆盖模块全局变量
signal_engine_v8.GROUP = BACKTEST_GROUP              # 👈 强行覆盖模块全局变量

execution_engine_v8.REDIS_CFG = copy.deepcopy(config.REDIS_CFG)
execution_engine_v8.REDIS_CFG['input_stream'] = "orch_trade_signals"
execution_engine_v8.STREAM = "orch_trade_signals"    # 👈 确保 OMS 绝对不会去听行情流
execution_engine_v8.GROUP = BACKTEST_GROUP

# ================= 配置区域 =================
BASE_PROJECT = Path.home() / "quant_project"
DEFAULT_OFFLINE_DIR = Path(f"/home/kingfang007/quant_project/data/rl_feed_parquet_batch")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [REPLAY_MAIN] - %(message)s')
logger = logging.getLogger("ReplayMain")

def resolve_parquet_path(args):
    if args.parquet_dir:
        p = Path(args.parquet_dir)
        if not p.exists(): return None
        return p
    if args.date:
        p = Path.home() / f"quant_project/backtest/{args.date}/rl_feed_parquet_new_delta_backtest"
        if not p.exists(): return None
        return p
    if not DEFAULT_OFFLINE_DIR.exists(): return None
    return DEFAULT_OFFLINE_DIR

async def main():
    print(f"!!! EXECUTING UPDATED SCRIPT: {__file__} !!!")
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, default=None)
    parser.add_argument("--parquet-dir", type=str, default=None)
    args = parser.parse_args()
    
    parquet_path = resolve_parquet_path(args)
    if parquet_path is None: return
    
    V8_ROOT = Path(__file__).parent.parent
    config_paths = {
        'fast': str(V8_ROOT/"daily_backtest/fast_feature.json"), 
        'slow': str(V8_ROOT/"daily_backtest/slow_feature.json")
    }

    symbols =  [ 
        'NVDA', 'AAPL', 'META', 'PLTR', 'TSLA', 'UNH', 'AMZN', 'AMD', 
        'NFLX', 'CRWV', 'AVGO', 'MSFT', 'HOOD', 'MU', 'APP', 'GOOGL', 
        'SMCI', 'ADBE', 'CRM', 'ORCL', 'NKE', 'XOM', 'TSM', 
        'SPY', 'QQQ', 'IWM',
    ]

    logger.info(f"🛡️ Isolated Backtest Stream: {BACKTEST_STREAM} | Group: {BACKTEST_GROUP} | DB: 1")
    
    logger.info("🛠️ Building V8 Signal & Execution Engines...") 
    
    # 1. 正常初始化信号引擎 (SE)
    signal_engine = SignalEngineV8(
        symbols=symbols, 
        mode='backtest', 
        config_paths=config_paths, 
        model_paths={} 
    )
    
    # 🚀 [新增架构开关] 创建内存无锁队列，并打开共享内存模式
    import asyncio
    shared_signal_queue = asyncio.Queue()
    signal_engine.signal_queue = shared_signal_queue
    signal_engine.use_shared_mem = True
    
    # 2. 初始化执行引擎 (OMS)，并将 SE 的内存指针和队列强行注入！
    exec_engine = ExecutionEngineV8(
        symbols=symbols, 
        mode='backtest',
        shared_states=signal_engine.states, # 👈 物理内存打通：OMS 和 SE 共享同一个状态字典
        signal_queue=shared_signal_queue    # 👈 物理通信打通
    )
    exec_engine.use_shared_mem = True
    
    logger.info(f"🔌 Injecting Mock IBKR...")
    mock_ibkr = MockIBKRHistorical()
    await mock_ibkr.connect()
    
    # =======================================================
    # 🧹 [核弹级清理] 彻底清空回测专用数据库 (DB 1)
    # 消除所有的 P:50 挂起单、跨日时间戳幽灵、残留持仓缓存！
    # =======================================================
    logger.info("💥 Nuking Redis Backtest Database (DB 1)...")
    mock_ibkr.r.flushdb() 
    
    # [NEW] 额外确保信号流处于干净状态
    mock_ibkr.r.delete("orch_trade_signals")
    mock_ibkr.r.delete("sync:orch_done")
    
    import uuid
    RUN_ID = str(uuid.uuid4())[:8]
    
    signal_engine.ibkr = mock_ibkr
    exec_engine.ibkr = mock_ibkr
    signal_engine.mock_cash = mock_ibkr.initial_capital
    exec_engine.mock_cash = mock_ibkr.initial_capital
    
    def _run_driver_safe():
        logger.info(f"▶️ Starting S5 Parquet Driver from {parquet_path}...")
        time.sleep(2) 
        try:
            from replay_driver_s5_parquet import S5ParquetDriver
            driver = S5ParquetDriver(parquet_dir=parquet_path, stream_key=BACKTEST_STREAM, run_id=RUN_ID)
            driver.run()
        except Exception as e:
            logger.error(f"❌ Driver Crashed: {e}")

    t = threading.Thread(target=_run_driver_safe, daemon=True)
    t.start()
    
    # 🚨 启动双引擎任务
    sig_task = asyncio.create_task(signal_engine.run())
    exec_task = asyncio.create_task(exec_engine.run())
    
    logger.info("👀 Monitoring Replay Status...")
    start_time = time.time()
    
    try:
        status_key = f"replay:status:{RUN_ID}"
        last_lag = None
        while True:
            if sig_task.done() or exec_task.done():
                logger.error("❌ One of the engine tasks has terminated unexpectedly!")
                break

            status = mock_ibkr.r.get(status_key)
            
            if status and status.decode() == "DONE":
                try:
                    # 监控 SE 和 OMS 双重 Lag
                    groups_se = mock_ibkr.r.xinfo_groups(BACKTEST_STREAM)
                    se_group = next((g for g in groups_se if g['name'].decode() == BACKTEST_GROUP), None)
                    lag = se_group.get('lag', 0) if se_group else 0
                    pending = se_group.get('pending', 0) if se_group else 0
                    
                    # [V8 Shared Mem Mode] OMS no longer consumes from Redis stream in this script, skip OMS_Lag check
                    if lag == 0 and pending == 0:
                        logger.info(f"✅ All processing complete. SE(Lag:0). Finishing...")
                        break
                    
                    logger.info(f"⏳ Waiting... SE_Lag: {lag} (P:{pending})")
                except Exception as e:
                    pass
                
                # 防死锁退出
                if not hasattr(main, 'stuck_counter'): main.stuck_counter = 0
                if last_lag is not None and lag == last_lag:
                    main.stuck_counter += 1
                    if main.stuck_counter > 15:
                         logger.error("❌ Orchestrator STUCK! Forcing exit.")
                         break
                else:
                    main.stuck_counter = 0

            await asyncio.sleep(2)
            
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user.")
    finally:
        elapsed = time.time() - start_time
        print(f"\n✅ Replay Finished in {elapsed:.1f}s.")
        
        try:
            sig_task.cancel()
            await exec_engine.force_close_all()
            await asyncio.sleep(1)
            exec_task.cancel()
        except: pass
        
        await asyncio.sleep(1)
        
        # 📊 [NEW] Final Backtest Summary
        print("\n" + "="*50)
        print("📊 FINAL BACKTEST PERFORMANCE SUMMARY (V8 SPLIT-ENGINE)")
        print("="*50)
        exec_engine.accounting.print_backtest_summary()
        exec_engine.accounting.print_counter_trend_summary()
        print("="*50 + "\n")
        
        mock_ibkr.save_trades() 

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
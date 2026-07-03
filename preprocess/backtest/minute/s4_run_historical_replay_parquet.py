import asyncio
import threading
import time
import logging
from pathlib import Path
import os
import argparse 
import sys

# 导入 V8 引擎
from system_orchestrator_v8 import V8Orchestrator
# 导入 Driver 和 Mock IBKR
from replay_driver_s5_parquet import S5ParquetDriver
from mock_ibkr_historical import MockIBKRHistorical

# ================= 配置区域 =================
BASE_PROJECT = Path.home() / "quant_project"

# 离线半年回测数据默认路径 (与 replay_driver_s5_parquet.py 中 PARQUET_DIR 一致)
#DEFAULT_OFFLINE_DIR = Path('/mnt/s990/data/h5_unified_overlap_id/rl_feed_parquet_new_delta_backtest')
DEFAULT_OFFLINE_DIR = Path(f"/home/kingfang007/quant_project/data/rl_feed_parquet")
# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [REPLAY_MAIN] - %(message)s'
)
logger = logging.getLogger("ReplayMain")

def run_driver_thread(parquet_path):
    """独立线程运行 Driver，负责往 Redis 灌入历史数据"""
    logger.info(f"▶️ Starting S5 Parquet Driver from {parquet_path}...")
    time.sleep(2) # 给 Redis 一点准备时间
    try:
        driver = S5ParquetDriver(parquet_dir=parquet_path)
        driver.run()
    except Exception as e:
        logger.error(f"❌ Driver Crashed: {e}")

def resolve_parquet_path(args):
    """
    解析数据路径，优先级:
    1. --parquet-dir  (显式指定路径)
    2. --date         (按日期查找 ~/quant_project/backtest/{date}/...)
    3. 默认离线路径   (/mnt/s990/data/.../rl_feed_parquet_new_delta_backtest)
    """
    if args.parquet_dir:
        p = Path(args.parquet_dir)
        if not p.exists():
            logger.error(f"❌ Parquet dir not found: {p}")
            return None
        return p
    
    if args.date:
        p = Path.home() / f"quant_project/backtest/{args.date}/rl_feed_parquet_new_delta_backtest"
        if not p.exists():
            logger.error(f"❌ Data not found: {p}")
            return None
        return p
    
    # 默认: 离线半年数据
    if not DEFAULT_OFFLINE_DIR.exists():
        logger.error(f"❌ Default offline data not found: {DEFAULT_OFFLINE_DIR}")
        return None
    return DEFAULT_OFFLINE_DIR

async def main():
    print(f"!!! EXECUTING UPDATED SCRIPT: {__file__} !!!")
    parser = argparse.ArgumentParser(description="V8 历史回测 (支持单日/离线半年)")
    parser.add_argument("--date", type=str, default=None, 
                        help="回测日期 YYYY-MM-DD (指定后从 ~/quant_project/backtest/{date}/ 读取)")
    parser.add_argument("--parquet-dir", type=str, default=None,
                        help="直接指定 Parquet 数据目录 (最高优先级)")
    args = parser.parse_args()
    
    # 0. 解析数据路径
    parquet_path = resolve_parquet_path(args)
    if parquet_path is None:
        return
    
    # 判断回测模式
    if args.date:
        mode_label = f"单日回测 [{args.date}]"
    elif args.parquet_dir:
        mode_label = f"自定义路径回测"
    else:
        mode_label = f"离线半年全量回测"
    
    print("\n" + "="*60)
    print(f"🚀 V8 Historical Replay — {mode_label}")
    print(f"📂 Data: {parquet_path}")
    print("="*60 + "\n")

    # 1. 准备配置路径
    V8_ROOT = Path(__file__).parent.parent
    config_paths = {
        'fast': str(V8_ROOT/"daily_backtest/fast_feature.json"), 
        'slow': str(V8_ROOT/"daily_backtest/slow_feature.json")
    }
    if not os.path.exists(config_paths['slow']):
        logger.warning(f"Config file not found: {config_paths['slow']}, ensuring mock compatibility.")

    # 2. 定义回测标的
    # Driver 会自动根据 Parquet 文件里的 symbol 发送数据
    # symbols = [
    #     'NVDA', 'AAPL', 'META', 'MSTR', 'PLTR', 'TSLA', 'UNH', 'AMZN', 'AMD', 
    #     'NFLX', 'MSFT', 'GOOGL', 'SPY', 'QQQ', 'IWM' 
    # ]

    symbols =  [ 
          # # --- Tier 1: 巨无霸 ---
    'NVDA', 'AAPL', 'META', 'PLTR', 'TSLA', 'UNH', 'AMZN', 'AMD', 
    # --- Tier 2: 核心蓝筹 ---
    'NFLX', 'CRWV', 'AVGO', 'MSFT', 'HOOD', 'MU', 'APP', 'GOOGL', 
    # --- Tier 3: 高流动性 --- 
    'SMCI', 'ADBE', 'CRM', 'ORCL', 'NKE', 'XOM', 'TSM', 
    # --- Indices & Macro ---
    'SPY', 'QQQ', 'IWM',
    # --- Crypto & Metals --- 
        ]
    # [关键] 隔离回测环境，防止与 Live Feature Service (maxlen=1000) 冲突导致数据丢失
    import system_orchestrator_v8
    BACKTEST_STREAM = "backtest_stream"
    BACKTEST_GROUP  = "backtest_group"
    
    # Patch Orchestrator Config
    system_orchestrator_v8.REDIS_CFG['input_stream'] = BACKTEST_STREAM
    system_orchestrator_v8.REDIS_CFG['orch_group'] = BACKTEST_GROUP
    # 🚀 [新增] 强制让 V8 引擎在回测时使用 Redis DB 1，实现与实盘的物理级隔离！
    system_orchestrator_v8.REDIS_CFG['db'] = 1
    logger.info(f"🛡️ Isolated Backtest Stream: {BACKTEST_STREAM} | Group: {BACKTEST_GROUP} | DB: 1")
    
    # 3. 实例化 Orchestrator (Mode='backtest')
    logger.info("🛠️ Building V8 Orchestrator...") 
    orch = V8Orchestrator(
        symbols=symbols, 
        mode='backtest', 
        config_paths=config_paths, 
        model_paths={} 
    )
    
    # 4. 注入 Mock IBKR (核心步骤)
    logger.info(f"🔌 Injecting Mock IBKR...")
    mock_ibkr = MockIBKRHistorical()
    await mock_ibkr.connect()
    
    # Cleanup old backtest data & status
    mock_ibkr.r.delete(BACKTEST_STREAM)
    try:
        mock_ibkr.r.xgroup_destroy(BACKTEST_STREAM, BACKTEST_GROUP)
        logger.info(f"🧹 Cleaned stale consumer group: {BACKTEST_GROUP}")
    except: pass
    
    mock_ibkr.r.delete("replay:status") # 防止使用了上次残留的 DONE 标志导致秒退
    
    # [关键] 覆盖 Orchestrator 内部的 connector
    orch.ibkr = mock_ibkr
    # [关键] 将 Mock IBKR 的资金同步给 Orchestrator 用于仓位计算
    orch.mock_cash = mock_ibkr.initial_capital
    
    # [关键] 生成本次回测的唯一 ID，防止状态污染
    import uuid
    RUN_ID = str(uuid.uuid4())[:8]
    logger.info(f"🆔 Run ID: {RUN_ID}")

    # 5. 启动 Driver 线程 (Pass isolated stream key + RUN_ID)
    def _run_driver_safe():
        logger.info(f"▶️ Starting S5 Parquet Driver from {parquet_path}...")
        time.sleep(2) 
        try:
            # Pass isolated stream key & RUN ID
            driver = S5ParquetDriver(parquet_dir=parquet_path, stream_key=BACKTEST_STREAM, run_id=RUN_ID)
            driver.run()
        except Exception as e:
            logger.error(f"❌ Driver Crashed: {e}")

    t = threading.Thread(target=_run_driver_safe, daemon=True)
    t.start()
    
    # 6. 运行 Orchestrator 主循环
    orch_task = asyncio.create_task(orch.run())
    
    # 7. 监控回放进度
    logger.info("👀 Monitoring Replay Status...")
    start_time = time.time()
    
    try:
        status_key = f"replay:status:{RUN_ID}"
        while True:
            # 检查 Redis 里的状态标志 (Specific to this RUN)
            status = mock_ibkr.r.get(status_key)
            
            if status and status.decode() == "DONE":
                # [Fix] 等待所有消息被消费 (检查 Consumer Lag)
                stream_key = BACKTEST_STREAM
                group_name = BACKTEST_GROUP
                
                try:
                    last_lag = None
                    while True:
                        # [Chk] 检查 Driver 是否挂了 (如果 Driver 死了且没有 DONE 标志)
                        if not t.is_alive():
                            # Driver 死了，检查是否正常完成
                            status = mock_ibkr.r.get(status_key)
                            if not status or status.decode() != "DONE":
                                logger.error("❌ Driver Thread died unexpectedly! (No DONE flag)")
                                return # Exit

                        groups = mock_ibkr.r.xinfo_groups(stream_key)
                        target_group = next((g for g in groups if g['name'].decode() == group_name), None)
                        
                        if target_group:
                            lag = target_group.get('lag', 0)
                            pending = target_group['pending']
                            last_delivered = target_group['last-delivered-id']
                            
                            # 获取 Stream 最新 ID
                            stream_info = mock_ibkr.r.xinfo_stream(stream_key)
                            last_entry_id = stream_info['last-generated-id']
                            
                            # 判断是否消费完毕
                            # 注意: last_delivered 可能等于 last_entry_id，且 pending=0
                            is_caught_up = (last_delivered == last_entry_id) and (pending == 0)
                            
                            if is_caught_up:
                                logger.info(f"✅ All {stream_info['length']} messages processed. Lag: 0, Pending: 0")
                                break
                            
                            # Progress Log
                            if last_lag is not None and lag < last_lag:
                                speed = last_lag - lag
                                logger.info(f"📉 Consuming... Lag: {lag} (-{speed}/2s), Pending: {pending}")
                            else:
                                logger.info(f"⏳ Waiting for consumer catch-up... Lag: {lag}, Pending: {pending}, StreamLen: {stream_info['length']}")
                            last_lag = lag
                        else:
                            logger.warning(f"⚠️ Consumer group {group_name} not found yet...")
                        
                        await asyncio.sleep(2)
                except Exception as e:
                    logger.error(f"⚠️ Error checking stream lag: {e}")
                    # Fallback to simple wait
                    await asyncio.sleep(10)

                elapsed = time.time() - start_time
                print(f"\n✅ Replay Driver Finished in {elapsed:.1f}s.")
                
                # --- 保存结果 ---
                print("\n" + "="*40)
                print("💾 SAVING TRADE LOGS")
                print("="*40)
                # 强制清理未平仓位 (EOD)
                await orch.force_close_all()
                # 等待撤单/平仓日志刷入
                await asyncio.sleep(2)
                mock_ibkr.save_trades() 
                orch.print_counter_trend_summary()
                
                # --- 停止任务 ---
                orch_task.cancel()
                try:
                    await orch_task
                except asyncio.CancelledError:
                    logger.info("🛑 Orchestrator Stopped Gracefully.")
                break
            
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user.")
        mock_ibkr.save_trades()
        orch_task.cancel()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
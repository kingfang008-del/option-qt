#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
文件名: run_realtime_replay.py
描述: 全量 Alpha 批处理推理工厂 (纯净无交易版)
修复: 彻底剥离脆弱的文件隐藏(Mask)逻辑，利用引擎原生 REPLAY_START_TS 机制进行时间隔离。
"""

import asyncio
import threading
import time
import logging
import json
import pickle
import redis
import argparse
import subprocess
import uuid
import sys
import copy
import os
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import pytz
import datetime as dt

# 导入业务组件
from system_orchestrator_v8 import V8Orchestrator
from mock_ibkr_historical import MockIBKRHistorical
import system_orchestrator_v8
import data_persistence_service_v8_pg
import psycopg2

from config import (
    REDIS_CFG, DB_DIR, PROJECT_ROOT, FEATURE_SERVICE_STATE_FILE,
    STREAM_FUSED_MARKET, HASH_OPTION_SNAPSHOT, STREAM_INFERENCE, STREAM_TRADE_LOG
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [BATCH_INFERENCE] - %(message)s')
logger = logging.getLogger("BatchInference")

class BatchPostgresDriver:
    """全量连续流发球机 (PostgreSQL 版)"""
    def __init__(self, target_dates: list, run_id: str, speed_factor: float = 0.0):
        self.target_dates = sorted(target_dates) # 格式: ['20260101', '20260102', ...]
        self.run_id = run_id
        self.speed_factor = speed_factor 
        self.r = redis.Redis(**{k:v for k,v in REDIS_CFG.items() if k in ['host','port','db']})
        from config import PG_DB_URL
        self.pg_url = PG_DB_URL

    def run(self):
        total_days = len(self.target_dates)
        logger.info(f"🚀 [Driver] Starting Batch Inference from PostgreSQL for {total_days} days...")

        for idx, date_str in enumerate(self.target_dates, 1):
            logger.info(f"\n" + "="*60)
            logger.info(f"📅 [{idx}/{total_days}] Processing Date: {date_str}")
            
            # =========================================================
            # [新增] 动态打印每一天的预热上下文 (Warmup Context)
            # =========================================================
            if idx == 1:
                logger.info(f"🧊 [Cold Start] {date_str} 之前无历史数据，引擎将从零开始积累特征。")
            else:
                prev_date = self.target_dates[idx-2]
                logger.info(f"🔥 [Hot Start] 引擎内存无缝接力，自动包含前一日 ({prev_date}) 尾盘特征，无需读盘！")
            logger.info("="*60)

            # 1. 提取数据 (从 PostgreSQL)
            try:
                conn = psycopg2.connect(self.pg_url)
                # 使用 NY 时区计算起止时间戳
                ny_tz = pytz.timezone('America/New_York')
                start_dt = ny_tz.localize(dt.datetime.strptime(date_str, "%Y%m%d").replace(hour=0, minute=0, second=0))
                end_dt = start_dt + dt.timedelta(days=1)
                
                start_ts = start_dt.timestamp()
                end_ts = end_dt.timestamp()
                
                logger.info(f"📥 Fetching dual-resolution data from PG for {date_str} ({start_ts} -> {end_ts})...")
                
                df_bars_1m = pd.read_sql(
                    "SELECT symbol, ts, open, high, low, close, volume FROM market_bars_1m WHERE ts >= %s AND ts < %s ORDER BY ts ASC", 
                    conn, params=(start_ts, end_ts)
                )
                df_opts_1m = pd.read_sql(
                    "SELECT symbol, ts, buckets_json FROM option_snapshots_1m WHERE ts >= %s AND ts < %s ORDER BY ts ASC", 
                    conn, params=(start_ts, end_ts)
                )
                
                df_bars_5m = pd.read_sql(
                    "SELECT symbol, ts, open, high, low, close, volume FROM market_bars_5m WHERE ts >= %s AND ts < %s ORDER BY ts ASC", 
                    conn, params=(start_ts, end_ts)
                )
                df_opts_5m = pd.read_sql(
                    "SELECT symbol, ts, buckets_json FROM option_snapshots_5m WHERE ts >= %s AND ts < %s ORDER BY ts ASC", 
                    conn, params=(start_ts, end_ts)
                )
                conn.close()

                # === [TIME ALIGNMENT] ===
                # [FIX] 彻底移除 +60.0 / +300.0 偏移，对齐 SQLite 版本修复逻辑。
                df_bars_1m['ts_aligned'] = df_bars_1m['ts']
                df_opts_1m['ts_aligned'] = df_opts_1m['ts']
                
                df_bars_5m['ts_aligned'] = df_bars_5m['ts']
                df_opts_5m['ts_aligned'] = df_opts_5m['ts']

            except Exception as e:
                logger.error(f"❌ Failed to fetch data from PG for {date_str}: {e}")
                continue

            if df_bars_1m.empty:
                logger.warning(f"⚠️ No 1m bar data in PG for {date_str}, skipping.")
                continue

            # === [OPTIMIZATION: DICT MAPPING] ===
            # Convert DataFrames to nested dicts for O(1) lookup: {ts: {symbol: data}}
            def to_map(df, type_key):
                m = {}
                for _, row in df.iterrows():
                    ts = row['ts_aligned']
                    sym = row['symbol']
                    if ts not in m: m[ts] = {}
                    if type_key == 'bars':
                        m[ts][sym] = {
                            'open': row['open'], 'high': row['high'],
                            'low': row['low'], 'close': row['close'], 'volume': row['volume']
                        }
                    else: # options
                        opt_data = row['buckets_json']
                        if isinstance(opt_data, str): opt_data = json.loads(opt_data)
                        m[ts][sym] = opt_data
                return m

            map_b1 = to_map(df_bars_1m, 'bars')
            map_o1 = to_map(df_opts_1m, 'opts')
            map_b5 = to_map(df_bars_5m, 'bars')
            map_o5 = to_map(df_opts_5m, 'opts')

            # 全局时间轴 (对齐后的)
            all_ts = sorted(list(set(map_b1.keys()) | set(map_o1.keys()) | set(map_b5.keys()) | set(map_o5.keys())))
            
            # [精简] 严格限制在 09:30 - 16:00 交易时段，防止多出 16:05-16:10 的对齐尾巴
            filtered_ts = []
            for ts in all_ts:
                dt_ny = dt.datetime.fromtimestamp(ts, ny_tz)
                time_val = dt_ny.time()
                if dt.time(9, 30) <= time_val <= dt.time(16, 0):
                    filtered_ts.append(ts)
            all_ts = filtered_ts
            
            logger.info(f"✅ Loaded {len(all_ts)} synchronized frames (Filtered to 09:30-16:00). Streaming via O(N) Dict-Mapping...")
            
            # ==========================================================
            # 🚀 [核心修复] 5分钟状态缓存器 (State Retention)
            # ==========================================================
            last_5m_state = {}
            count = 0
            for ts_val in tqdm(all_ts, desc=f"Inferring {date_str}"):
                self.r.set("replay:current_ts", str(ts_val)) 
                
                symbols_at_ts = set(map_b1.get(ts_val, {}).keys()) | set(map_o1.get(ts_val, {}).keys()) | \
                                set(map_b5.get(ts_val, {}).keys()) | set(map_o5.get(ts_val, {}).keys())
                
                batch_payloads = []
                last_msg = self.r.xrevrange(STREAM_INFERENCE, count=1)
                start_id = last_msg[0][0] if last_msg else None
                
                for sym in symbols_at_ts:
                    payload = {'ts': ts_val, 'symbol': sym}
                    
                    # --- 1min 数据 (瞬时状态，无需保持) ---
                    if sym in map_b1.get(ts_val, {}):
                        payload['stock'] = map_b1[ts_val][sym]
                    if sym in map_o1.get(ts_val, {}):
                        opt_data = map_o1[ts_val][sym]
                        if isinstance(opt_data, dict):
                            payload['option_buckets'] = opt_data.get('buckets', [])
                            payload['option_contracts'] = opt_data.get('contracts', [])
                        else:
                            # 兼容旧版本直接存储 list 的情况
                            payload['option_buckets'] = opt_data
                            payload['option_contracts'] = []
                        opt_data_for_redis = opt_data if isinstance(opt_data, dict) else {'buckets': opt_data, 'ts': ts_val}
                        opt_data_for_redis['ts'] = ts_val
                        self.r.hset(HASH_OPTION_SNAPSHOT, sym, pickle.dumps(opt_data_for_redis))

                    # --- 5min 数据 (状态机保持与前向填充) ---
                    if sym in map_b5.get(ts_val, {}) or sym in map_o5.get(ts_val, {}):
                        if sym not in last_5m_state: last_5m_state[sym] = {}
                        if sym in map_b5.get(ts_val, {}):
                            last_5m_state[sym]['stock_5m'] = map_b5[ts_val][sym]
                        if sym in map_o5.get(ts_val, {}):
                            opt_data_5m = map_o5[ts_val][sym]
                            if isinstance(opt_data_5m, dict):
                                last_5m_state[sym]['option_buckets_5m'] = opt_data_5m.get('buckets', [])
                                last_5m_state[sym]['option_contracts_5m'] = opt_data_5m.get('contracts', [])
                            else:
                                last_5m_state[sym]['option_buckets_5m'] = opt_data_5m
                                last_5m_state[sym]['option_contracts_5m'] = []
                    
                    # 2. 将黑板上的 5min 数据挂载到当前 1min 的 Payload 中
                    if sym in last_5m_state:
                        payload.update(last_5m_state[sym])

                    batch_payloads.append(payload)
                
                # Send the entire minute worth of data in ONE message
                if batch_payloads:
                    self.r.xadd(STREAM_FUSED_MARKET, {'batch': pickle.dumps(batch_payloads)})
                
                # === [FRAME SYNCHRONIZATION LOCK] ===
                # 阻塞直到: 1. 特征引擎算完这一帧 2. 交易引擎(Orchestrator)处理完这一帧
                timeout = 0
                while True:
                    ack_feat = self.r.get("sync:feature_calc_done")
                    ack_orch = self.r.get("sync:orch_done")
                    
                    feat_ts = float(ack_feat) if ack_feat else 0.0
                    orch_ts = float(ack_orch) if ack_orch else 0.0
                    
                    if feat_ts >= ts_val and orch_ts >= ts_val:
                        break
                            
                    time.sleep(0.001) 
                    timeout += 1
                    if timeout > 5000: # 5 seconds
                        logger.warning(f"⚠️ [STALL] Sync Timeout at ts={ts_val}. Feat:{feat_ts} Orch:{orch_ts}")
                        break
                        
                count += 1
                
        status_key = f"replay:status:{self.run_id}"
        self.r.set(status_key, "DONE")
        logger.info(f"🏁 All Dates Processed. Status: {status_key}")


async def main():
    parser = argparse.ArgumentParser(description="Batch Alpha Inference Factory")
    parser.add_argument('--start-date', type=str, default="20260101", help="Start processing from this date (YYYYMMDD)")
    parser.add_argument('--end-date', type=str, default="20991231", help="Process up to this date (YYYYMMDD)")
    args = parser.parse_args()

    # ================= [核心升级] 切换至 PostgreSQL 发现日期 =================
    from config import PG_DB_URL
    conn_pg = psycopg2.connect(PG_DB_URL)
    conn_pg.autocommit = True
    c_pg = conn_pg.cursor()
    
    # 查找所有 1m 行情分区表，提取 YYYYMMDD 后缀
    logger.info("📡 Querying PostgreSQL for available dates...")
    c_pg.execute("SELECT table_name FROM information_schema.tables WHERE table_name LIKE 'market_bars_1m_20%'")
    rows = c_pg.fetchall()
    all_dates = sorted([r[0].split('_')[-1] for r in rows])
    
    target_dates = [d for d in all_dates if args.start_date <= d <= args.end_date]
    
    if not target_dates:
        logger.error(f"❌ No PostgreSQL data partitions found between {args.start_date} and {args.end_date}.")
        c_pg.close()
        conn_pg.close()
        return
        
    logger.info(f"🔍 Found {len(target_dates)} dates in PostgreSQL to process: {target_dates[0]} -> {target_dates[-1]}")

    # PostgreSQL 预清理
    if c_pg:
        # 此时 conn_pg 还是 autocommit=True
        for date_str in target_dates:
            try:
                # 清理推理日志
                c_pg.execute(f"TRUNCATE TABLE alpha_logs_{date_str}")
                c_pg.execute(f"TRUNCATE TABLE trade_logs_{date_str}")
                c_pg.execute(f"TRUNCATE TABLE trade_logs_backtest_{date_str}")

                # 🚀 [新增] 每次回测前强制删除 Debug 表，确保表结构永远和 JSON 完美同步！
                c_pg.execute(f"DROP TABLE IF EXISTS debug_fast_{date_str}")
                c_pg.execute(f"DROP TABLE IF EXISTS debug_slow_1m_{date_str}")
                c_pg.execute(f"DROP TABLE IF EXISTS debug_slow_5m_{date_str}")
            except Exception as e:
                # 分区不存在时忽略错误（数据可能本来就没有）
                logger.debug(f"⚠️ Cleanup skipped for {date_str}: {e}")
                pass
        c_pg.close()
        conn_pg.close()

    r = redis.Redis(**{k:v for k,v in REDIS_CFG.items() if k in ['host','port','db']})
    RUN_ID = str(uuid.uuid4())[:8]
    logger.info(f"🆔 Generated Replay RUN_ID: {RUN_ID}")
    
    # 清理环境
    for stream in [STREAM_FUSED_MARKET, STREAM_INFERENCE, STREAM_TRADE_LOG, HASH_OPTION_SNAPSHOT, f"replay:status:{RUN_ID}"]:
        r.delete(stream)

    for state_file in ["positions_ignore_v8.json"]:
        path = DB_DIR / state_file if state_file.endswith(".db") else PROJECT_ROOT / state_file
        if path.exists():
            try: path.unlink()
            except: pass

    # 预建 Redis 流与组
    for s_key in [STREAM_FUSED_MARKET, STREAM_INFERENCE, STREAM_TRADE_LOG]:
        try: r.xadd(s_key, {'init': '1'}) 
        except: pass
        for g_name in ['compute_group', 'persistence_group', 'persistence_group_pg', 'v8_orchestrator_group']:
            try: r.xgroup_create(s_key, g_name, id='0')
            except: pass 

    # ================= 精准计算 REPLAY_START_TS =================
    NY_TZ = pytz.timezone('America/New_York')
    actual_start_date = target_dates[0]
    target_dt = dt.datetime.strptime(actual_start_date, "%Y%m%d")
    # 锁定到批处理第一天的 09:30:00
    target_dt = NY_TZ.localize(target_dt.replace(hour=9, minute=30, second=0))
    replay_start_ts = target_dt.timestamp()

    r.set("replay:current_ts", str(replay_start_ts))
    
    state_file = PROJECT_ROOT / FEATURE_SERVICE_STATE_FILE
    if state_file.exists(): state_file.unlink()

    # 将精准时间戳传给特征引擎，它会自动无视大于等于这个时间的数据
    env = os.environ.copy()
    env['REPLAY_START_TS'] = str(replay_start_ts)

    feature_process = None
    try:
        logger.info("🚀 Starting Feature Compute Service (Subprocess)...")
        feature_process = subprocess.Popen(
            [sys.executable, "feature_compute_service_v8.py"],
            cwd=str(PROJECT_ROOT/"baseline"),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env 
        )

        ready_event = threading.Event()
        def monitor_stdout():
            for line in iter(feature_process.stdout.readline, ''):
                line = line.strip()
                if line:
                    logger.info(f"📤 [FeatureService] {line}")
                # 兼容了彻底的 Cold start 打印
                if ("Complete" in line and "Warmup" in line) or "Deep Warmup Complete" in line or "Cold start" in line:
                    ready_event.set()

        t_mon = threading.Thread(target=monitor_stdout, daemon=True)
        t_mon.start()

        logger.info("⏳ Waiting for Engine Deep Warmup to complete...")
        if not ready_event.wait(timeout=120):
            poll = feature_process.poll()
            if poll is not None:
                logger.error(f"❌ Feature service DIED with exit code {poll} before warmup finished!")
            else:
                logger.error("❌ Warmup Timeout! Feature service is still running but hasn't signaled readiness.")
            return
            
        logger.info("✅ Feature Engine Warmup Finished.")

        # ================= 时钟劫持与防死锁补丁 =================
        class ReplayDatetime(dt.datetime):
            @classmethod
            def now(cls, tz=None):
                ts_str = r.get("replay:current_ts")
                if ts_str: return dt.datetime.fromtimestamp(float(ts_str), tz=tz)
                return super().now(tz)
                
        def replay_time():
            ts_str = r.get("replay:current_ts")
            if ts_str: return float(ts_str)
            return time.time()

        system_orchestrator_v8.datetime = ReplayDatetime
        system_orchestrator_v8.time.time = replay_time
        data_persistence_service_v8_pg.datetime = ReplayDatetime
        data_persistence_service_v8_pg.time.time = replay_time

        import config
        persist_pg_cfg = copy.deepcopy(config.REDIS_CFG)
        persist_pg_cfg['group'] = 'persistence_group_pg'
        persist_pg_cfg['pg_group'] = 'persistence_group_pg'
        persist_pg_cfg['consumer'] = 'pg_writer_1m'
        data_persistence_service_v8_pg.REDIS_CFG = persist_pg_cfg
        
        orch_cfg = copy.deepcopy(config.REDIS_CFG)
        orch_cfg['group'] = 'v8_orchestrator_group'
        orch_cfg['consumer'] = 'v8_worker'
        orch_cfg['input_stream'] = STREAM_INFERENCE 
        system_orchestrator_v8.REDIS_CFG = orch_cfg
        config.REDIS_CFG = orch_cfg
        
        logger.info("🐘 Starting Data Persistence Service (PG)...")
        try:
            persistence_svc_pg = data_persistence_service_v8_pg.DataPersistenceServicePG()
            threading.Thread(target=persistence_svc_pg.run, daemon=True).start()
        except Exception as e:
            logger.error(f"Failed to start PG persistence service: {e}")

        # ================= 启动 Orchestrator (纯推理模式) =================
        V8_ROOT = Path(__file__).parent.parent
        config_paths = {
            'fast': str("/home/kingfang007/notebook/train/fast_feature.json"), 
            'slow': str("/home/kingfang007/notebook/train/slow_feature.json")
        }
        # 【重要】在这里填入你真实的 Checkpoint 路径
        model_paths = {
            'slow': str("/home/kingfang007/quant_project/checkpoints_advanced_alpha/advanced_alpha_best.pth"),
            'fast': str("/home/kingfang007/quant_project/checkpoints_fast_final/fast_final_best.pth")
        }

        from config import TARGET_SYMBOLS
        logger.info("🛠️ Building V8 Orchestrator (Inference Factory Mode)...")
        orchestrator =  V8Orchestrator(
            symbols=TARGET_SYMBOLS, 
            mode='realtime', 
            config_paths=config_paths, 
            model_paths=model_paths
        )
        
        # =====================================================================
        # [核心改造]: 完全阉割交易逻辑，只输出 Alpha 日志
        # =====================================================================
        orchestrator.strategy.decide_entry = lambda ctx: None
        orchestrator.strategy.check_exit = lambda ctx: None
        logger.info("🛡️ Trading disabled. System is now purely extracting Alpha/Features.")

        mock_ibkr = MockIBKRHistorical()
        orchestrator.ibkr = mock_ibkr
        orchestrator.mock_cash = mock_ibkr.initial_capital
        
        orch_task = asyncio.create_task(orchestrator.run())

        # ================= 启动流水线发球机 =================
        def _run_driver():
            try:
                driver = BatchPostgresDriver(target_dates=target_dates, run_id=RUN_ID, speed_factor=0)
                driver.run()
            except Exception as e:
                logger.error(f"❌ Driver Crashed: {e}")

        # [🔥 终极修复] 在启动发球机之前，给 Orchestrator 2秒钟的初始化缓冲时间
        logger.info("⏳ Giving Orchestrator 2s to initialize...")
        await asyncio.sleep(2.0)

        driver_thread = threading.Thread(target=_run_driver, daemon=True)
        driver_thread.start()

        # ================= 全链路进度监控与收尾 =================
        logger.info("👀 Monitoring Batch Progress...")
        start_time = time.time()
        
        status_key = f"replay:status:{RUN_ID}"
        while True:
            await asyncio.sleep(2.0)
            
            fused_lag, orch_lag, persist_lag = 0, 0, 0
            try:
                for g in r.xinfo_groups(STREAM_FUSED_MARKET):
                    if g['name'] in [b'compute_group', 'compute_group']: fused_lag = g['pending']
                for g in r.xinfo_groups(STREAM_INFERENCE):
                    if g['name'] in [b'orchestrator_group', 'orchestrator_group']: orch_lag = g['pending']
                if r.exists(STREAM_TRADE_LOG):
                    for g in r.xinfo_groups(STREAM_TRADE_LOG):
                        if g['name'] in [b'persist_group', 'persist_group']: persist_lag = g['pending']
            except: pass

            status_raw = r.get(status_key)
            status_str = status_raw.decode('utf-8') if status_raw else ""

            if status_str == "DONE" and fused_lag == 0 and orch_lag == 0 and persist_lag == 0:
                elapsed = time.time() - start_time
                logger.info(f"\n🎉 ALL {len(target_dates)} Days Processed Successfully in {elapsed:.1f}s!")
                
                orch_task.cancel()
                try: await orch_task
                except asyncio.CancelledError: pass
                break
            
            # 每 10 秒打印一次心跳，防止刷屏
            # if int(time.time()) % 10 == 0:
            #     logger.info(f"❤️ Heartbeat | Lag -> Fused:{fused_lag} Orch:{orch_lag} Persist:{persist_lag}")

    except KeyboardInterrupt:
        logger.warning("⚠️ Interrupted by user!")
        if 'orch_task' in locals(): orch_task.cancel()
        
    finally:
        if feature_process and feature_process.poll() is None:
            logger.info("🛑 Terminating Feature Compute Service...")
            feature_process.terminate()
            feature_process.wait()

if __name__ == "__main__":
    asyncio.run(main())
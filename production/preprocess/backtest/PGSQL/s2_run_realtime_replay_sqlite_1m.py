#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
文件名: run_realtime_replay.py
描述: 全量 Alpha 批处理推理工厂 (纯净无交易版)
修复: 彻底剥离脆弱的文件隐藏(Mask)逻辑，利用引擎原生 REPLAY_START_TS 机制进行时间隔离。
"""
import os
# 🚀 必须在最顶层注入，确保后续 import 的 config.py 以及启动的子进程都能继承！
os.environ['RUN_MODE'] = 'BACKTEST'
import asyncio
import threading
import time
import logging
import sqlite3
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

# [NEW] Add project root to sys.path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils import serialization_utils as ser

# 导入业务组件
from signal_engine_v8 import SignalEngineV8
from mock_ibkr_historical import MockIBKRHistorical
import signal_engine_v8
import data_persistence_service_v8_sqlite

from config import (
    REDIS_CFG, DB_DIR, PROJECT_ROOT, FEATURE_SERVICE_STATE_FILE,
    STREAM_FUSED_MARKET, HASH_OPTION_SNAPSHOT, STREAM_INFERENCE, STREAM_TRADE_LOG
)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [BATCH_INFERENCE] - %(message)s')
logger = logging.getLogger("BatchInference")
logger.info(f"DEBUG: Script Start - REDIS_DB env: {os.environ.get('REDIS_DB')}")
logger.info(f"DEBUG: REDIS_CFG['db']: {REDIS_CFG.get('db')}")

class BatchSQLiteDriver:
    """全量连续流发球机"""
    def __init__(self, target_dbs: list, run_id: str, speed_factor: float = 0.01):
        self.target_dbs = sorted(target_dbs)
        self.run_id = run_id
        self.speed_factor = speed_factor 
        self.r = redis.Redis(**{k:v for k,v in REDIS_CFG.items() if k in ['host','port','db']})
        print(f"DEBUG: Parent Replay Driver connected to Redis DB {REDIS_CFG['db']}")

    def run(self):
        total_dbs = len(self.target_dbs)
        logger.info(f"🚀 [Driver] Starting Batch Inference for {total_dbs} databases...")

        # [新增] 获取所有数据库列表，用于向用户展示上下文
        from config import DB_DIR
        all_dbs = sorted([f for f in DB_DIR.glob("market_*.db") if len(f.stem) == 15])

        for idx, db_path in enumerate(self.target_dbs, 1):
            date_str = db_path.stem.split('_')[1]
            
            logger.info(f"\n" + "="*60)
            logger.info(f"📂 [{idx}/{total_dbs}] Processing: {db_path.name}")
            
            # =========================================================
            # [新增] 动态打印每一天的预热上下文 (Warmup Context)
            # =========================================================
            if idx == 1:
                # 第一天：计算硬盘上到底有哪些合法的前置预热库
                prev_dbs = [f.name for f in all_dbs if f.stem.split('_')[1] < date_str][-3:]
                prev_dbs.reverse()
                if prev_dbs:
                    logger.info(f"🔄 [Warm Start] 引擎初始预热依赖的历史数据: {prev_dbs}")
                else:
                    logger.info(f"🧊 [Cold Start] {date_str} 之前无历史 DB，引擎将从零开始积累特征。")
            else:
                # 第二天以后：内存自然接力
                prev_date = self.target_dbs[idx-2].stem.split('_')[1]
                logger.info(f"🔥 [Hot Start] 引擎内存无缝接力，自动包含前一日 ({prev_date}) 尾盘特征，无需读盘！")
            logger.info("="*60)
            # =========================================================

            # 1. 战前清理: 清空本 DB 的旧日志，防止污染
            try:
                conn = sqlite3.connect(f"file:{db_path}?mode=rw", uri=True, timeout=60.0)
                c = conn.cursor()
                c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='alpha_logs'")
                if c.fetchone(): c.execute("DELETE FROM alpha_logs")
                c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='trade_logs'")
                if c.fetchone(): c.execute("DELETE FROM trade_logs")
                conn.commit()
                
                # 2. 提取双分辨率数据
                logger.info(f"📥 Fetching dual-resolution data from SQLite for {date_str}...")
                df_bars_1m = pd.read_sql("SELECT symbol, ts, open, high, low, close, volume FROM market_bars_1m ORDER BY ts ASC", conn)
                df_opts_1m = pd.read_sql("SELECT symbol, ts, buckets_json FROM option_snapshots_1m ORDER BY ts ASC", conn)
                
                # 检查是否存在 5m 表，不存在则跳过 5m 逻辑
                c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='market_bars_5m'")
                has_5m = c.fetchone() is not None
                
                if has_5m:
                    df_bars_5m = pd.read_sql("SELECT symbol, ts, open, high, low, close, volume FROM market_bars_5m ORDER BY ts ASC", conn)
                    df_opts_5m = pd.read_sql("SELECT symbol, ts, buckets_json FROM option_snapshots_5m ORDER BY ts ASC", conn)
                else:
                    df_bars_5m = pd.DataFrame()
                    df_opts_5m = pd.DataFrame()
                
                conn.close()
            except Exception as e:
                logger.warning(f"⚠️ Data fetch failed for {db_path.name}: {e}")
                continue

            if df_bars_1m.empty:
                logger.warning(f"⚠️ No 1m data in {db_path.name}, skipping.")
                continue

            # === [TIME ALIGNMENT] ===
            # [FIX] 彻底移除 +60.0 / +300.0 偏移，因为 s1_seed 已经对齐过闭合时间，此处再偏移会导致 1min 延迟。
            df_bars_1m['ts_aligned'] = df_bars_1m['ts'].astype(float)
            df_opts_1m['ts_aligned'] = df_opts_1m['ts'].astype(float)
            
            if not df_bars_5m.empty:
                df_bars_5m['ts_aligned'] = df_bars_5m['ts'].astype(float)
            if not df_opts_5m.empty:
                df_opts_5m['ts_aligned'] = df_opts_5m['ts'].astype(float)

            # === [OPTIMIZATION: DICT MAPPING] ===
            def to_map(df, type_key):
                m = {}
                if df.empty: return m
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
                        if isinstance(opt_data, str): 
                            try: opt_data = json.loads(opt_data)
                            except: opt_data = {}
                        m[ts][sym] = opt_data
                return m

            map_b1 = to_map(df_bars_1m, 'bars')
            map_o1 = to_map(df_opts_1m, 'opts')
            map_b5 = to_map(df_bars_5m, 'bars')
            map_o5 = to_map(df_opts_5m, 'opts')

            # 全局时间轴
            all_ts = sorted(list(set(map_b1.keys()) | set(map_o1.keys()) | set(map_b5.keys()) | set(map_o5.keys())))
            logger.info(f"✅ Loaded {len(all_ts)} synchronized frames. Streaming...")

            last_5m_state = {}
            count = 0
            for ts_val in tqdm(all_ts, desc=f"Inferring {db_path.name}"):
                self.r.set("replay:current_ts", str(ts_val)) 
                
                symbols_at_ts = set(map_b1.get(ts_val, {}).keys()) | set(map_o1.get(ts_val, {}).keys()) | \
                                set(map_b5.get(ts_val, {}).keys()) | set(map_o5.get(ts_val, {}).keys())
                
                batch_payloads = []
                last_msg = self.r.xrevrange(STREAM_INFERENCE, count=1)
                start_id = last_msg[0][0] if last_msg else None
                
                for sym in symbols_at_ts:
                    payload = {'ts': ts_val, 'symbol': sym}
                    
                    # --- 1min 数据 ---
                    if sym in map_b1.get(ts_val, {}):
                        payload['stock'] = map_b1[ts_val][sym]
                    if sym in map_o1.get(ts_val, {}):
                        opt_data = map_o1[ts_val][sym]
                        if isinstance(opt_data, dict):
                            payload['option_buckets'] = opt_data.get('buckets', [])
                            payload['option_contracts'] = opt_data.get('contracts', [])
                        else:
                            payload['option_buckets'] = opt_data
                            payload['option_contracts'] = []
                        
                        opt_data_for_redis = opt_data if isinstance(opt_data, dict) else {'buckets': opt_data, 'ts': ts_val}
                        opt_data_for_redis['ts'] = ts_val
                        self.r.hset(HASH_OPTION_SNAPSHOT, sym, ser.pack(opt_data_for_redis))

                    # --- 5min 数据 ---
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
                    
                    if sym in last_5m_state:
                        payload.update(last_5m_state[sym])

                    batch_payloads.append(payload)
                
                if batch_payloads:
                    self.r.xadd(STREAM_FUSED_MARKET, {'batch': ser.pack(batch_payloads)})
                
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
                    if timeout > 30000: # 5 seconds
                        logger.warning(f"⚠️ [STALL] Sync Timeout at ts={ts_val}. Feat:{feat_ts} Orch:{orch_ts}")
                        break
                        
                count += 1
                
        status_key = f"replay:status:{self.run_id}"
        self.r.set(status_key, "DONE")
        logger.info(f"🏁 All Databases Processed. Status: {status_key}")


async def main():
    parser = argparse.ArgumentParser(description="Batch Alpha Inference Factory")
    parser.add_argument('--start-date', type=str, default="20260101", help="Start processing from this date (YYYYMMDD)")
    parser.add_argument('--end-date', type=str, default="20991231", help="Process up to this date (YYYYMMDD)")
    parser.add_argument('--skip-warmup', action='store_true', help="Skip Feature Engine deep warmup for instant start")
    args = parser.parse_args()

    # ================= 获取需要处理的数据库列表 =================
    all_dbs = sorted([f for f in DB_DIR.glob("market_*.db") if f.stem.startswith("market_") and len(f.stem) == 15])
    target_dbs = []
    for db in all_dbs:
        date_str = db.stem.split('_')[1]
        if args.start_date <= date_str <= args.end_date:
            target_dbs.append(db)
            
    if not target_dbs:
        logger.error(f"❌ No databases found between {args.start_date} and {args.end_date}.")
        return
        
    logger.info(f"🔍 Found {len(target_dbs)} databases to process.")

    # ================= 战前全局清理 =================
    # 一次性清空所有目标数据库的脏日志，防止 Orchestrator 读取到未来的幽灵 Alpha
    logger.info("🧹 Pre-cleaning alpha_logs and trade_logs for all target databases...")
    for db_path in target_dbs:
        try:
            conn = sqlite3.connect(f"file:{db_path}?mode=rw", uri=True)
            c = conn.cursor()
            c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='alpha_logs'")
            if c.fetchone(): c.execute("DELETE FROM alpha_logs")
            c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='trade_logs'")
            if c.fetchone(): c.execute("DELETE FROM trade_logs")
            conn.commit()
            conn.close()
        except Exception:
            pass

    r = redis.Redis(**{k:v for k,v in REDIS_CFG.items() if k in ['host','port','db']})
    RUN_ID = str(uuid.uuid4())[:8]
    logger.info(f"🆔 Generated Replay RUN_ID: {RUN_ID}")
    
    # 清理环境
    for stream in [STREAM_FUSED_MARKET, STREAM_INFERENCE, STREAM_TRADE_LOG, HASH_OPTION_SNAPSHOT, f"replay:status:{RUN_ID}"]:
        r.delete(stream)

    for state_file in ["orchestrator_state.db", "positions_ignore_v8.json"]:
        path = DB_DIR / state_file if state_file.endswith(".db") else PROJECT_ROOT / state_file
        if path.exists():
            try: path.unlink()
            except: pass

    # 预建 Redis 流与组
    for s_key in [STREAM_FUSED_MARKET, STREAM_INFERENCE, STREAM_TRADE_LOG]:
        try: r.xadd(s_key, {'init': '1'}) 
        except: pass
        for g_name in ['feature_group', 'persistence_group', 'v8_orch_group']:
            try: r.xgroup_create(s_key, g_name, id='0')
            except: pass 

    # ================= 精准计算 REPLAY_START_TS =================
    NY_TZ = pytz.timezone('America/New_York')
    first_db_date = target_dbs[0].stem.split('_')[1]
    actual_start_date = args.start_date if args.start_date >= first_db_date else first_db_date
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
    if args.skip_warmup:
        env['SKIP_DEEP_WARMUP'] = '1'

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
                    logger.info(f"[FEAT] {line}")
                # 兼容了彻底的 Cold start 打印，以及手动跳过预热的打印
                if ("Complete" in line and "Warmup" in line) or "Deep Warmup Complete" in line or "Cold start" in line or "Skipped" in line:
                    ready_event.set()

        t_mon = threading.Thread(target=monitor_stdout, daemon=True)
        t_mon.start()

        logger.info("⏳ Waiting for Engine Deep Warmup to complete...")
        if not ready_event.wait(timeout=60):
            logger.error("❌ Warmup Timeout! Feature service failed to initialize.")
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

        signal_engine_v8.datetime = ReplayDatetime
        signal_engine_v8.time.time = replay_time
        data_persistence_service_v8_sqlite.datetime = ReplayDatetime
        data_persistence_service_v8_sqlite.time.time = replay_time

        orig_init_db = data_persistence_service_v8_sqlite.DataPersistenceServiceSQLite._init_db
        def robust_init_db(self):
            for attempt in range(5):
                try:
                    orig_init_db(self)
                    return
                except sqlite3.OperationalError as e:
                    if 'disk i/o' in str(e).lower():
                        logger.warning(f"⚠️ SQLite Disk I/O Lock detected. Auto-Healing attempt {attempt+1}/5...")
                        if hasattr(self, 'conn') and self.conn:
                            try: self.conn.close()
                            except: pass
                        time.sleep(1)
                    else: raise e
        data_persistence_service_v8_sqlite.DataPersistenceServiceSQLite._init_db = robust_init_db

        import config
        persist_cfg = copy.deepcopy(config.REDIS_CFG)
        persist_cfg['group'] = 'persistence_group'
        persist_cfg['consumer'] = 'sqlite_writer_1m'
        data_persistence_service_v8_sqlite.REDIS_CFG = persist_cfg
        
        orch_cfg = copy.deepcopy(config.REDIS_CFG)
        orch_cfg['group'] = 'v8_orch_group'
        orch_cfg['consumer'] = 'v8_worker'
        orch_cfg['input_stream'] = STREAM_INFERENCE 
        signal_engine_v8.REDIS_CFG = orch_cfg
        config.REDIS_CFG = orch_cfg
        
        logger.info("💾 Starting Data Persistence Service...")
        persistence_svc = data_persistence_service_v8_sqlite.DataPersistenceServiceSQLite()
        threading.Thread(target=persistence_svc.run, daemon=True).start()

        # ================= 启动 Orchestrator (纯推理模式) =================
        V8_ROOT = Path(__file__).parent.parent
        config_paths = {
            'fast': str("/home/kingfang007/notebook/train/fast_feature.json"), 
            'slow': str("/home/kingfang007/notebook/train/slow_feature.json")
        }
        # 【重要】在这里填入你真实的 Checkpoint 路径
        model_paths = {
            'slow': str("/home/kingfang007/quant_project/checkpoints_advanced_alpha/advanced_alpha_best.pth"),
            'fast': str("/home/kingfang007/quant_project/checkpoints_advanced_alpha/fast_final_best.pth")
        }

        from config import TARGET_SYMBOLS
        logger.info("🛠️ Building V8 Signal Engine (Inference Factory Mode)...")
        signal_engine = SignalEngineV8(
            symbols=TARGET_SYMBOLS, 
            mode='backtest', 
            config_paths=config_paths, 
            model_paths=model_paths
        )
        signal_engine.only_log_alpha = True
        logger.info("🛡️ Trading disabled. System is now purely extracting Alpha/Features.")

        signal_task = asyncio.create_task(signal_engine.run())

        # ================= 启动流水线发球机 =================
        def _run_driver():
            try:
                driver = BatchSQLiteDriver(target_dbs=target_dbs, run_id=RUN_ID, speed_factor=0)
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
                    if g['name'] in [b'feature_group', 'feature_group']: fused_lag = g['pending']
                for g in r.xinfo_groups(STREAM_INFERENCE):
                    if g['name'] in [b'v8_orch_group', 'v8_orch_group']: orch_lag = g['pending']
                if r.exists(STREAM_TRADE_LOG):
                    for g in r.xinfo_groups(STREAM_TRADE_LOG):
                        if g['name'] in [b'persistence_group', 'persistence_group']: persist_lag = g['pending']
            except: pass

            status_raw = r.get(status_key)
            status_str = status_raw.decode('utf-8') if status_raw else ""

            if status_str == "DONE" and fused_lag == 0 and orch_lag == 0 and persist_lag == 0:
                elapsed = time.time() - start_time
                logger.info(f"\n🎉 ALL {len(target_dbs)} Databases Processed Successfully in {elapsed:.1f}s!")
                
                signal_task.cancel()
                try: await signal_task
                except asyncio.CancelledError: pass
                break
            
            # 每 10 秒打印一次心跳，防止刷屏
            if int(time.time()) % 10 == 0:
                logger.info(f"❤️ Heartbeat | Lag -> Fused:{fused_lag} Orch:{orch_lag} Persist:{persist_lag}")

    except KeyboardInterrupt:
        logger.warning("⚠️ Interrupted by user!")
        if 'signal_task' in locals(): signal_task.cancel()
        
    finally:
        if feature_process and feature_process.poll() is None:
            logger.info("🛑 Terminating Feature Compute Service...")
            feature_process.terminate()
            feature_process.wait()

if __name__ == "__main__":
    asyncio.run(main())
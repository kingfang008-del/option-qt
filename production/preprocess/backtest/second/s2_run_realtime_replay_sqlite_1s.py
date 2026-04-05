#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
文件名: s2_run_realtime_replay_sqlite_1s.py
描述: 全量 Alpha 批处理推理工厂 (1秒级高频纯净压测版)
核心改动: 适配 history_sqlite_1s，移除跨越分钟的聚合缓冲池，直接按秒帧 (23400帧/天) 极速发车！
"""
import os
os.environ['RUN_MODE'] = 'LIVEREPLAY'
import asyncio
import threading
import time
import logging
import sqlite3
import json
import uuid
import sys
import copy
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import pytz
import datetime as dt
import argparse
import subprocess
from config import TARGET_SYMBOLS
# 将项目根目录加入路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils import serialization_utils as ser
from datetime import datetime
# 导入业务组件
from signal_engine_v8 import SignalEngineV8
from execution_engine_v8 import ExecutionEngineV8
import signal_engine_v8
import execution_engine_v8
from mock_ibkr_historical import MockIBKRHistorical
import data_persistence_service_v8_sqlite

from config import (
    REDIS_CFG, PROJECT_ROOT, FEATURE_SERVICE_STATE_FILE,STREAM_ORCH_SIGNAL,GROUP_FEATURE,GROUP_ORCH,GROUP_OMS,GROUP_PERSISTENCE,
    STREAM_FUSED_MARKET, HASH_OPTION_SNAPSHOT, STREAM_INFERENCE, STREAM_TRADE_LOG
)
import redis

# 🚀 强制指向 1s 的高频数据库目录
DB_DIR_1S = PROJECT_ROOT / "data" / "history_sqlite_1s"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [1s_INFERENCE] - %(message)s')
logger = logging.getLogger("1s_BatchInference")

class BatchSQLiteDriver1s:
    """1秒级全量连续流发球机"""
    def __init__(self, target_dbs: list, run_id: str):
        self.target_dbs = sorted(target_dbs)
        self.run_id = run_id
        self.r = redis.Redis(**{k:v for k,v in REDIS_CFG.items() if k in ['host','port','db']})
    
    def run(self):
        total_dbs = len(self.target_dbs)
        logger.info(f"🚀 [Driver] Starting High-Freq Batch Inference (1s -> 1m Resampled) for {total_dbs} databases...")

        # 获取所有数据库列表，用于向用户展示上下文
        from config import STREAM_FUSED_MARKET, HASH_OPTION_SNAPSHOT
        from utils import serialization_utils as ser
        import json
        import time
        from tqdm import tqdm
        
        import pandas as pd
        
        # 强制使用传入的 db_dir 查找文件
        all_dbs = sorted([f for f in self.target_dbs[0].parent.glob("market_*.db") if len(f.stem) == 15])

        for idx, db_path in enumerate(self.target_dbs, 1):
            date_str = db_path.stem.split('_')[1]
            
            logger.info(f"\n" + "="*60)
            logger.info(f"📂 [{idx}/{total_dbs}] Processing: {db_path.name}")
            
            if idx == 1:
                prev_dbs = [f.name for f in all_dbs if f.stem.split('_')[1] < date_str][-3:]
                prev_dbs.reverse()
                if prev_dbs:
                    logger.info(f"🔄 [Warm Start] 引擎初始预热依赖的历史数据: {prev_dbs}")
                else:
                    logger.info(f"🧊 [Cold Start] {date_str} 之前无历史 DB，引擎将从零开始积累特征。")
            else:
                prev_date = self.target_dbs[idx-2].stem.split('_')[1]
                logger.info(f"🔥 [Hot Start] 引擎内存无缝接力，自动包含前一日 ({prev_date}) 尾盘特征，无需读盘！")
            logger.info("="*60)

            try:
                conn = sqlite3.connect(f"file:{db_path}?mode=rw", uri=True, timeout=60.0)
                c = conn.cursor()
                c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='alpha_logs'")
                if c.fetchone(): c.execute("DELETE FROM alpha_logs")
                c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='trade_logs'")
                if c.fetchone(): c.execute("DELETE FROM trade_logs")
                conn.commit()
                
                logger.info(f"📥 Fetching 1s high-freq data from SQLite for {date_str}...")
                df_bars_1s = pd.read_sql("SELECT symbol, ts, open, high, low, close, volume FROM market_bars_1s ORDER BY ts ASC", conn)
                df_opts_1s = pd.read_sql("SELECT symbol, ts, buckets_json FROM option_snapshots_1s ORDER BY ts ASC", conn)
                
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

            if df_bars_1s.empty:
                logger.warning(f"⚠️ No 1s data in {db_path.name}, skipping.")
                continue

            # =====================================================================
            # 🚀 [真 1s 高频发车]
            # 彻底抛弃 1min 聚合，直接使用秒级原力！
            # =====================================================================
            logger.info("⚡ Using raw 1s data for high-frequency resolution...")
            
            df_bars_1m = df_bars_1s.copy()
            df_bars_1m['ts_aligned'] = df_bars_1m['ts'].astype(float)
            
            df_opts_1m = df_opts_1s.copy()
            df_opts_1m['ts_aligned'] = df_opts_1m['ts'].astype(float)
            # =====================================================================

            if not df_bars_5m.empty: df_bars_5m['ts_aligned'] = df_bars_5m['ts'].astype(float)
            if not df_opts_5m.empty: df_opts_5m['ts_aligned'] = df_opts_5m['ts'].astype(float)

            # === [🚀 极速向量化重构: 彻底抛弃 df.iterrows()] ===
            def to_map(df, type_key):
                from collections import defaultdict
                m = defaultdict(dict)
                if df.empty: return m
                
                # 绕过 Pandas 慢速封装，直接抽离 numpy 数组
                ts_vals = df['ts_aligned'].values
                sym_vals = df['symbol'].values
                
                if type_key == 'bars':
                    o_vals, h_vals, l_vals, c_vals, v_vals = df['open'].values, df['high'].values, df['low'].values, df['close'].values, df['volume'].values
                    for ts, sym, o, h, l, c, v in zip(ts_vals, sym_vals, o_vals, h_vals, l_vals, c_vals, v_vals):
                        m[ts][sym] = {'open': o, 'high': h, 'low': l, 'close': c, 'volume': v}
                else: 
                    json_vals = df['buckets_json'].values
                    for ts, sym, b_json in zip(ts_vals, sym_vals, json_vals):
                        opt_data = b_json
                        if isinstance(b_json, str): 
                            try: opt_data = json.loads(b_json)
                            except: opt_data = {}
                        m[ts][sym] = opt_data
                return dict(m)

            logger.info("⚡ Vectorizing and mapping data structures...")
            map_b1 = to_map(df_bars_1m, 'bars')
            map_o1 = to_map(df_opts_1m, 'opts')
            map_b5 = to_map(df_bars_5m, 'bars')
            map_o5 = to_map(df_opts_5m, 'opts')

            all_ts = sorted(list(set(map_b1.keys()) | set(map_o1.keys()) | set(map_b5.keys()) | set(map_o5.keys())))
            logger.info(f"✅ Loaded {len(all_ts)} synchronized 1s ticks. Streaming...")

            # [Fix C] 初始化所有标的的“状态缓存”，实现断流补偿
            last_known_payloads = {sym: {'ts': 0, 'symbol': sym, 'stock': {'open': 0, 'high': 0, 'low': 0, 'close': 0, 'volume': 0}, 'option_buckets': [], 'option_contracts': []} for sym in TARGET_SYMBOLS}
            last_5m_state = {}
            count = 0
            
            for ts_val in tqdm(all_ts, desc=f"Inferring {db_path.name}"):
                self.r.set("replay:current_ts", str(ts_val)) 
                
                # 预获取当前时间戳的引用
                b1_ts = map_b1.get(ts_val, {})
                o1_ts = map_o1.get(ts_val, {})
                b5_ts = map_b5.get(ts_val, {})
                o5_ts = map_o5.get(ts_val, {})
                
                batch_payloads = []
                hset_mapping = {}  
                
                # [Fix C] 强制遍历所有目标标的，即使本秒无成交也发送“维持帧”
                for sym in TARGET_SYMBOLS:
                    # 1. 获取或创建基础 Payload (继承上一状态)
                    payload = last_known_payloads[sym]
                    payload['ts'] = ts_val
                    
                    # 2. 如果本秒有股票成交，更新 Open/High/Low/Close/Volume
                    if sym in b1_ts:
                        payload['stock'] = b1_ts[sym]
                    else:
                        # [重要] 若无成交，则价格继承上一秒 Close，成交量强制归 0
                        # 这样特征引擎收到的 price 不会突降到 0
                        if payload['stock']['close'] > 0:
                            payload['stock'] = payload['stock'].copy()
                            payload['stock']['volume'] = 0.0
                    
                    # 3. 如果本秒有期权快照，更新
                    if sym in o1_ts:
                        opt_data = o1_ts[sym]
                        if isinstance(opt_data, dict):
                            payload['option_buckets'] = opt_data.get('buckets', [])
                            payload['option_contracts'] = opt_data.get('contracts', [])
                        else:
                            payload['option_buckets'] = opt_data
                            payload['option_contracts'] = []
                        
                        opt_data_for_redis = opt_data if isinstance(opt_data, dict) else {'buckets': opt_data, 'ts': ts_val}
                        opt_data_for_redis['ts'] = ts_val
                        hset_mapping[sym] = ser.pack(opt_data_for_redis)

                    # 4. 5分钟线处理 (透传)
                    if sym in b5_ts or sym in o5_ts:
                        if sym not in last_5m_state: last_5m_state[sym] = {}
                        if sym in b5_ts:
                            last_5m_state[sym]['stock_5m'] = b5_ts[sym]
                        if sym in o5_ts:
                            opt_data_5m = o5_ts[sym]
                            if isinstance(opt_data_5m, dict):
                                last_5m_state[sym]['option_buckets_5m'] = opt_data_5m.get('buckets', [])
                                last_5m_state[sym]['option_contracts_5m'] = opt_data_5m.get('contracts', [])
                            else:
                                last_5m_state[sym]['option_buckets_5m'] = opt_data_5m
                                last_5m_state[sym]['option_contracts_5m'] = []
                    
                    if sym in last_5m_state:
                        payload.update(last_5m_state[sym])

                    # 5. 更新缓存并加入批处理
                    last_known_payloads[sym] = payload
                    batch_payloads.append(payload)
                
                # 网络 I/O 终极优化：一次性推送整个市场的 Hash 和 Stream
                if hset_mapping:
                    self.r.hset(HASH_OPTION_SNAPSHOT, mapping=hset_mapping)
                    
                if batch_payloads:
                    self.r.xadd(STREAM_FUSED_MARKET, {'batch': ser.pack(batch_payloads)})
                
                # === [FRAME SYNCHRONIZATION LOCK] ===
                timeout = 0
                while True:
                    # 使用 mget 并发拉取两把锁，消除额外的一倍网络往返时间
                    ack_feat, ack_orch = self.r.mget("sync:feature_calc_done", "sync:orch_done")
                    
                    feat_ts = float(ack_feat) if ack_feat else 0.0
                    orch_ts = float(ack_orch) if ack_orch else 0.0
                    
                    if feat_ts >= ts_val and orch_ts >= ts_val:
                        break

                    time.sleep(0.0005) 
                    timeout += 1
                    if timeout > 60000: # 30s
                        logger.warning(f"⚠️ [STALL] Sync Timeout at ts={ts_val}. Feat:{feat_ts} Orch:{orch_ts}")
                        break
                        
                count += 1
                
        status_key = f"replay:status:{self.run_id}"
        self.r.set(status_key, "DONE")
        logger.info(f"🏁 All Databases Processed. Status: {status_key}")

    async def _load_all_data_to_memory(self, start_ts, end_ts):
        """🚀 [Turbo Mode] 一次性载入全天 1s/5m 数据，消除 SQLite 瓶颈"""
        logger.info(f"💾 [Turbo] Loading all data from {start_ts} to {end_ts}...")
        
        # 1. 加载 1s 股票数据
        df_b1 = pd.read_sql(f"SELECT * FROM market_bars_1s WHERE ts >= {start_ts} AND ts <= {end_ts}", self.conn)
        map_b1 = {}
        for ts, group in df_b1.groupby('ts'):
            map_b1[float(ts)] = {row['symbol']: {
                'open': row['open'], 'high': row['high'], 'low': row['low'], 'close': row['close'], 'volume': row['volume']
            } for _, row in group.iterrows()}
            
        # 2. 加载 1s 期权快照 ( buckets 为 JSON, 包含 buckets 和 contracts )
        df_o1 = pd.read_sql(f"SELECT * FROM option_snapshots_1s WHERE ts >= {start_ts} AND ts <= {end_ts}", self.conn)
        map_o1 = {}
        for ts, group in df_o1.groupby('ts'):
            day_snaps = {}
            for _, row in group.iterrows():
                blob = json.loads(row['buckets_json']) if isinstance(row['buckets_json'], str) else {}
                day_snaps[row['symbol']] = {
                    'buckets': blob.get('buckets', []),
                    'contracts': blob.get('contracts', [])
                }
            map_o1[float(ts)] = day_snaps
            
        # 3. 加载 5m 股票和期权数据
        df_b5 = pd.read_sql(f"SELECT * FROM market_bars_5m WHERE ts >= {start_ts} AND ts <= {end_ts}", self.conn)
        map_b5 = {float(ts): {row['symbol']: row.to_dict() for _, row in group.iterrows()} for ts, group in df_b5.groupby('ts')}
        
        df_o5 = pd.read_sql(f"SELECT * FROM option_snapshots_5m WHERE ts >= {start_ts} AND ts <= {end_ts}", self.conn)
        map_o5 = {}
        for ts, group in df_o5.groupby('ts'):
            day_snaps = {}
            for _, row in group.iterrows():
                blob = json.loads(row['buckets_json']) if isinstance(row['buckets_json'], str) else {}
                day_snaps[row['symbol']] = {
                    'buckets': blob.get('buckets', []),
                    'contracts': blob.get('contracts', [])
                }
            map_o5[float(ts)] = day_snaps
        
        logger.info(f"✅ Loaded {len(map_o1)} snapshots in memory.")
        return map_b1, map_o1, map_b5, map_o5

    async def _wait_for_sync(self, ts_val, enable_oms=False):
        """[Redis Mode Only] 等待后续两个引擎完成本秒处理"""
        if getattr(self, 'turbo', False): return
        
        timeout = 0
        while True:
            # 使用 mget 并发拉取两把锁，消除额外的一倍网络往返时间
            ack_feat, ack_orch = self.r.mget("sync:feature_calc_done", "sync:orch_done")
            
            feat_ts = float(ack_feat) if ack_feat else 0.0
            orch_ts = float(ack_orch) if ack_orch else 0.0
            
            if feat_ts >= ts_val and orch_ts >= ts_val:
                break

            time.sleep(0.0005) 
            timeout += 1
            if timeout > 60000: # 30s
                logger.warning(f"⚠️ [STALL] Sync Timeout at ts={ts_val}. Feat:{feat_ts} Orch:{orch_ts}")
                break

    async def run_turbo(self, feat_svc, signal_svc):
        """🚀 [Turbo Mode] 极速进程内回放：Driver -> Feat -> Orch 瞬时直连"""
        logger.info("🔥 [Turbo Mode] Starting vectorized-speed in-process replay...")
        NY_TZ = pytz.timezone('America/New_York')

        if not self.target_dbs:
            logger.error(f"❌ No databases to process for Turbo Mode.")
            return

        for db_path in self.target_dbs:
            date_str = db_path.stem.split('_')[1]
            logger.info(f"📂 [Turbo] Processing Database: {db_path.name}")
            
            # 精准计算本库的时间窗口
            start_dt = NY_TZ.localize(datetime.strptime(date_str + " 09:30:00", "%Y%m%d %H:%M:%S"))
            end_dt = NY_TZ.localize(datetime.strptime(date_str + " 16:00:00", "%Y%m%d %H:%M:%S"))
            start_ts, end_ts = int(start_dt.timestamp()), int(end_dt.timestamp())

             
            self.conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
            map_b1, map_o1, map_b5, map_o5 = await self._load_all_data_to_memory(start_ts, end_ts)
            self.conn.close()
            del self.conn
            
            if not map_o1:
                logger.warning(f"⚠️ No snapshots found in {db_path.name} for RTH (09:30-16:00). Skipping...")
                continue
        
            # 准备 Turbo 专用引擎
            signal_svc.turbo_mode = True
            signal_svc.alpha_buffer = []

            # [Fix] Monkey-patch _emit_trade_log to intercept alpha logs
            def turbo_emit_log(payload):
                if payload.get('action') == 'ALPHA':
                    # [🔥 Debug] 打印 Alpha 值，方便确认是否生成
                    #print(f"📡 [Turbo_Alpha] {payload.get('symbol')} | Alpha: {payload.get('alpha'):.4f} | TS: {payload.get('ts')}")
                    signal_svc.alpha_buffer.append(payload)
                else:
                    # Non-alpha logs (trade logs) still go to the original handler
                    signal_svc.accounting._emit_trade_log(payload)
            
            signal_svc._emit_trade_log = turbo_emit_log
            all_ts = sorted(list(set(map_b1.keys()) | set(map_o1.keys()) | set(map_b5.keys()) | set(map_o5.keys())))
            # 🚀 [Parity Fix] 强制从整分钟开始（例如 10:00:00），剔除开头的残缺秒数，对齐 1m 历史回测基准
            if all_ts:
                first_full_min = ((all_ts[0] + 59) // 60) * 60
                logger.info(f"🎯 [Parity] Alignment: Data starts at {datetime.fromtimestamp(all_ts[0], NY_TZ)}, truncating to {datetime.fromtimestamp(first_full_min, NY_TZ)}")
                all_ts = [t for t in all_ts if t >= first_full_min]

                # [Clean] 清理存量数据，防止旧回测残留干扰对比
                try:
                    with sqlite3.connect(db_path) as clean_conn:
                        clean_conn.execute("DELETE FROM alpha_logs")
                        clean_conn.commit()
                        logger.info(f"🧹 [Clean] Truncated alpha_logs in {db_path.name}")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to clean alpha_logs: {e}")

            # [Fix C] Turbo 模式同步开启断流补偿
            last_known_payloads = {sym: {'ts': 0, 'symbol': sym, 'stock': {'open': 0, 'high': 0, 'low': 0, 'close': 0, 'volume': 0}, 'option_buckets': [], 'option_contracts': []} for sym in TARGET_SYMBOLS}
            last_5m_state = {}
            
            from tqdm import tqdm
            consecutive_05 = 0
            for ts_val in tqdm(all_ts, desc="Turbo Inference"):
                # 1. 组装 1s Tick Batch
                b1_ts = map_b1.get(ts_val, {})
                o1_ts = map_o1.get(ts_val, {})
                b5_ts = map_b5.get(ts_val, {})
                o5_ts = map_o5.get(ts_val, {})
                
                batch_payloads = []
                for sym in TARGET_SYMBOLS:
                    payload = last_known_payloads[sym]
                    payload['ts'] = ts_val
                    
                    if sym in b1_ts:
                        payload['stock'] = b1_ts[sym]
                    else:
                        if payload['stock']['close'] > 0:
                            payload['stock'] = payload['stock'].copy()
                            payload['stock']['volume'] = 0.0
                            
                    if sym in o1_ts: payload.update(o1_ts[sym])
                    
                    if sym in b5_ts or sym in o5_ts:
                        if sym not in last_5m_state: last_5m_state[sym] = {}
                        if sym in b5_ts: last_5m_state[sym]['stock_5m'] = b5_ts[sym]
                        if sym in o5_ts: last_5m_state[sym].update(o5_ts[sym])
                    
                    if sym in last_5m_state: payload.update(last_5m_state[sym])
                    
                    last_known_payloads[sym] = payload
                    batch_payloads.append(payload)
                
                # 2. 直连 Feature Engine (无 Redis)
                await feat_svc.process_market_data(batch_payloads)
                feat_payload = await feat_svc.run_compute_cycle(ts_from_payload=ts_val, return_payload=True)
                
                # 3. 直连 Signal Engine (无 Redis)
                if feat_payload:
                    await signal_svc.process_batch(feat_payload)
                    
                    # 🔍 [终极诊断哨兵]
                    if any(abs(a.get('iv', 0) - 0.5) < 1e-6 for a in signal_svc.alpha_buffer[-len(TARGET_SYMBOLS):]):
                        consecutive_05 += 1
                    else:
                        consecutive_05 = 0
                        
                    # 如果在热身 30 秒后，依然连续 5 秒观测到 0.5，立即中断并打印现场
                    if consecutive_05 > 5 and ts_val > all_ts[0] + 30: 
                        logger.critical(f"🔥 [IV_FATAL_SENTINEL] IV is STUCK at 0.5 at {ts_val}")
                        logger.critical(f"Sample Alpha Buffer Tail: {signal_svc.alpha_buffer[-1]}")
                        # 确认 Feature Service 传过来的数据
                        nvda_iv = "N/A"
                        if 'cheat_call_iv' in feat_payload:
                             nvda_idx = TARGET_SYMBOLS.index('NVDA') if 'NVDA' in TARGET_SYMBOLS else 0
                             nvda_iv = feat_payload['cheat_call_iv'][nvda_idx]
                        logger.info(f"Feat Payload NVDA IV: {nvda_iv}")
                        
                        # 检查 live_options 结构
                        if 'live_options' in feat_payload and 'NVDA' in feat_payload['live_options']:
                            nvda_snap = feat_payload['live_options']['NVDA']
                            if isinstance(nvda_snap, dict):
                                iv_val = nvda_snap['buckets'][2, 7] # CALL_ATM is index 2
                                logger.info(f"LiveOptions NVDA IV: {iv_val:.4f}")
                            else:
                                logger.info(f"LiveOptions NVDA structure: {type(nvda_snap)}")

                        raise RuntimeError("SENTINEL_INTERRUPT_IV_STUCK_0.5")
            
            # 4. 批量存入结果
            logger.info(f"💾 Directing {len(signal_svc.alpha_buffer)} Alphas to SQLite (Redis Bypass)...")
            from data_persistence_service_v8_sqlite import DataPersistenceServiceSQLite
            persist_svc = DataPersistenceServiceSQLite(db_dir=DB_DIR_1S, start_date=db_path.stem.split('_')[1])
            
            # 🚀 [核心优化] 不再经过 Redis，直接将内存中的 Alpha 数据传输给持久化服务
            for p in signal_svc.alpha_buffer:
                if p.get('action') == 'ALPHA':
                    persist_svc.alpha_buffer.append((
                        p['ts'], p['symbol'], 
                        float(p.get('alpha', 0)),
                        float(p.get('iv', 0)),
                        float(p.get('price', 0)),
                        float(p.get('vol_z', 0))
                    ))
                else:
                    # 对于交易日志 (Trade Log)，依然同步一份到持久化
                    mode = p.get('mode', 'LIVEREPLAY')
                    persist_svc.trade_buffer.append((
                        p['ts'], p['symbol'], p['action'],
                        float(p.get('qty', 0)), float(p.get('price', 0)), 
                        json.dumps(p),
                        mode
                    ))
            
            logger.info("💾 Triggering final flush to SQLite...")
            persist_svc.flush() 
            
            # [可选] 清除 Redis 这里的残留，防止下次干扰
            try:
                r.delete(STREAM_TRADE_LOG)
            except: pass
            
            logger.info("✅ Direct persistence completed (Zero-Loss).")
        
async def main():
    parser = argparse.ArgumentParser(description="Batch Alpha Inference Factory (1-Second Edition)")
    parser.add_argument('--start-date', type=str, default="20260101", help="Start processing from this date (YYYYMMDD)")
    parser.add_argument('--end-date', type=str, default=None, help="End processing at this date")
    parser.add_argument('--skip-warmup', action='store_true', help="Skip Feature Engine deep warmup for instant start")
    parser.add_argument('--enable-oms', action='store_true', help="Enable full OMS and Mock Broker (Backtest mode)")
    parser.add_argument('--turbo', action='store_true', help="🚀 [Turbo Mode] In-process execution (Redis bypass) for Alpha-only mode")
    args = parser.parse_args()

    if not DB_DIR_1S.exists():
        logger.error(f"❌ DB Directory not found: {DB_DIR_1S}")
        return

    # ================= 获取需要处理的数据库列表 =================
    all_dbs = sorted([f for f in DB_DIR_1S.glob("market_*.db") if f.stem.startswith("market_") and len(f.stem) == 15])
    
    if not all_dbs:
        logger.error(f"❌ No 1s databases found in {DB_DIR_1S}.")
        return

    # [优化] 如果用户没有指定 start_date (即使用默认值)，且默认值不存在数据，则从第一个有数据的 DB 开始
    available_dates = sorted([f.stem.split('_')[1] for f in all_dbs])
    
    start_date_val = args.start_date
    if start_date_val == "20260101" and start_date_val not in available_dates:
        start_date_val = available_dates[0]
        logger.info(f"📅 Default start date 20260101 not found. Auto-detecting first available: {start_date_val}")

    end_date_val = args.end_date if args.end_date else start_date_val
    
    target_dbs = []
    for db in all_dbs:
        date_str = db.stem.split('_')[1]
        if start_date_val <= date_str <= end_date_val:
            # 🛡️ 快速数据校验：确保数据库里不是空的
            try:
                with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as conn:
                    cur = conn.cursor()
                    cur.execute("SELECT COUNT(*) FROM market_bars_1s LIMIT 1")
                    if cur.fetchone()[0] > 0:
                        target_dbs.append(db)
                    else:
                        logger.warning(f"⚠️ Skipping empty database: {db.name}")
            except Exception as e:
                logger.warning(f"⚠️ Could not verify {db.name}: {e}")
            
    if not target_dbs:
        logger.error(f"❌ No valid 1s databases found between {start_date_val} and {end_date_val} in {DB_DIR_1S}.")
        logger.info(f"ℹ️ Available dates: {available_dates}")
        return
        
    logger.info(f"🔍 Found {len(target_dbs)} valid 1s databases to process: {[f.name for f in target_dbs]}")

    r = redis.Redis(**{k:v for k,v in REDIS_CFG.items() if k in ['host','port','db']})
    RUN_ID = str(uuid.uuid4())[:8]
    logger.info(f"🆔 Generated 1s Replay RUN_ID: {RUN_ID}")
    
    # 清理 Redis 流
    for stream in [STREAM_FUSED_MARKET, STREAM_INFERENCE, STREAM_ORCH_SIGNAL, STREAM_TRADE_LOG, HASH_OPTION_SNAPSHOT, f"replay:status:{RUN_ID}"]:
        r.delete(stream)

    for state_file in ["orchestrator_state.db", "positions_ignore_v8.json"]:
        path = PROJECT_ROOT / "data" / state_file if state_file.endswith(".db") else PROJECT_ROOT / state_file
        if path.exists():
            try: path.unlink()
            except: pass

    # 预建 Redis 流与组 (统一使用 config.py 定义的常量)
    streams_and_groups = {
        STREAM_FUSED_MARKET: [GROUP_FEATURE, GROUP_ORCH, GROUP_OMS, GROUP_PERSISTENCE],
        STREAM_INFERENCE:    [GROUP_ORCH, GROUP_PERSISTENCE],
        STREAM_ORCH_SIGNAL:  [GROUP_OMS, GROUP_PERSISTENCE],
        STREAM_TRADE_LOG:    [GROUP_PERSISTENCE]
    }

    for s_key, groups in streams_and_groups.items():
        try: 
            r.xadd(s_key, {'init': '1'}) 
        except: pass
        for g_name in groups:
            try: 
                r.xgroup_create(s_key, g_name, id='0', mkstream=True)
            except: pass

    # ================= 精准计算 REPLAY_START_TS =================
    NY_TZ = pytz.timezone('America/New_York')
    first_db_date = target_dbs[0].stem.split('_')[1]
    actual_start_date = args.start_date if args.start_date >= first_db_date else first_db_date
    target_dt = dt.datetime.strptime(actual_start_date, "%Y%m%d")
    target_dt = NY_TZ.localize(target_dt.replace(hour=9, minute=30, second=0))
    replay_start_ts = target_dt.timestamp()

    r.set("replay:current_ts", str(replay_start_ts))
    
    state_file = PROJECT_ROOT / FEATURE_SERVICE_STATE_FILE
    if state_file.exists(): state_file.unlink()

    # 启动特征引擎
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
                sys.stdout.write(f"[FEAT] {line}")
                sys.stdout.flush()
                if ("Complete" in line and "Warmup" in line) or "Deep Warmup Complete" in line or "Cold start" in line or "Skipped" in line:
                    ready_event.set()

        threading.Thread(target=monitor_stdout, daemon=True).start()

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
        execution_engine_v8.datetime = ReplayDatetime
        execution_engine_v8.time.time = replay_time
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
                        logger.warning(f"⚠️ SQLite Disk I/O Lock detected. Auto-Healing {attempt+1}/5...")
                        if hasattr(self, 'conn') and self.conn:
                            try: self.conn.close()
                            except: pass
                        time.sleep(1)
                    else: raise e
        data_persistence_service_v8_sqlite.DataPersistenceServiceSQLite._init_db = robust_init_db

        # [核心修复] 为各组件注入独立的 REDIS_CFG，防止全局变量污染
        # 1. 信号引擎 (Orch)
        orch_cfg = copy.deepcopy(REDIS_CFG)
        orch_cfg['group'] = GROUP_ORCH
        orch_cfg['input_stream'] = STREAM_INFERENCE
        signal_engine_v8.REDIS_CFG = orch_cfg # 强制补丁
        
        # ================= 🚀 [Turbo Mode 分支] =================
        # [V8.1 Mac 适配] 优先尝试工程相对路径，增强跨平台兼容性
        base_dir = Path(__file__).resolve().parent.parent.parent
        config_paths = {
            'fast': str(base_dir / "daily_backtest/fast_feature.json"), 
            'slow': str(base_dir / "daily_backtest/slow_feature.json")
        }
        
        # 兼容性 Fallback: 使用 PROJECT_ROOT 
        if not Path(config_paths['fast']).exists():
            config_paths['fast'] = str(PROJECT_ROOT / "config/fast_feature.json")
            config_paths['slow'] = str(PROJECT_ROOT / "config/slow_feature.json")

        model_paths = {
            'slow': str(PROJECT_ROOT / "checkpoints_advanced_alpha/advanced_alpha_best.pth"),
            'fast': str(PROJECT_ROOT / "checkpoints_advanced_alpha/fast_final_best.pth")
        }

        if args.turbo:
            logger.info("⚡ [Turbo Mode] Initializing in-process engines for maximum speed...")
            if args.skip_warmup:
                os.environ['SKIP_DEEP_WARMUP'] = '1'
            
            # 1. 初始化 Feature Engine (In-process)
           # 1. 初始化 Feature Engine (In-process)
            from feature_compute_service_v8 import FeatureComputeService
            # 🚀 [修复] 继承全局 REDIS_CFG，防止 FCS 内部找不到 raw_stream 和连接参数
            feat_cfg = copy.deepcopy(REDIS_CFG)
            feat_cfg['input_stream'] = STREAM_FUSED_MARKET
            feat_cfg['output_stream'] = STREAM_INFERENCE
            feat_svc = FeatureComputeService(feat_cfg, TARGET_SYMBOLS, config_paths)
            
            if not args.skip_warmup:
                logger.info("⏳ [Turbo] Performing Feature Engine warmup...")
                if hasattr(feat_svc, '_deep_warmup_rth_bars'):
                    feat_svc._deep_warmup_rth_bars() 
            
            # 2. 初始化 Signal Engine (In-process)
            signal_svc = SignalEngineV8(TARGET_SYMBOLS, mode='backtest', config_paths=config_paths, model_paths=model_paths)
            signal_svc.only_log_alpha = True
            
            # 3. 启动持久化 (依然需要，用来写结果)
            persistence_svc = data_persistence_service_v8_sqlite.DataPersistenceServiceSQLite()
            threading.Thread(target=persistence_svc.run, daemon=True).start()
            
            # 4. 执行极速回放
            driver = BatchSQLiteDriver1s(target_dbs=target_dbs, run_id=RUN_ID)
            await driver.run_turbo(feat_svc, signal_svc)
            
            logger.info("🏁 [Turbo Mode] Replay Completed.")
            return

        # ================= 🐢 [Standard Redis Mode] =================
        # 2. 特征引擎 (Feat) - 已经在内部各自的进程/线程中通过环境变量或显式配置处理
        
        # 3. 执行引擎 (OMS)
        oms = None
        mock_ibkr = None
        oms_task = None
        if args.enable_oms:
            logger.info("🔌 Injecting Mock IBKR & Execution Engine (OMS)...")
            mock_ibkr = MockIBKRHistorical()
            await mock_ibkr.connect()

            oms_cfg = copy.deepcopy(REDIS_CFG)
            oms_cfg['group'] = GROUP_OMS
            oms_cfg['input_stream'] = STREAM_ORCH_SIGNAL
            
            oms = ExecutionEngineV8(symbols=TARGET_SYMBOLS, mode='backtest')
            execution_engine_v8.REDIS_CFG = oms_cfg # 强制补丁 (给内部 run 循环读取信号流用)
            oms.ibkr = mock_ibkr
            oms.mock_cash = mock_ibkr.initial_capital
            
            # 🛡️ Trading ENABLED. System is now performing full 1s Split-Engine Replay.
            oms_task = asyncio.create_task(oms.run())
        else:
            logger.info("🕵️ Shadow System: ONLY_LOG_ALPHA mode (OMS disabled).")

        # 4. 持久化服务 (Persistence)
        persist_cfg = copy.deepcopy(REDIS_CFG)
        persist_cfg['group'] = GROUP_PERSISTENCE
        persist_cfg['consumer'] = 'sqlite_writer_1s' 
        data_persistence_service_v8_sqlite.REDIS_CFG = persist_cfg
        
        logger.info("💾 Starting Data Persistence Service...")
        data_persistence_service_v8_sqlite.DB_DIR = DB_DIR_1S # 🚀 FORCE 1s DIR
        persistence_svc = data_persistence_service_v8_sqlite.DataPersistenceServiceSQLite(start_date=args.start_date)
        threading.Thread(target=persistence_svc.run, daemon=True).start()

        # ================= 启动 Orchestrator (纯推理模式) =================
        config_paths = {
            'fast': str("/home/kingfang007/notebook/train/fast_feature.json"), 
            'slow': str("/home/kingfang007/notebook/train/slow_feature.json")
        }
        model_paths = {
            'slow': str("/home/kingfang007/quant_project/checkpoints_advanced_alpha/advanced_alpha_best.pth"),
            'fast': str("/home/kingfang007/quant_project/checkpoints_advanced_alpha/fast_final_best.pth")
        }

        
        logger.info("🛠️ Building V8 Signal Engine (1s Inference Factory Mode)...")
        signal_engine = SignalEngineV8(
            symbols=TARGET_SYMBOLS, 
            mode='backtest', 
            config_paths=config_paths, 
            model_paths=model_paths
        )
        signal_engine.only_log_alpha = not args.enable_oms # 🚀 [按需开启核心交易模式]
        
        oms = None
        mock_ibkr = None
        oms_task = None
        if args.enable_oms:
            logger.info("🔌 Injecting Mock IBKR & Execution Engine (OMS)...")
            mock_ibkr = MockIBKRHistorical()
            await mock_ibkr.connect()
            
            oms = ExecutionEngineV8(symbols=TARGET_SYMBOLS, mode='backtest')
            oms.ibkr = mock_ibkr
            oms.mock_cash = mock_ibkr.initial_capital
            
            # 🛡️ Trading ENABLED. System is now performing full 1s Split-Engine Replay.
            oms_task = asyncio.create_task(oms.run())
        else:
            logger.info("🕵️ Shadow System: ONLY_LOG_ALPHA mode (OMS disabled).")

        signal_task = asyncio.create_task(signal_engine.run())

        # ================= 启动流水线发球机 =================
        def _run_driver():
            try:
                driver = BatchSQLiteDriver1s(target_dbs=target_dbs, run_id=RUN_ID)
                driver.run()
            except Exception as e:
                logger.error(f"❌ 1s Driver Crashed: {e}")

        logger.info("⏳ Giving Orchestrator 2s to initialize...")
        await asyncio.sleep(2.0)

        threading.Thread(target=_run_driver, daemon=True).start()

        # ================= 全链路进度监控与收尾 =================
        logger.info("👀 Monitoring 1s Batch Progress...")
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
                logger.info(f"\n🎉 ALL {len(target_dbs)} 1s Databases Processed Successfully in {elapsed:.1f}s!")
                
                # 收尾与统计
                signal_task.cancel()
                if oms_task: oms_task.cancel()
                try: 
                    await signal_task
                    if oms_task: await oms_task
                except asyncio.CancelledError: pass
                
                if oms:
                    await oms.force_close_all()
                    await asyncio.sleep(1)
                    mock_ibkr.save_trades()
                    
                    # 🚀 [新增] 打印最终财务汇总
                    print("\n" + "="*60)
                    print("📊 FINAL BACKTEST PERFORMANCE SUMMARY (V8 SPLIT-ENGINE 1S)")
                    print("="*60)
                    oms.accounting.print_backtest_summary()
                    oms.accounting.print_counter_trend_summary()
                
                # 🚀 [新增] 最终检查 Alpha 是否成功入库 (Standard Mode)
                logger.info("🔍 Verifying alpha_logs in SQLite...")
                try:
                    # 检查最后一个数据库
                    last_db = target_dbs[-1]
                    # 使用 uri=True 需要 sqlite3 (由于顶层已 import，此处无需重复)
                    conn = sqlite3.connect(f"file:{last_db}?mode=ro", uri=True)
                    cur = conn.cursor()
                    cur.execute("SELECT COUNT(*) FROM alpha_logs")
                    count = cur.fetchone()[0]
                    conn.close()
                    if count == 0:
                        logger.error(f"❌ [VERIFY_FAILURE] alpha_logs is EMPTY in {last_db.name}!")
                        sys.exit(1)
                    else:
                        logger.info(f"✅ Verified: {count} alphas found in {last_db.name}")
                except Exception as e:
                    logger.warning(f"⚠️ Could not verify alpha_logs: {e}")
                
                break
            
            if int(time.time()) % 10 == 0:
                logger.info(f"❤️ 1s Heartbeat | Lag -> Fused:{fused_lag} Orch:{orch_lag} Persist:{persist_lag}")

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
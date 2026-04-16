#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
文件名: feature_compute_service_v8.py
描述: [V8 Persistence & Robustness]
核心升级:
    1. [State Persistence]: 定期保存 Normalizer 的均值/方差到磁盘，重启后指标不跳变。
    2. [History]: HISTORY_LEN 增加到 500，覆盖全天分钟线，保证 VWAP 准确。
    3. [Catch-up]: 优化追赶逻辑，防止重启后数据乱序。
    4. [Backfill]: 从 SQLite 预加载当日历史数据。
"""

import sys
import os
import asyncio

# [NEW] Add project root to sys.path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import serialization_utils as ser
import logging
import json
import redis
import pandas as pd
import numpy as np
import torch
import time
import psycopg2
import psycopg2
from config import PG_DB_URL
import os
import sys
from datetime import datetime, timedelta, timezone
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, List, Optional
import pytz
import concurrent.futures
try:
    from realtime_feature_engine import RealTimeFeatureEngine
except ImportError:
    print("❌ Missing realtime_feature_engine.py")

from config import (
    REDIS_CFG as _REDIS_BASE, NY_TZ,
    STREAM_FUSED_MARKET, STREAM_INFERENCE,
    FEATURE_SERVICE_STATE_FILE as STATE_FILE, DB_DIR,
    LOG_DIR, USE_5M_OPTION_DATA
)

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "FeatureService.log", mode='a', encoding='utf-8')
    ],
    force=True
)
logger = logging.getLogger("FeatService")

# Feature service-specific Redis config (extends base)
REDIS_CFG = {
    **_REDIS_BASE,
    'raw_stream': STREAM_FUSED_MARKET,
    'option_stream': 'option_data_stream',
    'output_stream': STREAM_INFERENCE,
    'group': 'compute_group',
    'consumer': 'worker_v8_prod',
}

class RollingWindowNormalizer:
    def __init__(self, feature_names: List[str], config_dicts: Dict[str, dict], window=2000, use_tanh=True):
        self.window = window
        self.use_tanh = use_tanh
        self.feature_names = feature_names
        
        self.raw_buffer = deque(maxlen=window)
        self.count = 0
        
        n = len(feature_names)
        self.last_mean = np.zeros(n, dtype=np.float32)
        self.last_std = np.ones(n, dtype=np.float32)
        
        # 排除归一化的特征名单
        self.bounded_features = {
            'session', 'day_of_week', 'hour', 'is_holiday', 'rsi_divergence',
            'rsi', 'k', 'd', 'adx', 'rvi', 'vw_delta', 'vp_corr_15',
            'minute', 'is_expiry', 'is_fed_meeting', 'stock_id', 'timestamp', 'date',
            'symbol', 'open', 'high', 'low', 'close', 'volume',
            'fast_vol', 'spy_roc_5min', 'qqq_roc_5min'
        }
        
        self.categorical_mask = np.zeros(n, dtype=bool)
        for i, name in enumerate(feature_names):
            cfg = config_dicts.get(name, {})
            should_normalize = True
            if name in self.bounded_features: should_normalize = False
            if cfg.get('type') == 'categorical': should_normalize = False
            if cfg.get('calc') == 'raw': should_normalize = False
            if name.startswith('label_'): should_normalize = False
            
            if not should_normalize:
                self.categorical_mask[i] = True

    # --- 状态存取 ---
    def normalize_only(self, x_raw_1d: np.ndarray) -> np.ndarray:
        """
        [新增] 仅归一化，不更新 Buffer 和统计量 (用于盘前/盘后数据)
        """
        if not np.isfinite(x_raw_1d).all():
            x_raw_1d = np.nan_to_num(x_raw_1d, nan=0.0, posinf=0.0, neginf=0.0)

        # 直接使用现有统计量
        x_norm = (x_raw_1d - self.last_mean) / (self.last_std + 1e-6)
        x_norm = np.nan_to_num(x_norm, nan=0.0, posinf=10.0, neginf=-10.0)
        
        if self.use_tanh:
            real_mask = ~self.categorical_mask
            x_norm[real_mask] = np.tanh(x_norm[real_mask] / 3.0)
        else:
            x_norm = np.clip(x_norm, -10.0, 10.0)
            
        return x_norm

    def get_state(self):
        return {
            'buffer': list(self.raw_buffer),
            'mean': self.last_mean,
            'std': self.last_std,
            'count': self.count
        }

    def set_state(self, state):
        if not state: return
        self.raw_buffer = deque(state.get('buffer', []), maxlen=self.window)
        self.last_mean = state.get('mean', self.last_mean)
        self.last_std = state.get('std', self.last_std)
        self.count = state.get('count', 0)

    def process_frame(self, x_raw_1d: np.ndarray) -> np.ndarray:
        if not np.isfinite(x_raw_1d).all():
            x_raw_1d = np.nan_to_num(x_raw_1d, nan=0.0, posinf=0.0, neginf=0.0)

        self.raw_buffer.append(x_raw_1d)
        self.count += 1
        
        # 定期更新统计量
        if self.count < 100 or self.count % 10 == 0:
            if len(self.raw_buffer) >= 2:
                arr = np.array(self.raw_buffer, dtype=np.float64) 
                self.last_mean = np.mean(arr, axis=0).astype(np.float32)
                self.last_std = np.std(arr, axis=0).astype(np.float32)
                
                self.last_std[self.last_std < 1e-6] = 1.0
                self.last_mean[self.categorical_mask] = 0.0
                self.last_std[self.categorical_mask] = 1.0

        x_norm = (x_raw_1d - self.last_mean) / (self.last_std + 1e-6)
        x_norm = np.nan_to_num(x_norm, nan=0.0, posinf=10.0, neginf=-10.0)
        
        if self.use_tanh:
            real_mask = ~self.categorical_mask
            x_norm[real_mask] = np.tanh(x_norm[real_mask] / 3.0)
        else:
            x_norm = np.clip(x_norm, -10.0, 10.0)
            
        return x_norm

    def get_sequence(self, seq_len: int) -> np.ndarray:
        if len(self.raw_buffer) == 0:
            return np.zeros((seq_len, len(self.last_mean)), dtype=np.float32)
            
        full_data = np.array(self.raw_buffer, dtype=np.float32)
        if len(full_data) >= seq_len: data = full_data[-seq_len:]
        else: data = full_data
            
        x_norm = (data - self.last_mean) / (self.last_std + 1e-9)
        x_norm = np.nan_to_num(x_norm, nan=0.0, posinf=10.0, neginf=-10.0)
        
        if self.use_tanh:
            real_mask = ~self.categorical_mask
            x_norm[:, real_mask] = np.tanh(x_norm[:, real_mask] / 3.0)
        else:
            x_norm = np.clip(x_norm, -10.0, 10.0)
            
        if len(x_norm) < seq_len:
            pad_len = seq_len - len(x_norm)
            padding = np.zeros((pad_len, x_norm.shape[1]), dtype=np.float32)
            x_norm = np.vstack([padding, x_norm])
            
        return x_norm

class FeatureComputeService:
    def __init__(self, redis_cfg, symbols, config_paths):
        print(f"init Service... Symbols: {len(symbols)}")
        self.redis_cfg = redis_cfg
        self.symbols = symbols
        self.stock_id_map = {s: i for i, s in enumerate(symbols)}
        self.sector_id_map = {} 
        
        self.r = redis.Redis(**{k:v for k,v in redis_cfg.items() if k in ['host','port','db']})
        self._init_consumer_groups()
        
        # [Performance] Persistent PG Connection
        from config import PG_DB_URL
        try:
            self.pg_conn = psycopg2.connect(PG_DB_URL)
            self.pg_conn.autocommit = True
            logger.info("🐘 Persistent PostgreSQL connection established.")
        except Exception as e:
            logger.warning(f"❌ Failed to establish persistent PG connection: {e}")
            self.pg_conn = None
        
        self.last_cum_volume = {s: np.zeros(6, dtype=np.float32) for s in symbols}
        
        # Load Configs
        self.fast_feat_infos = self._load_json_info(config_paths['fast'])
        self.slow_feat_infos = self._load_json_info(config_paths['slow'])
        
        # 🚀 [防弹修复 1] 强制列表去重，防止 JSON 中误写重复特征导致 PG 报 DuplicateColumn
        self.fast_feat_names = list(dict.fromkeys([x['name'] for x in self.fast_feat_infos]))
        self.slow_feat_names = list(dict.fromkeys([x['name'] for x in self.slow_feat_infos]))


        self.all_feat_names = sorted(list(set(self.fast_feat_names + self.slow_feat_names )))
        self.feat_config_dict = {x['name']: x for x in (self.fast_feat_infos + self.slow_feat_infos)}
        
        # Indices
        self.feat_name_to_idx = {name: i for i, name in enumerate(self.all_feat_names)}
        self.fast_indices = [self.feat_name_to_idx[name] for name in self.fast_feat_names]

        self.feat_resolutions = {name: x.get('resolution', '1min') for name, x in self.feat_config_dict.items()}
        
        # =========================================================
        # 🚀 [动态分辨率解耦] 按 resolution 自动分组，拒绝硬编码
        # =========================================================
        self.slow_resolution_groups = defaultdict(list)
        for name in self.slow_feat_names:
            res = self.feat_resolutions.get(name, '1min')
            self.slow_resolution_groups[res].append(name)
            
        self.slow_resolution_indices = {
            res: [self.feat_name_to_idx[n] for n in names]
            for res, names in self.slow_resolution_groups.items()
        }
        # 🚀 [核心修复] 补上所有 slow 特征的全局索引，供组装给模型的 Tensor 使用！
        self.slow_indices = [self.feat_name_to_idx[name] for name in self.slow_feat_names]
        
        logger.info(f"📊 Feature Service Dynamic Indexing Complete:")
        logger.info(f"   - Fast Features: {len(self.fast_feat_names)}")
        for res, names in self.slow_resolution_groups.items():
            logger.info(f"   - Slow Features ({res}): {len(names)}")
        
        # DB & Engine

        # DB & Engine
        
        # =========================================================
        # [新增] 专属的后台单线程池，用于异步写入 SQLite
        # max_workers=1 既不阻塞主异步循环，又能完美避开 SQLite 并发锁死报错
        # =========================================================
        self.debug_db_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        
        # [Fix] Initialize Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if str(self.device) == 'cpu' and torch.backends.mps.is_available():
            self.device = torch.device('mps')
            
        self.engine = RealTimeFeatureEngine(stats_path=None, device=str(self.device))
        
        # Normalizers
        self.normalizers = {
            s: RollingWindowNormalizer(
                self.all_feat_names, 
                self.feat_config_dict, 
                window=2000, 
                use_tanh=True
            ) for s in symbols
        }

        # [修改] 历史窗口增加到 500，确保全天 VWAP 准确
        self.HISTORY_LEN = 500
        self.history_1min = {s: pd.DataFrame() for s in symbols}
        self.history_5min = {s: pd.DataFrame() for s in symbols}
        self.current_bars_5s = defaultdict(list)
        self.option_snapshot = {s: np.zeros((6, 12), dtype=np.float32) for s in symbols}
        self.option_snapshot_5m = {s: np.zeros((6, 12), dtype=np.float32) for s in symbols}
        self.latest_prices = {s: 0.0 for s in symbols}
        self.deriv_history = {s: deque(maxlen=10) for s in symbols}
        self.last_compute_ts = None
        self.warmup_needed = {s: True for s in symbols}
        self.warmup_needed_5m = {s: True for s in symbols}
        self.last_cum_volume_5m = {s: np.zeros(6, dtype=np.float32) for s in symbols}
        
        self.msg_count = 0
        self.last_log_time = time.time()
        self.last_wait_log_time = 0
        self.state_file = Path(STATE_FILE)

        self.latest_opt_buckets = {}
        self.latest_opt_contracts = {}


        # 尝试加载状态
        self._load_service_state()


    def _get_pg_conn(self):
        """获取 PostgreSQL 连接 (优先重用持久连接)"""
        import psycopg2
        from config import PG_DB_URL
        if hasattr(self, 'pg_conn') and self.pg_conn and not self.pg_conn.closed:
            return self.pg_conn
        try:
            self.pg_conn = psycopg2.connect(PG_DB_URL)
            self.pg_conn.autocommit = True
            return self.pg_conn
        except:
            return psycopg2.connect(PG_DB_URL)
 
    def _ensure_debug_tables(self, date_str):
        """[主线程同步建表 & 分区化] 升级为 PARTITION BY RANGE 模式"""
        import psycopg2
        from config import PG_DB_URL
        from datetime import datetime as _dt, timedelta
        
        # 计算分区时间戳范围 (NY 时间 00:00:00 -> 次日 00:00:00)
        day = _dt.strptime(date_str, '%Y%m%d')
        start_dt = NY_TZ.localize(day)
        end_dt = start_dt + timedelta(days=1)
        start_ts = start_dt.timestamp()
        end_ts = end_dt.timestamp()

        conn = None
        try:
            conn = psycopg2.connect(PG_DB_URL)
            conn.autocommit = True
            c = conn.cursor()
            
            # Helper: 获取表的所有列名
            def get_existing_cols(table_name):
                c.execute(f"SELECT column_name FROM information_schema.columns WHERE table_name = '{table_name}'")
                return [r[0] for r in c.fetchall()]

            # ---------------- 1. Fast Debug (Master) ----------------
            c.execute("""
                CREATE TABLE IF NOT EXISTS debug_fast (
                    ts DOUBLE PRECISION, symbol TEXT, created_at TEXT,
                    PRIMARY KEY (ts, symbol)
                ) PARTITION BY RANGE (ts);
            """)
            
            # 主表扩列 (会自动同步到分区)
            existing_cols_fast = get_existing_cols("debug_fast")
            for name in self.fast_feat_names:
                if name not in existing_cols_fast:
                    c.execute(f'ALTER TABLE debug_fast ADD COLUMN "{name}" DOUBLE PRECISION')
            
            # 创建日分区
            part_fast = f"debug_fast_{date_str}"
            c.execute(f"""
                CREATE TABLE IF NOT EXISTS {part_fast} PARTITION OF debug_fast
                FOR VALUES FROM ({start_ts}) TO ({end_ts});
            """)

            # ---------------- 2. Slow Debug (Master) ----------------
            c.execute("""
                CREATE TABLE IF NOT EXISTS debug_slow (
                    ts DOUBLE PRECISION, symbol TEXT, created_at TEXT,
                    PRIMARY KEY (ts, symbol)
                ) PARTITION BY RANGE (ts);
            """)
            
            # 主表扩列
            existing_cols_slow = get_existing_cols("debug_slow")
            for name in self.slow_feat_names:
                if name not in existing_cols_slow:
                    c.execute(f'ALTER TABLE debug_slow ADD COLUMN "{name}" DOUBLE PRECISION')
            
            # 创建日分区
            part_slow = f"debug_slow_{date_str}"
            c.execute(f"""
                CREATE TABLE IF NOT EXISTS {part_slow} PARTITION OF debug_slow
                FOR VALUES FROM ({start_ts}) TO ({end_ts});
            """)
                        
            logger.info(f"✅ Debug 表分区化确认完成: debug_fast_{date_str}, debug_slow_{date_str}")
        except Exception as e:
            logger.error(f"❌ Debug Postgres 分区化建表失败: {e}")
        finally:
            if conn: conn.close()
     
    def _write_debug_batch(self, ts, date_str, fast_data_list, slow_data_list):
        """[纯净写入版] 恢复宽表直接写入"""
        if not fast_data_list and not slow_data_list: return
        
        import psycopg2
        from config import PG_DB_URL
        from datetime import datetime
        from config import NY_TZ
        
        conn = None
        try:
            conn = psycopg2.connect(PG_DB_URL)
            conn.autocommit = True
            c = conn.cursor()
            
            dt_ny = datetime.fromtimestamp(ts, NY_TZ)
            created_at = dt_ny.strftime("%Y-%m-%d %H:%M:%S")
            
            # 1. 写入 Fast
            if fast_data_list:
                table_name = f"debug_fast_{date_str}"
                cols_str = "ts, symbol, created_at, " + ", ".join([f'"{name}"' for name in self.fast_feat_names])
                placeholders = ",".join(["%s"] * (3 + len(self.fast_feat_names)))
                
                rows_fast = []
                for sym, vals in fast_data_list:
                    clean_vals = [None if not np.isfinite(v) else float(v) for v in vals]
                    rows_fast.append([ts, sym, created_at] + clean_vals)
                if rows_fast:
                    c.executemany(f"INSERT INTO {table_name} ({cols_str}) VALUES ({placeholders}) ON CONFLICT (ts, symbol) DO NOTHING", rows_fast)
            
            # 2. 写入 Slow 宽表
            if slow_data_list:
                table_name = f"debug_slow_{date_str}"
                cols_str = "ts, symbol, created_at, " + ", ".join([f'"{name}"' for name in self.slow_feat_names])
                placeholders = ",".join(["%s"] * (3 + len(self.slow_feat_names)))
                
                rows_slow = []
                for sym, vals in slow_data_list:
                    clean_vals = [None if not np.isfinite(v) else float(v) for v in vals]
                    rows_slow.append([ts, sym, created_at] + clean_vals)
                    
                if rows_slow:
                    c.executemany(f"INSERT INTO {table_name} ({cols_str}) VALUES ({placeholders}) ON CONFLICT (ts, symbol) DO NOTHING", rows_slow)
            
            c.close()
        except Exception as e:
            if "does not exist" in str(e):
                logger.warning(f"⚠️ [跳过写入] 表 {date_str} 尚未建好: {e}")
            else:
                logger.error(f"❌ DB 写入报错: {e}")
        finally:
            if conn: conn.close()

    # --- 持久化方法 ---
    def _save_service_state(self):
        try:
            state = {
                'ts': time.time(),
                'normalizers': {s: norm.get_state() for s, norm in self.normalizers.items()}
            }
            temp_file = self.state_file.with_suffix('.tmp')
            with open(temp_file, 'wb') as f:
                f.write(ser.pack(state))
            if self.state_file.exists(): os.remove(self.state_file)
            os.rename(temp_file, self.state_file)
        except Exception as e:
            logger.error(f"Save State Error: {e}")

    def _load_service_state(self):
        """尝试从磁盘加载 Normalizer 状态，若失败、跨日或样本不足则启动深度预热"""
        if not self.state_file.exists():
            logger.info("ℹ️ No state file found, starting fresh (Deep Warmup).")
            # 状态丢失，启动深度预热
            self._warmup_from_history(replay_features=True)
            return

        try:
            with open(self.state_file, 'rb') as f:
                state = ser.unpack(f.read())
            
            # 严格日内检查 (Strict Intraday)
            state_time = datetime.fromtimestamp(state['ts'], NY_TZ)
            now_ny = datetime.now(NY_TZ)
            
            if state_time.date() != now_ny.date():
                logger.warning(f"⚠️ State file is from previous day ({state_time.date()}), starting fresh (Deep Warmup).")
                self._warmup_from_history(replay_features=True) 
                return
            
            # 恢复 Normalizer 的均值/方差状态
            is_fully_warmed = True
            for sym, sub_state in state.get('normalizers', {}).items():
                if sym in self.normalizers:
                    self.normalizers[sym].set_state(sub_state)
                    # =======================================================
                    # [🔥 核心验证] 就算状态是今天的，如果吃的样本数太少(比如只有16条)，
                    # 算出来的 Z-Score 也是乱飞的！必须判定为未完成预热！
                    # =======================================================
                    if self.normalizers[sym].count < 500:
                        is_fully_warmed = False
            
            if not is_fully_warmed:
                logger.warning("⚠️ State file is from today but INCOMPLETE (count < 500). Forcing Deep Warmup!")
                self._warmup_from_history(replay_features=True)
            else:
                logger.info(f"♻️ Service State Restored from disk (Fully Warmed). Loading base history...")
                self._warmup_from_history(replay_features=False)
            
        except Exception as e:
            logger.error(f"❌ State Load Error: {e}")
            self._warmup_from_history(replay_features=True)

    def _warmup_from_history(self, replay_features=True):
        """
        [统一跨日强预热与回填]
        参数:
            replay_features (bool): 是否将数据推入 Normalizer 重建 2000 窗口的 Z-Score 状态。
        """
        logger.info(f"🔥 Starting Warmup from SQLite (Replay Features={replay_features})...")
        from config import DB_DIR, NY_TZ
        from datetime import datetime, time as dt_time
        import os 
        
        target_bars = 2000 if replay_features else self.HISTORY_LEN
        
        # 获取当前引擎被设定的虚拟时间起点
        replay_start_ts = float(os.environ.get('REPLAY_START_TS', 'inf'))
        if replay_start_ts != float('inf'):
            logger.info(f"🛡️ Time Truncation Active: Ignoring DB records >= {replay_start_ts}")
            target_date_str = datetime.fromtimestamp(replay_start_ts, NY_TZ).strftime('%Y%m%d')
        else:
            target_date_str = datetime.now(NY_TZ).strftime('%Y%m%d')

        # ==========================================================
        # [核心拦截] 从环境变量获取发球机的起始时间，默认为无穷大(实盘不受影响)
        # ==========================================================
        # ==========================================================
        # 🚀 [终极优化] 使用批量查询，合并 N 个标的的查询为 1 个，极大提升预热速度
        if os.environ.get('SKIP_DEEP_WARMUP') == '1':
            logger.info("⏭️ [Warmup] Skipped Deep Warmup by environment flag.")
            return

        from collections import defaultdict
        sym_to_rows_1m = defaultdict(list)
        sym_to_rows_5m = defaultdict(list)
        sym_to_opt_row = {}
        sym_to_opt_row_5m = {}

        warmup_count = 0
        try:
            conn = self._get_pg_conn()
            c = conn.cursor()
            all_symbols = tuple(self.symbols)
            
            # 1. 批量抓取 1min 预热数据
            logger.info(f"📥 Bulk Loading 1m bars for {len(all_symbols)} symbols...")
            c.execute("""
                SELECT symbol, ts, open, high, low, close, volume 
                FROM market_bars_1m 
                WHERE symbol IN %s AND ts < %s 
                ORDER BY ts DESC
            """, (all_symbols, replay_start_ts))
            
            count_1m = 0
            for r in c.fetchall():
                sym = r[0]
                if len(sym_to_rows_1m[sym]) < (target_bars + 500):
                    sym_to_rows_1m[sym].append(r[1:])
                    count_1m += 1
            
            # 2. 批量抓取 5min 预热数据 
            logger.info(f"📥 Bulk Loading 5m bars...")
            c.execute("""
                SELECT symbol, ts, open, high, low, close, volume 
                FROM market_bars_5m 
                WHERE symbol IN %s AND ts < %s 
                ORDER BY ts DESC
            """, (all_symbols, replay_start_ts))
            for r in c.fetchall():
                sym = r[0]
                if len(sym_to_rows_5m[sym]) < 500:
                    sym_to_rows_5m[sym].append(r[1:])

            # 3. 批量抓取期权快照 (1m)
            logger.info(f"📥 Bulk Loading option snapshots...")
            c.execute("""
                SELECT symbol, buckets_json, ts
                FROM option_snapshots_1m 
                WHERE symbol IN %s AND ts < %s 
                ORDER BY ts DESC
            """, (all_symbols, replay_start_ts))
            for r in c.fetchall():
                if r[0] not in sym_to_opt_row:
                    sym_to_opt_row[r[0]] = r[1]

            if USE_5M_OPTION_DATA:
                c.execute("""
                    SELECT symbol, buckets_json, ts
                    FROM option_snapshots_5m 
                    WHERE symbol IN %s AND ts < %s 
                    ORDER BY ts DESC
                """, (all_symbols, replay_start_ts))
                for r in c.fetchall():
                    if r[0] not in sym_to_opt_row_5m:
                        sym_to_opt_row_5m[r[0]] = r[1]

            # --- 开始正式分发处理 ---
            for sym in self.symbols:
                # 1. 1min 预热解析
                rows = sym_to_rows_1m.get(sym, [])
                raw_bars_1m = []
                for r in rows:
                    dt_ny = datetime.fromtimestamp(r[0], NY_TZ).replace(second=0, microsecond=0)
                    t = dt_ny.time()
                    if dt_time(9, 30) <= t < dt_time(16, 0):
                        raw_bars_1m.append({'ts': dt_ny, 'open': r[1], 'high': r[2], 'low': r[3], 'close': r[4], 'volume': r[5]})
                
                if raw_bars_1m:
                    df_hist_1m = pd.DataFrame(raw_bars_1m).drop_duplicates(subset=['ts']).set_index('ts').sort_index()
                    self.history_1min[sym] = df_hist_1m.iloc[-self.HISTORY_LEN:]

                # 2. 5min 预热解析
                rows_5m = sym_to_rows_5m.get(sym, [])
                raw_bars_5m = []
                for r in rows_5m:
                    dt_ny = datetime.fromtimestamp(r[0], NY_TZ).replace(second=0, microsecond=0)
                    t = dt_ny.time()
                    if dt_time(9, 30) <= t < dt_time(16, 0):
                        raw_bars_5m.append({'ts': dt_ny, 'open': r[1], 'high': r[2], 'low': r[3], 'close': r[4], 'volume': r[5]})
                
                if raw_bars_5m:
                    df_hist_5m = pd.DataFrame(raw_bars_5m).drop_duplicates(subset=['ts']).set_index('ts').sort_index()
                    self.history_5min[sym] = df_hist_5m.iloc[-500:]

                # 3. 期权快照恢复
                opt_snap = sym_to_opt_row.get(sym)
                if opt_snap: 
                    try:
                        if isinstance(opt_snap, str): opt_snap = json.loads(opt_snap)
                        buckets = opt_snap.get('buckets', [])
                        arr = np.array(buckets, dtype=np.float32)
                        if arr.shape[0] < 6: arr = np.vstack([arr, np.zeros((6 - arr.shape[0], arr.shape[1]), dtype=np.float32)])
                        if arr.shape[1] < 12: arr = np.hstack([arr, np.zeros((arr.shape[0], 12 - arr.shape[1]), dtype=np.float32)])
                        self.option_snapshot[sym] = arr[:, :12]
                        if np.sum(arr[:, 6]) > 0:
                            self.last_cum_volume[sym] = arr[:, 6].copy()
                            self.warmup_needed[sym] = False
                    except: pass
                
                if USE_5M_OPTION_DATA:
                    opt_snap_5m = sym_to_opt_row_5m.get(sym)
                    if opt_snap_5m:
                        try:
                            if isinstance(opt_snap_5m, str): opt_snap_5m = json.loads(opt_snap_5m)
                            buckets = opt_snap_5m.get('buckets', [])
                            arr = np.array(buckets, dtype=np.float32)
                            if arr.shape[0] < 6: arr = np.vstack([arr, np.zeros((6 - arr.shape[0], arr.shape[1]), dtype=np.float32)])
                            if arr.shape[1] < 12: arr = np.hstack([arr, np.zeros((arr.shape[0], 12 - arr.shape[1]), dtype=np.float32)])
                            self.option_snapshot_5m[sym] = arr[:, :12]
                            if np.sum(arr[:, 6]) > 0:
                                self.last_cum_volume_5m[sym] = arr[:, 6].copy()
                                self.warmup_needed_5m[sym] = False
                        except: pass
                else:
                    logger.info(f"⏭️ [Warmup] Skipping 5m option snapshot for {sym} (Disabled by config).")

                # 4. Normalizer 重演 (针对 1min)
                if replay_features and not self.history_1min[sym].empty and len(self.history_1min[sym]) > 50:
                    sliced_snaps = {s: snap[:, :12] for s, snap in self.option_snapshot.items()}
                    res = self.engine.compute_all_inputs(
                        {sym: self.history_1min[sym]}, self.fast_feat_names, self.slow_feat_names, sliced_snaps, skip_scaling=True
                    )
                    if sym in res:
                        t_fast = res[sym]['fast_1m'][0].cpu().numpy()
                        t_slow = res[sym]['slow_1m'][0].cpu().numpy()
                        L = t_fast.shape[1]
                        norm = self.normalizers[sym]
                        for i in range(L):
                            raw_vec = np.zeros(len(self.all_feat_names), dtype=np.float32)
                            for k, fname in enumerate(self.fast_feat_names):
                                idx = self.feat_name_to_idx.get(fname)
                                if idx is not None: raw_vec[idx] = t_fast[k, i]
                            for k, fname in enumerate(self.slow_feat_names):
                                idx = self.feat_name_to_idx.get(fname)
                                if idx is not None: raw_vec[idx] = t_slow[k, i]
                            norm.process_frame(raw_vec)
                warmup_count += 1
                        
        except Exception as e:
            logger.error(f"Backfill Error: {e}", exc_info=True)
            
        logger.info(f"✅ Deep Warmup Complete for {warmup_count} symbols (Dual Resolution).")
        self._publish_warmup_status()

    def _robust_backfill_and_warmup(self):
        """[跨日强预热] 自动加载跨越 5.5 天 (约2500根) K线，瞬间预热 2000 窗口的 Normalizer"""
        logger.info("🔄 [CRITICAL] Starting 5-Day Backfill & Normalizer Warmup (Window=2000)...")
        from config import NY_TZ
        from datetime import datetime, time as dt_time
        import time

        warmup_count = 0
        try:
            conn = self._get_pg_conn()
            c = conn.cursor()
            for sym in self.symbols:
                raw_bars = []
                opt_snap = None
                
                # 抓取约 5.5 天的数据 (如果存在的话), 限制上限
                c.execute("SELECT ts, open, high, low, close, volume FROM market_bars_1m WHERE symbol=%s ORDER BY ts DESC LIMIT 2500", (sym,))
                rows = c.fetchall()
                
                c.execute("SELECT buckets_json FROM option_snapshots_1m WHERE symbol=%s ORDER BY ts DESC LIMIT 1", (sym,))
                opt_row = c.fetchone()
                if opt_row: 
                    opt_snap = opt_row[0]
                    if isinstance(opt_snap, str):
                        opt_snap = json.loads(opt_snap)
                
                # 过滤 RTH (09:30-16:00)
                for r in rows:
                    dt_ny = datetime.fromtimestamp(r[0], NY_TZ)
                    t = dt_ny.time()
                    if dt_time(9, 30) <= t < dt_time(16, 0):
                        raw_bars.append({'ts': dt_ny, 'open': r[1], 'high': r[2], 'low': r[3], 'close': r[4], 'volume': r[5]})
                        
                if not raw_bars: continue

            # 2. 按时间正序重排，并提取最后 2000 根用于 Normalizer 预热
            raw_bars = sorted(raw_bars, key=lambda x: x['ts'])
            if len(raw_bars) > 2000: raw_bars = raw_bars[-2000:]
            
            df_hist = pd.DataFrame(raw_bars).drop_duplicates(subset=['ts']).set_index('ts')
            
            # 将最后 HISTORY_LEN (500) 存入实盘滚动 Buffer 供引擎计算 SMA_200 等
            self.history_1min[sym] = df_hist.iloc[-self.HISTORY_LEN:] 
            
            # 恢复期权快照
            if opt_snap:
                try:
                    buckets = opt_snap.get('buckets', []) if isinstance(opt_snap, dict) else opt_snap
                    arr = np.array(buckets, dtype=np.float32)
                    if arr.shape[0] < 6: arr = np.vstack([arr, np.zeros((6 - arr.shape[0], 10), dtype=np.float32)])
                    self.option_snapshot[sym] = arr
                    if np.sum(arr[:, 6]) > 0:
                        self.last_cum_volume[sym] = arr[:, 6].copy()
                        self.warmup_needed[sym] = False
                except: pass

            # ==========================================================
            # 3. 核心大招：向量化计算 2000 根线的特征，秒级填满 Normalizer
            # ==========================================================
            if len(df_hist) > 50:
                # 创建前 8 列的切片
                sliced_snaps = {s: snap[:, :12] for s, snap in self.option_snapshot.items()}

                # 把整整 2000 根线一把丢给引擎算！
                res = self.engine.compute_all_inputs(
                    {sym: df_hist}, self.fast_feat_names, self.slow_feat_names, sliced_snaps, skip_scaling=True
                )
                
                if sym in res:
                    # 获取 [1, N_feats, 2000] 的张量，提取序列长度 L
                    t_fast = res[sym]['fast_1m'][0].cpu().numpy() # [N_fast, L]
                    t_slow = res[sym]['slow_1m'][0].cpu().numpy() # [N_slow, L]
                    
                    L = t_fast.shape[1]
                    norm = self.normalizers[sym]
                    
                    # 将引擎算出的 L 个步长的数据，顺次推入 Normalizer 建立 2000 窗口的真实均值/方差
                    for i in range(L):
                        raw_vec = np.zeros(len(self.all_feat_names), dtype=np.float32)
                        for k, name in enumerate(self.fast_feat_names):
                            if name in self.feat_name_to_idx: raw_vec[self.feat_name_to_idx[name]] = t_fast[k, i]
                        for k, name in enumerate(self.slow_feat_names):
                            if name in self.feat_name_to_idx: raw_vec[self.feat_name_to_idx[name]] = t_slow[k, i]
                        
                        # 静默处理历史数据，建立 Z-score 的完美基准
                        norm.process_frame(raw_vec)
                        
            warmup_count += 1
                        
        except Exception as e:
            logger.error(f"Backfill Error: {e}")
            
        logger.info(f"✅ Deep Warmup (Window=2000) Complete for {warmup_count} symbols.")
        self._publish_warmup_status()



     

    def _publish_warmup_status(self):
        """[Monitor] 将当前 Normalizer 计数发布到 Redis，供 Dashboard 实时读取"""
        try:
            status = {}
            for sym, norm in self.normalizers.items():
                # [🔥 恢复真实监控] 坚决使用 norm.count，它才是归一化健康的唯一真理！
                status[sym] = norm.count
            
            # 使用 Hash 结构: key=monitor:warmup:norm, field=symbol, value=count
            key = f"monitor:warmup:norm"
            # 批量写入
            if status:
                self.r.hset(key, mapping=status)
                # 设置过期时间防止残留 (比如 1 小时)
                self.r.expire(key, 3600)
        except Exception as e:
            pass # 监控不应阻塞主流程

    def _load_json_info(self, path):
        """[升级] 强力日志版 JSON 加载，格式错误直接报警！"""
        try:
            with open(path, 'r') as f: 
                data = json.load(f).get('features', [])
                logger.info(f"📄 成功加载 {len(data)} 个特征自: {path}")
                return data
        except Exception as e: 
            # 🚨 以前这里是静默 return []，导致建了一张没有列的空表，现在让它大声报错！
            logger.error(f"❌ 致命错误！无法解析 JSON 文件 {path}: {e}")
            return []

    def _init_consumer_groups(self):
        for s in [self.redis_cfg['raw_stream'], self.redis_cfg['option_stream']]:
            try:
                self.r.xgroup_create(s, self.redis_cfg['group'], mkstream=True, id='$')
            except redis.exceptions.ResponseError: pass
 
    async def process_fused_data(self, payload: Dict):
        self.msg_count += 1
        try:
            # 🚀 [架构升维: 事前缓存并合并] 
            # 只要 Fused 流里有期权快照，立刻更新内存 Cache，供下一轮 run_compute_cycle 直接 Pre-Join
            sym = payload.get('symbol')
            if sym:
                if 'option_buckets' in payload:
                    self.latest_opt_buckets[sym] = payload['option_buckets']
                if 'option_contracts' in payload:
                    self.latest_opt_contracts[sym] = payload['option_contracts']

            if sym not in self.symbols: return
            
            ts = float(payload['ts'])

            dt_utc = datetime.fromtimestamp(ts, timezone.utc)
            dt_ny = dt_utc.astimezone(NY_TZ)
            curr_minute = dt_ny.replace(second=0, microsecond=0)

            # ==========================================================
            # [🔥 升级] 盘前放行与 09:30 准点清洗机制 (Premarket Flush)
            # ==========================================================
            from datetime import time as dt_time
            current_time = dt_ny.time()
            
            # 1. 屏蔽 16:15 之后的盘后冗余数据 (防止撑爆硬盘)
            if current_time > dt_time(16, 15):
                return
                
            # 2. [核心] 09:30 准点大扫除！
            # 作用: 把今天 00:00 到 09:29 的所有脏 K 线从内存中剔除，
            # 保证接下来的 VWAP/SMA 等指标与回测历史完美对齐！
            if current_time >= dt_time(9, 30):
                if getattr(self, '_premarket_flushed_date', None) != dt_ny.date():
                    today_start = dt_ny.replace(hour=0, minute=0, second=0, microsecond=0)
                    rth_start = dt_ny.replace(hour=9, minute=30, second=0, microsecond=0)
                    for s in self.symbols:
                        df = self.history_1min[s]
                        if not df.empty:
                            # 仅保留: 昨天之前的历史数据，以及今天 09:30 之后的数据
                            mask = (df.index < today_start) | (df.index >= rth_start)
                            self.history_1min[s] = df[mask]
                    self._premarket_flushed_date = dt_ny.date()
                    logger.info("🧹 09:30 准点清洗完成！盘前测试数据已剔除，引擎进入纯净实盘(RTH)模式。")
            # ==========================================================
            
            
            # 1min Stock
            stock = payload.get('stock', {})
            if stock: self.latest_prices[sym] = float(stock.get('close', 0.0))
            
            # 5min Stock (仅在 5 分钟整数倍时间点入库，防止状态保持导致的历史污染)
            stock_5m = payload.get('stock_5m')
            if stock_5m and dt_ny.minute % 5 == 0:
                o, h, l, c, v = stock_5m['open'], stock_5m['high'], stock_5m['low'], stock_5m['close'], stock_5m['volume']
                self.history_5min[sym].loc[curr_minute, ['open', 'high', 'low', 'close', 'volume']] = [o, h, l, c, v]
                if len(self.history_5min[sym]) > 100: self.history_5min[sym] = self.history_5min[sym].iloc[-100:]
            

            # ========= 替换 process_fused_data 中的期权更新逻辑 =========
            buckets = payload.get('option_buckets')
            if buckets:
                arr = np.array(buckets, dtype=np.float32)
                # 维度补齐
                if arr.shape[1] < 12:
                    pad = np.zeros((arr.shape[0], 12 - arr.shape[1]), dtype=np.float32)
                    arr = np.hstack([arr, pad])
                if arr.shape[0] < 6: 
                    arr = np.vstack([arr, np.zeros((6 - arr.shape[0], 12), dtype=np.float32)])
                
                # 成交量增量计算
                if np.sum(arr[:, 6]) > 0.0001: 
                    curr_cum = arr[:, 6]
                    if self.warmup_needed[sym]:
                        self.last_cum_volume[sym] = curr_cum.copy()
                        minute_vol = np.zeros_like(curr_cum)
                        self.warmup_needed[sym] = False
                    else:
                        minute_vol = curr_cum - self.last_cum_volume[sym]
                        minute_vol = np.where(minute_vol < 0, curr_cum, minute_vol)
                        self.last_cum_volume[sym] = curr_cum.copy()
                    
                    arr[:, 6] = minute_vol
                    self.option_snapshot[sym] = arr
                    
                    # 🚀 [终极修复] 5m 期权快照直接实时镜像 1m 快照！
                    # 期权特征算的是瞬时截面 IV/Gamma，不需要时间聚合，直接复用最新状态即可，消灭断流！
                    if USE_5M_OPTION_DATA:
                        self.option_snapshot_5m[sym] = arr.copy()

            # 5min Options (仅在 5 分钟整数倍时间点入库)
            # 🚀 [Fix] 如果缺失 5m 专用桶数据，自动复用 1m 桶数据作为 Fallback
            if USE_5M_OPTION_DATA:
                buckets_5m = payload.get('option_buckets_5m')
                if not buckets_5m and dt_ny.minute % 5 == 0:
                    buckets_5m = payload.get('option_buckets')
                    
                if buckets_5m and dt_ny.minute % 5 == 0:
                    arr = np.array(buckets_5m, dtype=np.float32)
                    if arr.shape[1] < 12:
                        pad = np.zeros((arr.shape[0], 12 - arr.shape[1]), dtype=np.float32)
                        arr = np.hstack([arr, pad])
                    if arr.shape[0] < 6: arr = np.vstack([arr, np.zeros((6 - arr.shape[0], 12), dtype=np.float32)])
                    
                    if np.sum(arr[:, 6]) > 0.0001: 
                        curr_cum = arr[:, 6]
                        if self.warmup_needed_5m[sym]:
                            self.last_cum_volume_5m[sym] = curr_cum.copy()
                            self.warmup_needed_5m[sym] = False
                        else:
                            minute_vol = curr_cum - self.last_cum_volume_5m[sym]
                            minute_vol = np.where(minute_vol < 0, curr_cum, minute_vol)
                            self.last_cum_volume_5m[sym] = curr_cum.copy()
                            arr[:, 6] = minute_vol
                        self.option_snapshot_5m[sym] = arr

            # [Fix] 优化 1min Bar 归档逻辑 - 移动到 append 之前以防污染
            
            # =========================================================
            # [🔥 致命漏洞修复 1: 全局时间界限强制结算 (Global Flush)]
            # 解决 2 分钟延迟的核心！只要收到新的一分钟的任何一个 Tick，强制归档全市场上一分钟的真实 K 线！
            # =========================================================
            if not hasattr(self, 'global_last_minute'):
                self.global_last_minute = curr_minute
                
            if curr_minute > self.global_last_minute:
                # 跨分钟了，把所有股票上一分钟还没结算的 5s Buffer 全部归档
                for s in self.symbols:
                    if self.current_bars_5s[s]:
                        self._finalize_1min_bar(s, self.global_last_minute, cleanup=True)
                self.global_last_minute = curr_minute
            # =========================================================

            if not hasattr(self, 'last_processed_minute'):
                self.last_processed_minute = {}
                
            if sym not in self.last_processed_minute:
                self.last_processed_minute[sym] = curr_minute
            
            # 1. 检测个股 Minute Switch
            if curr_minute > self.last_processed_minute[sym]:
                self._finalize_1min_bar(sym, self.last_processed_minute[sym], cleanup=True)
                self.last_processed_minute[sym] = curr_minute

            # 2. Append 当前 tick 到 buffer
            self.current_bars_5s[sym].append(stock)
            
            # 3. 实时性优化 (Early Update): 在 :55秒及之后强制刷新当前分钟 Bar (Cleanup=False)
            # 因为 IBKR 的 5s Bar 的最大时间戳是 55 (00, 05, ..., 55)，所以必须是 >= 55
            # 这样保证 history_1min 能看到当前的最新状态，但不清空 buffer，防止后续 tick 丢失 Vol
            # 3. 实时性优化 (Early Update)
            from config import IS_SIMULATED
            # 🚨 修复：在仿真（回测/回放）模式下，数据本身就是 1min 完美的，无需等待 55 秒，直接结算放行！
            if dt_ny.second >= 55 or IS_SIMULATED:
                 self._finalize_1min_bar(sym, curr_minute, cleanup=False)

                
            # # [Core] 计算/发布特征 (Fast Channel - 5s)
            # inputs = self.engine.compute_all_inputs(
            #     self.history_1min,
            #     self.fast_feat_names,
            #     self.slow_feat_names,
            #     self.option_snapshot,
            #     skip_scaling=True # 后面手动 scale
            # )
            
            # feat_map = inputs.get(sym)
            # if feat_map:
            #     # 1. 组装 Raw Vector
            #     raw_vec = np.zeros(len(self.all_feat_names), dtype=np.float32)
            #     for name, val in feat_map.items():
            #         idx = self.feat_name_to_idx.get(name)
            #         if idx is not None: raw_vec[idx] = val
                
            #     # 2. 区分时段归一化 (RTH vs Non-RTH)
            #     # RTH (Regular Trading Hours): 09:30 - 16:00
            #     # 只有 RTH 期间的数据才更新 Normalizer 的 Buffer 和均值方差
            #     current_minutes = dt_ny.hour * 60 + dt_ny.minute
            #     is_rth = (570 <= current_minutes < 960)
                
            #     norm = self.normalizers[sym]
            #     if is_rth:
            #         norm_vec = norm.process_frame(raw_vec)
            #     else:
            #         norm_vec = norm.normalize_only(raw_vec)
                
            #     # 3. Split Fast/Slow
            #     fast_norm = norm_vec[self.fast_indices]
            #     slow_norm = norm_vec[self.slow_indices]
                
            #     # 4. NOTE: Per-tick publishing is DISABLED to avoid confusing the Orchestrator
            #     # The Orchestrator expects BATCH packets (from run_compute_cycle), not single symbol packets.
            #     # out_payload = {
            #     #     'symbol': sym,
            #     #     'ts': ts,
            #     #     'fast_1m': fast_norm, # 保持 1D
            #     #     'slow_1m': slow_norm,
            #     #     'raw_debug': raw_vec if self.msg_count % 100 == 0 else None
            #     # }
            #     # self.r.xadd(self.redis_cfg['output_stream'], {'data': pickle.dumps(out_payload)})
                
            #     # [Debug] 写入 DB (采样) -> REMOVED to fix duplicate/misaligned data
            #     # Only publish warmup status to Redis for monitoring (low frequency)
            #     if self.msg_count % 100 == 0: 
            #         self._publish_warmup_status()
            
            # [Fix] 删除遗留的 payload.get('last', 0) 覆写 Bug
            # 改为从安全的 stock 字典中再次兜底读取，或者干脆什么都不做
            if sym in self.latest_prices:
                 stock_data = payload.get('stock', {})
                 if stock_data and stock_data.get('close'):
                     self.latest_prices[sym] = float(stock_data.get('close', 0.0))

        except Exception as e:
            logger.error(f"Process Error: {e}")

    def _finalize_1min_bar(self, sym, dt, cleanup=True):
        bars = self.current_bars_5s[sym]
        if not bars: return
        
        # 兼容两种数据格式: S3 的简写 ('o', 'h', 'l', 'c', 'v') 和实盘的全写 ('open')
        o = float(bars[0].get('open', bars[0].get('o', 0)))
        h = max(float(b.get('high', b.get('h', 0))) for b in bars)
        l = min(float(b.get('low', b.get('l', 0))) for b in bars)
        c = float(bars[-1].get('close', bars[-1].get('c', 0)))
        v = sum(max(0.0, float(b.get('volume', b.get('v', 0)))) for b in bars)
        
        bar_time = dt.replace(second=0, microsecond=0)
        
        # =========================================================
        # [极速优化] 抛弃 pd.concat，直接原地 loc 赋值！速度提升百倍！
        # =========================================================
        hist = self.history_1min[sym]
        if hist.empty:
            self.history_1min[sym] = pd.DataFrame(
                [[o, h, l, c, v]], 
                columns=['open', 'high', 'low', 'close', 'volume'], 
                index=[bar_time]
            )
        else:
            # 使用 .at 逐个元素快速赋值，或者直接使用字典更新，彻底解决 Shape 不匹配问题
            new_data = {'open': o, 'high': h, 'low': l, 'close': c, 'volume': v}
            for col, val in new_data.items():
                self.history_1min[sym].at[bar_time, col] = val
            
            if len(self.history_1min[sym]) > self.HISTORY_LEN:
                self.history_1min[sym] = self.history_1min[sym].iloc[-self.HISTORY_LEN:]
                
        # =========================================================
        # 🚀 [新增 5min 动态聚合] 利用完美的 1min 历史，实时生成 5min K线
        # =========================================================
        # =========================================================
        # 🚀 [终极时序对齐] 严格适配离线 shift_minutes=5 的逻辑！
        # closed='left', label='right' 保证 09:30~09:34 的五根 1min K 线
        # 被完美聚合并贴上 09:35 的标签，与离线 LMDB 数据严丝合缝！
        # =========================================================
        df_1m = self.history_1min[sym]
        if not df_1m.empty:
            df_5m = df_1m.resample('5min', closed='left', label='right').agg({
                'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
            }).dropna()
            self.history_5min[sym] = df_5m.iloc[-100:]
                
        if cleanup:
            self.current_bars_5s[sym] = []

    def _calc_and_inject_option_features(self, sym):
        if self.history_1min[sym].empty: return
        # [🔥 修复] 从 10 列中切片前 8 列供 PyTorch 引擎计算，防止 Tensor Mismatch
        snap = self.option_snapshot[sym][:, :8]
        price = self.latest_prices[sym]
         
        
        # 简单过滤开盘不稳定期 (09:30-09:31)
        current_time = self.history_1min[sym].index[-1]
        minutes_from_midnight = current_time.hour * 60 + current_time.minute
        if 570 <= minutes_from_midnight < 571: # 09:30
             return

        feats = self.engine._calc_opt_feats_from_snap(
            torch.tensor(snap, dtype=torch.float32), 
            torch.tensor(price, dtype=torch.float32)
        )
        vals = {k: v.item() for k, v in feats.items()}
        
        # 动量计算
        curr_iv = vals.get('options_vw_iv', 0.0)
        curr_gamma = vals.get('options_vw_gamma', 0.0)
        self.deriv_history[sym].append({'iv': curr_iv, 'gamma': curr_gamma, 'price': price})
        
        mom_iv, acc_gamma, div_iv = 0.0, 0.0, 0.0
        if len(self.deriv_history[sym]) >= 6:
            prev = self.deriv_history[sym][-6]
            curr = self.deriv_history[sym][-1]
            eps = 1e-6
            mom_iv = (curr['iv'] - prev['iv']) / (prev['iv'] if abs(prev['iv'])>eps else 1.0)
            acc_gamma = (curr['gamma'] - prev['gamma']) / (prev['gamma'] if abs(prev['gamma'])>eps else 1.0)
            div_iv = mom_iv - ((curr['price']-prev['price']) / (prev['price'] if prev['price']>eps else 1.0))

        vals['options_iv_momentum'] = mom_iv
        vals['options_gamma_accel'] = acc_gamma
        vals['options_iv_divergence'] = div_iv
        
        idx = self.history_1min[sym].index[-1]
        for k, v in vals.items(): self.history_1min[sym].loc[idx, k] = v
        self.history_1min[sym].fillna(0.0, inplace=True)

     

    def _ensure_history_alignment(self, target_time):
        """对齐数据，填补缺失的 Bar"""
        for s in self.symbols:
            df = self.history_1min[s]
            if df.empty:
                ref_price = self.latest_prices.get(s, 0.0)
                if ref_price > 1e-6:
                    self.history_1min[s] = pd.DataFrame([{
                        'open': ref_price, 'high': ref_price, 'low': ref_price, 'close': ref_price, 'volume': 0.0
                    }], index=[target_time])
                continue

            last_ts = df.index[-1]
            if last_ts < target_time:
                # 限制最多补 10 分钟，防止卡死
                if (target_time - last_ts).total_seconds() > 300 * 60: continue
                
                curr_fill_ts = last_ts + timedelta(minutes=1)
                last_close = float(df.iloc[-1]['close'])
                rows_to_add = []
                count = 0
                
                while curr_fill_ts <= target_time and count < 10:
                    if curr_fill_ts not in df.index:
                        # [极速优化] 直接 loc 补空
                        df.loc[curr_fill_ts, ['open','high','low','close','volume']] = [last_close, last_close, last_close, last_close, 0.0]
                    curr_fill_ts += timedelta(minutes=1)
                    count += 1
                
                if count > 0:
                    df.sort_index(inplace=True)
                    if len(df) > self.HISTORY_LEN:
                        df = df.iloc[-self.HISTORY_LEN:]
                    self.history_1min[s] = df

    def _inject_temporal_derivatives(self, sym: str, raw_vec: np.ndarray):
        """
        [极速截胡] 计算并注入时序导数特征 (期权动量/加速度/量价背离)
        在引擎(Engine)处理完截面数据后，进入归一化(Normalizer)和写库前被调用。
        """
        idx_iv = self.feat_name_to_idx.get('options_vw_iv')
        idx_gamma = self.feat_name_to_idx.get('options_vw_gamma')
        
        # 提取当前时间步的基础值
        curr_iv = raw_vec[idx_iv] if idx_iv is not None else 0.0
        curr_gamma = raw_vec[idx_gamma] if idx_gamma is not None else 0.0
        curr_price = self.latest_prices.get(sym, 1.0)
        
        # 记录当前帧状态
        self.deriv_history[sym].append({'iv': curr_iv, 'gamma': curr_gamma, 'price': curr_price})
        mom_iv, acc_gamma, div_iv = 0.0, 0.0, 0.0
        
        # 提取 5 分钟前状态进行对比计算
        if len(self.deriv_history[sym]) >= 6:
            prev = self.deriv_history[sym][-6]
            curr = self.deriv_history[sym][-1]
            eps = 1e-6
            
            mom_iv = (curr['iv'] - prev['iv']) / (prev['iv'] if abs(prev['iv']) > eps else 1.0)
            acc_gamma = (curr['gamma'] - prev['gamma']) / (prev['gamma'] if abs(prev['gamma']) > eps else 1.0)
            price_ret = (curr['price'] - prev['price']) / (prev['price'] if prev['price'] > eps else 1.0)
            div_iv = mom_iv - price_ret
            
        # 获取目标导数特征的全局索引
        idx_mom = self.feat_name_to_idx.get('options_iv_momentum')
        idx_acc = self.feat_name_to_idx.get('options_gamma_accel')
        idx_div = self.feat_name_to_idx.get('options_iv_divergence')
        
        # 精准覆写原本为 0.0 的位置 (Numpy 数组原地修改)
        if idx_mom is not None: raw_vec[idx_mom] = mom_iv
        if idx_acc is not None: raw_vec[idx_acc] = acc_gamma
        if idx_div is not None: raw_vec[idx_div] = div_iv
     
    
     
    async def run_compute_cycle(self, ts_from_payload=None):
        """
        [V8 终极版] 特征计算主循环
        """
        # [🔥 修正] 优先使用来自消息 Payload 的显式时间戳，彻底消除 Redis 全局 Key 带来的竞争
        current_replay_ts = ts_from_payload if ts_from_payload else self.r.get("replay:current_ts")
         
        latest_timestamps = []
        for s in self.symbols:
            if not self.history_1min[s].empty:
                latest_timestamps.append(self.history_1min[s].index[-1])
        
        if not latest_timestamps:
            if current_replay_ts: 
                self.r.set("sync:feature_calc_done", current_replay_ts)
                self.r.set("sync:orch_done", current_replay_ts)  # 👈 [死锁修复] 连带解开下游锁
            return
        target_time = max(latest_timestamps)
        
        ready_symbols = [s for s in self.symbols if not self.history_1min[s].empty and self.history_1min[s].index[-1] == target_time]
        if len(ready_symbols) < 2:
            if current_replay_ts: 
                self.r.set("sync:feature_calc_done", current_replay_ts)
                self.r.set("sync:orch_done", current_replay_ts)  # 👈 [死锁修复] 连带解开下游锁
            return
            
        # ---------------- 拒止推理安全检查 ----------------
        sample_s = ready_symbols[0]
        curr_len = len(self.history_1min[sample_s])
        required_len = 30
        
        if curr_len < required_len:
            logger.debug(f"[数据不足-拒止推理] 历史K线 {curr_len} 根，要求 {required_len} 根，等待预热。")
            if current_replay_ts: 
                self.r.set("sync:feature_calc_done", current_replay_ts)
                self.r.set("sync:orch_done", current_replay_ts)  # 👈 [死锁修复] 连带解开下游锁
            return
        
        # ---------------- 1. 触发 Engine 计算 ----------------
        # 🚀 [核心修复] 切片提取前 12 列期权特征，并将其传给底层引擎
        sliced_snaps = {s: snap[:, :12] for s, snap in self.option_snapshot.items()}
        sliced_snaps_5m = {s: snap[:, :12] for s, snap in getattr(self, 'option_snapshot_5m', {}).items()}

        try:
            results_map = self.engine.compute_all_inputs(
                history_1min=self.history_1min,
                history_5min=getattr(self, 'history_5min', {}),
                fast_feats=self.fast_feat_names,
                slow_feats=self.slow_feat_names,
                option_snapshots=sliced_snaps,  
                option_snapshot_5m=sliced_snaps_5m, # 🚀 [新增] 传入 5min 期权快照
                feat_resolutions=getattr(self, 'feat_resolutions', {}) 
            )
        except Exception as e:
            logger.error(f"Engine Compute Error: {e}", exc_info=True)
            if current_replay_ts: 
                self.r.set("sync:feature_calc_done", current_replay_ts)
                self.r.set("sync:orch_done", current_replay_ts)  # 👈 [死锁修复] 计算崩溃时也要解锁！
            return

        # ---------------- 2. 解析与解包 ----------------
        lx_fast_debug = []
        lx_slow_debug = []  # 🚀 改回纯净的 List，不再用 Dict 分辨率
        batch_raw = []
        valid_mask = []  # 🛡️ 记录当前帧未断流的股票，防止 0 值污染归一化均值
        
        for sym in self.symbols:
            raw_vec = np.zeros(len(self.all_feat_names), dtype=np.float32)
            is_valid = False
            
            if sym in results_map:
                res_sym = results_map[sym]
                is_valid = True
                
                # 🚀 提取 Fast 
                if res_sym.get('fast_1m') is not None:
                    fast_latest = res_sym['fast_1m'][0, :, -1].cpu().numpy()
                    for i, fname in enumerate(self.fast_feat_names):
                        if i < len(fast_latest):
                            raw_vec[self.feat_name_to_idx[fname]] = fast_latest[i]
                            
                # 🚀 提取 Slow
                if res_sym.get('slow_1m') is not None:
                    slow_latest = res_sym['slow_1m'][0, :, -1].cpu().numpy()
                    for i, fname in enumerate(self.slow_feat_names):
                        if i < len(slow_latest):
                            raw_vec[self.feat_name_to_idx[fname]] = slow_latest[i]

                # =======================================================
                # 🚀 [极速截胡修复] 注入时序导数特征 (期权动量/加速度/量价背离)
                # =======================================================
                if hasattr(self, '_inject_temporal_derivatives'):
                    self._inject_temporal_derivatives(sym, raw_vec)
                
                # Debug DB 收集切片
                lx_fast_debug.append((sym, raw_vec[self.fast_indices]))
                lx_slow_debug.append((sym, raw_vec[self.slow_indices])) # 🚀 直接一刀切下模型使用的完全体！

            # 无论有效与否，都要 append，保证矩阵维度严格等于 [B, N_feats]
            batch_raw.append(raw_vec)
            valid_mask.append(is_valid)

        # ---------------- 3. 主线程同步建表 & 异步写入 Debug 数据库 ----------------
        data_ts = target_time.timestamp()
        
        # 🚀 [主线程同步建表] 根据数据的实际时间戳建表。保证在主线程完成，彻底杜绝并发锁死！
        dt_ny = datetime.fromtimestamp(data_ts, NY_TZ)
        date_str = dt_ny.strftime('%Y%m%d')
        
        if not hasattr(self, 'created_debug_dates'):
            self.created_debug_dates = set()
            
        if date_str not in self.created_debug_dates:
            self._ensure_debug_tables(date_str)
            self.created_debug_dates.add(date_str)

        if lx_fast_debug or any(lx_slow_debug_dict.values()):
            if hasattr(self, 'debug_db_executor'):
                self.debug_db_executor.submit(
                    self._write_debug_batch,
                    int(data_ts), 
                    date_str,  # 🚀 直接传入算好的日期字符串
                    lx_fast_debug, 
                    lx_slow_debug   # 🚀 传完整的 List
                )

       # ---------------- 4. 归一化与序列组装 (修复时间轴坍缩漏洞) ----------------
        if not hasattr(self, 'norm_history_30'): 
            self.norm_history_30 = []

        # [🔥 核心修复] 判断当前物理时间是否跨越了 1 分钟
        current_minute_ts = int(data_ts // 60) * 60
        is_new_minute = False
        if getattr(self, 'last_model_minute_ts', 0) < current_minute_ts:
            is_new_minute = True
            self.last_model_minute_ts = current_minute_ts
            
        batch_norm = []
        current_minutes = dt_ny.hour * 60 + dt_ny.minute
        is_rth = (570 <= current_minutes < 960)
        
        for b_idx, sym in enumerate(self.symbols):
            raw_vec = batch_raw[b_idx]
            is_valid = valid_mask[b_idx]
            norm = self.normalizers[sym]
            
            # 🛡️ 只有在有效数据、盘中时间，且【真正跨越了 1 分钟】时，才推进 Buffer 污染均值
            if is_valid and is_rth and is_new_minute:
                norm_vec = norm.process_frame(raw_vec)
            else:
                # 否则只是实时投影（比如 10秒级的高频 Tick），绝不污染底层统计池
                norm_vec = norm.normalize_only(raw_vec)
                
            batch_norm.append(norm_vec)
            
        norm_mat = np.stack(batch_norm) # [B, N_all_feats]
        
        # [🔥 核心修复] 正确维护 30 步的宏观时间序列
        if is_new_minute:
            # 物理分钟跨越，产生新的历史帧
            self.norm_history_30.append(norm_mat)
            if len(self.norm_history_30) > 30:
                self.norm_history_30.pop(0)
        else:
            # 仍在同一分钟内的 10秒级高频刷新！
            if len(self.norm_history_30) > 0:
                # 覆盖最后一帧，使模型看到最新盘口，但绝不拉长序列导致时间轴坍缩！
                self.norm_history_30[-1] = norm_mat
            else:
                self.norm_history_30.append(norm_mat)
            
        # ---------------- 5. 组装推流 Payload (The Dict-Payload) ----------------
        # ---------------- 5. 组装推流 Payload (The Dict-Payload) ----------------
        if len(self.norm_history_30) > 0:  # ✅ 改成 > 0，只要有数据就向左补齐并推流！
            # [🔥 新增] 动态补齐 30 长度，彻底解决重启后瞎眼 30 分钟的问题
            pad_len = 30 - len(self.norm_history_30)
            if pad_len > 0:
                # 用最老的一帧向左复制填充
                padded_history = [self.norm_history_30[0]] * pad_len + self.norm_history_30
            else:
                padded_history = self.norm_history_30
                
            norm_seq_30 = np.stack(padded_history, axis=1) # [B, 30, N_all_feats]
            
            # 🚀 [核心修复] 把 batch_raw 堆叠成矩阵，供下面提取宏观/杂项特征使用
            raw_mat = np.stack(batch_raw)
            
            batch_symbols = []
            batch_prices = []
            batch_stock_ids = []
            batch_spy_rocs = []
            batch_qqq_rocs = []
            batch_fast_vols = []
            
            # 期权行情的 Mock 注入 (供回测使用)
            cheat_call, cheat_put = [], []
            cheat_call_bid, cheat_call_ask = [], []
            cheat_put_bid, cheat_put_ask = [], []
            cheat_call_iv, cheat_put_iv = [], []
            
            valid_b_indices = []
            
            for b_idx, sym in enumerate(self.symbols):
                if sym not in ready_symbols or sym not in results_map:
                    continue
                    
                valid_b_indices.append(b_idx)
                batch_symbols.append(sym)
                batch_prices.append(self.latest_prices.get(sym, 0.0))
                batch_stock_ids.append(b_idx)
                
                # 杂项特征提取
                idx_spy = self.feat_name_to_idx.get('spy_roc_5min')
                batch_spy_rocs.append(raw_mat[b_idx, idx_spy] if idx_spy is not None else 0.0)
                idx_qqq = self.feat_name_to_idx.get('qqq_roc_5min')
                batch_qqq_rocs.append(raw_mat[b_idx, idx_qqq] if idx_qqq is not None else 0.0)
                idx_vol = self.feat_name_to_idx.get('fast_vol')
                batch_fast_vols.append(raw_mat[b_idx, idx_vol] if idx_vol is not None else 0.0)
                
                # 提取期权快照供回测 (Mock)
                snap = self.option_snapshot.get(sym)
                if snap is not None and snap.shape[0] >= 6 and snap.shape[1] >= 10:
                    cheat_call.append(snap[2, 0]); cheat_call_iv.append(snap[2, 7])
                    cheat_call_bid.append(snap[2, 8]); cheat_call_ask.append(snap[2, 9])
                    cheat_put.append(snap[0, 0]); cheat_put_iv.append(snap[0, 7])
                    cheat_put_bid.append(snap[0, 8]); cheat_put_ask.append(snap[0, 9])
                else:
                    for lst in [cheat_call, cheat_call_iv, cheat_call_bid, cheat_call_ask, 
                                cheat_put, cheat_put_iv, cheat_put_bid, cheat_put_ask]:
                        lst.append(0.0)

            if batch_symbols:
                valid_b_indices = np.array(valid_b_indices)
                valid_norm_seq = norm_seq_30[valid_b_indices] # [B_valid, 30, N_all_feats]
                
                features_dict = {}
                for fname in self.slow_feat_names:
                    f_idx = self.feat_name_to_idx[fname]
                    features_dict[fname] = valid_norm_seq[:, :, f_idx]

                # =================================================================
                # 🚀 [架构升维：事前合并期权字典 (Pre-Join)]
                # 直接将最新的期权 Buckets 字典放入 Payload，不改变其原生结构！
                # =================================================================
                live_options = {}
                for sym in batch_symbols:
                    live_options[sym] = {
                    'buckets': getattr(self, 'latest_opt_buckets', {}).get(sym, []),
                    'contracts': getattr(self, 'latest_opt_contracts', {}).get(sym, [])
                }
                
                payload = {
                    'ts': data_ts,
                    'symbols': batch_symbols,
                    'stock_price': batch_prices,
                    'stock_id': np.array(batch_stock_ids),
                    'sector_id': np.zeros(len(batch_symbols)),
                    'fast_vol': np.array(batch_fast_vols),
                    'spy_roc_5min': np.array(batch_spy_rocs),
                    'qqq_roc_5min': np.array(batch_qqq_rocs),
                    'features_dict': features_dict,
                    # 👇 直接挂载结构化的期权大字典
                    'live_options': live_options,
                    'is_new_minute': is_new_minute,  # 🚀 [新增] 传导逻辑分钟切换标志，用于对齐 Orchestrator 的 30 分钟窗口
                    # 兼容回测环境的期权盘口
                    'cheat_call': cheat_call, 'cheat_put': cheat_put,
                    'cheat_call_bid': cheat_call_bid, 'cheat_call_ask': cheat_call_ask,
                    'cheat_put_bid': cheat_put_bid, 'cheat_put_ask': cheat_put_ask,
                    'cheat_call_iv': cheat_call_iv, 'cheat_put_iv': cheat_put_iv,
                }

               
                try:
                     # 🚀 [架构修复] 杜绝硬编码！使用引擎初始化时传入的 output_stream (即 STREAM_INFERENCE)
                     self.r.xadd(self.redis_cfg['input_stream'], {'data': ser.pack(payload)}, maxlen=100)
                except Exception as e:
                    logger.error(f"❌ Redis XADD Error: {e}")
                    
        # ✅ 严格在处理完本帧（包含 engine.compute 和 推流）后，再释放帧锁
        from config import IS_SIMULATED
        if current_replay_ts and IS_SIMULATED:
            self.r.set("sync:feature_calc_done", current_replay_ts)

    async def run(self):
        # [🔥 动态 DB 切换] 启动时自动识别模式，从 config 获取 Redis 数据库路由
        from config import get_redis_db
        target_db = get_redis_db()
        if self.r.connection_pool.connection_kwargs.get('db') != target_db:
            logger.info(f"🔄 Re-connecting Redis to DB {target_db} (Dynamic Mode Detection)")
            self.r = redis.Redis(host=self.redis_cfg['host'], port=self.redis_cfg['port'], db=target_db)
            # [🔥 关键修复] 必须在新 DB 中重新创建消费者组，在回放模式下从 '0' 开始
            from config import IS_SIMULATED
            group_id = '0' if IS_SIMULATED else '$'
            try:
                self.r.xgroup_create(self.redis_cfg['raw_stream'], self.redis_cfg['group'], mkstream=True, id=group_id)
            except Exception: pass # 忽略 Group 已存在的错误

        print(f"🚀 Feature Service Started on Redis DB {target_db}.")
        # 先进行跨日回填与强预热
        # ==========================================================
        # [🔥 优化] 启动时立刻为今天“占位”建表 (Eager Initialization)
        # 这样无论有没有真实数据，Dashboard 都能看到空表而不会报错！
        # ==========================================================
        from config import NY_TZ
        today_str = datetime.now(NY_TZ).strftime('%Y%m%d')
        self._ensure_debug_tables(today_str)
        logger.info(f"📁 提前初始化今日 Debug Postgres 数据库表: debug_fast_{today_str} and debug_slow_{today_str}")
        # ==========================================================
         
        
        last_save = time.time()
        streams = {self.redis_cfg['raw_stream']: '>'}
        
        while True:
            try:
                # 状态保存 (30s)
                if time.time() - last_save > 30:
                    self._save_service_state()
                    # [🔥 新增] 每 30 秒向 Redis 刷新一次状态，防止 1 小时后过期！
                    self._publish_warmup_status()
                    last_save = time.time()

                resp = self.r.xreadgroup(self.redis_cfg['group'], self.redis_cfg['consumer'], streams, count=100, block=100)
                if resp:
                    for _, msgs in resp:
                        for mid, data in msgs:
                            current_batch_ts = None
                            if b'batch' in data:
                                try:
                                    batch = ser.unpack(data[b'batch'])
                                    for payload in batch:
                                        await self.process_fused_data(payload)
                                        if not current_batch_ts: current_batch_ts = payload.get('ts')
                                except Exception as e:
                                    logger.error(f"❌ Batch Unpack Error: {e}")
                                    
                            elif b'pickle' in data:
                                payload = ser.unpack(data[b'pickle'])
                                await self.process_fused_data(payload)
                                current_batch_ts = payload.get('ts')
                            
                            # [🔥 核心修复] 每一帧数据（或 Batch）处理完后，立刻触发计算，并回传正确的同步旗标
                            if current_batch_ts:
                                await self.run_compute_cycle(ts_from_payload=current_batch_ts)
                            
                            self.r.xack(self.redis_cfg['raw_stream'], self.redis_cfg['group'], mid)
                
                await asyncio.sleep(0.001)
                
            except redis.exceptions.ResponseError as e:
                if "NOGROUP" in str(e):
                    logger.warning("⚠️ Consumer Group missing (likely flushed by Replay Driver). Recreating...")
                    self._init_consumer_groups()
                else:
                    logger.error(f"❌ Redis Response Error: {e}")
                    await asyncio.sleep(1)
            except Exception as e:
                print(f"❌ Run Loop Error: {e}")
                await asyncio.sleep(1)

if __name__ == "__main__":
    # [优化] 更加鲁棒的路径探测逻辑
    current_dir = Path(__file__).resolve().parent
    home_project = Path.home() / "quant_project"
    
    # 优先使用脚本同目录下的配置 (dev 模式)，其次使用 ~/quant_project/config (prod 模式)
    slow_path = current_dir / "slow_feature.json"
    if not slow_path.exists():
        slow_path = home_project / "config/slow_feature.json"
        
    fast_path = current_dir / "fast_feature.json"
    if not fast_path.exists():
        # Fallback to daily_backtest or config directory
        fast_path = current_dir.parent / "daily_backtest" / "fast_feature.json"
        if not fast_path.exists():
            fast_path = home_project / "config/fast_feature.json"
            
    paths = {
        'fast': str(fast_path),
        'slow': str(slow_path)
    }
    
    # [Fix] 从 config.py 加载统一标的
    try:
        from config import TARGET_SYMBOLS
    except ImportError:
        # 兼容当前路径没在 PYTHONPATH 里的情况
        sys.path.append(str(current_dir.parent / "baseline"))
        from config import TARGET_SYMBOLS
        
    print(f"🚀 Starting Feature Service for {len(TARGET_SYMBOLS)} symbols.")
    print(f"📄 Fast Config: {fast_path}")
    print(f"📄 Slow Config: {slow_path}")

    
    service = FeatureComputeService(REDIS_CFG, TARGET_SYMBOLS, paths)
    try:
        asyncio.run(service.run())
    except KeyboardInterrupt:
        print("🛑 Service Stopped.")
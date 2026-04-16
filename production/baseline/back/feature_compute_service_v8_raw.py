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
import copy
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

        # [对齐离线训练]：定义必须预先压制肥尾分布的特征名单
        self.FAT_TAIL_FEATURES = {'options_iv_momentum', 'options_gamma_accel', 'options_iv_divergence'}
        self.fat_tail_mask = np.array([name in self.FAT_TAIL_FEATURES for name in feature_names], dtype=bool)

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

        # [对齐离线训练]：核心预处理 - 肥尾特征 Signed Log1p 压制
        # 公式: sign(x) * log(1 + abs(x))
        if self.fat_tail_mask.any():
            target_vals = x_raw_1d[self.fat_tail_mask]
            x_raw_1d[self.fat_tail_mask] = np.sign(target_vals) * np.log1p(np.abs(target_vals))

        self.raw_buffer.append(x_raw_1d)
        self.count += 1
        
        # 定期更新统计量
        if self.count < 100 or self.count % 10 == 0:
            if len(self.raw_buffer) >= 2:
                raw_block = np.vstack(list(self.raw_buffer))
                self.last_mean = np.mean(raw_block, axis=0).astype(np.float32)
                self.last_std = np.std(raw_block, axis=0).astype(np.float32)
                
                self.last_std[self.last_std < 1e-6] = 1.0
                self.last_mean[self.categorical_mask] = 0.0
                self.last_std[self.categorical_mask] = 1.0

        # [对齐离线训练]：Z-Score (epsilon=1e-6) + Tanh(/3) 压缩
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
        
        # [动态分辨率解耦]
        self.slow_resolution_groups = defaultdict(list)
        for name in self.slow_feat_names:
            res = self.feat_resolutions.get(name, '1min')
            self.slow_resolution_groups[res].append(name)
            
        self.slow_resolution_indices = {
            res: [self.feat_name_to_idx[n] for n in names]
            for res, names in self.slow_resolution_groups.items()
        }
        self.slow_indices = [self.feat_name_to_idx[name] for name in self.slow_feat_names]
        
        self.debug_db_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if str(self.device) == 'cpu' and torch.backends.mps.is_available():
            self.device = torch.device('mps')
            
        self.engine = RealTimeFeatureEngine(stats_path=None, device=str(self.device))
        
        self.normalizers = {
            s: RollingWindowNormalizer(
                self.all_feat_names, 
                self.feat_config_dict, 
                window=2000, 
                use_tanh=True
            ) for s in symbols
        }

        self.HISTORY_LEN = 500
        # 🚀 [核心重构] 调用显式重置，确保初始化与重置逻辑合一
        self.reset_internal_memory()

    def reset_internal_memory(self):
        """🚀 [状态消磁] 物理抹除所有内部缓冲区，并由于由于初始化 SDS 4.0 状态机机机机"""
        logger.info("🧹 [State] Resetting all internal memory buffers (SDS 4.0)...")
        self.last_cum_volume = {s: np.zeros(6, dtype=np.float32) for s in self.symbols}
        self.last_total_volume = {s: 0.0 for s in self.symbols} # 🚀 [SDS 4.0] 存储昨日/上分末秒的累计成交量 (用于区间差分)
        self.history_1min = {s: pd.DataFrame() for s in self.symbols}
        self.history_5min = {s: pd.DataFrame() for s in self.symbols}
        self.current_bars_5s = {s: [] for s in self.symbols}    # 🚀 [SDS 4.0] 升级为 Full-Payload 缓冲区
        self.option_snapshot = {s: np.zeros((6, 12), dtype=np.float32) for s in self.symbols}
        self.option_snapshot_5m = {s: np.zeros((6, 12), dtype=np.float32) for s in self.symbols}
        self.latest_prices = {s: 0.0 for s in self.symbols}
        self.last_tick_price = {s: 0.0 for s in self.symbols}
        self.truth_map_1min = {}
        
        # 🚀 [架构由于瘦身] 抛弃本地 Frozen 冗余，全面拥抱 Redis 真相账本
        self.deriv_history = {s: deque(maxlen=10) for s in self.symbols}
        self.deriv_history = {s: deque(maxlen=10) for s in self.symbols}
        self.last_compute_ts = None
        self.warmup_needed = {s: True for s in self.symbols}
        self.warmup_needed_5m = {s: True for s in self.symbols}
        self.last_cum_volume_5m = {s: np.zeros(6, dtype=np.float32) for s in self.symbols}
        
        # [NEW] Parity & Speed Optimization
        self.last_model_minute_ts = 0
        self.cached_results_map = None
        self.cached_final_results = None
        self.cached_batch_raw = None
        self.cached_valid_mask = None
        self.cached_lx_fast_debug = None
        self.cached_lx_slow_debug = None
        
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

            # 🚀 [Surgery 19] 双保险同步：确保分区表本身也补齐了所有特征列
            # 解决 PostgreSQL 分区同步延迟或现有分区未自动更新的问题
            for part_name, feat_names in [(part_fast, self.fast_feat_names), (part_slow, self.slow_feat_names)]:
                existing_partition_cols = get_existing_cols(part_name)
                for name in feat_names:
                    if name not in existing_partition_cols:
                        try:
                            c.execute(f'ALTER TABLE {part_name} ADD COLUMN "{name}" DOUBLE PRECISION')
                            logger.info(f"➕ [Schema Sync] Added column '{name}' to partition {part_name}")
                        except Exception as ae:
                            logger.warn(f"Failed to add column {name} to {part_name}: {ae}")
                        
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
        # 🛡️ [防弹优化] 使用 .get() 安全获取，避免 KeyError
        streams = []
        if 'raw_stream' in self.redis_cfg: streams.append(self.redis_cfg['raw_stream'])
        if 'option_stream' in self.redis_cfg: streams.append(self.redis_cfg['option_stream'])
        
        target_group = self.redis_cfg.get('group', 'compute_group')
        for s in streams:
            try:
                self.r.xgroup_create(s, target_group, mkstream=True, id='$')
            except redis.exceptions.ResponseError: pass
 
    async def process_market_data(self, batch, current_replay_ts=None, return_payload=False):
        self.msg_count += 1
        try:
            if not hasattr(self, 'last_finalized_minute'): self.last_finalized_minute = {}
            if not hasattr(self, 'pending_finalization'): self.pending_finalization = {}
            if not hasattr(self, 'finalization_snapshots'): self.finalization_snapshots = {}

            for payload in batch:
                sym = payload.get('symbol')
                if sym not in self.symbols: continue
                
                ts = float(payload.get('ts', 0))
                dt_ny = datetime.fromtimestamp(ts, timezone.utc).astimezone(NY_TZ)
                curr_minute = dt_ny.replace(second=0, microsecond=0)
                
                # 初始个股时钟
                if sym not in self.last_finalized_minute:
                    self.last_finalized_minute[sym] = curr_minute

                # 🚀 [SDS 2.0 核心拦截] 
                # 当报文 TS 跨分时，触发“计算屏障”锁定，而非立即结算。
                # 这样可以确保当前的 option_snapshot (属于上一分钟末秒) 在被 10:14 报文污染前被物理冻结。
                while curr_minute > self.last_finalized_minute[sym]:
                    target_dt = self.last_finalized_minute[sym]
                    # 🔐 [Snapshot Lock] 深度拷贝当前所有期权状态，物理阻断未来数据
                    self.finalization_snapshots[sym] = {
                        'buckets': copy.deepcopy(self.latest_opt_buckets.get(sym, [])),
                        'contracts': copy.deepcopy(self.latest_opt_contracts.get(sym, [])),
                        'snapshot': self.option_snapshot.get(sym).copy() if self.option_snapshot.get(sym) is not None else None
                    }
                    self.pending_finalization[sym] = target_dt
                    self.last_finalized_minute[sym] += timedelta(minutes=1)
                    logger.info(f"❄️ [SDS 2.0 Lock] Queueing {sym} @ {target_dt.strftime('%H:%M:%S')} for Barrier Finalization")

                is_exact_boundary = (dt_ny.second == 0)
                if is_exact_boundary and self.current_bars_5s[sym]:
                    prev_minute = curr_minute - timedelta(minutes=1)
                    if prev_minute >= self.last_finalized_minute[sym] - timedelta(minutes=1):
                         # 这里的 prev_minute 结算同样进入 Barrier 逻辑，确保 IV 补算完成
                         if sym not in self.pending_finalization:
                            self.finalization_snapshots[sym] = {
                                'buckets': copy.deepcopy(self.latest_opt_buckets.get(sym, [])),
                                'contracts': copy.deepcopy(self.latest_opt_contracts.get(sym, [])),
                                'snapshot': self.option_snapshot.get(sym).copy() if self.option_snapshot.get(sym) is not None else None
                            }
                            self.pending_finalization[sym] = prev_minute

                # 2. [快照同步] 结算完成后，更新期权快照缓存
                b_raw = payload.get('buckets', payload.get('option_buckets'))
                if b_raw:
                    b_data = b_raw.get('buckets', []) if isinstance(b_raw, dict) else b_raw
                    if b_data and len(b_data) > 0: self.latest_opt_buckets[sym] = b_data
                        
                c_raw = payload.get('contracts', payload.get('option_contracts'))
                if c_raw:
                    c_data = c_raw.get('contracts', []) if isinstance(c_raw, dict) else c_raw
                    if c_data and len(c_data) > 0: self.latest_opt_contracts[sym] = c_data
                
    # ==========================================================
                # [🔥 升级] 盘前放行与 09:30 准点清洗机制 (Premarket Flush)
                # ==========================================================
                from datetime import time as dt_time
                current_time = dt_ny.time()
                
                # 1. 屏蔽 16:15 之后的盘后冗余数据 (防止撑爆硬盘)
                if current_time > dt_time(16, 15):
                    continue
                    
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
                 # =========================================================
    
                if not hasattr(self, 'last_processed_minute'):
                    self.last_processed_minute = {}
                    
                if sym not in self.last_processed_minute:
                    self.last_processed_minute[sym] = curr_minute
                
                # 1. 检测个股 Minute Switch
                # 🚀 [架构由于瘦身] 个股对位已由 global_last_minute 统一处理，此处仅作为占位
                pass
                
                
                # 🚀 [SDS 5.0 绝对放行] 取消 if stock 过滤器，物理由于确保全量 Ticks (含由于期权) 100% 入库
                self.current_bars_5s[sym].append(copy.deepcopy(payload)) 
                
                stock = payload.get('stock', {})
                if stock:
                    self.latest_prices[sym] = float(stock.get('close', 0.0))
                    self.last_tick_price[sym] = self.latest_prices[sym]
                
                # 5min Stock (仅在 5 分钟整数倍时间点入库，防止状态保持导致的历史污染)
                stock_5m = payload.get('stock_5m')
                if stock_5m and dt_ny.minute % 5 == 0:
                    o, h, l, c, v = stock_5m['open'], stock_5m['high'], stock_5m['low'], stock_5m['close'], stock_5m['volume']
                    self.history_5min[sym].loc[curr_minute, ['open', 'high', 'low', 'close', 'volume']] = [o, h, l, c, v]
                    if len(self.history_5min[sym]) > 100: self.history_5min[sym] = self.history_5min[sym].iloc[-100:]
                
    
                buckets = payload.get('buckets')
                if not buckets or len(buckets) == 0:
                    buckets = payload.get('option_buckets')
                
                if buckets:
                     
                    # 🚀🚀🚀 [核心修复 3：存量保护逻辑] 🚀🚀🚀
                    # 绝对不能直接覆写！因为 1s 流可能只带了价格和深度，会抹除初始化时的 Strike 信息。
                    new_arr = np.array(buckets, dtype=np.float32)
                    
                    # 维度补齐 (对齐到 12 列)
                    if new_arr.shape[1] < 12:
                        pad = np.zeros((new_arr.shape[0], 12 - new_arr.shape[1]), dtype=np.float32)
                        new_arr = np.hstack([new_arr, pad])
                    
                    # 取出存量快照作为底座
                    old_snap = self.option_snapshot.get(sym)
                    if old_snap is None:
                        old_snap = np.zeros((6, 12), dtype=np.float32)
                    
                    # 仅更新 Row 0~5 的 Price(0), Volume(6), Bid(8), Ask(9)
                    # 保留 Strike(5) 和 Expiry/Greeks 等存量
                    for i in range(min(len(new_arr), 6)):
                        # 强行重算 Price (bid+ask)/2
                        bid, ask = new_arr[i, 8], new_arr[i, 9]
                        if bid > 0 and ask > 0:
                            new_arr[i, 0] = (bid + ask) / 2.0
                        
                        # 物理合并
                        old_snap[i, 0] = new_arr[i, 0]       # Price
                        old_snap[i, 6] = max(0.0, new_arr[i, 6]) # Volume/Depth
                        old_snap[i, 8] = new_arr[i, 8]       # Bid
                        old_snap[i, 9] = new_arr[i, 9]       # Ask
                        # Index 1-4, 5, 7, 10, 11 保持不变 (由引擎后续计算填充)
                    
                    self.option_snapshot[sym] = old_snap
                    if USE_5M_OPTION_DATA:
                        self.option_snapshot_5m[sym] = old_snap.copy()
                    
                    self.warmup_needed[sym] = False
    
                # 5min Options (仅在 5 分钟整数倍时间点入库)
                # 🚀 [Fix] 如果缺失 5m 专用桶数据，自动复用 1m 桶数据作为 Fallback
                if USE_5M_OPTION_DATA:
                    buckets_5m = payload.get('buckets_5m')
                    if not buckets_5m or len(buckets_5m) == 0:
                        buckets_5m = payload.get('option_buckets_5m')
                        
                    if not buckets_5m or len(buckets_5m) == 0:
                        if dt_ny.minute % 5 == 0:
                            buckets_5m = payload.get('buckets')
                            if not buckets_5m or len(buckets_5m) == 0:
                                buckets_5m = payload.get('option_buckets')
                        
                    if buckets_5m and len(buckets_5m) > 0 and dt_ny.minute % 5 == 0:
                        arr = np.array(buckets_5m, dtype=np.float32)
                        if arr.shape[1] < 12:
                            pad = np.zeros((arr.shape[0], 12 - arr.shape[1]), dtype=np.float32)
                            arr = np.hstack([arr, pad])
                        if arr.shape[0] < 6: arr = np.vstack([arr, np.zeros((6 - arr.shape[0], 12), dtype=np.float32)])
                        
                        # 🚀 [5m 期权同样强行重算]
                        valid_quote_mask_5m = (arr[:, 8] > 0) & (arr[:, 9] > 0)
                        arr[valid_quote_mask_5m, 0] = (arr[valid_quote_mask_5m, 8] + arr[valid_quote_mask_5m, 9]) / 2.0

                        if np.sum(arr[:, 6]) > 0.0001: 
                            # 🚀 [核心对齐] 5m 桶同样直接使用 Depth 快照
                            minute_vol = arr[:, 6]
                            self.last_cum_volume_5m[sym] = minute_vol.copy()
                            arr[:, 6] = minute_vol
                            self.option_snapshot_5m[sym] = arr
    
                # 4. [实时更新] :55 秒早起预刷新
                if not is_exact_boundary and dt_ny.second >= 55:
                    self._finalize_1min_bar(sym, curr_minute, cleanup=False)

            
            # [Fix] 删除遗留的 payload.get('last', 0) 覆写 Bug
            # 改为从安全的 stock 字典中再次兜底读取，或者干脆什么都不做
                if sym in self.latest_prices:
                     stock_data = payload.get('stock', {})
                     if stock_data and stock_data.get('close'):
                         self.latest_prices[sym] = float(stock_data.get('close', 0.0))
                         self.last_tick_price[sym] = self.latest_prices[sym]

        except Exception as e:
            logger.error(f"Process Error: {e}")

    def _get_dynamic_atm_iv(self, snap, price):
        """🚀 [SDS 5.0 动态探测] 函数式提取最接近现价的 ATM IV 对"""
        if snap is None or len(snap) < 2: return 0.0, 0.0
        
        try:
            # 1. 物理由于由于由于提取所有 Strike (Row 0, 2, 4 的 Col 0)
            strikes = np.array([snap[0, 0], snap[2, 0], snap[4, 0]])
            # 2. 物理由于由于寻找最近行索引 k
            k_idx = np.argmin(np.abs(strikes - price))
            row_start = k_idx * 2
            
            # Row k: Put | Row k+1: Call (符合 greeks_math 矩阵定义)
            p_iv = float(snap[row_start, 7])
            c_iv = float(snap[row_start + 1, 7])
            
            # 审计日志
            if price > 0:
                logger.debug(f"🎯 [ATM Pick] Price: {price:.2f} | Best Strike: {strikes[k_idx]:.2f} (Row {row_start}/{row_start+1}) | P_IV: {p_iv:.4f} | C_IV: {c_iv:.4f}")
            
            return c_iv, p_iv
        except Exception as e:
            logger.warning(f"⚠️ [ATM Pick Error]: {e}")
            return 0.0, 0.0

    def _finalize_1min_bar(self, sym, dt, cleanup=True):
        """🚀 [SDS 5.0 终极确定性管道] 全量缓冲区聚合 -> 动态 ATM 探测 -> 差分成交量对位"""
        bars = self.current_bars_5s.get(sym, [])
        
        # 1. 指纹提取助手
        def gvx(b, k, alt, default=0): 
            s_data = b.get('stock', {})
            return float(s_data.get(k, s_data.get(alt, default)))
            
        # 2. 确定性采样段 (Last-Tick Sequence)
        # 筛选有效 Ticks (即便没有成交，replayer 也会带 close 价)
        valid_ticks = [b for b in bars if b.get('stock') and gvx(b, 'close', 'c') > 0]
        
        if not valid_ticks:
            # [市场静默逻辑 - 保持价格连续性]
            c_raw = self.last_tick_price.get(sym, 0.0)
            if c_raw == 0: c_raw = self.latest_prices.get(sym, 0.0)
            if c_raw == 0: return 
            o = h = l = c_raw
            v = 0.0
            vwap = c_raw
            last_minute_accum = self.last_total_volume.get(sym, 0.0)
        else:
            # 🚀 [SDS 5.0 物理对锁] 取分钟缓冲区最后一帧作为结算基准
            last_tick = valid_ticks[-1]
            o = gvx(valid_ticks[0], 'open', 'o')
            c_raw = gvx(last_tick, 'close', 'c')
            h = max(max(gvx(b, 'high', 'h', c_raw), gvx(b, 'close', 'c', c_raw)) for b in valid_ticks)
            l = min(min(gvx(b, 'low', 'l', c_raw), gvx(b, 'close', 'c', c_raw)) for b in valid_ticks)
            
            # 🚀 [🔥 成交量精度复位] 物理计算当日累积量的区间差值
            # 注意：Polygon 1s 原始回放中 volume 字段是当日累计量
            current_total_vol = gvx(last_tick, 'volume', 'v')
            # 只有在初次运行且没有 last_total_volume 时才使用 last_accum 作为基准
            prev_total_vol = self.last_total_volume.get(sym, current_total_vol)
            v = max(0.0, current_total_vol - prev_total_vol)
            
            # 记录此时刻的累积成交量，用于下一分钟差分
            last_minute_accum = current_total_vol
            
            # VWAP 物理由于计算
            pv_sum = sum(gvx(b, 'vwap', 'close', gvx(b, 'c', c_raw)) * max(1e-10, gvx(b, 'volume', 'v', 0) - gvx(valid_ticks[i-1], 'volume', 'v', 0) if i > 0 else 0) for i, b in enumerate(valid_ticks))
            vwap = pv_sum / (v + 1e-10) if v > 0 else c_raw

        # 3. 期权特征动态映射 (ATM Selection Parity)
        # 逆向搜索缓冲区，找到这一分钟最后一条报文
        last_opt_tick = next((b for b in reversed(bars) if b.get('buckets') or b.get('option_buckets')), None)
        
        if last_opt_tick:
            o_buckets_raw = last_opt_tick.get('buckets', last_opt_tick.get('option_buckets', []))
            o_contracts = last_opt_tick.get('contracts', last_opt_tick.get('option_contracts', []))
        else:
            o_buckets_raw = self.latest_opt_buckets.get(sym, [])
            o_contracts = self.latest_opt_contracts.get(sym, [])

        # 🚀 [SDS 5.0 核心探测器] 动态由于寻找最接近现价的行索引，终结 0.81x 常态偏位
        snap = self.option_snapshot.get(sym)
        atm_c_iv, atm_p_iv = self._get_dynamic_atm_iv(snap, c_raw)
        
        if sym == 'NVDA':
            logger.info(f"📈 [SDS 5.0 Parity] {dt.strftime('%H:%M:%S')} | c: {c_raw:.2f} | v: {v:.0f} | p_iv: {atm_p_iv:.4f} | c_iv: {atm_c_iv:.4f}")

        # 持久化成交量游标
        self.last_total_volume[sym] = last_minute_accum
        
        # 4. 指纹组装与 Redis 持久化
        ts_key = int(dt.replace(second=0, microsecond=0).timestamp())
        o_buckets = o_buckets_raw.tolist() if hasattr(o_buckets_raw, 'tolist') else o_buckets_raw
        if isinstance(o_contracts, np.ndarray): o_contracts = o_contracts.tolist()

        bar_payload = {
            'open': float(o), 'high': float(h), 'low': float(l), 
            'close': float(c_raw), 'volume': float(v), 'vwap': float(vwap)
        }
        opt_payload = {
            'buckets': o_buckets,
            'contracts': o_contracts,
            'atm_c_iv': atm_c_iv,
            'atm_p_iv': atm_p_iv
        }

        # 4. 写入 Redis 真相账本
        success = False
        try:
            r1 = self.r.hset(f"BAR:1M:{sym}", str(int(ts_key)), json.dumps(bar_payload))
            r2 = self.r.hset(f"BAR_OPT:1M:{sym}", str(int(ts_key)), json.dumps(opt_payload))
            self.r.expire(f"BAR:1M:{sym}", 172800)
            self.r.expire(f"BAR_OPT:1M:{sym}", 172800)
            success = True
            if sym == 'NVDA' and ts_key % 300 == 0:
                logger.info(f"✅ [Redis Flush] Persisted 1m bar for {sym} at {ts_key}")
        except Exception as se:
            logger.warning(f"❌ [Redis Write Error] Failed to write 1m/opt bars for {sym}: {se}")

        # 5. [🚀 极速对位] 增量更新内存状态，取代昂贵的全量回读
        should_full_sync = False
        if not success:
            should_full_sync = True
        elif self.history_1min[sym].empty:
            should_full_sync = True
        else:
            # 检查内存中的最后一个时间戳，如果有 Gap (漏单)，必须强制同步
            last_ts_loc = self.history_1min[sym].index[-1].timestamp()
            if ts_key - last_ts_loc > 65: # 允许少量偏移，但不能跳过整分钟
                should_full_sync = True

        if should_full_sync:
            # [异常路径] 触发昂贵的全量回读解析
            self._sync_history_from_redis(sym)
            self._sync_option_history_from_redis(sym)
        else:
            # [极速路径] 直接增量更新内存 DataFrame，开销忽略不计
            new_ts = pd.Timestamp(ts_key, unit='s', tz=NY_TZ)
            for k, v in bar_payload.items():
                self.history_1min[sym].loc[new_ts, k] = v
            # 限制长度
            if len(self.history_1min[sym]) > 500:
                self.history_1min[sym] = self.history_1min[sym].iloc[-500:]
            
            # 期权快照已在 process_market_data 中实时通过 option_snapshot 更新，
            # 且刚才已经物理 reconstruct 过了（如果需要同步），此处对位已完成。
        
        # 6. 生成 5min K 线 (Resample)
        df_1m = self.history_1min[sym]
        if not df_1m.empty and len(df_1m) >= 5:
            # 🚀 [优化] 只有在跨 5 分钟边界或 1min 更新后触发生
            df_5m = df_1m.resample('5min', closed='left', label='left').agg({
                'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
            }).dropna()
            self.history_5min[sym] = df_5m.iloc[-100:]
            
        # 7. 物理内存回收到到到 (SDS 4.0 Cleanup)
        if cleanup:
            self.current_bars_5s[sym] = []
            self.finalization_snapshots.pop(sym, None)
            self.pending_finalization.pop(sym, None)
        
        return True

        if cleanup:
            self.current_bars_5s[sym] = []

    def finalization_barrier(self):
        """🚀 [SDS 2.0 结算屏障] 确保在希腊值计算全部完成后执行分钟级 K 线落盘"""
        if not hasattr(self, 'pending_finalization') or not self.pending_finalization:
            return
            
        pending_items = list(self.pending_finalization.items())
        for sym, dt in pending_items:
            # 执行结算
            self._finalize_1min_bar(sym, dt, cleanup=True)
            # 释放锁
            self.pending_finalization.pop(sym, None)
            self.finalization_snapshots.pop(sym, None)
            
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

    def _inject_temporal_derivatives(self, sym: str, raw_vec: np.ndarray, is_new_minute: bool = False):
        """
        [极速截胡] 计算并注入时序导数特征 (期权动量/加速度/量价背离)
        """
        idx_iv = self.feat_name_to_idx.get('options_vw_iv')
        idx_gamma = self.feat_name_to_idx.get('options_vw_gamma')
        
        curr_iv = raw_vec[idx_iv] if idx_iv is not None else 0.0
        curr_gamma = raw_vec[idx_gamma] if idx_gamma is not None else 0.0
        curr_price = self.latest_prices.get(sym, 1.0)
        
        # 🚀 [致命时序坍缩修复]：只有物理时间跨越了 1 分钟，才往队列里 append
        # 这样保证队列的长度代表了真正的“物理分钟数”，而不是“计算触发次数”！
        if not self.deriv_history[sym]:
            self.deriv_history[sym].append({'iv': curr_iv, 'gamma': curr_gamma, 'price': curr_price})
        elif is_new_minute:
            self.deriv_history[sym].append({'iv': curr_iv, 'gamma': curr_gamma, 'price': curr_price})
            
        mom_iv, acc_gamma, div_iv = 0.0, 0.0, 0.0
        
        # 提取 5 分钟前状态进行对比计算 (如果不足 5 分钟，取最老的一帧)
        if len(self.deriv_history[sym]) > 1:
            lookback_idx = max(0, len(self.deriv_history[sym]) - 6)
            prev = self.deriv_history[sym][lookback_idx]
            eps = 1e-6
            
            mom_iv = (curr_iv - prev['iv']) / (prev['iv'] if abs(prev['iv']) > eps else 1.0)
            acc_gamma = (curr_gamma - prev['gamma']) / (prev['gamma'] if abs(prev['gamma']) > eps else 1.0)
            price_ret = (curr_price - prev['price']) / (prev['price'] if prev['price'] > eps else 1.0)
            div_iv = mom_iv - price_ret
            
        idx_mom = self.feat_name_to_idx.get('options_iv_momentum')
        idx_acc = self.feat_name_to_idx.get('options_gamma_accel')
        idx_div = self.feat_name_to_idx.get('options_iv_divergence')
        
        if idx_mom is not None: raw_vec[idx_mom] = mom_iv
        if idx_acc is not None: raw_vec[idx_acc] = acc_gamma
        if idx_div is not None: raw_vec[idx_div] = div_iv
     
    def _align_inference_timing(self, ts_from_payload):
        """[V8 Helper] 核心对位层：确定推断目标时间戳并执行时序审计"""
        current_replay_ts = ts_from_payload if ts_from_payload else self.r.get("replay:current_ts")
        sync_ts = ts_from_payload if ts_from_payload else time.time()
        
        target_time = datetime.fromtimestamp(sync_ts, NY_TZ)
        current_minute_ts = int(sync_ts // 60) * 60
        alpha_label_ts = current_minute_ts - 60
        
        # 1. 拒止检查与符号隔离 (🚨 物理隔离：排除指数不参与交易)
        NON_TRADE_SYMBOLS = {'SPY', 'QQQ', 'VIXY'}
        ready_trading_symbols = [s for s in self.symbols if s not in NON_TRADE_SYMBOLS and not self.history_1min[s].empty]
        
        if not ready_trading_symbols or len(self.history_1min[ready_trading_symbols[0]]) < 30:
            missing = [s for s in self.symbols if s not in NON_TRADE_SYMBOLS and s not in ready_trading_symbols]
            if len(ready_trading_symbols) < 30:
                logger.info(f"⏳ [Warmup Audit] {len(ready_trading_symbols)} symbols ready, {len(missing)} missing: {missing[:5]}")
            return None

        # 2. 时序审计 (10s 汇报频率)
        data_ts = float(current_replay_ts) if current_replay_ts else target_time.timestamp()
        if int(sync_ts) % 10 == 0:
            logger.info(f"🕒 [Time-Sync] Tick:{target_time.strftime('%H:%M:%S')} | Alpha_Log_TS:{datetime.fromtimestamp(data_ts, NY_TZ).strftime('%H:%M:%S')} | Trading_Symbols:{len(ready_trading_symbols)}")

        # 3. 边界判定 (Align 1m Reference)
        is_boundary = (int(sync_ts) % 60 == 0)
        is_new_minute = False
        if getattr(self, 'last_model_minute_ts', 0) == 0 and not is_boundary:
            return None
        if is_boundary and getattr(self, 'last_model_minute_ts', 0) < current_minute_ts:
            is_new_minute = True
            self.last_model_minute_ts = current_minute_ts
            
        return alpha_label_ts, data_ts, is_new_minute, ready_trading_symbols, sync_ts

    def _step_engine_compute(self, data_ts, is_new_minute, alpha_label_ts):
        """[V8 Helper] 引擎计算层：执行底层指标运算与 Redis 同步"""
        # 只有在整分，或者缓存失效时重新计算 engine
        if is_new_minute or getattr(self, 'cached_batch_raw', None) is None:
            source_snap = self.option_snapshot 
            source_snap_5m = getattr(self, 'option_snapshot_5m', {})
            # 物理截断前 12 列，防止维数震荡
            sliced_snaps = {s: snap[:, :12].copy() for s, snap in source_snap.items()}
            sliced_snaps_5m = {s: (snap[:, :12].copy() if isinstance(snap, np.ndarray) else np.zeros((6,12))) for s, snap in source_snap_5m.items()}

            try:
                results_map = self.engine.compute_all_inputs(
                    history_1min=self.history_1min, history_5min=getattr(self, 'history_5min', {}),
                    fast_feats=self.fast_feat_names, slow_feats=self.slow_feat_names,
                    option_snapshots=sliced_snaps,  option_contracts=self.latest_opt_contracts,
                    option_snapshot_5m=sliced_snaps_5m, feat_resolutions=getattr(self, 'feat_resolutions', {}),
                    current_ts=data_ts 
                )
            except Exception as e:
                logger.error(f"Engine Compute Error: {e}", exc_info=True); return None

            batch_raw, valid_mask = [], []
            for sym in self.symbols:
                raw_vec = np.zeros(len(self.all_feat_names), dtype=np.float32)
                is_valid = False
                if sym in results_map:
                    res_sym = results_map[sym]; is_valid = True
                    # 🚀 [🔥 终极闭环修复] 回写与同步 IV/Greeks
                    if res_sym.get('updated_buckets') is not None:
                        enriched = res_sym['updated_buckets']
                        self.latest_opt_buckets[sym] = enriched
                        try:
                            # 🚀 [🔥 终极报文扩容] 显式由于由于由于提取并写入原子化字段，方便 verify_parity 对位
                            atm_c_iv = float(enriched[2, 7]) if enriched.shape[0] > 2 else 0.0
                            atm_p_iv = float(enriched[0, 7]) if enriched.shape[0] > 0 else 0.0
                            
                            payload = {
                                'buckets': enriched.tolist() if hasattr(enriched, 'tolist') else enriched, 
                                'contracts': self.latest_opt_contracts.get(sym, []),
                                'atm_c_iv': atm_c_iv,
                                'atm_p_iv': atm_p_iv
                            }
                            self.r.hset(f"BAR_OPT:1M:{sym}", str(int(alpha_label_ts)), json.dumps(payload))
                        except Exception as e:
                            logger.error(f"❌ [Redis Write Error] Failed to write optic bar for {sym}: {e}")
                    
                    # 提取特征并物理回写至 history_1min
                    for ftype in ['fast_1m', 'slow_1m']:
                        if res_sym.get(ftype) is not None:
                            latest = res_sym[ftype][0, :, -1].cpu().numpy()
                            names = self.fast_feat_names if ftype == 'fast_1m' else self.slow_feat_names
                            for i, fname in enumerate(names):
                                if i < len(latest): 
                                    val = float(latest[i])
                                    raw_vec[self.feat_name_to_idx[fname]] = val
                                    # 🚀 [核心对冲] 同步指标回 DataFrame，确保下游能取到 EMA_ROC 等特征
                                    if is_new_minute:
                                        ts_logi = pd.Timestamp(alpha_label_ts, unit='s', tz=NY_TZ)
                                        self.history_1min[sym].loc[ts_logi, fname] = val

                # 🚀 [回填基线] 每分钟的基础期权数据物理落盘
                if is_new_minute:
                    row_ts = pd.Timestamp(alpha_label_ts, unit='s', tz=NY_TZ)
                    if row_ts in self.history_1min[sym].index:
                        for fname in self.slow_feat_names:
                            if 'options_' in fname: self.history_1min[sym].at[row_ts, fname] = raw_vec[self.feat_name_to_idx[fname]]

                # 时序导数计算 (Momentum/Accel)
                self._inject_temporal_derivatives(sym, raw_vec, is_new_minute=is_new_minute)
                batch_raw.append(raw_vec); valid_mask.append(is_valid)

            self.cached_results_map, self.cached_batch_raw, self.cached_valid_mask = results_map, batch_raw, valid_mask
        else:
            results_map, batch_raw, valid_mask = self.cached_results_map, self.cached_batch_raw, self.cached_valid_mask

        return batch_raw, valid_mask, results_map

    def _apply_normalization_sequence(self, batch_raw, valid_mask, data_ts, is_new_minute):
        """[V8 Helper] 归一化层：执行 Z-Score 并维护 30 步长宏观历史"""
        dt_ny = datetime.fromtimestamp(data_ts, NY_TZ)
        date_str = dt_ny.strftime('%Y%m%d')
        if not hasattr(self, 'created_debug_dates'): self.created_debug_dates = set()
        if date_str not in self.created_debug_dates: self._ensure_debug_tables(date_str); self.created_debug_dates.add(date_str)
        
        batch_norm = []
        current_minutes = dt_ny.hour * 60 + dt_ny.minute
        is_rth = (570 <= current_minutes < 960)
        
        for b_idx, sym in enumerate(self.symbols):
            raw_vec = batch_raw[b_idx]; is_valid = valid_mask[b_idx]; norm = self.normalizers[sym]
            if is_valid and is_rth and is_new_minute: norm_vec = norm.process_frame(raw_vec)
            else: norm_vec = norm.normalize_only(raw_vec)
            batch_norm.append(norm_vec)
            
        norm_mat = np.stack(batch_norm)
        if not hasattr(self, 'norm_history_30'): self.norm_history_30 = []
        if is_new_minute:
            self.norm_history_30.append(norm_mat)
            if len(self.norm_history_30) > 30: self.norm_history_30.pop(0)
        elif self.norm_history_30: self.norm_history_30[-1] = norm_mat
        else: self.norm_history_30.append(norm_mat)
        
        padded_hist = [self.norm_history_30[0]] * (30 - len(self.norm_history_30)) + self.norm_history_30
        return np.stack(padded_hist, axis=1)
    def _assemble_compute_payload(
        self, 
        norm_seq_30: np.ndarray, 
        batch_raw: list, 
        valid_mask: list, 
        results_map: dict, 
        alpha_label_ts: float, 
        data_ts: float, 
        is_new_minute: bool, 
        ready_symbols: list
    ) -> Optional[dict]:
        """
        组装发送给 SignalEngine 或 Redis 的最终 Payload。
        已完美适配上游参数，并集成时序防坍塌、成交量获取、时光冻结等全部核心修复。
        """
        from datetime import datetime
        import numpy as np
        
        # 将传入的时间和列表转化为内部所需的格式
        dt_ny_payload = datetime.fromtimestamp(alpha_label_ts, NY_TZ)
        raw_mat = np.stack(batch_raw)
        
        batch_symbols, batch_prices, batch_volumes, batch_stock_ids = [], [], [], []
        batch_spy_rocs, batch_qqq_rocs, batch_fast_vols = [], [], []
        cheat_call, cheat_put, cheat_call_bid, cheat_call_ask = [], [], [], []
        cheat_put_bid, cheat_put_ask, cheat_call_iv, cheat_put_iv = [], [], [], []
        valid_b_indices = []

        # 🚀 [时光冻结机制] 取回含有完美 Greeks 的上一分钟末期权快照，保证时序严格对齐
        source_opt_buckets = getattr(self, 'frozen_latest_opt_buckets', self.latest_opt_buckets) if is_new_minute else self.latest_opt_buckets
        source_snap_for_payload = getattr(self, 'frozen_option_snapshot', self.option_snapshot) if is_new_minute else self.option_snapshot

        for b_idx, sym in enumerate(self.symbols):
            if sym not in ready_symbols or sym not in results_map:
                continue
                
            # 过滤无效数据
            if not valid_mask[b_idx]:
                continue
                
            valid_b_indices.append(b_idx)
            batch_symbols.append(sym)
            
            # 🚀 [成交量修复] 同时提取 close 和 volume，供下游写入 Redis 账本
            if is_new_minute:
                try: 
                    target_time = dt_ny_payload.replace(second=0, microsecond=0)
                    p_close = self.history_1min[sym].at[target_time, 'close']
                    p_vol = self.history_1min[sym].at[target_time, 'volume']
                except: 
                    p_close = self.latest_prices.get(sym, 0.0)
                    p_vol = 0.0
                batch_prices.append(float(p_close))
                batch_volumes.append(float(p_vol))
            else:
                batch_prices.append(self.latest_prices.get(sym, 0.0))
                batch_volumes.append(0.0)
            
            batch_stock_ids.append(b_idx)
            
            idx_spy = self.feat_name_to_idx.get('spy_roc_5min')
            batch_spy_rocs.append(raw_mat[b_idx, idx_spy] if idx_spy is not None else 0.0)
            idx_qqq = self.feat_name_to_idx.get('qqq_roc_5min')
            batch_qqq_rocs.append(raw_mat[b_idx, idx_qqq] if idx_qqq is not None else 0.0)
            idx_vol = self.feat_name_to_idx.get('fast_vol')
            batch_fast_vols.append(raw_mat[b_idx, idx_vol] if idx_vol is not None else 0.0)
            
            # 提取期权快照
            snap = source_opt_buckets.get(sym)
            if snap is None: 
                snap = source_snap_for_payload.get(sym)

            if snap is not None and isinstance(snap, np.ndarray) and snap.shape[0] >= 6:
                c_iv, p_iv = snap[2, 7], snap[0, 7]
                cheat_call.append(snap[2, 0]); cheat_call_iv.append(c_iv)
                cheat_call_bid.append(snap[2, 8]); cheat_call_ask.append(snap[2, 9])
                cheat_put.append(snap[0, 0]); cheat_put_iv.append(p_iv)
                cheat_put_bid.append(snap[0, 8]); cheat_put_ask.append(snap[0, 9])
            else:
                for lst in [cheat_call, cheat_call_iv, cheat_call_bid, cheat_call_ask, cheat_put, cheat_put_iv, cheat_put_bid, cheat_put_ask]:
                    lst.append(0.0)

        if not batch_symbols:
            return None

        valid_b_indices = np.array(valid_b_indices)
        valid_norm_seq = norm_seq_30[valid_b_indices] 
        
        # 🚀 [致命修复：时空倒置与维度坍塌] 
        # 直接提取 [Batch, Time] 切片，绝对不能使用 .transpose(1, 0)
        features_dict = {
            fname: valid_norm_seq[:, :, self.feat_name_to_idx[fname]] 
            for fname in self.slow_feat_names
        }

        live_options = {}
        live_options_5m = {} 
        
        from config import USE_5M_OPTION_DATA
        source_snap_5m_for_payload = getattr(self, 'frozen_option_snapshot_5m', getattr(self, 'option_snapshot_5m', {})) if is_new_minute else getattr(self, 'option_snapshot_5m', {})

        for sym in batch_symbols:
            live_options[sym] = {
                'buckets': source_opt_buckets.get(sym, []),
                'contracts': getattr(self, 'latest_opt_contracts', {}).get(sym, [])
            }
            if USE_5M_OPTION_DATA:
                snap_5m = source_snap_5m_for_payload.get(sym)
                if snap_5m is not None:
                    live_options_5m[sym] = {
                        'buckets': snap_5m.tolist() if isinstance(snap_5m, np.ndarray) else snap_5m,
                        'contracts': getattr(self, 'latest_opt_contracts', {}).get(sym, [])
                    }
        
        # 🚀 [动态预热识别] 基于底层 Normalizer 的实际投喂次数
        sample_sym = batch_symbols[0]
        is_warmed_up = bool(self.normalizers[sample_sym].count >= 30)

        payload = {
            'ts': alpha_label_ts, 
            'log_ts': data_ts, 
            'symbols': batch_symbols,
            'stock_price': batch_prices,
            'stock_volume': np.array(batch_volumes), # 🚀 修复丢失的成交量
            'stock_id': np.array(batch_stock_ids),
            'sector_id': np.zeros(len(batch_symbols)),
            'fast_vol': np.array(batch_fast_vols),
            'spy_roc_5min': np.array(batch_spy_rocs),
            'qqq_roc_5min': np.array(batch_qqq_rocs),
            'features_dict': features_dict,
            'live_options': live_options,
            'live_options_5m': live_options_5m,
            'is_new_minute': is_new_minute,  
            'is_warmed_up': is_warmed_up,
            'cheat_call': cheat_call, 'cheat_put': cheat_put,
            'cheat_call_bid': cheat_call_bid, 'cheat_call_ask': cheat_call_ask,
            'cheat_put_bid': cheat_put_bid, 'cheat_put_ask': cheat_put_ask,
            'cheat_call_iv': cheat_call_iv, 'cheat_put_iv': cheat_put_iv,
        }
        
        return payload
          
    async def run_compute_cycle(self, ts_from_payload=None, return_payload=False):
        """[V8 终极模块化版] 特征计算主循环"""
        # 1. 时序对齐与预热审计
        t_ctx = self._align_inference_timing(ts_from_payload)
        if not t_ctx:
            f_ts = ts_from_payload if ts_from_payload else time.time()
            self.r.set("sync:feature_calc_done", f_ts); self.r.set("sync:orch_done", f_ts); return None
        alpha_label_ts, data_ts, is_new_minute, ready_symbols, sync_ts = t_ctx
        
        # 2. 引擎计算
        engine_ctx = self._step_engine_compute(data_ts, is_new_minute, alpha_label_ts)
        if not engine_ctx: return None
        batch_raw, valid_mask, results_map = engine_ctx

        # 3. 归一化与 30 帧缓存维护
        norm_seq_30 = self._apply_normalization_sequence(batch_raw, valid_mask, data_ts, is_new_minute)

        # 4. 组装推流 Payload
        payload = self._assemble_compute_payload(norm_seq_30, batch_raw, valid_mask, results_map, alpha_label_ts, data_ts, is_new_minute, ready_symbols)
        if not payload: return None
        if return_payload: return payload
        
        # 5. Redis 发送与状态同步
        payload_sent = False
        try:
            self.r.xadd(self.redis_cfg['output_stream'], {'data': ser.pack(payload)}, maxlen=100)
            payload_sent = True 
        except Exception as e: logger.error(f"❌ Redis XADD Error: {e}")
            
        from config import IS_SIMULATED
        if IS_SIMULATED:
            self.r.set("sync:feature_calc_done", sync_ts)
            if not payload_sent: self.r.set("sync:orch_done", sync_ts)

        return None

    async def run_compute_cycle_raw(self, ts_from_payload=None, return_payload=False):
        """
        [V8 终极版] 特征计算主循环 - (完美兼容秒级与分钟级对齐版)
        """
        # [🔥 修正] 优先使用来自消息 Payload 的显式时间戳
        current_replay_ts = ts_from_payload if ts_from_payload else self.r.get("replay:current_ts")
        
        # 🚀 [暴力调试] 直接打印每一秒的时间戳到 stdout
        
        sync_ts = ts_from_payload if ts_from_payload else time.time()
        target_time = datetime.fromtimestamp(sync_ts, NY_TZ)
        current_minute_ts = int(sync_ts // 60) * 60
        
        # 🚀 [🔥 终极协议对齐：Label 修正]
        # 基准协议规定：10:00:00-10:00:59 的 Bar 标签为 10:00:00。
        # 我们在 10:01:00 附近完成该 Bar 的结算与特征计算。
        # 因此，此处的 alpha_label_ts 必须回退 60s，以确保 Alpha 日志中的 10:00:00 行对应 10:00:00 的数据。
        alpha_label_ts = current_minute_ts - 60

        # 🚀 [Parity Fix] 重新定义 ready_symbols，确保作用域内所有索引对位
        ready_symbols = [s for s in self.symbols if not self.history_1min[s].empty]
        if not ready_symbols:
            self.r.set("sync:feature_calc_done", sync_ts)
            self.r.set("sync:orch_done", sync_ts)
            return None
            
        # ---------------- 拒止推理安全检查 ----------------
        sample_s = ready_symbols[0]
        curr_len = len(self.history_1min[sample_s])
        required_len = 30
        
        if curr_len < required_len:
            logger.debug(f"[数据不足-拒止推理] 历史K线 {curr_len} 根，要求 {required_len} 根，等待预热。")
            self.r.set("sync:feature_calc_done", sync_ts)
            self.r.set("sync:orch_done", sync_ts)
            return

        # =======================================================
        # 🚀 [修复] 分钟跨越标志 & 时间戳分离
        # - data_ts: 用于归一化/序列组装，始终用 target_time（分钟级精度）
        # =======================================================
        # 必须使用外部传入的精确到 1s 的 current_replay_ts！
        # 绝不能用 target_time.timestamp()，否则时间戳会被强行拉回整分钟 (00秒)，导致 OMS 同步死锁！
        data_ts = float(current_replay_ts) if current_replay_ts else target_time.timestamp()
        
        # [Time-Guard] 严格校时与强制异常终止
        # 计算 数据采集时间 (Payload) 与 模型推断时间 (Nominal) 的偏差
        time_skew = abs(data_ts - sync_ts)
        if time_skew > 60.0:
            logger.critical(f"🛑 [TIME_FATAL] 检测到严重时序偏移! Tick:{sync_ts} Data:{data_ts} Skew:{time_skew:.1f}s")
            logger.critical(f"可能原因: 1. Driver 推送过快累积延迟 2. 内部 Buffer 堆积 3. 数据源本身时间错乱")
            # 这里的 terminte 逻辑可以根据需要调整，暂时打印并报错
            # sys.exit(1) 
        
        if int(sync_ts) % 10 == 0: # 提高频率到 10s 一次
            logger.info(f"🕒 [Time-Sync] Tick:{datetime.fromtimestamp(sync_ts, NY_TZ).strftime('%H:%M:%S')} | Alpha_Log_TS:{datetime.fromtimestamp(data_ts, NY_TZ).strftime('%H:%M:%S')} | Symbols:{len(self.symbols)}")

        is_new_minute = False
        is_boundary = (int(sync_ts) % 60 == 0)
        
        # 🚀 [Parity Fix] 严格过滤：在遇到当天的第一个“整分钟边界”之前，抛弃所有 tick。
        # 这样确保 alpha_logs 的第一条记录必然是整分钟（比如 10:00:00），避免在 09:59:55 时抢跑输出。
        if getattr(self, 'last_model_minute_ts', 0) == 0 and not is_boundary:
            if int(sync_ts) % 10 == 0: logger.debug("Waiting for the first minute boundary to align with 1m reference...")
            if current_replay_ts:
                self.r.set("sync:feature_calc_done", current_replay_ts)
                self.r.set("sync:orch_done", current_replay_ts)
            return

        if is_boundary and getattr(self, 'last_model_minute_ts', 0) < current_minute_ts:
            is_new_minute = True
            self.last_model_minute_ts = current_minute_ts
        
        # [Parity & Speed] Only recompute raw features from engine at minute boundaries
        if is_new_minute or getattr(self, 'cached_batch_raw', None) is None:
            # ---------------- 1. 触发 Engine 计算 ----------------
            # 🚀 [SECURITY_FIX] 使用 .copy() 彻底隔断原本 snap 的内存引用。
            
            # 👇 [手术 3 简化版：抛弃本地平移，拥抱 Redis 对位]
            # 既然已有 Redis 标准化账本，我们直接引用实时同步后的 option_snapshot
            source_snap = self.option_snapshot 
            source_snap_5m = getattr(self, 'option_snapshot_5m', {})
            
            sliced_snaps = {s: snap[:, :12].copy() for s, snap in source_snap.items()}
            sliced_snaps_5m = {s: (snap[:, :12].copy() if isinstance(snap, np.ndarray) else np.zeros((6,12))) for s, snap in source_snap_5m.items()}
            # 👆 [手术 3 简化结束]

            try:
                results_map = self.engine.compute_all_inputs(
                    history_1min=self.history_1min,
                    history_5min=getattr(self, 'history_5min', {}),
                    fast_feats=self.fast_feat_names,
                    slow_feats=self.slow_feat_names,
                    option_snapshots=sliced_snaps,  
                    option_contracts=self.latest_opt_contracts,
                    option_snapshot_5m=sliced_snaps_5m,
                    feat_resolutions=getattr(self, 'feat_resolutions', {}),
                    current_ts=data_ts 
                )
            except Exception as e:
                logger.error(f"Engine Compute Error: {e}", exc_info=True)
                if current_replay_ts: 
                    self.r.set("sync:feature_calc_done", current_replay_ts)
                    self.r.set("sync:orch_done", current_replay_ts)
                return

            # ---------------- 2. 解析与解包 ----------------
            batch_raw = []
            valid_mask = []
            lx_fast_debug = []
            lx_slow_debug = []
            
            for sym in self.symbols:
                raw_vec = np.zeros(len(self.all_feat_names), dtype=np.float32)
                is_valid = False
                
                if sym in results_map:
                    res_sym = results_map[sym]
                    is_valid = True
                    
                    # 🚀 [🔥 终极闭环修复] 将引擎中算出的正确希腊值阵列，回写覆盖系统级快照并同步 Redis！
                    if res_sym.get('updated_buckets') is not None:
                        enriched_buckets = res_sym['updated_buckets']
                        self.latest_opt_buckets[sym] = enriched_buckets
                        
                        # 🚀 [核心对位] IV 算完后，必须立刻把带希腊值的快照同步回 Redis
                        # 否则 Redis 里的 Bar 永远只有原始价格。
                        try:
                            opt_p = {
                                'buckets': enriched_buckets,
                                'contracts': self.latest_opt_contracts.get(sym, [])
                            }
                            # alpha_label_ts 通常是分钟边界起始点 (如 10:00:00)
                            self.r.hset(f"BAR_OPT:1M:{sym}", str(int(alpha_label_ts)), json.dumps(opt_p))
                        except Exception: pass
                    
                    if res_sym.get('fast_1m') is not None:
                        fast_latest = res_sym['fast_1m'][0, :, -1].cpu().numpy()
                        for i, fname in enumerate(self.fast_feat_names):
                            if i < len(fast_latest):
                                raw_vec[self.feat_name_to_idx[fname]] = fast_latest[i]
                                
                    if res_sym.get('slow_1m') is not None:
                        slow_latest = res_sym['slow_1m'][0, :, -1].cpu().numpy()
                        for i, fname in enumerate(self.slow_feat_names):
                            if i < len(slow_latest):
                                raw_vec[self.feat_name_to_idx[fname]] = slow_latest[i]

                # 🚀 [🔥 终极闭环：特征回填 (Feature Back-Fill)]
                # 将本分钟算出的 IV/Gamma 等高阶指标物理回写到 history_1min 账本中。
                # 只有这样，下一分钟算“动量 (Momentum)”时才能读到本分钟的基础数据。
                if is_new_minute:
                    target_row_ts = pd.Timestamp(alpha_label_ts, unit='s', tz=NY_TZ)
                    if target_row_ts in self.history_1min[sym].index:
                        # 挑选需要回填的期权特征列
                        for fname in self.slow_feat_names:
                            if 'options_' in fname:
                                val = raw_vec[self.feat_name_to_idx[fname]]
                                self.history_1min[sym].at[target_row_ts, fname] = val

                self._inject_temporal_derivatives(sym, raw_vec, is_new_minute=is_new_minute)
                batch_raw.append(raw_vec)
                valid_mask.append(is_valid)
                lx_fast_debug.append((sym, raw_vec[self.fast_indices]))
                lx_slow_debug.append((sym, raw_vec[self.slow_indices]))

            self.cached_results_map = results_map
            self.cached_batch_raw = batch_raw
            self.cached_valid_mask = valid_mask
            self.cached_lx_fast_debug = lx_fast_debug
            self.cached_lx_slow_debug = lx_slow_debug
            if getattr(self, '_batch_trace_count', 0) < 3:
                logger.info(f"🧪 [BATCH_TRACE] Branch: FULL_COMPUTE | Symbols: {len(self.symbols)} | BatchRaw_Len: {len(batch_raw)}")
                self._batch_trace_count = getattr(self, '_batch_trace_count', 0) + 1
        else:
            results_map = self.cached_results_map
            batch_raw = self.cached_batch_raw
            valid_mask = self.cached_valid_mask
            lx_fast_debug = self.cached_lx_fast_debug
            lx_slow_debug = self.cached_lx_slow_debug
            if getattr(self, '_batch_trace_count', 0) < 3:
                logger.info(f"🧪 [BATCH_TRACE] Branch: CACHE_LOAD | BatchRaw_Len: {len(batch_raw) if batch_raw else 'NONE'}")
                self._batch_trace_count = getattr(self, '_batch_trace_count', 0) + 1

        # 🚀 [终极防御] 强制兜底，防止 NoneType 导致回测崩溃
        if batch_raw is None:
            logger.warning("⚠️ [ROBUSTNESS_GUARD] batch_raw was None! Initializing to empty list.")
            batch_raw = []
            valid_mask = [False] * len(self.symbols)
            lx_fast_debug = []; lx_slow_debug = []

        # ---------------- 3. 主线程同步建表 & 异步写入 Debug 数据库 ----------------
        # 🚀 [主线程同步建表] 根据数据的实际时间戳建表。保证在主线程完成，彻底杜绝并发锁死！
        dt_ny = datetime.fromtimestamp(data_ts, NY_TZ)
        date_str = dt_ny.strftime('%Y%m%d')
        
        if not hasattr(self, 'created_debug_dates'):
            self.created_debug_dates = set()
            
        if date_str not in self.created_debug_dates:
            self._ensure_debug_tables(date_str)
            self.created_debug_dates.add(date_str)

        if lx_fast_debug or lx_slow_debug:
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
            # 仍在同一分钟内的 1s 级高频刷新！
            if len(self.norm_history_30) > 0:
                # 覆盖最后一帧，使模型看到最新盘口，但绝不拉长序列导致时间轴坍缩！
                self.norm_history_30[-1] = norm_mat
            else:
                self.norm_history_30.append(norm_mat)
            
        # ---------------- 5. 组装推流 Payload (The Dict-Payload) ----------------
        payload_sent = False  # 👈 追踪本帧是否发出了数据
        if len(self.norm_history_30) > 0:  
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
                
                # 🚀 [Surgery 23] 逻辑价格对齐：区分整分结算与秒级Tick
                if is_new_minute:
                    # 既然 payload_ts 是 data_ts - 60s (如 10:00:00)，我们就取该时刻结算收盘价
                    from datetime import timedelta
                    target_time_logi = target_time.replace(second=0, microsecond=0) - timedelta(minutes=1)
                    try:
                        p_close = self.history_1min[sym].at[target_time_logi, 'close']
                    except:
                        p_close = self.latest_prices.get(sym, 0.0)
                    batch_prices.append(float(p_close))
                else:
                    # 秒级采样时，使用实时最准价格
                    batch_prices.append(self.latest_prices.get(sym, 0.0))
                
                batch_stock_ids.append(b_idx)
                
                # 杂项特征提取
                idx_spy = self.feat_name_to_idx.get('spy_roc_5min')
                batch_spy_rocs.append(raw_mat[b_idx, idx_spy] if idx_spy is not None else 0.0)
                idx_qqq = self.feat_name_to_idx.get('qqq_roc_5min')
                batch_qqq_rocs.append(raw_mat[b_idx, idx_qqq] if idx_qqq is not None else 0.0)
                idx_vol = self.feat_name_to_idx.get('fast_vol')
                batch_fast_vols.append(raw_mat[b_idx, idx_vol] if idx_vol is not None else 0.0)
                
                # 👇 [手术 4a 简化版] 直接使用实时同步后的最新 Buckets
                source_opt_buckets = self.latest_opt_buckets
                source_snap_for_payload = self.option_snapshot 

                snap = source_opt_buckets.get(sym)
                if snap is None: snap = source_snap_for_payload.get(sym)

                if snap is not None and isinstance(snap, np.ndarray) and snap.shape[0] >= 6:
                    c_iv = snap[2, 7]
                    p_iv = snap[0, 7]
                    cheat_call.append(snap[2, 0]); cheat_call_iv.append(c_iv)
                    cheat_call_bid.append(snap[2, 8]); cheat_call_ask.append(snap[2, 9])
                    cheat_put.append(snap[0, 0]); cheat_put_iv.append(p_iv)
                    cheat_put_bid.append(snap[0, 8]); cheat_put_ask.append(snap[0, 9])
                    
                    if sym == 'NVDA' and getattr(self, '_iv_cycle_count', 0) < 5:
                        logger.info(f"🧪 [IV_TRACE_2] {sym} | Cycle IV Extract | Call_IV: {c_iv:.4f} | Put_IV: {p_iv:.4f}")
                        self._iv_cycle_count = getattr(self, '_iv_cycle_count', 0) + 1
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
                    # 🚀 [Parity Fix] 提取最后 30 帧，并转置为 [B, 30] 以对位 SignalEngineV8 的推理逻辑
                    # 只有这样，模型才能精准提取到“当前时刻” (Index -1) 的信号，而非 30 分钟前的残影。
                    # 🚀 提取的切片本身就是 [Batch, 30]，绝对不要做任何转置！
                    features_dict[fname] = valid_norm_seq[:, :, f_idx]
    
                # =================================================================
                # 🚀 [架构升维：事前合并期权字典 (Pre-Join)]
                # =================================================================
                live_options = {}
                live_options_5m = {} 
                
                source_snap_5m_for_payload = getattr(self, 'option_snapshot_5m', {})

                for sym in batch_symbols:
                    live_options[sym] = {
                        'buckets': source_opt_buckets.get(sym, []),
                        'contracts': getattr(self, 'latest_opt_contracts', {}).get(sym, [])
                    }
                    if USE_5M_OPTION_DATA:
                        snap_5m = source_snap_5m_for_payload.get(sym)

                        if snap_5m is not None:
                            live_options_5m[sym] = {
                                'buckets': snap_5m.tolist() if isinstance(snap_5m, np.ndarray) else snap_5m,
                                'contracts': getattr(self, 'latest_opt_contracts', {}).get(sym, [])
                            }
                
                # =================================================================
                # 🚀 [核心修复：实现 Alpha 标签的物理级准时达]
                # 强制所有在该分钟产生的推断结果（包含 1s 刷新）全部打上上一分钟的标签
                # 这样 10:01:00 到 10:01:59 产生的所有特征，在 DB 中都会对位到 10:00:00
                # =================================================================
                payload_ts = alpha_label_ts
              
                
                payload = {  
                    'ts': payload_ts, 
                    'log_ts': data_ts, 
                    'symbols': batch_symbols,
                    'stock_price': batch_prices,
                    'stock_id': np.array(batch_stock_ids),
                    'sector_id': np.zeros(len(batch_symbols)),
                    'fast_vol': np.array(batch_fast_vols),
                    'spy_roc_5min': np.array(batch_spy_rocs),
                    'qqq_roc_5min': np.array(batch_qqq_rocs),
                    'features_dict': features_dict,
                    'live_options': live_options,
                    'live_options_5m': live_options_5m, # 🚀 [修复] 补齐 Payload 字段
                    'is_new_minute': is_new_minute,  
                    
                    'cheat_call': cheat_call, 'cheat_put': cheat_put,
                    'cheat_call_bid': cheat_call_bid, 'cheat_call_ask': cheat_call_ask,
                    'cheat_put_bid': cheat_put_bid, 'cheat_put_ask': cheat_put_ask,
                    'cheat_call_iv': cheat_call_iv, 'cheat_put_iv': cheat_put_iv,
                }

                if return_payload:
                    return payload
 

                try:
                     self.r.xadd(self.redis_cfg['output_stream'], {'data': ser.pack(payload)}, maxlen=100)
                     payload_sent = True 
                except Exception as e:
                    logger.error(f"❌ Redis XADD Error: {e}")
                    
        # ✅ 严格在处理完本帧（包含 engine.compute 和 推流）后，再释放帧锁
        from config import IS_SIMULATED
        if IS_SIMULATED:
            self.r.set("sync:feature_calc_done", sync_ts)
            
            # 👇 [🔥 致命黑洞修复] 如果因为没有有效股票导致没发消息，代为解开下游 Orch 的锁！
            if not payload_sent:
                self.r.set("sync:orch_done", sync_ts)

        return None

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
                                        await self.process_market_data([payload]) # 🚀 [Unified] Use list for single payload
                                        if not current_batch_ts: current_batch_ts = payload.get('ts')
                                except Exception as e:
                                    logger.error(f"❌ Batch Unpack Error: {e}")
                                    
                            elif b'pickle' in data:
                                payload = ser.unpack(data[b'pickle'])
                                await self.process_market_data([payload]) # 🚀 [Unified] Use list for single payload
                                current_batch_ts = payload.get('ts')
                            
                            # [🔥 核心修复] 每一帧数据（或 Batch）处理完后，立刻触发计算，并回传正确的同步旗标
                            
                                # 🚀 [FCS-Trace] Trace NVDA at 10:00:00
                                last_row = self.history_1min["NVDA"].iloc[-1]
                                logger.info(f"📁 [TRACE-NVDA] OHLCV Parity | TS: {current_batch_ts} | Close: {last_row['close']:.4f} | Vol: {last_row['volume']:.0f}")
                            
                            # 🚀 [性能优化] 移除计算期间的 Redis 强制同步。
                            # 所有的 1m 结算已在 process_market_data 中完成了内存与 Redis 的写透同步。

                            await self.run_compute_cycle(ts_from_payload=current_batch_ts)
                            
                            self.r.xack(self.redis_cfg['raw_stream'], self.redis_cfg['group'], mid)
                
                from config import IS_SIMULATED
                if not IS_SIMULATED:
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

    def _sync_history_from_redis(self, symbol, limit=500):
        """🚀 [Redis 唯一标准] 从 Redis 物理回读历史序列，确保内存状态与系统账本绝对对齐"""
        try:
            key = f"BAR:1M:{symbol}"
            raw_data = self.r.hgetall(key)
            if not raw_data: return
            data_list = []
            for ts_bytes, payload_bytes in raw_data.items():
                ts = int(ts_bytes.decode('utf-8'))
                p = json.loads(payload_bytes.decode('utf-8'))
                data_list.append({
                    'timestamp': NY_TZ.localize(datetime.fromtimestamp(ts)),
                    'open': p['open'], 'high': p['high'], 'low': p['low'], 
                    'close': p['close'], 'volume': p['volume'], 'vwap': p.get('vwap', p['close'])
                })
            data_list.sort(key=lambda x: x['timestamp'])
            data_list = data_list[-limit:]
            df = pd.DataFrame(data_list)
            df.set_index('timestamp', inplace=True)
            self.history_1min[symbol] = df
        except Exception as e:
            logger.error(f"❌ [Redis Back-Read] Failed to sync {symbol} from Redis: {e}")

    def _sync_option_history_from_redis(self, symbol, limit=500):
        """🚀 [期权真相回读] 从 Redis 拉取期权历史，并物理重构 NumPy 快照"""
        try:
            key = f"BAR_OPT:1M:{symbol}"
            raw_data = self.r.hgetall(key)
            if not raw_data: return
            sorted_keys = sorted([int(k.decode('utf-8')) for k in raw_data.keys()])
            if not sorted_keys: return
            last_ts = sorted_keys[-1]
            p = json.loads(raw_data[str(last_ts).encode('utf-8')].decode('utf-8'))
            
            buckets = p.get('buckets', [])
            self.latest_opt_buckets[symbol] = buckets
            self.latest_opt_contracts[symbol] = p.get('contracts', [])
            
            # 🚀 [真相重构] 物理映射回推断引擎使用的 NumPy 快照
            if buckets:
                self._reconstruct_option_snapshot(symbol, buckets)
                
        except Exception as e:
            logger.error(f"❌ [Redis Opt-Back-Read] Failed to sync {symbol} option from Redis: {e}")

    def _reconstruct_option_snapshot(self, symbol, buckets_list):
        """将 JSON 格式的 buckets 物理重构为 6x12 的 NumPy 张量"""
        try:
            arr = np.array(buckets_list, dtype=np.float32)
            # 维度保护：强制对齐到 6x12
            if arr.shape[0] < 6:
                padding_row = np.zeros((6 - arr.shape[0], arr.shape[1]), dtype=np.float32)
                arr = np.vstack([arr, padding_row])
            if arr.shape[1] < 12:
                padding_col = np.zeros((arr.shape[0], 12 - arr.shape[1]), dtype=np.float32)
                arr = np.hstack([arr, padding_col])
                
            self.option_snapshot[symbol] = arr[:6, :12] # 严格锁死 6x12
        except Exception as e:
            logger.warning(f"⚠️ Snapshot reconstruction failed for {symbol}: {e}")

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
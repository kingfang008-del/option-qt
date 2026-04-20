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
from fcs_engine_adapter import build_feature_engine_adapter
from fcs_market_profile import build_market_profile
from fcs_realtime_pipeline import FCSRealtimePipeline
from fcs_persistence_handler import FCSPersistenceHandler
from fcs_warmup_handler import FCSWarmupHandler
from fcs_support_handler import FCSSupportHandler
from fcs_parity_snapshot_utils import (
    build_symbol_feature_parity_snapshot,
    save_feature_parity_snapshot,
    should_capture_feature_parity,
)

from config import (
    REDIS_CFG as _REDIS_BASE, NY_TZ,
    STREAM_FUSED_MARKET, STREAM_INFERENCE,
    get_feature_service_state_file,
    GROUP_FEATURE,
    LOG_DIR, USE_5M_OPTION_DATA, NON_TRADABLE_SYMBOLS,
    get_option_gate_profile,
     
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


def _safe_float_env(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return float(default)
    try:
        return float(raw)
    except Exception:
        return float(default)

# Feature service-specific Redis config (extends base)
REDIS_CFG = {
    **_REDIS_BASE,
    'raw_stream': STREAM_FUSED_MARKET,
    'option_stream': 'option_data_stream',
    'output_stream': STREAM_INFERENCE,
    'group': GROUP_FEATURE,
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
        if len(full_data) >= seq_len:
            data = full_data[-seq_len:]
        else: 
            data = full_data
            
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


class OptionMinuteAggregator:
    """
    将秒级期权 quote 流归约为“分钟唯一快照”。
    当前采用与分钟基准库最接近的语义：
    - 价格/盘口/size：取该分钟最后一个有效 quote
    - volume：统一强制定义为 bid_size + ask_size（与训练口径一致）
    """
    def __init__(self, rows: int = 6, cols: int = 12):
        self.rows = rows
        self.cols = cols
        self.state = {}

    def reset(self):
        self.state = {}

    def update(self, symbol: str, minute_dt, snapshot_arr: np.ndarray, contracts, update_ts: float = None):
        if snapshot_arr is None:
            return
        arr = np.asarray(snapshot_arr, dtype=np.float32)
        if arr.ndim != 2:
            return
        if arr.shape[0] < self.rows:
            arr = np.vstack([arr, np.zeros((self.rows - arr.shape[0], arr.shape[1]), dtype=np.float32)])
        if arr.shape[1] < self.cols:
            arr = np.hstack([arr, np.zeros((arr.shape[0], self.cols - arr.shape[1]), dtype=np.float32)])
        arr = arr[:self.rows, :self.cols].copy()

        # 统一分钟语义：volume 永远强制使用最后有效盘口 size 之和。
        size_sum = np.maximum(arr[:, 10], 0.0) + np.maximum(arr[:, 11], 0.0)
        arr[:, 6] = size_sum

        self.state[symbol] = {
            'minute_dt': minute_dt,
            'snapshot': arr,
            'contracts': list(contracts) if contracts else [],
            'update_ts': float(update_ts) if update_ts is not None else None,
        }

    def finalize(self, symbol: str, minute_dt):
        st = self.state.get(symbol)
        if not st:
            return None, [], None
        if st.get('minute_dt') != minute_dt:
            return None, [], None
        snap = np.asarray(st.get('snapshot'), dtype=np.float32).copy()
        size_sum = np.maximum(snap[:, 10], 0.0) + np.maximum(snap[:, 11], 0.0)
        snap[:, 6] = size_sum
        return snap, list(st.get('contracts', [])), st.get('update_ts')

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
            
        adapter_name = os.environ.get("FEATURE_ENGINE_ADAPTER", "equity_options_v1")
        self.engine_adapter = build_feature_engine_adapter(adapter_name, device=str(self.device))
        # 兼容旧路径：保留 self.engine 供历史方法调用。
        self.engine = self.engine_adapter.engine
        logger.info(f"🧩 Feature Engine Adapter: {adapter_name}")
        profile_name = os.environ.get("MARKET_PROFILE", "equity_us")
        warmup_len = int(os.environ.get("FCS_WARMUP_REQUIRED_LEN", "31"))
        self.market_profile = build_market_profile(
            profile_name,
            ny_tz=NY_TZ,
            warmup_required_len=warmup_len,
            non_tradable_symbols=NON_TRADABLE_SYMBOLS,
        )
        logger.info(f"🧭 Market Profile: {self.market_profile.name} | warmup_required_len={self.market_profile.warmup_required_len}")
        recalc_env = os.environ.get("RECALC_GREEKS", "1").strip().lower()
        self.recalc_greeks = recalc_env not in {"0", "false", "no", "off"}
        logger.info(f"🧮 Greeks Recalc Enabled: {self.recalc_greeks}")
        gate_profile = get_option_gate_profile()
        self.option_gate_min_pass = int(gate_profile["min_pass"])
        self.option_gate_max_fail = int(gate_profile["max_fail"])
        self.option_gate_grace_minutes = int(gate_profile["grace_minutes"])
        self.option_gate_min_iv = float(gate_profile["min_iv"])
        self.option_gate_require_frame_consistency = bool(gate_profile["require_frame_consistency"])
        logger.info(
            f"🛡️ Option Gate Config | min_pass={self.option_gate_min_pass} "
            f"| max_fail={self.option_gate_max_fail} | grace={self.option_gate_grace_minutes}m "
            f"| min_iv={self.option_gate_min_iv:.4f} | frame_consistency={self.option_gate_require_frame_consistency}"
        )
        
        self.normalizers = {
            s: RollingWindowNormalizer(
                self.all_feat_names, 
                self.feat_config_dict, 
                window=2000, 
                use_tanh=True
            ) for s in symbols
        }

        # 先初始化处理器，避免 reset_internal_memory() -> _load_service_state()
        # 期间访问 support_handler / warmup_handler 时出现属性缺失。
        self.realtime_pipeline = FCSRealtimePipeline(self)
        self.persistence_handler = FCSPersistenceHandler(self)
        self.warmup_handler = FCSWarmupHandler(self)
        self.support_handler = FCSSupportHandler(self)

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
        self.committed_history_1min = {s: pd.DataFrame() for s in self.symbols}
        self.committed_history_5min = {s: pd.DataFrame() for s in self.symbols}
        self.current_bars_5s = {s: [] for s in self.symbols}    # 🚀 [SDS 4.0] 升级为 Full-Payload 缓冲区
        self.minute_working_state = self.current_bars_5s
        self.option_snapshot = {s: np.zeros((6, 12), dtype=np.float32) for s in self.symbols}
        self.option_snapshot_5m = {s: np.zeros((6, 12), dtype=np.float32) for s in self.symbols}
        self.committed_option_snapshot = {s: np.zeros((6, 12), dtype=np.float32) for s in self.symbols}
        self.committed_option_contracts = {s: [] for s in self.symbols}
        self.committed_latest_opt_buckets = {s: np.zeros((6, 12), dtype=np.float32) for s in self.symbols}
        self.latest_prices = {s: 0.0 for s in self.symbols}
        self.last_tick_price = {s: 0.0 for s in self.symbols}
        self.last_stock_update_ts = {s: None for s in self.symbols}
        self.last_option_update_ts = {s: None for s in self.symbols}
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
        self.state_file = Path(get_feature_service_state_file())

        self.latest_opt_buckets = {s: np.zeros((6, 12), dtype=np.float32) for s in self.symbols}
        self.latest_opt_contracts = {s: [] for s in self.symbols}
        self.option_minute_agg = OptionMinuteAggregator()
        self.pending_finalization = {} # 🚀 [SDS 2.0] 结算挂起队列
        # 全局 minute commit watermark：只有超过分钟末 + grace 才允许统一结算上一分钟
        self.minute_commit_grace_sec = max(
            0.0,
            float(os.environ.get("FCS_MINUTE_COMMIT_GRACE_SEC", "1.0"))
        )
        self.committed_last_minute = None
        self.global_last_minute = None
        self.preaggregated_1m_mode = False
        self._preagg_mode_logged = False
        self.sym_vol_mean = {}
        self.sym_vol_var = {}
        self.sym_last_vol_price = {}
        self.cached_vol_z = {}
        self.option_gate_state = {
            s: {'pass_streak': 0, 'fail_streak': 0, 'ready': False, 'grace_until_ts': None} for s in self.symbols
        }
        self.option_frame_state = {
            s: {'minute_flags': {}, 'last_seq': None} for s in self.symbols
        }
        self._last_gate_metrics_minute_ts = None
        self.publish_frame_seq = 0
        self.runtime_payload_audit_enabled = os.environ.get("FCS_RUNTIME_AUDIT", "0").strip().lower() in {"1", "true", "yes", "on"}
        self.runtime_payload_audit_symbol = os.environ.get("FCS_RUNTIME_AUDIT_SYMBOL", "NVDA").strip() or "NVDA"
        self.runtime_payload_audit_ts = int(_safe_float_env("FCS_RUNTIME_AUDIT_TS", 1767373980.0))
        self.minute_write_audit_enabled = os.environ.get("FCS_MINUTE_WRITE_AUDIT", "1").strip().lower() in {"1", "true", "yes", "on"}
        self.minute_write_audit_symbol_limit = max(1, int(_safe_float_env("FCS_MINUTE_WRITE_AUDIT_SYMBOL_LIMIT", 5)))
        audit_dir_default = Path("/tmp/fcs_runtime_audit")
        self.runtime_payload_audit_dir = Path(os.environ.get("FCS_RUNTIME_AUDIT_DIR", str(audit_dir_default))).expanduser()
        self.runtime_payload_audit_written = set()
        self.feature_parity_symbol = os.environ.get("FCS_FEATURE_PARITY_SYMBOL", "NVDA").strip() or "NVDA"
        self.feature_parity_ts = int(_safe_float_env("FCS_FEATURE_PARITY_TS", 1767366000.0))
        parity_out_default = f"{self.feature_parity_symbol.lower()}_fcs_parity_{self.feature_parity_ts}.npz"
        self.feature_parity_output = os.environ.get("FCS_FEATURE_PARITY_OUTPUT", parity_out_default).strip() or parity_out_default
        if self.runtime_payload_audit_enabled:
            logger.info(
                f"🧪 Runtime payload audit enabled | symbol={self.runtime_payload_audit_symbol} "
                f"| ts={self.runtime_payload_audit_ts} | dir={self.runtime_payload_audit_dir}"
            )
        if self.minute_write_audit_enabled:
            logger.info(
                f"🧭 Minute write audit enabled | grace={self.minute_commit_grace_sec:.3f}s "
                f"| sample_symbols={self.minute_write_audit_symbol_limit}"
            )


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

    def _to_audit_matrix(self, snap):
        return self.support_handler.to_audit_matrix(snap)

    def _to_audit_row(self, snap, row_idx: int):
        return self.support_handler.to_audit_row(snap, row_idx)

    def _build_greeks_input_audit(
        self,
        *,
        symbol: str,
        buckets,
        contracts,
        stock_price: float,
        timestamp: float,
        bucket_id: int = 5,
    ):
        return self.support_handler.build_greeks_input_audit(
            symbol=symbol,
            buckets=buckets,
            contracts=contracts,
            stock_price=stock_price,
            timestamp=timestamp,
            bucket_id=bucket_id,
        )

    def _get_max_committable_label_ts(self, wall_ts: Optional[float] = None) -> int:
        wall_ts = float(time.time() if wall_ts is None else wall_ts)
        return int((wall_ts - float(self.minute_commit_grace_sec)) // 60) * 60 - 60

    def _get_symbol_last_raw_tick_ts(self, symbol: str) -> Optional[float]:
        bars = self.current_bars_5s.get(symbol, []) or []
        for item in reversed(bars):
            try:
                ts_val = float(item.get('ts', 0.0))
                if ts_val > 0:
                    return ts_val
            except Exception:
                continue
        return None

    def _get_symbol_last_history_ts(self, symbol: str) -> Optional[float]:
        df = self.history_1min.get(symbol)
        if df is None or df.empty:
            return None
        try:
            return float(df.index[-1].timestamp())
        except Exception:
            return None

    def _build_minute_write_audit(
        self,
        *,
        label_ts: float,
        data_ts: Optional[float] = None,
        wall_ts: Optional[float] = None,
        symbols: Optional[list] = None,
        extra: Optional[dict] = None,
    ) -> dict:
        label_ts = int(float(label_ts))
        wall_ts = float(time.time() if wall_ts is None else wall_ts)
        data_ts_val = None if data_ts is None else float(data_ts)
        max_committable_label_ts = self._get_max_committable_label_ts(wall_ts)
        earliest_commit_wall_ts = float(label_ts) + 60.0 + float(self.minute_commit_grace_sec)
        chosen_symbols = list(symbols or self.symbols)[:self.minute_write_audit_symbol_limit]
        symbol_state = {}
        for sym in chosen_symbols:
            symbol_state[sym] = {
                'last_raw_tick_ts': self._get_symbol_last_raw_tick_ts(sym),
                'last_history_ts': self._get_symbol_last_history_ts(sym),
                'last_stock_update_ts': self.last_stock_update_ts.get(sym),
                'last_option_update_ts': self.last_option_update_ts.get(sym),
            }
        payload = {
            'label_ts': label_ts,
            'label_ny': datetime.fromtimestamp(label_ts, NY_TZ).strftime('%Y-%m-%d %H:%M:%S'),
            'data_ts': data_ts_val,
            'data_ny': datetime.fromtimestamp(data_ts_val, NY_TZ).strftime('%Y-%m-%d %H:%M:%S') if data_ts_val is not None else None,
            'wall_ts': wall_ts,
            'wall_ny': datetime.fromtimestamp(wall_ts, NY_TZ).strftime('%Y-%m-%d %H:%M:%S'),
            'grace_sec': float(self.minute_commit_grace_sec),
            'earliest_commit_wall_ts': earliest_commit_wall_ts,
            'earliest_commit_wall_ny': datetime.fromtimestamp(earliest_commit_wall_ts, NY_TZ).strftime('%Y-%m-%d %H:%M:%S'),
            'max_committable_label_ts': max_committable_label_ts,
            'max_committable_label_ny': datetime.fromtimestamp(max_committable_label_ts, NY_TZ).strftime('%Y-%m-%d %H:%M:%S'),
            'is_early_vs_wall': bool(wall_ts + 1e-9 < earliest_commit_wall_ts),
            'is_early_vs_max_label': bool(label_ts > max_committable_label_ts),
            'symbols': chosen_symbols,
            'symbol_state': symbol_state,
        }
        if extra:
            payload.update(extra)
        return payload

    def _log_minute_write_audit(
        self,
        *,
        stage: str,
        label_ts: float,
        data_ts: Optional[float] = None,
        wall_ts: Optional[float] = None,
        symbols: Optional[list] = None,
        extra: Optional[dict] = None,
        level: str = "info",
        force: bool = False,
    ):
        if not force and not self.minute_write_audit_enabled:
            return
        audit = self._build_minute_write_audit(
            label_ts=label_ts,
            data_ts=data_ts,
            wall_ts=wall_ts,
            symbols=symbols,
            extra=extra,
        )
        log_fn = getattr(logger, level, logger.info)
        log_fn(f"🧭 [Minute-Write-Audit] stage={stage} | {json.dumps(audit, ensure_ascii=False, default=str)}")

    def _maybe_write_runtime_payload_audit(
        self,
        *,
        alpha_label_ts: float,
        data_ts: float,
        symbol: str,
        payload_stock_price: float,
        latest_stock_price: float,
        last_stock_update_ts: float,
        last_option_update_ts: float,
        source_option_snapshot,
        frozen_option_snapshot,
        payload_option_snapshot,
        latest_opt_buckets,
        frozen_latest_opt_buckets,
        contracts,
        pre_supplement_greeks_input=None,
        post_supplement_greeks_input=None,
        bucket_id: int = 5,
    ):
        return self.support_handler.maybe_write_runtime_payload_audit(
            alpha_label_ts=alpha_label_ts,
            data_ts=data_ts,
            symbol=symbol,
            payload_stock_price=payload_stock_price,
            latest_stock_price=latest_stock_price,
            last_stock_update_ts=last_stock_update_ts,
            last_option_update_ts=last_option_update_ts,
            source_option_snapshot=source_option_snapshot,
            frozen_option_snapshot=frozen_option_snapshot,
            payload_option_snapshot=payload_option_snapshot,
            latest_opt_buckets=latest_opt_buckets,
            frozen_latest_opt_buckets=frozen_latest_opt_buckets,
            contracts=contracts,
            pre_supplement_greeks_input=pre_supplement_greeks_input,
            post_supplement_greeks_input=post_supplement_greeks_input,
            bucket_id=bucket_id,
        )
 
    def _ensure_debug_tables(self, date_str):
        return self.support_handler.ensure_debug_tables(date_str)
     
    def _write_debug_batch(self, ts, date_str, fast_data_list, slow_data_list, source_ts=None):
        return self.support_handler.write_debug_batch(ts, date_str, fast_data_list, slow_data_list, source_ts=source_ts)

    # --- 持久化方法 ---
    def _save_service_state(self):
        return self.support_handler.save_service_state()

    def _load_service_state(self):
        return self.support_handler.load_service_state()

    def _warmup_from_history(self, replay_features=True):
        return self.warmup_handler.warmup_from_history(replay_features=replay_features)

    def _robust_backfill_and_warmup(self):
        return self.warmup_handler.robust_backfill_and_warmup()

    def _publish_warmup_status(self):
        return self.warmup_handler.publish_warmup_status()

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
        return await self.realtime_pipeline.process_market_data(
            batch=batch,
            current_replay_ts=current_replay_ts,
            return_payload=return_payload,
        )

    def _get_dynamic_atm_iv(self, snap, price):
        return self.support_handler.get_dynamic_atm_iv(snap, price)

    def _extract_semantic_atm_iv(self, buckets, contracts, spot_price):
        return self.support_handler.extract_semantic_atm_iv(buckets, contracts, spot_price)

    def _extract_tagged_atm_iv(self, buckets):
        return self.support_handler.extract_tagged_atm_iv(buckets)

    def _merge_option_snapshot_with_greeks(self, raw_buckets, enriched_buckets):
        return self.support_handler.merge_option_snapshot_with_greeks(raw_buckets, enriched_buckets)

    def _is_option_snapshot_complete(self, buckets, contracts=None, min_iv=None):
        return self.support_handler.is_option_snapshot_complete(buckets, contracts=contracts, min_iv=min_iv)

    def _update_option_frame_state(self, sym: str, payload: dict, ts: float):
        return self.support_handler.update_option_frame_state(sym, payload, ts)

    def _is_option_frame_consistent(self, sym: str, gate_minute_ts: Optional[float]) -> bool:
        return self.support_handler.is_option_frame_consistent(sym, gate_minute_ts)

    def _update_option_gate_state(
        self,
        sym: str,
        snapshot_ok: bool,
        frame_ok: bool,
        is_new_minute: bool,
        gate_minute_ts: Optional[float] = None
    ) -> bool:
        return self.support_handler.update_option_gate_state(
            sym=sym,
            snapshot_ok=snapshot_ok,
            frame_ok=frame_ok,
            is_new_minute=is_new_minute,
            gate_minute_ts=gate_minute_ts,
        )

    def _publish_option_gate_metrics(self, gate_minute_ts: float, gate_audit: Dict[str, dict]):
        return self.support_handler.publish_option_gate_metrics(gate_minute_ts, gate_audit)

    def _resolve_gate_status(
        self,
        sym: str,
        candidate_buckets,
        candidate_contracts,
        *,
        is_new_minute: bool,
        gate_minute_ts: float,
    ):
        """
        统一门控判定入口（偏函数式）：输入固定，输出固定，不在调用侧散落分支。
        """
        if is_new_minute:
            snapshot_ok = self._is_option_snapshot_complete(candidate_buckets, candidate_contracts)
            frame_ok = self._is_option_frame_consistent(sym, gate_minute_ts)
            allow = self._update_option_gate_state(
                sym,
                snapshot_ok=snapshot_ok,
                frame_ok=frame_ok,
                is_new_minute=True,
                gate_minute_ts=gate_minute_ts
            )
        else:
            # 秒级帧只复用上一分钟 gate 结果，未初始化时默认不放行。
            gate_state = self.option_gate_state.get(sym, {})
            ready = bool(gate_state.get('ready', False))
            snapshot_ok = ready
            frame_ok = True
            allow = ready
        gate_state = self.option_gate_state.get(sym, {})
        return snapshot_ok, frame_ok, bool(allow), gate_state

    def _iter_feature_sources(self, res_sym: dict):
        """
        以可组合的数据源形式暴露特征块，便于后续 map/filter 风格处理。
        """
        blocks = (
            ('fast_1m', self.fast_feat_names),
            ('slow_1m', self.slow_feat_names),
        )
        return [(ftype, names, res_sym.get(ftype)) for ftype, names in blocks if res_sym.get(ftype) is not None]

    def _fill_raw_vec_from_result(
        self,
        *,
        sym: str,
        res_sym: dict,
        raw_vec: np.ndarray,
        alpha_label_ts: float,
        is_new_minute: bool,
    ):
        """
        将最新特征块写入 raw_vec；函数式输入输出，避免调用侧堆叠条件。
        """
        target_history_map = self.committed_history_1min if is_new_minute else self.history_1min
        for ftype, names, tensor_blk in self._iter_feature_sources(res_sym):
            latest = tensor_blk[0, :, -1].cpu().numpy()
            for i, fname in enumerate(names):
                val = float(latest[i]) if i < len(latest) else np.nan
                if (not np.isfinite(val)) and ftype == 'slow_1m':
                    hist_df = target_history_map.get(sym)
                    if hist_df is not None and fname in hist_df.columns:
                        prev_series = pd.to_numeric(hist_df[fname], errors='coerce').dropna()
                        if not prev_series.empty:
                            val = float(prev_series.iloc[-1])
                if np.isfinite(val):
                    raw_vec[self.feat_name_to_idx[fname]] = val
                    if is_new_minute:
                        ts_logi = pd.Timestamp(alpha_label_ts, unit='s', tz=NY_TZ)
                        target_history_map[sym].loc[ts_logi, fname] = val
                        self.history_1min[sym].loc[ts_logi, fname] = val

    def _finalize_1min_bar(self, sym, dt, cleanup=True):
        return self.persistence_handler.finalize_1min_bar(sym, dt, cleanup=cleanup)

    def commit_ready_minutes(self, target_minute_dt):
        return self.persistence_handler.commit_ready_minutes(target_minute_dt)

    def finalization_barrier(self):
        return self.persistence_handler.finalization_barrier()
            
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
        non_trade_symbols = self.market_profile.get_non_tradable_set()
        ready_trading_symbols = [s for s in self.symbols if s not in non_trade_symbols and not self.history_1min[s].empty]
        
        if not ready_trading_symbols or len(self.history_1min[ready_trading_symbols[0]]) < 30:
            missing = [s for s in self.symbols if s not in non_trade_symbols and s not in ready_trading_symbols]
            if len(ready_trading_symbols) < 30:
                logger.info(f"⏳ [Warmup Audit] {len(ready_trading_symbols)} symbols ready, {len(missing)} missing: {missing[:5]}")
            return None

        # 2. 时序审计 (10s 汇报频率)
        data_ts = float(current_replay_ts) if current_replay_ts else target_time.timestamp()
        if int(sync_ts) % 10 == 0:
            logger.info(f"🕒 [Time-Sync] Tick:{target_time.strftime('%H:%M:%S')} | Alpha_Log_TS:{datetime.fromtimestamp(data_ts, NY_TZ).strftime('%H:%M:%S')} | Trading_Symbols:{len(ready_trading_symbols)}")

        # 3. 分钟边界判定（容错：不依赖“恰好 :00 秒”）
        # 实盘 tick 可能不会在整秒到达，如果只看 sync_ts % 60 == 0，
        # 会出现整分钟长期不触发 is_new_minute，导致 alpha_logs 断档。
        is_new_minute = False
        last_model_minute_ts = int(getattr(self, 'last_model_minute_ts', 0) or 0)
        if last_model_minute_ts == 0:
            # 启动首帧只建立锚点，避免冷启动立即“补算”未知分钟。
            self.last_model_minute_ts = current_minute_ts
            return None
        if current_minute_ts > last_model_minute_ts:
            is_new_minute = True
            self.last_model_minute_ts = current_minute_ts
            
        return alpha_label_ts, data_ts, is_new_minute, ready_trading_symbols, sync_ts
    
    def _step_engine_compute(self, data_ts, is_new_minute, alpha_label_ts):
        """[V8 Helper] 引擎计算层：执行底层指标运算与 Redis 同步"""
        # 是否触发重算（含 Greeks）：分钟翻页 或 首次 / cached 丢失时冷启动
        need_full_recompute = is_new_minute or getattr(self, 'cached_batch_raw', None) is None
        if need_full_recompute:
            # 🚀 [致命修复] 严防未来数据污染！
            # 必须使用被严格对齐在上一分钟末秒的 frozen_snapshot 作为 BSM 计算基底！
            history_1min_source = self.committed_history_1min if is_new_minute else self.history_1min
            history_5min_source = self.committed_history_5min if is_new_minute else getattr(self, 'history_5min', {})
            source_snap = getattr(self, 'committed_option_snapshot', self.option_snapshot) if is_new_minute else self.option_snapshot
            source_snap_5m = getattr(self, 'frozen_option_snapshot_5m', getattr(self, 'option_snapshot_5m', {})) if is_new_minute else getattr(self, 'option_snapshot_5m', {})
            source_contracts = getattr(self, 'committed_option_contracts', self.latest_opt_contracts) if is_new_minute else self.latest_opt_contracts
            
            sliced_snaps = {s: snap[:, :12].copy() for s, snap in source_snap.items()}
            sliced_snaps_5m = {s: (snap[:, :12].copy() if isinstance(snap, np.ndarray) else np.zeros((6,12))) for s, snap in source_snap_5m.items()}
            compute_ts = alpha_label_ts if is_new_minute else data_ts
            self.runtime_pre_greeks_input_audit = {}
            self.runtime_post_greeks_input_audit = {}
            if self.runtime_payload_audit_enabled:
                for s in self.symbols:
                    if s not in sliced_snaps:
                        continue
                    df_s = history_1min_source.get(s)
                    if df_s is None or df_s.empty:
                        continue
                    try:
                        stock_price_input = float(df_s.iloc[-1]['close'])
                    except Exception:
                        continue
                    self.runtime_pre_greeks_input_audit[s] = self._build_greeks_input_audit(
                        symbol=s,
                        buckets=sliced_snaps.get(s),
                        contracts=source_contracts.get(s, []),
                        stock_price=stock_price_input,
                        timestamp=compute_ts,
                        bucket_id=5,
                    )

            try:
                results_map = self.engine_adapter.compute_all_inputs(
                    history_1min=history_1min_source, history_5min=history_5min_source,
                    fast_feats=self.fast_feat_names, slow_feats=self.slow_feat_names,
                    option_snapshots=sliced_snaps,  
                    option_contracts=source_contracts,  # 🚀 传入严格冻结版的合约，确保 100% 对齐
                    option_snapshot_5m=sliced_snaps_5m, feat_resolutions=getattr(self, 'feat_resolutions', {}),
                    current_ts=compute_ts, recalc_greeks=self.recalc_greeks,
                    is_new_minute=need_full_recompute
                )
            except Exception as e:
                logger.error(f"Engine Compute Error: {e}", exc_info=True); return None

            run_mode = os.environ.get("RUN_MODE", "").strip().upper()
            dry_mode = (run_mode == "REALTIME_DRY")
            batch_raw, valid_mask = [], []
            gate_audit = {}
            for sym in self.symbols:
                raw_vec = np.zeros(len(self.all_feat_names), dtype=np.float32)
                is_valid = False
                if sym in results_map:
                    res_sym = results_map[sym]
                    candidate_buckets = res_sym.get('updated_buckets')
                    if candidate_buckets is None:
                        candidate_buckets = source_snap.get(sym)
                    candidate_contracts = source_contracts.get(sym, [])

                    gate_minute_ts = alpha_label_ts if is_new_minute else data_ts
                    snapshot_ok, frame_ok, is_valid, gate_state = self._resolve_gate_status(
                        sym,
                        candidate_buckets,
                        candidate_contracts,
                        is_new_minute=is_new_minute,
                        gate_minute_ts=gate_minute_ts
                    )
                    gate_audit[sym] = {
                        'snapshot_ok': bool(snapshot_ok),
                        'frame_ok': bool(frame_ok),
                        'ready': bool(gate_state.get('ready', False)),
                        'allow': bool(is_valid)
                    }

                    enriched = res_sym.get('updated_buckets')
                    if enriched is not None:
                        self.latest_opt_buckets[sym] = enriched
                        if self.runtime_payload_audit_enabled:
                            df_s = history_1min_source.get(sym)
                            if df_s is not None and not df_s.empty:
                                try:
                                    stock_price_input = float(df_s.iloc[-1]['close'])
                                    self.runtime_post_greeks_input_audit[sym] = self._build_greeks_input_audit(
                                        symbol=sym,
                                        buckets=enriched,
                                        contracts=source_contracts.get(sym, []),
                                        stock_price=stock_price_input,
                                        timestamp=compute_ts,
                                        bucket_id=5,
                                    )
                                except Exception:
                                    pass
                        
                        # 🚀 [核心补漏] 必须同步更新 Frozen 字典，这样 downstream 的 payload 才能吃到被补算了 Greeks 的版本！
                        if is_new_minute and hasattr(self, 'frozen_latest_opt_buckets'):
                            self.frozen_latest_opt_buckets[sym] = enriched
                        if is_new_minute and hasattr(self, 'frozen_option_snapshot'):
                            self.frozen_option_snapshot[sym] = np.asarray(enriched, dtype=np.float32).copy()
                        if is_new_minute and hasattr(self, 'committed_latest_opt_buckets'):
                            self.committed_latest_opt_buckets[sym] = np.asarray(enriched, dtype=np.float32).copy()


                        try:
                            atm_c_iv = float(enriched[2, 7]) if enriched.shape[0] > 2 else 0.0
                            atm_p_iv = float(enriched[0, 7]) if enriched.shape[0] > 0 else 0.0
                            
                            payload = {
                                'buckets': enriched.tolist() if hasattr(enriched, 'tolist') else enriched, 
                                'contracts': source_contracts.get(sym, []), # 🚀 使用严密对应的合约
                                'atm_c_iv': atm_c_iv,
                                'atm_p_iv': atm_p_iv
                            }
                            self.r.hset(f"BAR_OPT:1M:{sym}", str(int(alpha_label_ts)), json.dumps(payload))
                        except Exception as e:
                            logger.error(f"❌ [Redis Write Error] Failed to write optic bar for {sym}: {e}")
                    
                    # Dry 旁路时也必须填充有效特征，避免“空向量入模”。
                    if is_valid or dry_mode:
                        self._fill_raw_vec_from_result(
                            sym=sym,
                            res_sym=res_sym,
                            raw_vec=raw_vec,
                            alpha_label_ts=alpha_label_ts,
                            is_new_minute=is_new_minute,
                        )

                if is_new_minute and (is_valid or dry_mode):
                    row_ts = pd.Timestamp(alpha_label_ts, unit='s', tz=NY_TZ)
                    if row_ts in self.committed_history_1min[sym].index:
                        for fname in self.slow_feat_names:
                            if 'options_' in fname:
                                self.committed_history_1min[sym].at[row_ts, fname] = raw_vec[self.feat_name_to_idx[fname]]
                                self.history_1min[sym].at[row_ts, fname] = raw_vec[self.feat_name_to_idx[fname]]

                self._inject_temporal_derivatives(sym, raw_vec, is_new_minute=is_new_minute)
                batch_raw.append(raw_vec); valid_mask.append(is_valid)

            self.cached_results_map, self.cached_batch_raw, self.cached_valid_mask = results_map, batch_raw, valid_mask
            
            if is_new_minute:
                self._publish_option_gate_metrics(alpha_label_ts, gate_audit)
        else:
            results_map, batch_raw, valid_mask = self.cached_results_map, self.cached_batch_raw, self.cached_valid_mask

        return batch_raw, valid_mask, results_map

    def _apply_normalization_sequence(self, batch_raw, valid_mask, data_ts, is_new_minute):
        """[V8 Helper] 归一化层：执行 Z-Score，并通过 normalizer.get_sequence(30) 生成输入序列"""
        dt_ny = datetime.fromtimestamp(data_ts, NY_TZ)
        date_str = dt_ny.strftime('%Y%m%d')
        if not hasattr(self, 'created_debug_dates'): self.created_debug_dates = set()
        if date_str not in self.created_debug_dates: self._ensure_debug_tables(date_str); self.created_debug_dates.add(date_str)
        
        batch_norm = []
        is_rth = bool(self.market_profile.is_rth_minute(dt_ny))
        
        for b_idx, sym in enumerate(self.symbols):
            raw_vec = batch_raw[b_idx]
            is_valid = valid_mask[b_idx]
            norm = self.normalizers[sym]
            if is_valid and is_rth and is_new_minute:
                norm.process_frame(raw_vec)
            else:
                norm.normalize_only(raw_vec)
            batch_norm.append(norm.get_sequence(30))

        # [B, 30, F]
        return np.stack(batch_norm, axis=0)
    
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

        # 🚀 [Greeks 写盘修复] payload 必须优先使用补算后的 latest_opt_buckets，
        # 否则会把 raw snapshot（Greeks 常为 0）写到 option_snapshots_1m。
        source_opt_buckets = getattr(self, 'committed_latest_opt_buckets', self.latest_opt_buckets) if is_new_minute else self.latest_opt_buckets
        source_snap_for_payload = getattr(self, 'committed_option_snapshot', self.option_snapshot) if is_new_minute else self.option_snapshot
        history_1min_source = self.committed_history_1min if is_new_minute else self.history_1min

        run_mode = os.environ.get("RUN_MODE", "").strip().upper()
        dry_mode = (run_mode == "REALTIME_DRY")
        for b_idx, sym in enumerate(self.symbols):
            if sym not in ready_symbols or sym not in results_map:
                continue
                
            # 过滤无效数据
            if not valid_mask[b_idx]:
                if not dry_mode:
                    continue
                if getattr(self, "_dry_valid_bypass_log_count", 0) < 50:
                    logger.warning(
                        f"🧪 [FCS-Dry-Valid-Bypass] keep symbol despite invalid gate | sym={sym} | ts={int(alpha_label_ts)}"
                    )
                    self._dry_valid_bypass_log_count = getattr(self, "_dry_valid_bypass_log_count", 0) + 1
                
            valid_b_indices.append(b_idx)
            batch_symbols.append(sym)
            
            # 🚀 [成交量修复] 同时提取 close 和 volume，供下游写入 Redis 账本
            if is_new_minute:
                try: 
                    target_time = dt_ny_payload.replace(second=0, microsecond=0)
                    p_close = history_1min_source[sym].at[target_time, 'close']
                    p_vol = history_1min_source[sym].at[target_time, 'volume']
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
            
            # 提取期权快照：raw 为底座，Greeks/IV 用补算结果覆盖
            snap = self._merge_option_snapshot_with_greeks(
                source_snap_for_payload.get(sym),
                source_opt_buckets.get(sym),
            )

            if snap is not None and isinstance(snap, np.ndarray) and snap.shape[0] >= 6:
                c_iv, p_iv = snap[2, 7], snap[0, 7]
                cheat_call.append(snap[2, 0])
                cheat_call_iv.append(c_iv)
                cheat_call_bid.append(snap[2, 8])
                cheat_call_ask.append(snap[2, 9])
                cheat_put.append(snap[0, 0])
                cheat_put_iv.append(p_iv)
                cheat_put_bid.append(snap[0, 8])
                cheat_put_ask.append(snap[0, 9])
            else:
                for lst in [cheat_call, cheat_call_iv, cheat_call_bid, cheat_call_ask, cheat_put, cheat_put_iv, cheat_put_bid, cheat_put_ask]:
                    lst.append(0.0)

        if not batch_symbols:
            if getattr(self, "_empty_payload_log_count", 0) < 30:
                valid_cnt = int(np.count_nonzero(valid_mask)) if valid_mask is not None else 0
                logger.warning(
                    f"⚠️ [FCS-Payload-Empty] no symbols passed to inference payload "
                    f"| ready={len(ready_symbols)} | valid={valid_cnt} | is_new_minute={bool(is_new_minute)} | ts={int(alpha_label_ts)}"
                )
                self._empty_payload_log_count = getattr(self, "_empty_payload_log_count", 0) + 1
            return None

        valid_b_indices = np.array(valid_b_indices)
        valid_norm_seq = norm_seq_30[valid_b_indices] 
        
        # 🚀 [致命修复：时空倒置与维度坍塌] 
        # 直接提取 [Batch, Time] 切片，绝对不能使用 .transpose(1, 0)
        features_dict = {
            fname: valid_norm_seq[:, :, self.feat_name_to_idx[fname]] 
            for fname in self.slow_feat_names
        }

        # 🚀 [架构升维：事前合并期权字典 (Pre-Join)]
        live_options = {}
        live_options_5m = {} 
 
        source_snap_5m_for_payload = getattr(self, 'frozen_option_snapshot_5m', getattr(self, 'option_snapshot_5m', {})) if is_new_minute else getattr(self, 'option_snapshot_5m', {})
        source_contracts = getattr(self, 'committed_option_contracts', self.latest_opt_contracts) if is_new_minute else self.latest_opt_contracts # 🚀

        for sym in batch_symbols:
            # live_options：将 raw 分钟快照与 Greeks 补算结果融合后再下发
            snap_live = self._merge_option_snapshot_with_greeks(
                source_snap_for_payload.get(sym),
                source_opt_buckets.get(sym),
            )
            contracts_live = source_contracts.get(sym, [])
            greeks_ready = self._is_option_snapshot_complete(snap_live, contracts_live)
            live_options[sym] = {
                'buckets': snap_live.tolist() if isinstance(snap_live, np.ndarray) else snap_live,
                'contracts': contracts_live, # 🚀 强绑定
                # 下游持久化据此决定是否允许写入 option_snapshots_1m
                'greeks_ready': bool(greeks_ready),
            }
            if USE_5M_OPTION_DATA:
                snap_5m = source_snap_5m_for_payload.get(sym)
                if snap_5m is not None:
                    live_options_5m[sym] = {
                        'buckets': snap_5m.tolist() if isinstance(snap_5m, np.ndarray) else snap_5m,
                        'contracts': getattr(self, 'latest_opt_contracts', {}).get(sym, []) # 5m暂保持宽松要求
                    }
        
        # 🚀 [Warmup 对齐修复 v2]
        # 仅统计 RTH(09:30-16:00) 的分钟历史，并对 alpha_label_ts 的“回退 1 分钟标签”做补偿。
        # 目标行为：
        # - 无预热：9:30 + 30 根后，首条 alpha 标签应为 10:00；
        # - 有预热：开盘即可放行（由历史条数天然满足门槛）。
        dt_label_ny = datetime.fromtimestamp(alpha_label_ts, NY_TZ)
        label_floor = dt_label_ny.replace(second=0, microsecond=0)

        hist_lens = []
        total_hist_lens = []
        for s in batch_symbols:
            df_hist = history_1min_source.get(s, pd.DataFrame())
            if df_hist is None or df_hist.empty:
                hist_lens.append(0)
                total_hist_lens.append(0)
                continue
            try:
                total_hist_lens.append(int(len(df_hist)))
                idx = df_hist.index
                if getattr(idx, "tz", None) is None:
                    idx = idx.tz_localize(NY_TZ)
                hist_lens.append(int(self.market_profile.count_effective_history(idx, label_floor)))
            except Exception:
                hist_lens.append(0)
                total_hist_lens.append(0)

        real_history_len = int(min(hist_lens)) if hist_lens else 0
        total_history_len = int(min(total_hist_lens)) if total_hist_lens else 0
        # 为保证秒级/分钟级发球机起跑时点一致（无预热时均从 10:00 开始），
        # 两种模式统一采用 31 根门槛。
        warmup_required_len = int(getattr(self.market_profile, "warmup_required_len", 31))
        # 额外保留归一化样本长度，便于诊断但不用于全局放行门控。
        valid_counts = [int(self.normalizers[s].count) for s in batch_symbols if s in self.normalizers]
        real_norm_history_len = int(min(valid_counts)) if valid_counts else 0
        has_cross_day_warmup = bool(
            real_history_len > 0
            and total_history_len >= warmup_required_len
            and real_norm_history_len >= warmup_required_len
        )
        is_warmed_up = bool(real_history_len >= warmup_required_len or has_cross_day_warmup)

        vol_z_dict = self._compute_payload_vol_z(
            batch_symbols,
            batch_prices,
            batch_fast_vols,
            is_new_minute=is_new_minute
        )

        payload = {
            'ts': alpha_label_ts, 
            'log_ts': data_ts, 
            'source_ts': data_ts,
            'frame_id': str(int(data_ts)),
            'symbols': batch_symbols,
            'stock_price': batch_prices,
            'stock_volume': np.array(batch_volumes), # 🚀 修复丢失的成交量
            'stock_id': np.array(batch_stock_ids),
            'sector_id': np.zeros(len(batch_symbols)),
            'fast_vol': np.array(batch_fast_vols),
            'vol_z_dict': vol_z_dict,
            'spy_roc_5min': np.array(batch_spy_rocs),
            'qqq_roc_5min': np.array(batch_qqq_rocs),
            'features_dict': features_dict,
            'live_options': live_options,
            'live_options_5m': live_options_5m,
            'is_new_minute': is_new_minute,  
            'is_warmed_up': is_warmed_up,
            'real_history_len': real_history_len,
            'total_history_len': total_history_len,
            'real_norm_history_len': real_norm_history_len,
            'warmup_required_len': int(warmup_required_len),
            'has_cross_day_warmup': has_cross_day_warmup,
            'cheat_call': cheat_call, 'cheat_put': cheat_put,
            'cheat_call_bid': cheat_call_bid, 'cheat_call_ask': cheat_call_ask,
            'cheat_put_bid': cheat_put_bid, 'cheat_put_ask': cheat_put_ask,
            'cheat_call_iv': cheat_call_iv, 'cheat_put_iv': cheat_put_iv,
        }

        for idx_sym, sym in enumerate(batch_symbols):
            payload_snap = live_options.get(sym, {}).get('buckets', [])
            self._maybe_write_runtime_payload_audit(
                alpha_label_ts=alpha_label_ts,
                data_ts=data_ts,
                symbol=sym,
                payload_stock_price=batch_prices[idx_sym],
                latest_stock_price=self.latest_prices.get(sym, 0.0),
                last_stock_update_ts=self.last_stock_update_ts.get(sym),
                last_option_update_ts=self.last_option_update_ts.get(sym),
                source_option_snapshot=self.option_snapshot.get(sym),
                frozen_option_snapshot=getattr(self, 'frozen_option_snapshot', {}).get(sym),
                payload_option_snapshot=payload_snap,
                latest_opt_buckets=self.latest_opt_buckets.get(sym),
                frozen_latest_opt_buckets=getattr(self, 'frozen_latest_opt_buckets', {}).get(sym),
                contracts=source_contracts.get(sym, []),
                pre_supplement_greeks_input=getattr(self, 'runtime_pre_greeks_input_audit', {}).get(sym),
                post_supplement_greeks_input=getattr(self, 'runtime_post_greeks_input_audit', {}).get(sym),
                bucket_id=5,
            )

        if should_capture_feature_parity(
            batch_symbols=batch_symbols,
            alpha_label_ts=alpha_label_ts,
            target_symbol=self.feature_parity_symbol,
            target_ts=self.feature_parity_ts,
        ):
            try:
                symbol = self.feature_parity_symbol
                symbol_idx = batch_symbols.index(symbol)
                raw_symbol_idx = self.symbols.index(symbol)
                snapshot = build_symbol_feature_parity_snapshot(
                    symbol=symbol,
                    symbol_idx=symbol_idx,
                    features_dict=features_dict,
                    history_1min=self.history_1min.get(symbol, pd.DataFrame()),
                    valid_norm_seq=valid_norm_seq,
                    feat_name_to_idx=self.feat_name_to_idx,
                    raw_mat=raw_mat,
                    raw_symbol_idx=raw_symbol_idx,
                    normalizer=self.normalizers.get(symbol),
                    batch_price=batch_prices[symbol_idx],
                    batch_fast_vol=batch_fast_vols[symbol_idx],
                    cheat_call_iv=cheat_call_iv[symbol_idx],
                    cheat_put_iv=cheat_put_iv[symbol_idx],
                    source_opt_buckets=source_opt_buckets,
                    source_snap_for_payload=source_snap_for_payload,
                    frozen_option_snapshot=getattr(self, "frozen_option_snapshot", {}).get(symbol),
                    frozen_latest_opt_buckets=getattr(self, "frozen_latest_opt_buckets", {}).get(symbol),
                    valid_mask_value=bool(valid_mask[raw_symbol_idx]),
                    real_history_len=int(real_history_len),
                    total_history_len=int(total_history_len),
                    real_norm_history_len=int(real_norm_history_len),
                    has_cross_day_warmup=bool(has_cross_day_warmup),
                    alpha_label_ts=alpha_label_ts,
                )
                out_path = save_feature_parity_snapshot(self.feature_parity_output, snapshot)
                logger.info(f"💾 [FCS_TRACE] Saved feature parity snapshot to {out_path}")
            except Exception as trace_e:
                logger.warning(f"⚠️ [FCS_TRACE] Failed to save feature parity snapshot: {trace_e}")
        
        return payload

    def _compute_payload_vol_z(self, symbols: list, prices: list, raw_vols: list, is_new_minute: bool) -> Dict[str, float]:
        """在特征服务内生成分钟级 vol_z 事实，供下游直接消费。"""
        import math

        vol_z_dict = {}
        use_price_proxy_env = os.environ.get('VOL_Z_USE_PRICE_PROXY', '1').strip().lower()
        use_price_proxy = use_price_proxy_env not in {'0', 'false', 'no', 'off'}

        for idx, sym in enumerate(symbols):
            if not is_new_minute:
                vol_z_dict[sym] = float(self.cached_vol_z.get(sym, 0.0))
                continue

            px = float(prices[idx]) if idx < len(prices) else 0.0
            r_v = float(raw_vols[idx]) if idx < len(raw_vols) else 0.0

            if use_price_proxy and px > 0:
                prev_px = float(self.sym_last_vol_price.get(sym, 0.0))
                if prev_px > 0:
                    r_v = abs((px - prev_px) / prev_px) * 30.0
                else:
                    r_v = float(self.sym_vol_mean.get(sym, 0.01))
                self.sym_last_vol_price[sym] = px

            if abs(r_v) < 1e-9:
                r_v = float(self.sym_vol_mean.get(sym, 0.01))

            if sym not in self.sym_vol_mean:
                self.sym_vol_mean[sym] = r_v
                self.sym_vol_var[sym] = 0.1106 ** 2

            diff = r_v - float(self.sym_vol_mean[sym])
            vol_ewma = 2.0 / (15 + 1) if diff > 0 else 2.0 / (60 + 1)
            self.sym_vol_mean[sym] += vol_ewma * diff
            self.sym_vol_var[sym] = (1 - vol_ewma) * float(self.sym_vol_var[sym]) + vol_ewma * (diff ** 2)

            v_std = math.sqrt(self.sym_vol_var[sym]) if self.sym_vol_var[sym] > 1e-9 else 1e-4
            vz = (r_v - float(self.sym_vol_mean[sym])) / (v_std + 1e-6)
            if abs(vz) > 10.0:
                vz = 0.0

            clipped_vz = max(-5.0, min(5.0, float(vz)))
            self.cached_vol_z[sym] = clipped_vz
            vol_z_dict[sym] = clipped_vz

        return vol_z_dict

    def _atomic_commit_minute_payload(self, payload: dict) -> bool:
        return self.persistence_handler.atomic_commit_minute_payload(payload)

    def _set_feature_sync_ack(self, ts_val: Optional[float], frame_id: Optional[str] = None):
        return self.persistence_handler.set_feature_sync_ack(ts_val, frame_id=frame_id)
    
    async def run_compute_cycle(self, ts_from_payload=None, return_payload=False):
        return await self.realtime_pipeline.run_compute_cycle(
            ts_from_payload=ts_from_payload,
            return_payload=return_payload,
        )

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
        last_stats_log = time.time()
        stats = {'msgs': 0, 'acks': 0, 'compute': 0}
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
                    for stream_key, msgs in resp:
                        stream_name = (
                            stream_key.decode("utf-8")
                            if isinstance(stream_key, (bytes, bytearray))
                            else str(stream_key)
                        )
                        stats['msgs'] += len(msgs)
                        for mid, data in msgs:
                            current_batch_ts = None
                            try:
                                def _coerce_batch_ts(raw_ts):
                                    try:
                                        return float(self.realtime_pipeline._coerce_payload_ts(raw_ts))
                                    except Exception:
                                        return None

                                def _fallback_replay_ts():
                                    try:
                                        replay_ts_raw = self.r.get("replay:current_ts")
                                        if replay_ts_raw is None:
                                            return None
                                        return float(replay_ts_raw)
                                    except Exception:
                                        return None

                                msg_id_str = (
                                    mid.decode("utf-8")
                                    if isinstance(mid, (bytes, bytearray))
                                    else str(mid)
                                )
                                if b'batch' in data:
                                    try:
                                        batch = ser.unpack(data[b'batch'])
                                        for payload in batch:
                                            if isinstance(payload, dict):
                                                payload['_fcs_stream_name'] = stream_name
                                                payload['_fcs_stream_msg_id'] = msg_id_str
                                                payload['_fcs_recv_wall_ts'] = float(time.time())
                                            await self.process_market_data([payload]) # 🚀 [Unified] Use list for single payload
                                            p_ts = _coerce_batch_ts(payload.get('ts') if isinstance(payload, dict) else None)
                                            if p_ts is not None:
                                                current_batch_ts = p_ts if current_batch_ts is None else max(float(current_batch_ts), p_ts)
                                    except Exception as e:
                                        logger.error(f"❌ Batch Unpack Error: {e}")
                                        
                                elif b'pickle' in data:
                                    payload = ser.unpack(data[b'pickle'])
                                    if isinstance(payload, dict):
                                        payload['_fcs_stream_name'] = stream_name
                                        payload['_fcs_stream_msg_id'] = msg_id_str
                                        payload['_fcs_recv_wall_ts'] = float(time.time())
                                    await self.process_market_data([payload]) # 🚀 [Unified] Use list for single payload
                                    p_ts = _coerce_batch_ts(payload.get('ts') if isinstance(payload, dict) else None)
                                    if p_ts is not None:
                                        current_batch_ts = p_ts
                                elif b'data' in data:
                                    payload = ser.unpack(data[b'data'])
                                    if isinstance(payload, dict):
                                        payload['_fcs_stream_name'] = stream_name
                                        payload['_fcs_stream_msg_id'] = msg_id_str
                                        payload['_fcs_recv_wall_ts'] = float(time.time())
                                    await self.process_market_data([payload])
                                    p_ts = _coerce_batch_ts(payload.get('ts') if isinstance(payload, dict) else None)
                                    if p_ts is not None:
                                        current_batch_ts = p_ts

                                if current_batch_ts is None:
                                    fb_ts = _fallback_replay_ts()
                                    if fb_ts is not None:
                                        current_batch_ts = fb_ts
                                        now_wall = time.time()
                                        if now_wall - getattr(self, "_last_batch_ts_fallback_log_at", 0.0) >= 10.0:
                                            logger.warning(
                                                "⚠️ [FCS-BatchTS-Fallback] message has no coercible payload ts; "
                                                "use replay:current_ts=%.3f | stream=%s msg_id=%s",
                                                fb_ts,
                                                stream_name,
                                                msg_id_str,
                                            )
                                            self._last_batch_ts_fallback_log_at = now_wall

                                # [🔥 核心修复] 每一帧数据（或 Batch）处理完后，立刻触发计算，并回传正确的同步旗标
                                if current_batch_ts is not None:
                                    # 🚀 [FCS-Trace] Trace NVDA at minute boundary
                                    try:
                                        if "NVDA" in self.history_1min and not self.history_1min["NVDA"].empty:
                                            last_row = self.history_1min["NVDA"].iloc[-1]
                                            logger.info(f"📁 [TRACE-NVDA] OHLCV Parity | TS: {current_batch_ts} | Close: {last_row['close']:.4f} | Vol: {last_row['volume']:.0f}")
                                    except Exception:
                                        pass

                                    # 🚀 [性能优化] 移除计算期间的 Redis 强制同步。
                                    # 所有的 1m 结算已在 process_market_data 中完成了内存与 Redis 的写透同步。
                                    try:
                                        compute_payload = await self.run_compute_cycle(ts_from_payload=current_batch_ts, return_payload=True)
                                    except Exception as ce:
                                        logger.error(
                                            f"❌ [FCS-Compute-Crash] ts={current_batch_ts} stream={stream_name} msg_id={msg_id_str} err={ce}",
                                            exc_info=True,
                                        )
                                        # 防止 compute 异常导致发球机永远等待 sync:feature_calc_done
                                        self._set_feature_sync_ack(float(current_batch_ts), frame_id=str(int(float(current_batch_ts))))
                                        continue
                                    stats['compute'] += 1

                                    # 🚀 [Persistence-Check Log] 帮助用户排查为何不写库
                                    if compute_payload:
                                        p_min = compute_payload.get('is_new_minute', False)
                                        p_warm = compute_payload.get('is_warmed_up', False)
                                        # 仅在整分或者每 30s 打印一次状态，避免日志爆炸
                                        if p_min or (int(time.time()) % 30 == 0):
                                            logger.info(f"🔍 [Cycle-Status] ts={current_batch_ts} | is_new_minute={p_min} | is_warmed_up={p_warm} | Ready_Syms={len(compute_payload.get('symbols', []))}")

                                        # 🚀 [Parity Fix] 恢复丢失的 PostgreSQL Debug 表持久化写入
                                        # [优化] 只要是分钟边界就尝试写入特征，不再等待 30 根线的 Alpha 预热
                                        if p_min:
                                            try:
                                                ts_val = compute_payload.get('ts')
                                                self._log_minute_write_audit(
                                                    stage="debug_slow:prepare",
                                                    label_ts=float(ts_val or 0.0),
                                                    data_ts=float(compute_payload.get('log_ts', current_batch_ts) or current_batch_ts or 0.0),
                                                    wall_ts=time.time(),
                                                    symbols=compute_payload.get('symbols', []),
                                                    extra={
                                                        'ready_symbols': len(compute_payload.get('symbols', [])),
                                                        'is_warmed_up': bool(compute_payload.get('is_warmed_up', False)),
                                                    },
                                                )
                                                run_mode = os.environ.get("RUN_MODE", "").strip().upper()
                                                commit_grace_sec = float(getattr(self, "minute_commit_grace_sec", 1.0) or 0.0)
                                                if run_mode in {"REALTIME", "REALTIME_DRY"}:
                                                    max_committable_label_ts = int((time.time() - commit_grace_sec) // 60) * 60 - 60
                                                    if int(float(ts_val or 0.0)) > max_committable_label_ts:
                                                        self._log_minute_write_audit(
                                                            stage="debug_slow:reject_early",
                                                            label_ts=float(ts_val or 0.0),
                                                            data_ts=float(compute_payload.get('log_ts', current_batch_ts) or current_batch_ts or 0.0),
                                                            wall_ts=time.time(),
                                                            symbols=compute_payload.get('symbols', []),
                                                            extra={
                                                                'ready_symbols': len(compute_payload.get('symbols', [])),
                                                                'is_warmed_up': bool(compute_payload.get('is_warmed_up', False)),
                                                            },
                                                            level="error",
                                                            force=True,
                                                        )
                                                        logger.error(
                                                            f"🛑 [DebugSlow-Guard] skip early debug_slow write | ts={int(float(ts_val or 0.0))} "
                                                            f"> max_committable={max_committable_label_ts}"
                                                        )
                                                        continue
                                                date_str = datetime.fromtimestamp(ts_val, NY_TZ).strftime('%Y%m%d')
                                                
                                                # 准备 slow_data_list
                                                batch_syms = compute_payload.get('symbols', [])
                                                feats_dict = compute_payload.get('features_dict', {})
                                                
                                                slow_data_list = []
                                                for i, sym in enumerate(batch_syms):
                                                    # 提取该 symbol 在当前分钟 (索引 -1) 的所有 slow 特征值
                                                    vals = []
                                                    for fn in self.slow_feat_names:
                                                        f_tensor = feats_dict.get(fn)
                                                        if f_tensor is not None and i < f_tensor.shape[0]:
                                                            vals.append(float(f_tensor[i, -1]))
                                                        else:
                                                            vals.append(0.0)
                                                    slow_data_list.append((sym, vals))
                                                
                                                if slow_data_list:
                                                    sample_sym, sample_vals = slow_data_list[0]
                                                    logger.info(f"📊 [Persistence-Prep] ts={ts_val} | Symbols={len(slow_data_list)} | Sample={sample_sym} | First 3 Feats={sample_vals[:3]}")
                                                    self._write_debug_batch(
                                                        ts_val,
                                                        date_str,
                                                        fast_data_list=[],
                                                        slow_data_list=slow_data_list,
                                                        source_ts=float(compute_payload.get('log_ts', current_batch_ts) or current_batch_ts or ts_val),
                                                    )
                                                else:
                                                    logger.warning(f"⚠️ [Persistence-Skip] slow_data_list is empty for ts={ts_val}")
                                            except Exception as pe:
                                                logger.error(f"❌ Persistence Error in loop: {pe}")
                            finally:
                                self.r.xack(self.redis_cfg['raw_stream'], self.redis_cfg['group'], mid)
                                stats['acks'] += 1
                
                from config import IS_SIMULATED
                if not IS_SIMULATED:
                    await asyncio.sleep(0.001)

                if time.time() - last_stats_log >= 60:
                    logger.info(
                        f"📊 [FCS-Stats] 60s msgs={stats['msgs']} acked={stats['acks']} compute={stats['compute']}"
                    )
                    stats = {'msgs': 0, 'acks': 0, 'compute': 0}
                    last_stats_log = time.time()
                
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
                    # Rebuild directly in New York timezone. Localizing a naive
                    # datetime.fromtimestamp(ts) will reinterpret local wall time
                    # as New York time and create ghost bars such as 22:30.
                    'timestamp': datetime.fromtimestamp(ts, NY_TZ),
                    'open': p['open'], 'high': p['high'], 'low': p['low'], 
                    'close': p['close'], 'volume': p['volume'], 'vwap': p.get('vwap', p['close'])
                })
            data_list.sort(key=lambda x: x['timestamp'])
            data_list = data_list[-limit:]
            df = pd.DataFrame(data_list)
            df.set_index('timestamp', inplace=True)
            self.history_1min[symbol] = df
            self.committed_history_1min[symbol] = df.copy()
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
            self.committed_latest_opt_buckets[symbol] = np.array(buckets, dtype=np.float32) if buckets else np.zeros((6, 12), dtype=np.float32)
            self.committed_option_contracts[symbol] = list(p.get('contracts', []))
            
            # 🚀 [真相重构] 物理映射回推断引擎使用的 NumPy 快照
            if buckets:
                self._reconstruct_option_snapshot(symbol, buckets)
                self.committed_option_snapshot[symbol] = self.option_snapshot[symbol].copy()
                
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

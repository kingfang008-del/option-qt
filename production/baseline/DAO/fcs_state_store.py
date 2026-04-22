"""
fcs_state_store.py
==================

FCS 状态持久化后端 (PostgreSQL 主存 + 本地 pkl 二级缓存)

替代原 pkl-only 方案:
    production/baseline/DAO/fcs_support_handler.py :: save_service_state / load_service_state

设计目标:
  1. 与 OMS 的 symbol_state 对齐, 统一走 PG, 运维一套;
  2. 支持多实例 / 容器化 / 蓝绿部署, 本地磁盘不再是硬依赖;
  3. 通过 namespace 隔离 realtime_live / realtime_paper / realtime_dry / livereplay;
  4. 带 schema_version + feature/symbols hash 校验, 不兼容时优雅降级到 Deep Warmup;
  5. payload 是 per-symbol 的 blob, 方便后续按 symbol 并发 upsert 和差异更新.

表结构:
    CREATE TABLE fcs_state_snapshot (
        namespace       TEXT NOT NULL,
        symbol          TEXT NOT NULL,    -- 正常为 ticker; '_META_' 存命名空间级元数据
        schema_version  INTEGER NOT NULL,
        updated_at      DOUBLE PRECISION NOT NULL,  -- epoch seconds
        payload         BYTEA NOT NULL,             -- pickle + zlib
        PRIMARY KEY (namespace, symbol)
    );

每个 symbol row 的 payload (pickle.dumps 后 zlib.compress):
    {
        'v': SCHEMA_VERSION,
        'ts': save_ts,
        'normalizer':         norm.get_state(),
        'history_1min':       pd.DataFrame,
        'history_5min':       pd.DataFrame,
        'option_snapshot':    np.ndarray (6, 12),
        'option_snapshot_5m': np.ndarray (6, 12),
        'last_cum_volume':    np.ndarray (6,),
        'last_cum_volume_5m': np.ndarray (6,),
        'warmup_needed':      bool,
        'warmup_needed_5m':   bool,
    }

_META_ 行 payload:
    {
        'v': SCHEMA_VERSION,
        'ts': save_ts,
        'feature_names_hash': hash over all_feat_names (str),
        'symbols_hash':       hash over symbols (str),
        'symbol_count':       int,
        'run_mode':           str,
    }
"""

from __future__ import annotations

import hashlib
import logging
import os
import pickle
import threading
import time
import zlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extras

logger = logging.getLogger("FeatService.StateStore")


SCHEMA_VERSION = 1
META_SYMBOL = "_META_"


# ------------------------------------------------------------
# 序列化工具 (pickle + zlib, 对 deque/ndarray/DataFrame 都适用)
# ------------------------------------------------------------
def _encode(obj) -> bytes:
    try:
        raw = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
        return zlib.compress(raw, level=3)
    except Exception as e:
        logger.error(f"[StateStore] encode failed: {e}")
        raise


def _decode(blob: bytes):
    try:
        raw = zlib.decompress(blob)
        return pickle.loads(raw)
    except Exception as e:
        logger.error(f"[StateStore] decode failed: {e}")
        raise


def _hash_list(items) -> str:
    if not items:
        return ""
    h = hashlib.sha256()
    for it in sorted(map(str, items)):
        h.update(it.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:16]


# ------------------------------------------------------------
# LoadResult
# ------------------------------------------------------------
@dataclass
class LoadResult:
    found:         bool = False
    schema_match:  bool = False
    feature_match: bool = False
    symbols_match: bool = False
    saved_at:      Optional[float] = None
    age_hours:     Optional[float] = None
    symbol_count:  int = 0
    fully_warmed:  bool = False
    note:          str = ""


# ------------------------------------------------------------
# FCSStateStore
# ------------------------------------------------------------
class FCSStateStore:
    """PG-backed FCS state store."""

    def __init__(self, namespace: str, pg_url: str):
        self.namespace = namespace
        self.pg_url    = pg_url
        self._lock     = threading.Lock()
        self._last_save_ts = 0.0
        self._ensured = False

    # ------------------------------ connection
    def _conn(self):
        return psycopg2.connect(self.pg_url)

    def ensure_table(self) -> bool:
        if self._ensured:
            return True
        try:
            conn = self._conn(); c = conn.cursor()
            c.execute("""
                CREATE TABLE IF NOT EXISTS fcs_state_snapshot (
                    namespace       TEXT NOT NULL,
                    symbol          TEXT NOT NULL,
                    schema_version  INTEGER NOT NULL,
                    updated_at      DOUBLE PRECISION NOT NULL,
                    payload         BYTEA NOT NULL,
                    PRIMARY KEY (namespace, symbol)
                );
            """)
            c.execute("""
                CREATE INDEX IF NOT EXISTS idx_fcs_state_updated_at
                ON fcs_state_snapshot(updated_at);
            """)
            conn.commit()
            conn.close()
            self._ensured = True
            return True
        except Exception as e:
            logger.error(f"[StateStore] ensure_table failed: {e}")
            return False

    # ------------------------------ payload builder
    def _build_symbol_payload(self, service, sym: str, save_ts: float) -> Optional[bytes]:
        svc = service
        try:
            norm_state = svc.normalizers[sym].get_state() if sym in svc.normalizers else None
            hist_1m = svc.history_1min.get(sym)
            hist_5m = svc.history_5min.get(sym)
            opt_1m  = svc.option_snapshot.get(sym)
            opt_5m  = getattr(svc, 'option_snapshot_5m', {}).get(sym)
            lcv_1m  = getattr(svc, 'last_cum_volume',    {}).get(sym)
            lcv_5m  = getattr(svc, 'last_cum_volume_5m', {}).get(sym)
            wn      = bool(getattr(svc, 'warmup_needed',    {}).get(sym, True))
            wn5     = bool(getattr(svc, 'warmup_needed_5m', {}).get(sym, True))

            # 只保留 tail 避免 payload 膨胀
            try:
                history_len = int(getattr(svc, 'HISTORY_LEN', 500))
            except Exception:
                history_len = 500

            if isinstance(hist_1m, pd.DataFrame) and not hist_1m.empty:
                hist_1m = hist_1m.iloc[-history_len:].copy()
            if isinstance(hist_5m, pd.DataFrame) and not hist_5m.empty:
                hist_5m = hist_5m.iloc[-100:].copy()

            payload = {
                'v':                   SCHEMA_VERSION,
                'ts':                  save_ts,
                'normalizer':          norm_state,
                'history_1min':        hist_1m  if isinstance(hist_1m, pd.DataFrame) else None,
                'history_5min':        hist_5m  if isinstance(hist_5m, pd.DataFrame) else None,
                'option_snapshot':     np.asarray(opt_1m, dtype=np.float32) if opt_1m is not None else None,
                'option_snapshot_5m':  np.asarray(opt_5m, dtype=np.float32) if opt_5m is not None else None,
                'last_cum_volume':     np.asarray(lcv_1m, dtype=np.float32) if lcv_1m is not None else None,
                'last_cum_volume_5m':  np.asarray(lcv_5m, dtype=np.float32) if lcv_5m is not None else None,
                'warmup_needed':       wn,
                'warmup_needed_5m':    wn5,
            }
            return _encode(payload)
        except Exception as e:
            logger.warning(f"[StateStore] build_symbol_payload({sym}) failed: {e}")
            return None

    def _build_meta_payload(self, service, save_ts: float) -> bytes:
        svc = service
        feat_names = list(getattr(svc, 'all_feat_names', []) or [])
        syms = list(getattr(svc, 'symbols', []) or [])
        meta = {
            'v':                   SCHEMA_VERSION,
            'ts':                  save_ts,
            'feature_names_hash':  _hash_list(feat_names),
            'symbols_hash':        _hash_list(syms),
            'symbol_count':        len(syms),
            'run_mode':            os.environ.get("RUN_MODE", "").upper(),
        }
        return _encode(meta)

    # ------------------------------ save / load
    def save(self, service, sync: bool = False) -> bool:
        """Upsert 全部 symbol + meta 行. 默认 async 线程, 不阻塞主循环."""
        if not self.ensure_table():
            return False
        save_ts = time.time()

        def _task():
            with self._lock:
                try:
                    rows: List[Tuple[str, str, int, float, bytes]] = []
                    meta_blob = self._build_meta_payload(service, save_ts)
                    rows.append((self.namespace, META_SYMBOL, SCHEMA_VERSION, save_ts, meta_blob))

                    for sym in getattr(service, 'symbols', []) or []:
                        blob = self._build_symbol_payload(service, sym, save_ts)
                        if blob is None:
                            continue
                        rows.append((self.namespace, sym, SCHEMA_VERSION, save_ts, blob))

                    if not rows:
                        return
                    conn = self._conn(); c = conn.cursor()
                    psycopg2.extras.execute_batch(c, """
                        INSERT INTO fcs_state_snapshot
                            (namespace, symbol, schema_version, updated_at, payload)
                        VALUES (%s, %s, %s, %s, %s)
                        ON CONFLICT (namespace, symbol) DO UPDATE SET
                            schema_version = EXCLUDED.schema_version,
                            updated_at     = EXCLUDED.updated_at,
                            payload        = EXCLUDED.payload
                    """, rows, page_size=64)
                    conn.commit()
                    conn.close()
                    self._last_save_ts = save_ts
                    logger.info(
                        f"💾 [StateStore] saved ns={self.namespace} rows={len(rows)} "
                        f"(incl. _META_) v{SCHEMA_VERSION}"
                    )
                except Exception as e:
                    logger.error(f"[StateStore] save failed: {e}")

        if sync:
            _task()
            return True
        threading.Thread(target=_task, daemon=True).start()
        return True

    def load(self, service) -> LoadResult:
        """把 PG 里保存的状态应用到 service. 返回 LoadResult."""
        result = LoadResult()
        if not self.ensure_table():
            return result
        try:
            conn = self._conn(); c = conn.cursor()
            c.execute(
                "SELECT symbol, schema_version, updated_at, payload "
                "FROM fcs_state_snapshot WHERE namespace = %s",
                (self.namespace,),
            )
            rows = c.fetchall()
            conn.close()
        except Exception as e:
            logger.error(f"[StateStore] load query failed: {e}")
            return result

        if not rows:
            result.note = "empty"
            return result
        result.found = True

        meta = None
        per_symbol: Dict[str, dict] = {}
        oldest_ts = None
        for sym, ver, updated_at, blob in rows:
            if ver != SCHEMA_VERSION:
                result.note = f"schema mismatch v={ver}"
                continue
            try:
                obj = _decode(blob)
            except Exception:
                continue
            if sym == META_SYMBOL:
                meta = obj
            else:
                per_symbol[sym] = obj
                if oldest_ts is None or updated_at < oldest_ts:
                    oldest_ts = updated_at
            result.saved_at = float(updated_at)

        if meta is None:
            result.note = "no _META_"
            return result

        result.schema_match = (meta.get('v') == SCHEMA_VERSION)

        svc_feat_hash = _hash_list(getattr(service, 'all_feat_names', []) or [])
        svc_syms_hash = _hash_list(getattr(service, 'symbols',        []) or [])
        result.feature_match = (meta.get('feature_names_hash') == svc_feat_hash)
        result.symbols_match = (meta.get('symbols_hash')       == svc_syms_hash)
        result.symbol_count  = len(per_symbol)

        if not result.schema_match:
            result.note = "schema version mismatch"
            return result
        if not result.feature_match:
            result.note = f"feature_names hash mismatch (svc={svc_feat_hash} db={meta.get('feature_names_hash')})"
            return result
        if not result.symbols_match:
            result.note = "symbols hash mismatch"
            return result
        if result.saved_at:
            result.age_hours = max(0.0, (time.time() - result.saved_at) / 3600.0)

        # 应用到 service
        applied = 0
        fully_count = 0
        for sym, obj in per_symbol.items():
            try:
                if self._apply_symbol(service, sym, obj):
                    applied += 1
                    # 看 normalizer 的 count
                    try:
                        norm = service.normalizers.get(sym)
                        if norm is not None and norm.count >= 500:
                            fully_count += 1
                    except Exception:
                        pass
            except Exception as e:
                logger.warning(f"[StateStore] apply {sym} failed: {e}")

        result.fully_warmed = (applied > 0 and fully_count >= max(1, int(0.8 * len(per_symbol))))
        result.note = f"restored {applied}/{len(per_symbol)} symbols"
        return result

    def _apply_symbol(self, service, sym: str, obj: dict) -> bool:
        svc = service
        if sym not in getattr(svc, 'normalizers', {}):
            return False
        try:
            norm_state = obj.get('normalizer')
            if norm_state:
                svc.normalizers[sym].set_state(norm_state)
            h1 = obj.get('history_1min')
            if isinstance(h1, pd.DataFrame) and not h1.empty:
                svc.history_1min[sym] = h1
            h5 = obj.get('history_5min')
            if isinstance(h5, pd.DataFrame) and not h5.empty:
                svc.history_5min[sym] = h5
            o1 = obj.get('option_snapshot')
            if isinstance(o1, np.ndarray) and o1.size > 0:
                svc.option_snapshot[sym] = o1.astype(np.float32, copy=False)
            o5 = obj.get('option_snapshot_5m')
            if isinstance(o5, np.ndarray) and o5.size > 0 and hasattr(svc, 'option_snapshot_5m'):
                svc.option_snapshot_5m[sym] = o5.astype(np.float32, copy=False)
            lcv1 = obj.get('last_cum_volume')
            if isinstance(lcv1, np.ndarray) and lcv1.size > 0 and hasattr(svc, 'last_cum_volume'):
                svc.last_cum_volume[sym] = lcv1.astype(np.float32, copy=False)
            lcv5 = obj.get('last_cum_volume_5m')
            if isinstance(lcv5, np.ndarray) and lcv5.size > 0 and hasattr(svc, 'last_cum_volume_5m'):
                svc.last_cum_volume_5m[sym] = lcv5.astype(np.float32, copy=False)
            if hasattr(svc, 'warmup_needed'):
                svc.warmup_needed[sym] = bool(obj.get('warmup_needed', True))
            if hasattr(svc, 'warmup_needed_5m'):
                svc.warmup_needed_5m[sym] = bool(obj.get('warmup_needed_5m', True))
            return True
        except Exception as e:
            logger.warning(f"[StateStore] _apply_symbol({sym}) failed: {e}")
            return False

    # ------------------------------ maintenance
    def drop_namespace(self) -> int:
        if not self.ensure_table():
            return 0
        try:
            conn = self._conn(); c = conn.cursor()
            c.execute("DELETE FROM fcs_state_snapshot WHERE namespace = %s", (self.namespace,))
            conn.commit()
            n = c.rowcount
            conn.close()
            logger.warning(f"🗑 [StateStore] dropped ns={self.namespace} rows={n}")
            return int(n)
        except Exception as e:
            logger.error(f"[StateStore] drop_namespace failed: {e}")
            return 0

    def summary(self) -> dict:
        """只读摘要, 供 Dashboard 消费."""
        out = {
            'namespace':      self.namespace,
            'table_exists':   False,
            'rows':           0,
            'meta_row':       None,
            'latest_saved':   None,
            'symbol_count':   0,
            'schema_version': SCHEMA_VERSION,
        }
        try:
            conn = self._conn(); c = conn.cursor()
            c.execute("SELECT to_regclass('public.fcs_state_snapshot')")
            if c.fetchone()[0] is None:
                conn.close()
                return out
            out['table_exists'] = True
            c.execute(
                "SELECT symbol, schema_version, updated_at, payload "
                "FROM fcs_state_snapshot WHERE namespace = %s",
                (self.namespace,),
            )
            rows = c.fetchall()
            conn.close()
        except Exception as e:
            logger.error(f"[StateStore] summary failed: {e}")
            return out

        out['rows'] = len(rows)
        latest = None
        for sym, ver, updated_at, blob in rows:
            if latest is None or updated_at > latest:
                latest = updated_at
            if sym == META_SYMBOL:
                try:
                    out['meta_row'] = _decode(blob)
                except Exception:
                    pass
            else:
                out['symbol_count'] += 1
        out['latest_saved'] = latest
        return out

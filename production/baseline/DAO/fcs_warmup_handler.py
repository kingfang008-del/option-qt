from __future__ import annotations

import json
import logging
import os
from datetime import datetime, time as dt_time
from collections import defaultdict

import numpy as np
import pandas as pd

from config import NY_TZ, USE_5M_OPTION_DATA


logger = logging.getLogger("FeatService")


class FCSWarmupHandler:
    """预热与回填处理器。"""

    def __init__(self, service):
        self.service = service

    def warmup_from_history(self, replay_features=True):
        svc = self.service
        logger.info(f"🔥 Starting Warmup from PostgreSQL (Replay Features={replay_features})...")

        target_bars = 2000 if replay_features else svc.HISTORY_LEN
        replay_start_ts = float(os.environ.get('REPLAY_START_TS', 'inf'))
        if replay_start_ts != float('inf'):
            logger.info(f"🛡️ Time Truncation Active: Ignoring DB records >= {replay_start_ts}")

        if os.environ.get('SKIP_DEEP_WARMUP') == '1':
            logger.info("⏭️ [Warmup] Skipped Deep Warmup by environment flag.")
            return

        sym_to_rows_1m = defaultdict(list)
        sym_to_rows_5m = defaultdict(list)
        sym_to_opt_row = {}
        sym_to_opt_row_5m = {}

        warmup_count = 0
        try:
            conn = svc._get_pg_conn()
            c = conn.cursor()
            all_symbols = tuple(svc.symbols)

            logger.info(f"📥 Bulk Loading 1m bars for {len(all_symbols)} symbols...")
            c.execute("""
                SELECT symbol, ts, open, high, low, close, volume, vwap
                FROM market_bars_1m
                WHERE symbol IN %s AND ts < %s
                ORDER BY ts DESC
            """, (all_symbols, replay_start_ts))
            for r in c.fetchall():
                sym = r[0]
                if len(sym_to_rows_1m[sym]) < (target_bars + 500):
                    sym_to_rows_1m[sym].append(r[1:])

            logger.info("📥 Bulk Loading 5m bars...")
            c.execute("""
                SELECT symbol, ts, open, high, low, close, volume, vwap
                FROM market_bars_5m
                WHERE symbol IN %s AND ts < %s
                ORDER BY ts DESC
            """, (all_symbols, replay_start_ts))
            for r in c.fetchall():
                sym = r[0]
                if len(sym_to_rows_5m[sym]) < 500:
                    sym_to_rows_5m[sym].append(r[1:])

            logger.info("📥 Bulk Loading option snapshots...")
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

            for sym in svc.symbols:
                rows = sym_to_rows_1m.get(sym, [])
                raw_bars_1m = []
                for r in rows:
                    dt_ny = datetime.fromtimestamp(r[0], NY_TZ).replace(second=0, microsecond=0)
                    t = dt_ny.time()
                    if dt_time(9, 30) <= t < dt_time(16, 0):
                        raw_bars_1m.append({
                            'ts': dt_ny, 'open': r[1], 'high': r[2], 'low': r[3],
                            'close': r[4], 'volume': r[5], 'vwap': r[6] if len(r) > 6 else r[4]
                        })
                if raw_bars_1m:
                    df_hist_1m = pd.DataFrame(raw_bars_1m).drop_duplicates(subset=['ts']).set_index('ts').sort_index()
                    svc.history_1min[sym] = df_hist_1m.iloc[-svc.HISTORY_LEN:]

                rows_5m = sym_to_rows_5m.get(sym, [])
                raw_bars_5m = []
                for r in rows_5m:
                    dt_ny = datetime.fromtimestamp(r[0], NY_TZ).replace(second=0, microsecond=0)
                    t = dt_ny.time()
                    if dt_time(9, 30) <= t < dt_time(16, 0):
                        raw_bars_5m.append({
                            'ts': dt_ny, 'open': r[1], 'high': r[2], 'low': r[3],
                            'close': r[4], 'volume': r[5], 'vwap': r[6] if len(r) > 6 else r[4]
                        })
                if raw_bars_5m:
                    df_hist_5m = pd.DataFrame(raw_bars_5m).drop_duplicates(subset=['ts']).set_index('ts').sort_index()
                    svc.history_5min[sym] = df_hist_5m.iloc[-500:]

                opt_snap = sym_to_opt_row.get(sym)
                if opt_snap:
                    try:
                        if isinstance(opt_snap, str):
                            opt_snap = json.loads(opt_snap)
                        buckets = opt_snap.get('buckets', [])
                        arr = np.array(buckets, dtype=np.float32)
                        if arr.shape[0] < 6:
                            arr = np.vstack([arr, np.zeros((6 - arr.shape[0], arr.shape[1]), dtype=np.float32)])
                        if arr.shape[1] < 12:
                            arr = np.hstack([arr, np.zeros((arr.shape[0], 12 - arr.shape[1]), dtype=np.float32)])
                        svc.option_snapshot[sym] = arr[:, :12]
                        if np.sum(arr[:, 6]) > 0:
                            svc.last_cum_volume[sym] = arr[:, 6].copy()
                            svc.warmup_needed[sym] = False
                    except Exception:
                        pass

                if USE_5M_OPTION_DATA:
                    opt_snap_5m = sym_to_opt_row_5m.get(sym)
                    if opt_snap_5m:
                        try:
                            if isinstance(opt_snap_5m, str):
                                opt_snap_5m = json.loads(opt_snap_5m)
                            buckets = opt_snap_5m.get('buckets', [])
                            arr = np.array(buckets, dtype=np.float32)
                            if arr.shape[0] < 6:
                                arr = np.vstack([arr, np.zeros((6 - arr.shape[0], arr.shape[1]), dtype=np.float32)])
                            if arr.shape[1] < 12:
                                arr = np.hstack([arr, np.zeros((arr.shape[0], 12 - arr.shape[1]), dtype=np.float32)])
                            svc.option_snapshot_5m[sym] = arr[:, :12]
                            if np.sum(arr[:, 6]) > 0:
                                svc.last_cum_volume_5m[sym] = arr[:, 6].copy()
                                svc.warmup_needed_5m[sym] = False
                        except Exception:
                            pass

                if replay_features and not svc.history_1min[sym].empty and len(svc.history_1min[sym]) > 50:
                    sliced_snaps = {s: snap[:, :12] for s, snap in svc.option_snapshot.items()}
                    res = svc.engine.compute_all_inputs(
                        {sym: svc.history_1min[sym]}, svc.fast_feat_names, svc.slow_feat_names, sliced_snaps,
                        skip_scaling=True, recalc_greeks=svc.recalc_greeks
                    )
                    if sym in res:
                        t_fast = res[sym]['fast_1m'][0].cpu().numpy()
                        t_slow = res[sym]['slow_1m'][0].cpu().numpy()
                        L = t_fast.shape[1]
                        norm = svc.normalizers[sym]
                        for i in range(L):
                            raw_vec = np.zeros(len(svc.all_feat_names), dtype=np.float32)
                            for k, fname in enumerate(svc.fast_feat_names):
                                idx = svc.feat_name_to_idx.get(fname)
                                if idx is not None:
                                    raw_vec[idx] = t_fast[k, i]
                            for k, fname in enumerate(svc.slow_feat_names):
                                idx = svc.feat_name_to_idx.get(fname)
                                if idx is not None:
                                    raw_vec[idx] = t_slow[k, i]
                            norm.process_frame(raw_vec)
                warmup_count += 1

        except Exception as e:
            logger.error(f"Backfill Error: {e}", exc_info=True)

        logger.info(f"✅ Deep Warmup Complete for {warmup_count} symbols (Dual Resolution).")
        self.publish_warmup_status()

    def robust_backfill_and_warmup(self):
        svc = self.service
        logger.info("🔄 [CRITICAL] Starting 5-Day Backfill & Normalizer Warmup (Window=2000)...")

        warmup_count = 0
        try:
            conn = svc._get_pg_conn()
            c = conn.cursor()
            for sym in svc.symbols:
                raw_bars = []
                opt_snap = None

                c.execute("SELECT ts, open, high, low, close, volume FROM market_bars_1m WHERE symbol=%s ORDER BY ts DESC LIMIT 2500", (sym,))
                rows = c.fetchall()
                c.execute("SELECT buckets_json FROM option_snapshots_1m WHERE symbol=%s ORDER BY ts DESC LIMIT 1", (sym,))
                opt_row = c.fetchone()
                if opt_row:
                    opt_snap = opt_row[0]
                    if isinstance(opt_snap, str):
                        opt_snap = json.loads(opt_snap)

                for r in rows:
                    dt_ny = datetime.fromtimestamp(r[0], NY_TZ)
                    t = dt_ny.time()
                    if dt_time(9, 30) <= t < dt_time(16, 0):
                        raw_bars.append({'ts': dt_ny, 'open': r[1], 'high': r[2], 'low': r[3], 'close': r[4], 'volume': r[5]})

                if not raw_bars:
                    continue

                raw_bars = sorted(raw_bars, key=lambda x: x['ts'])
                if len(raw_bars) > 2000:
                    raw_bars = raw_bars[-2000:]

                df_hist = pd.DataFrame(raw_bars).drop_duplicates(subset=['ts']).set_index('ts')
                svc.history_1min[sym] = df_hist.iloc[-svc.HISTORY_LEN:]

                if opt_snap:
                    try:
                        buckets = opt_snap.get('buckets', []) if isinstance(opt_snap, dict) else opt_snap
                        arr = np.array(buckets, dtype=np.float32)
                        if arr.shape[0] < 6:
                            arr = np.vstack([arr, np.zeros((6 - arr.shape[0], 10), dtype=np.float32)])
                        svc.option_snapshot[sym] = arr
                        if np.sum(arr[:, 6]) > 0:
                            svc.last_cum_volume[sym] = arr[:, 6].copy()
                            svc.warmup_needed[sym] = False
                    except Exception:
                        pass

                if len(df_hist) > 50:
                    sliced_snaps = {s: snap[:, :12] for s, snap in svc.option_snapshot.items()}
                    res = svc.engine.compute_all_inputs(
                        {sym: df_hist}, svc.fast_feat_names, svc.slow_feat_names, sliced_snaps,
                        skip_scaling=True, recalc_greeks=svc.recalc_greeks
                    )
                    if sym in res:
                        t_fast = res[sym]['fast_1m'][0].cpu().numpy()
                        t_slow = res[sym]['slow_1m'][0].cpu().numpy()
                        L = t_fast.shape[1]
                        norm = svc.normalizers[sym]
                        for i in range(L):
                            raw_vec = np.zeros(len(svc.all_feat_names), dtype=np.float32)
                            for k, name in enumerate(svc.fast_feat_names):
                                if name in svc.feat_name_to_idx:
                                    raw_vec[svc.feat_name_to_idx[name]] = t_fast[k, i]
                            for k, name in enumerate(svc.slow_feat_names):
                                if name in svc.feat_name_to_idx:
                                    raw_vec[svc.feat_name_to_idx[name]] = t_slow[k, i]
                            norm.process_frame(raw_vec)
                warmup_count += 1

        except Exception as e:
            logger.error(f"Backfill Error: {e}")

        logger.info(f"✅ Deep Warmup (Window=2000) Complete for {warmup_count} symbols.")
        self.publish_warmup_status()

    def publish_warmup_status(self):
        svc = self.service
        try:
            status = {}
            for sym, norm in svc.normalizers.items():
                status[sym] = norm.count
            key = "monitor:warmup:norm"
            if status:
                svc.r.hset(key, mapping=status)
                svc.r.expire(key, 3600)
        except Exception:
            pass

from __future__ import annotations

import json
import logging
import os
import re
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from config import ENABLE_DEEP_WARMUP, NY_TZ, PG_DB_URL, FCS_RECENT_STATE_MAX_HOURS
from utils import serialization_utils as ser


logger = logging.getLogger("FeatService")


class FCSSupportHandler:
    """审计、状态持久化、门控与辅助工具处理器。"""

    def __init__(self, service):
        self.service = service

    def to_audit_matrix(self, snap):
        if snap is None:
            return []
        arr = np.asarray(snap, dtype=np.float64)
        if arr.ndim != 2:
            return []
        return arr[:, :12].tolist()

    def to_audit_row(self, snap, row_idx: int):
        arr = np.asarray(snap, dtype=np.float64)
        if arr.ndim != 2 or row_idx >= arr.shape[0]:
            return {}
        row = arr[row_idx]
        return {
            'price': float(row[0]) if row.shape[0] > 0 else 0.0,
            'delta': float(row[1]) if row.shape[0] > 1 else 0.0,
            'gamma': float(row[2]) if row.shape[0] > 2 else 0.0,
            'vega': float(row[3]) if row.shape[0] > 3 else 0.0,
            'theta': float(row[4]) if row.shape[0] > 4 else 0.0,
            'strike': float(row[5]) if row.shape[0] > 5 else 0.0,
            'volume': float(row[6]) if row.shape[0] > 6 else 0.0,
            'iv': float(row[7]) if row.shape[0] > 7 else 0.0,
            'bid': float(row[8]) if row.shape[0] > 8 else 0.0,
            'ask': float(row[9]) if row.shape[0] > 9 else 0.0,
            'bid_size': float(row[10]) if row.shape[0] > 10 else 0.0,
            'ask_size': float(row[11]) if row.shape[0] > 11 else 0.0,
        }

    def build_greeks_input_audit(
        self, *, symbol: str, buckets, contracts, stock_price: float, timestamp: float, bucket_id: int = 5
    ):
        svc = self.service
        arr = np.asarray(buckets, dtype=np.float64)
        if arr.ndim != 2 or bucket_id >= arr.shape[0]:
            return {}
        row = self.to_audit_row(arr, bucket_id)
        contract = contracts[bucket_id] if contracts and bucket_id < len(contracts) else ""
        payload = {
            'symbol': symbol, 'bucket_id': int(bucket_id), 'contract': contract,
            'timestamp': float(timestamp) if timestamp is not None else None,
            'stock_price_input': float(stock_price) if stock_price is not None else None,
            'row_before_or_after': row,
        }
        try:
            ts_ny = pd.Timestamp(timestamp, unit='s', tz='UTC').tz_convert('America/New_York')
            ts_anchor = ts_ny.floor('1min')
            payload['timestamp_ny'] = ts_ny.isoformat()
            payload['ts_anchor_ny'] = ts_anchor.isoformat()

            r_val = None
            date_str = ts_ny.strftime('%Y%m%d')
            if not hasattr(svc.engine, 'rfr_cache'):
                svc.engine.rfr_cache = {}
            if date_str in svc.engine.rfr_cache:
                r_val = svc.engine.rfr_cache[date_str]
            else:
                rfr_candidates = [
                    Path("/home/kingfang007/risk_free_rates.parquet"),
                    Path(__file__).resolve().parents[3] / "risk_free_rates.parquet",
                    Path("risk_free_rates.parquet"),
                ]
                rfr_file = next((p for p in rfr_candidates if p.exists()), None)
                if rfr_file is not None:
                    df_rfr = pd.read_parquet(rfr_file)
                    search_date = ts_ny.replace(hour=0, minute=0, second=0, microsecond=0).tz_localize(None)
                    idx = df_rfr.index.searchsorted(search_date)
                    idx = int(np.clip(idx, 0, len(df_rfr) - 1))
                    r_val = float(df_rfr['DGS3MO'].iloc[idx])
                    if r_val > 1.0:
                        r_val /= 100.0
                    svc.engine.rfr_cache[date_str] = r_val
                    payload['rfr_source'] = str(rfr_file)
            payload['r'] = float(r_val) if r_val is not None else None

            ext = str(contract).replace('O:', '')
            m = re.search(r'\d{6}', ext)
            if m:
                exp_str = m.group(0)
                expiry_dt = pd.to_datetime(exp_str, format='%y%m%d').tz_localize('America/New_York') + pd.Timedelta(hours=16)
                t_years = max(1e-6, (expiry_dt - ts_anchor).total_seconds() / 31557600.0)
                payload['expiry_ny'] = expiry_dt.isoformat()
                payload['T_years'] = float(t_years)
        except Exception as audit_e:
            payload['input_audit_error'] = str(audit_e)
        return payload

    def maybe_write_runtime_payload_audit(self, **kwargs):
        svc = self.service
        if not svc.runtime_payload_audit_enabled:
            return
        symbol = kwargs.get('symbol')
        alpha_label_ts = kwargs.get('alpha_label_ts')
        if symbol != svc.runtime_payload_audit_symbol:
            return
        target_ts = int(float(alpha_label_ts))
        if target_ts != svc.runtime_payload_audit_ts:
            return
        audit_key = (symbol, target_ts)
        if audit_key in svc.runtime_payload_audit_written:
            return

        bucket_id = int(kwargs.get('bucket_id', 5))
        contracts = kwargs.get('contracts') or []
        contract = contracts[bucket_id] if contracts and bucket_id < len(contracts) else ""

        payload = {
            'ts': target_ts,
            'source_ts': float(kwargs.get('data_ts')) if kwargs.get('data_ts') is not None else None,
            'symbol': symbol,
            'bucket_id': bucket_id,
            'bucket_name': 'NEXT_CALL_ATM' if bucket_id == 5 else str(bucket_id),
            'contract': contract,
            'stock_price': float(kwargs.get('payload_stock_price', 0.0)),
            'latest_stock_price': float(kwargs.get('latest_stock_price', 0.0)),
            'last_stock_update_ts': float(kwargs.get('last_stock_update_ts')) if kwargs.get('last_stock_update_ts') is not None else None,
            'last_option_update_ts': float(kwargs.get('last_option_update_ts')) if kwargs.get('last_option_update_ts') is not None else None,
            'contracts': list(contracts) if contracts else [],
            'option_snapshot': self.to_audit_matrix(kwargs.get('source_option_snapshot')),
            'frozen_option_snapshot': self.to_audit_matrix(kwargs.get('frozen_option_snapshot')),
            'payload_option_snapshot': self.to_audit_matrix(kwargs.get('payload_option_snapshot')),
            'latest_opt_buckets': self.to_audit_matrix(kwargs.get('latest_opt_buckets')),
            'frozen_latest_opt_buckets': self.to_audit_matrix(kwargs.get('frozen_latest_opt_buckets')),
            'pre_supplement_greeks_input': kwargs.get('pre_supplement_greeks_input') or {},
            'post_supplement_greeks_input': kwargs.get('post_supplement_greeks_input') or {},
            'focus_rows': {
                'option_snapshot': self.to_audit_row(kwargs.get('source_option_snapshot'), bucket_id),
                'frozen_option_snapshot': self.to_audit_row(kwargs.get('frozen_option_snapshot'), bucket_id),
                'payload_option_snapshot': self.to_audit_row(kwargs.get('payload_option_snapshot'), bucket_id),
                'latest_opt_buckets': self.to_audit_row(kwargs.get('latest_opt_buckets'), bucket_id),
                'frozen_latest_opt_buckets': self.to_audit_row(kwargs.get('frozen_latest_opt_buckets'), bucket_id),
            }
        }
        try:
            svc.runtime_payload_audit_dir.mkdir(parents=True, exist_ok=True)
            out_path = svc.runtime_payload_audit_dir / f"{symbol}_{target_ts}_runtime_audit.json"
            out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
            svc.runtime_payload_audit_written.add(audit_key)
            logger.info(f"💾 [FCS_RUNTIME_AUDIT] Wrote runtime audit to {out_path}")
        except Exception as audit_e:
            logger.warning(f"⚠️ [FCS_RUNTIME_AUDIT] Failed to write audit JSON: {audit_e}")

    def ensure_debug_tables(self, date_str):
        import psycopg2
        day = datetime.strptime(date_str, '%Y%m%d')
        start_dt = NY_TZ.localize(day)
        end_dt = start_dt + timedelta(days=1)
        start_ts, end_ts = start_dt.timestamp(), end_dt.timestamp()
        svc = self.service
        conn = None
        try:
            conn = psycopg2.connect(PG_DB_URL)
            conn.autocommit = True
            c = conn.cursor()

            def get_existing_cols(table_name):
                c.execute(f"SELECT column_name FROM information_schema.columns WHERE table_name = '{table_name}'")
                return [r[0] for r in c.fetchall()]

            c.execute("""CREATE TABLE IF NOT EXISTS debug_fast (ts DOUBLE PRECISION, symbol TEXT, created_at TEXT, write_wall_ts DOUBLE PRECISION, write_wall_at TEXT, source_ts DOUBLE PRECISION, PRIMARY KEY (ts, symbol)) PARTITION BY RANGE (ts);""")
            existing_cols_fast = get_existing_cols("debug_fast")
            for fixed_col, fixed_type in [("write_wall_ts", "DOUBLE PRECISION"), ("write_wall_at", "TEXT"), ("source_ts", "DOUBLE PRECISION")]:
                if fixed_col not in existing_cols_fast:
                    c.execute(f'ALTER TABLE debug_fast ADD COLUMN "{fixed_col}" {fixed_type}')
            for name in svc.fast_feat_names:
                if name not in existing_cols_fast:
                    c.execute(f'ALTER TABLE debug_fast ADD COLUMN "{name}" DOUBLE PRECISION')
            part_fast = f"debug_fast_{date_str}"
            c.execute(f"CREATE TABLE IF NOT EXISTS {part_fast} PARTITION OF debug_fast FOR VALUES FROM ({start_ts}) TO ({end_ts});")

            c.execute("""CREATE TABLE IF NOT EXISTS debug_slow (ts DOUBLE PRECISION, symbol TEXT, created_at TEXT, write_wall_ts DOUBLE PRECISION, write_wall_at TEXT, source_ts DOUBLE PRECISION, PRIMARY KEY (ts, symbol)) PARTITION BY RANGE (ts);""")
            existing_cols_slow = get_existing_cols("debug_slow")
            for fixed_col, fixed_type in [("write_wall_ts", "DOUBLE PRECISION"), ("write_wall_at", "TEXT"), ("source_ts", "DOUBLE PRECISION")]:
                if fixed_col not in existing_cols_slow:
                    c.execute(f'ALTER TABLE debug_slow ADD COLUMN "{fixed_col}" {fixed_type}')
            for name in svc.slow_feat_names:
                if name not in existing_cols_slow:
                    c.execute(f'ALTER TABLE debug_slow ADD COLUMN "{name}" DOUBLE PRECISION')
            part_slow = f"debug_slow_{date_str}"
            c.execute(f"CREATE TABLE IF NOT EXISTS {part_slow} PARTITION OF debug_slow FOR VALUES FROM ({start_ts}) TO ({end_ts});")

            for part_name, feat_names in [(part_fast, svc.fast_feat_names), (part_slow, svc.slow_feat_names)]:
                existing_partition_cols = get_existing_cols(part_name)
                for fixed_col, fixed_type in [("write_wall_ts", "DOUBLE PRECISION"), ("write_wall_at", "TEXT"), ("source_ts", "DOUBLE PRECISION")]:
                    if fixed_col not in existing_partition_cols:
                        try:
                            c.execute(f'ALTER TABLE {part_name} ADD COLUMN "{fixed_col}" {fixed_type}')
                            logger.info(f"➕ [Schema Sync] Added fixed column '{fixed_col}' to partition {part_name}")
                        except Exception as ae:
                            logger.warning(f"Failed to add fixed column {fixed_col} to {part_name}: {ae}")
                for name in feat_names:
                    if name not in existing_partition_cols:
                        try:
                            c.execute(f'ALTER TABLE {part_name} ADD COLUMN "{name}" DOUBLE PRECISION')
                            logger.info(f"➕ [Schema Sync] Added column '{name}' to partition {part_name}")
                        except Exception as ae:
                            logger.warning(f"Failed to add column {name} to {part_name}: {ae}")
            logger.info(f"✅ Debug 表分区化确认完成: debug_fast_{date_str}, debug_slow_{date_str}")
        except Exception as e:
            logger.error(f"❌ Debug Postgres 分区化建表失败: {e}")
        finally:
            if conn:
                conn.close()

    def write_debug_batch(self, ts, date_str, fast_data_list, slow_data_list, source_ts=None):
        if not fast_data_list and not slow_data_list:
            return
        import psycopg2
        svc = self.service
        conn = None
        try:
            conn = psycopg2.connect(PG_DB_URL)
            conn.autocommit = True
            c = conn.cursor()
            wall_ts = time.time()
            wall_dt_ny = datetime.fromtimestamp(wall_ts, NY_TZ)
            # created_at = 真实写入墙钟（NY 时区字符串）。label 时间随时可由 ts 字段反算。
            created_at = wall_dt_ny.strftime("%Y-%m-%d %H:%M:%S")
            write_wall_at = created_at
            source_ts = float(source_ts) if source_ts is not None else float(ts)
            label_ny = datetime.fromtimestamp(ts, NY_TZ).strftime("%Y-%m-%d %H:%M:%S")
            earliest_commit_wall_ts = float(ts) + 60.0 + float(getattr(svc, "minute_commit_grace_sec", 1.0) or 0.0)
            if slow_data_list:
                svc._log_minute_write_audit(
                    stage="debug_write:execute",
                    label_ts=float(ts),
                    data_ts=float(ts),
                    wall_ts=wall_ts,
                    symbols=[sym for sym, _ in slow_data_list],
                    extra={
                        'rows': len(slow_data_list),
                        'date_str': date_str,
                        'label_ny': label_ny,
                        'created_at_wall_ny': created_at,
                    },
                    level="error" if wall_ts + 1e-9 < earliest_commit_wall_ts else "info",
                    force=bool(wall_ts + 1e-9 < earliest_commit_wall_ts),
                )
                if wall_ts + 1e-9 < earliest_commit_wall_ts:
                    logger.error(
                        "🛑 [Debug-Write-WallClock] debug_slow actual write wall-clock is earlier than allowed commit time "
                        f"| wall_ts={wall_ts:.3f} earliest={earliest_commit_wall_ts:.3f} label_ts={float(ts):.3f}"
                    )

            if fast_data_list:
                table_name = f"debug_fast_{date_str}"
                fixed_cols = ["ts", "symbol", "created_at", "write_wall_ts", "write_wall_at", "source_ts"]
                cols_str = ", ".join(fixed_cols + [f'"{name}"' for name in svc.fast_feat_names])
                placeholders = ",".join(["%s"] * (len(fixed_cols) + len(svc.fast_feat_names)))
                rows_fast = []
                for sym, vals in fast_data_list:
                    clean_vals = [None if not np.isfinite(v) else float(v) for v in vals]
                    rows_fast.append([ts, sym, created_at, wall_ts, write_wall_at, source_ts] + clean_vals)
                if rows_fast:
                    update_cols = ["created_at", "write_wall_ts", "write_wall_at", "source_ts"] + [f'"{name}"' for name in svc.fast_feat_names]
                    update_sql = ", ".join([f"{col} = EXCLUDED.{col}" for col in update_cols])
                    c.executemany(
                        f"INSERT INTO {table_name} ({cols_str}) VALUES ({placeholders}) "
                        f"ON CONFLICT (ts, symbol) DO UPDATE SET {update_sql}",
                        rows_fast
                    )

            if slow_data_list:
                table_name = f"debug_slow_{date_str}"
                fixed_cols = ["ts", "symbol", "created_at", "write_wall_ts", "write_wall_at", "source_ts"]
                cols_str = ", ".join(fixed_cols + [f'"{name}"' for name in svc.slow_feat_names])
                placeholders = ",".join(["%s"] * (len(fixed_cols) + len(svc.slow_feat_names)))
                rows_slow = []
                for sym, vals in slow_data_list:
                    clean_vals = [None if not np.isfinite(v) else float(v) for v in vals]
                    rows_slow.append([ts, sym, created_at, wall_ts, write_wall_at, source_ts] + clean_vals)
                if rows_slow:
                    logger.info(f"📁 [Debug-Write] Writing {len(rows_slow)} symbols to {table_name} at ts={ts}")
                    update_cols = ["created_at", "write_wall_ts", "write_wall_at", "source_ts"] + [f'"{name}"' for name in svc.slow_feat_names]
                    update_sql = ", ".join([f"{col} = EXCLUDED.{col}" for col in update_cols])
                    c.executemany(
                        f"INSERT INTO {table_name} ({cols_str}) VALUES ({placeholders}) "
                        f"ON CONFLICT (ts, symbol) DO UPDATE SET {update_sql}",
                        rows_slow
                    )
                    logger.info(f"✅ [Debug-Write] Success for {table_name}")
            c.close()
        except Exception as e:
            if "does not exist" in str(e):
                logger.warning(f"⚠️ [跳过写入] 表 {date_str} 尚未建好: {e}")
            else:
                logger.error(f"❌ DB 写入报错: {e}")
        finally:
            if conn:
                conn.close()

    def save_service_state(self):
        svc = self.service
        try:
            state = {'ts': time.time(), 'normalizers': {s: norm.get_state() for s, norm in svc.normalizers.items()}}
            temp_file = svc.state_file.with_suffix('.tmp')
            with open(temp_file, 'wb') as f:
                f.write(ser.pack(state))
            if svc.state_file.exists():
                os.remove(svc.state_file)
            os.rename(temp_file, svc.state_file)
        except Exception as e:
            logger.error(f"Save State Error: {e}")

    def load_service_state(self):
        svc = self.service
        if not svc.state_file.exists():
            if not ENABLE_DEEP_WARMUP:
                logger.info("⏭️ [Warmup] No state file found and Deep Warmup is DISABLED via config. Starting cold.")
                return
            logger.info("ℹ️ No state file found, starting fresh (Deep Warmup).")
            svc._warmup_from_history(replay_features=True)
            return
        try:
            with open(svc.state_file, 'rb') as f:
                state = ser.unpack(f.read())
            state_time = datetime.fromtimestamp(state['ts'], NY_TZ)
            now_ny = datetime.now(NY_TZ)
            if state_time.date() != now_ny.date():
                age_hours = max(0.0, (now_ny - state_time).total_seconds() / 3600.0)
                can_reuse_recent = age_hours <= float(FCS_RECENT_STATE_MAX_HOURS)
                if can_reuse_recent:
                    logger.warning(
                        f"♻️ [Warmup] Reusing recent cross-day FCS state from {state_time.strftime('%Y-%m-%d %H:%M:%S %Z')} "
                        f"(age={age_hours:.1f}h <= {float(FCS_RECENT_STATE_MAX_HOURS):.1f}h)."
                    )
                else:
                    if not ENABLE_DEEP_WARMUP:
                        logger.info(
                            f"⏭️ [Warmup] State is stale ({age_hours:.1f}h > {float(FCS_RECENT_STATE_MAX_HOURS):.1f}h) "
                            "and Deep Warmup is DISABLED. Starting cold."
                        )
                        return
                    logger.warning(
                        f"⚠️ State file is from previous day ({state_time.date()}) and exceeds recent window "
                        f"({age_hours:.1f}h > {float(FCS_RECENT_STATE_MAX_HOURS):.1f}h), starting fresh (Deep Warmup)."
                    )
                    svc._warmup_from_history(replay_features=True)
                    return
            is_fully_warmed = True
            for sym, sub_state in state.get('normalizers', {}).items():
                if sym in svc.normalizers:
                    svc.normalizers[sym].set_state(sub_state)
                    if svc.normalizers[sym].count < 500:
                        is_fully_warmed = False
            if not is_fully_warmed:
                if not ENABLE_DEEP_WARMUP:
                    logger.info("⏭️ [Warmup] State incomplete but Deep Warmup is DISABLED via config. Starting cold.")
                    return
                logger.warning("⚠️ State file is from today but INCOMPLETE (count < 500). Forcing Deep Warmup!")
                svc._warmup_from_history(replay_features=True)
            else:
                logger.info(f"♻️ Service State Restored from disk: {svc.state_file.name} (Fully Warmed). Loading base history...")
                svc._warmup_from_history(replay_features=False)
        except Exception as e:
            logger.error(f"❌ State Load Error: {e}")
            svc._warmup_from_history(replay_features=True)

    def get_dynamic_atm_iv(self, snap, price):
        if snap is None or len(snap) < 2:
            return 0.0, 0.0
        try:
            strikes = np.array([snap[0, 5], snap[2, 5], snap[4, 5]], dtype=np.float32)
            k_idx = np.argmin(np.abs(strikes - price))
            row_start = k_idx * 2
            p_iv = float(snap[row_start, 7])
            c_iv = float(snap[row_start + 1, 7])
            if price > 0:
                logger.debug(f"🎯 [ATM Pick] Price: {price:.2f} | Best Strike: {strikes[k_idx]:.2f} (Row {row_start}/{row_start+1}) | P_IV: {p_iv:.4f} | C_IV: {c_iv:.4f}")
            return c_iv, p_iv
        except Exception as e:
            logger.warning(f"⚠️ [ATM Pick Error]: {e}")
            return 0.0, 0.0

    def extract_semantic_atm_iv(self, buckets, contracts, spot_price):
        if buckets is None:
            return 0.0, 0.0
        try:
            arr = buckets if isinstance(buckets, np.ndarray) else np.array(buckets, dtype=np.float32)
            if arr.ndim != 2 or arr.shape[0] == 0:
                return 0.0, 0.0
            contracts = contracts.tolist() if isinstance(contracts, np.ndarray) else (contracts or [])
            call_iv = put_iv = 0.0
            call_diff = put_diff = float('inf')
            for i in range(arr.shape[0]):
                row = arr[i]
                if row.shape[0] <= 7:
                    continue
                contract_name = str(contracts[i]) if i < len(contracts) else ""
                is_put = 'P' in contract_name
                is_call = 'C' in contract_name
                if not is_put and not is_call:
                    is_put = i in [0, 1, 4]
                    is_call = i in [2, 3, 5]
                strike = float(row[5]) if row.shape[0] > 5 else 0.0
                iv = float(row[7])
                diff = abs(strike - float(spot_price))
                if is_call and diff < call_diff:
                    call_diff = diff
                    call_iv = iv
                if is_put and diff < put_diff:
                    put_diff = diff
                    put_iv = iv
            return call_iv, put_iv
        except Exception as e:
            logger.warning(f"⚠️ [Semantic ATM IV Error]: {e}")
            return 0.0, 0.0

    def extract_tagged_atm_iv(self, buckets):
        if buckets is None:
            return 0.0, 0.0
        try:
            arr = buckets if isinstance(buckets, np.ndarray) else np.array(buckets, dtype=np.float32)
            if arr.ndim != 2 or arr.shape[0] == 0:
                return 0.0, 0.0
            c_iv = float(arr[2, 7]) if arr.shape[0] > 2 and arr.shape[1] > 7 else 0.0
            p_iv = float(arr[0, 7]) if arr.shape[0] > 0 and arr.shape[1] > 7 else 0.0
            return c_iv, p_iv
        except Exception as e:
            logger.warning(f"⚠️ [Tagged ATM IV Error]: {e}")
            return 0.0, 0.0

    def merge_option_snapshot_with_greeks(self, raw_buckets, enriched_buckets):
        def _to_2d_array(val):
            if val is None:
                return None
            try:
                arr = val if isinstance(val, np.ndarray) else np.array(val, dtype=np.float32)
                if arr.ndim != 2 or arr.size == 0:
                    return None
                return arr.astype(np.float32, copy=False)
            except Exception:
                return None
        raw_arr = _to_2d_array(raw_buckets)
        enriched_arr = _to_2d_array(enriched_buckets)
        if raw_arr is None and enriched_arr is None:
            return []
        if raw_arr is None:
            return enriched_arr
        if enriched_arr is None:
            return raw_arr
        out_rows = max(raw_arr.shape[0], enriched_arr.shape[0])
        out_cols = max(raw_arr.shape[1], enriched_arr.shape[1])
        merged = np.zeros((out_rows, out_cols), dtype=np.float32)
        merged[:raw_arr.shape[0], :raw_arr.shape[1]] = raw_arr
        rows = min(out_rows, enriched_arr.shape[0])
        for col in (1, 2, 3, 4, 7):
            if col >= out_cols or col >= enriched_arr.shape[1]:
                continue
            src = enriched_arr[:rows, col]
            dst = merged[:rows, col]
            merged[:rows, col] = np.where(np.isfinite(src), src, dst)
        # 强约束：价格列统一由 bid/ask 重建为 mid，避免沿用成交价。
        if out_cols > 9:
            bid = merged[:rows, 8]
            ask = merged[:rows, 9]
            valid = np.isfinite(bid) & np.isfinite(ask) & (bid > 0.0) & (ask >= bid)
            merged[:rows, 0] = np.where(valid, (bid + ask) * 0.5, 0.0)
        return merged

    def is_option_snapshot_complete(self, buckets, contracts=None, min_iv=None):
        svc = self.service
        if buckets is None:
            return False
        try:
            min_iv = svc.option_gate_min_iv if min_iv is None else float(min_iv)
            arr = buckets if isinstance(buckets, np.ndarray) else np.array(buckets, dtype=np.float32)
            if arr.ndim != 2 or arr.shape[0] <= 2 or arr.shape[1] <= 9:
                return False
            c_iv, p_iv = float(arr[2, 7]), float(arr[0, 7])
            c_bid, c_ask = float(arr[2, 8]), float(arr[2, 9])
            p_bid, p_ask = float(arr[0, 8]), float(arr[0, 9])
            c_mid = 0.5 * (c_bid + c_ask) if (c_bid > 0.0 and c_ask >= c_bid) else 0.0
            p_mid = 0.5 * (p_bid + p_ask) if (p_bid > 0.0 and p_ask >= p_bid) else 0.0
            vals = [c_mid, p_mid, c_iv, p_iv, c_bid, c_ask, p_bid, p_ask]
            if not np.isfinite(vals).all():
                return False
            iv_ok = (c_iv > float(min_iv)) and (p_iv > float(min_iv))
            px_ok = (c_mid > 0.0) and (p_mid > 0.0)
            ba_ok = (c_bid > 0.0) and (p_bid > 0.0) and (c_ask >= c_bid) and (p_ask >= p_bid)
            contracts_ok = True if contracts is None else (len(contracts) > 2)
            return bool(iv_ok and px_ok and ba_ok and contracts_ok)
        except Exception:
            return False

    def update_option_frame_state(self, sym: str, payload: dict, ts: float):
        svc = self.service
        if sym not in svc.option_frame_state:
            svc.option_frame_state[sym] = {'minute_flags': {}, 'last_seq': None}
        state = svc.option_frame_state[sym]
        minute_ts = int(float(ts) // 60) * 60
        flags = state['minute_flags'].setdefault(minute_ts, {'has_integrity': False, 'has_complete': False, 'seq_ok': True})
        has_seq_key = 'seq' in payload
        has_complete_key = 'frame_complete' in payload
        if has_seq_key or has_complete_key:
            flags['has_integrity'] = True
        if has_seq_key:
            try:
                seq_val = int(payload.get('seq'))
                last_seq = state.get('last_seq')
                if last_seq is not None and seq_val <= int(last_seq):
                    flags['seq_ok'] = False
                state['last_seq'] = seq_val
            except Exception:
                flags['seq_ok'] = False
        if bool(payload.get('frame_complete', False)):
            flags['has_complete'] = True
        keep_keys = sorted(state['minute_flags'].keys())[-5:]
        state['minute_flags'] = {k: state['minute_flags'][k] for k in keep_keys}

    def is_option_frame_consistent(self, sym: str, gate_minute_ts: Optional[float]) -> bool:
        svc = self.service
        if not svc.option_gate_require_frame_consistency:
            return True
        if gate_minute_ts is None:
            return True
        state = svc.option_frame_state.get(sym, {})
        minute_flags = state.get('minute_flags', {})
        target_minute = int(float(gate_minute_ts) // 60) * 60
        flags = minute_flags.get(target_minute)
        if not flags:
            return True
        if not flags.get('has_integrity', False):
            return True
        if not flags.get('has_complete', False):
            return False
        if not flags.get('seq_ok', True):
            return False
        return True

    def update_option_gate_state(self, sym: str, snapshot_ok: bool, frame_ok: bool, is_new_minute: bool, gate_minute_ts: Optional[float] = None) -> bool:
        svc = self.service
        state = svc.option_gate_state.setdefault(sym, {'pass_streak': 0, 'fail_streak': 0, 'ready': False, 'grace_until_ts': None})
        prev_ready = bool(state.get('ready', False))
        current_gate_ts = float(gate_minute_ts) if gate_minute_ts is not None else None
        gate_ok = bool(snapshot_ok and frame_ok)
        in_grace = False
        if is_new_minute:
            if gate_ok:
                state['pass_streak'] = int(state.get('pass_streak', 0)) + 1
                state['fail_streak'] = 0
                state['grace_until_ts'] = None
                if not prev_ready and state['pass_streak'] >= svc.option_gate_min_pass:
                    state['ready'] = True
            else:
                state['fail_streak'] = int(state.get('fail_streak', 0)) + 1
                state['pass_streak'] = 0
                if prev_ready and svc.option_gate_grace_minutes > 0 and current_gate_ts is not None:
                    grace_until = state.get('grace_until_ts')
                    if grace_until is None:
                        grace_until = current_gate_ts + svc.option_gate_grace_minutes * 60.0
                        state['grace_until_ts'] = grace_until
                    in_grace = current_gate_ts <= float(grace_until)
                if prev_ready and (not in_grace) and state['fail_streak'] >= svc.option_gate_max_fail:
                    state['ready'] = False
            if bool(state.get('ready', False)) != prev_ready:
                logger.info(
                    f"🔁 [IV-Gate] {sym} state changed: ready={state['ready']} "
                    f"(pass={state['pass_streak']}, fail={state['fail_streak']}, grace_until={state.get('grace_until_ts')}, "
                    f"min_pass={svc.option_gate_min_pass}, max_fail={svc.option_gate_max_fail}, grace_m={svc.option_gate_grace_minutes})"
                )
        allow = bool(gate_ok and state.get('ready', False))
        if is_new_minute and (not allow) and bool(state.get('ready', False)) and (not gate_ok):
            if svc.option_gate_grace_minutes > 0 and current_gate_ts is not None:
                grace_until = state.get('grace_until_ts')
                if grace_until is not None and current_gate_ts <= float(grace_until):
                    allow = True
        # REALTIME_DRY 下，IV-Gate 只做质量告警，不应中断 Alpha 产出链路。
        run_mode = os.environ.get("RUN_MODE", "").strip().upper()
        if run_mode == "REALTIME_DRY" and (not allow):
            if getattr(svc, "_dry_gate_bypass_log_count", 0) < 30 and is_new_minute:
                logger.warning(
                    f"🧪 [IV-Gate Dry-Bypass] {sym}: gate blocked but bypassed in REALTIME_DRY "
                    f"(snapshot_ok={snapshot_ok}, frame_ok={frame_ok}, ready={state.get('ready', False)})"
                )
                svc._dry_gate_bypass_log_count = getattr(svc, "_dry_gate_bypass_log_count", 0) + 1
            allow = True
        if is_new_minute and not allow:
            logger.warning(
                f"⏭️ [IV-Gate] Skip {sym}: snapshot_ok={snapshot_ok}, frame_ok={frame_ok}, ready={state.get('ready', False)}, "
                f"pass={state.get('pass_streak', 0)}, fail={state.get('fail_streak', 0)}, grace_until={state.get('grace_until_ts')}"
            )
        return allow

    def publish_option_gate_metrics(self, gate_minute_ts: float, gate_audit: Dict[str, dict]):
        svc = self.service
        try:
            minute_ts = int(float(gate_minute_ts) // 60) * 60
            if svc._last_gate_metrics_minute_ts == minute_ts:
                return
            svc._last_gate_metrics_minute_ts = minute_ts
            if not gate_audit:
                return
            total = len(gate_audit)
            snapshot_ok_cnt = sum(1 for v in gate_audit.values() if v.get('snapshot_ok', False))
            frame_ok_cnt = sum(1 for v in gate_audit.values() if v.get('frame_ok', False))
            ready_symbols = sorted([s for s, v in gate_audit.items() if v.get('ready', False)])
            allowed_symbols = sorted([s for s, v in gate_audit.items() if v.get('allow', False)])
            failed_symbols = sorted([s for s, v in gate_audit.items() if not v.get('allow', False)])
            metric = {
                'ts': float(minute_ts),
                'dt': datetime.fromtimestamp(minute_ts, NY_TZ).strftime('%Y-%m-%d %H:%M:%S'),
                'total_symbols': total,
                'snapshot_ok_count': snapshot_ok_cnt,
                'frame_ok_count': frame_ok_cnt,
                'ready_count': len(ready_symbols),
                'allow_count': len(allowed_symbols),
                'pass_rate': float(len(allowed_symbols) / total) if total > 0 else 0.0,
                'ready_symbols': ready_symbols,
                'allow_symbols': allowed_symbols,
                'failed_symbols': failed_symbols
            }
            svc.r.hset("monitor:option_gate:1m", str(minute_ts), json.dumps(metric))
            svc.r.expire("monitor:option_gate:1m", 3600 * 6)
            svc.r.set("monitor:option_gate:last", json.dumps(metric), ex=3600 * 6)
            try:
                conn = svc._get_pg_conn()
                c = conn.cursor()
                c.execute("""
                    CREATE TABLE IF NOT EXISTS option_gate_metrics (
                        ts DOUBLE PRECISION PRIMARY KEY,
                        dt TEXT,
                        total_symbols INT,
                        snapshot_ok_count INT,
                        frame_ok_count INT,
                        ready_count INT,
                        allow_count INT,
                        pass_rate DOUBLE PRECISION,
                        ready_symbols TEXT,
                        allow_symbols TEXT,
                        failed_symbols TEXT
                    )
                """)
                c.execute("""
                    INSERT INTO option_gate_metrics
                    (ts, dt, total_symbols, snapshot_ok_count, frame_ok_count, ready_count, allow_count, pass_rate, ready_symbols, allow_symbols, failed_symbols)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (ts) DO UPDATE SET
                        dt = EXCLUDED.dt,
                        total_symbols = EXCLUDED.total_symbols,
                        snapshot_ok_count = EXCLUDED.snapshot_ok_count,
                        frame_ok_count = EXCLUDED.frame_ok_count,
                        ready_count = EXCLUDED.ready_count,
                        allow_count = EXCLUDED.allow_count,
                        pass_rate = EXCLUDED.pass_rate,
                        ready_symbols = EXCLUDED.ready_symbols,
                        allow_symbols = EXCLUDED.allow_symbols,
                        failed_symbols = EXCLUDED.failed_symbols
                """, (
                    metric['ts'], metric['dt'], metric['total_symbols'],
                    metric['snapshot_ok_count'], metric['frame_ok_count'],
                    metric['ready_count'], metric['allow_count'], metric['pass_rate'],
                    json.dumps(metric['ready_symbols']), json.dumps(metric['allow_symbols']),
                    json.dumps(metric['failed_symbols'])
                ))
                c.close()
            except Exception as pg_e:
                logger.warning(f"⚠️ [IV-Gate] PG metrics write failed: {pg_e}")
        except Exception as e:
            logger.warning(f"⚠️ [IV-Gate] metrics publish failed: {e}")

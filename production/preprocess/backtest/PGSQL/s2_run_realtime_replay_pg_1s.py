#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys

PARITY_MODE_BOOT = '--parity-mode' in sys.argv
if PARITY_MODE_BOOT:
    os.environ['RUN_MODE'] = 'BACKTEST'
    os.environ['RECALC_GREEKS'] = '1'
    os.environ['FORCE_HIGH_FREQ'] = '0'
    os.environ['REPLAY_1S_PARITY_MODE'] = '1'
else:
    os.environ['RUN_MODE'] = 'LIVEREPLAY'
    os.environ['RECALC_GREEKS'] = '1'
    os.environ.setdefault('REPLAY_1S_PARITY_MODE', '0')

import argparse
import asyncio
import copy
import datetime as dt
import json
import logging
import subprocess
import threading
import time
import uuid
import warnings
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd
import psycopg2
import pytz
import redis

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils import serialization_utils as ser
from signal_engine_v8 import SignalEngineV8
import signal_engine_v8
from execution_engine_v8 import ExecutionEngineV8
import execution_engine_v8
from mock_ibkr_historical import MockIBKRHistorical
from config import TARGET_SYMBOLS, PG_DB_URL
import data_persistence_service_v8_pg

from config import (
    REDIS_CFG, PROJECT_ROOT, FEATURE_SERVICE_STATE_FILE, STREAM_ORCH_SIGNAL,
    GROUP_FEATURE, GROUP_ORCH, GROUP_OMS, GROUP_PERSISTENCE,
    STREAM_FUSED_MARKET, HASH_OPTION_SNAPSHOT, STREAM_INFERENCE, STREAM_TRADE_LOG
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [PG_1S_INFERENCE] - %(message)s')
logger = logging.getLogger("PG1sBatchInference")
warnings.filterwarnings(
    "ignore",
    message="The default fill_method='pad' in Series.pct_change is deprecated",
    category=FutureWarning,
)


def partition_exists(cursor, table_name):
    cursor.execute(
        "SELECT EXISTS(SELECT 1 FROM information_schema.tables WHERE table_name=%s)",
        (table_name,),
    )
    return bool(cursor.fetchone()[0])


class BatchPostgresDriver1s:
    def __init__(self, target_dates, run_id, parity_mode=False):
        self.target_dates = list(target_dates)
        self.run_id = run_id
        self.parity_mode = parity_mode
        self.r = redis.Redis(**{k: v for k, v in REDIS_CFG.items() if k in ['host', 'port', 'db']})

    def _fetch_day_frames(self, date_str):
        part_b1 = f"market_bars_1s_{date_str}"
        part_o1 = f"option_snapshots_1s_{date_str}"
        part_b5 = f"market_bars_5m_{date_str}"
        part_o5 = f"option_snapshots_5m_{date_str}"

        conn = psycopg2.connect(PG_DB_URL)
        cur = conn.cursor()

        if not partition_exists(cur, part_b1) or not partition_exists(cur, part_o1):
            cur.close()
            conn.close()
            return None, None, None, None

        try:
            # if partition_exists(cur, f"alpha_logs_{date_str}"):
            #     cur.execute(f"TRUNCATE TABLE alpha_logs_{date_str}")
            # if partition_exists(cur, f"trade_logs_{date_str}"):
            #     cur.execute(f"TRUNCATE TABLE trade_logs_{date_str}")
            # if partition_exists(cur, f"trade_logs_backtest_{date_str}"):
            #     cur.execute(f"TRUNCATE TABLE trade_logs_backtest_{date_str}")
            conn.commit()
        except Exception:
            conn.rollback()

        df_bars_1s = pd.read_sql(f"SELECT symbol, ts, open, high, low, close, volume FROM {part_b1} ORDER BY ts ASC", conn)
        df_opts_1s = pd.read_sql(f"SELECT symbol, ts, buckets_json FROM {part_o1} ORDER BY ts ASC", conn)
        df_bars_5m = pd.read_sql(f"SELECT symbol, ts, open, high, low, close, volume FROM {part_b5} ORDER BY ts ASC", conn) if partition_exists(cur, part_b5) else pd.DataFrame()
        df_opts_5m = pd.read_sql(f"SELECT symbol, ts, buckets_json FROM {part_o5} ORDER BY ts ASC", conn) if partition_exists(cur, part_o5) else pd.DataFrame()

        cur.close()
        conn.close()
        return df_bars_1s, df_opts_1s, df_bars_5m, df_opts_5m

    @staticmethod
    def _to_map(df, type_key):
        mapping = defaultdict(dict)
        if df.empty:
            return {}
        df = df.copy()
        df['ts_aligned'] = df['ts'].astype(int)
        ts_vals = df['ts_aligned'].values
        sym_vals = df['symbol'].values

        if type_key == 'bars':
            o_vals = df['open'].values
            h_vals = df['high'].values
            l_vals = df['low'].values
            c_vals = df['close'].values
            v_vals = df['volume'].values
            for ts, sym, o, h, l, c, v in zip(ts_vals, sym_vals, o_vals, h_vals, l_vals, c_vals, v_vals):
                mapping[ts][sym] = {'open': o, 'high': h, 'low': l, 'close': c, 'volume': v}
        else:
            json_vals = df['buckets_json'].values
            for ts, sym, b_json in zip(ts_vals, sym_vals, json_vals):
                opt_data = b_json
                if isinstance(b_json, str):
                    try:
                        opt_data = json.loads(b_json)
                    except Exception:
                        opt_data = {}
                mapping[ts][sym] = opt_data
        return dict(mapping)

    def _build_day_stream(self, date_str, df_bars_1s, df_opts_1s, df_bars_5m, df_opts_5m):
        if df_bars_1s.empty:
            return []

        if self.parity_mode:
            rth_start_dt = pytz.timezone('America/New_York').localize(datetime.strptime(date_str + " 09:30:00", "%Y%m%d %H:%M:%S"))
            rth_start_ts = int(rth_start_dt.timestamp())
            df_bars_1s = df_bars_1s[df_bars_1s['ts'] >= rth_start_ts].copy()
            df_opts_1s = df_opts_1s[df_opts_1s['ts'] >= rth_start_ts].copy()

        map_b1 = self._to_map(df_bars_1s, 'bars')
        map_o1 = self._to_map(df_opts_1s, 'opts')
        map_b5 = self._to_map(df_bars_5m, 'bars')
        map_o5 = self._to_map(df_opts_5m, 'opts')

        all_ts = sorted(list(set(map_b1.keys()) | set(map_o1.keys()) | set(map_b5.keys()) | set(map_o5.keys())))
        last_known_payloads = {
            sym: {'ts': 0, 'symbol': sym, 'stock': {'open': 0, 'high': 0, 'low': 0, 'close': 0, 'volume': 0}, 'option_buckets': [], 'option_contracts': []}
            for sym in TARGET_SYMBOLS
        }
        last_5m_state = {}
        global_seq = 0
        frames = []

        for ts_val in all_ts:
            frame_complete = (int(ts_val) % 60 == 59)
            frame_id = str(int(ts_val))
            b1_ts = map_b1.get(ts_val, {})
            o1_ts = map_o1.get(ts_val, {})
            b5_ts = map_b5.get(ts_val, {})
            o5_ts = map_o5.get(ts_val, {})
            batch_payloads = []
            hset_mapping = {}

            for sym in TARGET_SYMBOLS:
                payload = last_known_payloads[sym]
                payload['ts'] = ts_val
                global_seq += 1
                payload['frame_id'] = frame_id
                payload['seq'] = global_seq
                payload['frame_complete'] = frame_complete

                if sym in b1_ts:
                    payload['stock'] = b1_ts[sym]
                elif payload['stock']['close'] > 0:
                    prev_close = float(payload['stock']['close'])
                    payload['stock'] = {'open': prev_close, 'high': prev_close, 'low': prev_close, 'close': prev_close, 'volume': 0.0}

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

                if sym in b5_ts or sym in o5_ts:
                    if sym not in last_5m_state:
                        last_5m_state[sym] = {}
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

                last_known_payloads[sym] = payload
                batch_payloads.append(payload.copy())

            frames.append((ts_val, batch_payloads, hset_mapping))
        return frames

    def run(self):
        logger.info(f"🚀 [Driver] Starting PG 1s replay for {len(self.target_dates)} dates...")
        for idx, date_str in enumerate(self.target_dates, 1):
            logger.info(f"📂 [{idx}/{len(self.target_dates)}] Processing PG date: {date_str}")
            if idx == 1:
                logger.info(f"🧊 [Cold Start] {date_str} 之前无历史分区依赖，引擎从零开始积累特征。")
            else:
                logger.info(f"🔥 [Hot Start] 引擎内存无缝接力，自动包含前一日 ({self.target_dates[idx-2]}) 尾盘特征。")

            df_b1, df_o1, df_b5, df_o5 = self._fetch_day_frames(date_str)
            if df_b1 is None or df_b1.empty:
                logger.warning(f"⚠️ No PG 1s data found for {date_str}, skipping.")
                continue

            frames = self._build_day_stream(date_str, df_b1, df_o1, df_b5, df_o5)
            for ts_val, batch_payloads, hset_mapping in frames:
                self.r.set("replay:current_ts", str(ts_val))
                if hset_mapping:
                    self.r.hset(HASH_OPTION_SNAPSHOT, mapping=hset_mapping)
                if batch_payloads:
                    self.r.xadd(STREAM_FUSED_MARKET, {'batch': ser.pack(batch_payloads)})

                timeout = 0
                while True:
                    ack_feat, ack_orch = self.r.mget("sync:feature_calc_done", "sync:orch_done")
                    feat_ts = float(ack_feat) if ack_feat else 0.0
                    orch_ts = float(ack_orch) if ack_orch else 0.0
                    if feat_ts >= ts_val and orch_ts >= ts_val:
                        break
                    time.sleep(0.0005)
                    timeout += 1
                    if timeout > 60000:
                        logger.warning(f"⚠️ [STALL] Sync Timeout at ts={ts_val}. Feat:{feat_ts} Orch:{orch_ts}")
                        break

    async def run_turbo(
        self,
        feat_svc,
        signal_svc,
        *,
        write_market: bool = False,
        write_features: bool = False,
        write_option_1m: bool = True,
        write_redis_alpha: bool = False,
    ):
        logger.info("🔥 [Turbo Mode] Starting PG 1s in-process replay...")
        logger.info(
            "⚡ [Turbo Alpha-Only] write_market=%s | write_features=%s | write_option_1m=%s | write_redis_alpha=%s",
            write_market,
            write_features,
            write_option_1m,
            write_redis_alpha,
        )
        NY_TZ = pytz.timezone('America/New_York')
        if not self.target_dates:
            return

        persist_cfg = copy.deepcopy(REDIS_CFG)
        persist_cfg['pg_group'] = GROUP_PERSISTENCE
        data_persistence_service_v8_pg.REDIS_CFG = persist_cfg
        persist_svc = data_persistence_service_v8_pg.DataPersistenceServicePG(start_date=self.target_dates[0])
        signal_svc.turbo_mode = True
        original_xadd = signal_svc.r.xadd

        def intercept_redis_xadd(stream_name, mapping, *args, **kwargs):
            s_name = stream_name.decode('utf-8') if isinstance(stream_name, bytes) else stream_name
            if s_name == STREAM_TRADE_LOG:
                data_str = mapping.get(b'data') or mapping.get('data')
                if data_str:
                    try:
                        payload = ser.unpack(data_str)
                        if payload.get('action') == 'ALPHA':
                            persist_svc.alpha_buffer.append((
                                payload['ts'], payload['symbol'],
                                float(payload.get('alpha', 0)),
                                float(payload.get('iv', 0)),
                                float(payload.get('price', 0)),
                                float(payload.get('vol_z', 0)),
                            ))
                    except Exception:
                        pass
                if not write_redis_alpha:
                    return None
            try:
                original_xadd(stream_name, mapping, *args, **kwargs)
            except Exception:
                pass

        signal_svc.r.xadd = intercept_redis_xadd
        is_first_day = True
        for idx, date_str in enumerate(self.target_dates, 1):
            logger.info(f"📂 [Turbo {idx}/{len(self.target_dates)}] Processing PG date: {date_str}")
            if is_first_day:
                if hasattr(feat_svc, 'reset_internal_memory'):
                    feat_svc.reset_internal_memory()
                logger.info("❄️ [Turbo Cold Start] First replay day reinitializes FCS with REPLAY_START_TS-bounded PostgreSQL warmup.")
            else:
                logger.info(f"🔥 [Turbo Hot Start] Reusing in-memory rolling state carried from previous day {self.target_dates[idx-2]}.")

            df_b1, df_o1, df_b5, df_o5 = self._fetch_day_frames(date_str)
            if df_b1 is None or df_b1.empty:
                logger.warning(f"⚠️ No PG 1s data found for {date_str}, skipping.")
                continue

            frames = self._build_day_stream(date_str, df_b1, df_o1, df_b5, df_o5)
            first_full_min = 0
            if frames:
                # Match the SQLite 1s turbo launcher: the first available frame may
                # prime state, and model inference starts from the next full minute.
                first_full_min = ((int(frames[0][0]) + 59) // 60) * 60
                logger.info(f"🎯 [Parity] Alignment: Data starts at {datetime.fromtimestamp(frames[0][0], NY_TZ)}, truncating signals to {datetime.fromtimestamp(first_full_min, NY_TZ)}")

            for ts_val, batch_payloads, _ in frames:
                await feat_svc.process_market_data(batch_payloads)
                persist_svc._check_date_rotation(ts_val)
                if write_market:
                    for payload in batch_payloads:
                        persist_svc.process_market_data(payload)

                if ts_val < first_full_min:
                    continue

                feat_payload = await feat_svc.run_compute_cycle(ts_from_payload=ts_val, return_payload=True)
                if feat_payload:
                    if self.parity_mode and not bool(feat_payload.get('is_new_minute', False)):
                        continue
                    if write_features or write_option_1m:
                        persist_svc.process_feature_data(
                            feat_payload,
                            write_feature_logs=write_features,
                            write_option_snapshots=write_option_1m,
                        )
                    await signal_svc.process_batch(feat_payload)

                if ts_val % 60 == 0:
                    persist_svc.flush()

            persist_svc.flush()
            is_first_day = False


async def main():
    parser = argparse.ArgumentParser(description="Batch Alpha Inference Factory (PostgreSQL 1-Second Edition)")
    parser.add_argument('--start-date', type=str, default="20260102")
    parser.add_argument('--end-date', type=str, default=None)
    parser.add_argument('--skip-warmup', action='store_true')
    parser.add_argument('--enable-oms', action='store_true')
    parser.add_argument('--turbo', action='store_true')
    parser.add_argument('--parity-mode', action='store_true')
    parser.add_argument('--turbo-write-market', action='store_true', help='Turbo 模式下仍回写 market_bars/option_snapshots；默认关闭以避免 PG 重复写入。')
    parser.add_argument('--turbo-write-features', action='store_true', help='Turbo 模式下仍写 feature_logs；默认关闭，仅生成 alpha_logs。')
    parser.add_argument('--turbo-no-option-overwrite', action='store_true', help='Turbo 模式下关闭 option_snapshots_1m 覆写；默认开启用于修复/校验 Greeks。')
    parser.add_argument('--turbo-write-redis-alpha', action='store_true', help='Turbo 模式下仍把 ALPHA 写入 Redis Stream；默认关闭，仅截获后写 PG。')
    args = parser.parse_args()

    if args.parity_mode:
        os.environ['RUN_MODE'] = 'BACKTEST'
        os.environ['RECALC_GREEKS'] = '1'
        os.environ['FORCE_HIGH_FREQ'] = '0'
        os.environ['REPLAY_1S_PARITY_MODE'] = '1'

    conn_pg = psycopg2.connect(PG_DB_URL)
    cur_pg = conn_pg.cursor()
    cur_pg.execute("SELECT table_name FROM information_schema.tables WHERE table_name LIKE 'market_bars_1s_20%'")
    rows = cur_pg.fetchall()
    all_dates = sorted([r[0].split('_')[-1] for r in rows])
    end_date = args.end_date or args.start_date
    target_dates = [d for d in all_dates if args.start_date <= d <= end_date]
    cur_pg.close()
    conn_pg.close()

    if not target_dates:
        logger.error(f"❌ No valid PG 1s partitions found between {args.start_date} and {end_date}.")
        return

    logger.info(f"🔍 Found {len(target_dates)} valid PG 1s dates: {target_dates}")

    r = redis.Redis(**{k: v for k, v in REDIS_CFG.items() if k in ['host', 'port', 'db']})
    run_id = str(uuid.uuid4())[:8]
    logger.info(f"🆔 Generated PG 1s Replay RUN_ID: {run_id}")

    for stream in [STREAM_FUSED_MARKET, STREAM_INFERENCE, STREAM_ORCH_SIGNAL, STREAM_TRADE_LOG, HASH_OPTION_SNAPSHOT, f"replay:status:{run_id}"]:
        r.delete(stream)

    streams_and_groups = {
        STREAM_FUSED_MARKET: [GROUP_FEATURE, GROUP_ORCH, GROUP_OMS, GROUP_PERSISTENCE],
        STREAM_INFERENCE: [GROUP_ORCH, GROUP_PERSISTENCE],
        STREAM_ORCH_SIGNAL: [GROUP_OMS, GROUP_PERSISTENCE],
        STREAM_TRADE_LOG: [GROUP_PERSISTENCE],
    }
    for s_key, groups in streams_and_groups.items():
        try:
            r.xadd(s_key, {'init': '1'})
        except Exception:
            pass
        for g_name in groups:
            try:
                r.xgroup_create(s_key, g_name, id='0', mkstream=True)
            except Exception:
                pass

    NY_TZ = pytz.timezone('America/New_York')
    actual_start_date = target_dates[0]
    target_dt = dt.datetime.strptime(actual_start_date, "%Y%m%d")
    target_dt = NY_TZ.localize(target_dt.replace(hour=9, minute=30, second=0))
    replay_start_ts = target_dt.timestamp()
    r.set("replay:current_ts", str(replay_start_ts))

    state_file = PROJECT_ROOT / FEATURE_SERVICE_STATE_FILE
    if state_file.exists():
        state_file.unlink()

    env = os.environ.copy()
    env['REPLAY_START_TS'] = str(replay_start_ts)
    os.environ['REPLAY_START_TS'] = str(replay_start_ts)
    if args.skip_warmup:
        env['SKIP_DEEP_WARMUP'] = '1'
        os.environ['SKIP_DEEP_WARMUP'] = '1'

    feature_process = None
    try:
        if args.turbo:
            logger.info("⚡ [Turbo Mode] Skipping Feature Compute Service subprocess; in-process FCS will warm up from PostgreSQL with replay clock.")
        else:
            logger.info("🚀 Starting Feature Compute Service (Subprocess)...")
            baseline_dir = PROJECT_ROOT / "baseline"
            fcs_entry_candidates = [
                baseline_dir / "DAO" / "feature_compute_service_v8.py",
                baseline_dir / "feature_compute_service_v8.py",
            ]
            fcs_entry = next((p for p in fcs_entry_candidates if p.exists()), None)
            if fcs_entry is None:
                logger.error(
                    "❌ Feature Compute Service entry not found. checked=%s",
                    [str(p) for p in fcs_entry_candidates],
                )
                return
            feature_process = subprocess.Popen(
                [sys.executable, str(fcs_entry)],
                cwd=str(baseline_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
            )

            ready_event = threading.Event()

            def monitor_stdout():
                for line in iter(feature_process.stdout.readline, ''):
                    sys.stdout.write(f"[FEAT] {line}")
                    sys.stdout.flush()
                    if ("Complete" in line and "Warmup" in line) or "Deep Warmup Complete" in line or "Cold start" in line or "Skipped" in line:
                        ready_event.set()

            threading.Thread(target=monitor_stdout, daemon=True).start()
            if not ready_event.wait(timeout=60):
                rc = feature_process.poll()
                if rc is not None:
                    logger.error(f"❌ Feature service exited during warmup. exit_code={rc}")
                logger.error("❌ Warmup Timeout! Feature service failed to initialize.")
                return

        class ReplayDatetime(dt.datetime):
            @classmethod
            def now(cls, tz=None):
                ts_str = r.get("replay:current_ts")
                if ts_str:
                    return dt.datetime.fromtimestamp(float(ts_str), tz=tz)
                return super().now(tz)

        def replay_time():
            ts_str = r.get("replay:current_ts")
            if ts_str:
                return float(ts_str)
            return time.time()

        signal_engine_v8.datetime = ReplayDatetime
        signal_engine_v8.time.time = replay_time
        execution_engine_v8.datetime = ReplayDatetime
        execution_engine_v8.time.time = replay_time
        data_persistence_service_v8_pg.datetime = ReplayDatetime
        data_persistence_service_v8_pg.time.time = replay_time

        orch_cfg = copy.deepcopy(REDIS_CFG)
        orch_cfg['group'] = GROUP_ORCH
        orch_cfg['input_stream'] = STREAM_INFERENCE
        signal_engine_v8.REDIS_CFG = orch_cfg

        
        config_paths = {
            'fast': str("/home/kingfang007/notebook/train/fast_feature.json"),
            'slow': str("/home/kingfang007/notebook/train/slow_feature.json")
        }
        
        model_paths = {
            'slow': str(PROJECT_ROOT / "checkpoints_advanced_alpha/advanced_alpha_best.pth"),
            'fast': str(PROJECT_ROOT / "checkpoints_advanced_alpha/fast_final_best.pth")
        }


        driver = BatchPostgresDriver1s(target_dates=target_dates, run_id=run_id, parity_mode=args.parity_mode)
        if args.turbo:
            from feature_compute_service_v8 import FeatureComputeService

            feat_cfg = copy.deepcopy(REDIS_CFG)
            feat_cfg['input_stream'] = STREAM_FUSED_MARKET
            feat_cfg['output_stream'] = STREAM_INFERENCE
            feat_svc = FeatureComputeService(feat_cfg, TARGET_SYMBOLS, config_paths)
            signal_svc = SignalEngineV8(TARGET_SYMBOLS, mode='backtest', config_paths=config_paths, model_paths=model_paths)
            signal_svc.only_log_alpha = True
            await driver.run_turbo(
                feat_svc,
                signal_svc,
                write_market=args.turbo_write_market,
                write_features=args.turbo_write_features,
                write_option_1m=(not args.turbo_no_option_overwrite),
                write_redis_alpha=args.turbo_write_redis_alpha,
            )
            logger.info("🏁 [Turbo Mode] PG 1s Replay Completed.")
            return

        persist_cfg = copy.deepcopy(REDIS_CFG)
        persist_cfg['group'] = GROUP_PERSISTENCE  # Fallback
        persist_cfg['pg_group'] = GROUP_PERSISTENCE
        persist_cfg['consumer'] = 'pg_writer_1s'
        data_persistence_service_v8_pg.REDIS_CFG = persist_cfg
        persistence_svc = data_persistence_service_v8_pg.DataPersistenceServicePG(start_date=target_dates[0])
        threading.Thread(target=persistence_svc.run, daemon=True).start()

        signal_engine = SignalEngineV8(symbols=TARGET_SYMBOLS, mode='backtest', config_paths=config_paths, model_paths=model_paths)
        signal_engine.only_log_alpha = not args.enable_oms
        signal_task = asyncio.create_task(signal_engine.run())

        oms = None
        oms_task = None
        if args.enable_oms:
            mock_ibkr = MockIBKRHistorical()
            await mock_ibkr.connect()
            oms = ExecutionEngineV8(symbols=TARGET_SYMBOLS, mode='backtest')
            execution_engine_v8.REDIS_CFG = {'input_stream': STREAM_ORCH_SIGNAL, 'group': GROUP_OMS, **REDIS_CFG}
            oms.ibkr = mock_ibkr
            oms.mock_cash = mock_ibkr.initial_capital
            oms_task = asyncio.create_task(oms.run())

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, driver.run)
        await asyncio.sleep(2)
        signal_task.cancel()
        if oms_task:
            oms_task.cancel()

    finally:
        if feature_process:
            try:
                feature_process.terminate()
            except Exception:
                pass


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass

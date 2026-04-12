#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PostgreSQL 1s event-driven replay runner.

This runner keeps alpha generation and trade execution on the same logical
clock. It is intended for production-like system validation:

    PG market partitions -> FCS -> SignalEngineV8 -> ExecutionEngineV8 -> PG logs

It deliberately does not use the Redis async launcher path, because the goal is
to avoid fake desynchronization caused by an overly fast pitcher.
"""

import os
import sys

os.environ.setdefault("RUN_MODE", "LIVEREPLAY")
os.environ.setdefault("RECALC_GREEKS", "1")
os.environ.setdefault("REPLAY_1S_PARITY_MODE", "0")
os.environ.setdefault("EVENT_DRIVEN_REPLAY", "1")

import argparse
import asyncio
import copy
import datetime as dt
import json
import logging
import time
import uuid
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
import pytz
import redis
from tqdm import tqdm

MY_DIR = Path(__file__).resolve().parent
ROOT_DIR = MY_DIR.parents[3]
PRODUCTION_DIR = ROOT_DIR / "production"
BASELINE_DIR = ROOT_DIR / "production" / "baseline"
DAO_DIR = BASELINE_DIR / "DAO"
HISTORY_REPLAY_DIR = ROOT_DIR / "production" / "history_replay"
MODEL_DIR = ROOT_DIR / "production" / "model"

for p in [str(PRODUCTION_DIR), str(BASELINE_DIR), str(DAO_DIR), str(HISTORY_REPLAY_DIR), str(MODEL_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from utils import serialization_utils as ser
from config import (
    PG_DB_URL,
    PROJECT_ROOT,
    REDIS_CFG,
    STREAM_TRADE_LOG,
    TARGET_SYMBOLS,
)
from feature_compute_service_v8 import FeatureComputeService
import feature_compute_service_v8
from signal_engine_v8 import SignalEngineV8
import signal_engine_v8
from execution_engine_v8 import ExecutionEngineV8
import execution_engine_v8
import data_persistence_service_v8_pg
import mock_ibkr_historical_1s
from mock_ibkr_historical_1s import MockIBKRHistorical


logging.basicConfig(level=logging.INFO, format="%(asctime)s - [EVENT_PG_1S] - %(message)s")
logger = logging.getLogger("EventDrivenPG1S")


def partition_exists(cursor, table_name: str) -> bool:
    cursor.execute(
        "SELECT EXISTS(SELECT 1 FROM information_schema.tables WHERE table_name=%s)",
        (table_name,),
    )
    return bool(cursor.fetchone()[0])


def list_pg_1s_dates(start_date: str, end_date: str) -> list[str]:
    conn = psycopg2.connect(PG_DB_URL)
    cur = conn.cursor()
    cur.execute("SELECT table_name FROM information_schema.tables WHERE table_name LIKE 'market_bars_1s_20%'")
    rows = cur.fetchall()
    cur.close()
    conn.close()
    all_dates = sorted([r[0].split("_")[-1] for r in rows])
    return [d for d in all_dates if start_date <= d <= end_date]


def truncate_output_partitions(dates: list[str]):
    tables = ["alpha_logs", "trade_logs", "trade_logs_backtest", "feature_logs"]
    conn = psycopg2.connect(PG_DB_URL)
    cur = conn.cursor()
    try:
        for date_str in dates:
            for base in tables:
                part = f"{base}_{date_str}"
                if partition_exists(cur, part):
                    logger.warning(f"🧹 Truncating PG output partition: {part}")
                    cur.execute(f"TRUNCATE TABLE {part}")
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()
        conn.close()


def fetch_day_frames(date_str: str):
    part_b1 = f"market_bars_1s_{date_str}"
    part_o1 = f"option_snapshots_1s_{date_str}"
    part_b5 = f"market_bars_5m_{date_str}"
    part_o5 = f"option_snapshots_5m_{date_str}"

    conn = psycopg2.connect(PG_DB_URL)
    cur = conn.cursor()
    try:
        if not partition_exists(cur, part_b1) or not partition_exists(cur, part_o1):
            return None, None, None, None

        df_bars_1s = pd.read_sql(
            f"SELECT symbol, ts, open, high, low, close, volume FROM {part_b1} ORDER BY ts ASC",
            conn,
        )
        df_opts_1s = pd.read_sql(
            f"SELECT symbol, ts, buckets_json FROM {part_o1} ORDER BY ts ASC",
            conn,
        )
        df_bars_5m = (
            pd.read_sql(f"SELECT symbol, ts, open, high, low, close, volume FROM {part_b5} ORDER BY ts ASC", conn)
            if partition_exists(cur, part_b5)
            else pd.DataFrame()
        )
        df_opts_5m = (
            pd.read_sql(f"SELECT symbol, ts, buckets_json FROM {part_o5} ORDER BY ts ASC", conn)
            if partition_exists(cur, part_o5)
            else pd.DataFrame()
        )
        return df_bars_1s, df_opts_1s, df_bars_5m, df_opts_5m
    finally:
        cur.close()
        conn.close()


def to_map(df: pd.DataFrame, type_key: str) -> dict:
    mapping = defaultdict(dict)
    if df is None or df.empty:
        return {}

    df = df.copy()
    df["ts_aligned"] = df["ts"].astype(int)
    ts_vals = df["ts_aligned"].values
    sym_vals = df["symbol"].values

    if type_key == "bars":
        for ts, sym, o, h, l, c, v in zip(
            ts_vals,
            sym_vals,
            df["open"].values,
            df["high"].values,
            df["low"].values,
            df["close"].values,
            df["volume"].values,
        ):
            mapping[ts][sym] = {"open": o, "high": h, "low": l, "close": c, "volume": v}
    else:
        for ts, sym, b_json in zip(ts_vals, sym_vals, df["buckets_json"].values):
            opt_data = b_json
            if isinstance(opt_data, str):
                try:
                    opt_data = json.loads(opt_data)
                except Exception:
                    opt_data = {}
            mapping[ts][sym] = opt_data

    return dict(mapping)


def build_day_stream(date_str: str, df_bars_1s, df_opts_1s, df_bars_5m, df_opts_5m, symbols: list[str]):
    if df_bars_1s is None or df_bars_1s.empty:
        return []

    ny_tz = pytz.timezone("America/New_York")
    rth_start_dt = ny_tz.localize(datetime.strptime(date_str + " 09:30:00", "%Y%m%d %H:%M:%S"))
    rth_start_ts = int(rth_start_dt.timestamp())
    df_bars_1s = df_bars_1s[df_bars_1s["ts"] >= rth_start_ts].copy()
    df_opts_1s = df_opts_1s[df_opts_1s["ts"] >= rth_start_ts].copy()

    map_b1 = to_map(df_bars_1s, "bars")
    map_o1 = to_map(df_opts_1s, "opts")
    map_b5 = to_map(df_bars_5m, "bars")
    map_o5 = to_map(df_opts_5m, "opts")

    all_ts = sorted(list(set(map_b1.keys()) | set(map_o1.keys()) | set(map_b5.keys()) | set(map_o5.keys())))
    last_known_payloads = {
        sym: {
            "ts": 0,
            "symbol": sym,
            "stock": {"open": 0, "high": 0, "low": 0, "close": 0, "volume": 0},
            "option_buckets": [],
            "option_contracts": [],
        }
        for sym in symbols
    }
    last_5m_state = {}
    frames = []
    global_seq = 0

    for ts_val in all_ts:
        frame_complete = int(ts_val) % 60 == 59
        frame_id = str(int(ts_val))
        b1_ts = map_b1.get(ts_val, {})
        o1_ts = map_o1.get(ts_val, {})
        b5_ts = map_b5.get(ts_val, {})
        o5_ts = map_o5.get(ts_val, {})
        batch_payloads = []

        for sym in symbols:
            payload = copy.deepcopy(last_known_payloads[sym])
            payload["ts"] = float(ts_val)
            payload["frame_id"] = frame_id
            payload["frame_complete"] = frame_complete
            global_seq += 1
            payload["seq"] = global_seq

            if sym in b1_ts:
                payload["stock"] = b1_ts[sym]
            elif payload["stock"]["close"] > 0:
                prev_close = float(payload["stock"]["close"])
                payload["stock"] = {
                    "open": prev_close,
                    "high": prev_close,
                    "low": prev_close,
                    "close": prev_close,
                    "volume": 0.0,
                }

            if sym in o1_ts:
                opt_data = o1_ts[sym]
                if isinstance(opt_data, dict):
                    payload["option_buckets"] = opt_data.get("buckets", [])
                    payload["option_contracts"] = opt_data.get("contracts", [])
                else:
                    payload["option_buckets"] = opt_data
                    payload["option_contracts"] = []

            if sym in b5_ts or sym in o5_ts:
                if sym not in last_5m_state:
                    last_5m_state[sym] = {}
                if sym in b5_ts:
                    last_5m_state[sym]["stock_5m"] = b5_ts[sym]
                    last_5m_state[sym]["stock_5m_ts"] = int(float(ts_val)) // 60 * 60
                if sym in o5_ts:
                    last_5m_state[sym]["option_5m_ts"] = int(float(ts_val)) // 60 * 60
                    opt_data_5m = o5_ts[sym]
                    if isinstance(opt_data_5m, dict):
                        last_5m_state[sym]["option_buckets_5m"] = opt_data_5m.get("buckets", [])
                        last_5m_state[sym]["option_contracts_5m"] = opt_data_5m.get("contracts", [])
                    else:
                        last_5m_state[sym]["option_buckets_5m"] = opt_data_5m
                        last_5m_state[sym]["option_contracts_5m"] = []

            if sym in last_5m_state:
                payload.update(last_5m_state[sym])

            last_known_payloads[sym] = payload
            batch_payloads.append(payload)

        frames.append((float(ts_val), batch_payloads))

    return frames


def add_feed_aliases(payload: dict) -> dict:
    aliases = {
        "cheat_call": "feed_call_price",
        "cheat_put": "feed_put_price",
        "cheat_call_bid": "feed_call_bid",
        "cheat_call_ask": "feed_call_ask",
        "cheat_put_bid": "feed_put_bid",
        "cheat_put_ask": "feed_put_ask",
        "cheat_call_iv": "feed_call_iv",
        "cheat_put_iv": "feed_put_iv",
        "cheat_call_k": "feed_call_k",
        "cheat_put_k": "feed_put_k",
    }
    for src, dst in aliases.items():
        if src in payload and dst not in payload:
            payload[dst] = payload[src]

    n = len(payload.get("symbols", []))
    payload.setdefault("feed_call_bid_size", np.full(n, 100.0, dtype=np.float32))
    payload.setdefault("feed_call_ask_size", np.full(n, 100.0, dtype=np.float32))
    payload.setdefault("feed_put_bid_size", np.full(n, 100.0, dtype=np.float32))
    payload.setdefault("feed_put_ask_size", np.full(n, 100.0, dtype=np.float32))
    payload.setdefault("feed_call_vol", np.ones(n, dtype=np.float32))
    payload.setdefault("feed_put_vol", np.ones(n, dtype=np.float32))
    payload.setdefault("feed_call_id", [""] * n)
    payload.setdefault("feed_put_id", [""] * n)
    return payload


class ReplayClock:
    def __init__(self):
        self.current_ts = time.time()

    def time(self):
        return float(self.current_ts or time.time())


def install_replay_clock(clock: ReplayClock):
    class ReplayDatetime(dt.datetime):
        @classmethod
        def now(cls, tz=None):
            return dt.datetime.fromtimestamp(clock.time(), tz=tz)

    signal_engine_v8.time.time = clock.time
    signal_engine_v8.datetime = ReplayDatetime
    execution_engine_v8.time.time = clock.time
    execution_engine_v8.datetime = ReplayDatetime
    feature_compute_service_v8.time.time = clock.time
    feature_compute_service_v8.datetime = ReplayDatetime
    data_persistence_service_v8_pg.time.time = clock.time
    data_persistence_service_v8_pg.datetime = ReplayDatetime


def patch_trade_log_capture(engine, persist_svc):
    original_xadd = engine.r.xadd

    def intercept_xadd(stream_name, mapping, *args, **kwargs):
        s_name = stream_name.decode("utf-8") if isinstance(stream_name, bytes) else str(stream_name)
        if s_name == STREAM_TRADE_LOG:
            data_blob = mapping.get(b"data") or mapping.get("data")
            if data_blob:
                try:
                    payload = ser.unpack(data_blob)
                    action = payload.get("action", "")
                    if action == "ALPHA":
                        persist_svc.alpha_buffer.append(
                            (
                                payload["ts"],
                                payload["symbol"],
                                float(payload.get("alpha", 0.0)),
                                float(payload.get("iv", 0.0)),
                                float(payload.get("price", 0.0)),
                                float(payload.get("vol_z", 0.0)),
                            )
                        )
                    else:
                        mode = payload.get("mode", "BACKTEST")
                        persist_svc.trade_buffer.append(
                            (
                                payload["ts"],
                                payload["symbol"],
                                action,
                                float(payload.get("qty", 0.0)),
                                float(payload.get("price", 0.0)),
                                json.dumps(payload),
                                mode,
                            )
                        )
                except Exception as e:
                    logger.warning(f"⚠️ Failed to capture trade log payload: {e}")
        try:
            return original_xadd(stream_name, mapping, *args, **kwargs)
        except Exception:
            return None

    engine.r.xadd = intercept_xadd


async def drain_signal_queue(shared_signal_queue, exec_engine):
    while not shared_signal_queue.empty():
        try:
            sig_payload = shared_signal_queue.get_nowait()
            await exec_engine.process_trade_signal(sig_payload)
            shared_signal_queue.task_done()
        except Exception as e:
            logger.warning(f"⚠️ Signal queue drain error: {e}")
            break


async def run_event_replay(args):
    end_date = args.end_date or args.start_date
    target_dates = list_pg_1s_dates(args.start_date, end_date)
    if not target_dates:
        logger.error(f"❌ No PG 1s partitions found between {args.start_date} and {end_date}.")
        return

    symbols = [args.symbol] if args.symbol else TARGET_SYMBOLS
    run_id = str(uuid.uuid4())[:8]
    logger.info(f"🆔 Event-driven PG 1s RUN_ID: {run_id}")
    logger.info(f"📅 Target dates: {target_dates}")
    logger.info(f"📌 Symbols: {symbols if args.symbol else 'TARGET_SYMBOLS'}")
    logger.info(f"🔍 Mock IBKR 1s module: {mock_ibkr_historical_1s.__file__}")

    if args.truncate_output:
        truncate_output_partitions(target_dates)

    ny_tz = pytz.timezone("America/New_York")
    start_dt = ny_tz.localize(dt.datetime.strptime(target_dates[0] + " 09:30:00", "%Y%m%d %H:%M:%S"))
    replay_start_ts = float(start_dt.timestamp())
    os.environ["REPLAY_START_TS"] = str(replay_start_ts)
    if args.skip_warmup:
        os.environ["SKIP_DEEP_WARMUP"] = "1"

    clock = ReplayClock()
    clock.current_ts = replay_start_ts
    install_replay_clock(clock)

    r = redis.Redis(**{k: v for k, v in REDIS_CFG.items() if k in ["host", "port", "db"]})
    r.set("replay:current_ts", str(replay_start_ts))

    v8_root = BASELINE_DIR
    config_paths = {
        "fast": str(v8_root / "daily_backtest" / "fast_feature.json"),
        "slow": str(v8_root / "daily_backtest" / "slow_feature.json"),
    }
    model_paths = {
        "slow": str(PROJECT_ROOT / "checkpoints_advanced_alpha/advanced_alpha_best.pth"),
        "fast": str(PROJECT_ROOT / "checkpoints_advanced_alpha/fast_final_best.pth"),
    }

    feat_cfg = copy.deepcopy(REDIS_CFG)
    feat_svc = FeatureComputeService(feat_cfg, symbols, config_paths)

    signal_engine = SignalEngineV8(symbols=symbols, mode="backtest", config_paths=config_paths, model_paths=model_paths)
    signal_engine.only_log_alpha = False
    shared_signal_queue = asyncio.Queue()
    signal_engine.signal_queue = shared_signal_queue
    signal_engine.use_shared_mem = True

    exec_engine = ExecutionEngineV8(
        symbols=symbols,
        mode="backtest",
        shared_states=signal_engine.states,
        signal_queue=shared_signal_queue,
    )
    exec_engine.strategy.cfg = signal_engine.strategy.cfg
    exec_engine.cfg = signal_engine.strategy.cfg
    exec_engine.use_shared_mem = True

    mock_ibkr = MockIBKRHistorical()
    await mock_ibkr.connect()
    if args.flush_redis:
        mock_ibkr.r.flushdb()
    signal_engine.ibkr = mock_ibkr
    exec_engine.ibkr = mock_ibkr
    signal_engine.mock_cash = mock_ibkr.initial_capital
    exec_engine.mock_cash = mock_ibkr.initial_capital

    persist_svc = data_persistence_service_v8_pg.DataPersistenceServicePG(start_date=target_dates[0])
    patch_trade_log_capture(signal_engine, persist_svc)
    patch_trade_log_capture(exec_engine, persist_svc)

    start_wall = time.time()
    is_first_day = True

    for idx, date_str in enumerate(target_dates, 1):
        logger.info("=" * 80)
        logger.info(f"📂 [{idx}/{len(target_dates)}] Event replay PG date: {date_str}")
        logger.info("❄️ Cold start with PG warmup." if is_first_day else f"🔥 Hot start from previous day {target_dates[idx - 2]}.")

        df_b1, df_o1, df_b5, df_o5 = fetch_day_frames(date_str)
        if df_b1 is None or df_b1.empty:
            logger.warning(f"⚠️ No PG 1s data found for {date_str}, skipping.")
            continue

        frames = build_day_stream(date_str, df_b1, df_o1, df_b5, df_o5, symbols)
        if not frames:
            logger.warning(f"⚠️ No event frames built for {date_str}, skipping.")
            continue

        first_full_min = ((int(frames[0][0]) + 59) // 60) * 60
        logger.info(
            "🎯 First frame: %s | First full minute signal boundary: %s",
            datetime.fromtimestamp(frames[0][0], ny_tz),
            datetime.fromtimestamp(first_full_min, ny_tz),
        )

        for ts_val, batch_payloads in tqdm(frames, desc=f"EventReplay {date_str}", total=len(frames)):
            dt_ny = datetime.fromtimestamp(ts_val, ny_tz)
            time_str = dt_ny.strftime("%H:%M:%S")
            if time_str < "09:30:00" or time_str > "16:00:00":
                continue

            clock.current_ts = float(ts_val)
            r.set("replay:current_ts", str(float(ts_val)))

            await feat_svc.process_market_data(batch_payloads)
            persist_svc._check_date_rotation(ts_val)

            feat_payload = await feat_svc.run_compute_cycle(ts_from_payload=ts_val, return_payload=True)
            if not feat_payload:
                continue

            feat_payload = add_feed_aliases(feat_payload)
            persist_svc.process_feature_data(feat_payload)
            mock_ibkr.record_market_data(feat_payload)

            await signal_engine.process_batch(feat_payload)
            await drain_signal_queue(shared_signal_queue, exec_engine)
            await exec_engine.process_trade_signal({"action": "SYNC", "ts": float(ts_val), "payload": {}})
            signal_engine.mock_cash = exec_engine.mock_cash

            if int(ts_val) % 60 == 0:
                persist_svc.flush()

        persist_svc.flush()
        is_first_day = False

    await exec_engine.force_close_all()
    await asyncio.sleep(0.5)
    await drain_signal_queue(shared_signal_queue, exec_engine)
    persist_svc.flush()

    out_name = args.trades_csv or f"replay_trades_event_pg_1s_{target_dates[0]}_{target_dates[-1]}.csv"
    mock_ibkr.save_trades(filename=out_name)

    elapsed = time.time() - start_wall
    print("\n" + "=" * 60)
    print("📊 EVENT-DRIVEN PG 1S REPLAY SUMMARY")
    print("=" * 60)
    print(f"Dates:      {target_dates[0]} -> {target_dates[-1]}")
    print(f"Symbols:    {len(symbols)}")
    print(f"Elapsed:    {elapsed:.1f}s")
    print(f"Trades CSV: {out_name}")
    print("ℹ️ Alpha generation and execution were driven by the same 1s logical clock.")
    exec_engine.accounting.print_counter_trend_summary()
    print("=" * 60 + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Event-driven PostgreSQL 1s replay: FCS + Signal + Execution in one clock.")
    parser.add_argument("--start-date", type=str, default="20260102")
    parser.add_argument("--end-date", type=str, default=None)
    parser.add_argument("--symbol", type=str, default=None)
    parser.add_argument("--skip-warmup", action="store_true")
    parser.add_argument("--truncate-output", action="store_true", help="Truncate alpha/trade/feature partitions for target dates before replay.")
    parser.add_argument("--flush-redis", action="store_true", help="Flush Redis DB used by MockIBKR before replay.")
    parser.add_argument("--trades-csv", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    try:
        asyncio.run(run_event_replay(parse_args()))
    except KeyboardInterrupt:
        pass

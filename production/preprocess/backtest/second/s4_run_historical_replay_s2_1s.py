#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

os.environ['RUN_MODE'] = 'BACKTEST'
os.environ['DUAL_CONVERGE_TO_SINGLE'] = '1'

import asyncio
import time
import logging
from pathlib import Path
import argparse
import sys
import shutil
import pandas as pd
import sqlite3
import json
import numpy as np
from datetime import datetime
import pytz
from tqdm import tqdm

# MY_DIR = Path(__file__).resolve().parent
# PRODUCTION_DIR = MY_DIR.parents[2]
# sys.path.insert(0, str(PRODUCTION_DIR / "baseline"))
# sys.path.insert(0, str(PRODUCTION_DIR / "history_replay"))
# sys.path.insert(0, str(PRODUCTION_DIR / "model"))

from signal_engine_v8 import SignalEngineV8
from execution_engine_v8 import ExecutionEngineV8, ExecutionWindow
import mock_ibkr_historical_1s
from mock_ibkr_historical_1s import MockIBKRHistorical
print(f"🔍 [Import Trace] Using MockIBKR from: {mock_ibkr_historical_1s.__file__}")

import config
import signal_engine_v8
import execution_engine_v8
from config import TAG_TO_INDEX, option_bucket_tag
try:
    from Domain import ReplaySemanticAuditor
except Exception:  # pragma: no cover
    ReplaySemanticAuditor = None

BACKTEST_STREAM = "backtest_stream"
BACKTEST_GROUP = "backtest_group"

config.REDIS_CFG['input_stream'] = BACKTEST_STREAM
config.REDIS_CFG['orch_group'] = BACKTEST_GROUP
signal_engine_v8.REDIS_CFG['input_stream'] = BACKTEST_STREAM
signal_engine_v8.REDIS_CFG['orch_group'] = BACKTEST_GROUP
execution_engine_v8.REDIS_CFG['input_stream'] = BACKTEST_STREAM
execution_engine_v8.REDIS_CFG['orch_group'] = BACKTEST_GROUP

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [REPLAY_S2_1S] - %(message)s')
logger = logging.getLogger("ReplayS2_1S")
ALPHA_AVAILABLE_DELAY_SECONDS = 60.0


def resolve_db_path(args):
    current_dir = Path(__file__).resolve().parent
    #hist_dir = current_dir.parent / "preprocess" / "backtest" / "history_sqlite_1s"
    hist_dir = Path("/home/kingfang007/quant_project/data/history_sqlite_1s")
    if args.date:
        db_name = f"market_{args.date}.db"
        path = hist_dir / db_name
        if path.exists():
            return path
        logger.error(f"❌ 关键数据库未找到: {path}")
        return None

    all_dbs = sorted(hist_dir.glob("market_*.db"))
    return all_dbs[-1] if all_dbs else None


def _pg_float_env(name: str, default_val: float) -> float:
    try:
        return float(os.environ.get(name, default_val))
    except (TypeError, ValueError):
        return float(default_val)


def _pg_merge_asof_per_symbol(left, right, value_cols, *, tolerance=None):
    left = left.copy()
    if "_row_order" not in left.columns:
        left["_row_order"] = np.arange(len(left), dtype=np.int64)
    value_cols = [c for c in value_cols if c in right.columns]
    if not value_cols or right.empty:
        out = left.copy()
        for col in value_cols:
            out[col] = np.nan
        return out.sort_values("_row_order").reset_index(drop=True)
    right = (
        right[["ts", "symbol"] + value_cols]
        .dropna(subset=["ts", "symbol"])
        .drop_duplicates(["ts", "symbol"], keep="last")
    )
    parts = []
    for sym, left_g in left.groupby("symbol", sort=False):
        left_g = left_g.sort_values("ts")
        right_g = right[right["symbol"] == sym].sort_values("ts")[["ts"] + value_cols]
        if right_g.empty:
            merged = left_g.copy()
            for col in value_cols:
                merged[col] = np.nan
        else:
            merged = pd.merge_asof(left_g, right_g, on="ts", direction="backward", tolerance=tolerance)
        parts.append(merged)
    return pd.concat(parts, ignore_index=True).sort_values("_row_order").reset_index(drop=True)


def load_merged_df_pgsql(date_str: str, symbols: list, pg_url: str):
    """PG 日分区 dense 合并，产出列与 SQLite merge 后一致（供下方同一套 replay 使用）。"""
    try:
        import psycopg2
    except ImportError as e:
        logger.error("需要 psycopg2: %s", e)
        return None

    def _exists(cur, t):
        cur.execute(
            "SELECT EXISTS(SELECT 1 FROM information_schema.tables WHERE table_name=%s)", (t,)
        )
        return bool(cur.fetchone()[0])

    def _cols(cur, t):
        cur.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_name=%s", (t,)
        )
        return {r[0] for r in cur.fetchall()}

    pa, pb, po = f"alpha_logs_{date_str}", f"market_bars_1s_{date_str}", f"option_snapshots_1s_{date_str}"
    conn = psycopg2.connect(pg_url)
    cur = conn.cursor()
    for t in (pa, pb, po):
        if not _exists(cur, t):
            logger.error("PG 缺表: %s", t)
            cur.close()
            conn.close()
            return None
    ac = _cols(cur, pa)
    sel = ["ts", "symbol", "alpha as alpha_score"]
    if "vol_z" in ac:
        sel.append("vol_z as fast_vol")
    df_a = pd.read_sql(f"SELECT {', '.join(sel)} FROM {pa}", conn)
    if "fast_vol" not in df_a.columns:
        df_a["fast_vol"] = 0.0
    df_s = pd.read_sql(f"SELECT ts, symbol, close, open, high, low, volume FROM {pb}", conn)
    df_o = pd.read_sql(f"SELECT ts, symbol, buckets_json FROM {po}", conn)
    cur.close()
    conn.close()

    for x in (df_a, df_s, df_o):
        x["ts"] = x["ts"].astype(float)
        x["symbol"] = x["symbol"].astype(str)

    uts = sorted(df_s["ts"].unique())
    if not uts:
        return None
    dix = (
        df_s[df_s["symbol"].isin(["SPY", "QQQ"])]
        .drop_duplicates(["ts", "symbol"], keep="last")
        .pivot(index="ts", columns="symbol", values="close")
        .reindex(uts)
        .ffill()
    )
    roc = pd.DataFrame({"ts": uts})
    for c in ("SPY", "QQQ"):
        roc[f"{c.lower()}_roc_5min"] = dix[c].pct_change(periods=300).fillna(0.0).values if c in dix.columns else 0.0

    if symbols:
        df_a, df_s, df_o = [d[d["symbol"].isin(symbols)] for d in (df_a, df_s, df_o)]

    base = pd.MultiIndex.from_product([uts, symbols], names=["ts", "symbol"]).to_frame(index=False)
    base["_row_order"] = np.arange(len(base), dtype=np.int64)
    ts_s, ts_o, ts_a = (
        _pg_float_env("REPLAY_1S_STOCK_ASOF_TOLERANCE_SEC", 300.0),
        _pg_float_env("REPLAY_1S_OPTION_ASOF_TOLERANCE_SEC", 60.0),
        _pg_float_env("REPLAY_1S_ALPHA_ASOF_TOLERANCE_SEC", 120.0),
    )
    sc = [c for c in ["open", "high", "low", "close", "volume"] if c in df_s.columns]
    dm = _pg_merge_asof_per_symbol(base, df_s, sc, tolerance=None if ts_s <= 0 else ts_s)
    dm = dm.merge(roc, on="ts", how="left")
    df_a = df_a.assign(alpha_label_ts=df_a["ts"], ts=df_a["ts"] + ALPHA_AVAILABLE_DELAY_SECONDS).sort_values("ts")
    df_o = df_o.sort_values("ts")
    dm = _pg_merge_asof_per_symbol(dm, df_o, ["buckets_json"], tolerance=None if ts_o <= 0 else ts_o)
    dm = _pg_merge_asof_per_symbol(
        dm, df_a, ["alpha_score", "fast_vol", "alpha_label_ts"],
        tolerance=None if ts_a <= 0 else ts_a,
    )
    return dm.sort_values(["ts", "_row_order"]).reset_index(drop=True)


def safe_col(group, col, default_val, dtype=np.float32):
    if col in group.columns:
        return group[col].fillna(default_val).values.astype(dtype)
    return np.full(len(group), default_val, dtype=dtype)


def build_option_arrays(group):
    put_prices, call_prices = [], []
    put_ks, call_ks = [], []
    put_ivs, call_ivs = [], []
    put_bids, put_asks = [], []
    call_bids, call_asks = [], []
    symbols_with_data = set()
    put_idx = TAG_TO_INDEX.get(option_bucket_tag(-1), 0)
    call_idx = TAG_TO_INDEX.get(option_bucket_tag(1), 2)

    for row in group.itertuples():
        try:
            raw = getattr(row, "buckets_json", None)
            if isinstance(raw, dict):
                bk = raw.get("buckets", [])
            elif isinstance(raw, str):
                bk = json.loads(raw).get("buckets", []) if raw else []
            elif raw is None or (isinstance(raw, float) and np.isnan(raw)):
                bk = []
            else:
                bk = json.loads(str(raw)).get("buckets", [])
            p_bk = bk[put_idx] if len(bk) > put_idx else []
            c_bk = bk[call_idx] if len(bk) > call_idx else []
            p_p = float(p_bk[0]) if len(p_bk) > 0 else 0.0
            c_p = float(c_bk[0]) if len(c_bk) > 0 else 0.0

            put_prices.append(p_p)
            put_ks.append(float(p_bk[5]) if len(p_bk) > 5 else 0.0)
            put_ivs.append(float(p_bk[7]) if len(p_bk) > 7 else 0.0)
            put_bids.append(float(p_bk[8]) if len(p_bk) > 8 else 0.0)
            put_asks.append(float(p_bk[9]) if len(p_bk) > 9 else 0.0)

            call_prices.append(c_p)
            call_ks.append(float(c_bk[5]) if len(c_bk) > 5 else 0.0)
            call_ivs.append(float(c_bk[7]) if len(c_bk) > 7 else 0.0)
            call_bids.append(float(c_bk[8]) if len(c_bk) > 8 else 0.0)
            call_asks.append(float(c_bk[9]) if len(c_bk) > 9 else 0.0)

            if p_p > 0.01 or c_p > 0.01:
                symbols_with_data.add(row.symbol)
        except Exception:
            for arr in [
                put_prices, call_prices, put_ks, call_ks, put_ivs, call_ivs,
                put_bids, put_asks, call_bids, call_asks,
            ]:
                arr.append(0.0)

    return {
        'put_prices': np.array(put_prices, dtype=np.float32),
        'call_prices': np.array(call_prices, dtype=np.float32),
        'put_ks': np.array(put_ks, dtype=np.float32),
        'call_ks': np.array(call_ks, dtype=np.float32),
        'put_ivs': np.array(put_ivs, dtype=np.float32),
        'call_ivs': np.array(call_ivs, dtype=np.float32),
        'put_bids': np.array(put_bids, dtype=np.float32),
        'put_asks': np.array(put_asks, dtype=np.float32),
        'call_bids': np.array(call_bids, dtype=np.float32),
        'call_asks': np.array(call_asks, dtype=np.float32),
        'symbols_with_data': symbols_with_data,
    }


def build_replay_packet(group, ts_val, is_new_minute=False):
    """Build one 1s quote packet; callers decide whether it is also the minute signal packet."""
    symbols_list = group['symbol'].tolist()
    opt = build_option_arrays(group)
    return {
        'symbols': symbols_list,
        'ts': float(ts_val),
        'stock_price': safe_col(group, 'close', 0.0),
        'fast_vol': safe_col(group, 'fast_vol', 0.0),
        'precalc_alpha': safe_col(group, 'alpha_score', 0.0),
        'alpha_label_ts': safe_col(group, 'alpha_label_ts', 0.0, dtype=np.float64),
        'alpha_available_ts': np.full(len(group), float(ts_val), dtype=np.float64),
        'spy_roc_5min': safe_col(group, 'spy_roc_5min', 0.0),
        'qqq_roc_5min': safe_col(group, 'qqq_roc_5min', 0.0),
        'is_new_minute': bool(is_new_minute),
        'symbols_with_data': opt['symbols_with_data'],
        'feed_put_price': opt['put_prices'],
        'feed_call_price': opt['call_prices'],
        'feed_put_k': opt['put_ks'],
        'feed_call_k': opt['call_ks'],
        'feed_put_iv': opt['put_ivs'],
        'feed_call_iv': opt['call_ivs'],
        'feed_put_bid': opt['put_bids'],
        'feed_put_ask': opt['put_asks'],
        'feed_call_bid': opt['call_bids'],
        'feed_call_ask': opt['call_asks'],
        'feed_put_vol': np.ones(len(group), dtype=np.float32),
        'feed_call_vol': np.ones(len(group), dtype=np.float32),
        'feed_call_bid_size': np.full(len(group), 100.0, dtype=np.float32),
        'feed_call_ask_size': np.full(len(group), 100.0, dtype=np.float32),
        'feed_put_bid_size': np.full(len(group), 100.0, dtype=np.float32),
        'feed_put_ask_size': np.full(len(group), 100.0, dtype=np.float32),
        'slow_1m': np.zeros((len(symbols_list), 30, 1), dtype=np.float32),
        'feed_put_id': [""] * len(group),
        'feed_call_id': [""] * len(group),
    }


async def drain_signal_queue(shared_signal_queue, exec_engine):
    """Synchronously drain SE→OMS queue before advancing the replay clock."""
    # 与 ExecutionEngineV8.signal_queue 为同一 asyncio.Queue 时,
    # 等价于 exec_engine.drain_trade_signal_queue()。
    await exec_engine.drain_trade_signal_queue()


async def main():
    print(f"!!! EXECUTING S2 1S REPLAY SCRIPT: {__file__} !!!")
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, default="20260102")
    parser.add_argument("--symbol", type=str, default=None)
    parser.add_argument(
        "--use-pgsql",
        action="store_true",
        help="从 PostgreSQL 日分区读数（连接串固定为 config.PG_DB_URL），其余回放逻辑与 SQLite 相同",
    )
    parser.add_argument(
        "--legacy-per-second-strategy",
        action="store_true",
        help="Debug only: run the old per-second process_batch loop. Default is synchronized minute-window mode.",
    )
    parser.add_argument(
        "--domain-semantic-audit",
        action="store_true",
        help="Run Domain semantic audit alongside replay without changing mainline logic.",
    )
    parser.add_argument(
        "--domain-semantic-audit-strict",
        action="store_true",
        help="Raise assertion on first Domain semantic mismatch. Implies --domain-semantic-audit.",
    )
    args = parser.parse_args()

    from config import TARGET_SYMBOLS

    symbols = [args.symbol] if args.symbol else TARGET_SYMBOLS

    if not args.use_pgsql:
        db_path = resolve_db_path(args)
        if db_path is None:
            logger.error(f"❌ Target Database not found. (Date: {args.date})")
            return
        logger.info(f"📂 Loading data directly from SQLite: {db_path}...")

    v8_root = Path(__file__).parent.parent
    config_paths = {
        'fast': str(v8_root / "daily_backtest" / "fast_feature.json"),
        'slow': str(v8_root / "daily_backtest" / "slow_feature.json")
    }

    logger.info("🛠️ Building S2 dual engine (1s execution path)...")
    domain_auditor = None
    if ReplaySemanticAuditor is not None and (args.domain_semantic_audit or args.domain_semantic_audit_strict):
        domain_auditor = ReplaySemanticAuditor(
            enabled=True,
            strict=bool(args.domain_semantic_audit_strict),
        )
        logger.info(
            "🧪 [Domain Audit] enabled strict=%s log_every=%s",
            bool(args.domain_semantic_audit_strict),
            domain_auditor.log_every,
        )
    signal_engine = SignalEngineV8(
        symbols=symbols,
        mode='backtest',
        config_paths=config_paths,
        model_paths={}
    )
    logger.info(
        "🧭 [AlphaEngine Audit] cfg=%s INDEX_GUARD_ENABLED=%s INDEX_REVERSAL_EXIT_ENABLED=%s",
        signal_engine.cfg.__class__.__module__,
        getattr(signal_engine.cfg, 'INDEX_GUARD_ENABLED', None),
        getattr(signal_engine.cfg, 'INDEX_REVERSAL_EXIT_ENABLED', None),
    )
    shared_signal_queue = asyncio.Queue()
    signal_engine.signal_queue = shared_signal_queue
    signal_engine.use_shared_mem = True

    exec_engine = ExecutionEngineV8(
        symbols=symbols,
        mode='backtest',
        signal_queue=shared_signal_queue
    )
    logger.info(
        "🧭 [OMS Strategy Audit] core=%s cfg=%s INDEX_GUARD_ENABLED=%s INDEX_REVERSAL_EXIT_ENABLED=%s",
        exec_engine.strategy.__class__.__name__,
        exec_engine.cfg.__class__.__module__,
        getattr(exec_engine.cfg, 'INDEX_GUARD_ENABLED', None),
        getattr(exec_engine.cfg, 'INDEX_REVERSAL_EXIT_ENABLED', None),
    )

    logger.info("🔌 Injecting Mock IBKR 1s...")
    mock_ibkr = MockIBKRHistorical()
    await mock_ibkr.connect()
    mock_ibkr.r.flushdb()

    signal_engine.ibkr = mock_ibkr
    exec_engine.ibkr = mock_ibkr
    signal_engine.mock_cash = mock_ibkr.initial_capital
    exec_engine.mock_cash = mock_ibkr.initial_capital

    if args.use_pgsql:
        logger.info("📂 Loading from PostgreSQL via config.PG_DB_URL (partition %s)...", args.date)
        df = load_merged_df_pgsql(str(args.date), symbols, config.PG_DB_URL)
        if df is None or df.empty:
            logger.error("❌ PostgreSQL merged frame empty")
            return
    else:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        df_a = pd.read_sql("SELECT ts, symbol, alpha as alpha_score, vol_z as fast_vol FROM alpha_logs", conn)
        df_s = pd.read_sql("SELECT ts, symbol, close, open, high, low, volume FROM market_bars_1s", conn)
        df_o = pd.read_sql("SELECT ts, symbol, buckets_json FROM option_snapshots_1s", conn)
        conn.close()

        # 🚀 [NEW] Calculate Index ROCs from the full dataset before filtering
        # Ensure ts is float for all calculations
        df_s['ts'] = df_s['ts'].astype(float)
        df_idx = df_s[df_s['symbol'].isin(['SPY', 'QQQ'])].pivot(index='ts', columns='symbol', values='close')

        # Prepare a ROC map for all unique timestamps
        unique_ts = sorted(df_s['ts'].unique())
        df_roc_map = pd.DataFrame(index=unique_ts)
        for col in ['SPY', 'QQQ']:
            if col in df_idx.columns:
                # 5-minute ROC (1s bars = 300 periods)
                df_roc_map[f'{col.lower()}_roc_5min'] = df_idx[col].pct_change(periods=300).fillna(0.0)
            else:
                df_roc_map[f'{col.lower()}_roc_5min'] = 0.0

        # Merge ROCs into df_s so they carry over after symbol filtering
        df_s = df_s.merge(df_roc_map.reset_index().rename(columns={'index': 'ts'}), on='ts', how='left')

        if symbols:
            df_a = df_a[df_a['symbol'].isin(symbols)]
            df_s = df_s[df_s['symbol'].isin(symbols)]
            df_o = df_o[df_o['symbol'].isin(symbols)]

        for df_tmp in [df_a, df_s, df_o]:
            df_tmp['ts'] = df_tmp['ts'].astype(float)
            df_tmp['symbol'] = df_tmp['symbol'].astype(str)

        # alpha_logs 的 ts 是分钟标签时间（左对齐），但该分钟 alpha 只有到下一分钟边界才可见。
        # 这里先把“可交易时间”右移 60s，再和 1s 行情做 backward merge，避免 lookahead。
        df_a['alpha_label_ts'] = df_a['ts']
        df_a['ts'] = df_a['ts'] + ALPHA_AVAILABLE_DELAY_SECONDS
        df_a = df_a.sort_values('ts')
        df_s = df_s.sort_values('ts')
        df_o = df_o.sort_values('ts')

        df_market = pd.merge_asof(df_s, df_o, on='ts', by='symbol', direction='backward', tolerance=2)
        df = pd.merge_asof(df_market, df_a, on='ts', by='symbol', direction='backward', tolerance=120)

        if df.empty:
            logger.error("❌ Merged dataset is EMPTY! Check DB integrity.")
            return

    sort_by = ['ts', '_row_order'] if '_row_order' in df.columns else ['ts']
    grouped = df.sort_values(sort_by, ascending=True).groupby('ts')
    unique_groups = df['ts'].nunique()
    expected_symbol_count = len(symbols)
    symbol_rows_per_tick = df.groupby('ts')['symbol'].nunique()
    incomplete_ticks = int((symbol_rows_per_tick < expected_symbol_count).sum())
    if incomplete_ticks > 0:
        logger.warning(
            "⚠️ [REPLAY_S2_1S-DIAG] symbol rows missing on %d/%d ticks (expected=%d)",
            incomplete_ticks,
            int(unique_groups),
            expected_symbol_count,
        )

    logger.info(f"🚀 Starting S2 1s replay bus ({unique_groups} ticks)...")
    start_time = time.time()

    ny_tz = pytz.timezone('America/New_York')

    minute_windows = {}
    ordered_minute_ts = []

    for ts_val, group in tqdm(grouped, desc="Preparing minute windows", total=unique_groups):
        dt_ny = datetime.fromtimestamp(ts_val, tz=pytz.utc).astimezone(ny_tz)
        curr_time_str = dt_ny.strftime('%H:%M:%S')
        if curr_time_str < "09:35:00" or curr_time_str > "16:00:00":
            continue

        minute_ts = int(ts_val // 60) * 60
        packet = build_replay_packet(group, ts_val, is_new_minute=False)
        packet['minute_ts'] = float(minute_ts)

        if minute_ts not in minute_windows:
            minute_windows[minute_ts] = {
                'minute_ts': float(minute_ts),
                'alpha_label_ts': float(minute_ts - ALPHA_AVAILABLE_DELAY_SECONDS),
                'alpha_available_ts': float(minute_ts),
                'signal_packet': None,
                'quotes': [],
                'active_counts': [],
                'symbol_rows': [],
                'missing_symbol_ticks': 0,
                'zero_alpha_ticks': 0,
            }
            ordered_minute_ts.append(minute_ts)

        window = minute_windows[minute_ts]
        window['quotes'].append(packet)
        active_now = int((packet['precalc_alpha'] != 0).sum())
        symbol_rows_now = len(packet['symbols'])
        window['active_counts'].append(active_now)
        window['symbol_rows'].append(symbol_rows_now)
        if symbol_rows_now < expected_symbol_count:
            window['missing_symbol_ticks'] += 1
        if active_now == 0:
            window['zero_alpha_ticks'] += 1
        if window['signal_packet'] is None:
            # The minute signal is available at minute_ts and is evaluated once.
            # The remaining packets in this window are execution quotes only.
            sig_packet = dict(packet)
            sig_packet['ts'] = float(minute_ts)
            sig_packet['is_new_minute'] = True
            sig_packet['alpha_available_ts'] = np.full(len(packet['symbols']), float(minute_ts), dtype=np.float64)
            window['signal_packet'] = sig_packet

    logger.info(
        "🚀 Starting synchronized 1s execution windows (%d minutes, %d ticks)...",
        len(ordered_minute_ts),
        unique_groups,
    )

    for minute_ts in tqdm(ordered_minute_ts, desc="Replaying minute windows", total=len(ordered_minute_ts)):
        window = minute_windows[minute_ts]
        quotes = window['quotes']
        if not quotes:
            continue

        ac = window['active_counts']
        sc = window['symbol_rows']
        logger.info(
            "📊 [REPLAY_S2_1S-WINDOW] Minute %d | Active(min/avg/max)=%d/%.2f/%d | "
            "ticks=%d | symbol_rows(min/max)=%d/%d | missing_symbol_ticks=%d | zero_alpha_ticks=%d",
            int(minute_ts // 60),
            int(min(ac)) if ac else 0,
            float(sum(ac) / len(ac)) if ac else 0.0,
            int(max(ac)) if ac else 0,
            int(len(quotes)),
            int(min(sc)) if sc else 0,
            int(max(sc)) if sc else 0,
            int(window['missing_symbol_ticks']),
            int(window['zero_alpha_ticks']),
        )

        if args.legacy_per_second_strategy:
            # Debug-only compatibility path: old behavior, one process_batch per second.
            for quote_packet in quotes:
                if domain_auditor is not None:
                    domain_auditor.audit_quote_packet(quote_packet)
                quote_packet['is_new_minute'] = (float(quote_packet['ts']) == float(quotes[0]['ts']))
                await signal_engine.process_batch(quote_packet)
                exec_engine._cache_execution_market_packet(quote_packet)
                mock_ibkr.record_market_data(quote_packet, alphas=quote_packet['precalc_alpha'])
                await drain_signal_queue(shared_signal_queue, exec_engine)
                await exec_engine.process_trade_signal({'action': 'SYNC', 'ts': float(quote_packet['ts']), 'payload': {}})
                signal_engine.mock_cash = exec_engine.mock_cash
                if domain_auditor is not None:
                    domain_auditor.audit_post_window(minute_ts, exec_engine, quote_packet)
            continue

        signal_packet = window['signal_packet']
        if signal_packet is None:
            continue
        if domain_auditor is not None:
            domain_auditor.audit_pre_window(minute_ts, signal_packet)
            for quote_packet in quotes:
                domain_auditor.audit_quote_packet(quote_packet)

        # [Step A] 用 ExecutionWindow 把 alpha_frame + 60 秒 quotes 打包成一等契约,
        # OMS 通过 execute_window() 单入口编排 "分钟边界一次 + 秒级循环"。
        # 行为应 bit-identical 于之前三行散调用 (cache_minute + execute_phase).
        exec_window = ExecutionWindow.from_packets(
            minute_ts=int(minute_ts),
            alpha_frame=signal_packet,
            quotes_1s=quotes,
        )

        # SE: 仅 alpha 推理 (分钟级), 产物进 signal_queue / ALPHA_FRAME
        await signal_engine.process_batch(exec_window.alpha_frame)
        # OMS: 分钟边界 cache/drain + 秒级 ingest/SYNC
        await exec_engine.execute_window(exec_window)
        signal_engine.mock_cash = exec_engine.mock_cash
        if domain_auditor is not None:
            domain_auditor.audit_post_window(minute_ts, exec_engine, quotes[-1])

    elapsed = time.time() - start_time
    print(f"\n✅ S2 1S Replay Finished in {elapsed:.1f}s.")
    await exec_engine.force_close_all()
    await asyncio.sleep(0.5)
    mock_ibkr.save_trades(filename="replay_trades_s2_1s.csv")
    # 结果镜像到脚本目录，避免误读旧文件（常见：只看 second/ 下历史 CSV）。
    try:
        log_dir = Path.home() / "quant_project" / "logs"
        src_raw = log_dir / "replay_trades_s2_1s.csv"
        src_logical = log_dir / "replay_trades_s2_1s_logical.csv"
        dst_dir = Path(__file__).resolve().parent
        if src_raw.exists():
            dst_raw = dst_dir / src_raw.name
            shutil.copy2(src_raw, dst_raw)
            print(f"💾 [Mirror] raw trades copied to: {dst_raw}")
        if src_logical.exists():
            dst_logical = dst_dir / src_logical.name
            shutil.copy2(src_logical, dst_logical)
            print(f"💾 [Mirror] logical trades copied to: {dst_logical}")
    except Exception as e:
        print(f"⚠️ [Mirror] copy replay trades failed: {e}")

    print("\n" + "=" * 50)
    print("📊 FINAL BACKTEST PERFORMANCE SUMMARY (S2 1S DUAL)")
    print("=" * 50)
    if domain_auditor is not None:
        print(f"🧪 Domain semantic audit stats: {domain_auditor.stats()}")
    print("ℹ️ 最终收益统计以 MockIBKR 生成的成交账本为准（已包含 execution delay / FIFO / 手续费 / chunk 撮合）。")
    print("ℹ️ ExecutionEngine 内部 realized_pnl 口径与 Mock 重撮合价不同，这里不再重复打印，避免双账本混淆。")
    exec_engine.accounting.print_counter_trend_summary()
    print("=" * 50 + "\n")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass

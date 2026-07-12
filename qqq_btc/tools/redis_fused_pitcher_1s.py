#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
秒级 fused_market_stream 发球机 —— 支持两种数据源:

1. raw  : /mnt/s990/data/raw_1s/options_databento_v3/{SYM}/QQQ_YYYY-MM-DD.parquet (推荐,含 6 月)
2. sqlite: ~/quant_project/data/history_sqlite_1s/market_YYYYMMDD.db

用法:
    # 6 月原始秒级期权 + 1min 股票回退
    python qqq_btc/tools/redis_fused_pitcher_1s.py --source raw --date 2026-06-26 --speed 1.0

    # legacy sqlite
    python qqq_btc/tools/redis_fused_pitcher_1s.py --source sqlite --date 20260202
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
import sys
import time
import uuid
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytz
import redis

_REPO = Path(__file__).resolve().parents[2]
_BASELINE = _REPO / "New_Pro" / "baseline_qqq"
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
if str(_BASELINE) not in sys.path:
    sys.path.insert(0, str(_BASELINE))

import baseline_paths  # noqa: E402,F401

from config import (  # noqa: E402
    DB_DIR_1S,
    GROUP_FEATURE,
    GROUP_OMS,
    GROUP_ORCH,
    GROUP_PERSISTENCE,
    HASH_OPTION_SNAPSHOT,
    REDIS_CFG,
    STREAM_FUSED_MARKET,
    STREAM_INFERENCE,
    STREAM_ORCH_SIGNAL,
    STREAM_TRADE_LOG,
    TARGET_SYMBOLS,
    get_feature_service_state_file,
    get_redis_db,
)
from utils import serialization_utils as ser  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [PITCHER_1S] - %(message)s",
)
logger = logging.getLogger("Pitcher1s")

NY_TZ = pytz.timezone("America/New_York")


def _ts_key(ts_val: float | int) -> int:
    """统一 map 时间戳键为 int，避免 float/int 键导致 lookup 失败。"""
    return int(ts_val)


DEFAULT_OPTION_ROOT = Path("/mnt/s990/data/raw_1s/options_databento_v3")
DEFAULT_STOCK_ROOT = Path("/mnt/s990/data/raw_1s/stocks")
# 用 resample 源(spnq_train)而非 feature_merge 产物: 其 vwap 是交易所真实分钟 vwap,
# 对应实盘 feed 的 wap 字段;quote_features_test 的 vwap 已被日内累计口径覆盖,不能当数据源。
DEFAULT_STOCK_FALLBACK_ROOT = Path.home() / "train_data/spnq_train/QQQ"
DEFAULT_STOCK_FALLBACKS: dict[str, Path] = {
    "QQQ": DEFAULT_STOCK_FALLBACK_ROOT,
    "VIXY": Path.home() / "train_data/spnq_train/VIXY",
}
DEFAULT_GREEK_ROOT = Path.home() / "train_data/quote_options_day_iv"


def _normalize_yyyymmdd(date_str: str) -> str:
    s = date_str.strip().replace("-", "")
    if len(s) != 8 or not s.isdigit():
        raise ValueError(f"invalid date: {date_str!r}")
    return s


def _iso_from_yyyymmdd(date_str: str) -> str:
    ymd = _normalize_yyyymmdd(date_str)
    return f"{ymd[:4]}-{ymd[4:6]}-{ymd[6:8]}"


def _session_ts_list(date_str: str, *, include_preopen_minute: bool) -> list[int]:
    start_ts, end_ts = _session_bounds_ts(date_str, include_preopen_minute=include_preopen_minute)
    return list(range(start_ts, end_ts))


def _session_bounds_ts(date_str: str, *, include_preopen_minute: bool) -> tuple[int, int]:
    start_hhmmss = "09:29:00" if include_preopen_minute else "09:30:00"
    start_dt = NY_TZ.localize(datetime.strptime(f"{date_str} {start_hhmmss}", "%Y%m%d %H:%M:%S"))
    end_dt = NY_TZ.localize(datetime.strptime(f"{date_str} 16:00:00", "%Y%m%d %H:%M:%S"))
    return int(start_dt.timestamp()), int(end_dt.timestamp())


def _limit_session_ticks(all_ts: list[int], max_session_bars: int | None) -> list[int]:
    """仅保留 session 前 N 个分钟 bar 对应的秒级 tick（每 bar 60 秒）。"""
    if not max_session_bars or max_session_bars <= 0 or not all_ts:
        return all_ts
    cap = int(max_session_bars) * 60
    if len(all_ts) <= cap:
        return all_ts
    return all_ts[:cap]


def _filter_df_to_session(
    df: pd.DataFrame,
    date_str: str,
    *,
    include_preopen_minute: bool,
    label: str,
) -> pd.DataFrame:
    if df.empty or "ts" not in df.columns:
        return df
    start_ts, end_ts = _session_bounds_ts(date_str, include_preopen_minute=include_preopen_minute)
    before = len(df)
    out = df[(df["ts"] >= start_ts) & (df["ts"] < end_ts)].copy()
    dropped = before - len(out)
    if dropped > 0:
        logger.info("Session filter %s: dropped %d rows outside NY session for %s", label, dropped, date_str)
    return out


def _filter_ts_keys_to_session(ts_values, date_str: str, *, include_preopen_minute: bool):
    start_ts, end_ts = _session_bounds_ts(date_str, include_preopen_minute=include_preopen_minute)
    return [ts for ts in ts_values if start_ts <= int(ts) < end_ts]


def _redis_client() -> redis.Redis:
    return redis.Redis(
        host=REDIS_CFG["host"],
        port=REDIS_CFG["port"],
        db=get_redis_db(),
        decode_responses=False,
    )


def init_replay_redis(
    r: redis.Redis,
    *,
    run_id: str | None = None,
    reset: bool = True,
) -> str:
    """清空流/消费组并写入 replay 起始状态。"""
    run_id = run_id or str(uuid.uuid4())[:8]
    if not reset:
        return run_id

    for key in (
        STREAM_FUSED_MARKET,
        STREAM_INFERENCE,
        STREAM_ORCH_SIGNAL,
        STREAM_TRADE_LOG,
        HASH_OPTION_SNAPSHOT,
        f"replay:status:{run_id}",
    ):
        try:
            r.delete(key)
        except Exception:
            pass

    streams_and_groups = {
        STREAM_FUSED_MARKET: [GROUP_FEATURE, GROUP_ORCH, GROUP_OMS, GROUP_PERSISTENCE],
        STREAM_INFERENCE: [GROUP_ORCH, GROUP_PERSISTENCE],
        STREAM_ORCH_SIGNAL: [GROUP_OMS, GROUP_PERSISTENCE],
        STREAM_TRADE_LOG: [GROUP_PERSISTENCE],
    }
    for stream, groups in streams_and_groups.items():
        try:
            r.xadd(stream, {"init": "1"})
        except Exception:
            pass
        for group in groups:
            try:
                r.xgroup_create(stream, group, id="0", mkstream=True)
            except Exception:
                pass

    for sync_key in (
        "sync:feature_calc_done",
        "sync:feature_calc_done_frame_id",
        "sync:orch_done",
        "sync:orch_done_frame_id",
    ):
        r.delete(sync_key)

    for sym in TARGET_SYMBOLS:
        for bar_key in (f"BAR:1M:{sym}", f"BAR_OPT:1M:{sym}"):
            try:
                r.delete(bar_key)
            except Exception:
                pass

    try:
        for lock_key in r.keys("lock:oms_writer:*"):
            r.delete(lock_key)
    except Exception:
        pass

    state_file = get_feature_service_state_file()
    if state_file.exists():
        try:
            state_file.unlink()
            logger.info("Removed FCS state cache: %s", state_file)
        except OSError as exc:
            logger.warning("Could not remove FCS state: %s", exc)

    try:
        from config import PG_DB_URL  # noqa: WPS433
        from fcs_state_store import FCSStateStore  # noqa: WPS433

        ns = state_file.stem.replace("feature_service_state_", "") or "default"
        store = FCSStateStore(namespace=ns, pg_url=PG_DB_URL)
        store.ensure_table()
        dropped = store.drop_namespace()
        if dropped:
            logger.info("Dropped FCS PG namespace=%s rows=%d", ns, dropped)
    except Exception as exc:
        logger.warning("Could not drop FCS PG state: %s", exc)

    r.set("replay:status", "INIT")
    r.set(f"replay:status:{run_id}", "INIT")
    logger.info("Redis replay init OK | run_id=%s db=%d", run_id, get_redis_db())
    return run_id


def set_replay_start_ts(r: redis.Redis, date_str: str, *, include_preopen_minute: bool = False) -> float:
    start_ts, _ = _session_bounds_ts(date_str, include_preopen_minute=include_preopen_minute)
    r.set("replay:current_ts", str(start_ts))
    os.environ["REPLAY_START_TS"] = str(start_ts)
    return float(start_ts)


class FusedPitcher1s:
    """从 SQLite 1s 库逐秒推送 fused batch。"""

    def __init__(
        self,
        *,
        db_dir: Path | None = None,
        symbols: list[str] | None = None,
        run_id: str | None = None,
    ):
        self.db_dir = Path(db_dir or DB_DIR_1S)
        self.symbols = list(symbols or TARGET_SYMBOLS)
        self.run_id = run_id or str(uuid.uuid4())[:8]
        self.r = _redis_client()

    def _resolve_dbs(self, start_date: str | None, end_date: str | None) -> list[Path]:
        all_dbs = sorted(self.db_dir.glob("market_*.db"))
        if not all_dbs:
            raise FileNotFoundError(f"No market_*.db under {self.db_dir}")

        def _date_of(p: Path) -> str:
            return p.stem.split("_", 1)[1]

        if start_date:
            all_dbs = [p for p in all_dbs if _date_of(p) >= start_date]
        if end_date:
            all_dbs = [p for p in all_dbs if _date_of(p) <= end_date]
        if not all_dbs:
            raise FileNotFoundError(
                f"No sqlite db in [{start_date}, {end_date}] under {self.db_dir}"
            )
        return all_dbs

    @staticmethod
    def _to_bar_map(df: pd.DataFrame) -> dict:
        out: dict[float, dict] = defaultdict(dict)
        if df.empty:
            return {}
        for ts, sym, o, h, l, c, v in zip(
            df["ts_aligned"].values,
            df["symbol"].values,
            df["open"].values,
            df["high"].values,
            df["low"].values,
            df["close"].values,
            df["volume"].values,
        ):
            out[_ts_key(ts)][sym] = {
                "open": float(o),
                "high": float(h),
                "low": float(l),
                "close": float(c),
                "volume": float(v),
            }
        return dict(out)

    @staticmethod
    def _to_opt_map(df: pd.DataFrame) -> dict:
        out: dict[float, dict] = defaultdict(dict)
        if df.empty:
            return {}
        for ts, sym, b_json in zip(df["ts_aligned"].values, df["symbol"].values, df["buckets_json"].values):
            opt_data = b_json
            if isinstance(b_json, str):
                try:
                    opt_data = json.loads(b_json)
                except json.JSONDecodeError:
                    opt_data = {}
            out[_ts_key(ts)][sym] = opt_data
        return dict(out)

    def _stream_day(
        self,
        db_path: Path,
        *,
        speed_factor: float,
        sync_mode: bool,
        include_preopen_minute: bool,
        progress_every: int,
        max_session_bars: int | None = None,
    ) -> int:
        date_str = db_path.stem.split("_", 1)[1]
        logger.info("Streaming day %s from %s", date_str, db_path)

        with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=30.0) as conn:
            df_bars_1s = pd.read_sql(
                "SELECT symbol, ts, open, high, low, close, volume FROM market_bars_1s ORDER BY ts ASC",
                conn,
            )
            df_opts_1s = pd.read_sql(
                "SELECT symbol, ts, buckets_json FROM option_snapshots_1s ORDER BY ts ASC",
                conn,
            )
            try:
                df_bars_5m = pd.read_sql(
                    "SELECT symbol, ts, open, high, low, close, volume FROM market_bars_5m ORDER BY ts ASC",
                    conn,
                )
            except Exception:
                df_bars_5m = pd.DataFrame()
            try:
                df_opts_5m = pd.read_sql(
                    "SELECT symbol, ts, buckets_json FROM option_snapshots_5m ORDER BY ts ASC",
                    conn,
                )
            except Exception:
                df_opts_5m = pd.DataFrame()

        if df_bars_1s.empty and df_opts_1s.empty:
            logger.warning("Empty 1s tables in %s, skip.", db_path.name)
            return 0

        df_bars_1s = _filter_df_to_session(
            df_bars_1s, date_str, include_preopen_minute=include_preopen_minute, label="market_bars_1s"
        )
        df_opts_1s = _filter_df_to_session(
            df_opts_1s, date_str, include_preopen_minute=include_preopen_minute, label="option_snapshots_1s"
        )
        if not df_bars_5m.empty:
            df_bars_5m = _filter_df_to_session(
                df_bars_5m, date_str, include_preopen_minute=include_preopen_minute, label="market_bars_5m"
            )
        if not df_opts_5m.empty:
            df_opts_5m = _filter_df_to_session(
                df_opts_5m, date_str, include_preopen_minute=include_preopen_minute, label="option_snapshots_5m"
            )

        df_bars_1s["ts_aligned"] = df_bars_1s["ts"].astype(int)
        df_opts_1s["ts_aligned"] = df_opts_1s["ts"].astype(int)
        if not df_bars_5m.empty:
            df_bars_5m["ts_aligned"] = df_bars_5m["ts"].astype(int)
        if not df_opts_5m.empty:
            df_opts_5m["ts_aligned"] = df_opts_5m["ts"].astype(int)

        map_b1 = self._to_bar_map(df_bars_1s)
        map_o1 = self._to_opt_map(df_opts_1s)
        map_b5 = self._to_bar_map(df_bars_5m)
        map_o5 = self._to_opt_map(df_opts_5m)

        all_ts = sorted(set(map_b1) | set(map_o1) | set(map_b5) | set(map_o5))
        all_ts = _filter_ts_keys_to_session(all_ts, date_str, include_preopen_minute=include_preopen_minute)
        all_ts = _limit_session_ticks(all_ts, max_session_bars)
        logger.info("Day %s: %d synchronized 1s ticks (cap_bars=%s)", date_str, len(all_ts), max_session_bars)
        if not all_ts:
            return 0

        return self._stream_from_maps(
            date_str,
            map_b1=map_b1,
            map_o1=map_o1,
            map_b5=map_b5,
            map_o5=map_o5,
            all_ts=all_ts,
            speed_factor=speed_factor,
            sync_mode=sync_mode,
            include_preopen_minute=include_preopen_minute,
            progress_every=progress_every,
        )

    def _stream_from_maps(
        self,
        date_str: str,
        *,
        map_b1: dict,
        map_o1: dict,
        map_b5: dict,
        map_o5: dict,
        all_ts: list[int],
        speed_factor: float,
        sync_mode: bool,
        include_preopen_minute: bool,
        progress_every: int,
    ) -> int:
        set_replay_start_ts(self.r, date_str, include_preopen_minute=include_preopen_minute)

        last_known: dict[str, dict] = {
            sym: {
                "ts": 0,
                "symbol": sym,
                "stock": {"open": 0.0, "high": 0.0, "low": 0.0, "close": 0.0, "volume": 0.0},
                "option_buckets": [],
                "option_contracts": [],
            }
            for sym in self.symbols
        }
        last_5m_state: dict[str, dict] = {}
        global_seq = 0
        count = 0
        total = len(all_ts)

        for ts_val in all_ts:
            self.r.set("replay:current_ts", str(ts_val))
            frame_complete = int(ts_val) % 60 == 59
            frame_id = str(int(ts_val))

            b1_ts = map_b1.get(ts_val, {})
            o1_ts = map_o1.get(ts_val, {})
            b5_ts = map_b5.get(ts_val, {})
            o5_ts = map_o5.get(ts_val, {})

            batch_payloads: list[dict] = []
            hset_mapping: dict[str, bytes] = {}

            for sym in self.symbols:
                payload = last_known[sym]
                payload["ts"] = ts_val
                global_seq += 1
                payload["frame_id"] = frame_id
                payload["seq"] = global_seq
                payload["frame_complete"] = frame_complete

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
                    opt_for_redis = opt_data if isinstance(opt_data, dict) else {"buckets": opt_data, "ts": ts_val}
                    opt_for_redis["ts"] = ts_val
                    hset_mapping[sym] = ser.pack(opt_for_redis)

                if sym in b5_ts or sym in o5_ts:
                    if sym not in last_5m_state:
                        last_5m_state[sym] = {}
                    if sym in b5_ts:
                        last_5m_state[sym]["stock_5m"] = b5_ts[sym]
                    if sym in o5_ts:
                        opt_data_5m = o5_ts[sym]
                        if isinstance(opt_data_5m, dict):
                            last_5m_state[sym]["option_buckets_5m"] = opt_data_5m.get("buckets", [])
                            last_5m_state[sym]["option_contracts_5m"] = opt_data_5m.get("contracts", [])
                        else:
                            last_5m_state[sym]["option_buckets_5m"] = opt_data_5m
                            last_5m_state[sym]["option_contracts_5m"] = []

                if sym in last_5m_state:
                    payload.update(last_5m_state[sym])

                last_known[sym] = payload
                batch_payloads.append(payload)

            if hset_mapping:
                self.r.hset(HASH_OPTION_SNAPSHOT, mapping=hset_mapping)
            if batch_payloads:
                self.r.xadd(STREAM_FUSED_MARKET, {"batch": ser.pack(batch_payloads)}, maxlen=10000)

            if sync_mode:
                self._wait_sync(ts_val, frame_id)

            if 0 < speed_factor < float("inf"):
                time.sleep(1.0 / speed_factor)

            count += 1
            if progress_every > 0 and count % progress_every == 0:
                logger.info("Progress %s: tick %d/%d ts=%s", date_str, count, total, frame_id)

        status_key = f"replay:status:{self.run_id}"
        self.r.set(status_key, f"DONE:{date_str}")
        logger.info("Day %s finished (%d ticks). status=%s", date_str, count, status_key)
        return count

    @staticmethod
    def _wait_sync(ts_val: float, frame_id: str, *, timeout_loops: int = 60000) -> None:
        r = _redis_client()
        loops = 0
        while True:
            ack_feat, ack_orch, ack_feat_fid, ack_orch_fid = r.mget(
                "sync:feature_calc_done",
                "sync:orch_done",
                "sync:feature_calc_done_frame_id",
                "sync:orch_done_frame_id",
            )
            feat_ts = float(ack_feat) if ack_feat else 0.0
            orch_ts = float(ack_orch) if ack_orch else 0.0
            feat_fid = ack_feat_fid.decode("utf-8") if isinstance(ack_feat_fid, bytes) else (str(ack_feat_fid or ""))
            orch_fid = ack_orch_fid.decode("utf-8") if isinstance(ack_orch_fid, bytes) else (str(ack_orch_fid or ""))

            if (feat_fid == frame_id and orch_fid == frame_id) or (feat_ts >= ts_val and orch_ts >= ts_val):
                return

            time.sleep(0.0005)
            loops += 1
            if loops >= timeout_loops:
                logger.warning(
                    "Sync timeout ts=%s frame=%s feat_ts=%.0f orch_ts=%.0f feat_fid=%s orch_fid=%s",
                    ts_val,
                    frame_id,
                    feat_ts,
                    orch_ts,
                    feat_fid,
                    orch_fid,
                )
                return

    def run(
        self,
        *,
        start_date: str | None = None,
        end_date: str | None = None,
        speed_factor: float = float("inf"),
        sync_mode: bool = True,
        include_preopen_minute: bool = False,
        progress_every: int = 600,
        max_session_bars: int | None = None,
    ) -> int:
        dbs = self._resolve_dbs(start_date, end_date)
        total = 0
        self.r.set("replay:status", "RUNNING")
        for i, db_path in enumerate(dbs):
            day_cap = max_session_bars if (i == len(dbs) - 1 and max_session_bars) else None
            total += self._stream_day(
                db_path,
                speed_factor=speed_factor,
                sync_mode=sync_mode,
                include_preopen_minute=include_preopen_minute,
                progress_every=progress_every,
                max_session_bars=day_cap,
            )
        self.r.set("replay:status", "DONE")
        logger.info("All days finished. total_ticks=%d run_id=%s", total, self.run_id)
        return total


def _align_timestamp_series(df: pd.DataFrame, col: str = "timestamp") -> pd.Series:
    if col not in df.columns and "ts" in df.columns:
        col = "ts"
    s = df[col]
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_datetime(s, unit="s", utc=True).dt.tz_convert(NY_TZ).dt.round("1s")
    return pd.to_datetime(s, utc=True).dt.tz_convert(NY_TZ).dt.round("1s")


def _load_stock_1s_day(
    sym: str,
    date_iso: str,
    *,
    stock_root: Path,
    stock_fallback_root: Path | None,
) -> pd.DataFrame:
    day_path = stock_root / sym / f"{sym}_{date_iso}.parquet"
    if day_path.exists():
        df = pd.read_parquet(day_path)
        if df.empty:
            return pd.DataFrame()
        df = df.copy()
        df["timestamp"] = _align_timestamp_series(df)
        df = df.set_index("timestamp").between_time("09:30", "16:00").reset_index()
        if df.empty:
            return pd.DataFrame()
        price_col = "close" if "close" in df.columns else "price"
        for c in ("open", "high", "low"):
            if c not in df.columns:
                df[c] = df[price_col]
        if "volume" not in df.columns:
            df["volume"] = 0.0
        had_vwap_col = "vwap" in df.columns
        if not had_vwap_col:
            # 1s tick 无独立 wap 时先占位;下面用分钟 fallback 覆写交易所分钟 wap
            df["vwap"] = df[price_col]
        df["symbol"] = sym
        df["ts_aligned"] = df["timestamp"].map(lambda t: int(t.timestamp()))
        out = df[["symbol", "ts_aligned", "open", "high", "low", "close", "volume", "vwap"]].copy()
        # raw 1s 常无真实分钟 wap(或 vwap≡close)。用 spnq 分钟 wap 覆写，
        # 对齐 offline feature_merge 的 vwap_log_return 口径。
        need_overlay = (not had_vwap_col) or bool(
            (out["vwap"] - out["close"]).abs().fillna(0.0).lt(1e-9).mean() > 0.95
        )
        if need_overlay and stock_fallback_root is not None:
            minute_wap = _load_minute_exchange_vwap_map(sym, date_iso, stock_fallback_root)
            if minute_wap:
                minute_ts = (
                    pd.to_datetime(out["ts_aligned"], unit="s", utc=True)
                    .dt.tz_convert(NY_TZ)
                    .dt.floor("min")
                    .map(lambda t: int(t.timestamp()))
                )
                overlay = minute_ts.map(minute_wap)
                valid = overlay.notna() & (overlay.astype(float) > 0.0)
                if valid.any():
                    out.loc[valid, "vwap"] = overlay.loc[valid].astype(float).values
                    logger.info(
                        "Stock 1s VWAP overlay from minute fallback %s %s: %d/%d secs",
                        sym,
                        date_iso,
                        int(valid.sum()),
                        len(out),
                    )
        return out

    if stock_fallback_root is None:
        logger.warning("No 1s stock file for %s %s and no fallback root", sym, date_iso)
        return pd.DataFrame()

    month = date_iso[:7]
    month_path = stock_fallback_root / f"{month}.parquet"
    if not month_path.exists():
        logger.warning("Stock fallback missing: %s", month_path)
        return pd.DataFrame()

    dfm = pd.read_parquet(month_path)
    ts = _align_timestamp_series(dfm)
    target = pd.Timestamp(date_iso).date()
    dfm = dfm.assign(timestamp=ts)
    dfm = dfm[dfm["timestamp"].dt.date == target].copy()
    if dfm.empty:
        return pd.DataFrame()

    dfm = dfm.set_index("timestamp").between_time("09:30", "16:00").reset_index()
    if dfm.empty:
        return pd.DataFrame()

    price_col = "close" if "close" in dfm.columns else "vwap"
    if price_col not in dfm.columns:
        logger.warning("Stock fallback %s has no close/vwap column", month_path)
        return pd.DataFrame()

    rows = []
    for row in dfm.itertuples(index=False):
        minute_ts = pd.Timestamp(getattr(row, "timestamp")).floor("min")
        o = float(getattr(row, "open", getattr(row, price_col, 0.0)) or getattr(row, price_col))
        h = float(getattr(row, "high", getattr(row, price_col)))
        l = float(getattr(row, "low", getattr(row, price_col)))
        c = float(getattr(row, price_col))
        vol = float(getattr(row, "volume", 0.0) if hasattr(row, "volume") else 0.0)
        # 行情 feed 自带的分钟 wap(交易所口径),作为数据源字段随 tick 下发(对应 IBKR bar.wap)
        wap = float(getattr(row, "vwap", c) or c) if hasattr(row, "vwap") else c
        per_sec_vol = vol / 60.0
        for sec in range(60):
            ts_ny = minute_ts + pd.Timedelta(seconds=sec)
            # 因果价格路径: 分钟开始时只知道 open，收盘价只能在最后一秒出现。
            # 旧实现整分钟平推 close，会把 bar 收盘价提前 59s 泄露给
            # last_tick_price → 期权 BSM spot 用到"未来"价，IV 偏差 ~1 vol 点。
            sec_close = o + (c - o) * (sec / 59.0) if sec < 59 else c
            rows.append(
                {
                    "symbol": sym,
                    "ts_aligned": int(ts_ny.timestamp()),
                    "open": o if sec == 0 else sec_close,
                    "high": h,
                    "low": l,
                    "close": sec_close,
                    "volume": per_sec_vol,
                    "vwap": wap,
                }
            )
    out = pd.DataFrame(rows)
    logger.info(
        "Stock 1min→1s fallback %s %s: %d rows from %s",
        sym,
        date_iso,
        len(out),
        month_path,
    )
    return out


def _load_minute_exchange_vwap_map(
    sym: str,
    date_iso: str,
    stock_fallback_root: Path,
) -> dict[int, float]:
    """spnq 分钟 parquet → {minute_unix_ts: exchange_vwap}。"""
    month = date_iso[:7]
    month_path = Path(stock_fallback_root) / f"{month}.parquet"
    if not month_path.exists():
        return {}
    dfm = pd.read_parquet(month_path)
    if dfm.empty or "vwap" not in dfm.columns:
        return {}
    ts = _align_timestamp_series(dfm)
    target = pd.Timestamp(date_iso).date()
    dfm = dfm.assign(timestamp=ts)
    dfm = dfm[dfm["timestamp"].dt.date == target].copy()
    if dfm.empty:
        return {}
    dfm = dfm.set_index("timestamp").between_time("09:30", "16:00")
    out: dict[int, float] = {}
    for ts_idx, row in dfm.iterrows():
        wap = float(row.get("vwap", 0.0) or 0.0)
        if wap > 0.0:
            out[int(pd.Timestamp(ts_idx).floor("min").timestamp())] = wap
    return out


def _resolve_greek_day_path(sym: str, date_iso: str, greek_root: Path) -> Path | None:
    from qqq_btc.common.option_minute_ref import resolve_greek_day_path

    return resolve_greek_day_path(sym, date_iso, greek_root)


def _load_minute_greeks_lookup(
    sym: str,
    date_iso: str,
    *,
    greek_root: Path,
) -> dict[tuple[int, int], dict[str, float]]:
    """分钟预计算 Greeks + volume → {(minute_ts, bucket_id): {iv,delta,...,volume}}。"""
    from qqq_btc.common.option_minute_ref import load_minute_option_ref

    out = load_minute_option_ref(sym, date_iso, greek_root=greek_root)
    if not out:
        logger.warning("Greek parity: no minute IV parquet for %s %s under %s", sym, date_iso, greek_root)
        return {}
    path = _resolve_greek_day_path(sym, date_iso, greek_root)
    logger.info(
        "Greek parity: loaded %d minute×bucket rows from %s",
        len(out),
        path,
    )
    return out


def _build_option_snapshots_1s(
    sym: str,
    df_day: pd.DataFrame,
    *,
    minute_greeks: dict[tuple[int, int], dict[str, float]] | None = None,
) -> dict[float, dict]:
    if df_day.empty or "bucket_id" not in df_day.columns:
        return {}

    df = df_day.copy()
    df["timestamp"] = _align_timestamp_series(df)
    df = df.set_index("timestamp").between_time("09:30", "16:00").reset_index()
    if df.empty:
        return {}

    all_seconds = pd.date_range(
        start=df["timestamp"].min(),
        end=df["timestamp"].max(),
        freq="1s",
    )
    keep_cols = [
        "mid_price",
        "price",
        "close",
        "ticker",
        "strike_price",
        "strike",
        "volume",
        "bid",
        "ask",
        "bid_size",
        "ask_size",
    ]
    existing_cols = [c for c in keep_cols if c in df.columns]
    # 哪些 (floor_minute, bucket) 在该分钟确有真实 quote（对齐离线 volume 不 ffill）
    real_minute_buckets: set[tuple[int, int]] = set()
    for row in df.itertuples(index=False):
        b_id = int(row.bucket_id)
        if not (0 <= b_id <= 5):
            continue
        m_start = int(pd.Timestamp(row.timestamp).floor("min").timestamp())
        real_minute_buckets.add((m_start, b_id))

    parts = []
    for b_id, group in df.groupby("bucket_id"):
        # 先 last 再 ffill：盘口状态可延续；volume 另按 real_minute_buckets 门控
        resampled = (
            group.set_index("timestamp")[existing_cols]
            .resample("1s")
            .last()
            .reindex(all_seconds)
            .ffill()
        )
        resampled["bucket_id"] = int(b_id)
        parts.append(resampled.reset_index().rename(columns={"index": "timestamp"}))
    if not parts:
        return {}

    df_aligned = pd.concat(parts, ignore_index=True)
    out: dict[float, dict] = defaultdict(dict)
    for ts_idx, sec_df in df_aligned.groupby("timestamp"):
        ts_unix = _ts_key(pd.Timestamp(ts_idx).timestamp())
        buckets = np.zeros((6, 12), dtype=float)
        contracts = [""] * 6
        minute_start = int(pd.Timestamp(ts_idx).floor("min").timestamp())
        for row in sec_df.itertuples(index=False):
            b_id = int(row.bucket_id)
            if not (0 <= b_id <= 5):
                continue
            price = float(
                getattr(row, "mid_price", getattr(row, "price", getattr(row, "close", 0.0)))
            )
            strike = float(getattr(row, "strike_price", getattr(row, "strike", 0.0)))
            bid = float(getattr(row, "bid", price))
            ask = float(getattr(row, "ask", price))
            bid_size = float(getattr(row, "bid_size", 0.0) or 0.0)
            ask_size = float(getattr(row, "ask_size", 0.0) or 0.0)
            # Greeks/IV：day_iv / offline feature 用 ceil 结束标签。
            # FCS alpha_label_ts=T 对应股票 bar [T,T+60) ≡ offline stamp T+60；
            # 秒级落在 [T,T+60) 时必须注入 day_iv[T+60]，若用 floor=T 会拿到上一根
            # day_iv[T]，导致同排 debug_slow 上 options 比 stock 慢 1 分钟。
            greek: dict = {}
            end_label = int(minute_start) + 60
            if minute_greeks:
                # 精确匹配 end-label；缺则回退 floor（开盘首分钟等）
                greek = (
                    minute_greeks.get((end_label, b_id))
                    or minute_greeks.get((minute_start, b_id))
                    or {}
                )
                # 离线对拍：bid/ask/mid 冻结为 day_iv 分钟收盘
                d_bid = float(greek.get("bid", 0.0) or 0.0)
                d_ask = float(greek.get("ask", 0.0) or 0.0)
                d_mid = float(greek.get("mid", 0.0) or greek.get("close", 0.0) or 0.0)
                if d_bid > 1e-6 and d_ask > 1e-6:
                    bid, ask = d_bid, d_ask
                    price = d_mid if d_mid > 1e-6 else 0.5 * (d_bid + d_ask)
                    bs = float(greek.get("bid_size", 0.0) or 0.0)
                    asz = float(greek.get("ask_size", 0.0) or 0.0)
                    if bs > 0.0:
                        bid_size = bs
                    if asz > 0.0:
                        ask_size = asz
            # 离线 locked_feature：volume 不跨分钟 ffill；无真实 quote 的分钟 volume=0
            if (minute_start, b_id) in real_minute_buckets:
                tick_vol = float(getattr(row, "volume", 0.0) or 0.0)
                if tick_vol <= 0.0:
                    tick_vol = float(greek.get("volume", 0.0) or 0.0) if greek else 0.0
                if tick_vol <= 0.0:
                    tick_vol = max(bid_size, 0.0) + max(ask_size, 0.0)
            else:
                tick_vol = 0.0
            buckets[b_id] = [
                price,
                float(greek.get("delta", 0.0)),
                float(greek.get("gamma", 0.0)),
                float(greek.get("vega", 0.0)),
                float(greek.get("theta", 0.0)),
                strike,
                tick_vol,
                float(greek.get("iv", 0.0)),
                bid,
                ask,
                bid_size,
                ask_size,
            ]
            contracts[b_id] = str(getattr(row, "ticker", "") or "")
        out[ts_unix][sym] = {"buckets": buckets.tolist(), "contracts": contracts}
    return dict(out)


def _resample_bar_map(map_b1: dict, *, freq_min: int) -> dict:
    if not map_b1:
        return {}
    rows = []
    for ts_val, sym_map in map_b1.items():
        for sym, bar in sym_map.items():
            rows.append({"ts": float(ts_val), "symbol": sym, **bar})
    if not rows:
        return {}
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["ts"], unit="s", utc=True).dt.tz_convert(NY_TZ)
    out: dict[float, dict] = defaultdict(dict)
    for sym, grp in df.groupby("symbol"):
        g = grp.set_index("timestamp").resample(f"{freq_min}min").agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        ).dropna()
        for ts_idx, row in g.iterrows():
            out[_ts_key(ts_idx.timestamp())][sym] = {
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row["volume"]),
            }
    return dict(out)


def _resample_opt_map(map_o1: dict, *, freq_min: int) -> dict:
    if not map_o1:
        return {}
    out: dict[float, dict] = defaultdict(dict)
    bins: dict[tuple[str, int], list[tuple[float, dict]]] = defaultdict(list)
    for ts_val, sym_map in map_o1.items():
        ts = pd.Timestamp(float(ts_val), unit="s", tz=NY_TZ)
        bin_key = int(ts.floor(f"{freq_min}min").timestamp())
        for sym, payload in sym_map.items():
            bins[(sym, bin_key)].append((float(ts_val), payload))
    for (sym, bin_ts), items in bins.items():
        last_ts, payload = max(items, key=lambda x: x[0])
        out[float(bin_ts)][sym] = payload
    return dict(out)


class RawParquetPitcher1s(FusedPitcher1s):
    """从 raw_1s parquet 逐秒推送 fused batch。"""

    def __init__(
        self,
        *,
        option_root: Path | None = None,
        stock_root: Path | None = None,
        stock_fallback_root: Path | None = None,
        stock_fallback_roots: dict[str, Path] | None = None,
        greek_root: Path | None = None,
        greek_parity: bool = False,
        symbols: list[str] | None = None,
        run_id: str | None = None,
    ):
        super().__init__(symbols=symbols, run_id=run_id)
        self.option_root = Path(option_root or DEFAULT_OPTION_ROOT)
        self.stock_root = Path(stock_root or DEFAULT_STOCK_ROOT)
        self.stock_fallback_root = (
            Path(stock_fallback_root)
            if stock_fallback_root is not None
            else DEFAULT_STOCK_FALLBACK_ROOT
        )
        roots = dict(DEFAULT_STOCK_FALLBACKS)
        if stock_fallback_roots:
            roots.update({k: Path(v) for k, v in stock_fallback_roots.items()})
        if stock_fallback_root is not None:
            roots["QQQ"] = Path(stock_fallback_root)
        self.stock_fallback_roots = roots
        self.greek_root = Path(greek_root or DEFAULT_GREEK_ROOT)
        self.greek_parity = bool(greek_parity)

    def _resolve_dates(self, start_date: str | None, end_date: str | None) -> list[str]:
        sym = self.symbols[0]
        opt_dir = self.option_root / sym
        if not opt_dir.exists():
            raise FileNotFoundError(f"option root not found: {opt_dir}")
        dates = []
        for p in sorted(opt_dir.glob(f"{sym}_*.parquet")):
            iso = p.stem.split("_", 1)[1]
            ymd = iso.replace("-", "")
            if start_date and ymd < _normalize_yyyymmdd(start_date):
                continue
            if end_date and ymd > _normalize_yyyymmdd(end_date):
                continue
            dates.append(ymd)
        if not dates:
            raise FileNotFoundError(
                f"no option parquet in [{start_date}, {end_date}] under {opt_dir}"
            )
        return dates

    def _load_day_maps(self, ymd: str) -> tuple[dict, dict, dict, dict]:
        date_iso = _iso_from_yyyymmdd(ymd)
        map_b1: dict[float, dict] = defaultdict(dict)
        map_o1: dict[float, dict] = defaultdict(dict)

        for sym in self.symbols:
            opt_path = self.option_root / sym / f"{sym}_{date_iso}.parquet"
            if opt_path.exists():
                df_opt = pd.read_parquet(opt_path)
                # raw_1s 无 volume → 始终尝试分钟 IV parquet 注入 volume/Greeks
                minute_greeks = _load_minute_greeks_lookup(
                    sym,
                    date_iso,
                    greek_root=self.greek_root,
                ) or None
                snaps = _build_option_snapshots_1s(
                    sym,
                    df_opt,
                    minute_greeks=minute_greeks,
                )
                for ts_val, per_sym in snaps.items():
                    payload = per_sym.get(sym)
                    if payload:
                        map_o1[_ts_key(ts_val)][sym] = payload
            else:
                logger.warning("Option parquet missing: %s", opt_path)

            df_stock = _load_stock_1s_day(
                sym,
                date_iso,
                stock_root=self.stock_root,
                stock_fallback_root=self.stock_fallback_roots.get(sym),
            )
            if not df_stock.empty:
                has_vwap = "vwap" in df_stock.columns
                for row in df_stock.itertuples(index=False):
                    map_b1[_ts_key(row.ts_aligned)][sym] = {
                        "open": float(row.open),
                        "high": float(row.high),
                        "low": float(row.low),
                        "close": float(row.close),
                        "volume": float(row.volume),
                        "vwap": float(row.vwap) if has_vwap else float(row.close),
                    }

        map_b5 = _resample_bar_map(map_b1, freq_min=5)
        map_o5 = _resample_opt_map(map_o1, freq_min=5)
        return dict(map_b1), dict(map_o1), map_b5, map_o5

    def run(
        self,
        *,
        start_date: str | None = None,
        end_date: str | None = None,
        speed_factor: float = float("inf"),
        sync_mode: bool = True,
        include_preopen_minute: bool = False,
        progress_every: int = 600,
        max_session_bars: int | None = None,
    ) -> int:
        dates = self._resolve_dates(start_date, end_date)
        total = 0
        self.r.set("replay:status", "RUNNING")
        for i, ymd in enumerate(dates):
            date_iso = _iso_from_yyyymmdd(ymd)
            day_cap = max_session_bars if (i == len(dates) - 1 and max_session_bars) else None
            logger.info("Streaming raw day %s from %s", date_iso, self.option_root)
            map_b1, map_o1, map_b5, map_o5 = self._load_day_maps(ymd)
            all_ts = _session_ts_list(ymd, include_preopen_minute=include_preopen_minute)
            all_ts = _limit_session_ticks(all_ts, day_cap)
            logger.info(
                "Day %s: session grid=%d stock_ticks=%d option_ticks=%d cap_bars=%s",
                date_iso,
                len(all_ts),
                len(map_b1),
                len(map_o1),
                day_cap,
            )
            if not map_o1 and not map_b1:
                logger.warning("Skip %s: no stock/option data loaded", date_iso)
                continue
            total += self._stream_from_maps(
                ymd,
                map_b1=map_b1,
                map_o1=map_o1,
                map_b5=map_b5,
                map_o5=map_o5,
                all_ts=all_ts,
                speed_factor=speed_factor,
                sync_mode=sync_mode,
                include_preopen_minute=include_preopen_minute,
                progress_every=progress_every,
            )
        self.r.set("replay:status", "DONE")
        logger.info("All raw days finished. total_ticks=%d run_id=%s", total, self.run_id)
        return total


def create_pitcher(
    source: str,
    *,
    db_dir: Path | None = None,
    option_root: Path | None = None,
    stock_root: Path | None = None,
    stock_fallback_root: Path | None = None,
    stock_fallback_roots: dict[str, Path] | None = None,
    greek_root: Path | None = None,
    greek_parity: bool = False,
    symbols: list[str] | None = None,
    run_id: str | None = None,
) -> FusedPitcher1s:
    src = (source or "auto").lower()
    if src == "auto":
        sym = (symbols or TARGET_SYMBOLS)[0]
        probe = None
        if option_root:
            probe = Path(option_root)
        else:
            probe = DEFAULT_OPTION_ROOT / sym
        src = "raw" if probe.exists() else "sqlite"
        logger.info("Pitcher source auto → %s (probe=%s)", src, probe)
    if src == "raw":
        return RawParquetPitcher1s(
            option_root=option_root,
            stock_root=stock_root,
            stock_fallback_root=stock_fallback_root,
            stock_fallback_roots=stock_fallback_roots,
            greek_root=greek_root,
            greek_parity=greek_parity,
            symbols=symbols,
            run_id=run_id,
        )
    if src == "sqlite":
        return FusedPitcher1s(db_dir=db_dir, symbols=symbols, run_id=run_id)
    raise ValueError(f"unknown pitcher source: {source}")


def _parse_date_arg(value: str | None) -> tuple[str | None, str | None]:
    if not value:
        return None, None
    if "," in value:
        parts = [p.strip() for p in value.split(",") if p.strip()]
        if len(parts) == 1:
            return parts[0], parts[0]
        return parts[0], parts[-1]
    return value, value


def main() -> int:
    parser = argparse.ArgumentParser(description="1s Redis fused_market_stream pitcher")
    parser.add_argument("--date", type=str, default=None, help="YYYYMMDD / YYYY-MM-DD or start,end")
    parser.add_argument("--start-date", type=str, default=None)
    parser.add_argument("--end-date", type=str, default=None)
    parser.add_argument(
        "--source",
        choices=("auto", "raw", "sqlite"),
        default="auto",
        help="auto=优先 raw parquet; raw=options_databento_v3; sqlite=history_sqlite_1s",
    )
    parser.add_argument("--db-dir", type=str, default=None)
    parser.add_argument(
        "--option-root",
        type=str,
        default=str(DEFAULT_OPTION_ROOT),
        help="raw 期权根目录,如 /mnt/s990/data/raw_1s/options_databento_v3",
    )
    parser.add_argument("--stock-root", type=str, default=str(DEFAULT_STOCK_ROOT))
    parser.add_argument(
        "--stock-fallback-root",
        type=str,
        default=str(DEFAULT_STOCK_FALLBACK_ROOT),
        help="无 1s 股票文件时用 1min 特征 parquet 展开",
    )
    parser.add_argument(
        "--greek-parity",
        action="store_true",
        help="从 day_iv 注入分钟 Greeks/IV，并冻结 bid/ask 为 ceil 分钟收盘（离线对拍）",
    )
    parser.add_argument(
        "--greek-root",
        type=str,
        default=str(DEFAULT_GREEK_ROOT),
        help="分钟 IV/Greeks parquet 根目录",
    )
    parser.add_argument("--symbols", type=str, default=None, help="Comma-separated, default TARGET_SYMBOLS")
    parser.add_argument("--speed", type=float, default=float("inf"), help="1.0 = realtime 1Hz; inf = max")
    parser.add_argument("--no-sync", action="store_true", help="Do not wait for FCS/SE sync ack")
    parser.add_argument("--no-reset-redis", action="store_true")
    parser.add_argument("--include-preopen", action="store_true")
    parser.add_argument("--progress-every", type=int, default=600)
    parser.add_argument(
        "--max-session-bars",
        type=int,
        default=None,
        help="仅截断最后一个交易日的 session bar 数(快速首小时诊断,如 30)",
    )
    parser.add_argument("--run-id", type=str, default=None)
    args = parser.parse_args()

    if args.date:
        d0, d1 = _parse_date_arg(args.date)
        args.start_date = args.start_date or d0
        args.end_date = args.end_date or d1

    if not args.start_date:
        parser.error("Provide --date or --start-date")

    args.start_date = _normalize_yyyymmdd(args.start_date)
    if args.end_date:
        args.end_date = _normalize_yyyymmdd(args.end_date)

    os.environ.setdefault("RUN_MODE", "REALTIME_DRY")
    if args.greek_parity:
        os.environ["GREEK_PARITY_MODE"] = "1"
        os.environ["RECALC_GREEKS"] = "0"
    symbols = [s.strip() for s in args.symbols.split(",")] if args.symbols else None
    r = _redis_client()
    run_id = init_replay_redis(r, run_id=args.run_id, reset=not args.no_reset_redis)

    pitcher = create_pitcher(
        args.source,
        db_dir=Path(args.db_dir) if args.db_dir else None,
        option_root=Path(args.option_root) if args.option_root else None,
        stock_root=Path(args.stock_root) if args.stock_root else None,
        stock_fallback_root=Path(args.stock_fallback_root) if args.stock_fallback_root else None,
        greek_root=Path(args.greek_root) if args.greek_root else None,
        greek_parity=args.greek_parity,
        symbols=symbols,
        run_id=run_id,
    )
    total = pitcher.run(
        start_date=args.start_date,
        end_date=args.end_date or args.start_date,
        speed_factor=args.speed,
        sync_mode=not args.no_sync,
        include_preopen_minute=args.include_preopen,
        progress_every=args.progress_every,
        max_session_bars=args.max_session_bars,
    )
    logger.info("Pitcher exit | ticks=%d redis_db=%d", total, get_redis_db())
    return 0 if total >= 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

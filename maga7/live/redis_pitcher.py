"""Mag7 1s fused_market_stream pitcher — New_Pro bus, Mag7 data roots.

Publishes the same ``xadd(fused_market_stream, {batch: msgpack})`` shape as
``New_Pro/.../ibkr_connector_v8.py`` / ``qqq_btc/tools/redis_fused_pitcher_1s.py``,
but loads Mag7 open-ladder option quotes + multi-symbol stock 1s.
"""
from __future__ import annotations

import logging
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pytz

from maga7.common.bar_agg import load_stock_1s_day
from maga7.live.redis_fused import (
    init_maga7_redis,
    pack_batch,
    pack_obj,
    redis_client,
    run_keys,
)

logger = logging.getLogger("maga7.live.redis_pitcher")
NY = pytz.timezone("America/New_York")


def _session_bounds(date: str) -> tuple[int, int]:
    start = NY.localize(datetime.strptime(f"{date} 09:30:00", "%Y-%m-%d %H:%M:%S"))
    end = NY.localize(datetime.strptime(f"{date} 16:00:00", "%Y-%m-%d %H:%M:%S"))
    return int(start.timestamp()), int(end.timestamp())


def _contract_row(row: Any, ticker_col: str | None) -> dict[str, Any]:
    return {
        "localSymbol": str(getattr(row, ticker_col)).replace("O:", "") if ticker_col else "",
        "bid": float(row.bid) if hasattr(row, "bid") else None,
        "ask": float(row.ask) if hasattr(row, "ask") else None,
        "mid": float(row.mid_price) if hasattr(row, "mid_price") else None,
        "strike": float(row.strike) if hasattr(row, "strike") else None,
        "tag": str(row.tag) if hasattr(row, "tag") else "",
        "bucket_id": int(row.bucket_id) if hasattr(row, "bucket_id") else None,
    }


def _option_delta(prev: list[dict[str, Any]], curr: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Only contracts whose bid/ask changed (or are new)."""
    prev_m = {str(c.get("localSymbol") or ""): c for c in prev or []}
    out: list[dict[str, Any]] = []
    for c in curr or []:
        key = str(c.get("localSymbol") or "")
        p = prev_m.get(key)
        if p is None or p.get("bid") != c.get("bid") or p.get("ask") != c.get("ask"):
            out.append(c)
    return out


def _load_option_map(quote_root: Path, symbol: str, date: str) -> dict[int, list[dict[str, Any]]]:
    """ts → list of option_contracts for Mag7 quote parquet."""
    path = Path(quote_root) / symbol / f"{symbol}_{date}.parquet"
    if not path.is_file():
        return {}
    df = pd.read_parquet(path)
    if df.empty:
        return {}
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"])
        if getattr(ts.dt, "tz", None) is None:
            ts = ts.dt.tz_localize(NY)
        else:
            ts = ts.dt.tz_convert(NY)
    else:
        ts = pd.to_datetime(df["ts"], unit="s", utc=True).dt.tz_convert(NY)
    df = df.copy()
    df["_ts_key"] = ts.map(lambda x: int(pd.Timestamp(x).timestamp()))
    ticker_col = "ticker" if "ticker" in df.columns else None
    out: dict[int, list[dict[str, Any]]] = {}
    for key, g in df.groupby("_ts_key", sort=False):
        contracts = [_contract_row(row, ticker_col) for row in g.itertuples(index=False)]
        out[int(key)] = contracts
    return out


class Mag7FusedPitcher:
    """Push Mag7 stock (+ option) 1s frames onto fused_market_stream."""

    def __init__(
        self,
        *,
        symbols: list[str],
        stock_1s_root: Path,
        quote_1s_root: Path | None = None,
        host: str = "127.0.0.1",
        port: int = 6379,
        db: int = 1,
        run_id: str | None = None,
        publish_options: bool = True,
    ):
        self.symbols = list(symbols)
        self.stock_1s_root = Path(stock_1s_root)
        self.quote_1s_root = Path(quote_1s_root) if quote_1s_root else None
        self.publish_options = bool(publish_options) and self.quote_1s_root is not None
        self.r = redis_client(host=host, port=port, db=db)
        self.run_id = run_id

    def init_redis(self, *, reset: bool = True) -> str:
        self.run_id = init_maga7_redis(self.r, run_id=self.run_id, reset=reset)
        return self.run_id

    def _load_day_maps(self, date: str) -> tuple[dict[int, dict[str, dict]], dict[int, dict[str, list]]]:
        map_stock: dict[int, dict[str, dict]] = defaultdict(dict)
        map_opt: dict[int, dict[str, list]] = defaultdict(dict)
        for sym in self.symbols:
            sdf = load_stock_1s_day(self.stock_1s_root, sym, date)
            if not sdf.empty:
                for r in sdf.itertuples(index=False):
                    ts = int(pd.Timestamp(r.timestamp).timestamp())
                    map_stock[ts][sym] = {
                        "open": float(r.open),
                        "high": float(r.high),
                        "low": float(r.low),
                        "close": float(r.close),
                        "volume": float(getattr(r, "volume", 0.0) or 0.0),
                    }
            if self.publish_options and self.quote_1s_root is not None:
                om = _load_option_map(self.quote_1s_root, sym, date)
                for ts, contracts in om.items():
                    map_opt[ts][sym] = contracts
        return dict(map_stock), dict(map_opt)

    def _wait_sync(self, ts_val: float, frame_id: str, *, timeout_loops: int | None = None) -> None:
        if not self.run_id:
            raise RuntimeError("init_redis() must run before streaming")
        keys = run_keys(self.run_id)
        # Options payloads are heavier — allow ~10 min before warning.
        if timeout_loops is None:
            timeout_loops = 1_200_000 if self.publish_options else 120_000
        loops = 0
        while True:
            done_ts, done_fid = self.r.mget(keys["ack_ts"], keys["ack_frame"])
            fid = done_fid.decode() if isinstance(done_fid, bytes) else str(done_fid or "")
            if fid == frame_id:
                return
            time.sleep(0.0005)
            loops += 1
            if loops >= timeout_loops:
                raise TimeoutError(
                    f"Mag7 sync timeout run={self.run_id} ts={ts_val} frame={frame_id}"
                )

    def stream_day(
        self,
        date: str,
        *,
        speed: float = float("inf"),
        sync: bool = False,
        progress_every: int = 600,
        max_seconds: int | None = None,
    ) -> int:
        if not self.run_id:
            raise RuntimeError("init_redis() must run before stream_day()")
        keys = run_keys(self.run_id)
        map_stock, map_opt = self._load_day_maps(date)
        start_ts, end_ts = _session_bounds(date)
        all_ts = list(range(start_ts, end_ts))
        if max_seconds and max_seconds > 0:
            all_ts = all_ts[: int(max_seconds)]

        self.r.set(keys["clock"], str(start_ts))
        self.r.set(keys["status"], f"RUNNING:{date}")
        last_known: dict[str, dict] = {
            sym: {
                "ts": 0,
                "symbol": sym,
                "stock": {"open": 0.0, "high": 0.0, "low": 0.0, "close": 0.0, "volume": 0.0},
                "option_contracts": [],
                "option_buckets": [],
            }
            for sym in self.symbols
        }
        last_opt_wire: dict[str, list] = {sym: [] for sym in self.symbols}

        count = 0
        total = len(all_ts)
        logger.info(
            "Pitch Mag7 day=%s ticks=%d stock_keys=%d opt_keys=%d speed=%s sync=%s",
            date,
            total,
            len(map_stock),
            len(map_opt),
            speed,
            sync,
        )

        for ts_val in all_ts:
            self.r.set(keys["clock"], str(ts_val))
            frame_id = f"{self.run_id}:{int(ts_val)}"
            stock_ts = map_stock.get(ts_val, {})
            opt_ts = map_opt.get(ts_val, {})
            batch: list[dict[str, Any]] = []
            hset: dict[str, bytes] = {}
            # Full option snapshot every 60s so late subscribers can catch up.
            force_full_opt = int(ts_val) % 60 == 0

            for sym in self.symbols:
                payload = last_known[sym]
                payload["ts"] = ts_val
                payload["frame_id"] = frame_id
                payload["run_id"] = self.run_id
                payload["frame_complete"] = int(ts_val) % 60 == 59
                payload["source"] = "maga7_s5"

                if sym in stock_ts:
                    payload["stock"] = stock_ts[sym]
                elif float(payload["stock"].get("close") or 0) > 0:
                    c = float(payload["stock"]["close"])
                    payload["stock"] = {"open": c, "high": c, "low": c, "close": c, "volume": 0.0}

                wire = dict(payload)
                if sym in opt_ts:
                    full = opt_ts[sym]
                    payload["option_contracts"] = full
                    delta = full if force_full_opt else _option_delta(last_opt_wire[sym], full)
                    last_opt_wire[sym] = full
                    wire["option_contracts"] = delta
                    if force_full_opt or delta:
                        hset[sym] = pack_obj({"ts": ts_val, "contracts": full, "symbol": sym})
                else:
                    wire["option_contracts"] = []

                last_known[sym] = payload
                # only publish symbols that have ever seen a stock print
                if float(payload["stock"].get("close") or 0) > 0:
                    batch.append(wire)

            if hset:
                try:
                    self.r.hset(keys["option_snapshot"], mapping=hset)
                except Exception as exc:
                    logger.warning("hset option snapshot failed: %s", exc)
            if batch:
                # No maxlen trim during replay — consumer must see every second.
                self.r.xadd(keys["stream"], {"batch": pack_batch(batch)})

            if sync:
                self._wait_sync(float(ts_val), frame_id)
            if 0 < speed < float("inf"):
                time.sleep(1.0 / speed)

            count += 1
            if progress_every > 0 and count % progress_every == 0:
                logger.info("Pitch progress %s %d/%d", date, count, total)

        self.r.set(keys["status"], f"DAY_DONE:{date}")
        logger.info("Pitch day %s done ticks=%d", date, count)
        return count

    def run(
        self,
        dates: list[str],
        *,
        speed: float = float("inf"),
        sync: bool = False,
        progress_every: int = 600,
        max_seconds: int | None = None,
    ) -> int:
        if not self.run_id:
            raise RuntimeError("init_redis() must run before run()")
        keys = run_keys(self.run_id)
        total = 0
        self.r.set(keys["status"], "RUNNING")
        for date in dates:
            total += self.stream_day(
                date,
                speed=speed,
                sync=sync,
                progress_every=progress_every,
                max_seconds=max_seconds if date == dates[-1] else None,
            )
        self.r.set(keys["status"], f"DONE:{dates[-1] if dates else ''}")
        return total

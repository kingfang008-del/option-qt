"""Mag7 Redis consumer — fused_market_stream → scanner → OMS (S5).

Mirrors live topology (IBKR → Redis → strategy) without FCS/TFT.
Each Redis message is handled as one atomic frame:
options → existing-position exits → all stock ticks → new entries.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import pandas as pd
import pytz

from maga7.live.oms_stub import Mag7OmsStub
from maga7.live.redis_fused import (
    ack_maga7_frame,
    run_keys,
    unpack_batch,
)
from maga7.live.scanner import Mag7Scanner, ScannerSignal

logger = logging.getLogger("maga7.live.redis_consumer")
NY = pytz.timezone("America/New_York")


@dataclass
class Mag7RedisScannerLoop:
    """xreadgroup fused batches → options book → on_stock_second → OMS."""

    profile: dict[str, Any]
    redis: Any
    scanner: Mag7Scanner
    run_id: str
    stub: Mag7OmsStub | None = None
    consumer_name: str = "maga7_s5"
    block_ms: int = 200
    option_quotes: dict[tuple[str, str], dict[str, float]] = field(default_factory=dict)
    n_batches: int = 0
    n_ticks: int = 0
    n_option_prints: int = 0
    n_duplicate_frames: int = 0
    n_foreign_frames: int = 0
    n_rejected_frames: int = 0
    last_ts: float = 0.0
    seen_frame_ids: set[str] = field(default_factory=set)
    stop: bool = False

    @classmethod
    def from_profile(
        cls,
        profile: dict[str, Any],
        redis_client: Any,
        *,
        run_id: str,
        scheme: str = "m5_circuit",
        stub: Mag7OmsStub | None = None,
        consumer_name: str = "maga7_s5",
    ) -> "Mag7RedisScannerLoop":
        scanner = Mag7Scanner.from_profile(profile, scheme=scheme)
        loop = cls(
            profile=profile,
            redis=redis_client,
            scanner=scanner,
            run_id=run_id,
            stub=stub,
            consumer_name=consumer_name,
        )
        if stub is not None:
            stub.scanner = scanner
        # Frame coordinator commits entries only after all exits/stocks.
        scanner.on_signal = None
        return loop

    @property
    def keys(self) -> dict[str, str]:
        return run_keys(self.run_id)

    def _ensure_group(self) -> None:
        try:
            self.redis.xgroup_create(
                self.keys["stream"], self.keys["group"], id="0-0", mkstream=True
            )
        except Exception:
            pass

    def _ingest_options(self, payload: dict[str, Any]) -> None:
        sym = str(payload.get("symbol") or "")
        ts_raw = payload.get("ts")
        if not sym or ts_raw is None:
            return
        ts = float(ts_raw)
        contracts = payload.get("option_contracts") or []
        if self.stub is not None:
            self.stub.ingest_option_contracts(
                sym, ts, contracts, resolve_pending=False
            )
        for c in contracts:
            if not isinstance(c, dict):
                continue
            local = str(c.get("localSymbol") or c.get("ticker") or "")
            bid, ask = c.get("bid"), c.get("ask")
            if not local or bid is None or ask is None:
                continue
            self.option_quotes[(sym, local)] = {"bid": float(bid), "ask": float(ask), "ts": ts}
            self.n_option_prints += 1

    def _ingest_stock(self, payload: dict[str, Any]) -> ScannerSignal | None:
        sym = str(payload.get("symbol") or "")
        if not sym:
            return None
        stock = payload.get("stock") or {}
        ts_raw = payload.get("ts")
        if ts_raw is None:
            return None
        ts = float(ts_raw)
        self.last_ts = ts
        dt = datetime.fromtimestamp(ts, tz=NY)
        close = float(stock.get("close") or 0.0)
        if close <= 0:
            return None
        tick = {
            "timestamp": pd.Timestamp(dt),
            "open": float(stock.get("open") or close),
            "high": float(stock.get("high") or close),
            "low": float(stock.get("low") or close),
            "close": close,
            "volume": float(stock.get("volume") or 0.0),
        }
        sig = self.scanner.on_stock_second(sym, tick)
        self.n_ticks += 1
        return sig

    def _handle_message(self, mid: bytes | str, fields: dict) -> None:
        raw = fields.get(b"batch") if isinstance(fields, dict) else None
        if raw is None and isinstance(fields, dict):
            raw = fields.get("batch")
        if raw is None or raw == b"1" or raw == "1":
            return
        batch = unpack_batch(raw)
        if not batch:
            self.n_rejected_frames += 1
            return

        run_ids = {str(p.get("run_id") or "") for p in batch if isinstance(p, dict)}
        if run_ids != {self.run_id}:
            self.n_foreign_frames += 1
            logger.error("Reject foreign frame run_ids=%s expected=%s", run_ids, self.run_id)
            return
        frame_ids = {str(p.get("frame_id") or "") for p in batch if isinstance(p, dict)}
        ts_values = {float(p.get("ts")) for p in batch if isinstance(p, dict) and p.get("ts") is not None}
        symbols = [str(p.get("symbol") or "") for p in batch if isinstance(p, dict)]
        if len(frame_ids) != 1 or "" in frame_ids or len(ts_values) != 1:
            self.n_rejected_frames += 1
            raise RuntimeError(
                f"incoherent frame run={self.run_id} frame_ids={frame_ids} ts={ts_values}"
            )
        if len(symbols) != len(set(symbols)):
            self.n_rejected_frames += 1
            raise RuntimeError(f"duplicate symbols in frame {frame_ids}")

        frame_id = next(iter(frame_ids))
        ts_val = next(iter(ts_values))
        if frame_id in self.seen_frame_ids:
            self.n_duplicate_frames += 1
            ack_maga7_frame(
                self.redis, run_id=self.run_id, ts_val=ts_val, frame_id=frame_id
            )
            return
        if self.last_ts and ts_val <= self.last_ts:
            self.n_rejected_frames += 1
            raise RuntimeError(
                f"out-of-order frame run={self.run_id} ts={ts_val} last={self.last_ts}"
            )

        # Phase 1: make all same-second option quotes visible.
        for payload in batch:
            if not isinstance(payload, dict):
                continue
            self._ingest_options(payload)

        # Phase 2: close existing positions before evaluating new entries.
        if self.stub is not None and self.stub.prefer_redis_quotes:
            self.stub.try_resolve_pending(asof_ts=ts_val)

        # Phase 3: update all stock states without callback side effects.
        signals: list[ScannerSignal] = []
        for payload in batch:
            if not isinstance(payload, dict):
                continue
            sig = self._ingest_stock(payload)
            if sig is not None:
                signals.append(sig)

        # Phase 4: commit entries after the complete cross-symbol frame.
        if self.stub is not None:
            for sig in signals:
                self.stub.process_one(sig)

        self.seen_frame_ids.add(frame_id)
        self.last_ts = ts_val
        self.n_batches += 1
        ack_maga7_frame(
            self.redis, run_id=self.run_id, ts_val=ts_val, frame_id=frame_id
        )

    def poll_once(self) -> int:
        self._ensure_group()
        resp = self.redis.xreadgroup(
            self.keys["group"],
            self.consumer_name,
            {self.keys["stream"]: ">"},
            count=50,
            block=self.block_ms,
        )
        if not resp:
            return 0
        n = 0
        for _stream, messages in resp:
            for mid, fields in messages:
                self._handle_message(mid, fields)
                try:
                    self.redis.xack(self.keys["stream"], self.keys["group"], mid)
                except Exception:
                    pass
                n += 1
        return n

    def run_until_done(
        self,
        *,
        idle_sec: float = 2.0,
        max_wall_sec: float | None = None,
    ) -> dict[str, Any]:
        t0 = time.time()
        idle_since = None
        while not self.stop:
            n = self.poll_once()
            status = self.redis.get(self.keys["status"])
            status_s = status.decode() if isinstance(status, bytes) else str(status or "")
            if n > 0:
                idle_since = None
            else:
                if idle_since is None:
                    idle_since = time.time()
                elif status_s.startswith("DONE") and (time.time() - idle_since) >= idle_sec:
                    break
            if max_wall_sec is not None and (time.time() - t0) >= max_wall_sec:
                logger.warning("Mag7 consumer wall timeout %.0fs", max_wall_sec)
                break
        flushed = self.scanner.flush_seconds()
        if self.stub is not None:
            for sig in flushed:
                self.stub.process_one(sig)
        if self.stub is not None and getattr(self.stub, "prefer_redis_quotes", False):
            self.stub.flush_pending()
        summary = {
            "n_batches": self.n_batches,
            "n_ticks": self.n_ticks,
            "n_option_prints": self.n_option_prints,
            "n_duplicate_frames": self.n_duplicate_frames,
            "n_foreign_frames": self.n_foreign_frames,
            "n_rejected_frames": self.n_rejected_frames,
            "n_unique_frames": len(self.seen_frame_ids),
            "n_signals": len(self.scanner.signals),
            "last_ts": self.last_ts,
            "elapsed_sec": time.time() - t0,
        }
        if self.stub is not None:
            stub_sum = self.stub.finalize_summary(n_signals=len(self.scanner.signals))
            summary.update(stub_sum)
        return summary

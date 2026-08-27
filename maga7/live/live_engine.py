"""Async Redis consumer joining live fused frames, scanner, and broker OMS."""
from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from maga7.live.redis_fused import run_keys, unpack_batch
from maga7.live.redis_fused import pack_obj
from maga7.live.scanner_state import scanner_snapshot, write_scanner_snapshot

logger = logging.getLogger("maga7.live.engine")


@dataclass
class LiveEngineMetrics:
    frames: int = 0
    rejected: int = 0
    duplicates: int = 0
    foreign: int = 0
    signals: int = 0
    last_frame_ts: float = 0.0
    last_frame_id: str = ""


class Mag7LiveFrameEngine:
    def __init__(
        self,
        *,
        redis_client: Any,
        session_id: str,
        scanner: Any,
        oms: Any,
        connector: Any,
        consumer_name: str,
    ):
        self.redis = redis_client
        self.session_id = session_id
        self.keys = run_keys(session_id)
        self.scanner = scanner
        self.oms = oms
        self.connector = connector
        self.consumer_name = consumer_name
        self.metrics = LiveEngineMetrics()
        self.seen: set[str] = set()
        self._stop = asyncio.Event()
        self._last_disk_snapshot = 0.0
        self._last_redis_snapshot = 0.0
        self._snapshot_interval_sec = 5.0
        self._restore_progress()
        self._ensure_group()

    @property
    def health_key(self) -> str:
        return f"maga7:live_engine:{self.session_id}"

    @property
    def scanner_state_key(self) -> str:
        return f"maga7:scanner_state:{self.session_id}"

    @property
    def dead_letter_key(self) -> str:
        return f"maga7:live_dead_letter:{self.session_id}"

    def _ensure_group(self) -> None:
        try:
            self.redis.xgroup_create(
                self.keys["stream"],
                self.keys["group"],
                # Connector publishes while contracts are prepared/locked.
                # A fresh live engine must start at the current stream tail;
                # replaying that startup backlog makes valid old frames trip
                # the runtime stale-frame fail-closed guard.
                id="$",
                mkstream=True,
            )
        except Exception as exc:
            if "BUSYGROUP" not in str(exc):
                raise
            # Connector/init may have created the group earlier. For a fresh
            # engine (no restored progress) jump to the live tail.
            if float(self.metrics.last_frame_ts or 0.0) <= 0.0:
                try:
                    self.redis.xgroup_setid(
                        self.keys["stream"], self.keys["group"], id="$"
                    )
                except Exception as set_exc:
                    logger.warning("xgroup setid $ skipped: %s", set_exc)

    def _restore_progress(self) -> None:
        raw = self.redis.hgetall(self.health_key) or {}

        def _get(name: str) -> str:
            value = raw.get(name.encode()) if name.encode() in raw else raw.get(name)
            return value.decode() if isinstance(value, bytes) else str(value or "")

        frame_id = _get("last_frame_id")
        frame_ts = _get("last_frame_ts")
        if frame_id:
            self.seen.add(frame_id)
            self.metrics.last_frame_id = frame_id
        try:
            self.metrics.last_frame_ts = float(frame_ts or 0.0)
        except ValueError:
            self.metrics.last_frame_ts = 0.0

    def _append_signal_audit(self, signal: Any) -> None:
        """Append one accepted signal so mid-session tape parity can see it."""
        try:
            path = Path(self.oms.session_dir) / "signals.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = (
                signal.to_orch_payload()
                if hasattr(signal, "to_orch_payload")
                else {
                    "sig_ts": getattr(signal, "sig_ts", None),
                    "symbol": getattr(signal, "symbol", None),
                    "direction": getattr(signal, "direction", None),
                    "contract": getattr(signal, "contract", None),
                }
            )
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
        except Exception as exc:
            logger.warning("signal audit append failed: %s", exc)

    def _publish_health(self, state: str, error: str = "") -> None:
        payload = {
            **self.metrics.__dict__,
            "state": state,
            "error": error,
            "session_id": self.session_id,
            "consumer": self.consumer_name,
            "updated_at": time.time(),
        }
        self.redis.hset(
            self.health_key,
            mapping={
                key: json.dumps(value) if isinstance(value, (dict, list)) else str(value)
                for key, value in payload.items()
            },
        )

    def _quarantine_message(
        self, message_id: Any, fields: dict[Any, Any], exc: Exception
    ) -> None:
        raw = fields.get(b"batch") if b"batch" in fields else fields.get("batch")
        pipe = self.redis.pipeline(transaction=True)
        pipe.xadd(
            self.dead_letter_key,
            {
                "original_id": (
                    message_id.decode()
                    if isinstance(message_id, bytes)
                    else str(message_id)
                ),
                "error": f"{type(exc).__name__}: {exc}",
                "failed_at": str(time.time()),
                "batch": raw or b"",
            },
            maxlen=10_000,
            approximate=True,
        )
        pipe.xack(self.keys["stream"], self.keys["group"], message_id)
        pipe.execute()

    def _fail_closed_on_processing_error(self, exc: Exception) -> None:
        # Startup backlog / late delivery is already quarantined. Permanent
        # day halt is reserved for true processing faults, not stale catch-up.
        if "stale live frame" in str(exc):
            if hasattr(self.oms, "_note_gap_event"):
                self.oms._note_gap_event()
            logger.warning("skip stale live frame without day halt: %s", exc)
            return
        self.oms.day_halted = True
        if hasattr(self.oms, "_note_gap_event"):
            self.oms._note_gap_event()
        try:
            self.oms.force_flatten("GAP_FLATTEN")
        except Exception:
            logger.exception("force flatten after frame rejection failed: %s", exc)

    def _ingest_options(self, payload: dict[str, Any], frame_ts: float) -> None:
        symbol = str(payload.get("symbol") or "").upper()
        for row in payload.get("option_contracts") or []:
            local = str(
                row.get("localSymbol")
                or row.get("ticker")
                or row.get("contract")
                or ""
            ).replace("O:", "").strip()
            bid = float(row.get("bid") or row.get("b") or 0.0)
            ask = float(row.get("ask") or row.get("a") or 0.0)
            if local and ask >= bid > 0:
                self.connector.option_quotes[(symbol, local)] = {
                    **row,
                    "ts": float(row.get("ts") or frame_ts),
                    "bid": bid,
                    "ask": ask,
                    "mid": (bid + ask) / 2.0,
                }

    def _process_message(self, message_id: Any, fields: dict[Any, Any]) -> None:
        raw = fields.get(b"batch") if b"batch" in fields else fields.get("batch")
        batch = unpack_batch(raw)
        if not isinstance(batch, list) or not batch:
            raise ValueError("empty live fused batch")
        run_ids = {str(row.get("run_id") or "") for row in batch if isinstance(row, dict)}
        if run_ids != {self.session_id}:
            self.metrics.foreign += 1
            raise ValueError(f"foreign run ids: {run_ids}")
        frame_ids = {str(row.get("frame_id") or "") for row in batch}
        frame_ts_values = {float(row.get("ts") or 0.0) for row in batch}
        symbols = [str(row.get("symbol") or "").upper() for row in batch]
        if len(frame_ids) != 1 or len(frame_ts_values) != 1:
            raise ValueError("mixed frame id or timestamp")
        if len(symbols) != len(set(symbols)):
            raise ValueError("duplicate symbol in frame")
        frame_id = next(iter(frame_ids))
        frame_ts = next(iter(frame_ts_values))
        risk_cfg = getattr(self.oms, "risk_cfg", None)
        max_frame_age = float(getattr(risk_cfg, "max_signal_age_sec", 0.0) or 0.0)
        if max_frame_age > 0:
            frame_age = time.time() - frame_ts
            max_future_skew = float(
                getattr(risk_cfg, "max_future_skew_sec", 1.0) or 1.0
            )
            if frame_age > max_frame_age or frame_age < -max_future_skew:
                raise ValueError(
                    f"stale live frame age={frame_age:.3f}s "
                    f"limit={max_frame_age:.3f}s"
                )
        if frame_id in self.seen:
            self.metrics.duplicates += 1
            self.redis.xack(self.keys["stream"], self.keys["group"], message_id)
            return
        if self.metrics.last_frame_ts and frame_ts <= self.metrics.last_frame_ts:
            raise ValueError(
                f"out-of-order frame {frame_ts} <= {self.metrics.last_frame_ts}"
            )
        if (
            self.metrics.last_frame_ts
            and frame_ts > self.metrics.last_frame_ts + 1.0
        ):
            if hasattr(self.oms, "_note_gap_event"):
                self.oms._note_gap_event(now=frame_ts)
            if hasattr(self.oms, "_entry_quote_stable"):
                self.oms._entry_quote_stable.clear()
            if hasattr(self.oms, "_last_entry_quote_ts"):
                self.oms._last_entry_quote_ts.clear()
            if hasattr(self.oms, "_event"):
                self.oms._event(
                    "FEED_FRAME_GAP",
                    {
                        "previous_frame_ts": self.metrics.last_frame_ts,
                        "frame_ts": frame_ts,
                        "gap_sec": frame_ts - self.metrics.last_frame_ts,
                    },
                )

        # Same-second option quotes become visible before stock/minute decisions.
        # Drop callback-time/future quotes; OMS may only see quotes serialized in
        # this completed frame.
        self.connector.option_quotes = {}
        for payload in batch:
            self._ingest_options(payload, frame_ts)

        def _stock_tick(payload: dict[str, Any]) -> tuple[str, dict[str, Any]] | None:
            stock = payload.get("stock") or {}
            if not stock:
                return None
            symbol = str(payload["symbol"]).upper()
            tick = {
                "timestamp": pd.Timestamp(frame_ts, unit="s", tz="UTC").tz_convert(
                    "America/New_York"
                ),
                "open": float(stock.get("open") or stock.get("close") or 0.0),
                "high": float(stock.get("high") or stock.get("close") or 0.0),
                "low": float(stock.get("low") or stock.get("close") or 0.0),
                "close": float(stock.get("close") or 0.0),
                "volume": float(stock.get("volume") or 0.0),
                "previous_close": float(stock.get("previous_close") or 0.0),
            }
            return symbol, tick

        # Complete reference bars first so a Mag7 decision for minute M sees
        # the completed QQQ/VIXY minute M, never M-1 due to payload ordering.
        for payload in batch:
            item = _stock_tick(payload)
            if item is None:
                continue
            symbol, tick = item
            last_ticks = getattr(self.connector, "last_stock_tick", None)
            if isinstance(last_ticks, dict):
                last_ticks[symbol] = float(frame_ts)
            if symbol not in self.scanner.states:
                gate = getattr(self.scanner, "regime_gate", None)
                if gate is not None and hasattr(gate, "on_stock_second"):
                    gate.on_stock_second(symbol, tick)
                # Feed QQQ completed 1m into scanner.stock_by for Watchdog/Hunt.
                if hasattr(self.scanner, "on_reference_second"):
                    self.scanner.on_reference_second(symbol, tick)

        signals = []
        n_sig_before = len(getattr(self.scanner, "signals", []) or [])
        for payload in batch:
            item = _stock_tick(payload)
            if item is None:
                continue
            symbol, tick = item
            if symbol not in self.scanner.states:
                continue
            previous_close = float(tick.get("previous_close") or 0.0)
            state = self.scanner.states.get(symbol)
            if (
                state is not None
                and getattr(state, "prev_close", None) is None
                and previous_close > 0
            ):
                state.prev_close = previous_close
            signal = self.scanner.on_stock_second(symbol, tick)
            if signal is not None:
                signals.append(signal)
        # Hunt emits land in scanner.signals; also drain by frame clock.
        for sig in list(getattr(self.scanner, "signals", []) or [])[n_sig_before:]:
            if sig not in signals:
                signals.append(sig)
        frame_ts_ny = pd.Timestamp(frame_ts, unit="s", tz="UTC").tz_convert(
            "America/New_York"
        )
        if hasattr(self.scanner, "drain_hunts"):
            for sig in self.scanner.drain_hunts(frame_ts_ny):
                if sig not in signals:
                    signals.append(sig)
        if hasattr(self.scanner, "drain_open_cont"):
            for sig in self.scanner.drain_open_cont(frame_ts_ny):
                if sig not in signals:
                    signals.append(sig)
        if hasattr(self.scanner, "drain_am_pulse"):
            for sig in self.scanner.drain_am_pulse(frame_ts_ny):
                if sig not in signals:
                    signals.append(sig)
        if hasattr(self.scanner, "drain_am_pulse_extension"):
            for sig in self.scanner.drain_am_pulse_extension(frame_ts_ny):
                if sig not in signals:
                    signals.append(sig)
        if hasattr(self.scanner, "drain_am_v2"):
            for sig in self.scanner.drain_am_v2(frame_ts_ny):
                if sig not in signals:
                    signals.append(sig)

        # Scanner states now include every completed minute in this cross-symbol
        # frame. Resolve exits first, then admit new entries.
        self.oms.evaluate_exits(frame_ts)
        for signal in signals:
            if self.oms.process_signal(signal):
                self.metrics.signals += 1
                self._append_signal_audit(signal)

        self.seen.add(frame_id)
        self.metrics.frames += 1
        self.metrics.last_frame_ts = frame_ts
        self.metrics.last_frame_id = frame_id
        # Keep the IB asyncio loop light: snapshot+msgpack+fsync every N seconds
        # (not every fused frame). Otherwise pendingTickers stalls for 10–15s.
        now = time.time()
        do_snapshot = (
            hasattr(self.scanner, "day_fires")
            and (now - self._last_redis_snapshot) >= self._snapshot_interval_sec
        )
        state_payload = None
        if do_snapshot:
            state_payload = {
                **scanner_snapshot(self.scanner),
                "session_id": self.session_id,
                "frame_id": frame_id,
                "frame_ts": frame_ts,
                "live_fingerprint": self.oms.profile_hash,
            }
        pipe = self.redis.pipeline(transaction=True)
        pipe.xack(self.keys["stream"], self.keys["group"], message_id)
        pipe.set(self.keys["ack_ts"], str(frame_ts))
        pipe.set(self.keys["ack_frame"], frame_id)
        if state_payload is not None:
            pipe.set(self.scanner_state_key, pack_obj(state_payload))
        pipe.execute()
        if state_payload is not None:
            write_scanner_snapshot(
                self.oms.session_dir / "scanner_state.json",
                state_payload,
            )
            prev = state_payload.get("prevention")
            if isinstance(prev, dict):
                try:
                    path = Path(self.oms.session_dir) / "prevention.json"
                    path.write_text(
                        json.dumps(prev, indent=2, ensure_ascii=False, default=str),
                        encoding="utf-8",
                    )
                except Exception:
                    logger.exception("failed to write prevention.json")
            self._last_disk_snapshot = now
            self._last_redis_snapshot = now
        # Health is cheap; still useful every frame for Dash L0.
        self._publish_health("RUNNING")

    def _claim_stale(self) -> None:
        try:
            result = self.redis.xautoclaim(
                self.keys["stream"],
                self.keys["group"],
                self.consumer_name,
                min_idle_time=0,
                start_id="0-0",
                count=100,
            )
            messages = result[1] if isinstance(result, (tuple, list)) and len(result) > 1 else []
            for message_id, fields in messages:
                try:
                    self._process_message(message_id, fields)
                except Exception as exc:
                    self.metrics.rejected += 1
                    self._fail_closed_on_processing_error(exc)
                    self._quarantine_message(message_id, fields, exc)
                    self._publish_health("DEGRADED", error=str(exc))
        except Exception as exc:
            # Redis < 6.2 has no XAUTOCLAIM; a new live session normally starts
            # with a fresh group, so this is observable but not fatal.
            logger.warning("pending claim skipped: %s", exc)

    def _read_messages(self, block_ms: int = 1000):
        return self.redis.xreadgroup(
            self.keys["group"],
            self.consumer_name,
            {self.keys["stream"]: ">"},
            count=10,
            block=block_ms,
        )

    def _process_messages(self, messages) -> int:
        count = 0
        for _, entries in messages or []:
            for message_id, fields in entries:
                try:
                    self._process_message(message_id, fields)
                except Exception as exc:
                    self.metrics.rejected += 1
                    self._fail_closed_on_processing_error(exc)
                    self._quarantine_message(message_id, fields, exc)
                    self._publish_health("DEGRADED", error=str(exc))
                    continue
                count += 1
        return count

    def poll_once(self, block_ms: int = 1000) -> int:
        return self._process_messages(self._read_messages(block_ms))

    async def run(self) -> None:
        self._claim_stale()
        self._publish_health("RUNNING")
        while not self._stop.is_set():
            messages = await asyncio.to_thread(self._read_messages, 1000)
            # Process on the IB asyncio thread; placeOrder/cancelOrder and
            # ib_insync events are not safe from the Redis worker thread.
            # Yield between messages so pendingTickers/publish_loop stay live
            # during catch-up after a stall.
            for _, entries in messages or []:
                for message_id, fields in entries:
                    if self._stop.is_set():
                        return
                    try:
                        self._process_message(message_id, fields)
                    except Exception as exc:
                        self.metrics.rejected += 1
                        self._fail_closed_on_processing_error(exc)
                        self._quarantine_message(message_id, fields, exc)
                        self._publish_health("DEGRADED", error=str(exc))
                        continue
                    await asyncio.sleep(0)

    def stop(self) -> None:
        self._stop.set()
        self._publish_health("STOPPED")

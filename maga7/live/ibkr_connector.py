"""Mag7 IBKR market-data connector using the S5 fused payload contract."""
from __future__ import annotations

import asyncio
import json
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import redis

from maga7.live.live_contract_lock import (
    LiveOpenLadderLockService,
    LockedContract,
    atomic_write_lock_manifest,
    locked_contract_identity_ok,
)
from maga7.live.option_quote_diagnose import diagnose_missing_option_quotes
from maga7.live.redis_fused import init_maga7_redis, pack_batch, pack_obj, run_keys
from maga7.live.session_phase import (
    is_rth_authority_phase,
    session_phase,
    tape_phase_dir,
    tape_symbol_path,
)

logger = logging.getLogger("maga7.live.ibkr_connector")


@dataclass(frozen=True)
class Mag7IbkrConfig:
    host: str = "127.0.0.1"
    port: int = 4002
    client_id: int = 212
    account: str = ""
    redis_host: str = "127.0.0.1"
    redis_port: int = 6379
    redis_db: int = 0
    market_data_type: int = 1
    publish_interval_sec: float = 1.0
    max_stock_staleness_sec: float = 2.0
    max_option_staleness_sec: float = 5.0
    max_option_subscriptions: int = 90
    preferred_dte: int = 0
    # combined = legacy one-process; stock/options = split MD processes
    md_role: str = "combined"


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


class Mag7IbkrConnector:
    """Stocks/options → run-scoped Redis fused frames.

    The connector never places orders. Delayed/frozen data is observable but
    cannot arm the live OMS.
    """

    def __init__(
        self,
        *,
        session_id: str,
        symbols: list[str],
        reference_symbols: list[str] | None = None,
        trade_date: str,
        session_dir: Path,
        config: Mag7IbkrConfig,
        allowed_dte: tuple[int, ...] = (0, 1, 2),
        otm_rungs: int = 3,
        resume: bool = False,
        ib: Any | None = None,
    ):
        if ib is None:
            from ib_insync import IB

            ib = IB()
        self.ib = ib
        self.session_id = session_id
        self.trade_symbols = [str(symbol).upper() for symbol in symbols]
        refs = [str(symbol).upper() for symbol in (reference_symbols or [])]
        self.reference_symbols = [symbol for symbol in refs if symbol not in self.trade_symbols]
        self.symbols = self.trade_symbols + self.reference_symbols
        self.trade_date = trade_date
        self.session_dir = Path(session_dir)
        self.config = config
        self.redis = redis.Redis(
            host=config.redis_host,
            port=config.redis_port,
            db=config.redis_db,
            decode_responses=False,
        )
        init_maga7_redis(
            self.redis,
            run_id=session_id,
            reset=not resume,
            md_role=str(getattr(config, "md_role", "combined") or "combined"),
        )
        self.keys = run_keys(session_id)
        self.md_role = str(getattr(config, "md_role", "combined") or "combined").lower()
        self.lock_service = LiveOpenLadderLockService(
            self.ib,
            allowed_dte=allowed_dte,
            otm_rungs=otm_rungs,
        )
        self.stock_contracts: dict[str, Any] = {}
        self.option_contracts: dict[int, Any] = {}
        self.locks: dict[str, list[LockedContract]] = {}
        self.subscribed_option_ids: set[int] = set()
        self.initial_option_ids: set[int] = set()
        self.stock_bars: dict[str, dict[str, float]] = {}
        self.stock_bar_history: dict[str, dict[int, dict[str, float]]] = {}
        self.stock_total_volume: dict[str, float] = {}
        self.stock_previous_close: dict[str, float] = {}
        self.option_quotes: dict[tuple[str, str], dict[str, float]] = {}
        self.option_quote_history: dict[
            tuple[str, str], dict[int, dict[str, float]]
        ] = {}
        self.last_stock_tick: dict[str, float] = {}
        self.data_mode = "UNKNOWN"
        self.connected = False
        self.lock_status = "PENDING"
        self.errors: dict[str, str] = {}
        self.prepared_option_candidates: dict[str, list[Any]] = {}
        self.prepare_errors: dict[str, str] = {}
        self.prelock_completed_at: float = 0.0
        self.partial_frame_drops = 0
        self._partial_targets: set[int] = set()
        self.last_published_second = 0  # RTH authority stream clock
        self.last_validation_second = 0  # PRE/POST validation stream clock
        self.validation_publishes = 0
        self.tape_writes = 0
        self.option_tape_frames = 0
        self.option_tape_quotes = 0
        self._stop = asyncio.Event()
        self._callbacks_bound = False
        self.on_reconnect: Any | None = None
        self.reconnect_count = 0

    def _subscribe_locked_contracts(self) -> None:
        preferred = []
        for symbol in self.trade_symbols:
            symbol_locks = self.locks.get(symbol, [])
            active_dte = (
                self.config.preferred_dte
                if any(lock.front_dte == self.config.preferred_dte for lock in symbol_locks)
                else min((lock.front_dte for lock in symbol_locks), default=-1)
            )
            for lock in symbol_locks:
                if lock.front_dte == active_dte:
                    preferred.append(lock)
        for lock in preferred:
            if len(self.subscribed_option_ids) >= self.config.max_option_subscriptions:
                break
            self.ensure_option_subscription(lock.con_id)
        self.initial_option_ids = set(self.subscribed_option_ids)

    async def restore_locks(self) -> bool:
        path = self.session_dir / "locks.json"
        if not path.is_file():
            return False
        from ib_insync import Option

        import json

        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("session_id") != self.session_id:
            raise RuntimeError("lock manifest session mismatch")
        if payload.get("trade_date") != self.trade_date:
            raise RuntimeError("lock manifest trade date mismatch")
        if payload.get("status") != "LOCKED":
            raise RuntimeError("only a complete LOCKED manifest can be restored")
        restored: dict[str, list[LockedContract]] = {}
        contracts = []
        for symbol, values in (payload.get("locks") or {}).items():
            restored[symbol] = [LockedContract(**value) for value in values]
            for lock in restored[symbol]:
                if not locked_contract_identity_ok(lock):
                    raise RuntimeError(
                        "lock contract identity mismatch: "
                        f"{lock.symbol} expiry={lock.expiry} local={lock.local_symbol}"
                    )
                contract = Option(
                    lock.symbol,
                    lock.expiry,
                    lock.strike,
                    lock.right,
                    lock.exchange,
                    currency=lock.currency,
                )
                contract.conId = lock.con_id
                contract.localSymbol = lock.local_symbol
                contracts.append(contract)
        if contracts:
            await self.ib.qualifyContractsAsync(*contracts)
        self.option_contracts = {
            int(contract.conId): contract
            for contract in contracts
            if int(getattr(contract, "conId", 0) or 0) > 0
        }
        self.locks = restored
        self.lock_status = str(payload.get("status") or "LOCKED")
        if set(self.locks) != set(self.trade_symbols):
            raise RuntimeError("restored lock symbols incomplete")
        if not all(self.locks.values()):
            raise RuntimeError("restored lock contains empty symbol")
        self._subscribe_locked_contracts()
        self.publish_status("LOCKS_RESTORED")
        return self.lock_status == "LOCKED"

    def _feed_snapshot(self, now: float | None = None) -> dict[str, Any]:
        """Per-symbol stock/option freshness for Dash / ops alerts."""
        now = float(now if now is not None else time.time())
        stock_feed: dict[str, Any] = {}
        for symbol in self.symbols:
            last = float(self.last_stock_tick.get(symbol, 0.0) or 0.0)
            stock_feed[symbol] = {
                "last_ts": last,
                "lag_sec": round(now - last, 3) if last > 0 else None,
                "spot": float(self._stock_spot(symbol) or 0.0),
                "subscribed": symbol in self.stock_contracts,
            }
        option_feed: dict[str, Any] = {}
        for symbol in self.trade_symbols:
            quotes = [
                quote
                for (underlying, _), quote in self.option_quotes.items()
                if underlying == symbol
            ]
            locked_n = len(self.locks.get(symbol) or [])
            if not quotes:
                option_feed[symbol] = {
                    "n_quotes": 0,
                    "n_locked": locked_n,
                    "last_ts": 0.0,
                    "lag_sec": None,
                }
                continue
            last = max(float(quote.get("ts") or 0.0) for quote in quotes)
            option_feed[symbol] = {
                "n_quotes": len(quotes),
                "n_locked": locked_n,
                "last_ts": last,
                "lag_sec": round(now - last, 3) if last > 0 else None,
            }
        phase = session_phase(now, trade_date=self.trade_date)
        live_syms = sum(
            1
            for row in stock_feed.values()
            if row.get("lag_sec") is not None and float(row["lag_sec"]) <= 30.0
        )
        return {
            "ts": now,
            "session_id": self.session_id,
            "md_role": getattr(self, "md_role", "combined"),
            "data_mode": self.data_mode,
            "connected": bool(self.ib.isConnected()),
            "session_phase": phase,
            "max_stock_staleness_sec": float(self.config.max_stock_staleness_sec),
            "max_option_staleness_sec": float(self.config.max_option_staleness_sec),
            "stock_feed": stock_feed,
            "option_feed": option_feed,
            "stock_live_symbols": live_syms,
            "partial_frame_drops": self.partial_frame_drops,
            "validation_publishes": self.validation_publishes,
            "tape_writes": self.tape_writes,
            "option_tape_frames": self.option_tape_frames,
            "option_tape_quotes": self.option_tape_quotes,
            "stream": self.keys["stream"],
            "stream_stock": self.keys.get("stream_stock"),
            "stream_pre": self.keys.get("stream_pre"),
            "stream_post": self.keys.get("stream_post"),
            "tape_dir": str(tape_phase_dir(self.session_dir, phase)),
        }

    def publish_feed_health(self, now: float | None = None) -> None:
        payload = self._feed_snapshot(now)
        packed = pack_obj(payload)
        role = str(getattr(self, "md_role", "combined") or "combined").lower()
        pipe = self.redis.pipeline(transaction=True)
        if role in {"combined", "stock"}:
            pipe.set(f"maga7:feed_health_stock:{self.session_id}", packed)
        if role in {"combined", "options"}:
            pipe.set(f"maga7:feed_health_option:{self.session_id}", packed)
        # Combined view for Dash: merge stock+option role snapshots when split.
        merged = self._merged_feed_health(payload)
        merged_packed = pack_obj(merged)
        pipe.set(f"maga7:feed_health:{self.session_id}", merged_packed)
        pipe.hset(
            f"live_ibkr_connector:maga7:{self.session_id}",
            "feed_health",
            merged_packed,
        )
        pipe.execute()

    def _merged_feed_health(self, local: dict[str, Any]) -> dict[str, Any]:
        role = str(getattr(self, "md_role", "combined") or "combined").lower()
        if role == "combined":
            return local
        stock = dict(local)
        option = dict(local)
        try:
            from maga7.live.redis_fused import unpack_obj

            if role == "options":
                raw = self.redis.get(f"maga7:feed_health_stock:{self.session_id}")
                if raw:
                    stock = unpack_obj(raw) or stock
            elif role == "stock":
                raw = self.redis.get(f"maga7:feed_health_option:{self.session_id}")
                if raw:
                    option = unpack_obj(raw) or option
        except Exception:
            pass
        merged = dict(local)
        merged["md_role"] = "split"
        merged["stock_feed"] = (stock or {}).get("stock_feed") or local.get("stock_feed")
        merged["option_feed"] = (option or {}).get("option_feed") or local.get(
            "option_feed"
        )
        merged["stock_live_symbols"] = (stock or {}).get("stock_live_symbols")
        merged["stock_md"] = {
            "connected": (stock or {}).get("connected"),
            "data_mode": (stock or {}).get("data_mode"),
            "lag_hint": (stock or {}).get("ts"),
        }
        merged["option_md"] = {
            "connected": (option or {}).get("connected"),
            "data_mode": (option or {}).get("data_mode"),
            "lag_hint": (option or {}).get("ts"),
        }
        return merged

    def _status_payload(self, state: str, note: str = "", error: str = "") -> dict[str, Any]:
        quote_symbols = {symbol for symbol, _ in self.option_quotes}
        feed = self._feed_snapshot()
        return {
            "ts": feed["ts"],
            "session_id": self.session_id,
            "trade_date": self.trade_date,
            "state": state,
            "connected": bool(self.ib.isConnected()),
            "host": self.config.host,
            "port": int(self.config.port),
            "client_id": int(self.config.client_id),
            "md_role": getattr(self, "md_role", "combined"),
            "account": self.config.account or "",
            "data_mode": self.data_mode,
            "lock_status": self.lock_status,
            "stocks": len(self.stock_contracts),
            "trade_symbols": len(self.trade_symbols),
            "locked_contracts": sum(len(value) for value in self.locks.values()),
            "prepared_symbols": len(self.prepared_option_candidates),
            "prepared_contracts": sum(
                len(value) for value in self.prepared_option_candidates.values()
            ),
            "prelock_completed_at": self.prelock_completed_at,
            "option_subscriptions": len(self.subscribed_option_ids),
            "option_quotes": len(self.option_quotes),
            "option_quote_symbols": len(quote_symbols),
            "partial_frame_drops": self.partial_frame_drops,
            "validation_publishes": self.validation_publishes,
            "tape_writes": self.tape_writes,
            "session_phase": feed.get("session_phase"),
            "stream": self.keys["stream"],
            "stream_pre": self.keys.get("stream_pre"),
            "stream_post": self.keys.get("stream_post"),
            "tape_dir": feed.get("tape_dir"),
            "stock_live_symbols": feed.get("stock_live_symbols"),
            "note": note,
            "last_error": error,
            "stock_feed": feed["stock_feed"],
            "option_feed": feed["option_feed"],
            "max_stock_staleness_sec": feed["max_stock_staleness_sec"],
            "max_option_staleness_sec": feed["max_option_staleness_sec"],
        }

    def publish_status(self, state: str, note: str = "", error: str = "") -> None:
        payload = self._status_payload(state, note=note, error=error)
        packed = pack_obj(payload)
        feed_packed = pack_obj(self._feed_snapshot(payload.get("ts")))
        pipe = self.redis.pipeline(transaction=True)
        pipe.hset(f"live_ibkr_connector:maga7:{self.session_id}", "status", packed)
        pipe.hset(f"live_ibkr_connector:maga7:{self.session_id}", "feed_health", feed_packed)
        pipe.set(f"maga7:feed_health:{self.session_id}", feed_packed)
        # Compatibility projection for the unified/legacy dashboards.
        pipe.hset("live_ibkr_connector", "maga7_status", packed)
        pipe.set(self.keys["status"], state)
        pipe.execute()

    def _bind_callbacks(self) -> None:
        if self._callbacks_bound:
            return
        self.ib.pendingTickersEvent += self._on_pending_tickers
        self.ib.errorEvent += self._on_error
        self.ib.disconnectedEvent += self._on_disconnected
        self._callbacks_bound = True

    async def connect(self, retries: int = 0) -> None:
        self._bind_callbacks()
        attempt = 0
        while not self.ib.isConnected():
            try:
                self.publish_status("CONNECTING")
                await self.ib.connectAsync(
                    self.config.host,
                    self.config.port,
                    clientId=self.config.client_id,
                )
                accounts = list(self.ib.managedAccounts() or [])
                if self.config.account and self.config.account not in accounts:
                    self.ib.disconnect()
                    raise RuntimeError(
                        f"IBKR account {self.config.account!r} not in managed accounts"
                    )
                self.ib.reqMarketDataType(int(self.config.market_data_type))
                self.data_mode = {
                    1: "LIVE",
                    2: "FROZEN",
                    3: "DELAYED",
                    4: "DELAYED_FROZEN",
                }.get(int(self.config.market_data_type), "UNKNOWN")
                self.connected = True
                self.publish_status("CONNECTED")
                return
            except Exception as exc:
                attempt += 1
                self.publish_status("CONNECT_FAILED", error=str(exc))
                if retries and attempt >= retries:
                    raise
                await asyncio.sleep(min(30.0, 2.0 * attempt))

    async def subscribe_stocks(self) -> None:
        if str(getattr(self, "md_role", "combined")).lower() == "options":
            # Stock ticks come from the external stock publisher via Redis.
            # Still qualify underlyings so option chain discovery has conIds.
            await self._ensure_stock_contracts(self.trade_symbols)
            self.publish_status("STOCKS_EXTERNAL")
            return
        from ib_insync import Stock

        queries = [Stock(symbol, "SMART", "USD") for symbol in self.symbols]
        qualified = await self.ib.qualifyContractsAsync(*queries)
        for contract in qualified:
            symbol = str(contract.symbol).replace(" ", ".").upper()
            self.stock_contracts[symbol] = contract
            self.ib.reqMktData(contract, "100,233,236", False, False)
        missing = sorted(set(self.symbols) - set(self.stock_contracts))
        if missing:
            raise RuntimeError(f"IBKR stock qualification missing: {missing}")
        self.publish_status("STOCKS_SUBSCRIBED")

    async def _ensure_stock_contracts(self, symbols: list[str] | None = None) -> None:
        """Qualify underlyings without requiring a live stock MD subscription."""
        from ib_insync import Stock

        want = [
            str(symbol).upper()
            for symbol in (symbols or list(self.trade_symbols) or list(self.symbols) or [])
            if str(symbol).strip()
        ]
        missing = [symbol for symbol in want if symbol not in self.stock_contracts]
        if not missing:
            return
        queries = [Stock(symbol, "SMART", "USD") for symbol in missing]
        try:
            qualified = await self.ib.qualifyContractsAsync(*queries)
        except Exception:
            logger.exception("stock qualify failed symbols=%s", missing)
            return
        for contract in qualified or []:
            symbol = str(getattr(contract, "symbol", "") or "").replace(" ", ".").upper()
            if symbol:
                self.stock_contracts[symbol] = contract

    def _ingest_external_stock_batch(self, batch: list[dict[str, Any]]) -> None:
        """Apply stock-only frames published by the stock MD process."""
        now = time.time()
        for row in batch or []:
            symbol = str(row.get("symbol") or "").upper()
            if not symbol:
                continue
            stock = row.get("stock") or {}
            ts = int(float(row.get("ts") or 0.0))
            if ts <= 0:
                continue
            open_px = float(stock.get("open") or 0.0)
            close_px = float(stock.get("close") or open_px or 0.0)
            if close_px <= 0 and open_px <= 0:
                continue
            bar = {
                "ts": float(ts),
                "open": open_px or close_px,
                "high": float(stock.get("high") or max(open_px, close_px)),
                "low": float(stock.get("low") or min(open_px or close_px, close_px)),
                "close": close_px or open_px,
                "volume": float(stock.get("volume") or 0.0),
            }
            self.stock_bar_history.setdefault(symbol, {})[ts] = bar
            self.stock_bars[symbol] = bar
            self.last_stock_tick[symbol] = now
            prev = float(stock.get("previous_close") or 0.0)
            if prev > 0:
                self.stock_previous_close[symbol] = prev

    async def stock_ingest_loop(self) -> None:
        """Options MD process: follow the stock publisher stream."""
        if str(getattr(self, "md_role", "combined")).lower() != "options":
            return
        from maga7.live.redis_fused import unpack_batch

        stream = self.keys.get("stream_stock") or ""
        group = self.keys.get("group_stock") or ""
        consumer = f"options-{self.config.client_id}"
        if not stream or not group:
            return
        try:
            self.redis.xgroup_create(stream, group, id="$", mkstream=True)
        except Exception:
            pass
        while not self._stop.is_set():
            try:
                rows = self.redis.xreadgroup(
                    group,
                    consumer,
                    {stream: ">"},
                    count=50,
                    block=1000,
                )
            except Exception:
                await asyncio.sleep(1.0)
                continue
            if not rows:
                self.publish_feed_health()
                continue
            for _name, messages in rows:
                for message_id, fields in messages:
                    try:
                        batch = unpack_batch(fields.get(b"batch") or fields.get("batch"))
                        self._ingest_external_stock_batch(batch)
                        self.redis.xack(stream, group, message_id)
                    except Exception:
                        logger.exception("stock ingest failed id=%s", message_id)
            self.publish_feed_health()

    def _stock_spot(self, symbol: str) -> float:
        # Prefer live IB ticker when this process owns stock MD.
        contract = self.stock_contracts.get(symbol)
        if contract is not None:
            ticker = self.ib.ticker(contract) if contract is not None else None
            if ticker is not None:
                for value in (
                    getattr(ticker, "last", None),
                    ticker.marketPrice() if hasattr(ticker, "marketPrice") else None,
                    getattr(ticker, "close", None),
                ):
                    price = _finite(value)
                    if price > 0:
                        return price
                bid = _finite(getattr(ticker, "bid", None))
                ask = _finite(getattr(ticker, "ask", None))
                if ask >= bid > 0:
                    return (bid + ask) / 2.0
        bar = self.stock_bars.get(symbol) or {}
        for key in ("close", "open"):
            px = _finite(bar.get(key))
            if px > 0:
                return px
        return 0.0

    async def wait_for_stock_spots(
        self,
        timeout_sec: float = 60.0,
        *,
        min_tick_ts: float = 0.0,
    ) -> dict[str, float]:
        deadline = time.time() + float(timeout_sec)
        while time.time() < deadline:
            spots = {
                symbol: (
                    float(self.stock_bars.get(symbol, {}).get("open", 0.0))
                    if self.last_stock_tick.get(symbol, 0.0) >= min_tick_ts
                    else 0.0
                )
                for symbol in self.trade_symbols
            }
            if all(value > 0 for value in spots.values()):
                return spots
            await asyncio.sleep(0.25)
        spots = {
            symbol: (
                float(self.stock_bars.get(symbol, {}).get("open", 0.0))
                if self.last_stock_tick.get(symbol, 0.0) >= min_tick_ts
                else 0.0
            )
            for symbol in self.trade_symbols
        }
        missing = [symbol for symbol, value in spots.items() if value <= 0]
        raise TimeoutError(f"stock spot timeout: {missing}")

    async def prepare_contract_candidates(self) -> dict[str, list[Any]]:
        """Fetch option-chain metadata before RTH; do not choose strikes yet."""
        self.lock_status = "PREPARING"
        self.prepare_errors = {}
        self.publish_status("PREPARING", note="pre-open contract metadata")
        await self._ensure_stock_contracts(self.trade_symbols)

        async def _one(symbol: str):
            contract = self.stock_contracts.get(symbol)
            if contract is None:
                raise RuntimeError(f"stock contract missing for {symbol}")
            return symbol, await self.lock_service.prepare_symbol(
                contract,
                symbol=symbol,
                trade_date=self.trade_date,
            )

        results = await asyncio.gather(
            *[_one(symbol) for symbol in self.trade_symbols],
            return_exceptions=True,
        )
        prepared: dict[str, list[Any]] = {}
        for result in results:
            if isinstance(result, BaseException):
                logger.error("prelock task failed: %s", result)
                continue
            symbol, contracts = result
            if contracts:
                prepared[symbol] = list(contracts)
            else:
                self.prepare_errors[symbol] = "empty contract universe"
        missing = sorted(set(self.trade_symbols) - set(prepared))
        for symbol in missing:
            self.prepare_errors.setdefault(symbol, "prelock failed")
        self.prepared_option_candidates = prepared
        self.prelock_completed_at = time.time()
        self.lock_status = (
            "PREPARED"
            if set(prepared) == set(self.trade_symbols)
            else "PREPARE_PARTIAL"
        )
        self.publish_status(
            self.lock_status,
            note=(
                f"prepared_symbols={len(prepared)}/{len(self.trade_symbols)} "
                f"prepared_contracts={sum(len(value) for value in prepared.values())}"
            ),
            error="; ".join(
                f"{symbol}:{reason}"
                for symbol, reason in sorted(self.prepare_errors.items())
            ),
        )
        return self.prepared_option_candidates

    async def lock_and_subscribe(
        self,
        timeout_sec: float = 60.0,
        *,
        min_tick_ts: float = 0.0,
    ) -> dict[str, list[LockedContract]]:
        lock_requested_at = time.time()
        spots = await self.wait_for_stock_spots(
            timeout_sec=timeout_sec,
            min_tick_ts=min_tick_ts,
        )
        lock_started_at = time.time()
        prepared_symbols = set(self.prepared_option_candidates)
        self.errors = {}
        self.lock_status = "LOCKING"
        self.publish_status("LOCKING")
        await self._ensure_stock_contracts(self.trade_symbols)

        async def _one(symbol: str):
            contract = self.stock_contracts.get(symbol)
            if contract is None:
                raise RuntimeError(f"stock contract missing for {symbol}")
            return symbol, await self.lock_service.lock_symbol(
                contract,
                symbol=symbol,
                trade_date=self.trade_date,
                spot=spots[symbol],
                prepared_contracts=self.prepared_option_candidates.get(symbol),
            )

        results = await asyncio.gather(
            *[_one(symbol) for symbol in self.trade_symbols],
            return_exceptions=True,
        )
        for result in results:
            if isinstance(result, BaseException):
                logger.error("lock task failed: %s", result)
                continue
            symbol, (locks, contracts) = result
            self.locks[symbol] = locks
            self.option_contracts.update(contracts)
            if not locks:
                self.errors[symbol] = self.prepare_errors.get(symbol, "empty lock")

        # Subscribe the preferred DTE ladder first. Other DTE contracts remain
        # qualified and are subscribed on demand when the signal selects them.
        self._subscribe_locked_contracts()
        quote_diagnostics = await self._diagnose_locked_option_quotes()

        expected = set(self.trade_symbols)
        locked = {symbol for symbol, values in self.locks.items() if values}
        self.lock_status = "LOCKED" if locked == expected else "PARTIAL"
        manifest = atomic_write_lock_manifest(
            self.session_dir / "locks.json",
            session_id=self.session_id,
            trade_date=self.trade_date,
            locks=self.locks,
            status=self.lock_status,
            errors=self.errors,
            metadata={
                "prelock_completed_at": self.prelock_completed_at,
                "lock_requested_at": lock_requested_at,
                "lock_started_at": lock_started_at,
                "lock_completed_at": time.time(),
                "prepared_symbols": sorted(prepared_symbols),
                "fallback_symbols": sorted(expected - prepared_symbols),
                "quote_diagnostics": quote_diagnostics,
            },
        )
        self.redis.set(
            f"maga7:lock_manifest:{self.session_id}",
            pack_obj(manifest),
        )
        self.prepared_option_candidates.clear()
        self.publish_status(self.lock_status)
        return self.locks

    async def _diagnose_locked_option_quotes(
        self, timeout_sec: float = 20.0
    ) -> dict[str, dict[str, Any]]:
        """Wait briefly for NBBO, then localize any remaining no-quote symbols.

        Missing 0/1 DTE is handled by nearest-expiry fallback at lock time.
        A still-missing NBBO is diagnosed (adjusted stub / not subscribed /
        ticker alive without bid-ask / awaiting OPRA), and only structural
        stub locks are excluded.
        """
        pending = {
            symbol
            for symbol, locks in self.locks.items()
            if locks and symbol not in self.errors
        }
        diagnostics: dict[str, dict[str, Any]] = {}
        if not pending:
            return diagnostics
        deadline = time.time() + max(1.0, float(timeout_sec))
        while pending and time.time() < deadline:
            ready = set()
            for symbol in pending:
                locks = self.locks.get(symbol) or []
                if any(
                    (symbol, lock.local_symbol) in self.option_quotes
                    for lock in locks
                ):
                    ready.add(symbol)
            pending -= ready
            if pending:
                await asyncio.sleep(0.25)

        allowed = tuple(getattr(self.lock_service, "allowed_dte", ()) or ())
        for symbol in sorted(set(self.locks) | pending):
            locks = self.locks.get(symbol) or []
            if not locks:
                continue
            if symbol not in pending and any(
                (symbol, lock.local_symbol) in self.option_quotes for lock in locks
            ):
                diagnosis = diagnose_missing_option_quotes(
                    symbol=symbol,
                    locks=locks,
                    option_quotes=self.option_quotes,
                    subscribed_con_ids=self.subscribed_option_ids,
                    allowed_dte=allowed,
                )
                diagnostics[symbol] = diagnosis.to_dict()
                continue
            ticker_snapshots: dict[int, dict[str, Any]] = {}
            for lock in locks:
                contract = self.option_contracts.get(int(lock.con_id))
                if contract is None:
                    continue
                ticker = None
                try:
                    ticker = self.ib.ticker(contract)
                except Exception:
                    ticker = None
                if ticker is None:
                    continue
                ticker_snapshots[int(lock.con_id)] = {
                    "bid": getattr(ticker, "bid", None),
                    "ask": getattr(ticker, "ask", None),
                    "close": getattr(ticker, "close", None),
                    "has_model": bool(getattr(ticker, "modelGreeks", None)),
                }
            diagnosis = diagnose_missing_option_quotes(
                symbol=symbol,
                locks=locks,
                option_quotes=self.option_quotes,
                subscribed_con_ids=self.subscribed_option_ids,
                allowed_dte=allowed,
                ticker_snapshots=ticker_snapshots,
            )
            diagnostics[symbol] = diagnosis.to_dict()
            logger.error(
                "option quote diagnosis symbol=%s code=%s detail=%s actionable=%s",
                symbol,
                diagnosis.code,
                diagnosis.detail,
                diagnosis.actionable,
            )
            if diagnosis.exclude:
                self.errors[symbol] = diagnosis.code
                self.locks[symbol] = []
        return diagnostics

    def ensure_option_subscription(self, con_id: int) -> bool:
        con_id = int(con_id or 0)
        if con_id <= 0 or con_id in self.subscribed_option_ids:
            return con_id in self.subscribed_option_ids
        if len(self.subscribed_option_ids) >= self.config.max_option_subscriptions:
            return False
        contract = self.option_contracts.get(con_id)
        if contract is None:
            return False
        self.ib.reqMktData(contract, "100,101,106", False, False)
        self.subscribed_option_ids.add(con_id)
        return True

    def release_on_demand_subscription(self, con_id: int) -> None:
        con_id = int(con_id or 0)
        if con_id <= 0 or con_id in self.initial_option_ids:
            return
        contract = self.option_contracts.get(con_id)
        if contract is not None and con_id in self.subscribed_option_ids:
            try:
                self.ib.cancelMktData(contract)
            except Exception:
                logger.warning("failed to cancel option subscription conId=%s", con_id)
        self.subscribed_option_ids.discard(con_id)

    def _on_pending_tickers(self, tickers: list[Any]) -> None:
        now = time.time()
        second = float(int(now))
        stock_by_con = {
            int(contract.conId or 0): symbol
            for symbol, contract in self.stock_contracts.items()
        }
        lock_by_local = {
            lock.local_symbol: lock
            for values in self.locks.values()
            for lock in values
        }
        for ticker in tickers:
            contract = getattr(ticker, "contract", None)
            if contract is None:
                continue
            sec_type = str(getattr(contract, "secType", "") or "").upper()
            if sec_type == "STK":
                symbol = stock_by_con.get(int(getattr(contract, "conId", 0) or 0))
                if not symbol:
                    continue
                price = self._stock_spot(symbol)
                if price <= 0:
                    continue
                total_volume = _finite(getattr(ticker, "volume", None))
                previous_close = _finite(getattr(ticker, "close", None))
                if previous_close > 0:
                    self.stock_previous_close[symbol] = previous_close
                previous = self.stock_total_volume.get(symbol)
                delta = max(0.0, total_volume - previous) if previous is not None else 0.0
                if total_volume > 0:
                    self.stock_total_volume[symbol] = total_volume
                history = self.stock_bar_history.setdefault(symbol, {})
                bar = history.get(int(second))
                if bar is None:
                    bar = {
                        "ts": second,
                        "open": price,
                        "high": price,
                        "low": price,
                        "close": price,
                        "volume": delta,
                    }
                    history[int(second)] = bar
                else:
                    bar["high"] = max(bar["high"], price)
                    bar["low"] = min(bar["low"], price)
                    bar["close"] = price
                    bar["volume"] += delta
                self.stock_bars[symbol] = bar
                self.last_stock_tick[symbol] = now
            elif sec_type == "OPT":
                local = str(getattr(contract, "localSymbol", "") or "").strip()
                lock = lock_by_local.get(local)
                if lock is None:
                    continue
                bid = _finite(getattr(ticker, "bid", None))
                ask = _finite(getattr(ticker, "ask", None))
                if ask >= bid > 0:
                    bid_size = _finite(getattr(ticker, "bidSize", None))
                    ask_size = _finite(getattr(ticker, "askSize", None))
                    quote = {
                        "ts": now,
                        "bid": bid,
                        "ask": ask,
                        "mid": (bid + ask) / 2.0,
                        "bid_size": bid_size if bid_size > 0 else 0.0,
                        "ask_size": ask_size if ask_size > 0 else 0.0,
                        "strike": lock.strike,
                        "bucket_id": lock.bucket_id,
                    }
                    key = (lock.symbol, local)
                    self.option_quotes[key] = quote
                    self.option_quote_history.setdefault(key, {})[int(now)] = quote

    def _option_rows(self, symbol: str, now: float) -> list[dict[str, Any]]:
        rows = []
        for (underlying, local), history in self.option_quote_history.items():
            if underlying != symbol:
                continue
            eligible = [
                quote
                for second, quote in history.items()
                if second <= int(now)
            ]
            if not eligible:
                continue
            quote = max(eligible, key=lambda value: float(value["ts"]))
            if now - float(quote["ts"]) > self.config.max_option_staleness_sec:
                continue
            rows.append({"localSymbol": local, **quote})
        return rows

    def _append_tape(self, phase: str, batch: list[dict[str, Any]]) -> None:
        """Append validation/authority seconds under session_dir/tape/{pre|rth|post}/."""
        if not batch:
            return
        phase_dir = tape_phase_dir(self.session_dir, phase)
        phase_dir.mkdir(parents=True, exist_ok=True)
        for row in batch:
            symbol = str(row.get("symbol") or "").upper()
            if not symbol:
                continue
            path = tape_symbol_path(
                self.session_dir,
                phase=phase,
                symbol=symbol,
                trade_date=self.trade_date,
            )
            stock = row.get("stock") or {}
            line = {
                "ts": row.get("ts"),
                "symbol": symbol,
                "phase": phase,
                "frame_id": row.get("frame_id"),
                "open": stock.get("open"),
                "high": stock.get("high"),
                "low": stock.get("low"),
                "close": stock.get("close"),
                "volume": stock.get("volume"),
                "previous_close": stock.get("previous_close"),
                "n_options": len(row.get("option_contracts") or []),
            }
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(line, ensure_ascii=False, default=str) + "\n")
            self.tape_writes += 1
            self._maybe_persist_rth_open_from_tape(symbol, line)
            options = list(row.get("option_contracts") or [])
            if options:
                option_dir = phase_dir / "options"
                option_dir.mkdir(parents=True, exist_ok=True)
                option_path = option_dir / f"{symbol}_{self.trade_date}.jsonl"
                option_line = {
                    "ts": row.get("ts"),
                    "symbol": symbol,
                    "phase": phase,
                    "frame_id": row.get("frame_id"),
                    "quotes": options,
                }
                with option_path.open("a", encoding="utf-8") as handle:
                    handle.write(
                        json.dumps(option_line, ensure_ascii=False, default=str) + "\n"
                    )
                self.option_tape_frames = (
                    int(getattr(self, "option_tape_frames", 0)) + 1
                )
                self.option_tape_quotes = (
                    int(getattr(self, "option_tape_quotes", 0)) + len(options)
                )

    def _maybe_persist_rth_open_from_tape(
        self, symbol: str, line: dict[str, Any]
    ) -> None:
        """Persist 09:30 stock open at the tape layer (before option lock / scanner)."""
        ts = float(line.get("ts") or 0.0)
        open_px = float(line.get("open") or 0.0)
        if ts <= 0 or open_px <= 0:
            return
        try:
            from datetime import datetime
            from zoneinfo import ZoneInfo

            from maga7.live.rth_open_store import upsert_rth_open

            dt = datetime.fromtimestamp(ts, tz=ZoneInfo("America/New_York"))
            if not (dt.hour == 9 and dt.minute == 30):
                return
            live_root = Path(self.session_dir).parent.parent
            upsert_rth_open(
                live_root,
                self.trade_date,
                symbol,
                open_px,
                redis_client=self.redis,
                source="connector_tape_0930",
            )
        except Exception:
            logger.exception("failed to persist tape rth open symbol=%s", symbol)

    async def fetch_rth_opens_historical(
        self,
        symbols: list[str] | None = None,
        *,
        persist: bool = True,
    ) -> dict[str, float]:
        """Backfill official 09:30 opens from IB 1m RTH bars (late-start safe)."""
        from datetime import datetime
        from zoneinfo import ZoneInfo

        from ib_insync import Stock

        from maga7.live.rth_open_store import save_rth_opens

        ny = ZoneInfo("America/New_York")
        want = [
            str(symbol).upper()
            for symbol in (symbols or list(self.trade_symbols) or list(self.symbols) or [])
            if str(symbol).strip()
        ]
        # QQQ is used by open_cont / regime; include when subscribed.
        for symbol in list(self.stock_contracts) + list(self.reference_symbols or []):
            sym = str(symbol).upper()
            if sym and sym not in want:
                want.append(sym)
        found: dict[str, float] = {}
        end = f"{str(self.trade_date).replace('-', '')} 16:00:00 US/Eastern"
        for symbol in want:
            contract = self.stock_contracts.get(symbol)
            if contract is None:
                try:
                    qualified = await self.ib.qualifyContractsAsync(
                        Stock(symbol, "SMART", "USD")
                    )
                    contract = qualified[0] if qualified else None
                    if contract is not None:
                        self.stock_contracts[symbol] = contract
                except Exception:
                    logger.exception("qualify failed for historical open %s", symbol)
                    continue
            if contract is None:
                continue
            try:
                bars = await self.ib.reqHistoricalDataAsync(
                    contract,
                    endDateTime=end,
                    durationStr="1 D",
                    barSizeSetting="1 min",
                    whatToShow="TRADES",
                    useRTH=True,
                    formatDate=1,
                )
            except Exception:
                logger.exception("historical open fetch failed symbol=%s", symbol)
                continue
            for bar in bars or []:
                raw_date = getattr(bar, "date", None)
                try:
                    if isinstance(raw_date, datetime):
                        dt = raw_date
                        if dt.tzinfo is None:
                            dt = dt.replace(tzinfo=ny)
                        else:
                            dt = dt.astimezone(ny)
                    else:
                        dt = datetime.strptime(str(raw_date)[:19], "%Y%m%d %H:%M:%S").replace(
                            tzinfo=ny
                        )
                except Exception:
                    continue
                if not (dt.hour == 9 and dt.minute == 30):
                    continue
                open_px = float(getattr(bar, "open", 0.0) or 0.0)
                if open_px > 0:
                    found[symbol] = open_px
                    break
            await asyncio.sleep(0.2)
        if found and persist:
            live_root = Path(self.session_dir).parent.parent
            save_rth_opens(
                live_root,
                self.trade_date,
                found,
                redis_client=self.redis,
                source="ib_historical_0930",
            )
        return found

    def _append_option_tape(self, phase: str, batch: list[dict[str, Any]]) -> None:
        if not batch:
            return
        phase_dir = tape_phase_dir(self.session_dir, phase)
        option_dir = phase_dir / "options"
        option_dir.mkdir(parents=True, exist_ok=True)
        for row in batch:
            symbol = str(row.get("symbol") or "").upper()
            options = list(row.get("option_contracts") or [])
            if not symbol or not options:
                continue
            option_path = option_dir / f"{symbol}_{self.trade_date}.jsonl"
            option_line = {
                "ts": row.get("ts"),
                "symbol": symbol,
                "phase": phase,
                "frame_id": row.get("frame_id"),
                "quotes": options,
            }
            with option_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(option_line, ensure_ascii=False, default=str) + "\n"
                )
            self.option_tape_frames = int(getattr(self, "option_tape_frames", 0)) + 1
            self.option_tape_quotes = (
                int(getattr(self, "option_tape_quotes", 0)) + len(options)
            )

    def publish_frame(self, ts_val: float | None = None) -> int:
        """Publish one completed second.

        - RTH: full-symbol batch → authority Redis stream + ``tape/rth/``
        - PRE/POST: partial OK → ``stream_pre`` / ``stream_post`` + ``tape/pre|post/``
          (Scanner/OMS never consume validation streams)
        """
        now = float(ts_val or time.time())
        frame_ts = int(now)
        phase = session_phase(now, trade_date=self.trade_date)
        authority = is_rth_authority_phase(phase)
        clock = (
            self.last_published_second if authority else self.last_validation_second
        )
        if frame_ts <= clock:
            return 0
        ib = getattr(self, "ib", None)
        if authority and (
            (ib is not None and not bool(getattr(ib, "isConnected", lambda: True)()))
            or str(getattr(self, "data_mode", "LIVE")).upper() != "LIVE"
        ):
            self.last_published_second = frame_ts
            self._prune_histories(frame_ts)
            return 0
        frame_id = f"{self.session_id}:{frame_ts}"
        batch = []
        for symbol in self.symbols:
            history = self.stock_bar_history.get(symbol) or {}
            bar = history.get(frame_ts)
            if not bar:
                continue
            batch.append(
                {
                    "run_id": self.session_id,
                    "frame_id": frame_id,
                    "frame_complete": frame_ts % 60 == 59,
                    "session_phase": phase,
                    "source": (
                        "maga7_ibkr_stock"
                        if str(getattr(self, "md_role", "combined")).lower() == "stock"
                        else "maga7_ibkr_live"
                    ),
                    "symbol": symbol,
                    "ts": float(frame_ts),
                    "stock": {
                        "open": bar["open"],
                        "high": bar["high"],
                        "low": bar["low"],
                        "close": bar["close"],
                        "volume": float(bar.get("volume", 0.0)),
                        "previous_close": self.stock_previous_close.get(symbol, 0.0),
                    },
                    "option_contracts": (
                        []
                        if str(getattr(self, "md_role", "combined")).lower() == "stock"
                        else self._option_rows(symbol, now)
                    ),
                }
            )
        role = str(getattr(self, "md_role", "combined") or "combined").lower()
        if authority:
            if len(batch) != len(self.symbols):
                if frame_ts not in self._partial_targets:
                    self._partial_targets.add(frame_ts)
                    self.partial_frame_drops += 1
                self.last_published_second = frame_ts
                self._prune_histories(frame_ts)
                return 0
            stream_key = (
                self.keys.get("stream_stock")
                if role == "stock"
                else self.keys["stream"]
            )
        else:
            if not batch:
                return 0
            if role == "stock":
                stream_key = (
                    self.keys.get("stream_pre")
                    if phase == "PRE"
                    else self.keys.get("stream_post")
                ) or self.keys.get("stream_stock")
            else:
                stream_key = (
                    self.keys.get("stream_pre")
                    if phase == "PRE"
                    else self.keys.get("stream_post")
                ) or self.keys["stream"]
        if batch:
            self.redis.xadd(
                stream_key,
                {"batch": pack_batch(batch)},
                maxlen=100_000,
                approximate=True,
            )
            if role != "options":
                self._append_tape(phase, batch)
            else:
                # Stock tape is owned by the stock MD process; only append option path.
                self._append_option_tape(phase, batch)
            if authority:
                self.last_published_second = frame_ts
            else:
                self.last_validation_second = frame_ts
                self.validation_publishes += 1
            self._prune_histories(frame_ts)
        return len(batch)

    def _prune_histories(self, frame_ts: int) -> None:
        for symbol in self.symbols:
            history = self.stock_bar_history.get(symbol) or {}
            for second in [key for key in history if key <= frame_ts - 2]:
                history.pop(second, None)
        for history in self.option_quote_history.values():
            for second in [key for key in history if key <= frame_ts - 10]:
                history.pop(second, None)

    async def publish_loop(self) -> None:
        last_second = -1
        while not self._stop.is_set():
            second = int(time.time())
            if second != last_second:
                last_second = second
                # Per-second freshness for Dash even when frames are partial.
                self.publish_feed_health(float(second))
            self.publish_frame(second - 1)
            await asyncio.sleep(min(0.2, self.config.publish_interval_sec))

    async def heartbeat_loop(self, interval_sec: float = 15.0) -> None:
        while not self._stop.is_set():
            try:
                if not self.ib.isConnected():
                    self.connected = False
                    self.publish_status("DISCONNECTED")
                    await self.connect(retries=0)
                    await self.subscribe_stocks()
                    for con_id in list(self.subscribed_option_ids):
                        contract = self.option_contracts.get(con_id)
                        if contract is not None:
                            self.ib.reqMktData(contract, "100,101,106", False, False)
                    self.reconnect_count += 1
                    callback = self.on_reconnect
                    if callable(callback):
                        callback()
                else:
                    await self.ib.reqCurrentTimeAsync()
                    self.connected = True
                    self.publish_status("HEARTBEAT_OK")
            except Exception as exc:
                self.publish_status("HEARTBEAT_FAIL", error=str(exc))
            await asyncio.sleep(interval_sec)

    def _on_error(self, req_id, error_code, error_string, contract=None) -> None:
        code = int(error_code)
        # Farm / info / connectivity chatter — do not sticky-ERROR the Dash state.
        if code in {
            2103,
            2104,
            2105,
            2106,
            2107,
            2108,
            2119,
            2157,
            2158,
            2100,
        }:
            return
        if code == 1100:
            self.connected = False
            self.publish_status("DISCONNECTED", error=f"{code}:{error_string}")
            return
        if code in {1101, 1102}:
            # Connectivity restored / partial — heartbeat will flip to HEARTBEAT_OK.
            self.publish_status(
                "HEARTBEAT_OK" if self.ib.isConnected() else "CONNECTING",
                note=f"{code}:{error_string}",
            )
            return
        if code == 10197:
            # Never silently turn a live trading session into delayed data.
            self.data_mode = "DELAYED_BLOCKED"
            self.publish_status("DATA_DELAYED_BLOCKED", error=str(error_string))
            return
        # Soft market-data / contract noise during lock & ladder subscribe.
        if code in {200, 201, 203, 300, 354, 366, 10167}:
            logger.warning("IB soft error %s: %s", code, error_string)
            return
        logger.error("IB error %s: %s", code, error_string)
        self.publish_status("ERROR", error=f"{code}:{error_string}")

    def _on_disconnected(self) -> None:
        self.connected = False
        self.publish_status("DISCONNECTED")

    def stop(self) -> None:
        self._stop.set()
        self.publish_status("STOPPING")
        try:
            self.ib.disconnect()
        finally:
            self.publish_status("STOPPED")

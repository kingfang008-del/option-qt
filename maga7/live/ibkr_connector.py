"""Mag7 IBKR market-data connector using the S5 fused payload contract."""
from __future__ import annotations

import asyncio
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
)
from maga7.live.redis_fused import init_maga7_redis, pack_batch, pack_obj, run_keys

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
        otm_rungs: int = 5,
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
        init_maga7_redis(self.redis, run_id=session_id, reset=not resume)
        self.keys = run_keys(session_id)
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
        self.partial_frame_drops = 0
        self._partial_targets: set[int] = set()
        self.last_published_second = 0
        self._stop = asyncio.Event()
        self._callbacks_bound = False

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

    def _status_payload(self, state: str, note: str = "", error: str = "") -> dict[str, Any]:
        quote_symbols = {symbol for symbol, _ in self.option_quotes}
        return {
            "ts": time.time(),
            "session_id": self.session_id,
            "trade_date": self.trade_date,
            "state": state,
            "connected": bool(self.ib.isConnected()),
            "data_mode": self.data_mode,
            "lock_status": self.lock_status,
            "stocks": len(self.stock_contracts),
            "trade_symbols": len(self.trade_symbols),
            "locked_contracts": sum(len(value) for value in self.locks.values()),
            "option_subscriptions": len(self.subscribed_option_ids),
            "option_quotes": len(self.option_quotes),
            "option_quote_symbols": len(quote_symbols),
            "partial_frame_drops": self.partial_frame_drops,
            "stream": self.keys["stream"],
            "note": note,
            "last_error": error,
        }

    def publish_status(self, state: str, note: str = "", error: str = "") -> None:
        payload = self._status_payload(state, note=note, error=error)
        packed = pack_obj(payload)
        pipe = self.redis.pipeline(transaction=True)
        pipe.hset(f"live_ibkr_connector:maga7:{self.session_id}", "status", packed)
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

    def _stock_spot(self, symbol: str) -> float:
        contract = self.stock_contracts.get(symbol)
        ticker = self.ib.ticker(contract) if contract is not None else None
        if ticker is None:
            return 0.0
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
        return (bid + ask) / 2.0 if ask >= bid > 0 else 0.0

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

    async def lock_and_subscribe(
        self,
        timeout_sec: float = 60.0,
        *,
        min_tick_ts: float = 0.0,
    ) -> dict[str, list[LockedContract]]:
        spots = await self.wait_for_stock_spots(
            timeout_sec=timeout_sec,
            min_tick_ts=min_tick_ts,
        )
        self.lock_status = "LOCKING"
        self.publish_status("LOCKING")

        async def _one(symbol: str):
            return symbol, await self.lock_service.lock_symbol(
                self.stock_contracts[symbol],
                symbol=symbol,
                trade_date=self.trade_date,
                spot=spots[symbol],
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
                self.errors[symbol] = "empty lock"

        # Subscribe the preferred DTE ladder first. Other DTE contracts remain
        # qualified and are subscribed on demand when the signal selects them.
        self._subscribe_locked_contracts()

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
        )
        self.redis.set(
            f"maga7:lock_manifest:{self.session_id}",
            pack_obj(manifest),
        )
        self.publish_status(self.lock_status)
        return self.locks

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
                    quote = {
                        "ts": now,
                        "bid": bid,
                        "ask": ask,
                        "mid": (bid + ask) / 2.0,
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

    def publish_frame(self, ts_val: float | None = None) -> int:
        now = float(ts_val or time.time())
        frame_ts = int(now)
        if frame_ts <= self.last_published_second:
            return 0
        frame_id = f"{self.session_id}:{frame_ts}"
        batch = []
        for symbol in self.symbols:
            history = self.stock_bar_history.get(symbol) or {}
            bar = history.get(frame_ts)
            if not bar:
                continue
            pending_bars = [
                value
                for second, value in sorted(history.items())
                if self.last_published_second < second <= frame_ts
            ]
            volume = sum(float(value.get("volume", 0.0)) for value in pending_bars)
            batch.append(
                {
                    "run_id": self.session_id,
                    "frame_id": frame_id,
                    "frame_complete": frame_ts % 60 == 59,
                    "source": "maga7_ibkr_live",
                    "symbol": symbol,
                    "ts": float(frame_ts),
                    "stock": {
                        "open": pending_bars[0]["open"],
                        "high": max(value["high"] for value in pending_bars),
                        "low": min(value["low"] for value in pending_bars),
                        "close": bar["close"],
                        "volume": volume,
                        "previous_close": self.stock_previous_close.get(symbol, 0.0),
                    },
                    "option_contracts": self._option_rows(symbol, now),
                }
            )
        if len(batch) != len(self.symbols):
            if frame_ts not in self._partial_targets:
                self._partial_targets.add(frame_ts)
                self.partial_frame_drops += 1
            return 0
        if batch:
            self.redis.xadd(
                self.keys["stream"],
                {"batch": pack_batch(batch)},
                maxlen=100_000,
                approximate=True,
            )
            self.last_published_second = frame_ts
            for symbol in self.symbols:
                history = self.stock_bar_history.get(symbol) or {}
                for second in [key for key in history if key <= frame_ts - 2]:
                    history.pop(second, None)
            for history in self.option_quote_history.values():
                for second in [key for key in history if key <= frame_ts - 10]:
                    history.pop(second, None)
        return len(batch)

    async def publish_loop(self) -> None:
        last_second = -1
        while not self._stop.is_set():
            second = int(time.time())
            if second != last_second:
                last_second = second
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
                else:
                    await self.ib.reqCurrentTimeAsync()
                    self.connected = True
                    self.publish_status("HEARTBEAT_OK")
            except Exception as exc:
                self.publish_status("HEARTBEAT_FAIL", error=str(exc))
            await asyncio.sleep(interval_sec)

    def _on_error(self, req_id, error_code, error_string, contract=None) -> None:
        if int(error_code) in {2104, 2106, 2107, 2108, 2158}:
            return
        if int(error_code) == 10197:
            # Never silently turn a live trading session into delayed data.
            self.data_mode = "DELAYED_BLOCKED"
            self.publish_status("DATA_DELAYED_BLOCKED", error=str(error_string))
            return
        self.publish_status("ERROR", error=f"{error_code}:{error_string}")

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

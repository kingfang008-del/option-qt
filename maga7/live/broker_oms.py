"""Event-driven Mag7 Shadow/Paper/Live OMS with hard live-trading gates."""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import os
import tempfile
import threading
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import pandas as pd

from maga7.common.fills import FillSpec
from maga7.common.position_size import apply_size_scale, regime_scale_from_meta, resolve_size_frac
from maga7.common.replay import to_ny
from maga7.live.requote import (
    entry_requote_limit,
    exit_requote_limit,
    is_urgent_exit_reason,
    requote_config_from_trade,
)
from maga7.live.risk_guards import (
    entry_quote_ok,
    fill_adverse,
    is_fresh,
    observe_exit_mid,
    quote_mid,
    risk_config_from_trade,
)
from maga7.live.scanner import Mag7Scanner, ScannerSignal
from qqq_btc.common.fill_model import OptionSpreadFillModel
from qqq_btc.live.oms_adapter import audit_fill, limit_price_from_quote

logger = logging.getLogger("maga7.live.broker_oms")

TERMINAL_STATUSES = {
    "FILLED",
    "CANCELLED",
    "APICANCELLED",
    "INACTIVE",
    "REJECTED",
    "ERROR",
}

FORCE_EXIT_REASONS = {
    "DAY_CIRCUIT",
    "EOD",
    "EXIT_CHASE_CAP",
    "GAP_FLATTEN",
    "ADVERSE_FILL_FLATTEN",
}


@dataclass
class LivePosition:
    symbol: str
    contract: str
    con_id: int
    direction: str
    qty: int
    entry_price: float
    entry_ts: float
    signal_ts: float
    rank: int
    qty_frac: float
    entry_bid: float
    entry_ask: float
    status: str = "OPEN"
    exit_order_id: int = 0
    last_bid: float = 0.0
    last_ask: float = 0.0
    hold_extended: bool = False
    last_good_mid: float = 0.0
    gap_hold_count: int = 0
    exit_chase_count: int = 0


@dataclass
class PendingIntent:
    intent_id: str
    action: str
    symbol: str
    contract: str
    con_id: int
    qty: int
    limit_price: float
    reason: str
    created_at: float
    signal: ScannerSignal | None = None
    broker_order_id: int = 0
    perm_id: int = 0
    status: str = "CREATED"
    filled: float = 0.0
    avg_fill_price: float = 0.0
    requote_attempt: int = 0
    ref_price: float = 0.0
    replaced_by: str = ""
    parent_intent_id: str = ""


def profile_digest(profile: dict[str, Any]) -> str:
    raw = json.dumps(profile, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _signal_dict(signal: ScannerSignal | None) -> dict[str, Any] | None:
    if signal is None:
        return None
    return {
        "date": signal.date,
        "symbol": signal.symbol,
        "direction": signal.direction,
        "sig_ts": to_ny(signal.sig_ts).isoformat(),
        "spot": signal.spot,
        "rank": signal.rank,
        "bucket_id": signal.bucket_id,
        "contract": signal.contract,
        "moneyness": signal.moneyness,
        "meta": signal.meta,
    }


def _intent_dict(intent: PendingIntent) -> dict[str, Any]:
    payload = asdict(intent)
    payload["signal"] = _signal_dict(intent.signal)
    return payload


def _intent_from_dict(payload: dict[str, Any]) -> PendingIntent:
    data = dict(payload)
    signal = data.get("signal")
    if isinstance(signal, dict):
        signal = ScannerSignal(**{**signal, "sig_ts": to_ny(signal["sig_ts"])})
    else:
        signal = None
    data["signal"] = signal
    allowed = {item.name for item in fields(PendingIntent)}
    return PendingIntent(**{k: v for k, v in data.items() if k in allowed})


class Mag7BrokerOms:
    """One OMS for shadow, IBKR paper, and explicitly armed live trading."""

    def __init__(
        self,
        *,
        profile: dict[str, Any],
        scanner: Mag7Scanner,
        connector: Any,
        session_id: str,
        trade_date: str,
        session_dir: Path,
        mode: str = "shadow",
        max_qty: int = 1,
        equity: float = 100_000.0,
    ):
        self.profile = profile
        self.scanner = scanner
        self.connector = connector
        self.ib = connector.ib
        self.redis = connector.redis
        self.session_id = session_id
        self.trade_date = trade_date
        self.session_dir = Path(session_dir)
        self.mode = str(mode).strip().lower()
        if self.mode not in {"shadow", "paper", "live"}:
            raise ValueError(f"invalid Mag7 OMS mode: {mode}")
        self.max_qty = max(1, int(max_qty))
        self.equity = float(equity)
        self.day_start_equity = float(equity)
        self.realized_pnl = 0.0
        self.day_halted = False
        self.available_funds = float(equity)
        self.account_ready = self.mode == "shadow"
        self._restored_state = False
        fill_cfg = profile.get("fill") or {}
        self.fill = FillSpec(
            entry_frac=float(fill_cfg.get("entry_frac", 0.8)),
            exit_frac=float(fill_cfg.get("exit_frac", 0.8)),
        )
        self.fill_model = OptionSpreadFillModel(
            entry_frac=self.fill.entry_frac,
            exit_frac=self.fill.exit_frac,
        )
        self.trade_cfg = profile.get("trade") or {}
        self.risk_cfg = risk_config_from_trade(
            self.trade_cfg,
            getattr(connector, "config", None),
        )
        self.requote_cfg = requote_config_from_trade(self.trade_cfg)
        self.profile_hash = str(
            profile.get("_live_fingerprint") or profile_digest(profile)
        )
        self.positions: dict[str, LivePosition] = {}
        self.intents: dict[str, PendingIntent] = {}
        self.trades: dict[str, Any] = {}
        self.pending_signals: dict[str, tuple[ScannerSignal, float]] = {}
        self.open_until: dict[str, pd.Timestamp] = {}
        self.seen_fills: set[str] = set()
        self.seen_commissions: set[str] = set()
        self._bound_order_refs: set[str] = set()
        self._last_seen_option_mid: dict[tuple[str, str], float] = {}
        self._flattening_circuit = False
        self.reconcile_ok = self.mode == "shadow"
        self._lock = threading.RLock()
        self._event_seq = 0
        self.scanner.is_symbol_active = self.has_position
        if self.mode != "shadow" and hasattr(self.ib, "commissionReportEvent"):
            self.ib.commissionReportEvent += self._on_commission
        self.restore_state()
        for position in self.positions.values():
            self.connector.ensure_option_subscription(position.con_id)
        self.publish_state()

    @property
    def state_path(self) -> Path:
        return self.session_dir / "oms_state.json"

    @property
    def event_path(self) -> Path:
        return self.session_dir / "order_events.jsonl"

    @property
    def positions_key(self) -> str:
        return f"oms:live_positions:maga7:{self.session_id}"

    @property
    def intents_key(self) -> str:
        return f"oms:pending_orders:maga7:{self.session_id}"

    def has_position(self, symbol: str) -> bool:
        pos = self.positions.get(str(symbol).upper())
        return bool(pos and pos.status in {"OPEN", "EXIT_PENDING"})

    def _event(self, kind: str, payload: dict[str, Any]) -> None:
        with self._lock:
            self._event_seq += 1
            row = {
                "seq": self._event_seq,
                "ts": time.time(),
                "session_id": self.session_id,
                "mode": self.mode,
                "kind": kind,
                **payload,
            }
            self.event_path.parent.mkdir(parents=True, exist_ok=True)
            with self.event_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            self.redis.xadd(
                f"maga7:order_events:{self.session_id}",
                {"data": json.dumps(row, ensure_ascii=True, default=str)},
                maxlen=20_000,
                approximate=True,
            )

    def snapshot(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "session_id": self.session_id,
            "trade_date": self.trade_date,
            "mode": self.mode,
            "profile_hash": self.profile_hash,
            "updated_at": time.time(),
            "reconcile_ok": self.reconcile_ok,
            "equity": self.equity,
            "day_start_equity": self.day_start_equity,
            "realized_pnl": self.realized_pnl,
            "day_halted": self.day_halted,
            "available_funds": self.available_funds,
            "account_ready": self.account_ready,
            "seen_fills": sorted(self.seen_fills),
            "seen_commissions": sorted(self.seen_commissions),
            "positions": {key: asdict(value) for key, value in self.positions.items()},
            "intents": {key: _intent_dict(value) for key, value in self.intents.items()},
            "pending_signals": {
                symbol: {
                    "signal": _signal_dict(signal),
                    "created_at": created,
                }
                for symbol, (signal, created) in self.pending_signals.items()
            },
            "open_until": {key: value.isoformat() for key, value in self.open_until.items()},
        }

    def publish_state(self) -> None:
        payload = self.snapshot()
        _atomic_json(self.state_path, payload)
        pipe = self.redis.pipeline(transaction=True)
        pipe.delete(self.positions_key)
        if self.positions:
            pipe.hset(
                self.positions_key,
                mapping={
                    key: json.dumps(asdict(value), ensure_ascii=True)
                    for key, value in self.positions.items()
                },
            )
        pipe.delete(self.intents_key)
        active_intents = {
            key: value
            for key, value in self.intents.items()
            if value.status.upper() not in TERMINAL_STATUSES
        }
        if active_intents:
            pipe.hset(
                self.intents_key,
                mapping={
                    key: json.dumps(_intent_dict(value), ensure_ascii=True, default=str)
                    for key, value in active_intents.items()
                },
            )
        pipe.execute()

    def restore_state(self) -> None:
        if not self.state_path.is_file():
            return
        self._restored_state = True
        try:
            raw = json.loads(self.state_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise RuntimeError(f"cannot restore OMS state: {exc}") from exc
        if raw.get("profile_hash") != self.profile_hash:
            raise RuntimeError("OMS state profile hash mismatch")
        if raw.get("trade_date") != self.trade_date:
            raise RuntimeError("OMS state trade date mismatch")
        self.equity = float(raw.get("equity", self.equity))
        self.day_start_equity = float(
            raw.get("day_start_equity", self.day_start_equity)
        )
        self.realized_pnl = float(raw.get("realized_pnl", 0.0))
        self.day_halted = bool(raw.get("day_halted", False))
        self.available_funds = float(
            raw.get("available_funds", self.available_funds)
        )
        self.account_ready = bool(raw.get("account_ready", self.account_ready))
        self.seen_fills = set(raw.get("seen_fills") or [])
        self.seen_commissions = set(raw.get("seen_commissions") or [])
        self.positions = {
            key: LivePosition(**value)
            for key, value in (raw.get("positions") or {}).items()
        }
        self.intents = {
            key: _intent_from_dict(value)
            for key, value in (raw.get("intents") or {}).items()
        }
        self.pending_signals = {}
        for symbol, value in (raw.get("pending_signals") or {}).items():
            signal_data = value.get("signal") if isinstance(value, dict) else None
            if not isinstance(signal_data, dict):
                continue
            signal = ScannerSignal(
                **{
                    **signal_data,
                    "sig_ts": to_ny(signal_data["sig_ts"]),
                }
            )
            self.pending_signals[str(symbol)] = (
                signal,
                float(value.get("created_at") or time.time()),
            )
        self.open_until = {
            key: to_ny(value) for key, value in (raw.get("open_until") or {}).items()
        }
        self._event("STATE_RESTORED", {"positions": len(self.positions), "intents": len(self.intents)})

    def _runtime_armed(self) -> bool:
        raw = self.redis.hget("meta:runtime_trading_controls:maga7", "trading_enabled")
        if raw is None:
            return False
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="ignore")
        return str(raw).strip().lower() in {"1", "true", "yes", "on"}

    def live_gate(self) -> tuple[bool, str]:
        if self.mode == "shadow":
            return True, "shadow"
        if not self.ib.isConnected():
            return False, "ibkr_disconnected"
        if not self.account_ready:
            return False, "broker_account_unavailable"
        if not self.connector.config.account:
            return False, "broker_account_not_explicit"
        if self.connector.lock_status != "LOCKED":
            return False, f"lock_{self.connector.lock_status.lower()}"
        if self.mode == "paper":
            if int(self.connector.config.port) != 4002:
                return False, "paper_requires_port_4002"
            if self.connector.data_mode != "LIVE":
                return False, f"market_data_{self.connector.data_mode.lower()}"
            if not self.reconcile_ok:
                return False, "broker_reconcile_failed"
            return True, "paper"
        if int(self.connector.config.port) != 4001:
            return False, "live_requires_port_4001"
        if self.connector.data_mode != "LIVE":
            return False, f"market_data_{self.connector.data_mode.lower()}"
        if os.environ.get("MAG7_LIVE_TRADING", "0") != "1":
            return False, "MAG7_LIVE_TRADING_not_enabled"
        expected = f"{self.trade_date}:{self.profile_hash[:12]}"
        if os.environ.get("MAG7_LIVE_CONFIRM", "") != expected:
            return False, "MAG7_LIVE_CONFIRM_mismatch"
        if not self._runtime_armed():
            return False, "runtime_disarmed"
        if not self.reconcile_ok:
            return False, "broker_reconcile_failed"
        return True, "armed"

    def _quote(self, symbol: str, contract: str) -> dict[str, float] | None:
        quote = self.connector.option_quotes.get((symbol, contract))
        if not quote:
            return None
        age = time.time() - float(quote.get("ts", 0.0))
        if age > self.risk_cfg.max_option_staleness_sec:
            return None
        return quote

    def _stock_fresh(self, symbol: str, *, now: float | None = None) -> tuple[bool, str]:
        now_ts = time.time() if now is None else float(now)
        ticks = getattr(self.connector, "last_stock_tick", None) or {}
        last = ticks.get(str(symbol).upper())
        if last is None:
            return False, "stock_tick_missing"
        if not is_fresh(
            float(last),
            now=now_ts,
            max_age_sec=self.risk_cfg.max_stock_staleness_sec,
        ):
            return False, "stock_stale"
        return True, "ok"

    def _remember_option_mid(self, symbol: str, contract: str, mid: float) -> None:
        if math.isfinite(mid) and mid > 0:
            self._last_seen_option_mid[(symbol.upper(), contract)] = float(mid)

    def _trip_day_circuit(self, *, day_ret: float) -> None:
        was = self.day_halted
        self.day_halted = True
        if was or self._flattening_circuit:
            return
        self._event("DAY_CIRCUIT", {"day_return": day_ret})
        if not self.risk_cfg.day_circuit_force_flatten:
            return
        self._flattening_circuit = True
        try:
            self.force_flatten("DAY_CIRCUIT")
        finally:
            self._flattening_circuit = False

    def _lock_for_contract(self, symbol: str, contract: str):
        for lock in self.connector.locks.get(symbol, []):
            if lock.local_symbol == contract:
                return lock
        return None

    def _defer_signal(self, signal: ScannerSignal) -> None:
        symbol = signal.symbol.upper()
        if symbol not in self.pending_signals:
            self.pending_signals[symbol] = (signal, time.time())
            self.publish_state()

    def _position_quote_fallback(self, position: LivePosition) -> dict[str, float] | None:
        quote = self._quote(position.symbol, position.contract)
        if quote is not None:
            return quote
        if position.last_bid > 0 and position.last_ask >= position.last_bid:
            return {
                "bid": float(position.last_bid),
                "ask": float(position.last_ask),
                "ts": time.time(),
            }
        if position.last_good_mid > 0:
            mid = float(position.last_good_mid)
            return {"bid": mid, "ask": mid, "ts": time.time()}
        return None

    def _intent_id(
        self,
        action: str,
        symbol: str,
        contract: str,
        seed: str,
    ) -> str:
        raw = f"{self.session_id}|{action}|{symbol}|{contract}|{seed}"
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]
        return f"M7-{self.session_id[:8]}-{action[:1].upper()}-{digest}"

    def _size(
        self,
        symbol: str,
        entry_ts,
        entry_price: float,
        *,
        regime_scale: float = 1.0,
    ) -> tuple[int, float]:
        top_k = max(int((self.profile.get("signal") or {}).get("top_k", 2)), 1)
        sizing_clock = to_ny(entry_ts)
        occupancy = dict(self.open_until)
        for active_symbol, position in self.positions.items():
            if position.status in {"OPEN", "EXIT_PENDING"}:
                occupancy[active_symbol] = sizing_clock + pd.Timedelta(days=1)
        frac, _, _, allow, _ = resolve_size_frac(
            self.trade_cfg,
            top_k=top_k,
            open_until=occupancy,
            symbol=symbol,
            entry_ts=sizing_clock,
        )
        frac = apply_size_scale(frac, regime_scale)
        if (not allow) or frac <= 0.0:
            return 0, float(frac)
        capital = min(self.equity, self.available_funds)
        qty = int((capital * float(frac)) // max(entry_price * 100.0, 0.01))
        return max(1, min(self.max_qty, qty)), float(frac)

    async def initialize_account(self) -> None:
        if self.mode == "shadow":
            self.account_ready = True
            return
        try:
            await self.ib.reqAccountSummaryAsync()
        except Exception:
            pass
        values = []
        if hasattr(self.ib, "accountSummary"):
            values.extend(self.ib.accountSummary() or [])
        if hasattr(self.ib, "accountValues"):
            values.extend(self.ib.accountValues() or [])
        net_liq = 0.0
        available = 0.0
        for value in values:
            if (
                self.connector.config.account
                and str(getattr(value, "account", "") or "")
                != self.connector.config.account
            ):
                continue
            if str(getattr(value, "currency", "") or "") not in {"USD", "BASE"}:
                continue
            tag = str(getattr(value, "tag", "") or "")
            number = float(getattr(value, "value", 0.0) or 0.0)
            if tag.startswith("NetLiquidation"):
                net_liq = max(net_liq, number)
            elif tag == "AvailableFunds":
                available = max(available, number)
        if net_liq <= 0 or available <= 0:
            self.account_ready = False
            raise RuntimeError("IBKR NetLiquidation/AvailableFunds unavailable")
        if not self._restored_state:
            self.equity = net_liq
            self.day_start_equity = net_liq
        self.available_funds = available
        self.account_ready = True
        self._event(
            "ACCOUNT_READY",
            {"net_liquidation": net_liq, "available_funds": available},
        )
        self.publish_state()

    def process_signal(self, signal: ScannerSignal) -> bool:
        symbol = signal.symbol.upper()
        contract = str(signal.contract or "").replace("O:", "")
        if not contract or self.has_position(symbol):
            return False
        if self.day_halted:
            reason = "day_halted"
            meta = getattr(self.scanner, "event_blackout_meta", None) or {}
            if meta.get("active_today"):
                reason = "event_blackout"
            self._event("ENTRY_REJECT", {"symbol": symbol, "reason": reason})
            return False
        if self.risk_cfg.halt_entries_on_gap and any(
            pos.gap_hold_count > 0 for pos in self.positions.values()
        ):
            self._event("ENTRY_WAIT", {"symbol": symbol, "reason": "option_gap_hold"})
            self._defer_signal(signal)
            return False
        stock_ok, stock_reason = self._stock_fresh(symbol)
        if not stock_ok:
            self._event(
                "ENTRY_WAIT",
                {"symbol": symbol, "reason": stock_reason},
            )
            self._defer_signal(signal)
            return False
        lock = self._lock_for_contract(symbol, contract)
        if lock is None:
            self._event("ENTRY_REJECT", {"symbol": symbol, "reason": "contract_not_locked"})
            return False
        if not self.connector.ensure_option_subscription(lock.con_id):
            self._event("ENTRY_WAIT", {"symbol": symbol, "reason": "subscription_limit"})
            self._defer_signal(signal)
            return False
        quote = self._quote(symbol, contract)
        if quote is None:
            self._event("ENTRY_WAIT", {"symbol": symbol, "reason": "option_stale_or_missing"})
            self._defer_signal(signal)
            return False
        prev_mid = self._last_seen_option_mid.get((symbol, contract))
        quote_ok, quote_reason, mid = entry_quote_ok(
            bid=float(quote["bid"]),
            ask=float(quote["ask"]),
            prev_mid=prev_mid,
            cfg=self.risk_cfg,
        )
        if mid is not None and quote_reason != "entry_mid_jump":
            self._remember_option_mid(symbol, contract, mid)
        if not quote_ok:
            kind = "ENTRY_REJECT" if quote_reason == "entry_mid_jump" else "ENTRY_WAIT"
            self._event(kind, {"symbol": symbol, "reason": quote_reason, "mid": mid})
            if quote_reason != "entry_mid_jump":
                self._defer_signal(signal)
            return False
        allowed, reason = self.live_gate()
        if not allowed:
            self._event("ENTRY_REJECT", {"symbol": symbol, "reason": reason})
            return False
        limit_price = limit_price_from_quote(
            quote["bid"], quote["ask"], "BUY", self.fill_model
        )
        r_scale = regime_scale_from_meta(getattr(signal, "meta", None))
        qty, qty_frac = self._size(
            symbol, signal.sig_ts, limit_price, regime_scale=r_scale
        )
        if qty <= 0:
            self._event(
                "ENTRY_REJECT",
                {
                    "symbol": symbol,
                    "reason": "size_gate",
                    "regime_size_scale": r_scale,
                },
            )
            return False
        intent_id = self._intent_id(
            "BUY",
            symbol,
            contract,
            str(int(to_ny(signal.sig_ts).timestamp() * 1000)),
        )
        if intent_id in self.intents:
            return False
        mid = quote_mid(float(quote["bid"]), float(quote["ask"])) or float(limit_price)
        intent = PendingIntent(
            intent_id=intent_id,
            action="BUY",
            symbol=symbol,
            contract=contract,
            con_id=lock.con_id,
            qty=qty,
            limit_price=limit_price,
            reason="ENTRY",
            created_at=time.time(),
            signal=signal,
            ref_price=float(mid),
        )
        self.intents[intent.intent_id] = intent
        self._event("ENTRY_INTENT", _intent_dict(intent))
        if self.mode == "shadow":
            self._apply_open_fill(intent, qty, limit_price, qty_frac, quote)
        else:
            self._place_broker_order(intent)
        self.publish_state()
        return True

    def retry_pending_signals(self, timeout_sec: float = 15.0) -> None:
        changed = False
        for symbol, (signal, created) in list(self.pending_signals.items()):
            if time.time() - created > timeout_sec:
                self.pending_signals.pop(symbol, None)
                changed = True
                self._event("ENTRY_REJECT", {"symbol": symbol, "reason": "quote_timeout"})
                continue
            if self.process_signal(signal):
                self.pending_signals.pop(symbol, None)
                changed = True
        if changed:
            self.publish_state()

    def _place_broker_order(self, intent: PendingIntent) -> None:
        from ib_insync import LimitOrder

        contract = self.connector.option_contracts.get(intent.con_id)
        if contract is None:
            intent.status = "ERROR"
            self._event("ORDER_ERROR", {"intent_id": intent.intent_id, "reason": "contract_missing"})
            return
        order = LimitOrder(intent.action, intent.qty, round(intent.limit_price, 2), tif="DAY")
        order.orderRef = intent.intent_id[:32]
        if self.connector.config.account:
            order.account = self.connector.config.account
        trade = self.ib.placeOrder(contract, order)
        self.trades[intent.intent_id] = trade
        intent.broker_order_id = int(getattr(order, "orderId", 0) or 0)
        intent.perm_id = int(getattr(order, "permId", 0) or 0)
        intent.status = str(getattr(trade.orderStatus, "status", "SUBMITTED") or "SUBMITTED").upper()
        self._bind_trade(intent.intent_id, trade)
        self._event("ORDER_SUBMITTED", _intent_dict(intent))

    def _bind_trade(self, intent_id: str, trade: Any) -> None:
        if intent_id in self._bound_order_refs:
            return
        trade.statusEvent += lambda tr: self._on_trade_status(intent_id, tr)
        trade.fillEvent += lambda tr, fill: self._on_fill(intent_id, tr, fill)
        trade.cancelledEvent += lambda tr: self._on_trade_status(intent_id, tr)
        self._bound_order_refs.add(intent_id)
        self.trades[intent_id] = trade

    async def recover_broker_activity(self) -> None:
        """Rebind open orders and fold completed fills missed during downtime."""
        if self.mode == "shadow":
            return
        recovered_ids: set[str] = set()
        for trade in self.ib.openTrades() or []:
            intent_id = str(getattr(trade.order, "orderRef", "") or "")
            if intent_id in self.intents:
                recovered_ids.add(intent_id)
                self._bind_trade(intent_id, trade)
                self._on_trade_status(intent_id, trade)
        try:
            completed = await self.ib.reqCompletedOrdersAsync(apiOnly=False)
        except Exception as exc:
            self._event("RECOVERY_WARNING", {"reason": "completed_orders", "error": str(exc)})
            completed = []
        for trade in completed or []:
            intent_id = str(getattr(trade.order, "orderRef", "") or "")
            intent = self.intents.get(intent_id)
            if intent is None:
                continue
            recovered_ids.add(intent_id)
            if intent.status.upper() in TERMINAL_STATUSES:
                continue
            status = getattr(trade, "orderStatus", None)
            filled = int(round(float(getattr(status, "filled", 0.0) or 0.0)))
            avg_price = float(getattr(status, "avgFillPrice", 0.0) or 0.0)
            if filled <= 0 or avg_price <= 0:
                self._on_trade_status(intent_id, trade)
                continue
            if intent.action == "BUY" and intent.symbol not in self.positions:
                quote = self._quote(intent.symbol, intent.contract) or {
                    "bid": avg_price,
                    "ask": avg_price,
                }
                signal = intent.signal
                if signal is not None:
                    _, qty_frac = self._size(
                        intent.symbol,
                        signal.sig_ts,
                        avg_price,
                        regime_scale=regime_scale_from_meta(getattr(signal, "meta", None)),
                    )
                    self._apply_open_fill(intent, filled, avg_price, qty_frac, quote)
            elif intent.action == "SELL" and intent.symbol in self.positions:
                self._apply_close_fill(intent, filled, avg_price)
            self._on_trade_status(intent_id, trade)
        for intent_id, intent in self.intents.items():
            if intent.status.upper() in TERMINAL_STATUSES or intent_id in recovered_ids:
                continue
            intent.status = "ERROR"
            position = self.positions.get(intent.symbol)
            if intent.action == "SELL" and position is not None:
                position.status = "OPEN"
            elif intent.action == "BUY" and position is None:
                self.connector.release_on_demand_subscription(intent.con_id)
            self._event(
                "RECOVERY_ORDER_MISSING",
                {"intent_id": intent_id, "action": intent.action},
            )
        self._event(
            "BROKER_ACTIVITY_RECOVERED",
            {"bound_open_orders": len(self._bound_order_refs)},
        )
        self.publish_state()

    def _on_trade_status(self, intent_id: str, trade: Any) -> None:
        intent = self.intents.get(intent_id)
        if intent is None:
            return
        status = getattr(trade, "orderStatus", None)
        intent.status = str(getattr(status, "status", "") or "").upper()
        intent.filled = float(getattr(status, "filled", 0.0) or 0.0)
        intent.avg_fill_price = float(getattr(status, "avgFillPrice", 0.0) or 0.0)
        intent.perm_id = int(getattr(status, "permId", intent.perm_id) or intent.perm_id)
        self._event("ORDER_STATUS", _intent_dict(intent))
        if intent.status in {"CANCELLED", "APICANCELLED", "INACTIVE", "REJECTED", "ERROR"}:
            remaining = max(0, int(intent.qty) - int(round(float(intent.filled))))
            requoted = False
            if (
                remaining > 0
                and intent.status in {"CANCELLED", "APICANCELLED", "REJECTED"}
                and not intent.replaced_by
            ):
                requoted = self._try_requote(intent, remaining_qty=remaining)
            position = self.positions.get(intent.symbol)
            if intent.action == "SELL" and position is not None and not requoted:
                position.status = "OPEN"
                if intent.reason not in FORCE_EXIT_REASONS:
                    position.exit_chase_count += 1
                    self._event(
                        "EXIT_CHASE",
                        {
                            "symbol": position.symbol,
                            "chase": position.exit_chase_count,
                            "max_exit_chase": self.risk_cfg.max_exit_chase,
                            "failed_reason": intent.reason,
                            "status": intent.status,
                        },
                    )
                    if position.exit_chase_count >= self.risk_cfg.max_exit_chase:
                        quote = self._position_quote_fallback(position)
                        if quote is not None:
                            self._submit_exit(
                                position,
                                "EXIT_CHASE_CAP",
                                float(quote["bid"]),
                                quote,
                            )
            elif (
                intent.action == "BUY"
                and not requoted
                and intent.symbol not in self.positions
            ):
                self.connector.release_on_demand_subscription(intent.con_id)
            self.trades.pop(intent_id, None)
        self.publish_state()

    def _live_option_quote(
        self, symbol: str, contract: str, con_id: int
    ) -> dict[str, float] | None:
        quote = self._quote(symbol, contract)
        if quote is not None:
            return quote
        try:
            ib_contract = self.connector.option_contracts.get(con_id)
            if ib_contract is None or not self.ib.isConnected():
                return None
            ticker = self.ib.ticker(ib_contract)
            if ticker is None:
                return None
            bid = float(getattr(ticker, "bid", 0.0) or 0.0)
            ask = float(getattr(ticker, "ask", 0.0) or 0.0)
            if ask >= bid > 0:
                return {"bid": bid, "ask": ask, "ts": time.time()}
        except Exception:
            return None
        return None

    def _try_requote(self, intent: PendingIntent, *, remaining_qty: int) -> bool:
        """Cancel-replace with a more aggressive LMT. Returns True if replaced."""
        if self.mode == "shadow" or not self.requote_cfg.enabled:
            return False
        if intent.replaced_by or remaining_qty <= 0:
            return False
        next_attempt = int(intent.requote_attempt) + 1
        quote = self._live_option_quote(intent.symbol, intent.contract, intent.con_id)
        if quote is None and intent.action == "SELL":
            position = self.positions.get(intent.symbol)
            if position is not None:
                quote = self._position_quote_fallback(position)
        if quote is None:
            self._event(
                "REQUOTE_SKIP",
                {
                    "intent_id": intent.intent_id,
                    "reason": "quote_missing",
                    "attempt": next_attempt,
                },
            )
            return False

        if intent.action == "BUY":
            if next_attempt > self.requote_cfg.max_entry_requotes:
                return False
            new_px = entry_requote_limit(
                bid=float(quote["bid"]),
                ask=float(quote["ask"]),
                attempt_no=next_attempt,
                base_frac=float(self.fill.entry_frac),
                max_frac=float(self.risk_cfg.max_fill_spread_frac),
                ref_price=float(intent.ref_price or quote_mid(quote["bid"], quote["ask"]) or 0.0),
                prev_limit=float(intent.limit_price),
                cfg=self.requote_cfg,
            )
            if new_px is None:
                self._event(
                    "REQUOTE_CAP",
                    {
                        "intent_id": intent.intent_id,
                        "action": "BUY",
                        "attempt": next_attempt,
                    },
                )
                return False
        else:
            if next_attempt > self.requote_cfg.max_exit_requotes:
                return False
            urgent = is_urgent_exit_reason(intent.reason)
            new_px = exit_requote_limit(
                bid=float(quote["bid"]),
                ask=float(quote.get("ask") or 0.0),
                attempt_no=next_attempt,
                prev_limit=float(intent.limit_price),
                urgent=urgent,
                cfg=self.requote_cfg,
            )
            if new_px is None:
                return False
            position = self.positions.get(intent.symbol)
            if position is None:
                return False
            position.status = "OPEN"

        child_id = self._intent_id(
            intent.action,
            intent.symbol,
            intent.contract,
            f"{intent.intent_id}:rq{next_attempt}:{time.time():.3f}",
        )
        if child_id in self.intents:
            return False
        child = PendingIntent(
            intent_id=child_id,
            action=intent.action,
            symbol=intent.symbol,
            contract=intent.contract,
            con_id=intent.con_id,
            qty=int(remaining_qty),
            limit_price=float(new_px),
            reason=intent.reason,
            created_at=time.time(),
            signal=intent.signal,
            requote_attempt=next_attempt,
            ref_price=float(intent.ref_price or 0.0),
            parent_intent_id=intent.intent_id,
        )
        intent.replaced_by = child_id
        self.intents[child_id] = child
        self._event(
            "REQUOTE",
            {
                **_intent_dict(child),
                "from_intent_id": intent.intent_id,
                "from_limit": intent.limit_price,
                "bid": quote["bid"],
                "ask": quote.get("ask"),
            },
        )
        self._place_broker_order(child)
        if intent.action == "SELL":
            position = self.positions.get(intent.symbol)
            if position is not None:
                position.status = "EXIT_PENDING"
        return True

    async def _await_cancel_settle(self, trade: Any, max_wait: float) -> str:
        wait = max(0.0, float(max_wait or 0.0))
        if wait <= 0:
            status = getattr(getattr(trade, "orderStatus", None), "status", "") or ""
            return str(status).upper()
        deadline = time.time() + wait
        terminal = {"FILLED", "CANCELLED", "APICANCELLED", "INACTIVE", "REJECTED"}
        await asyncio.sleep(min(0.025, wait))
        while True:
            status = str(
                getattr(getattr(trade, "orderStatus", None), "status", "") or ""
            ).upper()
            if status in terminal:
                return status
            if time.time() >= deadline:
                return status
            await asyncio.sleep(min(0.05, max(deadline - time.time(), 0.01)))

    def _on_fill(self, intent_id: str, trade: Any, fill: Any) -> None:
        execution = getattr(fill, "execution", None)
        exec_id = str(getattr(execution, "execId", "") or "")
        if exec_id and exec_id in self.seen_fills:
            return
        if exec_id:
            self.seen_fills.add(exec_id)
        intent = self.intents.get(intent_id)
        if intent is None:
            return
        qty = int(round(float(getattr(execution, "shares", 0.0) or 0.0)))
        price = float(getattr(execution, "price", 0.0) or 0.0)
        if qty <= 0 or price <= 0:
            return
        if intent.action == "BUY":
            signal = intent.signal
            quote = self._quote(intent.symbol, intent.contract) or {
                "bid": intent.limit_price,
                "ask": intent.limit_price,
            }
            adverse, adverse_reason, frac = fill_adverse(
                bid=float(quote["bid"]),
                ask=float(quote["ask"]),
                fill_px=price,
                side="BUY",
                cfg=self.risk_cfg,
            )
            if adverse:
                self._event(
                    "ADVERSE_FILL",
                    {
                        "intent_id": intent_id,
                        "side": "BUY",
                        "reason": adverse_reason,
                        "fill_spread_frac": frac,
                        "price": price,
                        "bid": quote["bid"],
                        "ask": quote["ask"],
                    },
                )
            _, qty_frac = self._size(
                intent.symbol,
                signal.sig_ts if signal is not None else time.time(),
                price,
                regime_scale=regime_scale_from_meta(
                    getattr(signal, "meta", None) if signal is not None else None
                ),
            )
            self._apply_open_fill(intent, qty, price, qty_frac, quote)
            if adverse:
                position = self.positions.get(intent.symbol)
                if position is not None and position.status == "OPEN":
                    self._submit_exit(
                        position,
                        "ADVERSE_FILL_FLATTEN",
                        float(quote["bid"]),
                        quote,
                    )
        else:
            quote = self._quote(intent.symbol, intent.contract)
            if quote is not None:
                adverse, adverse_reason, frac = fill_adverse(
                    bid=float(quote["bid"]),
                    ask=float(quote["ask"]),
                    fill_px=price,
                    side="SELL",
                    cfg=self.risk_cfg,
                )
                if adverse:
                    self._event(
                        "ADVERSE_FILL",
                        {
                            "intent_id": intent_id,
                            "side": "SELL",
                            "reason": adverse_reason,
                            "fill_spread_frac": frac,
                            "price": price,
                            "bid": quote["bid"],
                            "ask": quote["ask"],
                        },
                    )
            self._apply_close_fill(intent, qty, price)
        self._event(
            "FILL",
            {
                "intent_id": intent_id,
                "exec_id": exec_id,
                "qty": qty,
                "price": price,
            },
        )
        self.publish_state()

    def _on_commission(self, trade: Any, fill: Any, report: Any) -> None:
        intent_id = str(getattr(getattr(trade, "order", None), "orderRef", "") or "")
        if intent_id not in self.intents:
            return
        execution = getattr(fill, "execution", None)
        exec_id = str(getattr(execution, "execId", "") or "")
        if exec_id and exec_id in self.seen_commissions:
            return
        if exec_id:
            self.seen_commissions.add(exec_id)
        commission = float(getattr(report, "commission", 0.0) or 0.0)
        if math.isfinite(commission) and commission > 0:
            self.realized_pnl -= commission
            self.equity -= commission
            circuit = self.trade_cfg.get("day_circuit")
            if circuit is not None and self.day_start_equity > 0:
                day_ret = self.equity / self.day_start_equity - 1.0
                if day_ret <= float(circuit):
                    self._trip_day_circuit(day_ret=day_ret)
        self._event(
            "COMMISSION",
            {
                "intent_id": intent_id,
                "exec_id": exec_id,
                "commission": commission,
                "currency": str(getattr(report, "currency", "") or ""),
                "realized_pnl": float(getattr(report, "realizedPNL", 0.0) or 0.0),
            },
        )
        self.publish_state()

    def _apply_open_fill(
        self,
        intent: PendingIntent,
        qty: int,
        price: float,
        qty_frac: float,
        quote: dict[str, float],
    ) -> None:
        signal = intent.signal
        if signal is None:
            return
        previous_filled = float(intent.filled)
        total_filled = previous_filled + int(qty)
        intent.avg_fill_price = (
            intent.avg_fill_price * previous_filled + float(price) * int(qty)
        ) / total_filled
        intent.filled = total_filled
        intent.status = "FILLED" if total_filled >= intent.qty else "PARTIALLY_FILLED"
        existing = self.positions.get(intent.symbol)
        if existing and existing.status in {"OPEN", "EXIT_PENDING"}:
            if existing.contract != intent.contract:
                return
            total_qty = existing.qty + int(qty)
            existing.entry_price = (
                existing.entry_price * existing.qty + float(price) * int(qty)
            ) / total_qty
            existing.qty = total_qty
            existing.status = "OPEN"
        else:
            mid = quote_mid(float(quote["bid"]), float(quote["ask"])) or float(price)
            self.positions[intent.symbol] = LivePosition(
                symbol=intent.symbol,
                contract=intent.contract,
                con_id=intent.con_id,
                direction=signal.direction,
                qty=int(qty),
                entry_price=float(price),
                entry_ts=time.time(),
                signal_ts=float(to_ny(signal.sig_ts).timestamp()),
                rank=signal.rank,
                qty_frac=float(qty_frac),
                entry_bid=float(quote["bid"]),
                entry_ask=float(quote["ask"]),
                last_good_mid=float(mid),
            )
            self._remember_option_mid(intent.symbol, intent.contract, float(mid))
        rec = audit_fill(
            float(quote["bid"]),
            float(quote["ask"]),
            float(price),
            "BUY",
            self.fill_model,
        )
        self._event(
            "POSITION_OPEN",
            {
                **asdict(self.positions[intent.symbol]),
                "fill_spread_frac": rec.fill_spread_frac,
            },
        )

    def evaluate_exits(self, asof_ts: float) -> None:
        grace = int(self.trade_cfg.get("exit_mf_grace_seconds", 60))
        hold = int(self.trade_cfg.get("hold_minutes", 30)) * 60
        ext_hold_m = int(self.trade_cfg.get("hold_extend_minutes", 45) or 45)
        ext_hold = max(hold, ext_hold_m * 60)
        ext_mtm_min = float(self.trade_cfg.get("hold_extend_mtm_min", 0.0) or 0.0)
        require_mf_align = bool(self.trade_cfg.get("hold_extend_require_mf", True))
        tp = float(self.trade_cfg.get("tp_mult", 1.6))
        sl = float(self.trade_cfg.get("sl_mult", 0.4))
        exit_mode = str(self.trade_cfg.get("exit_mode") or "none").strip().lower()
        early = str(self.trade_cfg.get("early_exit_mode") or "").strip().lower()
        if early and early not in {"", "none", "off"} and early not in exit_mode:
            exit_mode = f"{exit_mode}+{early}" if exit_mode not in {"", "none"} else early
        blob = exit_mode.replace(",", "+").replace("|", "+")
        use_extend = "hold_extend" in blob or "extend_hold" in blob
        use_floor = "mtm_floor" in blob or "mtm_defend" in blob
        use_mf_flip = "mf_flip" in blob or "mf_reversal" in blob
        use_streak = "streak_break" in blob
        min_hold = self.trade_cfg.get("exit_min_hold_minutes")
        if min_hold is None and ("mf_reversal" in blob or use_floor):
            min_hold = 10.0
        if min_hold is not None:
            grace = max(grace, int(float(min_hold) * 60))
        floor = float(self.trade_cfg.get("mtm_floor_ret", 0.0))
        for symbol, position in list(self.positions.items()):
            if position.status != "OPEN":
                continue
            quote = self._quote(symbol, position.contract)
            if quote is None:
                continue
            position.last_bid = float(quote["bid"])
            position.last_ask = float(quote["ask"])
            mid = quote_mid(float(quote["bid"]), float(quote["ask"]))
            if mid is None:
                continue
            self._remember_option_mid(symbol, position.contract, mid)
            new_good, new_hold, gap_status = observe_exit_mid(
                last_good_mid=position.last_good_mid,
                mid=mid,
                gap_hold_count=position.gap_hold_count,
                cfg=self.risk_cfg,
            )
            position.last_good_mid = new_good
            position.gap_hold_count = new_hold
            if gap_status == "gap":
                self._event(
                    "EXIT_GAP_HOLD",
                    {
                        "symbol": symbol,
                        "mid": mid,
                        "last_good_mid": position.last_good_mid,
                        "gap_hold_count": position.gap_hold_count,
                    },
                )
                continue
            sell_price = self.fills.sell(float(quote["bid"]), float(quote["ask"]))
            reason = ""
            held = asof_ts - position.entry_ts
            mtm_ret = sell_price / position.entry_price - 1.0 if position.entry_price > 0 else float("nan")
            if gap_status == "gap_force":
                reason = "GAP_FLATTEN"
            elif sell_price >= position.entry_price * tp:
                reason = "TP"
            elif sell_price <= position.entry_price * sl:
                reason = "SL"
            elif (
                held >= grace
                and use_floor
                and math.isfinite(mtm_ret)
                and mtm_ret <= floor
            ):
                reason = "MTM_FLOOR"
            elif held >= grace and (use_mf_flip or use_streak):
                state = self.scanner.states.get(symbol)
                mf10 = float(getattr(state, "mf10", float("nan")))
                if math.isfinite(mf10):
                    if use_mf_flip:
                        if position.direction == "UP" and mf10 < 0:
                            reason = "MF_FLIP"
                        elif position.direction == "DN" and mf10 > 0:
                            reason = "MF_FLIP"
                    elif use_streak:
                        if (
                            position.direction == "UP"
                            and int(getattr(state, "streak_up", 0)) == 0
                            and mf10 <= 0
                        ):
                            reason = "STREAK0"
                        elif (
                            position.direction == "DN"
                            and int(getattr(state, "streak_dn", 0)) == 0
                            and mf10 >= 0
                        ):
                            reason = "STREAK0"
            elif use_extend and held >= hold:
                if position.hold_extended:
                    if held >= ext_hold:
                        reason = f"T+{ext_hold_m}"
                elif held >= ext_hold:
                    reason = f"T+{hold // 60}"
                else:
                    mf_ok = True
                    if require_mf_align:
                        state = self.scanner.states.get(symbol)
                        mf10 = float(getattr(state, "mf10", float("nan")))
                        if not math.isfinite(mf10):
                            mf_ok = False
                        elif position.direction == "UP":
                            mf_ok = mf10 > 0
                        else:
                            mf_ok = mf10 < 0
                    mtm_ok = math.isfinite(mtm_ret) and mtm_ret >= ext_mtm_min
                    if mtm_ok and mf_ok and ext_hold > hold:
                        position.hold_extended = True
                    else:
                        reason = f"T+{hold // 60}"
            elif held >= hold:
                reason = f"T+{hold // 60}"
            if reason:
                # Aggressive limit for force exits; model sell for normal rails.
                if reason in FORCE_EXIT_REASONS:
                    limit = float(quote["bid"])
                else:
                    limit = float(sell_price)
                self._submit_exit(position, reason, limit, quote)
        self.retry_pending_signals()

    def _submit_exit(
        self,
        position: LivePosition,
        reason: str,
        limit_price: float,
        quote: dict[str, float],
    ) -> None:
        if position.status == "EXIT_PENDING" and reason not in FORCE_EXIT_REASONS:
            return
        if self.mode != "shadow" and not self.ib.isConnected():
            self._event(
                "EXIT_BLOCKED",
                {"symbol": position.symbol, "reason": reason, "gate": "ibkr_disconnected"},
            )
            return
        attempt = sum(
            1
            for item in self.intents.values()
            if item.action == "SELL"
            and item.symbol == position.symbol
            and item.contract == position.contract
            and item.reason == reason
        )
        intent_id = self._intent_id(
            "SELL",
            position.symbol,
            position.contract,
            f"{position.entry_ts:.3f}:{reason}:{attempt}",
        )
        if intent_id in self.intents:
            return
        mid = quote_mid(float(quote["bid"]), float(quote["ask"])) or float(limit_price)
        intent = PendingIntent(
            intent_id=intent_id,
            action="SELL",
            symbol=position.symbol,
            contract=position.contract,
            con_id=position.con_id,
            qty=position.qty,
            limit_price=float(limit_price),
            reason=reason,
            created_at=time.time(),
            ref_price=float(mid),
        )
        self.intents[intent.intent_id] = intent
        position.status = "EXIT_PENDING"
        self._event(
            "EXIT_INTENT",
            {
                **_intent_dict(intent),
                "bid": quote["bid"],
                "ask": quote["ask"],
                "exit_chase_count": position.exit_chase_count,
            },
        )
        if self.mode == "shadow":
            self._apply_close_fill(intent, position.qty, limit_price)
        else:
            self._place_broker_order(intent)
        self.publish_state()

    def _apply_close_fill(self, intent: PendingIntent, qty: int, price: float) -> None:
        position = self.positions.get(intent.symbol)
        if position is None:
            return
        previous_filled = float(intent.filled)
        total_filled = previous_filled + int(qty)
        intent.avg_fill_price = (
            intent.avg_fill_price * previous_filled + float(price) * int(qty)
        ) / total_filled
        intent.filled = total_filled
        intent.status = "FILLED" if total_filled >= intent.qty else "PARTIALLY_FILLED"
        filled_qty = min(int(qty), position.qty)
        fill_pnl = (float(price) - position.entry_price) * filled_qty * 100.0
        self.realized_pnl += fill_pnl
        self.equity += fill_pnl
        day_ret = (
            self.equity / self.day_start_equity - 1.0
            if self.day_start_equity > 0
            else 0.0
        )
        if qty < position.qty:
            position.qty -= qty
            position.status = "EXIT_PENDING"
            self._event(
                "POSITION_PARTIAL_CLOSE",
                {
                    **asdict(position),
                    "filled_qty": filled_qty,
                    "fill_price": float(price),
                    "fill_pnl": fill_pnl,
                    "remaining_qty": position.qty,
                    "reason": intent.reason,
                },
            )
            circuit = self.trade_cfg.get("day_circuit")
            if circuit is not None and day_ret <= float(circuit):
                self._trip_day_circuit(day_ret=day_ret)
            return
        ret = float(intent.avg_fill_price) / position.entry_price - 1.0
        exit_ts = pd.Timestamp.now(tz="America/New_York")
        self.open_until[intent.symbol] = exit_ts
        self.scanner.record_fill(intent.symbol, exit_ts=exit_ts, won=ret > 0)
        self.connector.release_on_demand_subscription(position.con_id)
        del self.positions[intent.symbol]
        circuit = self.trade_cfg.get("day_circuit")
        if circuit is not None and day_ret <= float(circuit):
            self._trip_day_circuit(day_ret=day_ret)
        self._event(
            "POSITION_CLOSE",
            {
                **asdict(position),
                "exit_price": float(price),
                "exit_ts": float(exit_ts.timestamp()),
                "reason": intent.reason,
                "ret": ret,
                "fill_pnl": fill_pnl,
                "realized_pnl": self.realized_pnl,
                "day_return": day_ret,
                "day_halted": self.day_halted,
            },
        )

    async def reconcile(self) -> bool:
        if self.mode == "shadow":
            self.reconcile_ok = True
            return True
        for value in self.ib.accountValues() or []:
            if (
                self.connector.config.account
                and str(getattr(value, "account", "") or "")
                != self.connector.config.account
            ):
                continue
            if (
                str(getattr(value, "tag", "") or "") == "AvailableFunds"
                and str(getattr(value, "currency", "") or "") in {"USD", "BASE"}
            ):
                available = float(getattr(value, "value", 0.0) or 0.0)
                if available > 0:
                    self.available_funds = available
        managed_contracts = {
            lock.local_symbol
            for values in self.connector.locks.values()
            for lock in values
        }
        managed_contracts.update(pos.contract for pos in self.positions.values())
        managed_contracts.update(intent.contract for intent in self.intents.values())
        broker = {}
        for row in self.ib.positions() or []:
            if (
                self.connector.config.account
                and str(getattr(row, "account", "") or "")
                != self.connector.config.account
            ):
                continue
            contract = getattr(row, "contract", None)
            if str(getattr(contract, "secType", "") or "").upper() != "OPT":
                continue
            local = str(getattr(contract, "localSymbol", "") or "").strip()
            qty = int(round(float(getattr(row, "position", 0.0) or 0.0)))
            if local in managed_contracts and qty:
                broker[local] = qty
        internal = {pos.contract: pos.qty for pos in self.positions.values()}
        self.reconcile_ok = broker == internal
        self._event(
            "RECONCILE",
            {"ok": self.reconcile_ok, "broker": broker, "internal": internal},
        )
        self.publish_state()
        return self.reconcile_ok

    async def reconcile_loop(self, interval_sec: float = 10.0) -> None:
        while True:
            try:
                await self.reconcile()
            except Exception as exc:
                self.reconcile_ok = False
                self._event("RECONCILE_ERROR", {"error": str(exc)})
            await asyncio.sleep(interval_sec)

    async def order_watchdog_loop(self, interval_sec: float = 1.0) -> None:
        if self.mode == "shadow":
            return
        # Prefer short per-round requote timeouts; fall back to legacy trade keys.
        entry_timeout = float(
            self.trade_cfg.get(
                "entry_order_timeout_seconds",
                self.requote_cfg.entry_timeout_sec,
            )
        )
        exit_timeout = float(
            self.trade_cfg.get(
                "exit_order_timeout_seconds",
                self.requote_cfg.exit_timeout_sec,
            )
        )
        if self.requote_cfg.enabled:
            entry_timeout = min(entry_timeout, self.requote_cfg.entry_timeout_sec)
            exit_timeout = min(exit_timeout, self.requote_cfg.exit_timeout_sec)
        while True:
            now = time.time()
            for intent_id, intent in list(self.intents.items()):
                if intent.status.upper() in TERMINAL_STATUSES:
                    continue
                if intent.status.upper() == "CANCEL_REQUESTED":
                    continue
                if intent.replaced_by:
                    continue
                timeout = entry_timeout if intent.action == "BUY" else exit_timeout
                if now - intent.created_at < timeout:
                    continue
                trade = self.trades.get(intent_id)
                if trade is None:
                    continue
                self._event(
                    "ORDER_TIMEOUT",
                    {
                        "intent_id": intent_id,
                        "action": intent.action,
                        "age_sec": now - intent.created_at,
                        "requote_attempt": intent.requote_attempt,
                    },
                )
                intent.status = "CANCEL_REQUESTED"
                try:
                    self.ib.cancelOrder(trade.order)
                except Exception as exc:
                    self._event(
                        "ORDER_CANCEL_ERROR",
                        {"intent_id": intent_id, "error": str(exc)},
                    )
                    continue
                status = await self._await_cancel_settle(
                    trade, self.requote_cfg.cancel_settle_sec
                )
                # Callback may already have requoted; if not, fold status once.
                fresh = self.intents.get(intent_id)
                if fresh is None:
                    continue
                if fresh.replaced_by or fresh.status.upper() in TERMINAL_STATUSES:
                    continue
                if status in TERMINAL_STATUSES or status in {
                    "CANCELLED",
                    "APICANCELLED",
                    "REJECTED",
                    "FILLED",
                }:
                    self._on_trade_status(intent_id, trade)
            await asyncio.sleep(interval_sec)

    async def broker_recovery_loop(self, interval_sec: float = 30.0) -> None:
        if self.mode == "shadow":
            return
        while True:
            await asyncio.sleep(interval_sec)
            if not self.ib.isConnected():
                continue
            try:
                await self.recover_broker_activity()
            except Exception as exc:
                self._event("RECOVERY_WARNING", {"reason": "periodic", "error": str(exc)})

    def force_flatten(self, reason: str = "EOD") -> None:
        for position in list(self.positions.values()):
            if position.status == "OPEN":
                pass
            elif (
                position.status == "EXIT_PENDING"
                and reason in FORCE_EXIT_REASONS
                and reason != "DAY_CIRCUIT"
            ):
                # Escalate soft/pending exits (EOD / chase / adverse), but do not
                # double-submit while a day-circuit close is already in flight.
                position.status = "OPEN"
            else:
                continue
            quote = self._position_quote_fallback(position)
            if quote is None:
                self._event(
                    "EXIT_BLOCKED",
                    {
                        "symbol": position.symbol,
                        "reason": reason,
                        "gate": "quote_missing",
                    },
                )
                continue
            limit_price = float(quote["bid"])
            self._submit_exit(position, reason, limit_price, quote)
        self.publish_state()

    def cancel_open_orders(self) -> None:
        if self.mode == "shadow":
            return
        for trade in self.ib.openTrades() or []:
            order_ref = str(getattr(trade.order, "orderRef", "") or "")
            if order_ref.startswith(f"M7-{self.session_id[:8]}-"):
                self.ib.cancelOrder(trade.order)

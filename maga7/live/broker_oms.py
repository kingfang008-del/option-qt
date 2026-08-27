"""Event-driven Mag7 Shadow/Paper/Live OMS with hard live-trading gates."""
from __future__ import annotations

import asyncio
import csv
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

from maga7.common.exit_arms import build_exit_arms, build_exit_health
from maga7.common.fills import FillSpec
from maga7.common.hold_watchdog import hold_watchdog_from_trade, qqq_adverse_from_prices
from maga7.common.profit_protect import profit_protect_from_raw, profit_protect_on_tick
from maga7.common.delta_time_stop import (
    StockRevExitConfig,
    stock_rev_applies_to_route,
    stock_rev_exit_from_trade,
)
from maga7.common.confirm_abort import (
    ConfirmAbortState,
    confirm_abort_applies,
    confirm_abort_from_raw,
    confirm_abort_on_tick,
)
from maga7.common.wave_confirm import WaveAbortState, wave_abort_from_trade, wave_abort_on_tick
from maga7.common.ladder_active import ladder_active_from_trade
from maga7.common.path_fast_pack import (
    apply_path_fast_pack_overrides,
    path_fast_pack_day_should_arm,
    path_fast_pack_from_trade,
)
from maga7.common.option_trades import (
    trade_toxic_cut_ret,
    trade_toxic_from_trade,
    trade_toxic_in_cut_window,
    trade_toxic_is_dig,
)
from maga7.common.position_size import apply_size_scale, regime_scale_from_meta, resolve_size_frac
from maga7.common.replay import to_ny
from maga7.live.iceberg import (
    decode_chunk_queue,
    encode_chunk_queue,
    iceberg_config_from_trade,
    plan_entry_chunks,
)
from maga7.live.requote import (
    entry_requote_limit,
    exit_requote_limit,
    is_urgent_exit_reason,
    requote_config_from_trade,
)
from maga7.live.risk_guards import (
    entry_feed_ok,
    entry_quote_ok,
    entry_stock_drift_ok,
    fill_adverse,
    is_fresh,
    next_entry_quote_stable_ticks,
    observe_exit_mid,
    quote_mid,
    quote_spread_fields,
    risk_config_from_trade,
    signal_quote_lag_ok,
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
    "TRADE_TOX_RECONNECT",
    "RECOVERY_GUARD_FLATTEN",
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
    # Quote-proxy MFE for live trade_toxic (offline uses OPRA last prints).
    peak_mfe: float = 0.0
    trade_dig_since: float = 0.0
    entry_stock_px: float = 0.0
    entry_qqq_px: float = 0.0
    # Post-fill wave confirm (revocable); see maga7.common.wave_confirm.
    wave_armed: bool = False
    wave_done: bool = False
    # Set on OMS restore; first good quote rechecks toxic with max_cut bypass.
    toxic_reconnect_pending: bool = False
    # Second-level ladder_active bookkeeping (research path).
    ladder_peak_ret: float = float("-inf")
    ladder_peak_ts: float = 0.0
    ladder_trail_armed: bool = False
    ladder_trail_dd: float = 0.05
    # Satellite sleeve exit overrides (e.g. qqq_open_cont / am_pulse tp/sl).
    exit_tp_mult: float | None = None
    exit_sl_mult: float | None = None
    exit_hold_sec: float | None = None
    exit_simple: bool = False
    exit_flatten_before: str | None = None  # HH:MM NY — force exit before CORE
    # AM_EXT post-fill confirm-or-abort (option mark); see confirm_abort.py.
    confirm_abort: dict[str, Any] | None = None
    confirm_abort_confirmed: bool = False
    confirm_abort_done: bool = False
    # Peak-armed profit floor for satellite sleeves.
    profit_protect: dict[str, Any] | None = None
    route: str = "baseline"  # baseline | hunt | am_pulse | …


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
    # Entry iceberg: remaining clips as "3,2"; empty = not an iceberg / done.
    iceberg_queue: str = ""
    iceberg_chunk_idx: int = 0
    iceberg_chunks: int = 1
    iceberg_total_qty: int = 0
    iceberg_qty_frac: float = 0.0
    iceberg_started_at: float = 0.0


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


def _live_position_from_dict(payload: dict[str, Any]) -> LivePosition:
    allowed = {item.name for item in fields(LivePosition)}
    return LivePosition(**{k: v for k, v in dict(payload or {}).items() if k in allowed})


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
        self.iceberg_cfg = iceberg_config_from_trade(self.trade_cfg)
        self.trade_toxic = trade_toxic_from_trade(self.trade_cfg)
        self.hold_watchdog = hold_watchdog_from_trade(self.trade_cfg)
        self._path_fast_pack = path_fast_pack_from_trade(self.trade_cfg)
        self._path_fast_armed: bool | None = None
        self.profile_hash = str(
            profile.get("_live_fingerprint") or profile_digest(profile)
        )
        self.positions: dict[str, LivePosition] = {}
        self.intents: dict[str, PendingIntent] = {}
        self.trades: dict[str, Any] = {}
        self.pending_signals: dict[str, tuple[ScannerSignal, float]] = {}
        self.open_until: dict[str, pd.Timestamp] = {}
        self.exit_reason_counts: dict[str, int] = {}
        self._last_size_reject: dict[str, Any] | None = None
        self.seen_fills: set[str] = set()
        self.seen_commissions: set[str] = set()
        self._bound_order_refs: set[str] = set()
        self._last_seen_option_mid: dict[tuple[str, str], float] = {}
        self._entry_quote_stable: dict[tuple[str, str], int] = {}
        self._last_entry_quote_ts: dict[tuple[str, str], float] = {}
        self._last_gap_event_ts: float = 0.0
        self._pending_force_exits: dict[str, str] = {}
        self._flattening_circuit = False
        self.reconcile_ok = self.mode == "shadow"
        self.last_reconcile: dict[str, Any] = {
            "ok": self.reconcile_ok,
            "broker": {},
            "internal": {},
            "ts": None,
        }
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
    def trade_spreads_path(self) -> Path:
        """One row per OPEN/CLOSE fill with bid/ask/spread for Dash."""
        return self.session_dir / "trade_spreads.csv"

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

    def _record_trade_spread(
        self,
        *,
        action: str,
        symbol: str,
        contract: str,
        side: str,
        fill_px: float,
        qty: int,
        bid: float,
        ask: float,
        reason: str = "",
        ret: float | None = None,
    ) -> dict[str, Any]:
        """Persist OPEN/CLOSE quote spread for Dash (csv + event payload fields)."""
        fields = quote_spread_fields(bid, ask, fill_px=fill_px, side=side)
        row = {
            "ts": time.time(),
            "session_id": self.session_id,
            "mode": self.mode,
            "action": str(action).upper(),
            "symbol": str(symbol).upper(),
            "contract": contract,
            "side": str(side).upper(),
            "qty": int(qty),
            "fill_px": float(fill_px),
            "bid": fields.get("bid"),
            "ask": fields.get("ask"),
            "spread": fields.get("spread"),
            "spread_pct": fields.get("spread_pct"),
            "fill_spread_frac": fields.get("fill_spread_frac"),
            "reason": reason or "",
            "ret": ret,
        }
        path = self.trade_spreads_path
        path.parent.mkdir(parents=True, exist_ok=True)
        cols = [
            "ts",
            "session_id",
            "mode",
            "action",
            "symbol",
            "contract",
            "side",
            "qty",
            "fill_px",
            "bid",
            "ask",
            "spread",
            "spread_pct",
            "fill_spread_frac",
            "reason",
            "ret",
        ]
        write_header = not path.is_file() or path.stat().st_size == 0
        with path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=cols, extrasaction="ignore")
            if write_header:
                writer.writeheader()
            writer.writerow(row)
        return row

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
            "last_reconcile": dict(self.last_reconcile or {}),
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
            "data_guard_state": {
                "last_gap_event_ts": float(self._last_gap_event_ts or 0.0),
                "entry_quote_stable": {
                    f"{symbol}\t{contract}": int(value)
                    for (symbol, contract), value in self._entry_quote_stable.items()
                },
                "last_entry_quote_ts": {
                    f"{symbol}\t{contract}": float(value)
                    for (symbol, contract), value in self._last_entry_quote_ts.items()
                },
                "last_seen_option_mid": {
                    f"{symbol}\t{contract}": float(value)
                    for (symbol, contract), value in self._last_seen_option_mid.items()
                },
                "pending_force_exits": dict(self._pending_force_exits),
            },
            "exit_reason_counts": dict(self.exit_reason_counts),
            "exit_arms": self.exit_arms_snapshot(),
            "exit_health": self.exit_health_snapshot(),
        }

    def exit_arms_snapshot(self) -> dict[str, Any]:
        return build_exit_arms(self.trade_cfg, reason_counts=self.exit_reason_counts)

    def exit_health_snapshot(self) -> dict[str, Any]:
        return build_exit_health(
            self.exit_reason_counts, arms=self.exit_arms_snapshot()
        )

    def _seed_exit_reason_counts_from_events(self) -> None:
        """Backfill day close reasons from order_events after resume."""
        path = self.event_path
        if not path.is_file():
            return
        counts: dict[str, int] = {}
        try:
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except Exception:
                        continue
                    if str(row.get("kind") or "") != "POSITION_CLOSE":
                        continue
                    reason = str(row.get("reason") or "UNKNOWN")
                    counts[reason] = int(counts.get(reason, 0)) + 1
        except Exception:
            logger.exception("seed exit_reason_counts from events failed")
            return
        if counts:
            self.exit_reason_counts = counts

    def _watchdog_hunt_snapshot(self) -> dict[str, Any]:
        """Compact Watchdog/Hunt counters for oms_meta / Dash (P3 observability)."""
        sc = self.scanner
        wd = getattr(sc, "watchdog", None)
        pending = list(getattr(sc, "pending_hunts", None) or [])
        return {
            "state": getattr(sc, "_watchdog_state", "off"),
            "reason": getattr(sc, "_watchdog_reason", "off"),
            "route": getattr(sc, "_watchdog_route", "baseline"),
            "day_halt": bool(getattr(sc, "_day_halt", False)),
            "hunt_armed": bool(getattr(wd, "hunt_armed", False)) if wd is not None else False,
            "n_hunt_candidates": int(len(getattr(wd, "hunt_candidates", None) or [])),
            "n_hunt_signals": int(getattr(sc, "n_hunt_signals", 0) or 0),
            "n_hunt_emitted": int(getattr(sc, "n_hunt_emitted", 0) or 0),
            "n_hunt_budget_skip": int(getattr(sc, "n_hunt_budget_skip", 0) or 0),
            "n_hunt_mutex_skip": int(getattr(sc, "n_hunt_mutex_skip", 0) or 0),
            "pending_hunts": int(len(pending)),
            "day_hunt_symbols": sorted(
                str(s) for s in (getattr(sc, "day_hunt_symbols", None) or set())
            ),
        }

    def _meta_payload(self, snapshot: dict[str, Any] | None = None) -> dict[str, Any]:
        snap = snapshot or self.snapshot()
        return {
            "schema_version": snap.get("schema_version"),
            "session_id": snap.get("session_id"),
            "trade_date": snap.get("trade_date"),
            "mode": snap.get("mode"),
            "profile_hash": snap.get("profile_hash"),
            "updated_at": snap.get("updated_at"),
            "reconcile_ok": snap.get("reconcile_ok"),
            "equity": snap.get("equity"),
            "day_start_equity": snap.get("day_start_equity"),
            "realized_pnl": snap.get("realized_pnl"),
            "day_halted": snap.get("day_halted"),
            "available_funds": snap.get("available_funds"),
            "account_ready": snap.get("account_ready"),
            "last_reconcile": snap.get("last_reconcile") or {},
            "n_positions": len(self.positions),
            "n_intents": len(
                [
                    intent
                    for intent in self.intents.values()
                    if intent.status.upper() not in TERMINAL_STATUSES
                ]
            ),
            "exit_arms": snap.get("exit_arms") or self.exit_arms_snapshot(),
            "exit_health": snap.get("exit_health") or self.exit_health_snapshot(),
            "exit_reason_counts": snap.get("exit_reason_counts")
            or dict(self.exit_reason_counts),
            "watchdog": self._watchdog_hunt_snapshot(),
        }

    def publish_state(self) -> None:
        payload = self.snapshot()
        _atomic_json(self.state_path, payload)
        try:
            _atomic_json(
                self.session_dir / "exit_health.json",
                {
                    "session_id": self.session_id,
                    "trade_date": self.trade_date,
                    "updated_at": payload.get("updated_at"),
                    "exit_arms": payload.get("exit_arms"),
                    "exit_health": payload.get("exit_health"),
                },
            )
        except Exception:
            logger.exception("failed to write exit_health.json")
        meta_packed = json.dumps(self._meta_payload(payload), ensure_ascii=True, default=str)
        pipe = self.redis.pipeline(transaction=True)
        pipe.set(f"maga7:oms_meta:{self.session_id}", meta_packed)
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
            if not bool(self.profile.get("_allow_code_drift", False)):
                raise RuntimeError("OMS state profile hash mismatch")
            logger.warning(
                "OMS resume profile_hash mismatch saved=%s live=%s "
                "(explicit code-drift override)",
                str(raw.get("profile_hash") or "")[:12],
                str(self.profile_hash or "")[:12],
            )
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
        if isinstance(raw.get("last_reconcile"), dict):
            self.last_reconcile = dict(raw["last_reconcile"])
        self.account_ready = bool(raw.get("account_ready", self.account_ready))
        self.seen_fills = set(raw.get("seen_fills") or [])
        self.seen_commissions = set(raw.get("seen_commissions") or [])
        self.positions = {
            key: _live_position_from_dict(value)
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
        guard = raw.get("data_guard_state") or {}

        def _restore_contract_map(name: str, cast):
            restored = {}
            for key, value in (guard.get(name) or {}).items():
                parts = str(key).split("\t", 1)
                if len(parts) != 2:
                    continue
                try:
                    restored[(parts[0].upper(), parts[1])] = cast(value)
                except (TypeError, ValueError):
                    continue
            return restored

        self._last_gap_event_ts = float(guard.get("last_gap_event_ts") or 0.0)
        self._entry_quote_stable = _restore_contract_map(
            "entry_quote_stable", int
        )
        self._last_entry_quote_ts = _restore_contract_map(
            "last_entry_quote_ts", float
        )
        self._last_seen_option_mid = _restore_contract_map(
            "last_seen_option_mid", float
        )
        self._pending_force_exits = {
            str(symbol).upper(): str(reason)
            for symbol, reason in (guard.get("pending_force_exits") or {}).items()
        }
        raw_counts = raw.get("exit_reason_counts") or {}
        if isinstance(raw_counts, dict):
            self.exit_reason_counts = {
                str(k): int(v) for k, v in raw_counts.items() if int(v) > 0
            }
        if not self.exit_reason_counts:
            self._seed_exit_reason_counts_from_events()
        armed = 0
        if self.trade_toxic.enabled:
            for position in self.positions.values():
                if position.status in {"OPEN", "EXIT_PENDING"}:
                    position.toxic_reconnect_pending = True
                    armed += 1
        reconcile = self._reconcile_restored_occupancy()
        self._event(
            "STATE_RESTORED",
            {
                "positions": len(self.positions),
                "intents": len(self.intents),
                "toxic_reconnect_armed": armed,
                "occupancy": sorted(self._active_occupancy_symbols()),
                "open_until": {
                    key: value.isoformat() for key, value in self.open_until.items()
                },
                **reconcile,
            },
        )
        if armed:
            self._event(
                "TOXIC_RECONNECT_ARMED",
                {"n_positions": armed, "cut_ret": float(self.trade_toxic.cut_ret)},
            )

    def _wall_clock_ny(self) -> pd.Timestamp:
        return pd.Timestamp.now(tz="America/New_York")

    def _sizing_clock(self, entry_ts) -> pd.Timestamp:
        """Concurrent seats must follow wall time on live deferred signals.

        ``open_until`` is stamped at wall-clock close. Comparing it to a lagged
        ``sig_ts`` (ENTRY_WAIT retry) keeps just-closed names occupied and
        falsely trips ``size_gate`` / max_concurrent.
        """
        signal_clock = to_ny(entry_ts)
        wall = self._wall_clock_ny()
        if signal_clock is None:
            return wall
        return wall if wall > signal_clock else signal_clock

    def _active_occupancy_symbols(self) -> list[str]:
        out: list[str] = []
        for symbol, position in self.positions.items():
            if position.status in {"OPEN", "EXIT_PENDING"}:
                out.append(str(symbol).upper())
        return out

    def _reconcile_restored_occupancy(self) -> dict[str, Any]:
        """Drop stale cooldowns; finish shadow EXIT_PENDING so seats free on resume."""
        wall = self._wall_clock_ny()
        pruned = [
            key
            for key, until in list(self.open_until.items())
            if until is None or to_ny(until) <= wall
        ]
        for key in pruned:
            self.open_until.pop(key, None)
        shadow_closed: list[str] = []
        if self.mode == "shadow":
            for symbol, position in list(self.positions.items()):
                if position.status != "EXIT_PENDING":
                    continue
                quote = self._position_quote_fallback(position)
                if quote is None:
                    mid = float(position.last_good_mid or position.entry_price or 0.0)
                    if mid <= 0:
                        # No mark — scrub ghost seat rather than block sizing forever.
                        self.positions.pop(symbol, None)
                        shadow_closed.append(str(symbol).upper())
                        self._event(
                            "RESUME_SCRUB_POSITION",
                            {
                                "symbol": symbol,
                                "reason": "exit_pending_no_quote",
                            },
                        )
                        continue
                    quote = {"bid": mid, "ask": mid, "ts": time.time()}
                limit = float(quote["bid"])
                self._event(
                    "RESUME_FORCE_CLOSE",
                    {
                        "symbol": symbol,
                        "reason": "exit_pending_on_restore",
                        "limit": limit,
                    },
                )
                # Re-open status so _submit_exit accepts the flatten.
                position.status = "OPEN"
                self._submit_exit(position, "RESUME_FLATTEN", limit, quote)
                if symbol not in self.positions:
                    shadow_closed.append(str(symbol).upper())
        return {
            "open_until_pruned": pruned,
            "shadow_exit_pending_closed": shadow_closed,
        }

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
        if not is_fresh(
            float(quote.get("ts", 0.0)),
            now=time.time(),
            max_age_sec=self.risk_cfg.max_option_staleness_sec,
            max_future_skew_sec=self.risk_cfg.max_future_skew_sec,
        ):
            return None
        return quote

    def _stock_last_close(self, symbol: str) -> float | None:
        state = getattr(self.scanner, "states", {}).get(str(symbol).upper())
        bars = getattr(state, "bars", None) or []
        if not bars:
            return None
        try:
            px = float(bars[-1]["close"])
        except Exception:
            return None
        if not math.isfinite(px) or px <= 0:
            return None
        return px

    def _qqq_last_close(self) -> float | None:
        # Live QQQ feeds scanner.stock_by via on_reference_second (not states).
        stock_by = getattr(self.scanner, "stock_by", None) or {}
        sdf = stock_by.get("QQQ")
        if sdf is not None and not getattr(sdf, "empty", True):
            try:
                px = float(sdf.iloc[-1]["close"])
                if math.isfinite(px) and px > 0:
                    return px
            except Exception:
                pass
        gate = getattr(self.scanner, "regime_gate", None)
        for attr in ("qqq_last_close", "last_qqq_close", "qqq_close"):
            raw = getattr(gate, attr, None)
            if raw is not None:
                try:
                    px = float(raw)
                    if math.isfinite(px) and px > 0:
                        return px
                except Exception:
                    pass
        return self._stock_last_close("QQQ")

    def _hold_shock_reason(
        self,
        position: LivePosition,
        *,
        mtm_ret: float,
        held: float,
    ) -> str:
        cfg = self.hold_watchdog
        if not cfg.enabled:
            return ""
        if held < float(cfg.min_hold_seconds or 0):
            return ""
        if position.entry_qqq_px <= 0:
            q0 = self._qqq_last_close()
            if q0 is not None:
                position.entry_qqq_px = float(q0)
        if position.entry_qqq_px <= 0:
            return ""
        now_qqq = self._qqq_last_close()
        if now_qqq is None:
            return ""
        fired, _signed = qqq_adverse_from_prices(
            entry_px=float(position.entry_qqq_px),
            now_px=float(now_qqq),
            direction=str(position.direction),
            thresh=float(cfg.qqq_adverse_from_entry),
        )
        if not fired:
            return ""
        mtm_gate = cfg.require_option_mtm_max
        if mtm_gate is not None and (
            not math.isfinite(mtm_ret) or float(mtm_ret) > float(mtm_gate)
        ):
            return ""
        return "HOLD_SHOCK"

    def _stock_adverse_from_entry(self, position: LivePosition) -> float | None:
        entry_px = float(position.entry_stock_px or 0.0)
        if entry_px <= 0:
            return None
        cur = self._stock_last_close(position.symbol)
        if cur is None:
            return None
        ret = cur / entry_px - 1.0
        if str(position.direction).upper() == "UP":
            return float(-ret)
        return float(ret)

    def _trade_toxic_reason(
        self,
        position: LivePosition,
        *,
        mtm_ret: float,
        held: float,
        asof_ts: float,
    ) -> str:
        """Return TRADE_TOX / TRADE_TOX_RECONNECT, or empty if no cut."""
        cfg = self.trade_toxic
        if not cfg.enabled:
            position.toxic_reconnect_pending = False
            return ""
        if math.isfinite(mtm_ret) and mtm_ret > float(position.peak_mfe):
            position.peak_mfe = float(mtm_ret)
        reconnect = bool(position.toxic_reconnect_pending)
        in_window = trade_toxic_in_cut_window(
            held, cfg, bypass_max_cut=reconnect
        )
        dig = trade_toxic_is_dig(
            mtm_ret=mtm_ret,
            peak_mfe=float(position.peak_mfe),
            cfg=cfg,
            stock_adverse=self._stock_adverse_from_entry(position),
            # Live positions are marked from executable quote-sell prices.
            cut_ret=trade_toxic_cut_ret(cfg, mark_source="quote"),
        )
        if dig and in_window:
            if position.trade_dig_since <= 0:
                position.trade_dig_since = float(asof_ts)
            persist_ok = (float(asof_ts) - float(position.trade_dig_since)) >= float(
                cfg.persist_seconds or 0
            )
            qconf = cfg.quote_confirm_ret
            quote_ok = True if qconf is None else float(mtm_ret) <= -float(qconf)
            if persist_ok and quote_ok:
                position.toxic_reconnect_pending = False
                return "TRADE_TOX_RECONNECT" if reconnect else "TRADE_TOX"
            # Keep reconnect arm while waiting on persist / quote confirm.
            return ""
        position.trade_dig_since = 0.0
        # Recheck complete: not currently toxic under reconnect rules.
        position.toxic_reconnect_pending = False
        return ""

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
            max_future_skew_sec=self.risk_cfg.max_future_skew_sec,
        ):
            return False, "stock_stale"
        return True, "ok"

    def _stock_lag_map(self, *, now: float | None = None) -> dict[str, float | None]:
        now_ts = time.time() if now is None else float(now)
        ticks = getattr(self.connector, "last_stock_tick", None) or {}
        symbols = list(getattr(self.connector, "symbols", None) or [])
        if not symbols:
            symbols = list(getattr(self.connector, "trade_symbols", None) or [])
        out: dict[str, float | None] = {}
        for raw in symbols:
            sym = str(raw).upper()
            last = ticks.get(sym)
            if last is None:
                out[sym] = None
            else:
                out[sym] = float(now_ts) - float(last)
        return out

    def _entry_data_ok(self, *, now: float | None = None) -> tuple[bool, str]:
        """Refuse opens when IB/MD/universe looks unhealthy or post-gap cooldown."""
        now_ts = time.time() if now is None else float(now)
        connected = bool(getattr(self.ib, "isConnected", lambda: False)())
        data_mode = getattr(self.connector, "data_mode", None)
        return entry_feed_ok(
            connected=connected,
            data_mode=str(data_mode) if data_mode is not None else None,
            stock_lags_sec=self._stock_lag_map(now=now_ts),
            cfg=self.risk_cfg,
            now=now_ts,
            last_gap_event_ts=float(self._last_gap_event_ts or 0.0),
        )

    def _remember_option_mid(self, symbol: str, contract: str, mid: float) -> None:
        if math.isfinite(mid) and mid > 0:
            self._last_seen_option_mid[(symbol.upper(), contract)] = float(mid)

    def _reset_entry_quote_stable(self, symbol: str, contract: str) -> None:
        key = (str(symbol).upper(), contract)
        self._entry_quote_stable.pop(key, None)
        self._last_entry_quote_ts.pop(key, None)

    def _note_gap_event(self, *, now: float | None = None) -> None:
        self._last_gap_event_ts = time.time() if now is None else float(now)

    def on_feed_reconnected(self) -> None:
        """Fail closed after an IB reconnect and flush queued force exits."""
        now = time.time()
        self._note_gap_event(now=now)
        self._entry_quote_stable.clear()
        self._last_entry_quote_ts.clear()
        for position in self.positions.values():
            if position.status in {"OPEN", "EXIT_PENDING"}:
                position.toxic_reconnect_pending = True
        queued = len(self._pending_force_exits)
        self._flush_pending_force_exits()
        self._event(
            "FEED_RECONNECTED",
            {"queued_force_exits": queued, "cooldown_started_at": now},
        )
        self.publish_state()

    def _flush_pending_force_exits(self) -> None:
        for symbol, reason in list(self._pending_force_exits.items()):
            position = self.positions.get(symbol)
            if position is None:
                self._pending_force_exits.pop(symbol, None)
                continue
            quote = self._position_quote_fallback(position)
            if quote is None:
                continue
            self._pending_force_exits.pop(symbol, None)
            position.status = "OPEN"
            self._submit_exit(position, reason, float(quote["bid"]), quote)

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
        position_frac_override: float | None = None,
    ) -> tuple[int, float]:
        top_k = max(int((self.profile.get("signal") or {}).get("top_k", 2)), 1)
        sizing_clock = self._sizing_clock(entry_ts)
        occupancy = dict(self.open_until)
        for active_symbol, position in self.positions.items():
            if position.status in {"OPEN", "EXIT_PENDING"}:
                occupancy[active_symbol] = sizing_clock + pd.Timedelta(days=1)
        frac, _, n_conc, allow, size_reason = resolve_size_frac(
            self.trade_cfg,
            top_k=top_k,
            open_until=occupancy,
            symbol=symbol,
            entry_ts=sizing_clock,
        )
        # Hunt may set meta.position_frac (mirror offline hunt_trade_overrides).
        if position_frac_override is not None and float(position_frac_override) > 0:
            frac = float(position_frac_override)
        frac = apply_size_scale(frac, regime_scale)
        if (not allow) or frac <= 0.0:
            self._last_size_reject = {
                "allow": bool(allow),
                "size_reason": str(size_reason),
                "n_concurrent": int(n_conc),
                "occupancy": sorted(str(k).upper() for k in occupancy),
                "sizing_clock": sizing_clock.isoformat(),
                "frac": float(frac),
            }
            return 0, float(frac)
        capital = min(self.equity, self.available_funds)
        qty = int((capital * float(frac)) // max(entry_price * 100.0, 0.01))
        if qty <= 0:
            self._last_size_reject = {
                "allow": True,
                "size_reason": "qty_floor",
                "n_concurrent": int(n_conc),
                "occupancy": sorted(str(k).upper() for k in occupancy),
                "sizing_clock": sizing_clock.isoformat(),
                "frac": float(frac),
            }
            return 0, float(frac)
        self._last_size_reject = None
        return min(self.max_qty, qty), float(frac)

    @staticmethod
    def _signal_source_fields(signal: ScannerSignal | None) -> dict[str, Any]:
        meta = getattr(signal, "meta", None) or {}
        return {
            "event_source": meta.get("event_source", "baseline"),
            "watchdog_state": meta.get("watchdog_state"),
            "watchdog_reason": meta.get("watchdog_reason"),
            "route": meta.get("route"),
            "hunt_detector": meta.get("hunt_detector"),
        }

    @staticmethod
    def _position_frac_override(signal: ScannerSignal | None) -> float | None:
        meta = getattr(signal, "meta", None) or {}
        raw = meta.get("position_frac")
        if raw is None:
            return None
        try:
            val = float(raw)
        except (TypeError, ValueError):
            return None
        return val if val > 0 else None

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

    def _has_active_buy(self, symbol: str) -> bool:
        terminal = TERMINAL_STATUSES | {"FILLED"}
        for intent in self.intents.values():
            if (
                intent.action == "BUY"
                and intent.symbol == str(symbol).upper()
                and str(intent.status).upper() not in terminal
            ):
                return True
        return False

    def process_signal(self, signal: ScannerSignal) -> bool:
        symbol = signal.symbol.upper()
        contract = str(signal.contract or "").replace("O:", "")
        if not contract or self.has_position(symbol) or self._has_active_buy(symbol):
            return False
        # AM satellite lanes default execute_mode=shadow — only fill in OMS shadow.
        meta0 = getattr(signal, "meta", None) or {}
        event_source0 = str(meta0.get("event_source") or "")
        route0 = str(meta0.get("route") or "")
        if event_source0 in {
            "am_pulse_sleeve",
            "am_pulse_extension_sleeve",
            "am_v2_sleeve",
        } or route0 in {"am_pulse", "am_pulse_extension", "am_v2"}:
            exec_mode = str(meta0.get("execute_mode") or "shadow").lower()
            off_reason = (
                "am_v2_execute_off"
                if event_source0 == "am_v2_sleeve" or route0 == "am_v2"
                else "am_pulse_execute_off"
            )
            shadow_reason = (
                "am_v2_shadow_only"
                if event_source0 == "am_v2_sleeve" or route0 == "am_v2"
                else "am_pulse_shadow_only"
            )
            if exec_mode in {"off", "audit", "false", "0"}:
                self._event(
                    "ENTRY_SHADOW",
                    {
                        "symbol": symbol,
                        "reason": off_reason,
                        **self._signal_source_fields(signal),
                    },
                )
                return False
            if exec_mode == "shadow" and self.mode != "shadow":
                self._event(
                    "ENTRY_REJECT",
                    {
                        "symbol": symbol,
                        "reason": shadow_reason,
                        "oms_mode": self.mode,
                        **self._signal_source_fields(signal),
                    },
                )
                return False
        if self.day_halted:
            reason = "day_halted"
            meta = getattr(self.scanner, "event_blackout_meta", None) or {}
            if meta.get("active_today"):
                reason = "event_blackout"
            self._event("ENTRY_REJECT", {"symbol": symbol, "reason": reason})
            return False
        feed_ok, feed_reason = self._entry_data_ok()
        if not feed_ok:
            kind = (
                "ENTRY_REJECT"
                if feed_reason.startswith("market_data_") or feed_reason == "ibkr_disconnected"
                else "ENTRY_WAIT"
            )
            self._event(kind, {"symbol": symbol, "reason": feed_reason})
            if kind == "ENTRY_WAIT":
                self._defer_signal(signal)
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
        drift_cfg = meta0.get("entry_stock_drift_gate")
        if isinstance(drift_cfg, dict) and drift_cfg.get("enabled"):
            stock_bar = (getattr(self.connector, "stock_bars", None) or {}).get(
                symbol
            ) or {}
            current_spot = float(stock_bar.get("close") or 0.0)
            drift_ok, drift_reason, directional_drift = entry_stock_drift_ok(
                signal_spot=float(getattr(signal, "spot", 0.0) or 0.0),
                current_spot=current_spot,
                direction=str(getattr(signal, "direction", "") or ""),
                max_chase=float(drift_cfg.get("max_chase", 0.003) or 0.003),
                max_reversal=float(
                    drift_cfg.get("max_reversal", 0.0015) or 0.0015
                ),
            )
            if not drift_ok:
                self._event(
                    "ENTRY_REJECT",
                    {
                        "symbol": symbol,
                        "reason": drift_reason,
                        "directional_drift": directional_drift,
                        "signal_spot": getattr(signal, "spot", None),
                        "current_spot": current_spot,
                        **self._signal_source_fields(signal),
                    },
                )
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
            self._reset_entry_quote_stable(symbol, contract)
            self._event("ENTRY_WAIT", {"symbol": symbol, "reason": "option_stale_or_missing"})
            self._defer_signal(signal)
            return False
        quote_key = (symbol, contract)
        prev_mid = self._last_seen_option_mid.get(quote_key)
        quote_ts = float(quote.get("ts") or 0.0)
        prev_quote_ts = self._last_entry_quote_ts.get(quote_key)
        route_max_lag = meta0.get("max_lag_sec")
        try:
            route_max_lag = float(route_max_lag) if route_max_lag is not None else None
        except (TypeError, ValueError):
            route_max_lag = None
        lag_anchor = to_ny(signal.sig_ts)
        decision_ts_raw = meta0.get("decision_ts")
        if decision_ts_raw:
            try:
                lag_anchor = to_ny(pd.Timestamp(decision_ts_raw))
            except Exception:
                pass
        lag_ok, lag_reason, quote_lag = signal_quote_lag_ok(
            signal_ts=float(lag_anchor.timestamp()),
            quote_ts=quote_ts,
            max_lag_sec=route_max_lag,
        )
        if not lag_ok:
            self._entry_quote_stable[quote_key] = 0
            kind = "ENTRY_WAIT" if lag_reason == "option_quote_before_signal" else "ENTRY_REJECT"
            self._event(
                kind,
                {
                    "symbol": symbol,
                    "reason": lag_reason,
                    "quote_lag_sec": quote_lag,
                    "max_lag_sec": route_max_lag,
                    "lag_anchor_ts": lag_anchor.isoformat(),
                    **self._signal_source_fields(signal),
                },
            )
            if kind == "ENTRY_WAIT":
                self._defer_signal(signal)
            return False
        route_spread_cap = meta0.get("max_spread_pct")
        try:
            route_spread_cap = (
                float(route_spread_cap) if route_spread_cap is not None else None
            )
        except (TypeError, ValueError):
            route_spread_cap = None
        route_min_mid = meta0.get("min_mid")
        try:
            route_min_mid = float(route_min_mid) if route_min_mid is not None else None
        except (TypeError, ValueError):
            route_min_mid = None
        quote_ok, quote_reason, mid = entry_quote_ok(
            bid=float(quote["bid"]),
            ask=float(quote["ask"]),
            prev_mid=prev_mid,
            cfg=self.risk_cfg,
            max_spread_pct=route_spread_cap,
            min_mid=route_min_mid,
        )
        if not quote_ok:
            self._entry_quote_stable[quote_key] = 0
            if mid is not None and quote_reason != "entry_mid_jump":
                self._remember_option_mid(symbol, contract, mid)
            kind = "ENTRY_REJECT" if quote_reason == "entry_mid_jump" else "ENTRY_WAIT"
            self._event(kind, {"symbol": symbol, "reason": quote_reason, "mid": mid})
            if quote_reason != "entry_mid_jump":
                self._defer_signal(signal)
            return False
        prev_stable = int(self._entry_quote_stable.get(quote_key, 0))
        stable, ready, warm_reason = next_entry_quote_stable_ticks(
            prev_stable=prev_stable,
            quote_ok=True,
            prev_mid=prev_mid,
            require_ticks=self.risk_cfg.require_entry_quote_stable_ticks,
            prev_quote_ts=prev_quote_ts,
            quote_ts=quote_ts,
        )
        self._entry_quote_stable[quote_key] = int(stable)
        if (
            math.isfinite(quote_ts)
            and quote_ts > 0
            and (prev_quote_ts is None or quote_ts > float(prev_quote_ts))
        ):
            self._last_entry_quote_ts[quote_key] = quote_ts
        if mid is not None:
            self._remember_option_mid(symbol, contract, mid)
        if not ready:
            self._event(
                "ENTRY_WAIT",
                {
                    "symbol": symbol,
                    "reason": warm_reason,
                    "mid": mid,
                    "stable_ticks": stable,
                    "need_ticks": self.risk_cfg.require_entry_quote_stable_ticks,
                },
            )
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
            symbol,
            signal.sig_ts,
            limit_price,
            regime_scale=r_scale,
            position_frac_override=self._position_frac_override(signal),
        )
        if qty <= 0:
            detail = dict(self._last_size_reject or {})
            self._event(
                "ENTRY_REJECT",
                {
                    "symbol": symbol,
                    "reason": "size_gate",
                    "regime_size_scale": r_scale,
                    **detail,
                    **self._signal_source_fields(signal),
                },
            )
            return False
        mid = quote_mid(float(quote["bid"]), float(quote["ask"])) or float(limit_price)
        ask_size = quote.get("ask_size")
        try:
            ask_size_f = float(ask_size) if ask_size is not None else None
        except (TypeError, ValueError):
            ask_size_f = None
        chunks = plan_entry_chunks(
            qty,
            mid=float(mid),
            ask_size=ask_size_f,
            cfg=self.iceberg_cfg,
        )
        if not chunks:
            self._event("ENTRY_REJECT", {"symbol": symbol, "reason": "iceberg_empty"})
            return False
        if len(chunks) > 1:
            self._event(
                "ICEBERG_PLAN",
                {
                    "symbol": symbol,
                    "contract": contract,
                    "total_qty": qty,
                    "chunks": chunks,
                    "ask_size": ask_size_f,
                    "mid": mid,
                    "fallback_notional": self.iceberg_cfg.fallback_notional,
                },
            )
        started = time.time()
        if self.mode == "shadow":
            # Shadow: walk clips immediately (same quote snapshot; still records per-clip spreads).
            for idx, clip_qty in enumerate(chunks):
                ok = self._submit_entry_chunk(
                    signal=signal,
                    symbol=symbol,
                    contract=contract,
                    con_id=lock.con_id,
                    clip_qty=int(clip_qty),
                    limit_price=float(limit_price),
                    qty_frac=float(qty_frac),
                    quote=quote,
                    mid=float(mid),
                    chunk_idx=idx,
                    chunks_total=len(chunks),
                    total_qty=qty,
                    queue=chunks[idx + 1 :],
                    started_at=started,
                    parent_intent_id="",
                )
                if not ok:
                    break
        else:
            self._submit_entry_chunk(
                signal=signal,
                symbol=symbol,
                contract=contract,
                con_id=lock.con_id,
                clip_qty=int(chunks[0]),
                limit_price=float(limit_price),
                qty_frac=float(qty_frac),
                quote=quote,
                mid=float(mid),
                chunk_idx=0,
                chunks_total=len(chunks),
                total_qty=qty,
                queue=chunks[1:],
                started_at=started,
                parent_intent_id="",
            )
        self.publish_state()
        return True

    def _submit_entry_chunk(
        self,
        *,
        signal: ScannerSignal,
        symbol: str,
        contract: str,
        con_id: int,
        clip_qty: int,
        limit_price: float,
        qty_frac: float,
        quote: dict[str, float],
        mid: float,
        chunk_idx: int,
        chunks_total: int,
        total_qty: int,
        queue: list[int],
        started_at: float,
        parent_intent_id: str,
    ) -> bool:
        intent_id = self._intent_id(
            "BUY",
            symbol,
            contract,
            f"{int(to_ny(signal.sig_ts).timestamp() * 1000)}:ibg{chunk_idx}:{time.time():.3f}",
        )
        if intent_id in self.intents:
            return False
        intent = PendingIntent(
            intent_id=intent_id,
            action="BUY",
            symbol=symbol,
            contract=contract,
            con_id=con_id,
            qty=int(clip_qty),
            limit_price=float(limit_price),
            reason="ENTRY" if chunks_total <= 1 else f"ICEBERG_{chunk_idx + 1}/{chunks_total}",
            created_at=time.time(),
            signal=signal,
            ref_price=float(mid),
            parent_intent_id=parent_intent_id,
            iceberg_queue=encode_chunk_queue(queue),
            iceberg_chunk_idx=int(chunk_idx),
            iceberg_chunks=int(chunks_total),
            iceberg_total_qty=int(total_qty),
            iceberg_qty_frac=float(qty_frac),
            iceberg_started_at=float(started_at),
        )
        self.intents[intent.intent_id] = intent
        self._event("ENTRY_INTENT", _intent_dict(intent))
        if self.mode == "shadow":
            self._apply_open_fill(
                intent, int(clip_qty), float(limit_price), float(qty_frac), quote
            )
        else:
            self._place_broker_order(intent)
        return True

    def _maybe_continue_iceberg(self, intent: PendingIntent) -> None:
        """After a BUY clip fully fills, place the next clip if any remain."""
        if self.mode == "shadow":
            return  # shadow walks all clips synchronously in process_signal
        if intent.action != "BUY":
            return
        if str(intent.status).upper() != "FILLED":
            return
        queue = decode_chunk_queue(intent.iceberg_queue)
        if not queue:
            return
        if intent.iceberg_started_at > 0 and (
            time.time() - float(intent.iceberg_started_at)
            > float(self.iceberg_cfg.max_total_sec)
        ):
            self._event(
                "ICEBERG_STOP",
                {
                    "symbol": intent.symbol,
                    "reason": "max_total_sec",
                    "remaining": queue,
                    "from_intent_id": intent.intent_id,
                },
            )
            return
        if self.day_halted:
            self._event(
                "ICEBERG_STOP",
                {
                    "symbol": intent.symbol,
                    "reason": "day_halted",
                    "remaining": queue,
                    "from_intent_id": intent.intent_id,
                },
            )
            return
        feed_ok, feed_reason = self._entry_data_ok()
        if not feed_ok:
            self._event(
                "ICEBERG_STOP",
                {
                    "symbol": intent.symbol,
                    "reason": feed_reason,
                    "remaining": queue,
                    "from_intent_id": intent.intent_id,
                },
            )
            return
        signal = intent.signal
        if signal is None:
            return
        quote = self._quote(intent.symbol, intent.contract)
        if quote is None:
            self._reset_entry_quote_stable(intent.symbol, intent.contract)
            self._event(
                "ICEBERG_STOP",
                {
                    "symbol": intent.symbol,
                    "reason": "option_stale_or_missing",
                    "remaining": queue,
                    "from_intent_id": intent.intent_id,
                },
            )
            return
        quote_ok, quote_reason, mid = entry_quote_ok(
            bid=float(quote["bid"]),
            ask=float(quote["ask"]),
            prev_mid=self._last_seen_option_mid.get((intent.symbol, intent.contract)),
            cfg=self.risk_cfg,
            max_spread_pct=(
                (getattr(signal, "meta", None) or {}).get("max_spread_pct")
            ),
        )
        if not quote_ok:
            self._entry_quote_stable[(intent.symbol, intent.contract)] = 0
            self._event(
                "ICEBERG_STOP",
                {
                    "symbol": intent.symbol,
                    "reason": quote_reason,
                    "remaining": queue,
                    "from_intent_id": intent.intent_id,
                },
            )
            return
        quote_key = (intent.symbol, intent.contract)
        quote_ts = float(quote.get("ts") or 0.0)
        prev_quote_ts = self._last_entry_quote_ts.get(quote_key)
        stable, ready, warm_reason = next_entry_quote_stable_ticks(
            prev_stable=int(self._entry_quote_stable.get(quote_key, 0)),
            quote_ok=True,
            prev_mid=self._last_seen_option_mid.get(quote_key),
            require_ticks=self.risk_cfg.require_entry_quote_stable_ticks,
            prev_quote_ts=prev_quote_ts,
            quote_ts=quote_ts,
        )
        self._entry_quote_stable[quote_key] = stable
        if not ready:
            self._event(
                "ICEBERG_STOP",
                {
                    "symbol": intent.symbol,
                    "reason": warm_reason,
                    "remaining": queue,
                    "from_intent_id": intent.intent_id,
                },
            )
            return
        self._last_entry_quote_ts[quote_key] = quote_ts
        if mid is not None:
            self._remember_option_mid(intent.symbol, intent.contract, mid)
        mid = mid or quote_mid(float(quote["bid"]), float(quote["ask"])) or float(
            intent.limit_price
        )
        limit_price = limit_price_from_quote(
            quote["bid"], quote["ask"], "BUY", self.fill_model
        )
        next_qty = int(queue[0])
        self._submit_entry_chunk(
            signal=signal,
            symbol=intent.symbol,
            contract=intent.contract,
            con_id=intent.con_id,
            clip_qty=next_qty,
            limit_price=float(limit_price),
            qty_frac=float(intent.iceberg_qty_frac or 0.0),
            quote=quote,
            mid=float(mid),
            chunk_idx=int(intent.iceberg_chunk_idx) + 1,
            chunks_total=int(intent.iceberg_chunks or 1),
            total_qty=int(intent.iceberg_total_qty or next_qty),
            queue=queue[1:],
            started_at=float(intent.iceberg_started_at or time.time()),
            parent_intent_id=intent.intent_id,
        )

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
                        position_frac_override=self._position_frac_override(signal),
                    )
                    self._apply_open_fill(intent, filled, avg_price, qty_frac, quote)
            elif intent.action == "SELL" and intent.symbol in self.positions:
                close_q = self._quote(intent.symbol, intent.contract) or {
                    "bid": avg_price,
                    "ask": avg_price,
                }
                self._apply_close_fill(intent, filled, avg_price, close_q)
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
        if intent.action == "BUY":
            feed_ok, feed_reason = self._entry_data_ok()
            if not feed_ok:
                self._event(
                    "REQUOTE_SKIP",
                    {
                        "intent_id": intent.intent_id,
                        "reason": feed_reason,
                        "attempt": next_attempt,
                    },
                )
                return False
            quote = self._quote(intent.symbol, intent.contract)
            quote_key = (intent.symbol, intent.contract)
            quote_ts = float((quote or {}).get("ts") or 0.0)
            prev_quote_ts = self._last_entry_quote_ts.get(quote_key)
            if (
                quote is None
                or prev_quote_ts is None
                or quote_ts <= float(prev_quote_ts)
            ):
                self._event(
                    "REQUOTE_SKIP",
                    {
                        "intent_id": intent.intent_id,
                        "reason": "option_quote_not_advanced",
                        "attempt": next_attempt,
                    },
                )
                return False
            self._last_entry_quote_ts[quote_key] = quote_ts
        else:
            quote = self._live_option_quote(
                intent.symbol, intent.contract, intent.con_id
            )
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
                position_frac_override=self._position_frac_override(signal),
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
            self._apply_close_fill(intent, qty, price, quote)
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
            entry_stock = self._stock_last_close(intent.symbol) or float(
                getattr(signal, "spot", 0.0) or 0.0
            )
            entry_qqq = self._qqq_last_close() or 0.0
            meta = getattr(signal, "meta", None) or {}
            def _opt_f(key: str) -> float | None:
                raw = meta.get(key)
                if raw is None:
                    return None
                try:
                    return float(raw)
                except (TypeError, ValueError):
                    return None

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
                entry_stock_px=float(entry_stock or 0.0),
                entry_qqq_px=float(entry_qqq or 0.0),
                exit_tp_mult=_opt_f("exit_tp_mult") or _opt_f("tp_mult"),
                exit_sl_mult=_opt_f("exit_sl_mult") or _opt_f("sl_mult"),
                exit_hold_sec=_opt_f("exit_hold_sec"),
                exit_simple=bool(meta.get("exit_simple", False)),
                exit_flatten_before=(
                    str(meta["exit_flatten_before"])
                    if meta.get("exit_flatten_before")
                    else None
                ),
                confirm_abort=(
                    dict(meta["confirm_abort"])
                    if isinstance(meta.get("confirm_abort"), dict)
                    else None
                ),
                profit_protect=(
                    dict(meta["profit_protect"])
                    if isinstance(meta.get("profit_protect"), dict)
                    else None
                ),
                route=str(meta.get("route") or "baseline").strip().lower() or "baseline",
            )
            self._remember_option_mid(intent.symbol, intent.contract, float(mid))
        spread_row = self._record_trade_spread(
            action="OPEN",
            symbol=intent.symbol,
            contract=intent.contract,
            side="BUY",
            fill_px=float(price),
            qty=int(qty),
            bid=float(quote["bid"]),
            ask=float(quote["ask"]),
            reason=str(intent.reason or "ENTRY"),
        )
        self._event(
            "POSITION_OPEN",
            {
                **asdict(self.positions[intent.symbol]),
                **self._signal_source_fields(signal),
                "fill_px": float(price),
                "bid": spread_row["bid"],
                "ask": spread_row["ask"],
                "spread": spread_row["spread"],
                "spread_pct": spread_row["spread_pct"],
                "fill_spread_frac": spread_row["fill_spread_frac"],
                "iceberg_chunk": intent.iceberg_chunk_idx,
                "iceberg_chunks": intent.iceberg_chunks,
            },
        )
        position = self.positions[intent.symbol]
        feed_ok, guard_reason = self._entry_data_ok()
        quote_ok, quote_reason, _ = entry_quote_ok(
            bid=float(quote["bid"]),
            ask=float(quote["ask"]),
            prev_mid=position.last_good_mid,
            cfg=self.risk_cfg,
            max_spread_pct=(
                (getattr(signal, "meta", None) or {}).get("max_spread_pct")
            ),
        )
        if not feed_ok or not quote_ok:
            reason = guard_reason if not feed_ok else quote_reason
            self._event(
                "POST_FILL_GUARD_FLATTEN",
                {"symbol": intent.symbol, "reason": reason},
            )
            self._submit_exit(
                position,
                "RECOVERY_GUARD_FLATTEN",
                float(quote["bid"]),
                quote,
            )
            return
        if str(intent.status).upper() == "FILLED":
            self._maybe_continue_iceberg(intent)

    def _ensure_path_fast_armed(self) -> bool:
        if self._path_fast_armed is not None:
            return bool(self._path_fast_armed)
        pack = self._path_fast_pack
        if not pack.enabled:
            self._path_fast_armed = False
            return False
        stock_by = getattr(self.scanner, "stock_by", None) or {}
        if not stock_by:
            self._path_fast_armed = False
            return False
        wd = self.profile.get("watchdog") if isinstance(self.profile.get("watchdog"), dict) else {}
        prev = wd.get("prevention") if isinstance(wd.get("prevention"), dict) else {}
        try:
            self._path_fast_armed = bool(
                path_fast_pack_day_should_arm(
                    pack,
                    date=str(self.trade_date),
                    stock_by=stock_by,
                    qqq_df=stock_by.get("QQQ"),
                    symbols=list(self.profile.get("symbols") or []),
                    asof=str(pack.asof or prev.get("asof") or wd.get("asof") or "10:30"),
                    washout_breadth_min=int(
                        pack.washout_breadth_min
                        if pack.washout_breadth_min is not None
                        else prev.get("washout_breadth_min", 3)
                        or 3
                    ),
                    wash_drop_min=float(prev.get("wash_drop_min", 0.008) or 0.008),
                    frac_above_min=float(prev.get("frac_above_min", 0.35) or 0.35),
                    frac_above_max=float(prev.get("frac_above_max", 0.70) or 0.70),
                )
            )
        except Exception:
            self._path_fast_armed = False
        return bool(self._path_fast_armed)

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
        lac = ladder_active_from_trade(self.trade_cfg)
        use_ladder = bool(lac.enabled)
        use_extend = ("hold_extend" in blob or "extend_hold" in blob) and not use_ladder
        use_trail = ("mtm_trail" in blob or blob in {"trail"}) and not use_ladder
        trail_activate = float(self.trade_cfg.get("trail_activate", 0.20) or 0.20)
        trail_dd = float(self.trade_cfg.get("trail_dd", 0.15) or 0.15)
        srev_live = stock_rev_exit_from_trade(self.trade_cfg)
        if self._ensure_path_fast_armed():
            ov = apply_path_fast_pack_overrides(
                hold_minutes=int(hold // 60),
                trail_activate=trail_activate,
                trail_dd=trail_dd,
                stock_rev=srev_live,
                pack=self._path_fast_pack,
            )
            hold = int(ov["hold_minutes"]) * 60
            trail_activate = float(ov["trail_activate"])
            trail_dd = float(ov["trail_dd"])
            srev_live = ov["stock_rev_exit"]
            use_extend = False
            ext_hold = hold
        use_floor = "mtm_floor" in blob or "mtm_defend" in blob
        use_mf_flip = "mf_flip" in blob or "mf_reversal" in blob or (use_ladder and lac.mf_flip)
        use_streak = "streak_break" in blob
        min_hold = self.trade_cfg.get("exit_min_hold_minutes")
        if min_hold is None and ("mf_reversal" in blob or use_floor):
            min_hold = 10.0
        if min_hold is not None:
            grace = max(grace, int(float(min_hold) * 60))
        if use_ladder and lac.mf_flip:
            grace = int(lac.mf_grace_seconds)
        floor = float(self.trade_cfg.get("mtm_floor_ret", 0.0))
        for symbol, position in list(self.positions.items()):
            if position.status != "OPEN":
                continue
            quote = self._quote(symbol, position.contract)
            if quote is None:
                position.gap_hold_count += 1
                if position.gap_hold_count >= self.risk_cfg.max_gap_hold_ticks:
                    fallback = self._position_quote_fallback(position)
                    if fallback is not None:
                        self._note_gap_event(now=asof_ts)
                        self._event(
                            "OPTION_FEED_GAP_FLATTEN",
                            {
                                "symbol": symbol,
                                "missing_ticks": position.gap_hold_count,
                            },
                        )
                        self._submit_exit(
                            position,
                            "GAP_FLATTEN",
                            float(fallback["bid"]),
                            fallback,
                        )
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
            sell_price = self.fill.sell(float(quote["bid"]), float(quote["ask"]))
            reason = ""
            held = asof_ts - position.entry_ts
            mtm_ret = sell_price / position.entry_price - 1.0 if position.entry_price > 0 else float("nan")
            if math.isfinite(mtm_ret) and mtm_ret > float(position.peak_mfe):
                position.peak_mfe = float(mtm_ret)
            if position.entry_stock_px <= 0:
                stock_px = self._stock_last_close(symbol)
                if stock_px is not None:
                    position.entry_stock_px = float(stock_px)
            if position.entry_qqq_px <= 0:
                qqq_px = self._qqq_last_close()
                if qqq_px is not None:
                    position.entry_qqq_px = float(qqq_px)
            # Per-position rails for satellite sleeves (qqq_open_cont / am_pulse).
            pos_tp = float(position.exit_tp_mult) if position.exit_tp_mult else tp
            pos_sl = float(position.exit_sl_mult) if position.exit_sl_mult else sl
            pos_hold = (
                float(position.exit_hold_sec)
                if position.exit_hold_sec is not None and float(position.exit_hold_sec) > 0
                else float(hold)
            )
            simple = bool(position.exit_simple)
            toxic_reason = None if simple else self._trade_toxic_reason(
                position, mtm_ret=mtm_ret, held=held, asof_ts=asof_ts
            )
            shock_reason = None if simple else self._hold_shock_reason(
                position, mtm_ret=mtm_ret, held=held
            )
            flatten_hit = False
            if simple and position.exit_flatten_before:
                try:
                    # Sleeve flatten is a wall-clock hard stop (e.g. 10:45 before
                    # CORE). Use max(frame, wall) so lagged fused frames cannot
                    # spill holdings past the designed window.
                    flat_clock = max(float(asof_ts), float(time.time()))
                    ny = pd.Timestamp(flat_clock, unit="s", tz="UTC").tz_convert(
                        "America/New_York"
                    )
                    parts = str(position.exit_flatten_before).split(":")
                    flat_m = int(parts[0]) * 60 + int(parts[1])
                    flatten_hit = (ny.hour * 60 + ny.minute) >= flat_m
                except Exception:
                    flatten_hit = False
            if gap_status == "gap_force":
                reason = "GAP_FLATTEN"
                self._note_gap_event(now=float(asof_ts))
            elif flatten_hit:
                reason = "FLATTEN_BEFORE_CORE"
            elif sell_price >= position.entry_price * pos_tp and (
                simple or (not use_ladder or lac.keep_outer_rails)
            ):
                reason = "TP"
            elif (
                simple
                and math.isfinite(mtm_ret)
                and profit_protect_on_tick(
                    cfg=profit_protect_from_raw(position.profit_protect),
                    peak_mfe=float(position.peak_mfe),
                    opt_mtm=float(mtm_ret),
                )
            ):
                reason = "PROFIT_PROTECT"
            elif toxic_reason:
                reason = toxic_reason
            elif shock_reason:
                reason = shock_reason
            elif simple and sell_price <= position.entry_price * pos_sl:
                reason = "SL"
            elif simple:
                if position.confirm_abort and math.isfinite(mtm_ret):
                    ca_cfg = confirm_abort_from_raw(position.confirm_abort)
                    if confirm_abort_applies(
                        ca_cfg,
                        float(position.signal_ts or position.entry_ts),
                        direction=str(position.direction or ""),
                    ):
                        st = ConfirmAbortState(
                            confirmed=bool(position.confirm_abort_confirmed),
                            done=bool(position.confirm_abort_done),
                        )
                        do_abort, ca_reason, st = confirm_abort_on_tick(
                            st,
                            cfg=ca_cfg,
                            held_seconds=float(held),
                            opt_mtm=float(mtm_ret),
                        )
                        position.confirm_abort_confirmed = bool(st.confirmed)
                        position.confirm_abort_done = bool(st.done)
                        if do_abort:
                            reason = (
                                "CONFIRM_ABORT"
                                if ca_reason == "confirm_abort"
                                else "EARLY_ABORT"
                            )
                if not reason and held >= pos_hold:
                    reason = "MAX_HOLD"
            elif not simple:
                # Wave confirm: arm then revocable abort (before STOCK_REV / clock grind).
                wcfg = wave_abort_from_trade(self.trade_cfg)
                if (
                    wcfg.enabled
                    and not position.wave_done
                    and position.entry_stock_px > 0
                    and math.isfinite(mtm_ret)
                ):
                    cur_stock = self._stock_last_close(symbol)
                    if cur_stock is not None and float(cur_stock) > 0:
                        s_ret = float(cur_stock) / float(position.entry_stock_px) - 1.0
                        signed = s_ret if position.direction == "UP" else -s_ret
                        st = WaveAbortState(armed=bool(position.wave_armed), done=bool(position.wave_done))
                        do_abort, _, st = wave_abort_on_tick(
                            st,
                            cfg=wcfg,
                            held_seconds=float(held),
                            stock_signed=float(signed),
                            opt_mtm=float(mtm_ret),
                        )
                        position.wave_armed = bool(st.armed)
                        position.wave_done = bool(st.done)
                        if do_abort:
                            reason = "WAVE_ABORT"
                # Path risk: underlying reversed — cut before grinding to hard SL / clock.
                srev = srev_live
                if (
                    not reason
                    and srev.enabled
                    and stock_rev_applies_to_route(srev, getattr(position, "route", None))
                    and held >= float(srev.min_hold_minutes) * 60.0
                    and math.isfinite(mtm_ret)
                    and mtm_ret <= float(srev.opt_mtm_max)
                    and position.entry_stock_px > 0
                ):
                    cur_stock = self._stock_last_close(symbol)
                    if cur_stock is not None and float(cur_stock) > 0:
                        s_ret = float(cur_stock) / float(position.entry_stock_px) - 1.0
                        signed = s_ret if position.direction == "UP" else -s_ret
                        if float(signed) <= float(srev.stock_max):
                            reason = "STOCK_REV"
            if reason:
                pass
            elif simple:
                pass  # satellite rails already evaluated above
            elif (
                use_trail
                and math.isfinite(mtm_ret)
                and float(position.peak_mfe) >= float(trail_activate)
                and mtm_ret <= float(position.peak_mfe) - float(trail_dd)
            ):
                reason = "TRAIL"
            elif sell_price <= position.entry_price * pos_sl and (
                not use_ladder or lac.keep_outer_rails
            ):
                reason = "SL"
            elif use_ladder and math.isfinite(mtm_ret):
                if mtm_ret > position.ladder_peak_ret:
                    position.ladder_peak_ret = float(mtm_ret)
                    position.ladder_peak_ts = float(asof_ts)
                for rail in sorted(lac.sl_rails, key=lambda r: float(r.ret), reverse=True):
                    if mtm_ret <= float(rail.ret):
                        reason = f"SL_LADDER{int(round(abs(float(rail.ret)) * 100))}"
                        break
                if not reason:
                    for rail in lac.tp_rails:
                        if mtm_ret < float(rail.ret):
                            continue
                        if rail.action == "exit":
                            reason = f"TP_LADDER{int(round(float(rail.ret) * 100))}"
                            break
                        position.ladder_trail_armed = True
                        position.ladder_trail_dd = float(rail.trail_dd)
                if (
                    not reason
                    and position.ladder_trail_armed
                    and mtm_ret <= float(position.ladder_peak_ret) - float(position.ladder_trail_dd)
                ):
                    reason = "TRAIL_LADDER"
                if (
                    not reason
                    and float(position.ladder_peak_ret) >= float(lac.stall_min_peak)
                    and position.ladder_peak_ts > 0
                    and (float(asof_ts) - float(position.ladder_peak_ts)) >= float(lac.stall_seconds)
                ):
                    reason = "PROFIT_STALL"
                if not reason and held >= grace and use_mf_flip:
                    state = self.scanner.states.get(symbol)
                    mf10 = float(getattr(state, "mf10", float("nan")))
                    if math.isfinite(mf10):
                        if position.direction == "UP" and mf10 < 0:
                            reason = "MF_FLIP"
                        elif position.direction == "DN" and mf10 > 0:
                            reason = "MF_FLIP"
                if not reason and held >= float(lac.max_hold_seconds):
                    reason = "SEC_MAX"
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
                    giveback_ok = True
                    max_gb = self.trade_cfg.get("hold_extend_max_giveback")
                    if max_gb is not None and math.isfinite(mtm_ret):
                        min_peak = float(
                            self.trade_cfg.get("hold_extend_giveback_min_peak") or 0.0
                        )
                        peak = float(position.peak_mfe)
                        if peak >= min_peak:
                            giveback_ok = (peak - float(mtm_ret)) < float(max_gb)
                    if mtm_ok and mf_ok and giveback_ok and ext_hold > hold:
                        position.hold_extended = True
                    else:
                        reason = f"T+{hold // 60}"
            elif (not use_ladder) and held >= hold:
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
        active_sell = next(
            (
                item
                for item in self.intents.values()
                if item.action == "SELL"
                and item.symbol == position.symbol
                and item.contract == position.contract
                and item.status
                in {"PENDING", "SUBMITTED", "PARTIAL", "PARTIALLY_FILLED"}
            ),
            None,
        )
        if active_sell is not None:
            self._event(
                "EXIT_DEDUP",
                {
                    "symbol": position.symbol,
                    "reason": reason,
                    "active_intent_id": active_sell.intent_id,
                    "active_reason": active_sell.reason,
                },
            )
            return
        if position.status != "OPEN":
            return
        if self.mode != "shadow" and not self.ib.isConnected():
            position.status = "EXIT_PENDING"
            self._pending_force_exits[position.symbol] = str(reason)
            self._event(
                "EXIT_QUEUED_DISCONNECTED",
                {"symbol": position.symbol, "reason": reason},
            )
            self.publish_state()
            return
        exit_qty = int(position.qty)
        if self.mode != "shadow":
            broker_qty = 0
            try:
                for row in self.ib.positions() or []:
                    if (
                        self.connector.config.account
                        and str(getattr(row, "account", "") or "")
                        != self.connector.config.account
                    ):
                        continue
                    contract = getattr(row, "contract", None)
                    local = str(
                        getattr(contract, "localSymbol", "") or ""
                    ).strip()
                    if local == position.contract:
                        broker_qty += int(
                            round(float(getattr(row, "position", 0.0) or 0.0))
                        )
            except Exception as exc:
                if reason in FORCE_EXIT_REASONS:
                    position.status = "EXIT_PENDING"
                    self._pending_force_exits[position.symbol] = str(reason)
                self._event(
                    "EXIT_BLOCKED",
                    {
                        "symbol": position.symbol,
                        "reason": reason,
                        "gate": "broker_position_unavailable",
                        "error": str(exc),
                    },
                )
                return
            if broker_qty <= 0:
                if reason in FORCE_EXIT_REASONS:
                    position.status = "EXIT_PENDING"
                    self._pending_force_exits[position.symbol] = str(reason)
                self._event(
                    "EXIT_BLOCKED",
                    {
                        "symbol": position.symbol,
                        "reason": reason,
                        "gate": "broker_position_not_long",
                        "broker_qty": broker_qty,
                        "internal_qty": position.qty,
                    },
                )
                return
            exit_qty = min(exit_qty, broker_qty)
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
            qty=exit_qty,
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
            self._apply_close_fill(intent, exit_qty, limit_price, quote)
        else:
            self._place_broker_order(intent)
        self.publish_state()

    def _apply_close_fill(
        self,
        intent: PendingIntent,
        qty: int,
        price: float,
        quote: dict[str, float] | None = None,
    ) -> None:
        position = self.positions.get(intent.symbol)
        if position is None:
            return
        if quote is None:
            quote = self._quote(intent.symbol, position.contract)
        if quote is None and position.last_bid > 0 and position.last_ask >= position.last_bid:
            quote = {"bid": float(position.last_bid), "ask": float(position.last_ask)}
        if quote is None:
            quote = {"bid": float(price), "ask": float(price)}
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
        spread_row = self._record_trade_spread(
            action="CLOSE" if qty >= position.qty else "PARTIAL_CLOSE",
            symbol=intent.symbol,
            contract=position.contract,
            side="SELL",
            fill_px=float(price),
            qty=filled_qty,
            bid=float(quote["bid"]),
            ask=float(quote["ask"]),
            reason=intent.reason,
            ret=None
            if qty < position.qty
            else (float(intent.avg_fill_price) / position.entry_price - 1.0),
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
                    "bid": spread_row["bid"],
                    "ask": spread_row["ask"],
                    "spread": spread_row["spread"],
                    "spread_pct": spread_row["spread_pct"],
                    "fill_spread_frac": spread_row["fill_spread_frac"],
                },
            )
            circuit = self.trade_cfg.get("day_circuit")
            if circuit is not None and day_ret <= float(circuit):
                self._trip_day_circuit(day_ret=day_ret)
            return
        ret = float(intent.avg_fill_price) / position.entry_price - 1.0
        exit_ts = pd.Timestamp.now(tz="America/New_York")
        reason_key = str(intent.reason or "UNKNOWN")
        self.exit_reason_counts[reason_key] = int(
            self.exit_reason_counts.get(reason_key, 0)
        ) + 1
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
                "bid": spread_row["bid"],
                "ask": spread_row["ask"],
                "spread": spread_row["spread"],
                "spread_pct": spread_row["spread_pct"],
                "fill_spread_frac": spread_row["fill_spread_frac"],
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
        if not self.reconcile_ok:
            self.day_halted = True
        self.last_reconcile = {
            "ok": self.reconcile_ok,
            "broker": broker,
            "internal": internal,
            "ts": time.time(),
        }
        self._event(
            "RECONCILE",
            {"ok": self.reconcile_ok, "broker": broker, "internal": internal},
        )
        if self._pending_force_exits and self.ib.isConnected():
            self._flush_pending_force_exits()
        self.publish_state()
        return self.reconcile_ok

    async def reconcile_loop(self, interval_sec: float = 10.0) -> None:
        while True:
            try:
                await self.reconcile()
                # Heartbeat even on a flat book so Dash OMS age stays live.
                if self.mode == "shadow":
                    self.last_reconcile = {
                        "ok": True,
                        "broker": {},
                        "internal": {},
                        "ts": time.time(),
                    }
                self.publish_state()
            except Exception as exc:
                self.reconcile_ok = False
                self.last_reconcile = {
                    "ok": False,
                    "broker": (self.last_reconcile or {}).get("broker") or {},
                    "internal": (self.last_reconcile or {}).get("internal") or {},
                    "ts": time.time(),
                    "error": str(exc),
                }
                self._event("RECONCILE_ERROR", {"error": str(exc)})
                try:
                    self.publish_state()
                except Exception:
                    pass
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
            if position.status not in {"OPEN", "EXIT_PENDING"}:
                continue
            quote = self._position_quote_fallback(position)
            if quote is None:
                if reason in FORCE_EXIT_REASONS:
                    position.status = "EXIT_PENDING"
                    self._pending_force_exits[position.symbol] = str(reason)
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

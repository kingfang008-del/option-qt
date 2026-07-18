"""Mag7 multi-symbol Rule-A scanner — emits TopK signals for OMS / shadow.

Does NOT use QQQ TFT/FCS as the primary signal path. Optional regime filter
can be added later (QQQ state only gates Mag7 entries).
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from maga7.common.bar_agg import MultiSymbolMinuteAgg
from maga7.common.config import load_profile
from maga7.common.entry_contract import ContractBooks, resolve_entry_contract
from maga7.common.reentry import resolve_only_win_reenter
from maga7.common.replay import to_ny
from maga7.common.signals import StreamSignalState, count_peer_align

logger = logging.getLogger("maga7.live.scanner")


@dataclass
class ScannerSignal:
    date: str
    symbol: str
    direction: str  # UP / DN
    sig_ts: pd.Timestamp
    spot: float
    rank: int
    bucket_id: int
    contract: str | None
    moneyness: str
    meta: dict[str, Any] = field(default_factory=dict)

    def to_orch_payload(self) -> dict[str, Any]:
        """Audit-friendly payload (CALL/PUT). Not directly executable by QQQ OMS."""
        side = "CALL" if self.direction == "UP" else "PUT"
        return {
            "ts": self.sig_ts.isoformat(),
            "symbol": self.symbol,
            "side": side,
            "dir": self.direction,
            "rank": self.rank,
            "spot": self.spot,
            "contract": self.contract,
            "bucket_id": self.bucket_id,
            "moneyness": self.moneyness,
            "source": "maga7_mf10_top2",
            "meta": {
                **self.meta,
                "strategy": "maga7_mf10_top2_v1",
                "fill_frac": self.meta.get("fill_frac", 0.8),
                "watchdog_state": self.meta.get("watchdog_state"),
                "watchdog_reason": self.meta.get("watchdog_reason"),
                "route": self.meta.get("route"),
                "event_source": self.meta.get("event_source", "baseline"),
            },
        }

    def to_oms_exec_payload(
        self,
        *,
        action: str = "BUY",
        bid: float | None = None,
        ask: float | None = None,
        limit_px: float | None = None,
        qty: int = 1,
        ts: pd.Timestamp | None = None,
    ) -> dict[str, Any]:
        """Map to QQQ-like orch_trade_signals BUY/SELL shape (for optional Redis publish)."""
        opt_side = "CALL" if self.direction == "UP" else "PUT"
        dir_i = 1 if self.direction == "UP" else -1
        t = ts if ts is not None else self.sig_ts
        unix = float(pd.Timestamp(t).timestamp())
        return {
            "source": "maga7_mf10_top2",
            "action": action.upper(),
            "ts": unix,
            "symbol": self.symbol,
            "stock_price": self.spot,
            "sig": {
                "action": action.upper(),
                "dir": dir_i,
                "tag": f"{opt_side}_{self.moneyness}",
                "price": limit_px,
                "bid": bid,
                "ask": ask,
                "reason": f"maga7_rank{self.rank}",
                "meta": {
                    "contract_id": self.contract,
                    "requested_qty": int(qty),
                    "bucket_id": self.bucket_id,
                    "rank": self.rank,
                    "fill_frac": self.meta.get("fill_frac", 0.8),
                    "tp_mult": self.meta.get("tp_mult"),
                    "sl_mult": self.meta.get("sl_mult"),
                    "hold_minutes": self.meta.get("hold_minutes"),
                    "contract_source": self.meta.get("contract_source"),
                    "sig_dte": self.meta.get("sig_dte"),
                    "strategy": "maga7_mf10_top2_v1",
                    "watchdog_state": self.meta.get("watchdog_state"),
                    "watchdog_reason": self.meta.get("watchdog_reason"),
                    "route": self.meta.get("route"),
                    "event_source": self.meta.get("event_source", "baseline"),
                },
            },
        }


@dataclass
class Mag7Scanner:
    """Causal multi-symbol scanner.

    Decision clock = RTH **1m** bars (Rule-A). Prefer feeding **1s** ticks via
    ``on_stock_second`` (aggregates to 1m); ``on_stock_bar`` remains for parity
    with historical 1m parquet.
    """

    profile: dict[str, Any]
    on_signal: Callable[[ScannerSignal], None] | None = None
    is_symbol_active: Callable[[str], bool] | None = None
    states: dict[str, StreamSignalState] = field(default_factory=dict)
    day_fires: list[ScannerSignal] = field(default_factory=list)
    current_date: str | None = None
    books: ContractBooks | None = None
    signals: list[ScannerSignal] = field(default_factory=list)
    minute_agg: MultiSymbolMinuteAgg | None = None
    regime_gate: Any = None
    watchdog: Any = None
    _watchdog_snap: dict[str, Any] = field(default_factory=dict)
    _watchdog_date: str | None = None
    _watchdog_state: str = "off"
    _watchdog_reason: str = "off"
    _watchdog_route: str = "baseline"
    emit_all: bool = False
    n_done: dict[str, int] = field(default_factory=dict)
    last_exit: dict[str, pd.Timestamp | None] = field(default_factory=dict)
    last_win: dict[str, bool] = field(default_factory=dict)
    n_peer_block: int = 0
    n_regime_block: int = 0
    n_event_block: int = 0
    event_blackout: set = field(default_factory=set)
    event_blackout_meta: dict[str, Any] = field(default_factory=dict)
    # Optional feature frames for peer_align parity with offline/stream.
    # When set (stock-1s replay), peer uses count_peer_align(asof=feature_ts).
    # When None (pure live), peer uses per-symbol live StreamSignalState mf.
    stock_by: dict[str, Any] | None = None

    @classmethod
    def from_profile(cls, profile: dict[str, Any] | None = None, **kwargs) -> "Mag7Scanner":
        cfg = profile or load_profile()
        books = ContractBooks.from_profile(cfg)
        scheme = str(kwargs.pop("scheme", "single"))
        emit_all = scheme.startswith("m5")
        states = {s: StreamSignalState(s, cfg["signal"], emit_all=emit_all) for s in cfg["symbols"]}
        agg = MultiSymbolMinuteAgg(cfg["symbols"], rth_only=True)
        regime_gate = None
        try:
            from maga7.common.regime import Mag7RegimeGate
            from maga7.common.replay import month_list

            start = cfg["date_range"]["start"]
            end = cfg["date_range"]["end"]
            regime_gate = Mag7RegimeGate.from_profile(cfg, months=month_list(start, end))
        except Exception:
            regime_gate = None
        watchdog = None
        watchdog_snap: dict[str, Any] = {}
        try:
            from maga7.common.watchdog import RegimeWatchdog, snapshot_regime

            watchdog = RegimeWatchdog.from_profile(cfg)
            if watchdog is not None and regime_gate is not None:
                watchdog_snap = snapshot_regime(regime_gate.cfg)
        except Exception:
            watchdog = None
            watchdog_snap = {}
        return cls(
            profile=cfg,
            states=states,
            books=books,
            minute_agg=agg,
            regime_gate=regime_gate,
            watchdog=watchdog,
            _watchdog_snap=watchdog_snap,
            emit_all=emit_all,
            **kwargs,
        )

    def _topk(self) -> int:
        return int(self.profile["signal"].get("top_k", 2))

    def _peer_align_n(self, direction: str, *, date: str, feature_ts: pd.Timestamp) -> int:
        """Count peers aligned with ``direction`` (offline-compatible when stock_by set)."""
        import math

        sig_cfg = self.profile.get("signal") or {}
        peer_min = sig_cfg.get("peer_align_min")
        if peer_min is None or int(peer_min) <= 0:
            return 0
        peers = list(sig_cfg.get("peer_symbols") or self.profile.get("symbols") or [])
        mode = str(sig_cfg.get("peer_align_mode", "mf10")).strip().lower()
        streak_min = int(sig_cfg.get("streak_min", 8))
        if self.stock_by:
            return count_peer_align(
                self.stock_by,
                date=date,
                asof_ts=feature_ts,
                direction=direction,
                peer_symbols=peers,
                mode=mode,
                streak_min=streak_min,
            )
        n = 0
        for sym in peers:
            st = self.states.get(sym)
            if st is None or st.date != date:
                continue
            mf = float(st.mf10)
            if mode == "streak":
                ok = (direction == "UP" and int(st.streak_up) >= streak_min) or (
                    direction == "DN" and int(st.streak_dn) >= streak_min
                )
            else:
                ok = (direction == "UP" and math.isfinite(mf) and mf > 0) or (
                    direction == "DN" and math.isfinite(mf) and mf < 0
                )
            if ok:
                n += 1
        return n

    def _roll_day(self, date: str) -> None:
        if self.current_date == date:
            return
        self.current_date = date
        self.day_fires = []
        self.n_done = {s: 0 for s in self.profile["symbols"]}
        self.last_exit = {s: None for s in self.profile["symbols"]}
        self.last_win = {s: True for s in self.profile["symbols"]}
        self._refresh_watchdog(date)

    def _refresh_watchdog(self, date: str) -> None:
        """Day-scoped Watchdog evaluate + regime overlay (needs stock_by for rules)."""
        if self.watchdog is None:
            self._watchdog_date = str(date)
            self._watchdog_state = "off"
            self._watchdog_reason = "off"
            self._watchdog_route = "baseline"
            return
        if self._watchdog_date == str(date):
            return
        self._watchdog_date = str(date)
        if not self.stock_by:
            self._watchdog_state = "normal"
            self._watchdog_reason = "no_stock_by"
            self._watchdog_route = "baseline"
            logger.info("WATCHDOG %s skip evaluate (no stock_by)", date)
            return
        try:
            symbols = list(self.profile.get("symbols") or [])
            qqq = self.stock_by.get("QQQ")
            dec = self.watchdog.begin_day(
                str(date),
                stock_by=self.stock_by,
                qqq_df=qqq,
                symbols=symbols,
            )
            if self.regime_gate is not None and self._watchdog_snap is not None:
                self.watchdog.apply_to_regime(self.regime_gate.cfg, self._watchdog_snap)
            self._watchdog_state = dec.state.value
            self._watchdog_reason = dec.reason
            self._watchdog_route = dec.overlay.route_tag or "baseline"
            logger.info(
                "WATCHDOG %s state=%s reason=%s route=%s hunt_armed=%s",
                date,
                self._watchdog_state,
                self._watchdog_reason,
                self._watchdog_route,
                bool(self.watchdog.hunt_armed),
            )
        except Exception as exc:
            self._watchdog_state = "normal"
            self._watchdog_reason = f"error:{type(exc).__name__}"
            self._watchdog_route = "baseline"
            logger.warning("WATCHDOG %s evaluate failed: %s", date, exc)

    def set_event_blackout(
        self, blackout: set[str] | None, meta: dict[str, Any] | None = None
    ) -> None:
        self.event_blackout = set(blackout or ())
        self.event_blackout_meta = dict(meta or {})

    def is_event_blackout(self, date: str | None = None) -> bool:
        d = date or self.current_date
        return bool(d) and str(d) in self.event_blackout

    def on_stock_second(self, symbol: str, tick: dict[str, Any]) -> ScannerSignal | None:
        """Ingest 1s (or trade) print → maybe complete a 1m bar → Rule-A."""
        if self.minute_agg is None:
            self.minute_agg = MultiSymbolMinuteAgg(self.profile["symbols"], rth_only=True)
        bar = self.minute_agg.on_second(symbol, tick)
        if bar is None:
            return None
        bar = {
            **bar,
            "bar_source": "1s_agg",
            "available_ts": to_ny(tick["timestamp"]),
        }
        return self.on_stock_bar(symbol, bar)

    def flush_seconds(self) -> list[ScannerSignal]:
        """Flush open minute bars (e.g. end of day / stream)."""
        if self.minute_agg is None:
            return []
        out: list[ScannerSignal] = []
        for bar in self.minute_agg.flush_all():
            bar = {**bar, "bar_source": "1s_agg"}
            sig = self.on_stock_bar(bar["symbol"], bar)
            if sig is not None:
                out.append(sig)
        return out

    def on_stock_bar(self, symbol: str, bar: dict[str, Any]) -> ScannerSignal | None:
        feature_ts = to_ny(bar["timestamp"])
        delay = int(
            (self.profile.get("trade") or {}).get(
                "bar_availability_delay_seconds",
                0,
            )
            or 0
        )
        ts = to_ny(
            bar.get("available_ts")
            or (feature_ts + pd.Timedelta(seconds=delay))
        )
        date = feature_ts.strftime("%Y-%m-%d")
        self._roll_day(date)
        st = self.states.get(symbol)
        if st is None:
            return None
        fire = st.on_bar(bar)
        if fire is None:
            return None
        if self.is_event_blackout(date):
            self.n_event_block += 1
            logger.info("EVENT_BLACKOUT %s skip %s %s", date, symbol, fire.get("dir"))
            return None

        trade = self.profile.get("trade") or {}
        use_reentry = self.emit_all
        max_n = int(trade.get("max_entries_per_symbol", 5)) if use_reentry else 1
        cooldown = int(trade.get("cooldown_minutes", 5))
        only_win = resolve_only_win_reenter(trade)

        already = any(f.symbol == symbol for f in self.day_fires)
        if already and not use_reentry:
            return None
        if not already and len(self.day_fires) >= self._topk() and not use_reentry:
            return None
        # TopK admission: first fires of new symbols only until K filled
        # (match stream/offline: reserve slot BEFORE regime — blocked names still fill K)
        if not already and len({f.symbol for f in self.day_fires}) >= self._topk():
            return None
        if use_reentry:
            if self.is_symbol_active is not None and self.is_symbol_active(symbol):
                return None
            if self.n_done.get(symbol, 0) >= max_n:
                return None
            if self.last_exit.get(symbol) is not None and ts < self.last_exit[symbol] + pd.Timedelta(minutes=cooldown):
                return None
            if only_win and self.n_done.get(symbol, 0) > 0 and not self.last_win.get(symbol, True):
                return None

        money = str(trade.get("moneyness", "ATM"))
        direction = fire["dir"]
        spot = float(fire["spot"])
        books = self.books or ContractBooks.from_profile(self.profile)
        pick = resolve_entry_contract(
            books,
            symbol=symbol,
            date=date,
            direction=direction,
            moneyness=money,
            sig_ts=ts,
            spot=spot,
        )
        fill = self.profile.get("fill") or {}
        rank = len({f.symbol for f in self.day_fires}) + (0 if already else 1)
        sig = ScannerSignal(
            date=date,
            symbol=symbol,
            direction=direction,
            sig_ts=ts,
            spot=spot,
            rank=rank,
            bucket_id=pick.bucket_id,
            contract=pick.ticker,
            moneyness=money,
            meta={
                "fill_frac": float(fill.get("entry_frac", 0.8)),
                "tp_mult": float(trade.get("tp_mult", 1.6)),
                "sl_mult": float(trade.get("sl_mult", 0.4)),
                "hold_minutes": int(trade.get("hold_minutes", 30)),
                "bar_source": bar.get("bar_source", "1m"),
                "contract_source": pick.source,
                "sig_dte": pick.dte,
                "sig_strike": pick.strike,
                "contract_mode": books.mode,
                "feature_ts": feature_ts.isoformat(),
                "decision_ts": ts.isoformat(),
                "bar_availability_delay_seconds": delay,
                "watchdog_state": self._watchdog_state,
                "watchdog_reason": self._watchdog_reason,
                "route": self._watchdog_route,
                "event_source": "baseline",
            },
        )
        if not already:
            self.day_fires.append(sig)

        if self.regime_gate is not None:
            dec = self.regime_gate.check(direction, feature_ts)
            sig.meta["regime_reason"] = getattr(dec, "reason", None)
            sig.meta["regime_size_scale"] = float(getattr(dec, "size_scale", 1.0) or 1.0)
            if not dec.allow:
                self.n_regime_block += 1
                return None

        sig_cfg = self.profile.get("signal") or {}
        peer_min = sig_cfg.get("peer_align_min")
        if peer_min is not None and int(peer_min) > 0:
            peer_n = self._peer_align_n(direction, date=date, feature_ts=feature_ts)
            sig.meta["peer_align_n"] = peer_n
            sig.meta["peer_align_min"] = int(peer_min)
            if peer_n < int(peer_min):
                self.n_peer_block += 1
                logger.info(
                    "PEER_BLOCK %s %s %s peer=%d<%d",
                    date,
                    symbol,
                    direction,
                    peer_n,
                    int(peer_min),
                )
                return None

        self.signals.append(sig)
        # m5/emit_all: n_done + last_exit/win only after OMS record_fill (only_win sequencing).
        # single TopK: count emit as taken (no re-entry path).
        if not use_reentry:
            self.n_done[symbol] = self.n_done.get(symbol, 0) + 1
        logger.info(
            "TOPK signal %s %s %s rank=%d contract=%s src=%s dte=%s wd=%s route=%s r_scale=%s",
            date,
            symbol,
            direction,
            sig.rank,
            pick.ticker,
            pick.source,
            pick.dte,
            self._watchdog_state,
            self._watchdog_route,
            sig.meta.get("regime_size_scale", 1.0),
        )
        if self.on_signal:
            self.on_signal(sig)
        return sig

    def record_fill(
        self,
        symbol: str,
        *,
        exit_ts: pd.Timestamp,
        won: bool,
    ) -> None:
        """OMS callback after a filled round-trip (m5 only_win / cooldown)."""
        self.n_done[symbol] = self.n_done.get(symbol, 0) + 1
        self.last_exit[symbol] = to_ny(exit_ts)
        self.last_win[symbol] = bool(won)


def write_signal_audit(signals: list[ScannerSignal], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [s.to_orch_payload() for s in signals]
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")
    flat = [
        {
            "ts": r["ts"],
            "symbol": r["symbol"],
            "dir": r["dir"],
            "side": r["side"],
            "rank": r["rank"],
            "spot": r["spot"],
            "contract": r["contract"],
            "bucket_id": r["bucket_id"],
            "moneyness": r["moneyness"],
            "source": r["source"],
            "contract_source": (r.get("meta") or {}).get("contract_source"),
            "sig_dte": (r.get("meta") or {}).get("sig_dte"),
        }
        for r in rows
    ]
    pd.DataFrame(flat).to_csv(path.with_suffix(".csv"), index=False)

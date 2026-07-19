"""Mag7 multi-symbol Rule-A scanner — emits TopK + Hunt signals for OMS / shadow.

Does NOT use QQQ TFT/FCS as the primary signal path. Optional regime filter
can be added later (QQQ state only gates Mag7 entries).

Hunt: mirrors ``stream_engine`` — ``begin_day`` arms candidates, then
``_drain_hunts`` emits at feature_ts + bar_delay (does not consume TopK slots).
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
    _day_halt: bool = False
    # Live: re-eval Watchdog as morning 1m bars accumulate until hunt deadline.
    _watchdog_closed: bool = False
    _watchdog_last_eval_tod: str | None = None
    ref_agg: MultiSymbolMinuteAgg | None = None
    emit_all: bool = False
    n_done: dict[str, int] = field(default_factory=dict)
    last_exit: dict[str, pd.Timestamp | None] = field(default_factory=dict)
    last_win: dict[str, bool] = field(default_factory=dict)
    n_peer_block: int = 0
    n_regime_block: int = 0
    n_event_block: int = 0
    n_hunt_signals: int = 0
    n_hunt_emitted: int = 0
    n_hunt_budget_skip: int = 0
    n_hunt_mutex_skip: int = 0
    n_halt_skip: int = 0
    event_blackout: set = field(default_factory=set)
    event_blackout_meta: dict[str, Any] = field(default_factory=dict)
    # Optional feature frames for peer_align parity with offline/stream.
    # When set (stock-1s replay), peer uses count_peer_align(asof=feature_ts).
    # When None (pure live), peer uses per-symbol live StreamSignalState mf.
    # Also required for Watchdog Hunt arming (washout_reclaim needs day bars).
    stock_by: dict[str, Any] | None = None
    pending_hunts: list = field(default_factory=list)
    day_hunt_symbols: set = field(default_factory=set)
    day_hunt_dirs: set = field(default_factory=set)

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
        if "stock_by" not in kwargs:
            kwargs["stock_by"] = {}
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
        self.pending_hunts = []
        self.day_hunt_symbols = set()
        self.day_hunt_dirs = set()
        self._day_halt = False
        self._watchdog_closed = False
        self._watchdog_last_eval_tod = None
        self._watchdog_date = None
        # Live accumulation starts empty; preloaded research stock_by kept.
        if self.stock_by is None:
            self.stock_by = {}
        # If research preload already has this date, evaluate immediately.
        if self._stock_by_has_date(date):
            self._eval_watchdog(str(date), force=True)

    def _stock_by_has_date(self, date: str) -> bool:
        if not self.stock_by:
            return False
        for sdf in self.stock_by.values():
            if sdf is None or getattr(sdf, "empty", True):
                continue
            if "date" in sdf.columns and (sdf["date"].astype(str) == str(date)).any():
                return True
        return False

    def _hunt_signal_deadline(self, date: str) -> pd.Timestamp:
        wd = self.watchdog
        hhmm = "10:15"
        if wd is not None:
            hhmm = str(getattr(wd.cfg, "hunter_signal_deadline", None) or hhmm)
        return pd.Timestamp(f"{date} {hhmm}:00", tz="America/New_York")

    def _append_stock_bar(self, symbol: str, bar: dict[str, Any]) -> None:
        """Accumulate completed 1m OHLCV into stock_by for Watchdog/Hunt."""
        if self.stock_by is None:
            self.stock_by = {}
        ts = to_ny(bar["timestamp"])
        date = ts.strftime("%Y-%m-%d")
        row = {
            "timestamp": ts,
            "date": date,
            "open": float(bar["open"]),
            "high": float(bar["high"]),
            "low": float(bar["low"]),
            "close": float(bar["close"]),
            "volume": float(bar.get("volume") or 0.0),
        }
        prev = self.stock_by.get(symbol)
        if prev is not None and not getattr(prev, "empty", True):
            ts_col = pd.to_datetime(prev["timestamp"])
            if getattr(ts_col.dt, "tz", None) is None:
                ts_col = ts_col.dt.tz_localize("America/New_York")
            else:
                ts_col = ts_col.dt.tz_convert("America/New_York")
            if bool((ts_col == ts).any()):
                return
            self.stock_by[symbol] = pd.concat([prev, pd.DataFrame([row])], ignore_index=True)
        else:
            self.stock_by[symbol] = pd.DataFrame([row])
        self._stamp_from_prev(symbol)

    def _stamp_from_prev(self, symbol: str) -> None:
        df = self.stock_by.get(symbol)
        if df is None or df.empty:
            return
        pc = None
        st = self.states.get(symbol)
        if st is not None and getattr(st, "prev_close", None) is not None:
            try:
                pc = float(st.prev_close)
            except Exception:
                pc = None
        if pc is None and symbol == "QQQ":
            gate = self.regime_gate
            if gate is not None and getattr(gate, "qqq_previous_close", 0):
                try:
                    pc = float(gate.qqq_previous_close)
                except Exception:
                    pc = None
        if pc is None or not (pc > 0):
            pc = float(df.iloc[0]["open"])
        df = df.copy()
        df["prev_close"] = float(pc)
        df["from_prev"] = df["close"].astype(float) / float(pc) - 1.0
        self.stock_by[symbol] = df

    def on_reference_second(self, symbol: str, tick: dict[str, Any]) -> None:
        """Ingest QQQ (etc.) 1s → completed 1m into stock_by for Halt/Hunt."""
        symbol = str(symbol).upper()
        if symbol not in {"QQQ"}:
            return
        if self.ref_agg is None:
            self.ref_agg = MultiSymbolMinuteAgg(["QQQ"], rth_only=True)
        if isinstance(tick.get("timestamp"), (int, float)):
            tick = {
                **tick,
                "timestamp": pd.Timestamp(
                    float(tick["timestamp"]), unit="s", tz="UTC"
                ).tz_convert("America/New_York"),
            }
        bar = self.ref_agg.on_second(symbol, tick)
        if bar is None:
            return
        self._append_stock_bar(symbol, bar)
        self._maybe_refresh_watchdog(to_ny(bar["timestamp"]))

    def _maybe_refresh_watchdog(self, asof: pd.Timestamp) -> None:
        """Re-evaluate Halt/Degrade/Hunt as morning bars arrive (live causal)."""
        if self.watchdog is None:
            return
        asof = to_ny(asof)
        date = asof.strftime("%Y-%m-%d")
        if self.current_date is None:
            self._roll_day(date)
        if self._watchdog_closed:
            return
        if not self.stock_by:
            return
        tod = asof.strftime("%H:%M")
        # At most once per minute clock
        if self._watchdog_last_eval_tod == tod and self._watchdog_date == date:
            return
        deadline = self._hunt_signal_deadline(date)
        self._eval_watchdog(date)
        self._watchdog_last_eval_tod = tod
        if asof >= deadline:
            self._watchdog_closed = True
            logger.info(
                "WATCHDOG %s eval closed after deadline %s armed=%s pending=%d",
                date,
                deadline.strftime("%H:%M"),
                bool(getattr(self.watchdog, "hunt_armed", False)),
                len(self.pending_hunts),
            )

    def _eval_watchdog(self, date: str, *, force: bool = False) -> None:
        if self.watchdog is None:
            self._watchdog_state = "off"
            self._watchdog_reason = "off"
            self._watchdog_route = "baseline"
            self._day_halt = False
            return
        if not self.stock_by or not self._stock_by_has_date(date):
            self._watchdog_state = "normal"
            self._watchdog_reason = "no_stock_by"
            self._watchdog_route = "baseline"
            self._day_halt = False
            if force:
                logger.info("WATCHDOG %s skip evaluate (no stock_by yet)", date)
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
            regime_cfg = getattr(self.regime_gate, "cfg", None) if self.regime_gate else None
            if regime_cfg is not None and self._watchdog_snap is not None:
                self.watchdog.apply_to_regime(regime_cfg, self._watchdog_snap)
            self._watchdog_date = str(date)
            self._watchdog_state = dec.state.value
            self._watchdog_reason = dec.reason
            self._watchdog_route = dec.overlay.route_tag or "baseline"
            self._day_halt = str(self._watchdog_state) == "halt"
            # Reschedule Hunt only if we have not already filled today's Hunt budget.
            if self.n_hunt_emitted == 0 and not self._day_halt:
                self._schedule_hunts(str(date))
            elif self._day_halt:
                self.pending_hunts = []
            logger.info(
                "WATCHDOG %s state=%s reason=%s route=%s hunt_armed=%s n_hunt_cand=%d pending=%d",
                date,
                self._watchdog_state,
                self._watchdog_reason,
                self._watchdog_route,
                bool(self.watchdog.hunt_armed),
                len(getattr(self.watchdog, "hunt_candidates", None) or []),
                len(self.pending_hunts),
            )
        except Exception as exc:
            self._watchdog_state = "normal"
            self._watchdog_reason = f"error:{type(exc).__name__}"
            self._watchdog_route = "baseline"
            self._day_halt = False
            logger.warning("WATCHDOG %s evaluate failed: %s", date, exc)

    def _schedule_hunts(self, date: str) -> None:
        """Queue Hunt fires at feature_ts + bar_delay (same clock as stream/offline)."""
        self.pending_hunts = []
        if self._day_halt or self.watchdog is None:
            return
        if self.is_event_blackout(date):
            return
        if not bool(getattr(self.watchdog, "hunt_armed", False)):
            return
        trade = self.profile.get("trade") or {}
        bar_delay = int(trade.get("bar_availability_delay_seconds", 0) or 0)
        n_new = 0
        for hc in list(getattr(self.watchdog, "hunt_candidates", None) or []):
            feature_ts = to_ny(hc.sig_ts)
            entry_ts = feature_ts + pd.Timedelta(seconds=bar_delay)
            if entry_ts > to_ny(hc.armed_until):
                continue
            self.pending_hunts.append(
                {
                    "entry_ts": entry_ts,
                    "feature_ts": feature_ts,
                    "symbol": str(hc.symbol),
                    "dir": str(hc.direction),
                    "date": date,
                    "detector": str(getattr(hc, "detector", "") or ""),
                }
            )
            n_new += 1
        self.pending_hunts.sort(key=lambda x: (x["entry_ts"], x["symbol"]))
        # Count unique schedule waves once per non-empty rebuild
        if n_new:
            self.n_hunt_signals = max(int(self.n_hunt_signals), n_new)

    def _spot_at(self, sym: str, date: str, feature_ts: pd.Timestamp) -> float | None:
        if self.stock_by:
            sdf = self.stock_by.get(sym)
            if sdf is not None and not getattr(sdf, "empty", True):
                day = sdf[sdf["date"].astype(str) == str(date)]
                if not day.empty:
                    ts = pd.to_datetime(day["timestamp"])
                    if getattr(ts.dt, "tz", None) is None:
                        ts = ts.dt.tz_localize("America/New_York")
                    else:
                        ts = ts.dt.tz_convert("America/New_York")
                    day = day.assign()
                    day["_ts"] = ts
                    upto = day[day["_ts"] <= feature_ts]
                    if not upto.empty:
                        try:
                            return float(upto.iloc[-1]["close"])
                        except Exception:
                            pass
        st = self.states.get(sym)
        if st is not None and getattr(st, "bars", None):
            try:
                return float(st.bars[-1]["close"])
            except Exception:
                return None
        return None

    def drain_hunts(self, ts: pd.Timestamp) -> list[ScannerSignal]:
        """Emit Hunt signals due at/before ``ts``. Safe to call every tick/frame."""
        ts = to_ny(ts)
        if self.current_date is None:
            return []
        if self._day_halt or self.is_event_blackout(self.current_date):
            return []
        if not self.pending_hunts:
            return []
        due = [h for h in self.pending_hunts if h["entry_ts"] <= ts]
        if not due:
            return []
        self.pending_hunts = [h for h in self.pending_hunts if h["entry_ts"] > ts]
        out: list[ScannerSignal] = []
        for h in due:
            sig = self._emit_hunt(h)
            if sig is not None:
                out.append(sig)
        return out

    def _emit_hunt(self, h: dict[str, Any]) -> ScannerSignal | None:
        """Build + emit one Hunt ScannerSignal (does not consume TopK day_fires)."""
        if self.watchdog is not None and self.watchdog.hunt_budget_remaining() <= 0:
            self.n_hunt_budget_skip += 1
            return None
        symbol = str(h["symbol"])
        direction = str(h["dir"]).upper()
        date = str(h["date"])
        feature_ts = to_ny(h["feature_ts"])
        entry_ts = to_ny(h["entry_ts"])
        spot = self._spot_at(symbol, date, feature_ts)
        if spot is None or float(spot) <= 0:
            logger.warning("HUNT skip %s %s %s: no spot", date, symbol, direction)
            return None

        trade = self.profile.get("trade") or {}
        money = str(trade.get("moneyness", "ATM"))
        books = self.books or ContractBooks.from_profile(self.profile)
        pick = resolve_entry_contract(
            books,
            symbol=symbol,
            date=date,
            direction=direction,
            moneyness=money,
            sig_ts=entry_ts,
            spot=float(spot),
        )
        fill = self.profile.get("fill") or {}
        hold_minutes = int(trade.get("hold_minutes", 30))
        sl_mult = float(trade.get("sl_mult", 0.4))
        tp_mult = float(trade.get("tp_mult", 1.6))
        if self.watchdog is not None:
            from maga7.common.watchdog import hunt_trade_overrides

            hov = hunt_trade_overrides(self.watchdog.cfg)
            if hov.get("hold_minutes") is not None:
                hold_minutes = int(hov["hold_minutes"])
            if hov.get("sl_mult") is not None:
                sl_mult = float(hov["sl_mult"])
            if hov.get("tp_mult") is not None:
                tp_mult = float(hov["tp_mult"])

        delay = int(trade.get("bar_availability_delay_seconds", 0) or 0)
        det = str(h.get("detector") or "")
        wd_reason = f"hunt:{det}" if det else f"hunt:{self._watchdog_reason}"
        sig = ScannerSignal(
            date=date,
            symbol=symbol,
            direction=direction,
            sig_ts=entry_ts,
            spot=float(spot),
            rank=0,  # Hunt is outside TopK ranking
            bucket_id=pick.bucket_id,
            contract=pick.ticker,
            moneyness=money,
            meta={
                "fill_frac": float(fill.get("entry_frac", 0.8)),
                "tp_mult": tp_mult,
                "sl_mult": sl_mult,
                "hold_minutes": hold_minutes,
                "bar_source": "hunt",
                "contract_source": pick.source,
                "sig_dte": pick.dte,
                "sig_strike": pick.strike,
                "contract_mode": books.mode,
                "feature_ts": feature_ts.isoformat(),
                "decision_ts": entry_ts.isoformat(),
                "bar_availability_delay_seconds": delay,
                "watchdog_state": "hunt",
                "watchdog_reason": wd_reason,
                "route": "hunt",
                "event_source": "hunt",
                "hunt_detector": det,
            },
        )

        # Hunt skips peer / QQQ by hunter flags (mirror stream/offline).
        wd_cfg = getattr(self.watchdog, "cfg", None) if self.watchdog is not None else None
        skip_qqq = bool(getattr(wd_cfg, "hunter_skip_qqq_align", False)) if wd_cfg else False
        skip_peer = bool(getattr(wd_cfg, "hunter_skip_peer", True)) if wd_cfg else True

        if self.regime_gate is not None and not skip_qqq:
            dec = self.regime_gate.check(direction, feature_ts)
            sig.meta["regime_reason"] = getattr(dec, "reason", None)
            sig.meta["regime_size_scale"] = float(getattr(dec, "size_scale", 1.0) or 1.0)
            if not dec.allow:
                self.n_regime_block += 1
                logger.info("HUNT_REGIME_BLOCK %s %s %s", date, symbol, direction)
                return None

        sig_cfg = self.profile.get("signal") or {}
        peer_min = sig_cfg.get("peer_align_min")
        if (not skip_peer) and peer_min is not None and int(peer_min) > 0:
            peer_n = self._peer_align_n(direction, date=date, feature_ts=feature_ts)
            sig.meta["peer_align_n"] = peer_n
            if peer_n < int(peer_min):
                self.n_peer_block += 1
                logger.info("HUNT_PEER_BLOCK %s %s %s", date, symbol, direction)
                return None

        if self.watchdog is not None:
            self.watchdog.note_hunt_entry()
        self.day_hunt_symbols.add(symbol)
        self.day_hunt_dirs.add((symbol, direction))
        self.n_hunt_emitted += 1
        self.signals.append(sig)
        if not self.emit_all:
            self.n_done[symbol] = self.n_done.get(symbol, 0) + 1
        logger.info(
            "HUNT signal %s %s %s contract=%s src=%s dte=%s detector=%s",
            date,
            symbol,
            direction,
            pick.ticker,
            pick.source,
            pick.dte,
            det,
        )
        if self.on_signal:
            self.on_signal(sig)
        return sig

    def set_event_blackout(
        self, blackout: set[str] | None, meta: dict[str, Any] | None = None
    ) -> None:
        self.event_blackout = set(blackout or ())
        self.event_blackout_meta = dict(meta or {})

    def is_event_blackout(self, date: str | None = None) -> bool:
        d = date or self.current_date
        return bool(d) and str(d) in self.event_blackout

    def on_stock_second(self, symbol: str, tick: dict[str, Any]) -> ScannerSignal | None:
        """Ingest 1s (or trade) print → maybe complete a 1m bar → Rule-A / Hunt."""
        if self.minute_agg is None:
            self.minute_agg = MultiSymbolMinuteAgg(self.profile["symbols"], rth_only=True)
        ts = to_ny(tick["timestamp"])
        date = ts.strftime("%Y-%m-%d")
        self._roll_day(date)
        # Even without a completed 1m bar, Hunt may become due on this clock.
        self.drain_hunts(ts)
        bar = self.minute_agg.on_second(symbol, tick)
        if bar is None:
            return None
        bar = {
            **bar,
            "bar_source": "1s_agg",
            "available_ts": ts,
        }
        return self.on_stock_bar(symbol, bar)

    def flush_seconds(self) -> list[ScannerSignal]:
        """Flush open minute bars (e.g. end of day / stream)."""
        out: list[ScannerSignal] = []
        if self.minute_agg is not None:
            for bar in self.minute_agg.flush_all():
                bar = {**bar, "bar_source": "1s_agg"}
                before = len(self.signals)
                self.on_stock_bar(bar["symbol"], bar)
                out.extend(self.signals[before:])
        if self.pending_hunts:
            last = max(h["entry_ts"] for h in self.pending_hunts)
            for hs in self.drain_hunts(last):
                if hs not in out:
                    out.append(hs)
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
        # Live/research: keep growing stock_by so Watchdog can arm after wash window.
        self._append_stock_bar(symbol, bar)
        self._maybe_refresh_watchdog(feature_ts)
        # Hunt first (time-driven); may emit via on_signal / self.signals.
        self.drain_hunts(ts)

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
        if self._day_halt:
            self.n_halt_skip += 1
            return None

        trade = self.profile.get("trade") or {}
        use_reentry = self.emit_all
        max_n = int(trade.get("max_entries_per_symbol", 5)) if use_reentry else 1
        cooldown = int(trade.get("cooldown_minutes", 5))
        only_win = resolve_only_win_reenter(trade)
        direction = str(fire["dir"]).upper()

        # Mutex vs prior Hunt (mirror stream/offline).
        if self.watchdog is not None and bool(
            getattr(self.watchdog.cfg, "hunter_mutex_with_baseline", False)
        ):
            scope = str(
                getattr(self.watchdog.cfg, "hunter_mutex_scope", "symbol") or "symbol"
            ).lower()
            if scope in {"symbol_dir", "dir", "same_dir"}:
                mutex_hit = (str(symbol), direction) in self.day_hunt_dirs
            else:
                mutex_hit = str(symbol) in self.day_hunt_symbols
            if mutex_hit:
                self.n_hunt_mutex_skip += 1
                return None

        already = any(f.symbol == symbol for f in self.day_fires)
        if already and not use_reentry:
            # Hunt may have taken n_done; still allow opposite baseline once.
            allow_opp = (
                bool(getattr(self.watchdog.cfg, "hunter_allow_baseline_opposite", False))
                if self.watchdog is not None
                else False
            )
            allow_opp = (
                allow_opp
                and str(symbol) in self.day_hunt_symbols
                and (str(symbol), direction) not in self.day_hunt_dirs
                and any(s == str(symbol) and d != direction for s, d in self.day_hunt_dirs)
            )
            if not allow_opp:
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
                allow_opp = (
                    bool(getattr(self.watchdog.cfg, "hunter_allow_baseline_opposite", False))
                    if self.watchdog is not None
                    else False
                )
                allow_opp = (
                    allow_opp
                    and str(symbol) in self.day_hunt_symbols
                    and (str(symbol), direction) not in self.day_hunt_dirs
                    and any(
                        s == str(symbol) and d != direction for s, d in self.day_hunt_dirs
                    )
                )
                if not allow_opp:
                    return None
            if self.last_exit.get(symbol) is not None and ts < self.last_exit[symbol] + pd.Timedelta(minutes=cooldown):
                return None
            if only_win and self.n_done.get(symbol, 0) > 0 and not self.last_win.get(symbol, True):
                return None

        money = str(trade.get("moneyness", "ATM"))
        spot = float(fire["spot"])

        # Entry confirm (weekday-gated); parity with offline / stream.
        confirm_bars_raw = trade.get("entry_confirm_bars") or (
            self.profile.get("signal") or {}
        ).get("entry_confirm_bars")
        confirm_bars_n = int(confirm_bars_raw) if confirm_bars_raw is not None else 0
        confirm_mode = str(
            trade.get("entry_confirm_mode")
            or (self.profile.get("signal") or {}).get("entry_confirm_mode")
            or "mf"
        ).strip().lower()
        confirm_wd_raw = trade.get("entry_confirm_weekdays") or (
            self.profile.get("signal") or {}
        ).get("entry_confirm_weekdays")
        use_confirm = confirm_bars_n > 0
        if use_confirm and confirm_wd_raw is not None:
            if isinstance(confirm_wd_raw, str):
                confirm_wds = {
                    int(x.strip()) for x in confirm_wd_raw.split(",") if str(x).strip() != ""
                }
            else:
                confirm_wds = {int(x) for x in confirm_wd_raw}
            try:
                wd0 = int(pd.Timestamp(str(date)).weekday())
            except Exception:
                wd0 = -1
            use_confirm = wd0 in confirm_wds
        if use_confirm:
            from maga7.common.replay import entry_confirm_ok

            sdf_c = (self.stock_by or {}).get(symbol)
            stock_day_c = None
            if sdf_c is not None and not getattr(sdf_c, "empty", True):
                stock_day_c = sdf_c[sdf_c["date"].astype(str) == str(date)]
            ok_c, confirm_ft, _, _, _ = entry_confirm_ok(
                stock_day_c,
                direction=direction,
                feature_ts=feature_ts,
                confirm_bars=confirm_bars_n,
                mode=confirm_mode,
            )
            if not ok_c:
                return None
            ts = to_ny(confirm_ft) + pd.Timedelta(seconds=delay)

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
        if pick.ticker is None:
            return None
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

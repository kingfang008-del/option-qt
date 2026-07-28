"""Mag7 multi-symbol Rule-A scanner — emits TopK + Hunt signals for OMS / shadow.

Does NOT use QQQ TFT/FCS as the primary signal path. Optional regime filter
can be added later (QQQ state only gates Mag7 entries).

Hunt: mirrors ``stream_engine`` — ``begin_day`` arms candidates, then
``drain_hunts`` emits at confirm_ft + bar_delay (entry_confirm when gated;
does not consume TopK slots).

Stock path confirm (S1): after fill clock, wait for +thr_pos before -thr_neg
using ``stock_path_confirm_ok(..., asof_ts=)``; pending candidates sit in
``pending_path`` until pos / neg / timeout.
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
from maga7.common.replay import stock_path_confirm_ok, to_ny
from maga7.common.signals import (
    StreamSignalState,
    attach_mf_features,
    count_peer_align,
    resolve_mf_fast_window,
)

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
    # Offline-parity entry morph gates (feature / entry clock).
    n_fo_lod_chase_block: int = 0
    n_up_gap_stall_block: int = 0
    n_dn_gap_stall_block: int = 0
    n_peer_gap_block: int = 0
    n_overnight_gap_block: int = 0
    n_range_stall_block: int = 0
    n_entry_morph_scale: int = 0
    # Satellite sleeve: qqq_open_cont (09:45 continuation; not TopK / not Hunt).
    n_open_cont_signals: int = 0
    n_open_cont_emitted: int = 0
    n_open_cont_skip: int = 0
    _open_cont_done_date: str | None = None
    _qqq_rth_open: float | None = None
    _qqq_rth_open_date: str | None = None
    _qqq_last_px: float | None = None
    _qqq_last_ts: pd.Timestamp | None = None
    # Satellite sleeve: am_pulse (Mag7 FO 09:30–10:25; not TopK / not Hunt).
    n_am_pulse_signals: int = 0
    n_am_pulse_emitted: int = 0
    n_am_pulse_skip: int = 0
    n_am_pulse_shadow: int = 0
    pending_am_pulse: list = field(default_factory=list)
    _am_pulse_scout: Any = None
    _am_pulse_scout_date: str | None = None
    # Independent AM_EXT lane (10:25–11:30); never shares AM trigger budget.
    n_am_pulse_extension_signals: int = 0
    n_am_pulse_extension_emitted: int = 0
    n_am_pulse_extension_skip: int = 0
    n_am_pulse_extension_shadow: int = 0
    pending_am_pulse_extension: list = field(default_factory=list)
    _am_pulse_extension_scout: Any = None
    _am_pulse_extension_scout_date: str | None = None
    event_blackout: set = field(default_factory=set)
    event_symbol_blackout: dict = field(default_factory=dict)  # date -> {SYM}
    event_blackout_meta: dict[str, Any] = field(default_factory=dict)
    # Optional feature frames for peer_align parity with offline/stream.
    # When set (stock-1s replay), peer uses count_peer_align(asof=feature_ts).
    # When None (pure live), peer uses per-symbol live StreamSignalState mf.
    # Also required for Watchdog Hunt arming (washout_reclaim needs day bars).
    stock_by: dict[str, Any] | None = None
    pending_hunts: list = field(default_factory=list)
    # Path-confirm wait queue (baseline TopK + Hunt) — live asof pending.
    pending_path: list = field(default_factory=list)
    day_hunt_symbols: set = field(default_factory=set)
    day_hunt_dirs: set = field(default_factory=set)
    # Earliest-TopK seats reserved at first Rule-A (before confirm/regime/peer).
    day_topk_syms: set = field(default_factory=set)
    # When True, stock_by is research-preloaded (lookback+features); do not mutate.
    stock_by_frozen: bool = False
    n_stock_path_confirm_block: int = 0
    n_stock_path_confirm_ok: int = 0

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

    def _entry_morph_feature_gates(
        self,
        *,
        symbol: str,
        date: str,
        direction: str,
        feature_ts: pd.Timestamp,
    ) -> tuple[bool, float, dict[str, Any]]:
        """Offline-parity feature-clock gates (fo_lod / gap stalls / peer_gap / overnight).

        Runs before TopK seat reservation so chase names do not consume seats.
        """
        from maga7.common.dn_gap_stall_gate import (
            parse_dn_gap_stall_gate,
            resolve_dn_gap_stall_gate,
        )
        from maga7.common.fo_lod_chase_gate import (
            parse_fo_lod_chase_gate,
            resolve_fo_lod_chase_gate,
        )
        from maga7.common.overnight_gap_gate import (
            parse_overnight_gap_gate,
            resolve_overnight_gap_gate,
        )
        from maga7.common.peer_gap_gate import parse_peer_gap_gate, resolve_peer_gap_gate
        from maga7.common.up_gap_stall_gate import (
            parse_up_gap_stall_gate,
            resolve_up_gap_stall_gate,
        )

        trade = self.profile.get("trade") or {}
        sdf = (self.stock_by or {}).get(symbol)
        size_mult = 1.0
        meta: dict[str, Any] = {}
        peer_n = self._peer_align_n(direction, date=date, feature_ts=feature_ts)
        meta["peer_align_n_gate"] = int(peer_n)

        og_cfg = parse_overnight_gap_gate(trade.get("overnight_gap_gate"))
        if og_cfg.enabled:
            adv = None
            try:
                from maga7.common.adverse_vol_share import (
                    adverse_vol_share_asof,
                    prepare_stock_1s_arrays,
                )
                from maga7.common.bar_agg import load_stock_1s_day

                paths = self.profile.get("_paths") or self.profile.get("paths") or {}
                s1s = paths.get("stock_1s_root")
                lag = int(getattr(og_cfg, "lag_seconds", 0) or 0)
                og_ts = (
                    to_ny(feature_ts) + pd.Timedelta(seconds=lag)
                    if lag > 0
                    else to_ny(feature_ts)
                )
                if s1s and getattr(og_cfg, "require_adv_share", None) is not None:
                    day1s = load_stock_1s_day(Path(s1s), symbol, date)
                    if day1s is not None and not day1s.empty:
                        adv = adverse_vol_share_asof(
                            prepare_stock_1s_arrays(day1s),
                            now_ts=og_ts,
                            window_seconds=120,
                            direction=str(direction),
                        )
            except Exception:
                adv = None
            og = resolve_overnight_gap_gate(
                og_cfg,
                stock_df=sdf,
                date=str(date),
                direction=str(direction),
                adv_share=adv,
            )
            meta["overnight_gap_reason"] = og.reason
            if not og.allow:
                self.n_overnight_gap_block += 1
                logger.info(
                    "OVERNIGHT_GAP_BLOCK %s %s %s reason=%s",
                    date,
                    symbol,
                    direction,
                    og.reason,
                )
                return False, 0.0, meta
            if abs(float(og.size_scale) - 1.0) > 1e-12:
                size_mult *= float(og.size_scale)
                self.n_entry_morph_scale += 1

        pg_cfg = parse_peer_gap_gate(trade.get("peer_gap_gate"))
        if pg_cfg.enabled:
            pg = resolve_peer_gap_gate(
                pg_cfg,
                stock_df=sdf,
                date=str(date),
                direction=str(direction),
                peer_n=peer_n,
                from_open=None,
            )
            meta["peer_gap_reason"] = pg.reason
            if not pg.allow:
                self.n_peer_gap_block += 1
                logger.info(
                    "PEER_GAP_BLOCK %s %s %s reason=%s peer=%s",
                    date,
                    symbol,
                    direction,
                    pg.reason,
                    peer_n,
                )
                return False, 0.0, meta
            if abs(float(pg.size_scale) - 1.0) > 1e-12:
                size_mult *= float(pg.size_scale)
                self.n_entry_morph_scale += 1

        dgs_cfg = parse_dn_gap_stall_gate(trade.get("dn_gap_stall_gate"))
        if dgs_cfg.enabled:
            dgs = resolve_dn_gap_stall_gate(
                dgs_cfg,
                stock_df=sdf,
                date=str(date),
                asof_ts=feature_ts,
                direction=str(direction),
                peer_n=peer_n,
            )
            meta["dn_gap_stall_reason"] = dgs.reason
            if not dgs.allow:
                self.n_dn_gap_stall_block += 1
                logger.info(
                    "DN_GAP_STALL_BLOCK %s %s %s reason=%s",
                    date,
                    symbol,
                    direction,
                    dgs.reason,
                )
                return False, 0.0, meta
            if abs(float(dgs.size_scale) - 1.0) > 1e-12:
                size_mult *= float(dgs.size_scale)
                self.n_entry_morph_scale += 1

        ugs_cfg = parse_up_gap_stall_gate(trade.get("up_gap_stall_gate"))
        if ugs_cfg.enabled:
            ugs = resolve_up_gap_stall_gate(
                ugs_cfg,
                stock_df=sdf,
                date=str(date),
                asof_ts=feature_ts,
                direction=str(direction),
            )
            meta["up_gap_stall_reason"] = ugs.reason
            if not ugs.allow:
                self.n_up_gap_stall_block += 1
                logger.info(
                    "UP_GAP_STALL_BLOCK %s %s %s reason=%s",
                    date,
                    symbol,
                    direction,
                    ugs.reason,
                )
                return False, 0.0, meta
            if abs(float(ugs.size_scale) - 1.0) > 1e-12:
                size_mult *= float(ugs.size_scale)
                self.n_entry_morph_scale += 1

        flc_cfg = parse_fo_lod_chase_gate(trade.get("fo_lod_chase_gate"))
        if flc_cfg.enabled:
            flc = resolve_fo_lod_chase_gate(
                flc_cfg,
                stock_df=sdf,
                date=str(date),
                asof_ts=feature_ts,
                direction=str(direction),
            )
            meta["fo_lod_chase_reason"] = flc.reason
            meta["fo_lod_fav"] = flc.fav_from_open
            meta["fo_lod_chase"] = flc.chase
            meta["fo_lod_dist_ext"] = flc.dist_ext
            if not flc.allow:
                self.n_fo_lod_chase_block += 1
                logger.info(
                    "FO_LOD_CHASE_BLOCK %s %s %s reason=%s fo=%s chase=%s dist=%s",
                    date,
                    symbol,
                    direction,
                    flc.reason,
                    flc.fav_from_open,
                    flc.chase,
                    flc.dist_ext,
                )
                return False, 0.0, meta
            if abs(float(flc.size_scale) - 1.0) > 1e-12:
                size_mult *= float(flc.size_scale)
                self.n_entry_morph_scale += 1

        return True, float(size_mult), meta

    def _entry_morph_range_stall(
        self,
        *,
        symbol: str,
        date: str,
        direction: str,
        entry_ts: pd.Timestamp,
        peer_n: int | None,
    ) -> tuple[bool, float, dict[str, Any]]:
        """Range-chase stall at final entry clock (offline parity)."""
        from maga7.common.range_stall_gate import (
            parse_range_stall_gate,
            resolve_range_stall_gate,
        )

        trade = self.profile.get("trade") or {}
        cfg = parse_range_stall_gate(trade.get("range_stall_gate"))
        meta: dict[str, Any] = {}
        if not cfg.enabled:
            return True, 1.0, meta
        sdf = (self.stock_by or {}).get(symbol)
        rs = resolve_range_stall_gate(
            cfg,
            stock_df=sdf,
            date=str(date),
            asof_ts=entry_ts,
            direction=str(direction),
            peer_n=peer_n,
        )
        meta["range_stall_reason"] = rs.reason
        if not rs.allow:
            self.n_range_stall_block += 1
            logger.info(
                "RANGE_STALL_BLOCK %s %s %s reason=%s",
                date,
                symbol,
                direction,
                rs.reason,
            )
            return False, 0.0, meta
        if abs(float(rs.size_scale) - 1.0) > 1e-12:
            self.n_entry_morph_scale += 1
            return True, float(rs.size_scale), meta
        return True, 1.0, meta

    def _roll_day(self, date: str) -> None:
        if self.current_date == date:
            return
        prev = self.current_date
        if prev is not None:
            # Resolve pending Hunt / path-confirm before dropping day state.
            # Otherwise cross-day rolls silently drop S1 waits (missed Hunt emits).
            eod = pd.Timestamp(f"{prev} 20:00:00", tz="America/New_York")
            self.drain_hunts(eod)
            self.drain_path_confirms(eod)
        self.current_date = date
        self.day_fires = []
        self.n_done = {s: 0 for s in self.profile["symbols"]}
        self.last_exit = {s: None for s in self.profile["symbols"]}
        self.last_win = {s: True for s in self.profile["symbols"]}
        self.pending_hunts = []
        self.pending_path = []
        self.pending_am_pulse = []
        self.pending_am_pulse_extension = []
        self.day_hunt_symbols = set()
        self.day_hunt_dirs = set()
        self.day_topk_syms = set()
        self._day_halt = False
        self._watchdog_closed = False
        self._watchdog_last_eval_tod = None
        self._watchdog_date = None
        # Open-cont is once/day; clear RTH open tracker on day roll.
        if self._qqq_rth_open_date != date:
            self._qqq_rth_open = None
            self._qqq_rth_open_date = None
            self._qqq_last_px = None
            self._qqq_last_ts = None
        # Am-pulse scout resets each session day.
        self._am_pulse_scout = None
        self._am_pulse_scout_date = None
        self._am_pulse_extension_scout = None
        self._am_pulse_extension_scout_date = None
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
        if self.stock_by_frozen:
            return
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
        # entry_confirm / peer_align need mf10+streak on stock_by (same as offline bars)
        sig = self.profile.get("signal") or {}
        mf_w = int(sig.get("mf_window", 10) or 10)
        vol_w = int(sig.get("vol_ma_window", 20) or 20)
        fast_w = resolve_mf_fast_window(sig)
        df = attach_mf_features(
            df,
            mf_window=mf_w,
            vol_ma_window=vol_w,
            mf_fast_window=fast_w,
        )
        # Keep overnight prev_close if StreamSignalState provided one
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
        # Track RTH open / last for satellite qqq_open_cont (clock 09:45).
        try:
            ts = to_ny(tick.get("timestamp"))
            px = float(tick.get("close", tick.get("price", tick.get("last")) or 0.0))
        except Exception:
            ts, px = None, 0.0
        if ts is not None and px > 0:
            d = ts.strftime("%Y-%m-%d")
            t = ts.time()
            if t >= pd.Timestamp("09:30").time() and t < pd.Timestamp("16:00").time():
                if self._qqq_rth_open_date != d or self._qqq_rth_open is None:
                    self._qqq_rth_open = float(px)
                    self._qqq_rth_open_date = d
                self._qqq_last_px = float(px)
                self._qqq_last_ts = ts
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
            # Reschedule Hunt while *today's* budget remains.
            # Do NOT gate on cumulative ``n_hunt_emitted`` — that blocked Hunt on
            # later sessions after the first Hunt of a multi-day replay.
            if self._day_halt:
                self.pending_hunts = []
            elif (
                self.watchdog is None
                or self.watchdog.hunt_budget_remaining() > 0
            ):
                self._schedule_hunts(str(date))
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
            if str(dec.reason or "").startswith("prevention:"):
                logger.info(
                    "PREVENTION %s expert=%s prefer_risk_off=%s",
                    date,
                    dec.expert,
                    bool((self.watchdog.cfg.prevention_router_cfg or {}).get("prefer_risk_off")),
                )
        except Exception as exc:
            self._watchdog_state = "normal"
            self._watchdog_reason = f"error:{type(exc).__name__}"
            self._watchdog_route = "baseline"
            self._day_halt = False
            logger.warning("WATCHDOG %s evaluate failed: %s", date, exc)

    def _path_confirm_cfg(self) -> dict[str, Any] | None:
        trade = self.profile.get("trade") or {}
        raw = trade.get("stock_path_confirm") or {}
        if not isinstance(raw, dict) or not bool(raw.get("enabled", False)):
            return None
        return raw

    def _path_confirm_applies(
        self, cfg: dict[str, Any], *, date: str, entry_ts: pd.Timestamp
    ) -> bool:
        wd_raw = cfg.get("weekdays")
        if wd_raw is not None:
            if isinstance(wd_raw, str):
                wds = {int(x.strip()) for x in wd_raw.split(",") if str(x).strip() != ""}
            else:
                wds = {int(x) for x in wd_raw}
            try:
                wd = int(pd.Timestamp(str(date)).weekday())
            except Exception:
                wd = -1
            if wd not in wds:
                return False
        tod_start = cfg.get("tod_start")
        tod_end = cfg.get("tod_end")
        if tod_start is not None and tod_end is not None:
            try:
                sh, sm = str(tod_start).split(":", 1)
                eh, em = str(tod_end).split(":", 1)
                t0 = int(sh) * 60 + int(sm)
                t1 = int(eh) * 60 + int(em)
            except Exception:
                return True
            hm = int(to_ny(entry_ts).hour) * 60 + int(to_ny(entry_ts).minute)
            if not (t0 <= hm <= t1):
                return False
        return True

    def _eval_path_confirm(
        self,
        *,
        symbol: str,
        date: str,
        direction: str,
        entry_ts: pd.Timestamp,
        asof_ts: pd.Timestamp,
        cfg: dict[str, Any],
    ) -> tuple[str, pd.Timestamp | None, str]:
        """Return ``(ok|block|pending, confirm_ts, reason)``."""
        sdf = (self.stock_by or {}).get(symbol)
        stock_day = None
        if sdf is not None and not getattr(sdf, "empty", True):
            stock_day = sdf[sdf["date"].astype(str) == str(date)]
        ok_p, path_ts, reason = stock_path_confirm_ok(
            stock_day,
            direction=direction,
            entry_ts=entry_ts,
            thr_pos=float(cfg.get("thr_pos", 0.0015) or 0.0015),
            thr_neg=float(cfg.get("thr_neg", -0.003) or -0.003),
            max_wait_seconds=int(cfg.get("max_wait_seconds", 300) or 300),
            on_timeout=str(cfg.get("on_timeout", "block") or "block"),
            asof_ts=asof_ts,
        )
        if reason == "pending":
            return "pending", None, reason
        if not ok_p:
            return "block", path_ts, reason
        return "ok", path_ts, reason

    def _gate_path_confirm(
        self,
        *,
        route: str,
        symbol: str,
        date: str,
        direction: str,
        feature_ts: pd.Timestamp,
        entry_ts: pd.Timestamp,
        asof_ts: pd.Timestamp,
        stash: dict[str, Any],
    ) -> tuple[str, pd.Timestamp, str | None]:
        """Apply S1 path gate. Returns ``(ok|block|pending|skip, entry_ts, reason)``."""
        cfg = self._path_confirm_cfg()
        if cfg is None or not self._path_confirm_applies(
            cfg, date=date, entry_ts=entry_ts
        ):
            return "skip", entry_ts, None
        status, path_ts, reason = self._eval_path_confirm(
            symbol=symbol,
            date=date,
            direction=direction,
            entry_ts=entry_ts,
            asof_ts=asof_ts,
            cfg=cfg,
        )
        if status == "pending":
            self.pending_path.append(
                {
                    **stash,
                    "route": route,
                    "symbol": symbol,
                    "date": date,
                    "direction": direction,
                    "feature_ts": feature_ts,
                    "entry_ts": entry_ts,
                    "path_cfg": cfg,
                }
            )
            return "pending", entry_ts, reason
        if status == "block":
            self.n_stock_path_confirm_block += 1
            logger.info(
                "PATH_CONFIRM_BLOCK %s %s %s route=%s reason=%s",
                date,
                symbol,
                direction,
                route,
                reason,
            )
            return "block", entry_ts, reason
        self.n_stock_path_confirm_ok += 1
        if (
            bool(cfg.get("delay_on_pos", True))
            and path_ts is not None
            and reason == "pos"
        ):
            delay = int(
                (self.profile.get("trade") or {}).get(
                    "bar_availability_delay_seconds", 0
                )
                or 0
            )
            entry_ts = to_ny(path_ts) + pd.Timedelta(seconds=delay)
        return "ok", entry_ts, reason

    def drain_path_confirms(self, ts: pd.Timestamp) -> list[ScannerSignal]:
        """Resolve pending path-confirm candidates as of ``ts``."""
        if not self.pending_path:
            return []
        asof = to_ny(ts)
        out: list[ScannerSignal] = []
        rest: list[dict[str, Any]] = []
        for item in self.pending_path:
            cfg = item.get("path_cfg") or self._path_confirm_cfg() or {}
            status, path_ts, reason = self._eval_path_confirm(
                symbol=str(item["symbol"]),
                date=str(item["date"]),
                direction=str(item["direction"]),
                entry_ts=to_ny(item["entry_ts"]),
                asof_ts=asof,
                cfg=cfg,
            )
            if status == "pending":
                rest.append(item)
                continue
            if status == "block":
                self.n_stock_path_confirm_block += 1
                logger.info(
                    "PATH_CONFIRM_BLOCK %s %s %s route=%s reason=%s",
                    item["date"],
                    item["symbol"],
                    item["direction"],
                    item.get("route"),
                    reason,
                )
                continue
            self.n_stock_path_confirm_ok += 1
            entry_ts = to_ny(item["entry_ts"])
            if (
                bool(cfg.get("delay_on_pos", True))
                and path_ts is not None
                and reason == "pos"
            ):
                delay = int(
                    (self.profile.get("trade") or {}).get(
                        "bar_availability_delay_seconds", 0
                    )
                    or 0
                )
                entry_ts = to_ny(path_ts) + pd.Timedelta(seconds=delay)
            item = {**item, "entry_ts": entry_ts, "path_confirm_reason": reason}
            if str(item.get("route") or "") == "hunt":
                sig = self._emit_hunt_resolved(item)
            else:
                sig = self._emit_topk_resolved(item)
            if sig is not None:
                out.append(sig)
        self.pending_path = rest
        return out

    def _entry_confirm_cfg(self, date: str) -> tuple[bool, int, str]:
        """Weekday-gated entry_confirm knobs (parity with offline / baseline emit)."""
        trade = self.profile.get("trade") or {}
        sig = self.profile.get("signal") or {}
        confirm_bars_raw = trade.get("entry_confirm_bars") or sig.get("entry_confirm_bars")
        confirm_bars_n = int(confirm_bars_raw) if confirm_bars_raw is not None else 0
        confirm_mode = str(
            trade.get("entry_confirm_mode") or sig.get("entry_confirm_mode") or "mf"
        ).strip().lower()
        confirm_wd_raw = trade.get("entry_confirm_weekdays") or sig.get(
            "entry_confirm_weekdays"
        )
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
        return use_confirm, confirm_bars_n, confirm_mode

    def _schedule_hunts(self, date: str) -> None:
        """Queue Hunt fires at confirm_ft + bar_delay (same clock as stream/offline)."""
        self.pending_hunts = []
        if self._day_halt or self.watchdog is None:
            return
        if self.is_event_blackout(date):  # full-day only (no symbol arg)
            return
        if not bool(getattr(self.watchdog, "hunt_armed", False)):
            return
        trade = self.profile.get("trade") or {}
        bar_delay = int(trade.get("bar_availability_delay_seconds", 0) or 0)
        use_confirm, confirm_bars_n, _ = self._entry_confirm_cfg(date)
        n_new = 0
        for hc in list(getattr(self.watchdog, "hunt_candidates", None) or []):
            if self.is_event_blackout(date, symbol=str(hc.symbol)):
                self.n_event_block += 1
                continue
            feature_ts = to_ny(hc.sig_ts)
            # Match offline: fill clock = (feature + confirm_bars) + bar_delay when gated.
            confirm_ft = (
                feature_ts + pd.Timedelta(minutes=confirm_bars_n)
                if use_confirm
                else feature_ts
            )
            entry_ts = confirm_ft + pd.Timedelta(seconds=bar_delay)
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

    def _hunt_confirm_ready(self, h: dict[str, Any]) -> bool:
        """True when stock_by has the confirm bar (avoid early mf-from-prior-bar)."""
        date = str(h["date"])
        use_confirm, confirm_bars_n, _ = self._entry_confirm_cfg(date)
        if not use_confirm:
            return True
        confirm_ft = to_ny(h["feature_ts"]) + pd.Timedelta(minutes=confirm_bars_n)
        sdf = (self.stock_by or {}).get(str(h["symbol"]))
        if sdf is None or getattr(sdf, "empty", True):
            return False
        day = sdf[sdf["date"].astype(str) == date]
        if day.empty or "timestamp" not in day.columns:
            return False
        ts = pd.to_datetime(day["timestamp"])
        if getattr(ts.dt, "tz", None) is None:
            ts = ts.dt.tz_localize("America/New_York")
        else:
            ts = ts.dt.tz_convert("America/New_York")
        return bool((ts >= confirm_ft).any())

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
        rest = [h for h in self.pending_hunts if h["entry_ts"] > ts]
        out: list[ScannerSignal] = []
        for h in due:
            if self.is_event_blackout(
                h.get("date") or self.current_date, symbol=h.get("symbol")
            ):
                self.n_event_block += 1
                continue
            # Wait for confirm minute bar before deciding (causal vs full-day offline).
            if not self._hunt_confirm_ready(h):
                rest.append(h)
                continue
            sig = self._emit_hunt({**h, "_asof": ts})
            if sig is not None:
                out.append(sig)
        self.pending_hunts = sorted(rest, key=lambda x: (x["entry_ts"], x["symbol"]))
        return out

    def drain_open_cont(self, ts: pd.Timestamp) -> list[ScannerSignal]:
        """Emit at most one QQQ open_cont satellite signal when clock is due.

        Independent of Rule-A TopK and Hunt. Requires ``profile.qqq_open_cont.enabled``.
        """
        from maga7.common.qqq_open_cont import (
            load_champion,
            open_cont_enabled,
            resolve_atm_ticker,
            signal_at_clock,
            signal_from_open_spot,
        )

        ts = to_ny(ts)
        if not open_cont_enabled(self.profile):
            return []
        date = self.current_date or ts.strftime("%Y-%m-%d")
        if self.current_date is None:
            self._roll_day(date)
            date = self.current_date or date
        if self._open_cont_done_date == date:
            return []
        if self.is_event_blackout(date, symbol="QQQ"):
            self._open_cont_done_date = date
            self.n_open_cont_skip += 1
            return []
        champ = load_champion(self.profile)
        clock = str(champ.get("clock") or "09:45")
        clock_ts = pd.Timestamp(f"{date} {clock}", tz="America/New_York")
        if ts < clock_ts:
            return []
        # Mark attempted so we only fire once even on no-signal / no-contract.
        self._open_cont_done_date = date
        self.n_open_cont_signals += 1

        oc = None
        if (
            self._qqq_rth_open_date == date
            and self._qqq_rth_open is not None
            and self._qqq_last_px is not None
        ):
            oc = signal_from_open_spot(
                date=date,
                open_px=float(self._qqq_rth_open),
                spot=float(self._qqq_last_px),
                entry_ts=clock_ts,
                from_open_min=float(champ.get("from_open_min", 0.002)),
            )
        if oc is None:
            paths = self.profile.get("_paths") or self.profile.get("paths") or {}
            s1s = paths.get("stock_1s_root")
            if s1s is not None:
                try:
                    oc = signal_at_clock(
                        Path(s1s),
                        date,
                        clock=clock,
                        from_open_min=float(champ.get("from_open_min", 0.002)),
                    )
                except Exception:
                    oc = None
        if oc is None:
            self.n_open_cont_skip += 1
            logger.info("OPEN_CONT skip %s: no signal (|fo| or no QQQ tape)", date)
            return []

        quote_root = champ.get("quote_1s_root")
        if not quote_root:
            paths = self.profile.get("_paths") or self.profile.get("paths") or {}
            quote_root = paths.get("quote_1s_root_qqq") or paths.get("quote_1s_root")
            # Mag7 ladder root is multi-symbol; prefer dedicated QQQ dte0 if present.
            if quote_root and Path(quote_root).name != "QQQ":
                qqq_guess = Path("/mnt/s990/data/raw_1s/dte0_options/QQQ")
                if qqq_guess.is_dir():
                    quote_root = qqq_guess
        ticker, strike = resolve_atm_ticker(quote_root, date, oc.direction)
        books = self.books or ContractBooks.from_profile(self.profile)
        pick = resolve_entry_contract(
            books,
            symbol="QQQ",
            date=date,
            direction=oc.direction,
            moneyness="ATM",
            sig_ts=oc.entry_ts,
            spot=float(oc.spot),
        )
        contract = pick.ticker or ticker
        if contract is None:
            self.n_open_cont_skip += 1
            logger.info("OPEN_CONT skip %s %s: no QQQ ATM contract", date, oc.direction)
            return []

        hold_sec = int(champ.get("max_hold_sec", 900) or 900)
        tp = float(champ.get("tp", 0.10) or 0.10)
        sl = float(champ.get("sl", 0.25) or 0.25)
        fill_frac = float(champ.get("entry_frac", 0.75) or 0.75)
        pos_frac = float(champ.get("position_frac", 0.10) or 0.10)
        meta = {
            "fill_frac": fill_frac,
            # Mag7 OMS uses price multiples: TP at entry*(1+tp), SL at entry*(1-sl).
            "tp_mult": 1.0 + tp,
            "sl_mult": 1.0 - sl,
            "hold_minutes": max(1, int(round(hold_sec / 60.0))),
            "exit_hold_sec": hold_sec,
            "exit_tp_mult": 1.0 + tp,
            "exit_sl_mult": 1.0 - sl,
            "exit_simple": True,
            "bar_source": "qqq_open_cont",
            "contract_source": pick.source if pick.ticker else "qqq_dte0_bucket",
            "sig_dte": pick.dte if pick.ticker else 0,
            "sig_strike": pick.strike if pick.strike is not None else strike,
            "route": "qqq_open_cont",
            "event_source": "qqq_open_cont",
            "from_open": float(oc.from_open),
            "position_frac": pos_frac,
            "watchdog_state": "satellite",
            "watchdog_reason": "qqq_open_cont",
        }
        sig = ScannerSignal(
            date=date,
            symbol="QQQ",
            direction=oc.direction,
            sig_ts=oc.entry_ts,
            spot=float(oc.spot),
            rank=0,
            bucket_id=int(pick.bucket_id) if pick.ticker and pick.bucket_id is not None else (
                2 if oc.direction == "UP" else 0
            ),
            contract=str(contract),
            moneyness="ATM",
            meta=meta,
        )
        self.n_open_cont_emitted += 1
        self.signals.append(sig)
        logger.info(
            "OPEN_CONT signal %s %s fo=%.4f contract=%s",
            date,
            oc.direction,
            oc.from_open,
            contract,
        )
        if self.on_signal:
            self.on_signal(sig)
        return [sig]

    @staticmethod
    def _am_pulse_lane_names(lane: str) -> tuple[str, str, str, str, str, str]:
        if str(lane) == "am_pulse_extension":
            return (
                "_am_pulse_extension_scout",
                "_am_pulse_extension_scout_date",
                "pending_am_pulse_extension",
                "n_am_pulse_extension_signals",
                "n_am_pulse_extension_emitted",
                "n_am_pulse_extension_skip",
            )
        return (
            "_am_pulse_scout",
            "_am_pulse_scout_date",
            "pending_am_pulse",
            "n_am_pulse_signals",
            "n_am_pulse_emitted",
            "n_am_pulse_skip",
        )

    def _ensure_am_pulse_scout(self, date: str, lane: str = "am_pulse"):
        from maga7.common.am_pulse_scout import (
            AmPulseScout,
            am_pulse_lane_enabled,
            load_am_pulse_lane_cfg,
            scout_config_from_live,
        )

        if not am_pulse_lane_enabled(self.profile, lane):
            return None
        scout_attr, date_attr, *_ = self._am_pulse_lane_names(lane)
        current = getattr(self, scout_attr)
        if current is not None and getattr(self, date_attr) == date:
            return current
        cfg = scout_config_from_live(load_am_pulse_lane_cfg(self.profile, lane))
        scout = AmPulseScout(cfg=cfg)
        scout.begin_day(str(date))
        setattr(self, scout_attr, scout)
        setattr(self, date_attr, str(date))
        return scout

    def _feed_am_pulse_lane_bar(
        self,
        lane: str,
        symbol: str,
        bar: dict[str, Any],
    ) -> None:
        """Push one completed bar into an independent AM pulse lane."""
        from maga7.common.am_pulse_scout import (
            am_pulse_lane_enabled,
            load_am_pulse_lane_cfg,
        )

        if not am_pulse_lane_enabled(self.profile, lane):
            return
        live = load_am_pulse_lane_cfg(self.profile, lane)
        if str(live.get("execute_mode") or "shadow") == "off":
            return
        sym = str(symbol).upper()
        if sym not in set(self.profile.get("symbols") or []):
            return
        ts = to_ny(bar.get("timestamp"))
        date = ts.strftime("%Y-%m-%d")
        scout = self._ensure_am_pulse_scout(date, lane)
        if scout is None:
            return
        arm_want = str(live.get("arm") or "FO").upper()
        # Prefer scanner's official RTH open over late-first-bar latch.
        st = (self.states or {}).get(sym)
        day_open = getattr(st, "day_open", None) if st is not None else None
        if day_open is not None and float(day_open) > 0:
            scout.seed_day_open(sym, float(day_open))
        alert = scout.on_bar(
            symbol=sym,
            ts=ts,
            open_=float(bar.get("open") or 0.0),
            high=float(bar.get("high") or 0.0),
            low=float(bar.get("low") or 0.0),
            close=float(bar.get("close") or 0.0),
        )
        if alert is None:
            return
        if arm_want in {"FO", "LB"} and alert.arm != arm_want:
            return
        _, _, pending_attr, signals_attr, _, _ = self._am_pulse_lane_names(lane)
        setattr(self, signals_attr, int(getattr(self, signals_attr)) + 1)
        getattr(self, pending_attr).append(alert)

    def _feed_am_pulse_bar(self, symbol: str, bar: dict[str, Any]) -> None:
        """Backward-compatible feed for the original AM lane."""
        self._feed_am_pulse_lane_bar("am_pulse", symbol, bar)

    def _drain_am_pulse_lane(
        self,
        lane: str,
        ts: pd.Timestamp,
    ) -> list[ScannerSignal]:
        """Emit one AM lane without touching baseline TopK accounting."""
        from maga7.common.am_pulse_scout import am_pulse_lane_enabled

        ts = to_ny(ts)
        if not am_pulse_lane_enabled(self.profile, lane):
            return []
        date = self.current_date or ts.strftime("%Y-%m-%d")
        if self.current_date is None:
            self._roll_day(date)
        _, _, pending_attr, *_ = self._am_pulse_lane_names(lane)
        pending = getattr(self, pending_attr)
        if not pending:
            return []
        out: list[ScannerSignal] = []
        rest: list = []
        for alert in list(pending):
            # Defer if wall clock still before alert bar (shouldn't happen live).
            alert_ts = to_ny(pd.Timestamp(alert.ts))
            if ts < alert_ts:
                rest.append(alert)
                continue
            sig = self._emit_am_pulse_lane(lane, alert)
            if sig is not None:
                out.append(sig)
        setattr(self, pending_attr, rest)
        return out

    def drain_am_pulse(self, ts: pd.Timestamp) -> list[ScannerSignal]:
        """Drain A lane using its configured window (LOCK: 09:30–10:30)."""
        return self._drain_am_pulse_lane("am_pulse", ts)

    def drain_am_pulse_extension(self, ts: pd.Timestamp) -> list[ScannerSignal]:
        """Drain independent B lane (LOCK: 10:30–11:30)."""
        return self._drain_am_pulse_lane("am_pulse_extension", ts)

    def _emit_am_pulse_lane(self, lane: str, alert: Any) -> ScannerSignal | None:
        from maga7.common.am_pulse_scout import load_am_pulse_lane_cfg

        live = load_am_pulse_lane_cfg(self.profile, lane)
        is_ext = lane == "am_pulse_extension"
        route = "am_pulse_extension" if is_ext else "am_pulse"
        event_source = "am_pulse_extension_sleeve" if is_ext else "am_pulse_sleeve"
        log_tag = "AM_PULSE_EXTENSION" if is_ext else "AM_PULSE"
        *_, emitted_attr, skip_attr = self._am_pulse_lane_names(lane)
        shadow_attr = (
            "n_am_pulse_extension_shadow" if is_ext else "n_am_pulse_shadow"
        )
        date = str(alert.date)
        sym = str(alert.symbol).upper()
        direction = str(alert.dir).upper()
        if self.is_event_blackout(date, symbol=sym):
            setattr(self, skip_attr, int(getattr(self, skip_attr)) + 1)
            logger.info("%s skip %s %s: event blackout", log_tag, date, sym)
            return None
        arm_ts = to_ny(pd.Timestamp(alert.ts))
        books = self.books or ContractBooks.from_profile(self.profile)
        # Lane may override lock DTE (e.g. AM_EXT 0DTE-only).
        lane_prefer = live.get("prefer_dte")
        lane_allowed = live.get("allowed_dte")
        prefer_kw = int(lane_prefer) if lane_prefer is not None else None
        allowed_kw = None
        if isinstance(lane_allowed, (list, tuple)) and lane_allowed:
            allowed_kw = [int(x) for x in lane_allowed]
        elif isinstance(lane_allowed, str) and lane_allowed.strip():
            allowed_kw = [
                int(x.strip()) for x in lane_allowed.split(",") if x.strip()
            ]
        pick = resolve_entry_contract(
            books,
            symbol=sym,
            date=date,
            direction=direction,
            moneyness=str(live.get("moneyness") or "ATM"),
            sig_ts=arm_ts,
            spot=float(alert.px),
            prefer_dte=prefer_kw,
            allowed_dte=allowed_kw,
        )
        if not pick.ticker:
            setattr(self, skip_attr, int(getattr(self, skip_attr)) + 1)
            logger.info("%s skip %s %s %s: no ATM contract", log_tag, date, sym, direction)
            return None

        hold_sec = int(live.get("max_hold_sec", 900) or 900)
        # Cap hold so position cannot spill into CORE (flatten_before).
        flat_hhmm = str(live.get("flatten_before") or "10:30")
        try:
            flat_ts = pd.Timestamp(f"{date} {flat_hhmm}", tz="America/New_York")
            max_to_flat = max(1, int((flat_ts - arm_ts).total_seconds()))
            hold_sec = min(hold_sec, max_to_flat)
        except Exception:
            pass
        tp = float(live.get("tp", 0.15) or 0.15)
        sl = float(live.get("sl", 0.20) or 0.20)
        fill_frac = float(live.get("entry_frac", 0.75) or 0.75)
        pos_frac = float(live.get("position_frac", 0.10) or 0.10)
        exec_mode = str(live.get("execute_mode") or "shadow")
        meta = {
            "fill_frac": fill_frac,
            "tp_mult": 1.0 + tp,
            "sl_mult": 1.0 - sl,
            "hold_minutes": max(1, int(round(hold_sec / 60.0))),
            "exit_hold_sec": hold_sec,
            "exit_tp_mult": 1.0 + tp,
            "exit_sl_mult": 1.0 - sl,
            "exit_simple": True,
            "exit_flatten_before": flat_hhmm,
            "bar_source": route,
            "contract_source": pick.source,
            "sig_dte": pick.dte,
            "sig_strike": pick.strike,
            "route": route,
            "event_source": event_source,
            "am_pulse_arm": str(alert.arm),
            "fav_from_open": float(alert.fav_from_open),
            "lookback_ret": alert.lookback_ret,
            "position_frac": pos_frac,
            "execute_mode": exec_mode,
            "max_lag_sec": float(live.get("max_lag_sec", 5.0) or 5.0),
            "max_spread_pct": float(live.get("max_spread_pct", 0.15) or 0.15),
            "min_mid": float(live.get("min_mid", 0.05) or 0.05),
            "watchdog_state": "satellite",
            "watchdog_reason": route,
        }
        ca_raw = live.get("confirm_abort")
        if isinstance(ca_raw, dict) and ca_raw.get("enabled"):
            meta["confirm_abort"] = dict(ca_raw)
        protect_raw = live.get("profit_protect")
        if isinstance(protect_raw, dict) and protect_raw.get("enabled"):
            meta["profit_protect"] = dict(protect_raw)
        sig = ScannerSignal(
            date=date,
            symbol=sym,
            direction=direction,
            sig_ts=arm_ts,
            spot=float(alert.px),
            rank=0,
            bucket_id=int(pick.bucket_id) if pick.bucket_id is not None else (
                2 if direction == "UP" else 0
            ),
            contract=str(pick.ticker),
            moneyness=str(live.get("moneyness") or "ATM"),
            meta=meta,
        )
        setattr(self, emitted_attr, int(getattr(self, emitted_attr)) + 1)
        if exec_mode == "shadow":
            setattr(self, shadow_attr, int(getattr(self, shadow_attr)) + 1)
        self.signals.append(sig)
        logger.info(
            "%s signal %s %s %s arm=%s fo=%.4f exec=%s contract=%s",
            log_tag,
            date,
            sym,
            direction,
            alert.arm,
            float(alert.fav_from_open),
            exec_mode,
            pick.ticker,
        )
        if self.on_signal:
            self.on_signal(sig)
        return sig

    def _emit_am_pulse(self, alert: Any) -> ScannerSignal | None:
        """Backward-compatible emitter for the original AM lane."""
        return self._emit_am_pulse_lane("am_pulse", alert)

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
        delay = int(trade.get("bar_availability_delay_seconds", 0) or 0)
        use_confirm, confirm_bars_n, confirm_mode = self._entry_confirm_cfg(date)
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
            if not ok_c or confirm_ft is None:
                logger.info(
                    "HUNT_CONFIRM_BLOCK %s %s %s bars=%s mode=%s",
                    date,
                    symbol,
                    direction,
                    confirm_bars_n,
                    confirm_mode,
                )
                return None
            entry_ts = to_ny(confirm_ft) + pd.Timedelta(seconds=delay)

        asof = to_ny(h.get("_asof") or entry_ts)
        status, entry_ts, _ = self._gate_path_confirm(
            route="hunt",
            symbol=symbol,
            date=date,
            direction=direction,
            feature_ts=feature_ts,
            entry_ts=entry_ts,
            asof_ts=asof,
            stash={**h, "spot": float(spot), "delay": delay},
        )
        if status in {"pending", "block"}:
            return None

        return self._emit_hunt_resolved(
            {
                **h,
                "symbol": symbol,
                "dir": direction,
                "date": date,
                "feature_ts": feature_ts,
                "entry_ts": entry_ts,
                "spot": float(spot),
                "delay": delay,
            }
        )

    def _emit_hunt_resolved(self, h: dict[str, Any]) -> ScannerSignal | None:
        """Emit Hunt after entry_confirm + path confirm have resolved."""
        if self.watchdog is not None and self.watchdog.hunt_budget_remaining() <= 0:
            self.n_hunt_budget_skip += 1
            return None
        symbol = str(h["symbol"])
        direction = str(h.get("dir") or h.get("direction") or "").upper()
        date = str(h["date"])
        feature_ts = to_ny(h["feature_ts"])
        entry_ts = to_ny(h["entry_ts"])
        spot = h.get("spot")
        if spot is None:
            spot = self._spot_at(symbol, date, feature_ts)
        if spot is None or float(spot) <= 0:
            logger.warning("HUNT skip %s %s %s: no spot", date, symbol, direction)
            return None
        trade = self.profile.get("trade") or {}
        delay = int(h.get("delay") if h.get("delay") is not None else trade.get("bar_availability_delay_seconds", 0) or 0)

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
        if pick.ticker is None:
            logger.info(
                "HUNT skip %s %s %s: no contract src=%s",
                date,
                symbol,
                direction,
                pick.source,
            )
            return None
        fill = self.profile.get("fill") or {}
        hold_minutes = int(trade.get("hold_minutes", 30))
        sl_mult = float(trade.get("sl_mult", 0.4))
        tp_mult = float(trade.get("tp_mult", 1.6))
        hunt_pos_frac: float | None = None
        if self.watchdog is not None:
            from maga7.common.watchdog import hunt_trade_overrides

            hov = hunt_trade_overrides(self.watchdog.cfg)
            if hov.get("hold_minutes") is not None:
                hold_minutes = int(hov["hold_minutes"])
            if hov.get("sl_mult") is not None:
                sl_mult = float(hov["sl_mult"])
            if hov.get("tp_mult") is not None:
                tp_mult = float(hov["tp_mult"])
            if hov.get("position_frac") is not None:
                hunt_pos_frac = float(hov["position_frac"])

        det = str(h.get("detector") or "")
        wd_reason = f"hunt:{det}" if det else f"hunt:{self._watchdog_reason}"
        path_reason = h.get("path_confirm_reason")
        meta = {
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
            "stock_path_confirm_reason": path_reason,
        }
        if hunt_pos_frac is not None:
            meta["position_frac"] = float(hunt_pos_frac)
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
            meta=meta,
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
        peer_n: int | None = None
        from maga7.common.range_stall_gate import parse_range_stall_gate

        rs_cfg = parse_range_stall_gate(trade.get("range_stall_gate"))
        need_peer_for_min = (not skip_peer) and peer_min is not None and int(peer_min) > 0
        need_peer_for_hunt_rs = bool(rs_cfg.enabled and rs_cfg.hunt_peer_align)
        if need_peer_for_min or need_peer_for_hunt_rs:
            peer_n = self._peer_align_n(direction, date=date, feature_ts=feature_ts)
            sig.meta["peer_align_n"] = peer_n
            if need_peer_for_min and peer_n < int(peer_min):
                self.n_peer_block += 1
                logger.info("HUNT_PEER_BLOCK %s %s %s", date, symbol, direction)
                return None

        rs_asof = entry_ts
        asof_mode = str(getattr(rs_cfg, "hunt_asof", None) or "").strip()
        if rs_cfg.enabled and rs_cfg.hunt_peer_align and asof_mode:
            tod = asof_mode
            if asof_mode in {"signal_deadline", "deadline", "wash_end"}:
                tod = str(getattr(wd_cfg, "hunter_signal_deadline", "10:00") or "10:00")
            try:
                floor = pd.Timestamp(f"{date} {tod}", tz=to_ny(entry_ts).tz)
                if to_ny(rs_asof) < floor:
                    rs_asof = floor
            except Exception:
                pass
        allow_rs, rs_scale, rs_meta = self._entry_morph_range_stall(
            symbol=symbol,
            date=date,
            direction=direction,
            entry_ts=rs_asof,
            peer_n=peer_n if need_peer_for_hunt_rs else None,
        )
        if not allow_rs:
            logger.info(
                "HUNT_RANGE_STALL_BLOCK %s %s %s reason=%s",
                date,
                symbol,
                direction,
                rs_meta.get("range_stall_reason"),
            )
            return None
        sig.meta.update(rs_meta)
        if abs(float(rs_scale) - 1.0) > 1e-12:
            base_scale = float(sig.meta.get("regime_size_scale", 1.0) or 1.0)
            sig.meta["regime_size_scale"] = base_scale * float(rs_scale)
            sig.meta["entry_morph_range_scale"] = float(rs_scale)

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
        plan = (meta or {}).get("event_plan")
        if plan is not None and hasattr(plan, "symbol_days"):
            self.event_symbol_blackout = {
                str(d): {str(s).upper() for s in syms}
                for d, syms in (plan.symbol_days or {}).items()
            }
        else:
            raw = (meta or {}).get("symbol_blackout") or {}
            self.event_symbol_blackout = {
                str(d): {str(s).upper() for s in (syms or [])}
                for d, syms in raw.items()
            }

    def is_event_blackout(
        self, date: str | None = None, *, symbol: str | None = None
    ) -> bool:
        """Full-day block, or symbol-scoped when ``symbol`` is set."""
        d = date or self.current_date
        if not d:
            return False
        if str(d) in self.event_blackout:
            return True
        if symbol:
            return str(symbol).upper() in (
                self.event_symbol_blackout.get(str(d)) or set()
            )
        return False

    def on_stock_second(self, symbol: str, tick: dict[str, Any]) -> ScannerSignal | None:
        """Ingest 1s (or trade) print → maybe complete a 1m bar → Rule-A / Hunt."""
        if self.minute_agg is None:
            self.minute_agg = MultiSymbolMinuteAgg(self.profile["symbols"], rth_only=True)
        ts = to_ny(tick["timestamp"])
        date = ts.strftime("%Y-%m-%d")
        self._roll_day(date)
        # Even without a completed 1m bar, Hunt / path-confirm may become due.
        self.drain_hunts(ts)
        self.drain_path_confirms(ts)
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
        if self.pending_path:
            lasts = [to_ny(p["entry_ts"]) for p in self.pending_path]
            # Flush path waits at deadline so timeout_allow/block can resolve.
            for p in self.pending_path:
                cfg = p.get("path_cfg") or {}
                mw = int(cfg.get("max_wait_seconds", 300) or 300)
                lasts.append(to_ny(p["entry_ts"]) + pd.Timedelta(seconds=mw))
            last_p = max(lasts)
            for ps in self.drain_path_confirms(last_p):
                if ps not in out:
                    out.append(ps)
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
        clock_ts = ts
        date = feature_ts.strftime("%Y-%m-%d")
        self._roll_day(date)
        # Live/research: keep growing stock_by so Watchdog can arm after wash window.
        self._append_stock_bar(symbol, bar)
        self._maybe_refresh_watchdog(feature_ts)
        # Hunt / path-confirm first (time-driven); may emit via on_signal.
        self.drain_hunts(ts)
        self.drain_path_confirms(ts)
        # Independent AM lanes: each detector owns its trigger budget/runtime.
        self._feed_am_pulse_bar(symbol, bar)
        self._feed_am_pulse_lane_bar("am_pulse_extension", symbol, bar)
        self.drain_am_pulse(ts)
        self.drain_am_pulse_extension(ts)

        st = self.states.get(symbol)
        if st is None:
            return None
        fire = st.on_bar(bar)
        if fire is None:
            return None
        if self.is_event_blackout(date, symbol=symbol):
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

        # Offline-parity morph gates at feature clock — before TopK seat reserve.
        allow_m, morph_scale, morph_meta = self._entry_morph_feature_gates(
            symbol=symbol,
            date=date,
            direction=direction,
            feature_ts=feature_ts,
        )
        if not allow_m:
            return None

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

        already = str(symbol) in self.day_topk_syms or any(
            f.symbol == symbol for f in self.day_fires
        )
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
        # Earliest-TopK: reserve seat at first Rule-A, before confirm/regime/peer
        # (offline top2 is fixed; confirm fails must not free the seat for later names).
        if not already and not use_reentry:
            if len(self.day_topk_syms) >= self._topk():
                return None
            self.day_topk_syms.add(str(symbol))
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
        use_confirm, confirm_bars_n, confirm_mode = self._entry_confirm_cfg(date)
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

        # Path confirm uses the scan clock (bar available_ts), not the fill clock.
        status, ts, path_reason = self._gate_path_confirm(
            route="baseline",
            symbol=symbol,
            date=date,
            direction=direction,
            feature_ts=feature_ts,
            entry_ts=ts,
            asof_ts=clock_ts,
            stash={
                "already": already,
                "use_reentry": use_reentry,
                "spot": spot,
                "money": money,
                "delay": delay,
                "bar_source": bar.get("bar_source", "1m"),
                "morph_size_scale": morph_scale,
                "morph_meta": morph_meta,
            },
        )
        if status in {"pending", "block"}:
            return None

        return self._emit_topk_resolved(
            {
                "symbol": symbol,
                "direction": direction,
                "date": date,
                "feature_ts": feature_ts,
                "entry_ts": ts,
                "spot": spot,
                "already": already,
                "use_reentry": use_reentry,
                "money": money,
                "delay": delay,
                "bar_source": bar.get("bar_source", "1m"),
                "path_confirm_reason": path_reason,
                "morph_size_scale": morph_scale,
                "morph_meta": morph_meta,
            }
        )

    def _emit_topk_resolved(self, ctx: dict[str, Any]) -> ScannerSignal | None:
        """Emit TopK after entry_confirm + path confirm have resolved."""
        symbol = str(ctx["symbol"])
        direction = str(ctx["direction"]).upper()
        date = str(ctx["date"])
        feature_ts = to_ny(ctx["feature_ts"])
        ts = to_ny(ctx["entry_ts"])
        spot = float(ctx["spot"])
        already = bool(ctx.get("already"))
        use_reentry = bool(ctx.get("use_reentry"))
        trade = self.profile.get("trade") or {}
        money = str(ctx.get("money") or trade.get("moneyness", "ATM"))
        delay = int(
            ctx["delay"]
            if ctx.get("delay") is not None
            else trade.get("bar_availability_delay_seconds", 0) or 0
        )
        path_reason = ctx.get("path_confirm_reason")
        morph_scale = float(ctx.get("morph_size_scale") or 1.0)
        morph_meta = dict(ctx.get("morph_meta") or {})

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
                "bar_source": ctx.get("bar_source", "1m"),
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
                "stock_path_confirm_reason": path_reason,
                **{k: v for k, v in morph_meta.items() if k not in {"peer_align_n_gate"}},
                "entry_morph_feature_scale": morph_scale,
            },
        )
        if not already:
            self.day_fires.append(sig)

        r_scale = 1.0
        if self.regime_gate is not None:
            dec = self.regime_gate.check(direction, feature_ts)
            sig.meta["regime_reason"] = getattr(dec, "reason", None)
            r_scale = float(getattr(dec, "size_scale", 1.0) or 1.0)
            if not dec.allow:
                self.n_regime_block += 1
                return None

        sig_cfg = self.profile.get("signal") or {}
        peer_min = sig_cfg.get("peer_align_min")
        peer_n: int | None = None
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
        elif morph_meta.get("peer_align_n_gate") is not None:
            peer_n = int(morph_meta["peer_align_n_gate"])

        allow_rs, rs_scale, rs_meta = self._entry_morph_range_stall(
            symbol=symbol,
            date=date,
            direction=direction,
            entry_ts=ts,
            peer_n=peer_n,
        )
        if not allow_rs:
            return None
        sig.meta.update(rs_meta)
        sig.meta["entry_morph_range_scale"] = float(rs_scale)
        # OMS reads regime_size_scale; fold offline morph scales into the same knob.
        combined = float(r_scale) * float(morph_scale) * float(rs_scale)
        sig.meta["regime_size_scale"] = combined

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

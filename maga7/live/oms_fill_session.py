"""Shared OMS fill helpers: mf_flip exit + concurrent sizing aligned with offline."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from maga7.common.fills import FillSpec
from maga7.common.position_size import apply_size_scale, resolve_size_frac
from maga7.common.replay import load_quotes, month_list, path_for_ticker, simulate_trade, to_ny
from maga7.common.signals import attach_mf_features, load_stock_month_files
from maga7.live.redis_quotes import RedisQuoteBook
from maga7.live.scanner import ScannerSignal


@dataclass
class PendingRedisSignal:
    sig: ScannerSignal
    hold_minutes: int


class QuoteSimSession:
    """Cache quotes + stock mf frames for OMS shadow fills.

    When ``prefer_redis=True``, option paths come from ``quote_book`` (fed by
    fused_market_stream). Signals are deferred until the book covers the hold
    window or a causal TP/SL/mf exit becomes observable.
    """

    def __init__(self, profile: dict[str, Any], *, prefer_redis: bool = False):
        self.profile = profile
        self.paths = profile["_paths"]
        self.trade = profile.get("trade") or {}
        self.fill = FillSpec(
            entry_frac=float(profile["fill"].get("entry_frac", 0.8)),
            exit_frac=float(profile["fill"].get("exit_frac", 0.8)),
        )
        self.prefer_redis = bool(prefer_redis)
        self.quote_book = RedisQuoteBook()
        self.quote_cache: dict[tuple[str, str], pd.DataFrame] = {}
        self.stock_by: dict[str, pd.DataFrame] = {}
        self.open_until: dict[str, Any] = {}
        self.pending: list[PendingRedisSignal] = []
        self.n_path_redis = 0
        self.n_path_disk = 0
        self._load_stock()

    def _load_stock(self) -> None:
        start = self.profile["date_range"]["start"]
        end = self.profile["date_range"]["end"]
        months = month_list(start, end)
        sig = self.profile.get("signal") or {}
        for sym in self.profile["symbols"]:
            raw = load_stock_month_files(self.paths["stock_root"], sym, months)
            if raw.empty:
                continue
            feat = attach_mf_features(
                raw[(raw["date"] >= start) & (raw["date"] <= end)],
                mf_window=int(sig.get("mf_window", 10)),
                vol_ma_window=int(sig.get("vol_ma_window", 20)),
            )
            self.stock_by[sym] = feat

    def ingest_redis_contracts(self, symbol: str, ts: float, contracts: list[dict[str, Any]]) -> None:
        for c in contracts or []:
            if not isinstance(c, dict):
                continue
            local = str(c.get("localSymbol") or c.get("ticker") or "")
            bid, ask = c.get("bid"), c.get("ask")
            if local and bid is not None and ask is not None:
                self.quote_book.update(symbol, local, ts, bid=float(bid), ask=float(ask))

    def get_path_disk(self, symbol: str, date: str, ticker: str) -> pd.DataFrame | None:
        k = (symbol, date)
        if k not in self.quote_cache:
            self.quote_cache[k] = load_quotes(self.paths["quote_1s_root"], symbol, date)
        qdf = self.quote_cache[k]
        if not ticker:
            return qdf
        return path_for_ticker(qdf, ticker)

    def get_path(
        self, symbol: str, date: str, ticker: str, *, allow_disk_fallback: bool = True
    ) -> tuple[pd.DataFrame | None, str]:
        if self.prefer_redis:
            rdf = self.quote_book.path(symbol, ticker)
            if rdf is not None and not rdf.empty:
                self.n_path_redis += 1
                return rdf, "redis"
            if not allow_disk_fallback:
                return None, "none"
        path = self.get_path_disk(symbol, date, ticker)
        if path is not None and not path.empty:
            self.n_path_disk += 1
            return path, "disk"
        return None, "none"

    def simulate_on_path(self, sig: ScannerSignal, path: pd.DataFrame):
        sdf = self.stock_by.get(sig.symbol)
        stock_day = None if sdf is None or sdf.empty else sdf[sdf["date"] == sig.date]
        return simulate_trade(
            path,
            sig.sig_ts,
            fill=self.fill,
            tp_mult=float(self.trade.get("tp_mult", 1.6)),
            sl_mult=float(self.trade.get("sl_mult", 0.4)),
            hold_minutes=int(self.trade.get("hold_minutes", 30)),
            direction=sig.direction,
            stock_day=stock_day,
            exit_mode=str(self.trade.get("exit_mode") or "none"),
            exit_mf_grace_seconds=int(self.trade.get("exit_mf_grace_seconds", 60)),
            exit_min_hold_minutes=self.trade.get("exit_min_hold_minutes"),
            mtm_floor_ret=self.trade.get("mtm_floor_ret"),
            flow_cum_floor=self.trade.get("flow_cum_floor"),
            stock_bar_delay_seconds=int(
                self.trade.get("bar_availability_delay_seconds", 0) or 0
            ),
            trail_activate=self.trade.get("trail_activate"),
            trail_dd=self.trade.get("trail_dd"),
            hold_extend_minutes=self.trade.get("hold_extend_minutes"),
            hold_extend_mtm_min=self.trade.get("hold_extend_mtm_min"),
            hold_extend_require_mf=bool(self.trade.get("hold_extend_require_mf", True)),
        )

    def simulate_signal(self, sig: ScannerSignal):
        if not sig.contract:
            return None
        path, _src = self.get_path(sig.symbol, sig.date, sig.contract, allow_disk_fallback=True)
        if path is None or path.empty:
            return None
        return self.simulate_on_path(sig, path)

    def size_frac_for(
        self,
        symbol: str,
        entry_ts,
        *,
        regime_scale: float = 1.0,
    ) -> tuple[float, bool, int]:
        top_k = max(int(self.profile["signal"].get("top_k", 2)), 1)
        size, _, n_conc, allow, _ = resolve_size_frac(
            self.trade,
            top_k=top_k,
            open_until=self.open_until,
            symbol=symbol,
            entry_ts=to_ny(entry_ts),
        )
        if allow:
            size = apply_size_scale(size, regime_scale)
            if float(regime_scale) <= 0.0:
                return 0.0, False, n_conc
        return size, allow, n_conc

    def mark_closed(self, symbol: str, exit_ts) -> None:
        self.open_until[symbol] = to_ny(exit_ts)

    def hold_minutes(self) -> int:
        return int(self.trade.get("hold_minutes", 30))

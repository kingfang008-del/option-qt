"""Shared OMS fill helpers: mf_flip exit + concurrent sizing aligned with offline."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from maga7.common.fills import FillSpec
from maga7.common.option_trades import (
    load_option_trades,
    path_for_ticker_trades,
    trade_toxic_from_trade,
)
from maga7.common.position_size import apply_size_scale, resolve_size_frac
from maga7.common.replay import load_quotes, month_list, path_for_ticker, simulate_trade, to_ny
from maga7.common.signals import attach_mf_features, load_stock_month_files, resolve_mf_fast_window
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
        self.fills = FillSpec(
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
        self.trade_day_cache: dict[tuple[str, str], pd.DataFrame | None] = {}
        self._ttox = trade_toxic_from_trade(self.trade)
        self._load_stock()

    def _load_stock(self) -> None:
        start = self.profile["date_range"]["start"]
        end = self.profile["date_range"]["end"]
        sig = self.profile.get("signal") or {}
        wd = self.profile.get("watchdog") or {}
        tcn = self.profile.get("tcn_gate") or sig.get("tcn_gate") or {}
        mf_idio_on = str(sig.get("mf_idio_mode") or "off").strip().lower() not in {
            "",
            "off",
            "none",
            "false",
            "0",
        }
        load_start = start
        if mf_idio_on or wd.get("enabled") or tcn.get("enabled"):
            lookback_days = max(14, int(sig.get("mf_idio_beta_days", 5) or 5) * 3)
            load_start = (pd.Timestamp(start) - pd.Timedelta(days=lookback_days)).strftime(
                "%Y-%m-%d"
            )
        months = month_list(load_start, end)
        load_syms = list(
            dict.fromkeys(
                list(self.profile.get("symbols") or [])
                + list(sig.get("peer_symbols") or [])
                + ["QQQ"]
            )
        )
        for sym in load_syms:
            raw = load_stock_month_files(self.paths["stock_root"], sym, months)
            if raw.empty:
                continue
            sliced = raw[(raw["date"] >= load_start) & (raw["date"] <= end)]
            if sliced.empty:
                continue
            feat = attach_mf_features(
                sliced,
                mf_window=int(sig.get("mf_window", 10)),
                vol_ma_window=int(sig.get("vol_ma_window", 20)),
                mf_fast_window=resolve_mf_fast_window(sig),
            )
            self.stock_by[sym] = feat

    def _get_trade_path(self, symbol: str, date: str, ticker: str) -> pd.DataFrame | None:
        if not self._ttox.enabled or not ticker:
            return None
        root = self.paths.get("option_trades_root")
        if root is None:
            return None
        k = (symbol, date)
        if k not in self.trade_day_cache:
            try:
                self.trade_day_cache[k] = load_option_trades(root, symbol, date)
            except Exception:
                self.trade_day_cache[k] = None
        return path_for_ticker_trades(self.trade_day_cache[k], ticker)

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
        from maga7.common.delta_time_stop import (
            StockRevExitConfig,
            delta_time_stop_from_trade,
            roi_time_stop_from_trade,
            stock_rev_applies_to_route,
            stock_rev_day_should_arm,
            stock_rev_exit_from_trade,
        )
        from maga7.common.hold_watchdog import hold_watchdog_from_trade
        from maga7.common.ladder_active import (
            LadderActiveConfig,
            ladder_active_from_trade,
            ladder_day_should_arm,
        )
        from maga7.common.path_fast_pack import (
            apply_path_fast_pack_overrides,
            path_fast_pack_day_should_arm,
            path_fast_pack_from_trade,
        )

        sdf = self.stock_by.get(sig.symbol)
        stock_day = None
        if sdf is not None and not getattr(sdf, "empty", True):
            stock_day = sdf[sdf["date"].astype(str) == str(sig.date)]
        hwd = hold_watchdog_from_trade(self.trade)
        qqq_day = None
        if hwd.enabled:
            qdf = self.stock_by.get("QQQ")
            if qdf is not None and not getattr(qdf, "empty", True):
                qqq_day = qdf[qdf["date"].astype(str) == str(sig.date)]
        hold_minutes = int(sig.meta.get("hold_minutes") or self.trade.get("hold_minutes", 30))
        tp_mult = float(sig.meta.get("tp_mult") or self.trade.get("tp_mult", 1.6))
        sl_mult = float(sig.meta.get("sl_mult") or self.trade.get("sl_mult", 0.4))
        tpath = self._get_trade_path(sig.symbol, sig.date, str(sig.contract or ""))
        lac = ladder_active_from_trade(self.trade)
        srev = stock_rev_exit_from_trade(self.trade)
        exit_mode = str(self.trade.get("exit_mode") or "none")
        prev = {}
        wd = self.profile.get("watchdog") if isinstance(self.profile.get("watchdog"), dict) else {}
        if isinstance(wd.get("prevention"), dict):
            prev = wd["prevention"]
        wash_kw = dict(
            asof=str(prev.get("asof") or wd.get("asof") or "10:30"),
            washout_breadth_min=int(prev.get("washout_breadth_min", 3) or 3),
            wash_drop_min=float(prev.get("wash_drop_min", 0.008) or 0.008),
            frac_above_min=float(prev.get("frac_above_min", 0.35) or 0.35),
            frac_above_max=float(prev.get("frac_above_max", 0.70) or 0.70),
        )
        if lac.enabled:
            arm = ladder_day_should_arm(
                lac,
                date=str(sig.date),
                stock_by=self.stock_by,
                qqq_df=self.stock_by.get("QQQ"),
                symbols=list(self.profile.get("symbols") or []),
                **wash_kw,
            )
            if arm:
                exit_mode = "ladder_active"
            else:
                exit_mode = str(self.trade.get("ladder_fallback_exit_mode") or "hold_extend")
                lac = LadderActiveConfig(enabled=False)
                hold_minutes = int(self.trade.get("hold_minutes", 30))
        if srev.enabled:
            if not stock_rev_day_should_arm(
                srev,
                date=str(sig.date),
                stock_by=self.stock_by,
                qqq_df=self.stock_by.get("QQQ"),
                symbols=list(self.profile.get("symbols") or []),
                **wash_kw,
            ):
                srev = StockRevExitConfig(enabled=False)
            elif not stock_rev_applies_to_route(
                srev, str((sig.meta or {}).get("route") or "baseline")
            ):
                srev = StockRevExitConfig(enabled=False)
        trail_activate = self.trade.get("trail_activate")
        trail_dd = self.trade.get("trail_dd")
        hold_extend_minutes = self.trade.get("hold_extend_minutes")
        fast = path_fast_pack_from_trade(self.trade)
        if fast.enabled and path_fast_pack_day_should_arm(
            fast,
            date=str(sig.date),
            stock_by=self.stock_by,
            qqq_df=self.stock_by.get("QQQ"),
            symbols=list(self.profile.get("symbols") or []),
            **wash_kw,
        ):
            ov = apply_path_fast_pack_overrides(
                hold_minutes=hold_minutes,
                trail_activate=trail_activate,
                trail_dd=trail_dd,
                stock_rev=srev,
                pack=fast,
            )
            hold_minutes = int(ov["hold_minutes"])
            trail_activate = ov["trail_activate"]
            trail_dd = ov["trail_dd"]
            srev = ov["stock_rev_exit"]
            hold_extend_minutes = ov.get("hold_extend_minutes")
        return simulate_trade(
            path,
            sig.sig_ts,
            fill=self.fills,
            tp_mult=tp_mult,
            sl_mult=sl_mult,
            hold_minutes=hold_minutes,
            direction=sig.direction,
            stock_day=stock_day,
            exit_mode=exit_mode,
            exit_mf_grace_seconds=int(self.trade.get("exit_mf_grace_seconds", 60)),
            exit_min_hold_minutes=self.trade.get("exit_min_hold_minutes"),
            mtm_floor_ret=self.trade.get("mtm_floor_ret"),
            flow_cum_floor=self.trade.get("flow_cum_floor"),
            stock_bar_delay_seconds=int(
                self.trade.get("bar_availability_delay_seconds", 0) or 0
            ),
            trail_activate=trail_activate,
            trail_dd=trail_dd,
            hold_extend_minutes=hold_extend_minutes,
            hold_extend_mtm_min=self.trade.get("hold_extend_mtm_min"),
            hold_extend_require_mf=bool(self.trade.get("hold_extend_require_mf", True)),
            hold_extend_require_stock=bool(self.trade.get("hold_extend_require_stock", False)),
            hold_extend_stock_min=float(self.trade.get("hold_extend_stock_min", 0.0) or 0.0),
            hold_extend_min_peak_mfe=self.trade.get("hold_extend_min_peak_mfe"),
            hold_extend_max_qqq_adverse=self.trade.get("hold_extend_max_qqq_adverse"),
            hold_watchdog=hwd,
            qqq_day=qqq_day,
            trade_path=tpath,
            trade_toxic=self._ttox,
            delta_time_stop=delta_time_stop_from_trade(self.trade),
            roi_time_stop=roi_time_stop_from_trade(self.trade),
            stock_rev_exit=srev,
            ladder_active=lac,
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
        from maga7.common.ladder_active import ladder_active_from_trade

        lac = ladder_active_from_trade(self.trade)
        if lac.enabled:
            # Pending resolve clock in whole minutes (ceil).
            return max(1, int((int(lac.max_hold_seconds) + 59) // 60))
        return int(self.trade.get("hold_minutes", 30))

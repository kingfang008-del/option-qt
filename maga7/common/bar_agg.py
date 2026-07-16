"""Causal 1s stock ticks → RTH 1m OHLCV bars for Mag7 Rule-A.

Signal clock stays on 1m (mf10 / streak / vol_z). Seconds are only the
ingest source; do not evaluate Rule-A on second-level features.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

NY = "America/New_York"
RTH_START = pd.Timestamp("09:30").time()
RTH_END = pd.Timestamp("16:00").time()


def to_ny_ts(ts) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        return t.tz_localize(NY)
    return t.tz_convert(NY)


def minute_floor(ts: pd.Timestamp) -> pd.Timestamp:
    t = to_ny_ts(ts)
    return t.floor("min")


def in_rth(ts: pd.Timestamp) -> bool:
    t = to_ny_ts(ts).time()
    return RTH_START <= t < RTH_END


@dataclass
class MinuteBarAggregator:
    """Accumulate 1s OHLCV; emit a completed left-labeled 1m bar on minute roll."""

    symbol: str
    rth_only: bool = True
    cur_minute: pd.Timestamp | None = None
    open: float | None = None
    high: float | None = None
    low: float | None = None
    close: float | None = None
    volume: float = 0.0

    def _emit(self) -> dict[str, Any] | None:
        if self.cur_minute is None or self.open is None:
            return None
        return {
            "symbol": self.symbol,
            "timestamp": self.cur_minute,
            "open": float(self.open),
            "high": float(self.high),
            "low": float(self.low),
            "close": float(self.close),
            "volume": float(self.volume),
        }

    def _start(self, minute: pd.Timestamp, o: float, h: float, l: float, c: float, v: float) -> None:
        self.cur_minute = minute
        self.open = o
        self.high = h
        self.low = l
        self.close = c
        self.volume = v

    def on_second(
        self,
        ts,
        *,
        open: float | None = None,
        high: float | None = None,
        low: float | None = None,
        close: float,
        volume: float = 0.0,
    ) -> dict[str, Any] | None:
        """Ingest one second (or trade) print. Return completed prior 1m bar, else None."""
        t = to_ny_ts(ts)
        if self.rth_only and not in_rth(t):
            # Outside RTH: flush open RTH bar if we rolled past 16:00, else ignore.
            if self.cur_minute is not None and t.time() >= RTH_END:
                done = self._emit()
                self.reset()
                return done
            return None

        o = float(open if open is not None else close)
        h = float(high if high is not None else close)
        l = float(low if low is not None else close)
        c = float(close)
        v = float(volume)
        minute = minute_floor(t)

        if self.cur_minute is None:
            self._start(minute, o, h, l, c, v)
            return None

        if minute < self.cur_minute:
            # Out-of-order late print — fold into current minute if same calendar minute
            # already closed; otherwise ignore (causal path expects sorted feed).
            return None

        if minute == self.cur_minute:
            self.high = max(self.high, h)  # type: ignore[arg-type]
            self.low = min(self.low, l)  # type: ignore[arg-type]
            self.close = c
            self.volume += v
            return None

        # Minute rolled forward → emit completed bar, start new.
        done = self._emit()
        self._start(minute, o, h, l, c, v)
        return done

    def flush(self) -> dict[str, Any] | None:
        done = self._emit()
        self.reset()
        return done

    def reset(self) -> None:
        self.cur_minute = None
        self.open = self.high = self.low = self.close = None
        self.volume = 0.0


@dataclass
class MultiSymbolMinuteAgg:
    """Per-symbol aggregators; feed chronologically across symbols."""

    symbols: Iterable[str]
    rth_only: bool = True
    aggs: dict[str, MinuteBarAggregator] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for s in self.symbols:
            self.aggs[s] = MinuteBarAggregator(s, rth_only=self.rth_only)

    def on_second(self, symbol: str, row: dict[str, Any]) -> dict[str, Any] | None:
        agg = self.aggs.get(symbol)
        if agg is None:
            agg = MinuteBarAggregator(symbol, rth_only=self.rth_only)
            self.aggs[symbol] = agg
        return agg.on_second(
            row["timestamp"],
            open=row.get("open"),
            high=row.get("high"),
            low=row.get("low"),
            close=float(row["close"]),
            volume=float(row.get("volume") or 0.0),
        )

    def flush_all(self) -> list[dict[str, Any]]:
        out = []
        for sym in sorted(self.aggs):
            bar = self.aggs[sym].flush()
            if bar is not None:
                out.append(bar)
        return out


def load_stock_1s_day(stock_1s_root: Path | str, symbol: str, date: str) -> pd.DataFrame:
    """Load one day of stock 1s parquet: `{root}/{SYM}/{SYM}_{date}.parquet`."""
    root = Path(stock_1s_root)
    path = root / symbol / f"{symbol}_{date}.parquet"
    if not path.is_file():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    if "timestamp" not in df.columns and "ts" in df.columns:
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["ts"], unit="s", utc=True).dt.tz_convert(NY)
    else:
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        if getattr(df["timestamp"].dt, "tz", None) is None:
            df["timestamp"] = df["timestamp"].dt.tz_localize(NY)
        else:
            df["timestamp"] = df["timestamp"].dt.tz_convert(NY)
    keep = [c for c in ("timestamp", "open", "high", "low", "close", "volume") if c in df.columns]
    df = df[keep].sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    return df.reset_index(drop=True)


def aggregate_1s_to_1m(df: pd.DataFrame, *, symbol: str = "", rth_only: bool = True) -> pd.DataFrame:
    """Batch-resample a day (or multi-day) 1s frame to left-labeled 1m bars."""
    if df.empty:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    agg = MinuteBarAggregator(symbol or "X", rth_only=rth_only)
    bars: list[dict[str, Any]] = []
    has_o = "open" in df.columns
    has_h = "high" in df.columns
    has_l = "low" in df.columns
    has_v = "volume" in df.columns
    for r in df.itertuples(index=False):
        c = float(r.close)
        done = agg.on_second(
            r.timestamp,
            open=float(r.open) if has_o else c,
            high=float(r.high) if has_h else c,
            low=float(r.low) if has_l else c,
            close=c,
            volume=float(r.volume) if has_v else 0.0,
        )
        if done is not None:
            bars.append(done)
    last = agg.flush()
    if last is not None:
        bars.append(last)
    if not bars:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    out = pd.DataFrame(bars)
    if "symbol" in out.columns:
        out = out.drop(columns=["symbol"])
    return out

"""Redis-fed option quote book for Mag7 S5 OMS."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from maga7.common.replay import to_ny

NY = "America/New_York"


def _norm_ticker(ticker: str) -> str:
    return str(ticker or "").replace("O:", "").strip()


@dataclass
class RedisQuoteBook:
    """Accumulate option prints from fused_market_stream."""

    # (symbol, localSymbol) -> rows
    _rows: dict[tuple[str, str], list[dict[str, Any]]] = field(default_factory=dict)
    n_updates: int = 0

    def update(
        self,
        symbol: str,
        ticker: str,
        ts: float | pd.Timestamp,
        *,
        bid: float,
        ask: float,
    ) -> None:
        ticker = _norm_ticker(ticker)
        if not symbol or not ticker:
            return
        if not (np.isfinite(bid) and np.isfinite(ask) and ask >= bid > 0):
            return
        if isinstance(ts, (int, float)) and not isinstance(ts, bool):
            t = pd.Timestamp(float(ts), unit="s", tz="UTC").tz_convert(NY)
        else:
            t = to_ny(ts)
        key = (symbol, ticker)
        bucket = self._rows.setdefault(key, [])
        # replace same-second print
        if bucket and int(bucket[-1]["timestamp"].timestamp()) == int(t.timestamp()):
            bucket[-1] = {"timestamp": t, "bid": float(bid), "ask": float(ask)}
        else:
            bucket.append({"timestamp": t, "bid": float(bid), "ask": float(ask)})
        self.n_updates += 1

    def path(self, symbol: str, ticker: str) -> pd.DataFrame | None:
        rows = self._rows.get((symbol, _norm_ticker(ticker)))
        if not rows:
            return None
        df = pd.DataFrame(rows)
        return df.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)

    def quote_at_or_after(
        self, symbol: str, ticker: str, ts
    ) -> tuple[float, float, pd.Timestamp] | None:
        df = self.path(symbol, ticker)
        if df is None or df.empty:
            return None
        ts = to_ny(ts)
        after = df[df["timestamp"] >= ts]
        if after.empty:
            return None
        r = after.iloc[0]
        return float(r["bid"]), float(r["ask"]), to_ny(r["timestamp"])

    def max_ts(self, symbol: str, ticker: str) -> pd.Timestamp | None:
        df = self.path(symbol, ticker)
        if df is None or df.empty:
            return None
        return to_ny(df["timestamp"].iloc[-1])

    def clear_symbol_day(self, symbol: str) -> None:
        drop = [k for k in self._rows if k[0] == symbol]
        for k in drop:
            del self._rows[k]

"""Unit tests for 1s → Ns bar aggregation."""
from __future__ import annotations

import pandas as pd

from maga7.common.bar_agg import aggregate_1s_to_1m, aggregate_1s_to_bars

NY = "America/New_York"


def _synth_1s(start: str = "2026-01-02 09:30:00", n: int = 120) -> pd.DataFrame:
    idx = pd.date_range(start, periods=n, freq="1s", tz=NY)
    close = pd.Series(range(100, 100 + n), dtype=float)
    return pd.DataFrame(
        {
            "timestamp": idx,
            "open": close,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": 1.0,
        }
    )


def test_aggregate_5s_bar_count_and_ohlc():
    raw = _synth_1s(n=60)  # one RTH minute
    bars = aggregate_1s_to_bars(raw, bar_seconds=5, symbol="QQQ")
    assert len(bars) == 12
    b0 = bars.iloc[0]
    assert float(b0["open"]) == 100.0
    assert float(b0["close"]) == 104.0  # seconds 0..4
    assert float(b0["volume"]) == 5.0
    assert float(bars.iloc[-1]["close"]) == 159.0


def test_aggregate_15s_and_60_delegate():
    raw = _synth_1s(n=60)
    b15 = aggregate_1s_to_bars(raw, bar_seconds=15)
    assert len(b15) == 4
    b60 = aggregate_1s_to_bars(raw, bar_seconds=60)
    m1 = aggregate_1s_to_1m(raw, symbol="X")
    assert len(b60) == len(m1) == 1
    assert abs(float(b60.iloc[0]["close"]) - float(m1.iloc[0]["close"])) < 1e-9

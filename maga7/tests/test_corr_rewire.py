"""Unit tests for Mag7–QQQ corr rewire gate."""
from __future__ import annotations

import numpy as np
import pandas as pd

from maga7.common.corr_rewire import corr_rewire_asof


def _synth_sym(name: str, n: int, seed: int, beta: float = 1.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2026-02-17 09:30", periods=n, freq="1min", tz="America/New_York")
    # shared factor + noise
    f = rng.normal(0, 0.001, size=n)
    noise = rng.normal(0, 0.0005, size=n)
    rets = beta * f + noise
    px = 100 * np.cumprod(1 + rets)
    return pd.DataFrame(
        {
            "timestamp": ts,
            "close": px,
            "date": ts.strftime("%Y-%m-%d"),
            "high": px * 1.001,
            "low": px * 0.999,
            "volume": rng.integers(1_000, 5_000, size=n),
        }
    )


def test_corr_rewire_triggers_on_break():
    # calm: high beta to QQQ; event: decorrelated
    n = 300
    qqq = _synth_sym("QQQ", n, seed=1, beta=1.0)
    # rebuild with shared factor access — simpler: copy qqq for calm portion
    rng = np.random.default_rng(0)
    ts = qqq["timestamp"]
    q_ret = qqq["close"].pct_change().fillna(0).to_numpy()
    # Mag7 follows QQQ for first 200 bars, then independent
    m_ret = q_ret.copy()
    m_ret[200:] = rng.normal(0, 0.0015, size=n - 200)
    m_px = 100 * np.cumprod(1 + m_ret)
    mag = qqq.copy()
    mag["close"] = m_px
    stock_by = {"QQQ": qqq, "NVDA": mag, "AAPL": mag, "MSFT": mag, "AMD": mag}
    asof = ts.iloc[-1]
    snap = corr_rewire_asof(
        stock_by,
        asof_ts=asof,
        symbols=["NVDA", "AAPL", "MSFT", "AMD"],
        event_bars=60,
        calm_bars=120,
        min_bars=30,
        rewire_min=0.15,
        scale=0.5,
    )
    assert snap.rho_event is not None and snap.rho_calm is not None
    assert snap.rewire is not None and snap.rewire >= 0.15
    assert snap.trigger
    assert snap.size_scale == 0.5


def test_corr_rewire_ok_when_stable():
    n = 300
    qqq = _synth_sym("QQQ", n, seed=2, beta=1.0)
    mag = _synth_sym("NVDA", n, seed=2, beta=1.0)  # same seed → same path
    # force identical
    mag = qqq.copy()
    stock_by = {"QQQ": qqq, "NVDA": mag, "AAPL": mag, "MSFT": mag}
    asof = qqq["timestamp"].iloc[-1]
    snap = corr_rewire_asof(
        stock_by,
        asof_ts=asof,
        symbols=["NVDA", "AAPL", "MSFT"],
        event_bars=60,
        calm_bars=120,
        rewire_min=0.25,
        rho_event_min=0.2,
    )
    assert not snap.trigger
    assert snap.size_scale == 1.0

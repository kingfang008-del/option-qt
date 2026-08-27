"""Feature gates on hold_extend / stale_cut."""
from __future__ import annotations

import numpy as np
import pandas as pd

from maga7.common.fills import FillSpec
from maga7.common.replay import simulate_trade

NY = "America/New_York"


def _path(entry=1.0, n=40, end=1.02):
    """1m quotes spanning ~40 minutes."""
    ts = pd.date_range("2026-05-01 10:00", periods=n, freq="1min", tz=NY)
    mid = np.linspace(entry, end, n)
    return pd.DataFrame(
        {
            "timestamp": ts,
            "bid": mid * 0.99,
            "ask": mid * 1.01,
            "ticker": "TEST",
        }
    )


def _stock(n=40, end_ret=0.01):
    ts = pd.date_range("2026-05-01 10:00", periods=n, freq="1min", tz=NY)
    px = 100 * np.linspace(1.0, 1.0 + end_ret, n)
    return pd.DataFrame(
        {
            "timestamp": ts,
            "date": "2026-05-01",
            "open": px,
            "high": px * 1.001,
            "low": px * 0.999,
            "close": px,
            "volume": 1000.0,
            "mf10": np.full(n, 0.01),
        }
    )


def test_extend_requires_stock_blocks_when_stock_red():
    path = _path(end=1.05)  # option green → would extend on MTM alone
    stock = _stock(end_ret=-0.005)  # stock red
    sim = simulate_trade(
        path,
        entry_ts=pd.Timestamp("2026-05-01 10:00", tz=NY),
        fill=FillSpec(0.5, 0.5),
        hold_minutes=10,
        direction="UP",
        stock_day=stock,
        exit_mode="hold_extend",
        hold_extend_minutes=20,
        hold_extend_mtm_min=0.0,
        hold_extend_require_mf=False,
        hold_extend_require_stock=True,
        hold_extend_stock_min=0.0,
        tp_mult=10.0,
        sl_mult=0.9,
    )
    assert sim is not None
    assert sim.reason == "T+10"  # denied extend


def test_extend_max_giveback_blocks_after_peak_fade():
    # Spike early, then largely give back before T+10.
    ts = pd.date_range("2026-05-01 10:00", periods=25, freq="1min", tz=NY)
    mid = np.array(
        [1.00, 1.05, 1.15, 1.28, 1.30]
        + list(np.linspace(1.20, 1.04, 20)),
        dtype=float,
    )
    path = pd.DataFrame(
        {
            "timestamp": ts,
            "bid": mid * 0.99,
            "ask": mid * 1.01,
            "ticker": "TEST",
        }
    )
    sim = simulate_trade(
        path,
        entry_ts=pd.Timestamp("2026-05-01 10:00", tz=NY),
        fill=FillSpec(0.5, 0.5),
        hold_minutes=10,
        direction="DN",
        exit_mode="hold_extend",
        hold_extend_minutes=20,
        hold_extend_mtm_min=0.0,
        hold_extend_require_mf=False,
        hold_extend_max_giveback=0.12,
        hold_extend_giveback_min_peak=0.15,
        tp_mult=10.0,
        sl_mult=0.5,
    )
    assert sim is not None
    assert sim.reason == "T+10"


def test_stale_cut_fires():
    # option drifts down, stock also down
    path = _path(entry=1.0, n=25, end=0.95)
    stock = _stock(n=25, end_ret=-0.004)
    sim = simulate_trade(
        path,
        entry_ts=pd.Timestamp("2026-05-01 10:00", tz=NY),
        fill=FillSpec(0.5, 0.5),
        hold_minutes=30,
        direction="UP",
        stock_day=stock,
        exit_mode="hold_extend",
        hold_extend_minutes=45,
        hold_extend_mtm_min=0.0,
        hold_extend_require_mf=False,
        stale_cut_minutes=5,
        stale_cut_mtm_max=0.0,
        stale_cut_stock_max=0.0,
        tp_mult=10.0,
        sl_mult=0.9,
    )
    assert sim is not None
    assert sim.reason == "STALE_CUT"

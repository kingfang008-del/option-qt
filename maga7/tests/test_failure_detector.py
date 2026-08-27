"""Failure Detector unit tests."""
from __future__ import annotations

import numpy as np
import pandas as pd

from maga7.common.failure_detector import (
    FailureDetectorConfig,
    evaluate_failure,
    failure_cfg_for_sleeve,
    simulate_stock_with_failure,
)
from maga7.common.smooth_trend import SmoothStockTradeConfig

NY = "America/New_York"


def test_impulse_tighter_than_smooth():
    s = failure_cfg_for_sleeve("smooth")
    i = failure_cfg_for_sleeve("impulse")
    assert i.early_mae_cut <= s.early_mae_cut
    assert i.max_eval_minutes <= s.max_eval_minutes


def test_early_mae_exits():
    cfg = FailureDetectorConfig(enabled=True, min_hold_minutes=0.5, early_mae_cut=0.003)
    action, reason = evaluate_failure(
        direction="UP",
        entry_px=100.0,
        entry_ts=pd.Timestamp("2026-01-02 10:00", tz=NY),
        now_ts=pd.Timestamp("2026-01-02 10:03", tz=NY),
        now_px=99.5,
        peak=100.0,
        trough=99.5,
        closes_since_entry=np.array([100.0, 99.7, 99.5]),
        structure_extreme=99.0,
        open_px=99.8,
        vwap=100.0,
        cfg=cfg,
    )
    assert action == "EXIT"
    assert reason == "FD_EARLY_MAE"


def test_disabled_holds():
    cfg = FailureDetectorConfig(enabled=False)
    action, reason = evaluate_failure(
        direction="UP",
        entry_px=100.0,
        entry_ts=pd.Timestamp("2026-01-02 10:00", tz=NY),
        now_ts=pd.Timestamp("2026-01-02 10:03", tz=NY),
        now_px=99.0,
        peak=100.0,
        trough=99.0,
        closes_since_entry=np.array([100.0, 99.0]),
        structure_extreme=None,
        open_px=None,
        vwap=None,
        cfg=cfg,
    )
    assert action == "HOLD"
    assert reason is None


def test_simulate_fd_cuts_fade():
    date = "2026-01-02"
    start = pd.Timestamp(f"{date} 09:30:00", tz=NY)
    ts = pd.date_range(start, periods=120, freq="1min")
    # grind up then hard fade after entry at 10:00
    px = np.concatenate(
        [np.linspace(100, 101, 30), np.linspace(101, 99.2, 90)]
    )
    day = pd.DataFrame(
        {
            "timestamp": ts,
            "date": date,
            "open": px,
            "high": px * 1.0002,
            "low": px * 0.9998,
            "close": px,
            "volume": 1000.0,
        }
    )
    et = pd.Timestamp(f"{date} 10:00:00", tz=NY)
    base = simulate_stock_with_failure(
        day,
        entry_ts=et,
        direction="UP",
        trade_cfg=SmoothStockTradeConfig(break_max_adverse=0.05, max_hold_minutes=180),
        fd_cfg=FailureDetectorConfig(enabled=False),
        date=date,
    )
    fd = simulate_stock_with_failure(
        day,
        entry_ts=et,
        direction="UP",
        trade_cfg=SmoothStockTradeConfig(break_max_adverse=0.05, max_hold_minutes=180),
        fd_cfg=FailureDetectorConfig(
            enabled=True, min_hold_minutes=1.0, early_mae_cut=0.003, max_eval_minutes=15.0
        ),
        date=date,
    )
    assert base is not None and fd is not None
    assert fd["fd_fired"] is True
    assert fd["hold_minutes"] < base["hold_minutes"]
    assert fd["ret"] > base["ret"]  # cut fade earlier → less loss

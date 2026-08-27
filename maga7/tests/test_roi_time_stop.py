"""Unit tests for V0-style option ROI time-stop rails."""
from __future__ import annotations

import pandas as pd

from maga7.common.delta_time_stop import roi_time_stop_from_trade
from maga7.common.fills import FillSpec
from maga7.common.replay import simulate_trade


def test_roi_time_stop_config():
    cfg = roi_time_stop_from_trade(
        {"roi_time_stop": {"enabled": True, "mid_mins": 15, "mid_min_roi": 0.05}}
    )
    assert cfg.enabled
    assert cfg.rails[0] == (15.0, 0.05)


def test_roi_time15_cuts_flat_option():
    ts0 = pd.Timestamp("2026-02-18 10:31:00", tz="America/New_York")
    # Option drifts mildly negative for 20 minutes — never reaches +5%.
    qrows = []
    for i in range(25):
        mid = 1.00 - 0.002 * i
        t = ts0 + pd.Timedelta(minutes=i)
        qrows.append({"timestamp": t, "bid": mid - 0.01, "ask": mid + 0.01})
    sim = simulate_trade(
        pd.DataFrame(qrows),
        ts0,
        fill=FillSpec(0.5, 0.5),
        direction="UP",
        hold_minutes=30,
        exit_mode="hold_extend",
        roi_time_stop={
            "enabled": True,
            "rails": [{"mins": 15, "min_roi": 0.05}],
        },
    )
    assert sim is not None
    assert sim.reason == "ROI_TIME15"
    assert sim.ret < 0.05


def test_roi_time_keeps_strong_winner():
    ts0 = pd.Timestamp("2026-05-06 10:31:00", tz="America/New_York")
    qrows = []
    for i, mid in enumerate([1.0, 1.1, 1.3, 1.5, 1.7, 1.9]):
        t = ts0 + pd.Timedelta(minutes=i * 5)
        qrows.append({"timestamp": t, "bid": mid - 0.02, "ask": mid + 0.02})
    sim = simulate_trade(
        pd.DataFrame(qrows),
        ts0,
        fill=FillSpec(0.5, 0.5),
        direction="UP",
        hold_minutes=30,
        tp_mult=1.55,
        exit_mode="hold_extend",
        roi_time_stop={
            "enabled": True,
            "mid_mins": 15,
            "mid_min_roi": 0.05,
            "late_mins": 30,
            "late_min_roi": 0.05,
        },
    )
    assert sim is not None
    assert not str(sim.reason).startswith("ROI_TIME")

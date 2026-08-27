"""Unit tests for Δ-aware time-stop."""
from __future__ import annotations

import pandas as pd

from maga7.common.delta_time_stop import delta_time_stop_from_trade
from maga7.common.fills import FillSpec
from maga7.common.replay import simulate_trade


def test_delta_time_stop_config_default_off():
    assert delta_time_stop_from_trade({}).enabled is False
    assert delta_time_stop_from_trade({"delta_time_stop": {"enabled": True}}).enabled is True


def test_delta_stop_cuts_when_stock_stalls():
    ts0 = pd.Timestamp("2026-02-18 10:31:00", tz="America/New_York")
    # Option bleeds slowly; stock barely moves.
    qrows = []
    for i, mid in enumerate([1.00, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70]):
        t = ts0 + pd.Timedelta(minutes=i)
        qrows.append({"timestamp": t, "bid": mid - 0.01, "ask": mid + 0.01})
    srows = []
    for i in range(10):
        t = ts0 + pd.Timedelta(minutes=i)
        # +5bp then flat — never reaches +15bp
        px = 100.0 + (0.05 if i >= 1 else 0.0)
        srows.append(
            {
                "timestamp": t,
                "close": px,
                "mf10": 1.0,
                "streak_up": 1,
                "streak_dn": 0,
                "high": px,
                "low": px,
            }
        )
    sim = simulate_trade(
        pd.DataFrame(qrows),
        ts0,
        fill=FillSpec(0.5, 0.5),
        direction="UP",
        hold_minutes=30,
        stock_day=pd.DataFrame(srows),
        exit_mode="hold_extend",
        delta_time_stop={
            "enabled": True,
            "check_seconds": 180,
            "max_seconds": 600,
            "min_stock_move": 0.0015,
            "opt_mtm_max": 0.0,
        },
    )
    assert sim is not None
    assert sim.reason == "DELTA_STOP"
    assert sim.ret < 0


def test_delta_stop_keeps_when_stock_confirms():
    ts0 = pd.Timestamp("2026-02-18 10:31:00", tz="America/New_York")
    qrows = []
    for i, mid in enumerate([1.00, 0.98, 1.05, 1.20, 1.40, 1.60]):
        t = ts0 + pd.Timedelta(minutes=i)
        qrows.append({"timestamp": t, "bid": mid - 0.01, "ask": mid + 0.01})
    srows = []
    for i in range(8):
        t = ts0 + pd.Timedelta(minutes=i)
        px = 100.0 * (1.0 + 0.003 * i)  # +30bp/min
        srows.append(
            {
                "timestamp": t,
                "close": px,
                "mf10": 1.0,
                "streak_up": 1,
                "streak_dn": 0,
                "high": px,
                "low": px,
            }
        )
    sim = simulate_trade(
        pd.DataFrame(qrows),
        ts0,
        fill=FillSpec(0.5, 0.5),
        direction="UP",
        hold_minutes=30,
        tp_mult=1.5,
        stock_day=pd.DataFrame(srows),
        exit_mode="hold_extend",
        delta_time_stop={
            "enabled": True,
            "check_seconds": 120,
            "min_stock_move": 0.0015,
            "opt_mtm_max": 0.0,
        },
    )
    assert sim is not None
    assert sim.reason != "DELTA_STOP"

"""Unit tests for second-level ladder_active exits."""
from __future__ import annotations

import pandas as pd

from maga7.common.fills import FillSpec
from maga7.common.ladder_active import ladder_active_from_trade
from maga7.common.replay import simulate_trade


def test_ladder_config_from_exit_mode():
    cfg = ladder_active_from_trade({"exit_mode": "ladder_active"})
    assert cfg.enabled is True
    assert cfg.max_hold_seconds == 300


def test_ladder_config_default_off():
    assert ladder_active_from_trade({}).enabled is False
    assert ladder_active_from_trade({"exit_mode": "hold_extend"}).enabled is False


def test_ladder_when_parsed():
    cfg = ladder_active_from_trade(
        {"ladder_active": {"enabled": True, "when": "mixed_wash_up"}}
    )
    assert cfg.enabled is True
    assert cfg.when == "mixed_wash_up"


def _quotes(mids: list[float], ts0: pd.Timestamp, step_s: int = 5) -> pd.DataFrame:
    rows = []
    for i, mid in enumerate(mids):
        t = ts0 + pd.Timedelta(seconds=i * step_s)
        rows.append({"timestamp": t, "bid": mid - 0.01, "ask": mid + 0.01})
    return pd.DataFrame(rows)


def test_sl_ladder_cuts_before_outer_sl():
    ts0 = pd.Timestamp("2026-02-18 10:31:00", tz="America/New_York")
    # Bleed to -12% quickly; outer SL is -55%.
    mids = [1.00, 0.95, 0.90, 0.88, 0.85, 0.80]
    sim = simulate_trade(
        _quotes(mids, ts0),
        ts0,
        fill=FillSpec(0.5, 0.5),
        direction="UP",
        hold_minutes=30,
        tp_mult=1.6,
        sl_mult=0.45,
        exit_mode="ladder_active",
        ladder_active={
            "enabled": True,
            "max_hold_seconds": 300,
            "keep_outer_rails": True,
            "sl_rails": [{"ret": -0.10}, {"ret": -0.18}],
            "tp_rails": [{"ret": 0.20, "action": "exit"}],
            "profit_stall": {"min_peak": 0.50, "stall_seconds": 999},
            "mf_flip": False,
        },
    )
    assert sim is not None
    assert str(sim.reason).startswith("SL_LADDER")
    assert sim.ret <= -0.10


def test_tp_ladder_hard_exit():
    ts0 = pd.Timestamp("2026-02-18 10:31:00", tz="America/New_York")
    mids = [1.00, 1.05, 1.10, 1.15, 1.22]
    sim = simulate_trade(
        _quotes(mids, ts0),
        ts0,
        fill=FillSpec(0.5, 0.5),
        direction="UP",
        hold_minutes=30,
        exit_mode="ladder_active",
        ladder_active={
            "enabled": True,
            "max_hold_seconds": 300,
            "keep_outer_rails": False,
            "sl_rails": [{"ret": -0.50}],
            "tp_rails": [{"ret": 0.20, "action": "exit"}],
            "profit_stall": {"min_peak": 0.99, "stall_seconds": 999},
            "mf_flip": False,
        },
    )
    assert sim is not None
    assert sim.reason == "TP_LADDER20"
    assert sim.ret >= 0.18


def test_profit_stall_exits():
    ts0 = pd.Timestamp("2026-02-18 10:31:00", tz="America/New_York")
    # Peak +10%, then flat for >20s (5s steps → 5 quotes after peak).
    mids = [1.00, 1.10, 1.10, 1.10, 1.10, 1.09, 1.09]
    sim = simulate_trade(
        _quotes(mids, ts0, step_s=5),
        ts0,
        fill=FillSpec(0.5, 0.5),
        direction="UP",
        hold_minutes=30,
        exit_mode="ladder_active",
        ladder_active={
            "enabled": True,
            "max_hold_seconds": 300,
            "keep_outer_rails": False,
            "sl_rails": [{"ret": -0.50}],
            "tp_rails": [{"ret": 0.50, "action": "exit"}],
            "profit_stall": {"min_peak": 0.08, "stall_seconds": 20},
            "mf_flip": False,
        },
    )
    assert sim is not None
    assert sim.reason == "PROFIT_STALL"


def test_sec_max_hard_cap():
    ts0 = pd.Timestamp("2026-02-18 10:31:00", tz="America/New_York")
    # Drift slowly; never hit rails; should flatten at 30s max hold.
    mids = [1.00 + 0.001 * i for i in range(20)]
    sim = simulate_trade(
        _quotes(mids, ts0, step_s=5),
        ts0,
        fill=FillSpec(0.5, 0.5),
        direction="UP",
        hold_minutes=30,
        exit_mode="ladder_active",
        ladder_active={
            "enabled": True,
            "max_hold_seconds": 30,
            "keep_outer_rails": False,
            "sl_rails": [{"ret": -0.90}],
            "tp_rails": [{"ret": 0.90, "action": "exit"}],
            "profit_stall": {"min_peak": 0.99, "stall_seconds": 999},
            "mf_flip": False,
        },
    )
    assert sim is not None
    assert sim.reason == "SEC_MAX"
    held = (sim.exit_ts - sim.entry_ts).total_seconds()
    assert held <= 35

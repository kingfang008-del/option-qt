"""Unit tests for split-entry scale-in helper + simulate_trade hook."""
from __future__ import annotations

import pandas as pd

from maga7.common.fills import FillSpec
from maga7.common.replay import simulate_trade
from maga7.common.scale_in import blend_scale_in_ret, confirm_scale_in, scale_in_from_trade


def test_confirm_modes():
    assert confirm_scale_in(mode="always", direction="UP", mf=-1.0) is True
    assert confirm_scale_in(mode="never", direction="UP", mf=1.0) is False
    assert confirm_scale_in(mode="mf", direction="UP", mf=1.0) is True
    assert confirm_scale_in(mode="mf", direction="UP", mf=-1.0) is False
    assert confirm_scale_in(mode="mf", direction="DN", mf=-1.0) is True
    assert confirm_scale_in(mode="mf_streak", direction="UP", mf=1.0, streak_up=0) is False
    assert confirm_scale_in(mode="mf_streak", direction="UP", mf=1.0, streak_up=2) is True


def test_blend_ret():
    r, dep, added = blend_scale_in_ret(
        entry1=1.0, entry2=0.7, exit_px=1.6, first_frac=0.5, add_frac=0.5
    )
    # r1=+60%, r2≈+128.6% → blend ≈ 0.5*0.6 + 0.5*1.2857
    assert added and abs(dep - 1.0) < 1e-9
    assert r > 0.9
    r2, dep2, added2 = blend_scale_in_ret(
        entry1=1.0, entry2=None, exit_px=1.6, first_frac=0.5, add_frac=0.5
    )
    assert not added2 and abs(dep2 - 0.5) < 1e-9
    assert abs(r2 - 0.3) < 1e-9  # half of +60%


def test_simulate_trade_scale_in_adds_on_pullback():
    # Path: entry mid~1.0 → dips to ~0.7 → recovers to TP 1.6
    ts0 = pd.Timestamp("2026-05-11 10:00:00", tz="America/New_York")
    rows = []
    # flat then dump then recover
    for i, mid in enumerate([1.00, 0.95, 0.85, 0.70, 0.90, 1.20, 1.65]):
        t = ts0 + pd.Timedelta(minutes=i)
        half = 0.02
        rows.append({"timestamp": t, "bid": mid - half, "ask": mid + half})
    path = pd.DataFrame(rows)
    # stock mf stays favorable for UP
    stock = pd.DataFrame(
        {
            "timestamp": [ts0 + pd.Timedelta(minutes=i) for i in range(7)],
            "close": [100.0 + i * 0.1 for i in range(7)],
            "mf10": [1e6] * 7,
            "streak_up": [3] * 7,
            "streak_dn": [0] * 7,
        }
    )
    sim = simulate_trade(
        path,
        ts0,
        fill=FillSpec(entry_frac=0.5, exit_frac=0.5),
        direction="UP",
        stock_day=stock,
        hold_minutes=30,
        stock_bar_delay_seconds=0,
        scale_in={
            "enabled": True,
            "first_frac": 0.5,
            "add_frac": 0.5,
            "pullback_ret": 0.25,
            "confirm_mode": "mf",
            "min_hold_seconds": 60,
        },
    )
    assert sim is not None
    assert sim.scale_in_added
    assert sim.scale_in_entry2 is not None
    assert sim.scale_in_entry2 < sim.entry
    # Blended ret should beat first-only half of full-path ret when add is cheaper.
    assert sim.ret > 0.5 * (sim.exit / sim.entry - 1.0) - 1e-6


def test_scale_in_from_trade_default_off():
    cfg = scale_in_from_trade({})
    assert cfg.enabled is False

"""Unit tests for AM pocket scale-out exit simulator."""
from __future__ import annotations

import numpy as np

from maga7.tools.scan_am_pocket_scaleout import simulate_scaleout


def _path(rets: list[float], dt: float = 1.0):
    r = np.asarray(rets, dtype=float)
    h = np.arange(len(r), dtype=float) * dt
    return r, h


def test_full_tp_like_frac1():
    rets, holds = _path([0.0, 0.02, 0.05, 0.09, 0.12])
    sim = simulate_scaleout(
        rets, holds, frac1=1.0, tp1=0.08, sl=0.15, max_hold=100, runner="hold"
    )
    assert sim["scaled"] is True
    assert sim["reason"] == "scale_full"
    assert abs(sim["ret"] - 0.09) < 1e-9


def test_hard_sl_before_scale():
    rets, holds = _path([0.0, -0.05, -0.16, -0.20])
    sim = simulate_scaleout(
        rets, holds, frac1=0.5, tp1=0.08, sl=0.15, max_hold=100, runner="hold"
    )
    assert sim["scaled"] is False
    assert sim["reason"] == "sl"
    assert abs(sim["ret"] - (-0.16)) < 1e-9


def test_scale_then_be_stop():
    # hit +8%, then give back through 0 on runner with BE
    rets, holds = _path([0.0, 0.04, 0.09, 0.05, -0.01, -0.05])
    sim = simulate_scaleout(
        rets,
        holds,
        frac1=0.5,
        tp1=0.08,
        sl=0.15,
        max_hold=100,
        runner="hold",
        be_after_scale=True,
        floor=0.0,
    )
    assert sim["scaled"] is True
    assert sim["reason"] == "runner_sl"
    # blend: 0.5*0.09 + 0.5*(-0.01)
    assert abs(sim["ret"] - (0.5 * 0.09 + 0.5 * (-0.01))) < 1e-9


def test_scale_then_trail():
    rets, holds = _path([0.0, 0.07, 0.10, 0.18, 0.22, 0.12, 0.05])
    sim = simulate_scaleout(
        rets,
        holds,
        frac1=0.67,
        tp1=0.06,
        sl=0.15,
        max_hold=100,
        runner="trail",
        arm=0.15,
        trail=0.10,
        be_after_scale=True,
        floor=0.0,
    )
    assert sim["scaled"] is True
    assert sim["reason"] == "runner_trail"
    # peak 0.22, giveback 0.10 -> exit at 0.12
    assert abs(sim["r2"] - 0.12) < 1e-9
    assert abs(sim["ret"] - (0.67 * 0.07 + 0.33 * 0.12)) < 1e-9


def test_time_cut_before_scale():
    rets, holds = _path([0.0, 0.01, -0.02, -0.01], dt=60.0)
    sim = simulate_scaleout(
        rets,
        holds,
        frac1=0.5,
        tp1=0.08,
        sl=0.15,
        max_hold=600,
        runner="hold",
        time_cut=120,
        time_cut_min=0.0,
    )
    assert sim["scaled"] is False
    assert sim["reason"] == "time_cut"

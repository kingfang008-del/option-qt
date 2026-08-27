"""Call/put flip short-hold simulator smoke tests."""
from __future__ import annotations

import numpy as np

from maga7.tools.scan_am_pocket_flip import (
    simulate_flip_once,
    simulate_opp_first,
    simulate_scalp,
)


def _path(vals: list[float], dt: float = 1.0):
    rets = np.asarray(vals, dtype=float)
    holds = np.arange(len(rets), dtype=float) * dt
    return rets, holds


def test_scalp_hits_tp_fast():
    r, h = _path([0.0, 0.03, 0.09, 0.12])
    sim = simulate_scalp(r, h, tp=0.08, sl=0.10, max_hold=30)
    assert sim["reason"] == "tp"
    assert abs(sim["ret"] - 0.09) < 1e-9


def test_flip_once_on_adverse():
    # primary dumps to -10% then would recover; we flip at -8%
    prim = _path([0.0, -0.04, -0.09, -0.05, 0.20])
    # opposite rallies after t=2
    opp = _path([0.0, 0.02, 0.05, 0.12, 0.18])
    sim = simulate_flip_once(
        prim,
        opp,
        adv=0.08,
        tp=0.10,
        sl=0.10,
        max_leg=30,
        window=60,
        reentry_slip=0.0,
    )
    assert sim["flipped"] is True
    assert sim["n_legs"] == 2
    assert sim["ret"] > -0.08  # recovered via opp leg


def test_opp_first_switches_side():
    prim = _path([0.0, -0.05, -0.09, -0.12, -0.20])
    opp = _path([0.0, 0.04, 0.08, 0.11, 0.15])
    sim = simulate_opp_first(prim, opp, look_t=10, dip=0.08, tp=0.10, sl=0.10, max_hold=30)
    assert sim["flipped"] is True
    assert sim["reason"] == "tp"
    assert abs(sim["ret"] - 0.11) < 1e-9

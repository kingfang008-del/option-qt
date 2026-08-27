"""Path-state exit simulator smoke tests."""
from __future__ import annotations

import numpy as np

from maga7.tools.scan_am_pocket_path_exit import simulate_path_exit


def _ramp(n: int = 120, opt_step: float = 0.001, stk_step: float = 0.00005):
    holds = np.arange(n, dtype=float)
    opt = holds * opt_step
    stk_px = 100.0 * (1.0 + holds * stk_step)
    return opt, holds, holds.copy(), stk_px


def test_tpsl_only_hits_tp():
    opt, holds, sh, px = _ramp()
    sim = simulate_path_exit(
        opt,
        holds,
        stock_holds=sh,
        stock_px=px,
        direction="UP",
        mode="tpsl_only",
        params={"tp": 0.08, "sl": 0.15, "max_hold": 200},
    )
    assert sim["reason"] == "tp"
    assert abs(sim["ret"] - 0.08) < 1e-9


def test_stock_adv_cuts_before_tp():
    holds = np.arange(100, dtype=float)
    opt = holds * 0.0002  # slow, never hits 8%
    # stock dumps after t=40
    px = np.full(100, 100.0)
    px[40:] = 100.0 * (1.0 - 0.003)  # -30bp adverse for UP
    sim = simulate_path_exit(
        opt,
        holds,
        stock_holds=holds,
        stock_px=px,
        direction="UP",
        mode="stock_adv",
        params={
            "tp": 0.08,
            "sl": 0.15,
            "max_hold": 200,
            "min_hold": 10,
            "stock_adv": 0.002,
            "stock_adv_opt_max": 0.05,
        },
    )
    assert sim["reason"] == "stock_adv"
    assert sim["hold_sec"] >= 40


def test_fail_fast_after_t():
    holds = np.arange(80, dtype=float)
    opt = np.full(80, -0.01)
    px = 100.0 * (1.0 - holds * 0.0001)  # drifting adverse
    sim = simulate_path_exit(
        opt,
        holds,
        stock_holds=holds,
        stock_px=px,
        direction="UP",
        mode="fail_fast",
        params={
            "tp": 0.08,
            "sl": 0.50,
            "max_hold": 200,
            "min_hold": 5,
            "stock_adv": 9.0,
            "fail_t": 30,
            "fail_stock": 0.0,
            "fail_opt": 0.0,
        },
    )
    assert sim["reason"] == "fail_fast"
    assert sim["hold_sec"] >= 30

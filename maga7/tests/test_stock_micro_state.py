"""Tests for stock micro-state detector."""
from __future__ import annotations

import numpy as np
import pandas as pd

from maga7.common.stock_micro_state import MicroStateConfig, detect_micro_edges, window_state

NY = "America/New_York"


def test_window_state_up_trend_snr():
    t = np.arange(40, dtype=float)
    c = 100.0 + 0.05 * t
    st = window_state(c)
    assert st is not None
    assert st["slope_bp_per_min"] > 0
    assert st["snr"] > 2.0


def test_detect_up_cross():
    rng = np.random.default_rng(0)
    flat = 100 + rng.normal(0, 0.01, size=80)
    up = flat[-1] + np.arange(40) * 0.04
    close = np.concatenate([flat, up])
    idx = pd.date_range("2026-05-01 09:30", periods=len(close), freq="s", tz=NY)
    df = pd.DataFrame({"timestamp": idx, "close": close})
    cfg = MicroStateConfig(
        short_sec=12,
        long_sec=24,
        stride_sec=2,
        min_snr=1.0,
        min_slope_bp_per_min=0.5,
        scan_start="09:30",
        scan_end="10:15",
        cooldown_sec=30,
        require_accel=False,
    )
    evs = detect_micro_edges(df, symbol="TEST", date="2026-05-01", cfg=cfg)
    assert any(e["dir"] == "UP" for e in evs)

"""Unit tests for causal 1s smooth regression detector."""
from __future__ import annotations

import numpy as np
import pandas as pd

from maga7.common.smooth_regress_1s import (
    SmoothRegressConfig,
    detect_day_edges,
    fit_window,
    is_smooth,
)

NY = "America/New_York"


def test_fit_window_up_trend_high_r2():
    t = np.arange(60, dtype=float)
    c = 100.0 + 0.02 * t + np.random.default_rng(0).normal(0, 0.001, size=60)
    feat = fit_window(c)
    assert feat is not None
    assert feat["r2"] > 0.95
    assert feat["slope_bp_per_min"] > 0


def test_detect_rising_edge_on_synthetic():
    rng = np.random.default_rng(1)
    # chop then smooth grind
    chop = 100 + rng.normal(0, 0.05, size=120).cumsum() * 0.01
    grind = chop[-1] + np.arange(90) * 0.03
    close = np.concatenate([chop, grind])
    idx = pd.date_range("2026-05-01 09:30", periods=len(close), freq="s", tz=NY)
    df = pd.DataFrame({"timestamp": idx, "close": close})
    cfg = SmoothRegressConfig(
        win_sec=60,
        stride_sec=5,
        min_r2=0.85,
        max_resid_bp=8.0,
        min_slope_bp_per_min=1.0,
        min_path_eff=0.25,
        scan_start="09:30",
        scan_end="12:00",
        cooldown_sec=60,
    )
    evs = detect_day_edges(df, symbol="TEST", date="2026-05-01", cfg=cfg)
    assert len(evs) >= 1
    assert evs[0]["dir"] == "UP"
    assert is_smooth(
        {
            "r2": evs[0]["r2"],
            "resid_bp": evs[0]["resid_bp"],
            "slope_bp_per_min": evs[0]["slope_bp_per_min"],
            "path_eff": evs[0]["path_eff"],
        },
        cfg,
    )

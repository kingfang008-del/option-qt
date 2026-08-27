"""Unit tests for launch-slope second-level detector."""
from __future__ import annotations

import numpy as np
import pandas as pd

from maga7.common.launch_slope import (
    attach_launch_slope_features,
    launch_edges,
)


def _synth_spike() -> pd.DataFrame:
    """Flat then a sharp 5s UP impulse, then flat."""
    ts0 = pd.Timestamp("2026-05-01 09:35:00", tz="America/New_York")
    rows = []
    px = 100.0
    for i in range(120):
        if 40 <= i < 45:
            px *= 1.0015  # ~0.15%/s → ~0.75% over 5s
        t = ts0 + pd.Timedelta(seconds=i)
        rows.append(
            {
                "timestamp": t,
                "open": px,
                "high": px * 1.0001,
                "low": px * 0.9999,
                "close": px,
                "volume": 1000.0 + (5000.0 if 40 <= i < 45 else 0.0),
            }
        )
    return pd.DataFrame(rows)


def test_attach_ret_k_and_local_peak():
    feat = attach_launch_slope_features(
        _synth_spike(), slope_sec=5, peak_lookback_sec=30, mf_window_sec=None
    )
    assert not feat.empty
    assert "ret_k" in feat.columns
    # During/after spike, some ret_k should be clearly positive
    assert float(np.nanmax(feat["ret_k"])) > 0.005
    assert bool(feat["is_local_max_up"].any())


def test_launch_edges_up_rising():
    feat = attach_launch_slope_features(
        _synth_spike(), slope_sec=5, peak_lookback_sec=30, mf_window_sec=None
    )
    edges = launch_edges(feat, direction="UP", abs_ret_min=0.004, require_local_peak=True)
    assert len(edges) >= 1
    # First edge should be around the spike window (index ~44+)
    assert int(edges[0]) >= 40
    assert int(edges[0]) <= 55


def test_launch_edges_respects_threshold():
    feat = attach_launch_slope_features(
        _synth_spike(), slope_sec=5, peak_lookback_sec=30, mf_window_sec=None
    )
    loose = launch_edges(feat, direction="UP", abs_ret_min=0.002, require_local_peak=False)
    tight = launch_edges(feat, direction="UP", abs_ret_min=0.05, require_local_peak=False)
    assert len(loose) >= 1
    assert len(tight) == 0

import numpy as np
import pandas as pd

from maga7.tools.scan_convexity_scout_tick import (
    _adaptive_gap,
    _flow_snapshot,
    _second_median_path,
)


def test_second_median_path_rejects_single_tick_outlier():
    ts = np.array(
        [
            1_000_000_000,
            1_100_000_000,
            1_200_000_000,
            2_000_000_000,
        ],
        dtype=np.int64,
    )
    out_ts, out_px = _second_median_path(ts, np.array([1.0, 100.0, 1.0, 2.0]))
    assert out_ts.tolist() == [1_000_000_000, 2_000_000_000]
    assert out_px.tolist() == [1.0, 2.0]


def test_adaptive_gap_increases_with_underlying_impulse():
    weak = {"r3": -0.0015, "r15": -0.0030}
    strong = {"r3": -0.011, "r15": -0.015}
    assert _adaptive_gap(weak) == 0.005
    assert _adaptive_gap(strong) == 0.030


def test_flow_snapshot_uses_only_data_at_or_before_signal():
    base = pd.Timestamp("2026-07-15 09:35:00", tz="America/New_York")
    ts_ns = np.array(
        [int((base + pd.Timedelta(seconds=i)).value) for i in range(401)],
        dtype=np.int64,
    )
    flow = {
        "ts_ns": ts_ns,
        "put_v": np.ones(401),
        "call_v": np.ones(401),
        "n": 401,
        "source": "tick",
    }
    snap = _flow_snapshot(flow, base + pd.Timedelta(seconds=300))
    assert snap["put_share_60"] == 0.5
    assert snap["put_v_60"] == 61.0

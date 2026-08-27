from __future__ import annotations

import numpy as np
import pandas as pd

from maga7.common.option_trade_tpsl import simulate_trade_tpsl_confirm_abort


def test_trade_confirm_abort_cuts_loser():
    idx = pd.date_range("2026-07-24 09:30:00", periods=120, freq="1s", tz="America/New_York")
    # entry mid=1.0 path; drops to -12% by ~40s without ever +2%
    last = np.linspace(1.00, 0.85, len(idx))
    ts_ns = idx.view("int64")
    # pandas 2 may need .asi8
    ts_ns = idx.astype("int64").to_numpy()
    sim = simulate_trade_tpsl_confirm_abort(
        ts_ns,
        last,
        idx[0],
        tp=0.15,
        sl=0.25,
        max_hold_sec=900,
        confirm_sec=60,
        confirm_thr=0.02,
        abort_thr=0.10,
        on_timeout="abort",
        slip=0.0,
    )
    assert sim is not None
    assert sim["reason"] in {"early_abort", "confirm_abort"}
    assert sim["ret"] < 0

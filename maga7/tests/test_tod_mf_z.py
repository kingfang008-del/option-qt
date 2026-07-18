import numpy as np
import pandas as pd

from maga7.common.signals import tod_mf_z_ok


def _frame() -> pd.DataFrame:
    """20 quiet days then one spike day at 10:35."""
    rows = []
    for i in range(20):
        d = (pd.Timestamp("2026-04-01") + pd.Timedelta(days=i)).strftime("%Y-%m-%d")
        # skip weekends roughly by using consecutive calendar — ok for unit test
        ts = pd.Timestamp(f"{d} 10:35", tz="America/New_York")
        rows.append(
            {
                "timestamp": ts,
                "date": d,
                "tod": "10:35",
                "mf10": 1e6,
                "close": 100.0,
                "net$": 1e5,
            }
        )
    spike = {
        "timestamp": pd.Timestamp("2026-04-21 10:35", tz="America/New_York"),
        "date": "2026-04-21",
        "tod": "10:35",
        "mf10": 20e6,
        "close": 101.0,
        "net$": 2e6,
    }
    rows.append(spike)
    return pd.DataFrame(rows)


def test_tod_z_fires_on_spike():
    df = _frame()
    ok, z = tod_mf_z_ok(
        df,
        asof_ts=pd.Timestamp("2026-04-21 10:35", tz="America/New_York"),
        direction="UP",
        lookback_days=20,
        z_min=2.0,
        min_periods=5,
    )
    assert z is not None and z > 2.0
    assert ok


def test_tod_z_blocks_normal():
    df = _frame()
    # mid history day should be near z~0 after warmup
    ok, z = tod_mf_z_ok(
        df,
        asof_ts=pd.Timestamp("2026-04-15 10:35", tz="America/New_York"),
        direction="UP",
        lookback_days=20,
        z_min=2.0,
        min_periods=5,
    )
    assert z is not None
    assert abs(z) < 2.0
    assert not ok

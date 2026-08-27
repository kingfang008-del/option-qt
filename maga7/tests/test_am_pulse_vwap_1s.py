from __future__ import annotations

import numpy as np
import pandas as pd

from maga7.common.am_pulse_scout import AmPulseScoutConfig, scan_day_1s_vwap
from maga7.common.session_1s_features import prepare_day_arrays, rolling_vwap_at


def _grind_1s_dn(*, open_px: float = 100.0) -> pd.DataFrame:
    """Linear grind down: after ~20s, 30s VWAP is clearly below -1%."""
    idx = pd.date_range("2026-07-24 09:30:00", periods=120, freq="1s", tz="America/New_York")
    rows = []
    for i, t in enumerate(idx):
        # drop 2bp per second → ~2.4% in 2 minutes
        px = open_px * (1.0 - 0.0002 * i)
        rows.append(
            {
                "timestamp": t,
                "open": px,
                "high": px * 1.0001,
                "low": px * 0.9999,
                "close": px,
                "volume": 1000.0,
            }
        )
    return pd.DataFrame(rows)


def test_rolling_vwap_matches_manual():
    day = _grind_1s_dn()
    arr = prepare_day_arrays(day)
    i = 40
    vwap = rolling_vwap_at(arr, i, 10)
    sub = day.iloc[i - 9 : i + 1]  # approx; exact uses ts lookback
    # recompute via helper contract: prints in (t-10s, t]
    t_ns = int(arr["ts_ns"][i])
    mask = (arr["ts_ns"] > t_ns - 10_000_000_000) & (arr["ts_ns"] <= t_ns)
    c = arr["close"][mask]
    v = arr["volume"][mask]
    expect = float(np.sum(c * v) / np.sum(v))
    assert abs(vwap - expect) < 1e-9


def test_vwap_1s_fo_fires_before_1m_close_semantics():
    """30s VWAP FO can fire intra-minute; px is VWAP not last print."""
    cfg = AmPulseScoutConfig(
        enabled=True,
        feature_mode="vwap_1s",
        vwap_win_sec=30,
        sample_every_sec=10,
        min_fav_from_open=0.01,
        dirs=("DN",),
        window_start="09:30",
        window_end="10:30",
    )
    alerts = scan_day_1s_vwap(
        _grind_1s_dn(), date="2026-07-24", symbol="TSLA", cfg=cfg, day_open=100.0
    )
    assert len(alerts) == 1
    a = alerts[0]
    assert a.arm == "FO" and a.dir == "DN"
    assert a.feature_mode == "vwap_1s"
    assert a.vwap_win_sec == 30
    assert a.fav_from_open >= 0.01
    ts = pd.Timestamp(a.ts)
    # Warm 30s + grind → fires around 09:31:10, still before a 09:32 1m close.
    assert ts.hour == 9 and ts.minute in (30, 31)

def test_vwap_agree_requires_all_windows():
    cfg = AmPulseScoutConfig(
        enabled=True,
        feature_mode="vwap_1s",
        vwap_win_sec=30,
        vwap_agree_wins=(10, 20, 30),
        sample_every_sec=10,
        min_fav_from_open=0.01,
        dirs=("DN",),
    )
    alerts = scan_day_1s_vwap(
        _grind_1s_dn(), date="2026-07-24", symbol="TSLA", cfg=cfg, day_open=100.0
    )
    assert len(alerts) == 1
    assert alerts[0].dir == "DN"

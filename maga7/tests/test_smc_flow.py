from __future__ import annotations

import numpy as np
import pandas as pd

from maga7.common.smc_flow import (
    detect_smc_flow_dn,
    dn_vol_share_at,
    prepare_smc_flow_day,
)

NY = "America/New_York"


def _synth_day() -> pd.DataFrame:
    """Flat → spike above → dump through prior low (sweep then BOS-ish)."""
    n = 400
    idx = pd.date_range("2026-05-01 09:35:00", periods=n, freq="s", tz=NY)
    px = np.full(n, 100.0)
    # grind mild up
    px[:200] = 100 + np.linspace(0, 0.4, 200)
    # sweep high
    px[200:220] = 100.4 + np.linspace(0, 0.8, 20)
    # reclaim + displace down through lows
    px[220:] = 101.2 - np.linspace(0, 2.5, n - 220)
    high = px + 0.05
    low = px - 0.05
    high[205:215] = px[205:215] + 0.15
    vol = np.where(np.arange(n) >= 220, 5000.0, 800.0)
    # amplify down seconds
    d = np.diff(px, prepend=px[0])
    vol = np.where(d < 0, vol * 1.5, vol)
    return pd.DataFrame(
        {
            "timestamp": idx,
            "open": px,
            "high": high,
            "low": low,
            "close": px,
            "volume": vol,
        }
    )


def test_prepare_and_dn_vol_share():
    arrays = prepare_smc_flow_day(_synth_day())
    assert arrays is not None
    share = dn_vol_share_at(arrays, i=300, window_sec=60)
    assert share is not None
    assert 0.0 <= share <= 1.0


def test_sweep_and_bos_fire():
    arrays = prepare_smc_flow_day(_synth_day())
    assert arrays is not None
    # late in dump
    i = 320
    sweep = detect_smc_flow_dn(
        arrays,
        i=i,
        morph="sweep_rev_dn",
        swing_sec=180,
        disp_sec=60,
        disp_thr=0.003,
        flow_sec=120,
        min_dn_vol_share=0.50,
        min_streak_dn=0,
        require_mf_neg=False,
    )
    bos = detect_smc_flow_dn(
        arrays,
        i=i,
        morph="bos_disp_dn",
        swing_sec=180,
        disp_sec=60,
        disp_thr=0.003,
        flow_sec=120,
        min_dn_vol_share=None,
        min_streak_dn=0,
        require_mf_neg=False,
    )
    # At least one morph should arm on synthetic dump
    assert sweep is not None or bos is not None

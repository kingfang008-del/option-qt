"""Unit tests for short-window adverse volume share."""
from __future__ import annotations

import numpy as np
import pandas as pd

from maga7.common.adverse_vol_share import (
    adverse_vol_share_asof,
    adverse_vol_share_from_trade,
    prepare_stock_1s_arrays,
)
from maga7.common.fills import FillSpec
from maga7.common.replay import simulate_trade


def test_config_default_off():
    assert adverse_vol_share_from_trade({}).enabled is False
    cfg = adverse_vol_share_from_trade(
        {"adverse_vol_share": {"enabled": True, "min_share": 0.6, "mode": "soft_exit"}}
    )
    assert cfg.enabled and cfg.min_share == 0.6 and cfg.mode == "soft_exit"


def test_share_counts_adverse_ticks():
    ts0 = pd.Timestamp("2026-02-18 10:31:00", tz="America/New_York")
    rows = []
    px = 100.0
    for i in range(180):
        # mostly down ticks with volume 10; occasional up with vol 1
        if i % 5 == 0:
            px += 0.01
            v = 1.0
        else:
            px -= 0.01
            v = 10.0
        rows.append(
            {
                "timestamp": ts0 + pd.Timedelta(seconds=i),
                "close": px,
                "volume": v,
            }
        )
    arr = prepare_stock_1s_arrays(pd.DataFrame(rows))
    share = adverse_vol_share_asof(
        arr,
        now_ts=ts0 + pd.Timedelta(seconds=179),
        window_seconds=180,
        direction="UP",
    )
    assert share is not None and share > 0.7


def test_soft_exit_adv_vol():
    ts0 = pd.Timestamp("2026-02-18 10:31:00", tz="America/New_York")
    qrows = []
    for i, mid in enumerate([1.00, 0.95, 0.90, 0.85, 0.80]):
        t = ts0 + pd.Timedelta(minutes=i)
        qrows.append({"timestamp": t, "bid": mid - 0.01, "ask": mid + 0.01})
    # 1s path: heavy adverse volume
    s1 = []
    px = 100.0
    for i in range(200):
        px -= 0.01
        s1.append(
            {
                "timestamp": ts0 + pd.Timedelta(seconds=i),
                "close": px,
                "volume": 100.0,
            }
        )
    sim = simulate_trade(
        pd.DataFrame(qrows),
        ts0,
        fill=FillSpec(0.5, 0.5),
        direction="UP",
        hold_minutes=30,
        exit_mode="none",
        stock_1s=pd.DataFrame(s1),
        adverse_vol_share={
            "enabled": True,
            "mode": "soft_exit",
            "check_seconds": 120,
            "window_seconds": 120,
            "min_share": 0.55,
            "opt_mtm_max": 0.0,
        },
    )
    assert sim is not None
    assert sim.reason == "ADV_VOL"
    assert sim.ret < 0


def test_no_fire_when_share_low():
    ts0 = pd.Timestamp("2026-02-18 10:31:00", tz="America/New_York")
    qrows = [{"timestamp": ts0 + pd.Timedelta(minutes=i), "bid": 0.9, "ask": 0.92} for i in range(35)]
    s1 = []
    px = 100.0
    for i in range(400):
        # favorable drift for UP
        px += 0.01
        s1.append(
            {
                "timestamp": ts0 + pd.Timedelta(seconds=i),
                "close": px,
                "volume": 50.0,
            }
        )
    sim = simulate_trade(
        pd.DataFrame(qrows),
        ts0,
        fill=FillSpec(0.5, 0.5),
        direction="UP",
        hold_minutes=30,
        exit_mode="none",
        stock_1s=pd.DataFrame(s1),
        adverse_vol_share={
            "enabled": True,
            "mode": "soft_exit",
            "check_seconds": 120,
            "window_seconds": 120,
            "min_share": 0.55,
            "opt_mtm_max": 0.0,
        },
    )
    assert sim is not None
    assert sim.reason != "ADV_VOL"

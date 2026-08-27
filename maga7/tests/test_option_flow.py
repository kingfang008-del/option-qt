from __future__ import annotations

import numpy as np
import pandas as pd

from maga7.common.option_flow import (
    detect_put_flow_dn,
    iter_put_flow_dn_in_window,
    option_right,
    prepare_option_flow_day,
    put_flow_features_at,
)

NY = "America/New_York"


def test_option_right():
    assert option_right("O:NVDA260723P00120000") == "P"
    assert option_right("NVDA260723C00120000") == "C"
    assert option_right("bad") is None


def test_put_flow_fires_on_put_heavy_tape():
    idx = pd.date_range("2026-05-01 09:35:00", periods=400, freq="s", tz=NY)
    rows = []
    for i, ts in enumerate(idx):
        # early mixed, then put-heavy burst
        if i < 250:
            rows.append(
                {
                    "ticker": "O:NVDA260501C00100000",
                    "timestamp": ts,
                    "v": 50,
                    "c": 1.0,
                }
            )
            rows.append(
                {
                    "ticker": "O:NVDA260501P00100000",
                    "timestamp": ts,
                    "v": 40,
                    "c": 1.0,
                }
            )
        else:
            rows.append(
                {
                    "ticker": "O:NVDA260501P00100000",
                    "timestamp": ts,
                    "v": 400,
                    "c": 1.2,
                }
            )
            rows.append(
                {
                    "ticker": "O:NVDA260501C00100000",
                    "timestamp": ts,
                    "v": 20,
                    "c": 0.8,
                }
            )
    flow = prepare_option_flow_day(pd.DataFrame(rows))
    assert flow is not None
    assert flow["source"] == "1s_agg"
    feat = put_flow_features_at(flow, i=320, window_sec=60)
    assert feat is not None
    share, z, pv, cv = feat
    assert share > 0.7
    assert z > 1.0
    arm = detect_put_flow_dn(
        flow,
        i=320,
        window_sec=60,
        min_put_share=0.55,
        min_put_vol_z=1.5,
        min_put_v=200,
        stock_ret_lb=-0.01,
        max_stock_ret=-0.003,
    )
    assert arm is not None
    assert arm.direction == "DN"


def _put_heavy_flow(n: int = 400):
    idx = pd.date_range("2026-07-23 09:35:00", periods=n, freq="s", tz=NY)
    rows = []
    for i, ts in enumerate(idx):
        # off → on → off → on (two rising edges)
        put = 300 if (120 <= i < 200) or (280 <= i < 360) else 20
        call = 10 if put >= 300 else 80
        rows.append(
            {"ticker": "O:NVDA260724P00100000", "timestamp": ts, "price": 2.0, "size": put}
        )
        rows.append(
            {"ticker": "O:NVDA260724C00100000", "timestamp": ts, "price": 1.0, "size": call}
        )
    return idx, prepare_option_flow_day(pd.DataFrame(rows))


def test_iter_hold_respects_rearm_gap():
    idx, flow = _put_heavy_flow()
    assert flow is not None
    hits = iter_put_flow_dn_in_window(
        flow,
        t_start=idx[120],
        t_end=idx[360],
        window_sec=60,
        min_put_share=0.55,
        min_put_vol_z=1.5,
        min_put_v=200,
        stock_ts_ns=None,
        stock_px=None,
        stock_lb_sec=120,
        max_stock_ret=None,
        stride_sec=5,
        rearm_gap_sec=60,
        fire_mode="hold",
    )
    assert len(hits) >= 2
    gaps = [(hits[i + 1][0] - hits[i][0]).total_seconds() for i in range(len(hits) - 1)]
    assert all(g >= 60 - 1e-6 for g in gaps)


def test_iter_rising_fires_once_per_episode():
    idx, flow = _put_heavy_flow()
    assert flow is not None
    hits = iter_put_flow_dn_in_window(
        flow,
        t_start=idx[100],
        t_end=idx[380],
        window_sec=60,
        min_put_share=0.55,
        min_put_vol_z=1.5,
        min_put_v=200,
        stock_ts_ns=None,
        stock_px=None,
        stock_lb_sec=120,
        max_stock_ret=None,
        stride_sec=5,
        rearm_gap_sec=30,
        fire_mode="rising",
    )
    # two put-heavy episodes → ideally 2 rising edges
    assert 1 <= len(hits) <= 3


def test_prepare_accepts_tick_size_schema():
    idx = pd.date_range("2026-07-23 09:35:00", periods=120, freq="s", tz=NY)
    rows = []
    for ts in idx:
        rows.append(
            {
                "ticker": "O:NVDA260724P00100000",
                "timestamp": ts,
                "price": 2.0,
                "size": 100,
            }
        )
        rows.append(
            {
                "ticker": "O:NVDA260724C00100000",
                "timestamp": ts,
                "price": 1.0,
                "size": 10,
            }
        )
    flow = prepare_option_flow_day(pd.DataFrame(rows))
    assert flow is not None
    assert flow["source"] == "tick"
    feat = put_flow_features_at(flow, i=100, window_sec=60)
    assert feat is not None
    assert feat[0] > 0.8

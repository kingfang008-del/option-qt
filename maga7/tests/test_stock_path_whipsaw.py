"""Unit tests for stock hold-path whipsaw typology."""
from __future__ import annotations

import pandas as pd

from maga7.common.stock_path_whipsaw import (
    analyze_hold_path,
    classify_whipsaw_subtype,
    signed_stock_ret,
)


def test_signed_ret_direction():
    assert abs(signed_stock_ret(101, 100, "UP") - 0.01) < 1e-9
    assert abs(signed_stock_ret(99, 100, "UP") - (-0.01)) < 1e-9
    assert abs(signed_stock_ret(99, 100, "DN") - 0.01) < 1e-9


def test_classify_subtypes():
    assert classify_whipsaw_subtype(-0.0001, -0.0001) == "no_adverse"
    assert classify_whipsaw_subtype(-0.0008, 0.0001) == "shallow_wash_recover"
    assert classify_whipsaw_subtype(-0.0020, 0.0001) == "deep_adverse_recover"
    assert classify_whipsaw_subtype(-0.0020, -0.0018) == "deep_adverse_persist"
    assert classify_whipsaw_subtype(-0.0008, -0.0007) == "shallow_adverse_persist"


def test_analyze_hold_path_whipsaw_then_recover():
    # UP trade: dip 20bp then recover to flat
    ts0 = pd.Timestamp("2026-02-05 10:31:00", tz="America/New_York")
    rows = []
    px = 100.0
    for i in range(0, 601):
        if i <= 120:
            px = 100.0 - 0.20 * (i / 120.0)  # to 99.80
        else:
            # recover to 100.0 by 600s
            px = 99.80 + 0.20 * ((i - 120) / 480.0)
        rows.append(
            {
                "timestamp": ts0 + pd.Timedelta(seconds=i),
                "close": px,
                "volume": 100.0,
            }
        )
    bars = pd.DataFrame(rows)
    m = analyze_hold_path(
        bars,
        entry_ts=ts0,
        exit_ts=ts0 + pd.Timedelta(seconds=600),
        direction="UP",
    )
    assert m.mae is not None and m.mae <= -0.0015
    assert m.subtype == "deep_adverse_recover"
    assert m.h1.mae is not None and m.h1.mae < 0
    assert m.h5.mae is not None and m.h5.mae < 0

"""Causal quote flip helpers smoke tests."""
from __future__ import annotations

import pandas as pd

from maga7.common.fills import FillSpec
from maga7.tools.scan_am_pocket_flip_quote_dual import simulate_quote_flip_causal


def _qpath(rows: list[tuple[str, float, float]]) -> pd.DataFrame:
    ts = pd.to_datetime([r[0] for r in rows])
    ts = ts.tz_localize("America/New_York")
    return pd.DataFrame(
        {
            "timestamp": ts,
            "bid": [r[1] for r in rows],
            "ask": [r[2] for r in rows],
        }
    )


def test_causal_flip_eats_adverse_then_opp():
    prim = _qpath(
        [
            ("2026-05-01 09:30:00", 0.99, 1.01),
            ("2026-05-01 09:30:02", 0.94, 0.96),
            ("2026-05-01 09:30:04", 0.89, 0.91),
            ("2026-05-01 09:30:20", 0.87, 0.89),
        ]
    )
    opp = _qpath(
        [
            ("2026-05-01 09:30:00", 0.99, 1.01),
            ("2026-05-01 09:30:04", 0.99, 1.01),
            ("2026-05-01 09:30:10", 1.05, 1.07),
            ("2026-05-01 09:30:20", 1.13, 1.15),
        ]
    )
    fill = FillSpec(entry_frac=1.0, exit_frac=1.0)
    sim = simulate_quote_flip_causal(
        prim,
        opp,
        pd.Timestamp("2026-05-01 09:30:00", tz="America/New_York"),
        look_t=10,
        dip=0.08,
        prim_tp=0.20,
        prim_sl=0.20,
        prim_max_hold=60,
        opp_tp=0.12,
        opp_sl=0.10,
        opp_max_hold=45,
        fill=fill,
        max_lag_sec=5,
        max_spread_pct=0.5,
        min_mid=0.05,
    )
    assert sim is not None
    assert sim["flipped"] is True
    assert sim["n_legs"] == 2
    assert sim["leg1_ret"] < 0
    assert sim["ret"] > sim["leg1_ret"]

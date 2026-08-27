"""Tests for frozen Top2 decision funnel + first-passage labels."""
from __future__ import annotations

import numpy as np
import pandas as pd

from maga7.common.decision_funnel import (
    FUNNEL_VERSION,
    FunnelConfig,
    build_replacement_chain,
    collect_day_candidates,
    select_top_seats,
)
from maga7.common.first_passage import FirstPassageConfig, first_passage_label

NY = "America/New_York"


def _synth_day(symbol: str, date: str, *, drift: float, n: int = 200) -> pd.DataFrame:
    """1m bars from 09:30 with constant drift."""
    start = pd.Timestamp(f"{date} 09:30:00", tz=NY)
    ts = pd.date_range(start, periods=n, freq="1min")
    px = 100.0 * np.cumprod(1.0 + np.full(n, drift))
    return pd.DataFrame(
        {
            "timestamp": ts,
            "date": date,
            "symbol": symbol,
            "open": px,
            "high": px * 1.0005,
            "low": px * 0.9995,
            "close": px,
            "volume": np.full(n, 1000.0),
        }
    )


def test_funnel_version_frozen():
    assert FUNNEL_VERSION == "top2_smooth_impulse_v1"
    cfg = FunnelConfig()
    assert cfg.max_positions == 2
    assert cfg.smooth.scan_end == "11:30"
    assert cfg.impulse.min_look_ret == 0.004


def test_select_top_seats_unique_symbols():
    cands = [
        {"symbol": "AAA", "detect_ts": "2026-01-02 10:00:00", "score": 1.0},
        {"symbol": "BBB", "detect_ts": "2026-01-02 10:01:00", "score": 2.0},
        {"symbol": "AAA", "detect_ts": "2026-01-02 10:02:00", "score": 3.0},
        {"symbol": "CCC", "detect_ts": "2026-01-02 10:03:00", "score": 4.0},
    ]
    seats = select_top_seats(cands, max_positions=2)
    assert [s["symbol"] for s in seats] == ["AAA", "BBB"]
    assert seats[0]["seat_rank"] == 1


def test_replacement_chain_surfaces_next_name():
    cands = [
        {"symbol": "AAA", "detect_ts": pd.Timestamp("2026-01-02 10:00", tz=NY), "score": 1.0, "direction": "UP", "sleeve": "smooth", "look_ret": 0.01, "path_eff": 0.5, "up_frac": 0.7, "max_dd": -0.001, "from_extreme": 0.002, "price": 100.0, "date": "2026-01-02"},
        {"symbol": "BBB", "detect_ts": pd.Timestamp("2026-01-02 10:01", tz=NY), "score": 1.0, "direction": "UP", "sleeve": "smooth", "look_ret": 0.01, "path_eff": 0.5, "up_frac": 0.7, "max_dd": -0.001, "from_extreme": 0.002, "price": 100.0, "date": "2026-01-02"},
        {"symbol": "CCC", "detect_ts": pd.Timestamp("2026-01-02 10:02", tz=NY), "score": 1.0, "direction": "UP", "sleeve": "smooth", "look_ret": 0.01, "path_eff": 0.5, "up_frac": 0.7, "max_dd": -0.001, "from_extreme": 0.002, "price": 100.0, "date": "2026-01-02"},
    ]
    seats = select_top_seats(cands, max_positions=2)
    alts = build_replacement_chain(cands, seats[0], max_positions=2)
    assert any(a["symbol"] == "CCC" for a in alts)


def test_first_passage_clear_true_on_trend():
    date = "2026-01-02"
    day = _synth_day("NVDA", date, drift=0.0004)  # strong up
    # enter ~10:00 (30 bars in)
    et = pd.Timestamp(f"{date} 10:00:00", tz=NY)
    lab = first_passage_label(
        day,
        entry_ts=et,
        direction="UP",
        date=date,
        cfg=FirstPassageConfig(horizon_minutes=90, good_mfe_pct=0.005, toxic_mae_pct=0.01),
    )
    assert lab is not None
    assert lab["label_pct"] == "clear_true"
    assert lab["y_train_pct"] == 1


def test_first_passage_clear_false_on_fade():
    date = "2026-01-02"
    day = _synth_day("NVDA", date, drift=-0.0003)
    et = pd.Timestamp(f"{date} 10:00:00", tz=NY)
    lab = first_passage_label(
        day,
        entry_ts=et,
        direction="UP",
        date=date,
        cfg=FirstPassageConfig(horizon_minutes=90, good_mfe_pct=0.02, toxic_mae_pct=0.003),
    )
    assert lab is not None
    assert lab["label_pct"] == "clear_false"
    assert lab["y_train_pct"] == 0

from __future__ import annotations

import pandas as pd

from maga7.common.dvol_size_scale import parse_dvol_size_scale, resolve_dvol_size_scale


def _day(sym: str, close: float, volume: float) -> pd.DataFrame:
    idx = pd.date_range("2026-07-22 10:30", periods=5, freq="1min", tz="America/New_York")
    return pd.DataFrame(
        {
            "date": ["2026-07-22"] * len(idx),
            "timestamp": idx,
            "close": [close] * len(idx),
            "volume": [volume] * len(idx),
        }
    )


def test_parse_defaults_disabled():
    cfg = parse_dvol_size_scale(None)
    assert cfg.enabled is False


def test_cs_rank_boosts_top_liquidity():
    stock_by = {
        "NVDA": _day("NVDA", 210.0, 1_000_000),  # highest notional
        "AMD": _day("AMD", 550.0, 100_000),
        "MSFT": _day("MSFT", 390.0, 50_000),
    }
    cfg = parse_dvol_size_scale(
        {
            "enabled": True,
            "mode": "cs_rank",
            "scales": {"1": 1.25, "2": 1.15},
            "default_scale": 1.0,
            "min_scale": 1.0,
            "max_scale": 1.25,
        }
    )
    asof = pd.Timestamp("2026-07-22 11:00", tz="America/New_York")
    s1, r1, _ = resolve_dvol_size_scale(
        cfg, stock_by=stock_by, symbol="NVDA", date="2026-07-22", asof_ts=asof
    )
    s2, r2, _ = resolve_dvol_size_scale(
        cfg, stock_by=stock_by, symbol="AMD", date="2026-07-22", asof_ts=asof
    )
    s3, r3, _ = resolve_dvol_size_scale(
        cfg, stock_by=stock_by, symbol="MSFT", date="2026-07-22", asof_ts=asof
    )
    assert r1 == 1 and abs(s1 - 1.25) < 1e-9
    assert r2 == 2 and abs(s2 - 1.15) < 1e-9
    assert r3 == 3 and abs(s3 - 1.0) < 1e-9


def test_min_scale_prevents_cut():
    stock_by = {"NVDA": _day("NVDA", 210.0, 10_000), "AMD": _day("AMD", 550.0, 1_000_000)}
    cfg = parse_dvol_size_scale(
        {
            "enabled": True,
            "scales": {"1": 1.25, "2": 0.5},  # would cut rank2
            "min_scale": 1.0,
            "max_scale": 1.25,
        }
    )
    asof = pd.Timestamp("2026-07-22 11:00", tz="America/New_York")
    s, r, _ = resolve_dvol_size_scale(
        cfg, stock_by=stock_by, symbol="NVDA", date="2026-07-22", asof_ts=asof
    )
    assert r == 2
    assert abs(s - 1.0) < 1e-9  # clamped by min_scale

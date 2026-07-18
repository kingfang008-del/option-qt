"""Unit tests for open_washout + ORB fractal-high break."""
from __future__ import annotations

import pandas as pd

from maga7.common.orb_open import OrbOpenConfig, detect_open_washout, detect_orb_fractal_break


def _bars(rows: list[tuple[str, float, float, float, float]]) -> pd.DataFrame:
    """rows: (HH:MM, open, high, low, close)"""
    out = []
    for hhmm, o, h, l, c in rows:
        out.append(
            {
                "timestamp": pd.Timestamp(f"2026-05-01 {hhmm}", tz="America/New_York"),
                "open": o,
                "high": h,
                "low": l,
                "close": c,
                "volume": 1000.0,
                "date": "2026-05-01",
            }
        )
    return pd.DataFrame(out)


def test_open_washout_requires_drop():
    # tiny dip — not washout
    df = _bars(
        [
            ("09:30", 100, 100.1, 99.9, 100.0),
            ("09:31", 100, 100.0, 99.85, 99.9),
            ("09:32", 99.9, 99.95, 99.8, 99.85),
        ]
    )
    assert detect_open_washout(df, cfg=OrbOpenConfig(wash_drop_min=0.003)) is None

    # 0.4% wash
    df2 = _bars(
        [
            ("09:30", 100, 100.1, 99.7, 99.8),
            ("09:31", 99.8, 99.85, 99.5, 99.55),
            ("09:32", 99.55, 99.6, 99.4, 99.45),
        ]
    )
    w = detect_open_washout(df2, cfg=OrbOpenConfig(wash_drop_min=0.003))
    assert w is not None and w["wash_drop"] >= 0.003


def test_orb_fractal_break_fires_after_selloff():
    # Selloff 09:30-09:35, fractal high = 09:35 high (99.2), break close at 09:38
    rows = [
        ("09:30", 100.0, 100.2, 99.6, 99.7),
        ("09:31", 99.7, 99.75, 99.4, 99.45),
        ("09:32", 99.45, 99.5, 99.2, 99.25),
        ("09:33", 99.25, 99.3, 99.0, 99.05),
        ("09:34", 99.05, 99.1, 98.8, 98.85),
        ("09:35", 98.85, 99.2, 98.7, 98.75),  # last down; fractal_high=99.2
        ("09:36", 98.75, 99.0, 98.7, 98.9),  # no new low
        ("09:37", 98.9, 99.1, 98.85, 99.05),
        ("09:38", 99.05, 99.4, 99.0, 99.35),  # close > 99.2
    ]
    df = _bars(rows)
    cfg = OrbOpenConfig(wash_drop_min=0.003, selloff_min_bars=3, hold_confirm_bars=0)
    sig = detect_orb_fractal_break(df, symbol="NVDA", date="2026-05-01", cfg=cfg)
    assert sig is not None
    assert sig.direction == "UP"
    assert abs(sig.fractal_high - 99.2) < 1e-9
    assert sig.sig_ts.strftime("%H:%M") == "09:38"


def test_count_open_washout_breadth():
    from maga7.common.orb_open import count_open_washout

    wash = _bars(
        [
            ("09:30", 100, 100.1, 99.5, 99.6),
            ("09:31", 99.6, 99.65, 99.3, 99.35),
            ("09:32", 99.35, 99.4, 99.1, 99.15),
        ]
    )
    flat = _bars(
        [
            ("09:30", 100, 100.2, 99.95, 100.0),
            ("09:31", 100, 100.1, 99.9, 100.05),
            ("09:32", 100.05, 100.2, 100.0, 100.1),
        ]
    )
    stock_by = {"NVDA": wash, "TSLA": wash, "AAPL": wash, "AMZN": flat}
    n, hits = count_open_washout(
        stock_by,
        date="2026-05-01",
        symbols=["NVDA", "TSLA", "AAPL", "AMZN"],
        cfg=OrbOpenConfig(wash_drop_min=0.003),
    )
    assert n == 3
    assert set(hits) == {"NVDA", "TSLA", "AAPL"}


def test_orb_no_fire_without_break():
    rows = [
        ("09:30", 100.0, 100.2, 99.6, 99.7),
        ("09:31", 99.7, 99.75, 99.4, 99.45),
        ("09:32", 99.45, 99.5, 99.2, 99.25),
        ("09:33", 99.25, 99.3, 99.0, 99.05),
        ("09:34", 99.05, 99.1, 98.8, 98.85),
        ("09:35", 98.85, 99.2, 98.7, 98.75),
        ("09:36", 98.75, 99.0, 98.7, 98.9),
        ("09:37", 98.9, 99.15, 98.85, 99.0),
        ("09:38", 99.0, 99.15, 98.9, 99.1),  # never close above 99.2
    ]
    df = _bars(rows)
    sig = detect_orb_fractal_break(
        df, symbol="NVDA", date="2026-05-01", cfg=OrbOpenConfig(wash_drop_min=0.003)
    )
    assert sig is None

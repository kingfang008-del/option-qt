"""Unit tests for path-state exit stress simulator."""
from __future__ import annotations

import pandas as pd

from maga7.common.fills import FillSpec
from maga7.common.option_quote_exit_stress import policy_preset, simulate_quote_exit_stress


def _path(rows: list[tuple[str, float, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime([r[0] for r in rows], utc=True).tz_convert(
                "America/New_York"
            ),
            "bid": [r[1] for r in rows],
            "ask": [r[2] for r in rows],
        }
    )


def test_giveback_after_peak() -> None:
    # peak ≥+10%, then give back 8%+ → giveback before SL
    path = _path(
        [
            ("2026-05-01 09:40:00-04:00", 0.99, 1.01),
            ("2026-05-01 09:40:20-04:00", 1.12, 1.14),  # sell ~+12% after fill
            ("2026-05-01 09:40:40-04:00", 1.01, 1.03),  # giveback from peak
            ("2026-05-01 09:41:00-04:00", 0.78, 0.80),  # would SL
        ]
    )
    fill = FillSpec(0.75, 0.75)
    sim = simulate_quote_exit_stress(
        path,
        pd.Timestamp("2026-05-01 09:40:00-04:00"),
        policy_preset("gb08_p10"),
        fill=fill,
        max_lag_sec=5.0,
        max_spread_pct=0.20,
    )
    assert sim is not None
    assert sim["reason"] == "giveback"
    assert sim["ret"] > -0.20


def test_be_lock_raises_floor() -> None:
    path = _path(
        [
            ("2026-05-01 09:40:00-04:00", 0.99, 1.01),
            ("2026-05-01 09:40:15-04:00", 1.09, 1.11),  # peak >=8% after fill
            ("2026-05-01 09:40:45-04:00", 0.985, 1.005),  # back to ~0 → be_stop
        ]
    )
    fill = FillSpec(0.75, 0.75)
    sim = simulate_quote_exit_stress(
        path,
        pd.Timestamp("2026-05-01 09:40:00-04:00"),
        policy_preset("be_lock08"),
        fill=fill,
        max_lag_sec=5.0,
        max_spread_pct=0.20,
    )
    assert sim is not None
    assert sim["reason"] == "be_stop"
    assert sim["armed"] is True
    assert abs(sim["ret"]) < 0.05


def test_confirm_then_giveback_combo() -> None:
    path = _path(
        [
            ("2026-05-01 10:25:00-04:00", 0.99, 1.01),
            ("2026-05-01 10:25:20-04:00", 1.02, 1.04),  # confirm ~+3%
            ("2026-05-01 10:25:40-04:00", 1.12, 1.14),  # peak ≥10%
            ("2026-05-01 10:26:10-04:00", 1.01, 1.03),  # giveback
        ]
    )
    fill = FillSpec(0.75, 0.75)
    sim = simulate_quote_exit_stress(
        path,
        pd.Timestamp("2026-05-01 10:25:00-04:00"),
        policy_preset("ca_gb08_p10"),
        fill=fill,
        max_lag_sec=5.0,
        max_spread_pct=0.20,
    )
    assert sim is not None
    assert sim["confirmed"] is True
    assert sim["reason"] == "giveback"

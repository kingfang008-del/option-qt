"""Unit tests for post-fill confirm-or-abort quote simulator."""
from __future__ import annotations

import pandas as pd

from maga7.common.fills import FillSpec
from maga7.common.option_quote_tpsl import simulate_quote_tpsl_confirm_abort


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


def test_confirm_abort_timeout_flattens() -> None:
    # Entry mid~1.0; never reaches +3%; at T+60 still flat → confirm_abort
    path = _path(
        [
            ("2026-05-01 09:40:00-04:00", 0.99, 1.01),
            ("2026-05-01 09:40:30-04:00", 0.99, 1.01),
            ("2026-05-01 09:41:05-04:00", 0.985, 1.005),
            ("2026-05-01 09:42:00-04:00", 0.90, 0.92),  # would SL later
        ]
    )
    fill = FillSpec(0.75, 0.75)
    sim = simulate_quote_tpsl_confirm_abort(
        path,
        pd.Timestamp("2026-05-01 09:40:00-04:00"),
        tp=0.15,
        sl=0.20,
        confirm_sec=60,
        confirm_thr=0.03,
        abort_thr=None,
        on_timeout="abort",
        fill=fill,
        max_lag_sec=5.0,
        max_spread_pct=0.20,
    )
    assert sim is not None
    assert sim["reason"] == "confirm_abort"
    assert sim["hold_sec"] <= 70


def test_confirm_then_tp() -> None:
    # Quickly greens +5%, then hits TP
    path = _path(
        [
            ("2026-05-01 09:40:00-04:00", 0.99, 1.01),
            ("2026-05-01 09:40:10-04:00", 1.04, 1.06),  # confirm
            ("2026-05-01 09:40:40-04:00", 1.16, 1.18),  # tp ~15%+
        ]
    )
    fill = FillSpec(0.75, 0.75)
    sim = simulate_quote_tpsl_confirm_abort(
        path,
        pd.Timestamp("2026-05-01 09:40:00-04:00"),
        tp=0.15,
        sl=0.20,
        confirm_sec=120,
        confirm_thr=0.03,
        on_timeout="abort",
        fill=fill,
        max_lag_sec=5.0,
        max_spread_pct=0.20,
    )
    assert sim is not None
    assert sim["confirmed"] is True
    assert sim["reason"] == "tp"


def test_early_abort_before_confirm() -> None:
    path = _path(
        [
            ("2026-05-01 09:40:00-04:00", 0.99, 1.01),
            ("2026-05-01 09:40:20-04:00", 0.88, 0.90),  # ~-10% early abort
            ("2026-05-01 09:41:00-04:00", 0.70, 0.72),
        ]
    )
    fill = FillSpec(0.75, 0.75)
    sim = simulate_quote_tpsl_confirm_abort(
        path,
        pd.Timestamp("2026-05-01 09:40:00-04:00"),
        tp=0.15,
        sl=0.20,
        confirm_sec=120,
        confirm_thr=0.05,
        abort_thr=0.08,
        on_timeout="abort",
        fill=fill,
        max_lag_sec=5.0,
        max_spread_pct=0.25,
    )
    assert sim is not None
    assert sim["reason"] == "early_abort"

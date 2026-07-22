"""Unit tests for causal stock path confirm first-touch gate."""
from __future__ import annotations

import pandas as pd

from maga7.common.replay import stock_path_confirm_ok


def _day(closes: list[float], start: str = "2026-02-18 10:31:00") -> pd.DataFrame:
    ts = pd.date_range(start, periods=len(closes), freq="1min", tz="America/New_York")
    return pd.DataFrame(
        {
            "timestamp": ts,
            "close": closes,
            "mf10": [1.0] * len(closes),
            "streak_up": [1] * len(closes),
            "streak_dn": [0] * len(closes),
            "date": ["2026-02-18"] * len(closes),
        }
    )


def test_path_confirm_pos_before_neg():
    # +20bp by bar 2, never -30bp
    day = _day([100.0, 100.10, 100.20, 100.25])
    ok, ts, reason = stock_path_confirm_ok(
        day,
        direction="UP",
        entry_ts="2026-02-18 10:31:00-05:00",
        thr_pos=0.0015,
        thr_neg=-0.003,
        max_wait_seconds=300,
    )
    assert ok and reason == "pos"
    assert ts is not None
    assert ts.strftime("%H:%M") == "10:33"


def test_path_confirm_neg_first_blocks():
    day = _day([100.0, 99.60, 100.50])  # -40bp then recover
    ok, ts, reason = stock_path_confirm_ok(
        day,
        direction="UP",
        entry_ts="2026-02-18 10:31:00-05:00",
        thr_pos=0.0015,
        thr_neg=-0.003,
        max_wait_seconds=300,
    )
    assert not ok and reason == "neg"
    assert ts is not None


def test_path_confirm_timeout():
    day = _day([100.0, 100.05, 100.08, 100.10, 100.12, 100.14])  # never +15bp
    ok, ts, reason = stock_path_confirm_ok(
        day,
        direction="UP",
        entry_ts="2026-02-18 10:31:00-05:00",
        thr_pos=0.0015,
        thr_neg=-0.003,
        max_wait_seconds=180,
    )
    assert not ok and reason == "timeout"
    assert ts is None


def test_path_confirm_timeout_allow():
    day = _day([100.0, 100.05, 100.08, 100.10, 100.12, 100.14])
    ok, ts, reason = stock_path_confirm_ok(
        day,
        direction="UP",
        entry_ts="2026-02-18 10:31:00-05:00",
        thr_pos=0.0015,
        thr_neg=-0.003,
        max_wait_seconds=180,
        on_timeout="allow",
    )
    assert ok and reason == "timeout_allow"
    assert ts is None


def test_path_confirm_dn():
    day = _day([100.0, 99.90, 99.80])  # -20bp for DN = +20bp signed
    ok, ts, reason = stock_path_confirm_ok(
        day,
        direction="DN",
        entry_ts="2026-02-18 10:31:00-05:00",
        thr_pos=0.0015,
        thr_neg=-0.003,
        max_wait_seconds=300,
    )
    assert ok and reason == "pos"


def test_path_confirm_asof_pending_then_pos():
    day = _day([100.0, 100.10, 100.20, 100.25])
    ok, ts, reason = stock_path_confirm_ok(
        day,
        direction="UP",
        entry_ts="2026-02-18 10:31:00-05:00",
        thr_pos=0.0015,
        thr_neg=-0.003,
        max_wait_seconds=300,
        asof_ts="2026-02-18 10:31:00-05:00",
    )
    assert not ok and reason == "pending"
    assert ts is None
    ok2, ts2, reason2 = stock_path_confirm_ok(
        day,
        direction="UP",
        entry_ts="2026-02-18 10:31:00-05:00",
        thr_pos=0.0015,
        thr_neg=-0.003,
        max_wait_seconds=300,
        asof_ts="2026-02-18 10:33:00-05:00",
    )
    assert ok2 and reason2 == "pos"
    assert ts2 is not None


def test_path_confirm_asof_timeout_allow():
    day = _day([100.0, 100.05, 100.08, 100.10])
    ok, ts, reason = stock_path_confirm_ok(
        day,
        direction="UP",
        entry_ts="2026-02-18 10:31:00-05:00",
        thr_pos=0.0015,
        thr_neg=-0.003,
        max_wait_seconds=120,
        on_timeout="allow",
        asof_ts="2026-02-18 10:32:00-05:00",
    )
    assert not ok and reason == "pending"
    ok2, ts2, reason2 = stock_path_confirm_ok(
        day,
        direction="UP",
        entry_ts="2026-02-18 10:31:00-05:00",
        thr_pos=0.0015,
        thr_neg=-0.003,
        max_wait_seconds=120,
        on_timeout="allow",
        asof_ts="2026-02-18 10:34:00-05:00",
    )
    assert ok2 and reason2 == "timeout_allow"
    assert ts2 is None

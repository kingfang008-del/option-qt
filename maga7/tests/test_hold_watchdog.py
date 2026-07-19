"""Unit tests for hold-period QQQ shock flatten."""
from __future__ import annotations

import pandas as pd

from maga7.common.hold_watchdog import (
    hold_watchdog_from_trade,
    qqq_adverse_from_entry,
    qqq_close_at,
)


def _qqq_day() -> pd.DataFrame:
    # 10:00 = 100, 10:05 = 99.0 (−1%), 10:10 = 98.5 (−1.5% from entry)
    rows = []
    for i, px in enumerate([100.0, 99.5, 99.0, 98.5]):
        ts = pd.Timestamp("2026-05-01 10:00:00", tz="America/New_York") + pd.Timedelta(
            minutes=i * 5
        )
        rows.append({"timestamp": ts, "close": px, "date": "2026-05-01"})
    return pd.DataFrame(rows)


def test_qqq_close_at_and_adverse_up():
    day = _qqq_day()
    entry = pd.Timestamp("2026-05-01 10:00:00", tz="America/New_York")
    now = pd.Timestamp("2026-05-01 10:10:00", tz="America/New_York")
    assert abs(qqq_close_at(day, entry) - 100.0) < 1e-9
    fired, signed = qqq_adverse_from_entry(
        day,
        entry_ts=entry,
        now_ts=now,
        direction="UP",
        thresh=0.008,
        bar_delay_seconds=0,
    )
    assert fired
    assert signed is not None and signed < -0.01


def test_dn_not_fired_on_qqq_drop():
    day = _qqq_day()
    entry = pd.Timestamp("2026-05-01 10:00:00", tz="America/New_York")
    now = pd.Timestamp("2026-05-01 10:10:00", tz="America/New_York")
    fired, signed = qqq_adverse_from_entry(
        day,
        entry_ts=entry,
        now_ts=now,
        direction="DN",
        thresh=0.008,
        bar_delay_seconds=0,
    )
    assert not fired
    assert signed is not None and signed > 0


def test_config_parse():
    cfg = hold_watchdog_from_trade(
        {"hold_watchdog": {"enabled": True, "qqq_adverse_from_entry": 0.01}}
    )
    assert cfg.enabled
    assert abs(cfg.qqq_adverse_from_entry - 0.01) < 1e-12

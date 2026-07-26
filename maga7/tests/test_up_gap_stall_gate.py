from __future__ import annotations

import pandas as pd

from maga7.common.up_gap_stall_gate import (
    parse_up_gap_stall_gate,
    resolve_up_gap_stall_gate,
)


def _day_up_gap_flat() -> pd.DataFrame:
    # Prev close 100 → open 101.8 (gap +1.8%). Session stays near open (fo≈0).
    prev = pd.date_range("2026-06-10 15:50", periods=10, freq="1min", tz="America/New_York")
    day = pd.date_range("2026-06-11 09:30", periods=40, freq="1min", tz="America/New_York")
    rows = []
    for t in prev:
        rows.append(
            {
                "date": "2026-06-10",
                "timestamp": t,
                "open": 100.0,
                "high": 100.2,
                "low": 99.8,
                "close": 100.0,
            }
        )
    for i, t in enumerate(day):
        # Dip then reclaim near day high while staying ~open → chase high, |fo|≈0.
        px = 101.88
        rows.append(
            {
                "date": "2026-06-11",
                "timestamp": t,
                "open": 101.8 if i == 0 else px,
                "high": 101.90,
                "low": 101.70 if i == 1 else 101.80,
                "close": 101.72 if i == 1 else px,
            }
        )
    return pd.DataFrame(rows)


def test_block_up_gap_early_stall():
    cfg = parse_up_gap_stall_gate(
        {
            "enabled": True,
            "min_fav_gap": 0.015,
            "max_abs_from_open": 0.001,
            "min_chase": 0.9,
            "max_sess_min": 40.0,
        }
    )
    df = _day_up_gap_flat()
    asof = pd.Timestamp("2026-06-11 10:02:00", tz="America/New_York")
    d = resolve_up_gap_stall_gate(
        cfg, stock_df=df, date="2026-06-11", asof_ts=asof, direction="UP"
    )
    assert not d.allow
    assert d.fav_gap is not None and d.fav_gap >= 0.015
    assert d.from_open is not None and abs(d.from_open) <= 0.001


def test_pass_when_fo_extends():
    cfg = parse_up_gap_stall_gate({"enabled": True})
    df = _day_up_gap_flat()
    day = df["date"] == "2026-06-11"
    df = df.copy()
    df.loc[day, "close"] = 103.5
    asof = pd.Timestamp("2026-06-11 10:02:00", tz="America/New_York")
    d = resolve_up_gap_stall_gate(
        cfg, stock_df=df, date="2026-06-11", asof_ts=asof, direction="UP"
    )
    assert d.allow and d.reason == "fo_moved"


def test_dn_skipped():
    cfg = parse_up_gap_stall_gate({"enabled": True})
    df = _day_up_gap_flat()
    asof = pd.Timestamp("2026-06-11 10:02:00", tz="America/New_York")
    d = resolve_up_gap_stall_gate(
        cfg, stock_df=df, date="2026-06-11", asof_ts=asof, direction="DN"
    )
    assert d.allow and d.reason == "dir_skip"

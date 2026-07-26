from __future__ import annotations

import pandas as pd

from maga7.common.dn_gap_stall_gate import (
    parse_dn_gap_stall_gate,
    resolve_dn_gap_stall_gate,
)


def _day_dn_gap() -> pd.DataFrame:
    # Prev close 100 → open 98.1 (gap -1.9%). Session drifts to ~97.0 (fo≈-1.1%).
    prev = pd.date_range("2026-02-16 15:50", periods=10, freq="1min", tz="America/New_York")
    day = pd.date_range("2026-02-17 09:30", periods=70, freq="1min", tz="America/New_York")
    rows = []
    for t in prev:
        rows.append(
            {
                "date": "2026-02-16",
                "timestamp": t,
                "open": 100.0,
                "high": 100.2,
                "low": 99.8,
                "close": 100.0,
            }
        )
    # Grind down from 98.1 toward 97.0 over morning.
    for i, t in enumerate(day):
        px = 98.1 - min(i, 60) * (1.1 / 60.0)
        rows.append(
            {
                "date": "2026-02-17",
                "timestamp": t,
                "open": 98.1 if i == 0 else px + 0.05,
                "high": px + 0.1,
                "low": px - 0.1,
                "close": px,
            }
        )
    return pd.DataFrame(rows)


def test_block_dn_gap_mid_extension():
    cfg = parse_dn_gap_stall_gate(
        {
            "enabled": True,
            "min_fav_gap": 0.018,
            "min_fav_from_open": 0.008,
            "max_fav_from_open": 0.014,
            "min_peer": 6,
        }
    )
    df = _day_dn_gap()
    asof = pd.Timestamp("2026-02-17 10:33:00", tz="America/New_York")
    d = resolve_dn_gap_stall_gate(
        cfg, stock_df=df, date="2026-02-17", asof_ts=asof, direction="DN", peer_n=6
    )
    assert not d.allow
    assert d.fav_gap is not None and d.fav_gap >= 0.018
    assert d.fav_from_open is not None
    assert 0.008 <= d.fav_from_open <= 0.014


def test_pass_full_continuation():
    cfg = parse_dn_gap_stall_gate(
        {
            "enabled": True,
            "min_fav_gap": 0.018,
            "min_fav_from_open": 0.008,
            "max_fav_from_open": 0.014,
            "min_peer": 6,
        }
    )
    df = _day_dn_gap()
    # Push session much further → fo above band.
    day = df["date"] == "2026-02-17"
    df = df.copy()
    df.loc[day, "close"] = 95.0
    asof = pd.Timestamp("2026-02-17 10:33:00", tz="America/New_York")
    d = resolve_dn_gap_stall_gate(
        cfg, stock_df=df, date="2026-02-17", asof_ts=asof, direction="DN", peer_n=6
    )
    assert d.allow and d.reason == "fo_high"


def test_up_skipped():
    cfg = parse_dn_gap_stall_gate({"enabled": True})
    df = _day_dn_gap()
    asof = pd.Timestamp("2026-02-17 10:33:00", tz="America/New_York")
    d = resolve_dn_gap_stall_gate(
        cfg, stock_df=df, date="2026-02-17", asof_ts=asof, direction="UP", peer_n=6
    )
    assert d.allow and d.reason == "dir_skip"

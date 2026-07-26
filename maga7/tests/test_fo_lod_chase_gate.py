from __future__ import annotations

import pandas as pd

from maga7.common.fo_lod_chase_gate import (
    parse_fo_lod_chase_gate,
    resolve_fo_lod_chase_gate,
)


def _day_dn_on_lod() -> pd.DataFrame:
    # Open 320, grind to ~309 near LOD → fo≈−3.4%, chase high, dist_ext tiny.
    prev = pd.date_range("2026-07-23 15:50", periods=5, freq="1min", tz="America/New_York")
    day = pd.date_range("2026-07-24 09:30", periods=70, freq="1min", tz="America/New_York")
    rows = []
    for t in prev:
        rows.append(
            {
                "date": "2026-07-23",
                "timestamp": t,
                "open": 319.0,
                "high": 319.5,
                "low": 318.5,
                "close": 319.0,
            }
        )
    for i, t in enumerate(day):
        # Drift down from 320 to 309 over ~60 bars; sit on lows.
        px = 320.0 - min(i, 60) * (11.0 / 60.0)
        rows.append(
            {
                "date": "2026-07-24",
                "timestamp": t,
                "open": 320.0 if i == 0 else px + 0.2,
                "high": px + 0.3,
                "low": px - 0.05,
                "close": px,
            }
        )
    return pd.DataFrame(rows)


def test_block_dn_large_fo_on_lod():
    cfg = parse_fo_lod_chase_gate(
        {
            "enabled": True,
            "min_fav_from_open": 0.03,
            "min_chase": 0.9,
            "max_dist_ext": 0.003,
            "dirs": ["DN"],
        }
    )
    df = _day_dn_on_lod()
    asof = pd.Timestamp("2026-07-24 10:38:00", tz="America/New_York")
    d = resolve_fo_lod_chase_gate(
        cfg, stock_df=df, date="2026-07-24", asof_ts=asof, direction="DN"
    )
    assert not d.allow
    assert d.fav_from_open is not None and d.fav_from_open >= 0.03
    assert d.dist_ext is not None and d.dist_ext <= 0.003


def test_pass_when_off_lod():
    cfg = parse_fo_lod_chase_gate({"enabled": True, "dirs": ["DN"]})
    df = _day_dn_on_lod()
    day = df["date"] == "2026-07-24"
    df = df.copy()
    # Force fo≈3.4% with chase high but >30bp off LOD: open=320, px=309, lo=307.5.
    asof_mask = day & (
        df["timestamp"] <= pd.Timestamp("2026-07-24 10:38:00", tz="America/New_York")
    )
    df.loc[asof_mask, "high"] = 322.0
    df.loc[asof_mask, "low"] = 307.0
    df.loc[asof_mask, "close"] = 308.5  # chase=0.9, dist≈47bp, fo≈3.6%
    asof = pd.Timestamp("2026-07-24 10:38:00", tz="America/New_York")
    d = resolve_fo_lod_chase_gate(
        cfg, stock_df=df, date="2026-07-24", asof_ts=asof, direction="DN"
    )
    assert d.allow and d.reason == "off_extreme"
    assert d.fav_from_open is not None and d.fav_from_open >= 0.03
    assert d.dist_ext is not None and d.dist_ext > 0.003


def test_up_skipped_by_default():
    cfg = parse_fo_lod_chase_gate({"enabled": True})
    df = _day_dn_on_lod()
    asof = pd.Timestamp("2026-07-24 10:38:00", tz="America/New_York")
    d = resolve_fo_lod_chase_gate(
        cfg, stock_df=df, date="2026-07-24", asof_ts=asof, direction="UP"
    )
    assert d.allow and d.reason == "dir_skip"

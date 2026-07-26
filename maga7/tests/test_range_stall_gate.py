from __future__ import annotations

import pandas as pd

from maga7.common.range_stall_gate import (
    parse_range_stall_gate,
    resolve_range_stall_gate,
    session_chase_and_pre5,
)


def _day_up_chase() -> pd.DataFrame:
    # Open 100, grind to high 103 with flat last 5m → chase high, pre5≈0
    idx = pd.date_range("2026-04-06 09:30", periods=60, freq="1min", tz="America/New_York")
    close = [100.0 + min(i, 50) * 0.06 for i in range(60)]  # rises then flat ~103
    high = [c + 0.05 for c in close]
    low = [100.0] * 60
    return pd.DataFrame(
        {
            "date": ["2026-04-06"] * 60,
            "timestamp": idx,
            "open": [100.0] + close[:-1],
            "high": high,
            "low": low,
            "close": close,
        }
    )


def test_block_chase_and_flat_pre5():
    cfg = parse_range_stall_gate(
        {"enabled": True, "min_chase": 0.9, "max_pre5": 0.0, "mode": "block"}
    )
    df = _day_up_chase()
    asof = df["timestamp"].iloc[-1]
    chase, pre5, _ = session_chase_and_pre5(
        df, date="2026-04-06", asof_ts=asof, direction="UP", pre_seconds=300
    )
    assert chase is not None and chase >= 0.9
    assert pre5 is not None and pre5 <= 0.0 + 1e-9
    d = resolve_range_stall_gate(
        cfg, stock_df=df, date="2026-04-06", asof_ts=asof, direction="UP"
    )
    assert not d.allow


def test_pass_when_pre5_positive():
    cfg = parse_range_stall_gate(
        {"enabled": True, "min_chase": 0.9, "max_pre5": 0.0, "mode": "block"}
    )
    df = _day_up_chase()
    # rising last 5 minutes
    df = df.copy()
    df.loc[df.index[-5:], "close"] = [102.0, 102.3, 102.6, 102.9, 103.2]
    df.loc[df.index[-5:], "high"] = df.loc[df.index[-5:], "close"] + 0.05
    asof = df["timestamp"].iloc[-1]
    d = resolve_range_stall_gate(
        cfg, stock_df=df, date="2026-04-06", asof_ts=asof, direction="UP"
    )
    assert d.allow and d.reason == "pre5_ok"


def test_peer_pre5_arm():
    cfg = parse_range_stall_gate(
        {
            "enabled": True,
            "min_chase": 0.99,  # chase arm won't fire
            "max_pre5": 0.0,
            "peer_pre5_max_peer": 3,
            "mode": "block",
        }
    )
    df = _day_up_chase()
    asof = df["timestamp"].iloc[-1]
    hot = resolve_range_stall_gate(
        cfg, stock_df=df, date="2026-04-06", asof_ts=asof, direction="UP", peer_n=3
    )
    assert not hot.allow and "peer<=" in hot.reason
    cool = resolve_range_stall_gate(
        cfg, stock_df=df, date="2026-04-06", asof_ts=asof, direction="UP", peer_n=5
    )
    assert cool.allow


def test_crowd_chase_arm():
    """peer>=7 + chase + looser pre5 + ffo — 02-06-style unanimous chase."""
    cfg = parse_range_stall_gate(
        {
            "enabled": True,
            "min_chase": 0.9,
            "max_pre5": 0.0002,
            "max_peer": 5,  # Arm A skipped when peer=7
            "min_fav_from_open": 0.02,
            "crowd_min_peer": 7,
            "crowd_max_pre5": 0.001,
            "mode": "block",
        }
    )
    df = _day_up_chase()
    asof = df["timestamp"].iloc[-1]
    # mild positive pre5 still inside crowd_max_pre5 via flat tail
    blocked = resolve_range_stall_gate(
        cfg, stock_df=df, date="2026-04-06", asof_ts=asof, direction="UP", peer_n=7
    )
    assert not blocked.allow and "crowd>=" in blocked.reason
    skip = resolve_range_stall_gate(
        cfg, stock_df=df, date="2026-04-06", asof_ts=asof, direction="UP", peer_n=6
    )
    assert skip.allow


def test_crowd_min_fav_independent_of_arm_a():
    """crowd_min_fav_from_open can be looser than Arm A min_fav_from_open."""
    cfg = parse_range_stall_gate(
        {
            "enabled": True,
            "min_chase": 0.9,
            "max_pre5": 0.0002,
            "max_peer": 5,
            "min_fav_from_open": 0.05,  # Arm A too high for this day (~3%)
            "crowd_min_peer": 7,
            "crowd_max_pre5": 0.002,
            "crowd_min_fav_from_open": 0.01,
            "mode": "block",
        }
    )
    df = _day_up_chase()
    asof = df["timestamp"].iloc[-1]
    d = resolve_range_stall_gate(
        cfg, stock_df=df, date="2026-04-06", asof_ts=asof, direction="UP", peer_n=7
    )
    assert not d.allow and "crowd>=" in d.reason

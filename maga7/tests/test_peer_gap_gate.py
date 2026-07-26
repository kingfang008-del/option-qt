from __future__ import annotations

import pandas as pd

from maga7.common.peer_gap_gate import parse_peer_gap_gate, resolve_peer_gap_gate


def _book(prev_close: float, day_open: float) -> pd.DataFrame:
    idx_prev = pd.date_range("2026-04-07 09:30", periods=2, freq="1min", tz="America/New_York")
    idx_day = pd.date_range("2026-04-08 09:30", periods=2, freq="1min", tz="America/New_York")
    return pd.DataFrame(
        {
            "date": ["2026-04-07", "2026-04-07", "2026-04-08", "2026-04-08"],
            "timestamp": list(idx_prev) + list(idx_day),
            "open": [prev_close, prev_close, day_open, day_open],
            "close": [prev_close, prev_close, day_open, day_open],
        }
    )


def test_block_weak_peer_hot_gap():
    cfg = parse_peer_gap_gate(
        {"enabled": True, "min_fav_gap": 0.015, "max_peer": 3, "mode": "block"}
    )
    # +2% gap, peer=3 → block
    d = resolve_peer_gap_gate(
        cfg, stock_df=_book(100.0, 102.0), date="2026-04-08", direction="UP", peer_n=3
    )
    assert not d.allow and d.fav_gap is not None and d.fav_gap >= 0.015


def test_pass_strong_peer_or_small_gap():
    cfg = parse_peer_gap_gate(
        {"enabled": True, "min_fav_gap": 0.015, "max_peer": 3, "mode": "block"}
    )
    strong = resolve_peer_gap_gate(
        cfg, stock_df=_book(100.0, 102.0), date="2026-04-08", direction="UP", peer_n=5
    )
    assert strong.allow and strong.reason == "peer_strong"
    small = resolve_peer_gap_gate(
        cfg, stock_df=_book(100.0, 100.5), date="2026-04-08", direction="UP", peer_n=3
    )
    assert small.allow and small.reason == "gap_small"


def test_max_fav_from_open_filters_extended():
    cfg = parse_peer_gap_gate(
        {
            "enabled": True,
            "min_fav_gap": 0.015,
            "max_peer": 3,
            "max_fav_from_open": 0.01,
            "mode": "block",
        }
    )
    stall = resolve_peer_gap_gate(
        cfg,
        stock_df=_book(100.0, 102.0),
        date="2026-04-08",
        direction="UP",
        peer_n=3,
        from_open=0.003,
    )
    assert not stall.allow
    ext = resolve_peer_gap_gate(
        cfg,
        stock_df=_book(100.0, 102.0),
        date="2026-04-08",
        direction="UP",
        peer_n=3,
        from_open=0.03,
    )
    assert ext.allow and ext.reason == "ffo_extended"

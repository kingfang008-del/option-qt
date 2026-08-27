"""Unit tests for session entry reinforce gates."""
from __future__ import annotations

import pandas as pd

from maga7.common.session_entry_reinforce import (
    SessionReinforceConfig,
    evaluate_reinforce,
)


def _day(sym: str = "NVDA") -> dict[str, pd.DataFrame]:
    ts = pd.date_range("2026-05-01 09:31", periods=5, freq="min", tz="America/New_York")
    df = pd.DataFrame(
        {
            "date": ["2026-05-01"] * 5,
            "timestamp": ts,
            "open": [100.0, 100.2, 100.4, 100.6, 100.8],
            "close": [100.2, 100.4, 100.6, 100.8, 101.0],
            "mf10": [0.1, 0.2, 0.3, 0.4, 0.5],
            "streak_up": [1, 2, 3, 4, 5],
            "streak_dn": [0, 0, 0, 0, 0],
            "vol_z": [0.5, 1.0, 1.5, 2.0, 2.5],
            "from_prev": [0.01] * 5,
        }
    )
    peer = df.copy()
    peer["mf10"] = 0.2
    return {sym: df, "AMD": peer, "AAPL": peer}


def test_base_always_pass():
    ok, meta = evaluate_reinforce(
        stock_by=_day(),
        symbol="NVDA",
        date="2026-05-01",
        entry_ts=pd.Timestamp("2026-05-01 09:35", tz="America/New_York"),
        direction="UP",
        cfg=SessionReinforceConfig(),
        peer_symbols=["NVDA", "AMD", "AAPL"],
    )
    assert ok
    assert meta["reason"] == "pass"


def test_mf_blocks_against():
    stock = _day()
    stock["NVDA"].loc[:, "mf10"] = -0.3
    ok, meta = evaluate_reinforce(
        stock_by=stock,
        symbol="NVDA",
        date="2026-05-01",
        entry_ts=pd.Timestamp("2026-05-01 09:35", tz="America/New_York"),
        direction="UP",
        cfg=SessionReinforceConfig(require_mf=True),
        peer_symbols=["NVDA", "AMD", "AAPL"],
    )
    assert not ok
    assert meta["reason"] == "mf_against"


def test_peer_and_from_open():
    stock = _day()
    # huge from_open chase
    stock["NVDA"]["open"] = 90.0
    ok, meta = evaluate_reinforce(
        stock_by=stock,
        symbol="NVDA",
        date="2026-05-01",
        entry_ts=pd.Timestamp("2026-05-01 09:35", tz="America/New_York"),
        direction="UP",
        cfg=SessionReinforceConfig(require_mf=True, peer_min=2, from_open_max=0.03),
        peer_symbols=["NVDA", "AMD", "AAPL"],
    )
    assert not ok
    assert meta["reason"] == "from_open_chase"

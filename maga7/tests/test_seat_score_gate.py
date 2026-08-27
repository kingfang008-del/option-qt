from __future__ import annotations

import pandas as pd

from maga7.common.seat_score_gate import (
    candidate_gate_active,
    day_gate_armed,
    parse_seat_score_gate,
    seat_score_ok,
)


def _day(close: float, volume: float, vol_z: float = 2.0) -> pd.DataFrame:
    idx = pd.date_range("2026-07-22 10:30", periods=5, freq="1min", tz="America/New_York")
    return pd.DataFrame(
        {
            "date": ["2026-07-22"] * len(idx),
            "timestamp": idx,
            "close": [close] * len(idx),
            "volume": [volume] * len(idx),
            "vol_z": [vol_z] * len(idx),
        }
    )


def test_cs_dvol_skips_low_liquidity():
    stock_by = {
        "NVDA": _day(210.0, 1_000_000),
        "AMD": _day(550.0, 100_000),
        "MSFT": _day(390.0, 10_000),
    }
    cfg = parse_seat_score_gate({"enabled": True, "mode": "cs_dvol_max_rank", "max_rank": 2})
    asof = pd.Timestamp("2026-07-22 11:00", tz="America/New_York")
    ok_n, _, sc_n = seat_score_ok(
        cfg, stock_by=stock_by, symbol="NVDA", date="2026-07-22", asof_ts=asof
    )
    ok_m, reason_m, sc_m = seat_score_ok(
        cfg, stock_by=stock_by, symbol="MSFT", date="2026-07-22", asof_ts=asof
    )
    assert ok_n and sc_n == 1.0
    assert (not ok_m) and sc_m == 3.0 and "max" in reason_m


def test_topk_weak_arms_only_when_earliest_fails():
    stock_by = {
        "NVDA": _day(210.0, 1_000_000),
        "AMD": _day(550.0, 100_000),
        "MSFT": _day(390.0, 10_000),
    }
    cfg = parse_seat_score_gate(
        {
            "enabled": True,
            "mode": "cs_dvol_max_rank",
            "max_rank": 2,
            "when": "topk_weak",
        }
    )
    # Earliest topk both weak (AMD rk2 ok actually - use MSFT+AMD where MSFT is rk3)
    # With volumes NVDA>AMD>MSFT: AMD=2 ok, MSFT=3 fail → any → armed
    topk = pd.DataFrame(
        {
            "date": ["2026-07-22", "2026-07-22"],
            "symbol": ["MSFT", "AMD"],
            "dir": ["DN", "UP"],
            "sig_ts": [
                pd.Timestamp("2026-07-22 10:43", tz="America/New_York"),
                pd.Timestamp("2026-07-22 11:05", tz="America/New_York"),
            ],
            "from_prev": [-0.02, 0.02],
            "vol_z": [1.0, 2.0],
        }
    )
    armed, reason = day_gate_armed(
        cfg, topk_day=topk, stock_by=stock_by, date="2026-07-22"
    )
    assert armed and "topk_weak" in reason

    topk_strong = pd.DataFrame(
        {
            "date": ["2026-07-22", "2026-07-22"],
            "symbol": ["NVDA", "AMD"],
            "dir": ["UP", "UP"],
            "sig_ts": [
                pd.Timestamp("2026-07-22 11:08", tz="America/New_York"),
                pd.Timestamp("2026-07-22 11:05", tz="America/New_York"),
            ],
            "from_prev": [0.02, 0.02],
            "vol_z": [2.0, 2.0],
        }
    )
    armed2, reason2 = day_gate_armed(
        cfg, topk_day=topk_strong, stock_by=stock_by, date="2026-07-22"
    )
    assert (not armed2) and "topk_strong" in reason2


def test_morning_candidate_window():
    cfg = parse_seat_score_gate(
        {
            "enabled": True,
            "when": "morning",
            "tod_start": "10:30",
            "tod_end": "11:30",
        }
    )
    assert candidate_gate_active(
        cfg,
        day_armed=True,
        asof_ts=pd.Timestamp("2026-07-22 11:08", tz="America/New_York"),
    )
    assert not candidate_gate_active(
        cfg,
        day_armed=True,
        asof_ts=pd.Timestamp("2026-07-22 13:15", tz="America/New_York"),
    )
    assert not candidate_gate_active(
        cfg,
        day_armed=False,
        asof_ts=pd.Timestamp("2026-07-22 11:08", tz="America/New_York"),
    )


def test_apply_to_topk_members_spares_backfill():
    cfg = parse_seat_score_gate(
        {
            "enabled": True,
            "when": "always",
            "apply_to": "topk_members",
        }
    )
    ts = pd.Timestamp("2026-07-22 11:08", tz="America/New_York")
    assert candidate_gate_active(
        cfg, day_armed=True, asof_ts=ts, is_topk_member=True
    )
    assert not candidate_gate_active(
        cfg, day_armed=True, asof_ts=ts, is_topk_member=False
    )

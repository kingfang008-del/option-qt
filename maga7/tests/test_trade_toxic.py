"""Unit tests for trade-mark toxic early exit."""
from __future__ import annotations

import pandas as pd

from maga7.common.fills import FillSpec
from maga7.common.option_trades import trade_toxic_from_trade
from maga7.common.replay import simulate_trade


def test_trade_toxic_config_default_off():
    assert trade_toxic_from_trade({}).enabled is False
    assert trade_toxic_from_trade({"trade_toxic": {"enabled": True}}).enabled is True


def test_simulate_trade_trade_toxic_cuts_on_prints():
    ts0 = pd.Timestamp("2026-05-11 10:00:00", tz="America/New_York")
    # Quote path slowly bleeds; trade prints dig to -25% with no MFE.
    qrows = []
    for i, mid in enumerate([1.00, 0.95, 0.90, 0.80, 0.75, 0.70]):
        t = ts0 + pd.Timedelta(minutes=i)
        qrows.append({"timestamp": t, "bid": mid - 0.01, "ask": mid + 0.01})
    path = pd.DataFrame(qrows)
    trows = []
    for i, last in enumerate([1.00, 0.96, 0.88, 0.74, 0.70, 0.68]):
        trows.append({"timestamp": ts0 + pd.Timedelta(minutes=i), "last": last})
    trade_path = pd.DataFrame(trows)
    sim = simulate_trade(
        path,
        ts0,
        fill=FillSpec(0.5, 0.5),
        direction="UP",
        hold_minutes=30,
        trade_path=trade_path,
        trade_toxic={"enabled": True, "cut_ret": 0.25, "mfe_bypass": 0.05, "min_hold_seconds": 60},
    )
    assert sim is not None
    assert sim.reason == "TRADE_TOX"
    assert sim.ret <= -0.19


def test_trade_toxic_tracks_mfe_during_min_hold():
    """MFE printed inside min_hold must still arm the bypass."""
    ts0 = pd.Timestamp("2026-07-13 12:07:00", tz="America/New_York")
    # Quotes every 30s; dig after min_hold.
    qrows = []
    mids = [0.33, 0.38, 0.33, 0.30, 0.24, 0.22]
    for i, mid in enumerate(mids):
        t = ts0 + pd.Timedelta(seconds=30 * i)
        qrows.append({"timestamp": t, "bid": mid - 0.01, "ask": mid + 0.01})
    # Trade prints: +15% MFE at 30s (inside min_hold=60), then dig to -27%.
    trows = [
        {"timestamp": ts0, "last": 0.33},
        {"timestamp": ts0 + pd.Timedelta(seconds=30), "last": 0.38},
        {"timestamp": ts0 + pd.Timedelta(seconds=60), "last": 0.33},
        {"timestamp": ts0 + pd.Timedelta(seconds=90), "last": 0.30},
        {"timestamp": ts0 + pd.Timedelta(seconds=120), "last": 0.24},
        {"timestamp": ts0 + pd.Timedelta(seconds=150), "last": 0.22},
    ]
    sim = simulate_trade(
        pd.DataFrame(qrows),
        ts0,
        fill=FillSpec(0.5, 0.5),
        direction="DN",
        hold_minutes=30,
        sl_mult=0.4,
        trade_path=pd.DataFrame(trows),
        trade_toxic={"enabled": True, "cut_ret": 0.25, "mfe_bypass": 0.05, "min_hold_seconds": 60},
    )
    assert sim is not None
    assert sim.reason != "TRADE_TOX"


def test_trade_toxic_anchors_at_quote_fill_not_signal():
    """Pre-fill trade prints must not create phantom toxic MTM."""
    sig = pd.Timestamp("2026-05-15 09:50:00", tz="America/New_York")
    fill = pd.Timestamp("2026-05-15 10:00:00", tz="America/New_York")
    # Quotes only appear at fill (confirm/gap); then winner path.
    qrows = []
    for i, mid in enumerate([3.10, 3.20, 3.50, 4.00, 5.00]):
        t = fill + pd.Timedelta(minutes=i)
        qrows.append({"timestamp": t, "bid": mid - 0.05, "ask": mid + 0.05})
    path = pd.DataFrame(qrows)
    # Stale high prints between signal and fill, then normal prints at fill.
    trows = [
        {"timestamp": sig, "last": 5.65},
        {"timestamp": sig + pd.Timedelta(minutes=5), "last": 5.20},
        {"timestamp": fill, "last": 3.04},
        {"timestamp": fill + pd.Timedelta(minutes=1), "last": 3.20},
        {"timestamp": fill + pd.Timedelta(minutes=2), "last": 3.50},
        {"timestamp": fill + pd.Timedelta(minutes=3), "last": 4.00},
        {"timestamp": fill + pd.Timedelta(minutes=4), "last": 5.00},
    ]
    sim = simulate_trade(
        path,
        sig,
        fill=FillSpec(0.5, 0.5),
        direction="UP",
        hold_minutes=30,
        tp_mult=1.6,
        trade_path=pd.DataFrame(trows),
        trade_toxic={"enabled": True, "cut_ret": 0.25, "mfe_bypass": 0.05, "min_hold_seconds": 60},
    )
    assert sim is not None
    assert sim.reason != "TRADE_TOX"


def test_trade_toxic_persist_requires_sustained_dig():
    ts0 = pd.Timestamp("2026-05-11 10:00:00", tz="America/New_York")
    # Brief spike through -25% then recover — must not cut with persist=60.
    qrows = []
    mids = [1.00, 0.95, 0.70, 0.95, 0.90, 0.85]
    for i, mid in enumerate(mids):
        t = ts0 + pd.Timedelta(minutes=i)
        qrows.append({"timestamp": t, "bid": mid - 0.01, "ask": mid + 0.01})
    trows = [
        {"timestamp": ts0 + pd.Timedelta(minutes=i), "last": last}
        for i, last in enumerate([1.00, 0.96, 0.70, 0.95, 0.90, 0.85])
    ]
    sim = simulate_trade(
        pd.DataFrame(qrows),
        ts0,
        fill=FillSpec(0.5, 0.5),
        direction="UP",
        hold_minutes=30,
        trade_path=pd.DataFrame(trows),
        trade_toxic={
            "enabled": True,
            "cut_ret": 0.25,
            "mfe_bypass": 0.05,
            "min_hold_seconds": 60,
            "persist_seconds": 60,
        },
    )
    assert sim is not None
    assert sim.reason != "TRADE_TOX"


def test_trade_toxic_max_cut_window_blocks_late_dig():
    ts0 = pd.Timestamp("2026-06-11 10:00:00", tz="America/New_York")
    qrows = []
    trows = []
    for i in range(0, 25):
        t = ts0 + pd.Timedelta(minutes=i)
        # Flat then dig after 15m.
        last = 1.00 if i < 16 else 0.70
        mid = last
        qrows.append({"timestamp": t, "bid": mid - 0.01, "ask": mid + 0.01})
        trows.append({"timestamp": t, "last": last})
    sim = simulate_trade(
        pd.DataFrame(qrows),
        ts0,
        fill=FillSpec(0.5, 0.5),
        direction="UP",
        hold_minutes=30,
        trade_path=pd.DataFrame(trows),
        trade_toxic={
            "enabled": True,
            "cut_ret": 0.25,
            "mfe_bypass": 0.05,
            "min_hold_seconds": 60,
            "max_cut_seconds": 600,
        },
    )
    assert sim is not None
    assert sim.reason != "TRADE_TOX"


def test_trade_toxic_div_softens_mfe_when_stock_flat():
    """Option digs with peak MFE=5% but stock barely moves → soft bypass cuts."""
    ts0 = pd.Timestamp("2026-05-06 10:41:00", tz="America/New_York")
    qrows = []
    # Quote bleeds; trade peaks +5% then digs to -25%.
    for i, mid in enumerate([1.00, 1.05, 0.90, 0.80, 0.74, 0.70]):
        t = ts0 + pd.Timedelta(minutes=i)
        qrows.append({"timestamp": t, "bid": mid - 0.01, "ask": mid + 0.01})
    trows = [
        {"timestamp": ts0 + pd.Timedelta(minutes=i), "last": last}
        for i, last in enumerate([1.00, 1.05, 0.90, 0.80, 0.74, 0.70])
    ]
    # Stock almost flat (adverse << 0.5%).
    srows = []
    for i, px in enumerate([100.0, 100.0, 99.9, 99.8, 99.7, 99.7]):
        srows.append(
            {
                "timestamp": ts0 + pd.Timedelta(minutes=i),
                "close": px,
                "mf10": 0.0,
                "streak_up": 0,
                "streak_dn": 0,
            }
        )
    sim = simulate_trade(
        pd.DataFrame(qrows),
        ts0,
        fill=FillSpec(0.5, 0.5),
        direction="UP",
        hold_minutes=30,
        stock_day=pd.DataFrame(srows),
        trade_path=pd.DataFrame(trows),
        trade_toxic={
            "enabled": True,
            "cut_ret": 0.25,
            "mfe_bypass": 0.05,
            "min_hold_seconds": 60,
            "max_cut_seconds": 600,
            "div_mfe_bypass": 0.08,
            "div_stock_adverse_max": 0.005,
        },
    )
    assert sim is not None
    assert sim.reason == "TRADE_TOX"


def test_simulate_trade_skips_without_trade_path():
    ts0 = pd.Timestamp("2026-05-11 10:00:00", tz="America/New_York")
    path = pd.DataFrame(
        {
            "timestamp": [ts0 + pd.Timedelta(minutes=i) for i in range(5)],
            "bid": [0.99, 0.9, 0.8, 0.7, 0.6],
            "ask": [1.01, 0.92, 0.82, 0.72, 0.62],
        }
    )
    sim = simulate_trade(
        path,
        ts0,
        fill=FillSpec(0.5, 0.5),
        direction="UP",
        hold_minutes=30,
        sl_mult=0.4,
        trade_path=None,
        trade_toxic={"enabled": True, "cut_ret": 0.25, "mfe_bypass": 0.05, "min_hold_seconds": 60},
    )
    assert sim is not None
    assert sim.reason != "TRADE_TOX"

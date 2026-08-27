from __future__ import annotations

import pandas as pd

from maga7.common.am_pulse_scout import (
    AmPulseScout,
    AmPulseScoutConfig,
    parse_am_pulse_scout,
    scan_day,
)


def _grind_dn() -> pd.DataFrame:
    idx = pd.date_range("2026-07-24 09:30", periods=40, freq="1min", tz="America/New_York")
    rows = []
    for i, t in enumerate(idx):
        px = 320.0 - i * 0.35  # crosses -1% around bar ~9
        rows.append(
            {
                "date": "2026-07-24",
                "timestamp": t,
                "open": 320.0 if i == 0 else px + 0.2,
                "high": max(320.0, px) + 0.1,
                "low": px - 0.15,
                "close": px,
            }
        )
    return pd.DataFrame(rows)


def test_fo_alert_before_1030():
    cfg = parse_am_pulse_scout(
        {"enabled": True, "min_fav_from_open": 0.01, "dirs": ["DN"], "lookback_bars": 2}
    )
    alerts = scan_day(_grind_dn(), date="2026-07-24", symbol="TSLA", cfg=cfg)
    assert len(alerts) == 1
    a = alerts[0]
    assert a.event == "AM_SCOUT_ALERT"
    assert a.symbol == "TSLA" and a.dir == "DN" and a.arm == "FO"
    assert a.fav_from_open >= 0.01
    ts = pd.Timestamp(a.ts)
    assert ts.hour * 60 + ts.minute < 10 * 60 + 30


def test_no_alert_after_window():
    cfg = AmPulseScoutConfig(enabled=True, min_fav_from_open=0.01, dirs=("DN",))
    scout = AmPulseScout(cfg=cfg)
    scout.begin_day("2026-07-24")
    # only feed bars after 10:30 with large fo
    a = scout.on_bar(
        symbol="TSLA",
        ts=pd.Timestamp("2026-07-24 10:35:00", tz="America/New_York"),
        open_=320.0,
        high=320.0,
        low=300.0,
        close=300.0,
    )
    assert a is None


def test_one_alert_per_arm():
    cfg = AmPulseScoutConfig(
        enabled=True,
        min_fav_from_open=0.01,
        min_lookback_ret=0.008,
        lookback_bars=2,
        dirs=("DN",),
        max_alerts_per_symbol=1,
    )
    alerts = scan_day(_grind_dn(), date="2026-07-24", symbol="TSLA", cfg=cfg)
    arms = [a.arm for a in alerts]
    assert arms.count("FO") == 1
    # LB may or may not fire before FO depending on grind path; never more than 1 each
    assert arms.count("LB") <= 1
    assert len(alerts) == len(set(arms))


def test_late_bar_does_not_latch_day_open():
    cfg = AmPulseScoutConfig(enabled=True, min_fav_from_open=0.008, dirs=("DN",), rth_open_only=True)
    scout = AmPulseScout(cfg=cfg)
    scout.begin_day("2026-07-24")
    a = scout.on_bar(
        symbol="NVDA",
        ts=pd.Timestamp("2026-07-24 10:13:00", tz="America/New_York"),
        open_=202.84,
        high=203.0,
        low=201.0,
        close=201.03,
    )
    assert a is None
    assert "NVDA" not in scout._day_open


def test_seed_day_open_uses_rth_open():
    cfg = AmPulseScoutConfig(enabled=True, min_fav_from_open=0.008, dirs=("DN",), rth_open_only=True)
    scout = AmPulseScout(cfg=cfg)
    scout.begin_day("2026-07-24")
    scout.seed_day_open("NVDA", 208.02)
    a = scout.on_bar(
        symbol="NVDA",
        ts=pd.Timestamp("2026-07-24 10:13:00", tz="America/New_York"),
        open_=202.84,
        high=203.0,
        low=201.0,
        close=201.03,
    )
    # 3.36% FO from true open — fires, and day_open stays seeded
    assert a is not None and a.arm == "FO" and a.dir == "DN"
    assert abs(a.day_open - 208.02) < 1e-9
    assert a.fav_from_open > 0.03


def test_seed_force_overwrites_pseudo_open():
    cfg = AmPulseScoutConfig(enabled=True, min_fav_from_open=0.008, dirs=("DN",), rth_open_only=True)
    scout = AmPulseScout(cfg=cfg)
    scout.begin_day("2026-07-24")
    scout.seed_day_open("NVDA", 200.0)  # wrong / pseudo
    scout.seed_day_open("NVDA", 208.02, force=True)
    assert abs(scout._day_open["NVDA"] - 208.02) < 1e-9
    # Without force, later seed must not clobber.
    scout.seed_day_open("NVDA", 199.0, force=False)
    assert abs(scout._day_open["NVDA"] - 208.02) < 1e-9


def test_max_fav_from_open_blocks_chase():
    cfg = AmPulseScoutConfig(
        enabled=True,
        min_fav_from_open=0.008,
        max_fav_from_open=0.015,
        dirs=("DN",),
        rth_open_only=True,
    )
    scout = AmPulseScout(cfg=cfg)
    scout.begin_day("2026-07-24")
    scout.seed_day_open("NVDA", 208.02)
    a = scout.on_bar(
        symbol="NVDA",
        ts=pd.Timestamp("2026-07-24 10:13:00", tz="America/New_York"),
        open_=202.84,
        high=203.0,
        low=201.0,
        close=201.03,
    )
    assert a is None

"""Predictive prevention: mixed_wash_up + LiveRegimeGate overlays."""
from __future__ import annotations

import pandas as pd

from maga7.common.predictive_prevention import evaluate_prevention_rule
from maga7.common.watchdog import RegimeWatchdog, WatchdogState, eval_router_rule
from maga7.live.live_regime import LiveRegimeGate


def _day_bars(
    *,
    open_px: float,
    lod: float,
    close_1030: float,
    date: str = "2026-07-20",
) -> pd.DataFrame:
    """Minimal 09:30–10:30 1m path with a wash low before 10:00."""
    rows = []
    base = pd.Timestamp(f"{date} 09:30:00", tz="America/New_York")
    for i in range(61):
        ts = base + pd.Timedelta(minutes=i)
        if i == 0:
            o, h, l, c = open_px, open_px, open_px, open_px
        elif i < 30:
            # dip to lod then bounce a bit
            c = lod + (open_px - lod) * (i / 40.0)
            o, h, l = c, max(c, open_px * 0.999), min(c, lod)
        else:
            c = close_1030
            o, h, l = c, c, min(c, lod)
        rows.append(
            {
                "timestamp": ts,
                "date": date,
                "open": o,
                "high": h,
                "low": l,
                "close": c,
                "volume": 1000.0,
            }
        )
    return pd.DataFrame(rows)


def test_mixed_wash_up_triggers_risk_off_expert():
    date = "2026-07-20"
    # 4 names wash >=0.8%, frac_above ~0.5 (half above open at 10:30)
    stock_by = {
        "TSLA": _day_bars(open_px=100.0, lod=98.0, close_1030=99.0),  # wash, below
        "AAPL": _day_bars(open_px=100.0, lod=98.5, close_1030=99.2),  # wash, below
        "META": _day_bars(open_px=100.0, lod=98.2, close_1030=101.0),  # wash, above
        "AMD": _day_bars(open_px=100.0, lod=98.8, close_1030=100.5),  # wash, above
        "NVDA": _day_bars(open_px=100.0, lod=99.9, close_1030=100.2),  # no wash, above
        "MSFT": _day_bars(open_px=100.0, lod=99.95, close_1030=100.1),
        "AMZN": _day_bars(open_px=100.0, lod=99.9, close_1030=100.3),
        "GOOGL": _day_bars(open_px=100.0, lod=99.9, close_1030=99.8),
    }
    # QQQ slight low-open reclaim but bounce <0.8% so classic halt misses
    qqq = _day_bars(open_px=700.0, lod=698.5, close_1030=700.5)
    # stitch prev day close for low_open_reclaim helpers
    prev = pd.DataFrame(
        [
            {
                "timestamp": pd.Timestamp("2026-07-17 15:59:00", tz="America/New_York"),
                "date": "2026-07-17",
                "open": 705.0,
                "high": 705.0,
                "low": 705.0,
                "close": 705.0,
                "volume": 1.0,
            }
        ]
    )
    qqq = pd.concat([prev, qqq], ignore_index=True)
    symbols = list(stock_by)
    hit = eval_router_rule(
        "mixed_wash_up",
        date=date,
        stock_by=stock_by,
        qqq_df=qqq,
        symbols=symbols,
        asof_hhmm="10:30",
        router_cfg={
            "prefer_risk_off": True,
            "washout_breadth_min": 3,
            "wash_drop_min": 0.008,
            "frac_above_min": 0.35,
            "frac_above_max": 0.70,
        },
    )
    assert hit == "up_toxic_block"
    soft = evaluate_prevention_rule(
        date=date,
        stock_by=stock_by,
        qqq_df=qqq,
        symbols=symbols,
        prefer_risk_off=False,
    )
    assert soft == "up_toxic"


def test_watchdog_prevention_lane_blocks_via_expert():
    prof = {
        "watchdog": {
            "enabled": True,
            "asof": "10:30",
            "experts": {
                "up_toxic_block": {"regime": {"block_directions": ["UP"]}},
                "up_toxic": {"regime": {"direction_size_scale": {"UP": 0.5}}},
            },
            "degrade": {"enabled": False},
            "halt": {"enabled": False},
            "prevention": {
                "enabled": True,
                "rule": "mixed_wash_up",
                "prefer_risk_off": True,
                "washout_breadth_min": 3,
                "wash_drop_min": 0.008,
                "frac_above_min": 0.35,
                "frac_above_max": 0.70,
            },
            "hunter": {"enabled": False},
        }
    }
    wd = RegimeWatchdog.from_profile(prof)
    assert wd is not None
    assert wd.cfg.prevention_enabled
    date = "2026-07-20"
    stock_by = {
        "TSLA": _day_bars(open_px=100.0, lod=98.0, close_1030=99.0),
        "AAPL": _day_bars(open_px=100.0, lod=98.5, close_1030=99.2),
        "META": _day_bars(open_px=100.0, lod=98.2, close_1030=101.0),
        "AMD": _day_bars(open_px=100.0, lod=98.8, close_1030=100.5),
        "NVDA": _day_bars(open_px=100.0, lod=99.9, close_1030=100.2),
        "MSFT": _day_bars(open_px=100.0, lod=99.95, close_1030=100.1),
        "AMZN": _day_bars(open_px=100.0, lod=99.9, close_1030=100.3),
        "GOOGL": _day_bars(open_px=100.0, lod=99.9, close_1030=99.8),
    }
    qqq = _day_bars(open_px=700.0, lod=698.5, close_1030=700.5)
    dec = wd.begin_day(date, stock_by=stock_by, qqq_df=qqq, symbols=list(stock_by))
    assert dec.reason.startswith("prevention:")
    assert dec.expert == "up_toxic_block"
    assert dec.state == WatchdogState.DEGRADE
    assert "UP" in (dec.overlay.regime_patch.get("block_directions") or [])


def test_live_regime_honors_block_directions():
    gate = LiveRegimeGate({"qqq_align": False, "block_directions": ["UP"]})
    gate.qqq_previous_close = 100.0
    gate.qqq_close = 100.5
    gate.qqq_state.bars = [{"open": 100.0, "close": 100.5}]
    up = gate.check("UP", pd.Timestamp("2026-07-20 10:31", tz="America/New_York"))
    dn = gate.check("DN", pd.Timestamp("2026-07-20 10:31", tz="America/New_York"))
    assert up.allow is False
    assert "block_dir" in up.reason
    assert dn.allow is True

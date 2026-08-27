from __future__ import annotations

import pandas as pd

from maga7.common.delta_time_stop import (
    StockRevExitConfig,
    stock_rev_applies_to_route,
    stock_rev_day_should_arm,
    stock_rev_exit_from_trade,
)
from maga7.common.fills import FillSpec
from maga7.common.replay import simulate_trade

NY = "America/New_York"


def _opt_path(start: str, n: int, entry: float, step: float) -> pd.DataFrame:
    ts = pd.date_range(start, periods=n, freq="1min", tz=NY)
    px = [entry + i * step for i in range(n)]
    return pd.DataFrame({"timestamp": ts, "bid": px, "ask": [p + 0.05 for p in px]})


def _stock_path(start: str, n: int, entry: float, step: float) -> pd.DataFrame:
    ts = pd.date_range(start, periods=n, freq="1min", tz=NY)
    px = [entry + i * step for i in range(n)]
    return pd.DataFrame(
        {"timestamp": ts, "open": px, "high": px, "low": px, "close": px, "volume": 1000}
    )


def test_stock_rev_config_default_off():
    assert stock_rev_exit_from_trade({}).enabled is False
    assert stock_rev_exit_from_trade({"stock_rev_exit": {"enabled": True}}).enabled


def test_stock_rev_when_mixed_wash_parsed():
    cfg = stock_rev_exit_from_trade(
        {"stock_rev_exit": {"enabled": True, "when": "mixed_wash_up"}}
    )
    assert cfg.when == "mixed_wash_up"
    assert stock_rev_day_should_arm(
        StockRevExitConfig(enabled=False),
        date="2026-05-01",
        stock_by={},
        qqq_df=None,
        symbols=["NVDA"],
    ) is False
    assert stock_rev_day_should_arm(
        StockRevExitConfig(enabled=True, when="always"),
        date="2026-05-01",
        stock_by={},
        qqq_df=None,
        symbols=["NVDA"],
    ) is True


def test_stock_rev_hunt_only_routes():
    cfg = stock_rev_exit_from_trade(
        {"stock_rev_exit": {"enabled": True, "hunt_only": True, "stock_max": -0.005}}
    )
    assert cfg.routes == ("hunt",)
    assert stock_rev_applies_to_route(cfg, "hunt")
    assert not stock_rev_applies_to_route(cfg, "baseline")
    cfg2 = stock_rev_exit_from_trade(
        {"stock_rev_exit": {"enabled": True, "routes": ["hunt", "baseline"]}}
    )
    assert stock_rev_applies_to_route(cfg2, "baseline")
    assert stock_rev_applies_to_route(
        stock_rev_exit_from_trade({"stock_rev_exit": {"enabled": True}}), "baseline"
    )


def test_stock_rev_fires_when_underlying_flips():
    opt = _opt_path("2026-05-01 10:30", 40, 1.0, -0.002)
    stock = _stock_path("2026-05-01 10:30", 40, 100.0, -0.05)
    sim = simulate_trade(
        opt,
        entry_ts=pd.Timestamp("2026-05-01 10:30", tz=NY),
        fill=FillSpec(0.5, 0.5),
        tp_mult=10.0,
        sl_mult=0.9,
        hold_minutes=30,
        direction="UP",
        stock_day=stock,
        exit_mode="hold_extend",
        hold_extend_minutes=45,
        hold_extend_require_mf=False,
        stock_bar_delay_seconds=0,
        stock_rev_exit={
            "enabled": True,
            "min_hold_minutes": 10,
            "stock_max": 0.0,
            "opt_mtm_max": 0.50,
        },
    )
    assert sim is not None
    assert sim.reason == "STOCK_REV"


def test_stock_rev_spares_when_stock_still_up():
    opt = _opt_path("2026-05-01 10:30", 40, 1.0, 0.001)
    stock = _stock_path("2026-05-01 10:30", 40, 100.0, 0.05)
    sim = simulate_trade(
        opt,
        entry_ts=pd.Timestamp("2026-05-01 10:30", tz=NY),
        fill=FillSpec(0.5, 0.5),
        tp_mult=10.0,
        sl_mult=0.9,
        hold_minutes=30,
        direction="UP",
        stock_day=stock,
        exit_mode="none",
        stock_bar_delay_seconds=0,
        stock_rev_exit={
            "enabled": True,
            "min_hold_minutes": 10,
            "stock_max": 0.0,
            "opt_mtm_max": 0.50,
        },
    )
    assert sim is not None
    assert sim.reason != "STOCK_REV"

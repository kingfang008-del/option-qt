"""Unit tests for causal 5m adverse soft gate."""
from __future__ import annotations

import pandas as pd

from maga7.common.delta_time_stop import adverse_soft_from_trade
from maga7.common.fills import FillSpec
from maga7.common.replay import simulate_trade


def _quotes(ts0, mids):
    rows = []
    for i, mid in enumerate(mids):
        t = ts0 + pd.Timedelta(minutes=i)
        rows.append({"timestamp": t, "bid": mid - 0.01, "ask": mid + 0.01})
    return pd.DataFrame(rows)


def _stock(ts0, n, px_fn, direction="UP"):
    rows = []
    for i in range(n):
        t = ts0 + pd.Timedelta(minutes=i)
        px = float(px_fn(i))
        rows.append(
            {
                "timestamp": t,
                "close": px,
                "mf10": 1.0 if direction == "UP" else -1.0,
                "streak_up": 1 if direction == "UP" else 0,
                "streak_dn": 0 if direction == "UP" else 1,
                "high": px,
                "low": px,
            }
        )
    return pd.DataFrame(rows)


def test_adverse_soft_default_off():
    assert adverse_soft_from_trade({}).enabled is False
    cfg = adverse_soft_from_trade(
        {"adverse_soft": {"enabled": True, "mode": "soft_exit", "adverse_mae": 0.002}}
    )
    assert cfg.enabled and cfg.mode == "soft_exit"
    assert cfg.adverse_mae == 0.002


def test_soft_exit_on_deep_adverse():
    ts0 = pd.Timestamp("2026-02-18 10:31:00", tz="America/New_York")
    # Option red; stock dumps >15bp by minute 5.
    sim = simulate_trade(
        _quotes(ts0, [1.00, 0.95, 0.90, 0.88, 0.85, 0.82, 0.80]),
        ts0,
        fill=FillSpec(0.5, 0.5),
        direction="UP",
        hold_minutes=30,
        stock_day=_stock(ts0, 12, lambda i: 100.0 * (1.0 - 0.0005 * i)),
        exit_mode="hold_extend",
        adverse_soft={
            "enabled": True,
            "mode": "soft_exit",
            "check_seconds": 300,
            "adverse_mae": 0.0015,
            "opt_mtm_max": 0.0,
        },
    )
    assert sim is not None
    assert sim.reason == "ADVERSE_SOFT"
    assert sim.ret < 0


def test_tox_tighten_arms_without_hard_exit():
    ts0 = pd.Timestamp("2026-02-18 10:31:00", tz="America/New_York")
    # Stock adverse, option drifts but never hits tightened tox without trade path.
    # Without trade_path, tox is off → should hold to T+30, but armed=True.
    sim = simulate_trade(
        _quotes(ts0, [1.00] + [0.92] * 35),
        ts0,
        fill=FillSpec(0.5, 0.5),
        direction="UP",
        hold_minutes=30,
        stock_day=_stock(ts0, 40, lambda i: 100.0 * (1.0 - 0.0006 * min(i, 10))),
        exit_mode="none",
        adverse_soft={
            "enabled": True,
            "mode": "tox_tighten",
            "check_seconds": 300,
            "adverse_mae": 0.0015,
            "opt_mtm_max": 0.0,
            "tight_cut_ret": 0.15,
        },
    )
    assert sim is not None
    assert sim.reason.startswith("T+")
    assert sim.adverse_soft_armed is True


def test_no_soft_exit_when_mae_shallow():
    ts0 = pd.Timestamp("2026-02-18 10:31:00", tz="America/New_York")
    # Stock adverse but MAE only ~8bp — below deep threshold.
    sim = simulate_trade(
        _quotes(ts0, [1.00, 0.95, 0.90, 0.88, 0.85, 0.82, 0.80] + [0.78] * 25),
        ts0,
        fill=FillSpec(0.5, 0.5),
        direction="UP",
        hold_minutes=30,
        stock_day=_stock(ts0, 40, lambda i: 100.0 * (1.0 - 0.00015 * min(i, 8))),
        exit_mode="none",
        adverse_soft={
            "enabled": True,
            "mode": "soft_exit",
            "check_seconds": 300,
            "adverse_mae": 0.0015,
            "opt_mtm_max": 0.0,
        },
    )
    assert sim is not None
    assert sim.reason != "ADVERSE_SOFT"
    assert sim.adverse_soft_armed is False


def test_no_soft_exit_when_option_green():
    ts0 = pd.Timestamp("2026-02-18 10:31:00", tz="America/New_York")
    sim = simulate_trade(
        _quotes(ts0, [1.00, 1.05, 1.10, 1.15, 1.20, 1.25, 1.30]),
        ts0,
        fill=FillSpec(0.5, 0.5),
        direction="UP",
        hold_minutes=30,
        stock_day=_stock(ts0, 12, lambda i: 100.0 * (1.0 - 0.0005 * i)),
        exit_mode="none",
        adverse_soft={
            "enabled": True,
            "mode": "soft_exit",
            "check_seconds": 300,
            "adverse_mae": 0.0015,
            "opt_mtm_max": 0.0,
        },
    )
    assert sim is not None
    assert sim.reason != "ADVERSE_SOFT"


def test_max_opt_mfe_blocks_soft_exit():
    ts0 = pd.Timestamp("2026-02-18 10:31:00", tz="America/New_York")
    # Early green peak then dig; stock dumps. Narrow gate requires peak < 5%.
    sim = simulate_trade(
        _quotes(ts0, [1.00, 1.10, 0.95, 0.90, 0.85, 0.80, 0.75] + [0.70] * 25),
        ts0,
        fill=FillSpec(0.5, 0.5),
        direction="UP",
        hold_minutes=30,
        stock_day=_stock(ts0, 40, lambda i: 100.0 * (1.0 - 0.0005 * i)),
        exit_mode="none",
        adverse_soft={
            "enabled": True,
            "mode": "soft_exit",
            "check_seconds": 300,
            "adverse_mae": 0.0015,
            "opt_mtm_max": 0.0,
            "max_opt_mfe": 0.05,
        },
    )
    assert sim is not None
    assert sim.reason != "ADVERSE_SOFT"


def test_still_adverse_required():
    ts0 = pd.Timestamp("2026-02-18 10:31:00", tz="America/New_York")
    # Deep MAE early, then stock recovers by minute 5 → still_adverse blocks.

    def px(i):
        if i <= 2:
            return 100.0 * (1.0 - 0.001 * i)  # to -20bp
        return 100.0 * (1.0 + 0.001 * (i - 2))  # recover

    sim = simulate_trade(
        _quotes(ts0, [1.00, 0.95, 0.90, 0.88, 0.85, 0.82, 0.80] + [0.78] * 25),
        ts0,
        fill=FillSpec(0.5, 0.5),
        direction="UP",
        hold_minutes=30,
        stock_day=_stock(ts0, 40, px),
        exit_mode="none",
        adverse_soft={
            "enabled": True,
            "mode": "soft_exit",
            "check_seconds": 300,
            "adverse_mae": 0.0015,
            "opt_mtm_max": 0.0,
            "require_still_adverse": True,
            "still_adverse_max": -0.0010,
        },
    )
    assert sim is not None
    assert sim.reason != "ADVERSE_SOFT"

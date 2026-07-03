#!/usr/bin/env python3
"""Tests for bidirectional Phase 2–4 modules."""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pytz

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bidirectional_regime import (
    DayType,
    oracle_side_from_returns,
    pick_tradable_side,
    resolve_day_type,
)
from exec_profile import ExecProfile, resolve_effective_exec_profile, resolve_exec_band, resolve_exec_plan
from strategy_config0 import StrategyConfig
from strategy_core_v0 import StrategyCoreV0

NY = pytz.timezone("America/New_York")
CFG = StrategyConfig()
CORE = StrategyCoreV0(CFG)


def _ctx(**kwargs):
    base = {
        "symbol": "QQQ",
        "time": datetime(2025, 6, 2, 11, 0, tzinfo=NY),
        "curr_ts": 1_700_000_000.0,
        "is_ready": True,
        "is_banned": False,
        "position": 0,
        "cooldown_until": 0,
        "alpha": 0.025,
        "alpha_z": 0.025,
        "vol_z": 1.2,
        "stock_roc": -0.002,
        "spy_roc": -0.002,
        "qqq_roc": -0.002,
        "qqq_day_roc": -0.005,
        "index_trend": -1,
        "macd_hist": -0.012,
        "snap_roc": -0.0012,
        "curr_price": 0.45,
        "bid": 0.44,
        "ask": 0.46,
        "options_vw_spread": 0.06,
        "options_iv_momentum": 0.05,
        "exec_profile": "auto_hybrid",
    }
    base.update(kwargs)
    return base


def test_resolve_day_type_trend_down():
    dt = resolve_day_type(_ctx(qqq_day_roc=-0.006, stock_roc=-0.004))
    assert dt == DayType.TREND_DOWN


def test_pick_tradable_side_put_wins():
    d, a, _ = pick_tradable_side(0.01, 0.022, threshold=0.015)
    assert d == -1
    assert a < 0


def test_oracle_side():
    assert oracle_side_from_returns(0.03, 0.01) == 1
    assert oracle_side_from_returns(0.01, 0.04) == -1


def test_dislocation_put_entry_on_bear_day():
    sig = CORE._try_dislocation_entry(_ctx(alpha=-0.02, alpha_z=-0.02))
    assert sig is not None
    assert sig["dir"] == -1
    assert "CH_DISLOC_PUT" in sig["reason"]


def test_index_guard_allows_put_on_trend_down():
    ctx = _ctx(day_type="trend_down", spy_roc=-0.001, qqq_roc=-0.001)
    assert CORE._check_index_guard(ctx, -1) is True


def test_index_guard_blocks_put_on_strong_rally():
    ctx = _ctx(day_type="trend_down", spy_roc=0.002, qqq_roc=0.002)
    assert CORE._check_index_guard(ctx, -1) is False


def test_resolve_exec_band_put_snap():
    band, _ = resolve_exec_band(
        _ctx(alpha=-0.02, alpha_z=-0.02, curr_price=0.40, snap_roc=-0.001),
        CFG,
    )
    assert band is not None


def test_regime_routing_chop_to_scalp():
    prof, reason = resolve_effective_exec_profile(
        ExecProfile.AUTO_HYBRID,
        _ctx(qqq_day_roc=0.001, stock_roc=0.0002, snap_roc=0.0001),
        CFG,
        regime_routing_enabled=True,
    )
    assert prof == ExecProfile.SCALP_0DTE
    assert "chop" in reason


def test_regime_dislocation_to_multi_band():
    prof, _ = resolve_effective_exec_profile(
        ExecProfile.AUTO_HYBRID,
        _ctx(qqq_day_roc=0.001, stock_roc=-0.002, snap_roc=0.0012, alpha=0.02),
        CFG,
        regime_routing_enabled=True,
    )
    assert prof == ExecProfile.MULTI_BAND


def test_resolve_exec_plan_includes_regime_detail():
    plan = resolve_exec_plan(
        ExecProfile.AUTO_HYBRID,
        _ctx(qqq_day_roc=0.001, stock_roc=0.0002),
        CFG,
        regime_routing_enabled=True,
    )
    assert "regime" in plan.reason


def test_simple_trend_entry_call():
    sig = CORE.decide_entry(_ctx(
        alpha=0.025, alpha_z=0.025, stock_roc=0.002,
        spy_roc=0.0003, qqq_roc=0.0003, macd_hist=0.005,
        curr_price=2.5, exec_profile="auto_hybrid",
    ))
    assert sig is not None
    assert sig["dir"] == 1
    assert "CH_TREND" in sig["reason"]


def test_simple_trend_entry_put_on_bear_day():
    sig = CORE.decide_entry(_ctx(
        alpha=-0.025, alpha_z=-0.025, stock_roc=-0.002,
        qqq_day_roc=-0.006, day_type="trend_down",
        spy_roc=-0.0005, qqq_roc=-0.0005, macd_hist=-0.005,
        curr_price=2.5, exec_profile="auto_hybrid",
    ))
    assert sig is not None
    assert sig["dir"] == -1


def test_simple_dislocation_beats_trend():
    sig = CORE.decide_entry(_ctx(
        alpha=0.02, curr_price=0.40, snap_roc=0.001,
        stock_roc=-0.01, macd_hist=-0.01,
    ))
    assert sig is not None
    assert "CH_DISLOC" in sig["reason"]

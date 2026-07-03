#!/usr/bin/env python3
"""Unit tests for exec_profile routing (Path A / Path C)."""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pytz

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from exec_profile import (
    ExecBand,
    ExecMode,
    ExecProfile,
    effective_exit_param,
    multi_band_roll_cooldown_seconds,
    parse_exec_profile,
    resolve_exec_band,
    resolve_exec_plan,
)
from strategy_config0 import StrategyConfig

NY = pytz.timezone("America/New_York")


def _ctx(**kwargs):
    base = {
        "alpha": 0.02,
        "alpha_z": 0.02,
        "spy_roc": 0.0003,
        "qqq_roc": 0.0003,
        "options_vw_spread": 0.05,
        "options_iv_momentum": 0.1,
        "is_volatile_regime": False,
        "time": datetime(2025, 6, 2, 11, 0, tzinfo=NY),
    }
    base.update(kwargs)
    return base


def test_parse_exec_profile_aliases():
    assert parse_exec_profile("scalp") == ExecProfile.SCALP_0DTE
    assert parse_exec_profile("hybrid") == ExecProfile.AUTO_HYBRID
    assert parse_exec_profile("swing_1dte") == ExecProfile.SWING_1DTE


def test_scalp_profile_fixed():
    cfg = StrategyConfig()
    plan = resolve_exec_plan(ExecProfile.SCALP_0DTE, _ctx(), cfg)
    assert plan.mode == ExecMode.SCALP
    assert plan.target_dte == 0


def test_swing_profile_fixed():
    cfg = StrategyConfig()
    plan = resolve_exec_plan(ExecProfile.SWING_1DTE, _ctx(), cfg)
    assert plan.mode == ExecMode.SWING
    assert plan.target_dte == 1


def test_auto_hybrid_routes_scalp_on_strong_edge():
    cfg = StrategyConfig()
    plan = resolve_exec_plan(ExecProfile.AUTO_HYBRID, _ctx(alpha_z=0.035), cfg)
    assert plan.mode == ExecMode.SCALP


def test_auto_hybrid_routes_swing_on_weak_edge():
    cfg = StrategyConfig()
    plan = resolve_exec_plan(ExecProfile.AUTO_HYBRID, _ctx(alpha_z=0.018), cfg)
    assert plan.mode == ExecMode.SWING


def test_effective_exit_param_overlays():
    cfg = StrategyConfig()
    scalp_pos = {"exec_mode": "SCALP"}
    swing_pos = {"exec_mode": "SWING"}
    assert effective_exit_param(cfg, scalp_pos, "TIME_STOP_MINS") == cfg.SCALP_TIME_STOP_MINS
    assert effective_exit_param(cfg, swing_pos, "TIME_STOP_MINS") == cfg.SWING_TIME_STOP_MINS
    assert effective_exit_param(cfg, swing_pos, "TIME_STOP_MINS") != cfg.SCALP_TIME_STOP_MINS


def test_parse_multi_band_aliases():
    assert parse_exec_profile("multi_band") == ExecProfile.MULTI_BAND
    assert parse_exec_profile("roll") == ExecProfile.MULTI_BAND


def test_resolve_exec_band_tiers():
    cfg = StrategyConfig()
    b1, _ = resolve_exec_band(_ctx(curr_price=0.55, snap_roc=0.001, alpha_z=0.02), cfg)
    b2, _ = resolve_exec_band(_ctx(curr_price=1.20), cfg)
    b3, _ = resolve_exec_band(_ctx(curr_price=2.50, stock_roc=0.0005), cfg)
    assert b1 == ExecBand.BAND1
    assert b2 == ExecBand.BAND2
    assert b3 == ExecBand.BAND3


def test_multi_band_max_legs_blocks():
    cfg = StrategyConfig()
    plan = resolve_exec_plan(ExecProfile.MULTI_BAND, _ctx(curr_price=1.0), cfg, legs_today=3)
    assert plan.hold_profile == "band_blocked"


def test_band_exit_overlay_priority():
    cfg = StrategyConfig()
    band_pos = {"exec_band": "BAND1", "exec_mode": "SWING"}
    assert effective_exit_param(cfg, band_pos, "TIME_STOP_MINS") == cfg.BAND1_TIME_STOP_MINS


def test_multi_band_roll_cooldown():
    cfg = StrategyConfig()
    assert multi_band_roll_cooldown_seconds(cfg, profitable=True, reason="TIME_STOP") == 8 * 60
    assert multi_band_roll_cooldown_seconds(cfg, profitable=True, reason="TRAIL_EXIT") == 8 * 60
    assert multi_band_roll_cooldown_seconds(cfg, profitable=False, reason="HARD_STOP") == cfg.COOLDOWN_MINUTES * 60


def test_dislocation_band1_entry_gate():
    from strategy_core_v0 import StrategyCoreV0

    cfg = StrategyConfig()
    strategy = StrategyCoreV0(cfg)
    ctx = _ctx(
        exec_profile="multi_band",
        curr_price=0.55,
        snap_roc=0.0012,
        alpha=0.018,
        alpha_z=0.018,
        stock_roc=-0.003,
        bid=0.52,
        ask=0.58,
        options_vw_spread=0.10,
        macd_hist=0.005,
        vol_z=1.8,
        position=0,
        cooldown_until=0,
        curr_ts=1.0,
        is_ready=True,
        is_banned=False,
        spread_divergence=0.0,
        time=datetime(2025, 6, 2, 9, 51, tzinfo=NY),
    )
    sig = strategy.decide_entry(ctx)
    assert sig is not None
    assert str(sig.get("reason", "")).startswith("CH_DISLOC")


if __name__ == "__main__":
    test_parse_exec_profile_aliases()
    test_scalp_profile_fixed()
    test_swing_profile_fixed()
    test_auto_hybrid_routes_scalp_on_strong_edge()
    test_auto_hybrid_routes_swing_on_weak_edge()
    test_effective_exit_param_overlays()
    test_parse_multi_band_aliases()
    test_resolve_exec_band_tiers()
    test_multi_band_max_legs_blocks()
    test_band_exit_overlay_priority()
    test_multi_band_roll_cooldown()
    test_dislocation_band1_entry_gate()
    print("ok")

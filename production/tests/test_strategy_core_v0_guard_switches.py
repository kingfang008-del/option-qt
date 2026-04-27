#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path


def _bootstrap_imports() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    production_dir = repo_root / "production"
    baseline_dir = production_dir / "baseline"
    sys.path.insert(0, str(production_dir))
    sys.path.insert(0, str(baseline_dir))


def _entry_ctx(*, curr_price: float = 2.1, bid: float = 2.0, ask: float = 2.2) -> dict:
    return {
        "symbol": "NVDA",
        "time": datetime(2026, 4, 25, 10, 0, 0),
        "curr_ts": 1_777_100_000.0,
        "price": 100.0,
        "alpha_z": 1.2,
        "vol_z": 0.5,
        "stock_roc": 0.002,
        "macd_hist": 0.03,
        "spy_roc": 0.002,
        "qqq_roc": 0.002,
        "index_trend": 1,
        "position": 0,
        "cooldown_until": 0.0,
        "is_ready": True,
        "is_banned": False,
        "curr_price": curr_price,
        "bid": bid,
        "ask": ask,
        "spread_divergence": 0.0,
        "snap_roc": 0.002,
        "regime_reversal_count": 0,
        "is_volatile_regime": False,
    }


def _holding(*, entry_index_trend: int = 0, max_roi: float = 0.1) -> dict:
    return {
        "entry_price": 2.0,
        "entry_stock": 100.0,
        "entry_ts": 1_777_100_000.0,
        "dir": 1,
        "max_roi": max_roi,
        "entry_spy_roc": 0.0,
        "entry_index_trend": entry_index_trend,
    }


def _exit_ctx(
    *,
    held_mins: float = 6.0,
    curr_price: float = 2.2,
    curr_stock: float = 100.0,
    alpha_z: float = 0.5,
    index_trend: int = 0,
    bid: float = 2.18,
    ask: float = 2.22,
    macd_hist_slope: float = 0.01,
    max_roi: float = 0.1,
    entry_index_trend: int = 0,
) -> dict:
    curr_ts = 1_777_100_000.0 + held_mins * 60.0
    return {
        "symbol": "NVDA",
        "time": datetime(2026, 4, 25, 10, 0, 0),
        "curr_ts": curr_ts,
        "price": curr_stock,
        "alpha_z": alpha_z,
        "vol_z": 0.0,
        "stock_roc": 0.0,
        "event_prob": 0.0,
        "macd_hist": 0.01,
        "macd_hist_slope": macd_hist_slope,
        "spy_roc": 0.0,
        "qqq_roc": 0.0,
        "index_trend": index_trend,
        "position": 1,
        "cooldown_until": 0.0,
        "is_ready": True,
        "is_banned": False,
        "held_mins": held_mins,
        "stock_iv": 0.3,
        "holding": _holding(entry_index_trend=entry_index_trend, max_roi=max_roi),
        "curr_price": curr_price,
        "curr_stock": curr_stock,
        "bid": bid,
        "ask": ask,
        "spread_divergence": 0.0,
        "snap_roc": 0.0,
        "regime_reversal_count": 0,
        "is_volatile_regime": False,
        "regime_band": "calm",
        "regime_score": 0.0,
    }


def test_entry_liquidity_guard_can_be_disabled() -> None:
    _bootstrap_imports()
    from strategy_core_v0 import StrategyCoreV0  # noqa: E402
    from strategy_config0 import StrategyConfig  # noqa: E402

    cfg = StrategyConfig()
    core = StrategyCoreV0(cfg)
    tight_spread_ctx = _entry_ctx(curr_price=2.0, bid=1.8, ask=2.2)
    assert core.decide_entry(tight_spread_ctx) is None, "默认应被流动性门禁拦住"

    cfg.ENTRY_LIQUIDITY_GUARD_ENABLED = False
    core = StrategyCoreV0(cfg)
    sig = core.decide_entry(tight_spread_ctx)
    assert sig is not None, "关闭 ENTRY_LIQUIDITY_GUARD_ENABLED 后应允许信号通过"


def test_entry_momentum_guard_can_be_disabled() -> None:
    _bootstrap_imports()
    from strategy_core_v0 import StrategyCoreV0  # noqa: E402
    from strategy_config0 import StrategyConfig  # noqa: E402

    cfg = StrategyConfig()
    core = StrategyCoreV0(cfg)
    weak_momentum_ctx = _entry_ctx(curr_price=2.05, bid=2.00, ask=2.10)
    weak_momentum_ctx["stock_roc"] = -0.003
    weak_momentum_ctx["snap_roc"] = -0.002
    assert core.decide_entry(weak_momentum_ctx) is None, "默认应被 ENTRY momentum guard 拦住"

    cfg.ENTRY_MOMENTUM_GUARD_ENABLED = False
    core = StrategyCoreV0(cfg)
    sig = core.decide_entry(weak_momentum_ctx)
    assert sig is not None, "关闭 ENTRY_MOMENTUM_GUARD_ENABLED 后应允许通过动量校验"


def test_exit_counter_trend_guard_can_be_disabled() -> None:
    _bootstrap_imports()
    from strategy_core_v0 import StrategyCoreV0  # noqa: E402
    from strategy_config0 import StrategyConfig  # noqa: E402

    cfg = StrategyConfig()
    core = StrategyCoreV0(cfg)
    ctx = _exit_ctx(held_mins=6.0, index_trend=-1, entry_index_trend=-1)
    sig = core._check_exit_pre_conditions(ctx, ctx["holding"])
    assert sig is None, f"默认 10 分钟前不应触发 CT_TIMEOUT，实际: {sig}"

    ctx = _exit_ctx(held_mins=10.0, index_trend=-1, entry_index_trend=-1)
    sig = core._check_exit_pre_conditions(ctx, ctx["holding"])
    assert sig and "CT_TIMEOUT" in str(sig.get("reason", "")), f"满 10 分钟应触发 CT_TIMEOUT，实际: {sig}"

    cfg.EXIT_COUNTER_TREND_ENABLED = False
    core = StrategyCoreV0(cfg)
    sig = core._check_exit_pre_conditions(ctx, ctx["holding"])
    assert sig is None, "关闭 EXIT_COUNTER_TREND_ENABLED 后不应触发 CT_TIMEOUT"


def test_exit_index_reversal_guard_can_be_disabled() -> None:
    _bootstrap_imports()
    from strategy_core_v0 import StrategyCoreV0  # noqa: E402
    from strategy_config0 import StrategyConfig  # noqa: E402

    cfg = StrategyConfig()
    core = StrategyCoreV0(cfg)
    ctx = _exit_ctx(held_mins=3.0, index_trend=-1, entry_index_trend=1)
    sig = core._check_trend_reversal_guard(ctx, ctx["holding"])
    assert sig is None, f"默认关闭指数反转离场，实际: {sig}"

    cfg.EXIT_INDEX_REVERSAL_ENABLED = True
    core = StrategyCoreV0(cfg)
    sig = core._check_trend_reversal_guard(ctx, ctx["holding"])
    assert sig and "IDX_REVERSAL" in str(sig.get("reason", "")), f"显式开启后应触发指数反转离场，实际: {sig}"


def test_exit_stock_hard_stop_and_liquidity_guard_can_be_disabled() -> None:
    _bootstrap_imports()
    from strategy_core_v0 import StrategyCoreV0  # noqa: E402
    from strategy_config0 import StrategyConfig  # noqa: E402

    cfg = StrategyConfig()
    core = StrategyCoreV0(cfg)
    ctx = _exit_ctx(held_mins=3.0, curr_stock=99.0, bid=1.8, ask=2.4, curr_price=2.0)
    stock_sig = core._check_stock_hard_stop(ctx, ctx["holding"], 3.0, -0.01)
    spread_sig = core._check_exit_liquidity_guard(ctx, ctx["curr_price"])
    assert stock_sig and "STOCK_STOP" in str(stock_sig.get("reason", ""))
    assert spread_sig and "SPREAD_STOP" in str(spread_sig.get("reason", ""))

    cfg.EXIT_STOCK_HARD_STOP_ENABLED = False
    cfg.EXIT_LIQUIDITY_GUARD_ENABLED = False
    core = StrategyCoreV0(cfg)
    assert core._check_stock_hard_stop(ctx, ctx["holding"], 3.0, -0.01) is None
    assert core._check_exit_liquidity_guard(ctx, ctx["curr_price"]) is None


def test_exit_small_gain_and_zombie_guard_can_be_disabled() -> None:
    _bootstrap_imports()
    from strategy_core_v0 import StrategyCoreV0  # noqa: E402
    from strategy_config0 import StrategyConfig  # noqa: E402

    cfg = StrategyConfig()
    cfg.NO_MOMENTUM_MINS = 999
    cfg.MID_TIME_STOP_MINS = 999
    core = StrategyCoreV0(cfg)

    zombie_sig = core._check_time_and_inactivity_stops(
        _exit_ctx(held_mins=21.0, curr_price=2.02, max_roi=0.05),
        _holding(max_roi=0.05),
        21.0,
        0.01,
    )
    assert zombie_sig and "ZOMBIE_STOP" in str(zombie_sig.get("reason", ""))

    small_gain_sig = core._check_time_and_inactivity_stops(
        _exit_ctx(held_mins=16.0, curr_price=2.06, max_roi=0.10),
        _holding(max_roi=0.10),
        16.0,
        0.03,
    )
    assert small_gain_sig and "SMALL_GAIN" in str(small_gain_sig.get("reason", ""))

    cfg.EXIT_ZOMBIE_STOP_ENABLED = False
    cfg.EXIT_SMALL_GAIN_ENABLED = False
    core = StrategyCoreV0(cfg)
    assert core._check_time_and_inactivity_stops(_exit_ctx(held_mins=21.0, curr_price=2.02, max_roi=0.05), _holding(max_roi=0.05), 21.0, 0.01) is None
    assert core._check_time_and_inactivity_stops(_exit_ctx(held_mins=16.0, curr_price=2.06, max_roi=0.10), _holding(max_roi=0.10), 16.0, 0.03) is None


def test_exit_cond_stop_macd_fade_and_signal_flip_can_be_disabled() -> None:
    _bootstrap_imports()
    from strategy_core_v0 import StrategyCoreV0  # noqa: E402
    from strategy_config0 import StrategyConfig  # noqa: E402

    cfg = StrategyConfig()
    core = StrategyCoreV0(cfg)

    cond_ctx = _exit_ctx(held_mins=3.0, curr_price=1.78, curr_stock=99.0, max_roi=0.02)
    cond_sig = core._check_stop_loss_guards(cond_ctx, cond_ctx["holding"], -0.11, -0.01)
    assert cond_sig and "COND_STOP" in str(cond_sig.get("reason", ""))

    macd_ctx = _exit_ctx(held_mins=3.0, curr_price=2.08, macd_hist_slope=-0.02, max_roi=0.05)
    macd_sig = core._check_macd_fade(macd_ctx, macd_ctx["holding"], 3.0, 0.04)
    assert macd_sig and "MACD_FADE" in str(macd_sig.get("reason", ""))

    flip_ctx = _exit_ctx(held_mins=3.0, curr_price=2.20, alpha_z=-1.0, max_roi=0.10)
    flip_sig = core._check_signal_flip(flip_ctx, flip_ctx["holding"], 3.0)
    assert flip_sig and "FLIP" in str(flip_sig.get("reason", ""))

    cfg.EXIT_COND_STOP_ENABLED = False
    cfg.EXIT_MACD_FADE_ENABLED = False
    cfg.EXIT_SIGNAL_FLIP_ENABLED = False
    core = StrategyCoreV0(cfg)
    assert core._check_stop_loss_guards(cond_ctx, cond_ctx["holding"], -0.11, -0.01) is None
    assert core._check_macd_fade(macd_ctx, macd_ctx["holding"], 3.0, 0.04) is None
    assert core._check_signal_flip(flip_ctx, flip_ctx["holding"], 3.0) is None


def main() -> None:
    test_entry_liquidity_guard_can_be_disabled()
    test_entry_momentum_guard_can_be_disabled()
    test_exit_counter_trend_guard_can_be_disabled()
    test_exit_index_reversal_guard_can_be_disabled()
    test_exit_stock_hard_stop_and_liquidity_guard_can_be_disabled()
    test_exit_small_gain_and_zombie_guard_can_be_disabled()
    test_exit_cond_stop_macd_fade_and_signal_flip_can_be_disabled()
    print("[OK] strategy core v0 guard switch tests passed")


if __name__ == "__main__":
    main()

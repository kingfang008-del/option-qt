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


def _mk_ctx(*, curr_ts: float, entry_ts: float, curr_price: float, max_roi: float) -> dict:
    return {
        "symbol": "NVDA",
        "time": datetime(2026, 4, 24, 10, 6, 0),
        "curr_ts": curr_ts,
        "price": 100.0,
        "alpha_z": 1.0,
        "vol_z": 0.0,
        "stock_roc": 0.0,
        "event_prob": 0.0,
        "macd_hist": 0.01,
        "macd_hist_slope": 0.0,
        "spy_roc": 0.0,
        "qqq_roc": 0.0,
        "index_trend": 0,
        "position": 1,
        "cooldown_until": 0.0,
        "is_ready": True,
        "is_banned": False,
        "held_mins": (curr_ts - entry_ts) / 60.0,
        "stock_iv": 0.3,
        "holding": {
            "symbol": "NVDA",
            "entry_price": 1.0,
            "entry_stock": 100.0,
            "entry_ts": entry_ts,
            "dir": 1,
            "max_roi": max_roi,
            "entry_spy_roc": 0.0,
            "entry_index_trend": 0,
        },
        "curr_price": curr_price,
        "curr_stock": 100.0,
        "bid": curr_price * 0.995,
        "ask": curr_price * 1.005,
        "spread_divergence": 0.0,
        "snap_roc": 0.0,
        "global_regime_reversal_cnt": 0,
        "regime_reversal_count": 0,
        "is_volatile_regime": False,
        "regime_band": "calm",
        "regime_score": 0.0,
    }


def test_v0_no_momentum_triggers_at_five_minutes_when_current_roi_stays_weak() -> None:
    _bootstrap_imports()
    from strategy_core_v0 import StrategyCoreV0  # noqa: E402
    from strategy_config0 import StrategyConfig  # noqa: E402

    core = StrategyCoreV0(StrategyConfig())
    entry_ts = 1_777_000_000.0

    ctx = _mk_ctx(
        curr_ts=entry_ts + 5 * 60,
        entry_ts=entry_ts,
        curr_price=1.01,
        max_roi=0.03,
    )
    sig = core.check_exit(ctx)

    assert sig is not None, "满 5 分钟且当前 roi < 2% 应触发平仓"
    assert "NO_MOMENTUM" in str(sig.get("reason", "")), f"应触发 NO_MOMENTUM，实际: {sig}"


def test_v0_no_momentum_skips_when_current_roi_reached_threshold() -> None:
    _bootstrap_imports()
    from strategy_core_v0 import StrategyCoreV0  # noqa: E402
    from strategy_config0 import StrategyConfig  # noqa: E402

    core = StrategyCoreV0(StrategyConfig())
    entry_ts = 1_777_000_000.0

    ctx = _mk_ctx(
        curr_ts=entry_ts + 6 * 60,
        entry_ts=entry_ts,
        curr_price=1.03,
        max_roi=0.03,
    )
    sig = core.check_exit(ctx)

    assert not (sig and "NO_MOMENTUM" in str(sig.get("reason", ""))), f"当前 roi 已超过阈值时不应触发 NO_MOMENTUM，实际: {sig}"


def test_v0_no_momentum_skips_before_five_minutes() -> None:
    _bootstrap_imports()
    from strategy_core_v0 import StrategyCoreV0  # noqa: E402
    from strategy_config0 import StrategyConfig  # noqa: E402

    core = StrategyCoreV0(StrategyConfig())
    entry_ts = 1_777_000_000.0

    ctx = _mk_ctx(
        curr_ts=entry_ts + 4 * 60 + 59,
        entry_ts=entry_ts,
        curr_price=1.0,
        max_roi=0.0,
    )
    sig = core.check_exit(ctx)

    assert not (sig and "NO_MOMENTUM" in str(sig.get("reason", ""))), f"未满 5 分钟时不应触发 NO_MOMENTUM，实际: {sig}"


def test_v0_mid_time_stop_triggers_after_fifteen_minutes_when_roi_below_five_percent() -> None:
    _bootstrap_imports()
    from strategy_core_v0 import StrategyCoreV0  # noqa: E402
    from strategy_config0 import StrategyConfig  # noqa: E402

    core = StrategyCoreV0(StrategyConfig())
    entry_ts = 1_777_000_000.0

    ctx = _mk_ctx(
        curr_ts=entry_ts + 15 * 60,
        entry_ts=entry_ts,
        curr_price=1.04,
        max_roi=0.06,
    )
    sig = core.check_exit(ctx)

    assert sig is not None, "满 15 分钟且当前 roi < 5% 应触发中期时间止损"
    assert "TIME_STOP15" in str(sig.get("reason", "")), f"应触发 TIME_STOP15，实际: {sig}"


def test_v0_mid_time_stop_skips_when_roi_reached_five_percent() -> None:
    _bootstrap_imports()
    from strategy_core_v0 import StrategyCoreV0  # noqa: E402
    from strategy_config0 import StrategyConfig  # noqa: E402

    core = StrategyCoreV0(StrategyConfig())
    entry_ts = 1_777_000_000.0

    ctx = _mk_ctx(
        curr_ts=entry_ts + 15 * 60,
        entry_ts=entry_ts,
        curr_price=1.05,
        max_roi=0.06,
    )
    sig = core.check_exit(ctx)

    assert not (sig and "TIME_STOP15" in str(sig.get("reason", ""))), f"当前 roi 已达到 5% 时不应触发 TIME_STOP15，实际: {sig}"


def main() -> None:
    test_v0_no_momentum_triggers_at_five_minutes_when_current_roi_stays_weak()
    test_v0_no_momentum_skips_when_current_roi_reached_threshold()
    test_v0_no_momentum_skips_before_five_minutes()
    test_v0_mid_time_stop_triggers_after_fifteen_minutes_when_roi_below_five_percent()
    test_v0_mid_time_stop_skips_when_roi_reached_five_percent()
    print("[OK] strategy core v0 no momentum guards passed")


if __name__ == "__main__":
    main()

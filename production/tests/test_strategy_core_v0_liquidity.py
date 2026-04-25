#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import sys
from pathlib import Path


def _bootstrap_imports() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    production_dir = repo_root / "production"
    baseline_dir = production_dir / "baseline"
    sys.path.insert(0, str(production_dir))
    sys.path.insert(0, str(baseline_dir))


def _ctx(alpha_z: float, bid: float, ask: float, curr_price: float) -> dict:
    return {
        "alpha_z": alpha_z,
        "bid": bid,
        "ask": ask,
        "curr_price": curr_price,
        "spread_divergence": 0.0,
    }


def test_v0_call_uses_tighter_spread_cap_than_put() -> None:
    _bootstrap_imports()
    from strategy_core_v0 import StrategyCoreV0  # noqa: E402
    from strategy_config0 import StrategyConfig  # noqa: E402

    core = StrategyCoreV0(StrategyConfig())

    # 价格 2.0 时，原动态阈值约 16.67%，因此实际应由方向阈值决定：
    # CALL 用 8%，PUT 用 10%。
    call_ctx = _ctx(alpha_z=1.0, bid=1.91, ask=2.09, curr_price=2.0)  # 9%
    put_ctx = _ctx(alpha_z=-1.0, bid=1.91, ask=2.09, curr_price=2.0)  # 9%

    assert core._check_entry_liquidity_guard(call_ctx) is False, "CALL 方向 9% 点差应被 8% 阈值拦截"
    assert core._check_entry_liquidity_guard(put_ctx) is True, "PUT 方向 9% 点差应允许通过 10% 阈值"


def test_v0_blocks_entry_when_option_price_below_minimum() -> None:
    _bootstrap_imports()
    from strategy_core_v0 import StrategyCoreV0  # noqa: E402
    from strategy_config0 import StrategyConfig  # noqa: E402

    cfg = StrategyConfig()
    cfg.MIN_OPTION_PRICE = 1.0
    core = StrategyCoreV0(cfg)

    cheap_ctx = _ctx(alpha_z=1.0, bid=0.92, ask=0.98, curr_price=0.95)
    ok_ctx = _ctx(alpha_z=1.0, bid=1.00, ask=1.08, curr_price=1.04)

    assert core._check_entry_liquidity_guard(cheap_ctx) is False, "低于 MIN_OPTION_PRICE 的期权应被拦截"
    assert core._check_entry_liquidity_guard(ok_ctx) is True, "高于 MIN_OPTION_PRICE 的期权不应因最低价被拦截"


def main() -> None:
    test_v0_call_uses_tighter_spread_cap_than_put()
    test_v0_blocks_entry_when_option_price_below_minimum()
    print("[OK] strategy core v0 liquidity guards passed")


if __name__ == "__main__":
    main()

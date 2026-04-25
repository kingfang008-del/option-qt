#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace


def _bootstrap_imports() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    production_dir = repo_root / "production"
    baseline_dir = production_dir / "baseline"
    sys.path.insert(0, str(production_dir))
    sys.path.insert(0, str(baseline_dir))


def test_shared_entry_liquidity_matches_v0_strategy_guard() -> None:
    _bootstrap_imports()
    from entry_risk_rules import evaluate_entry_liquidity  # noqa: E402
    from strategy_core_v0 import StrategyCoreV0  # noqa: E402
    from strategy_config0 import StrategyConfig  # noqa: E402

    cfg = StrategyConfig()
    core = StrategyCoreV0(cfg)
    ctx = {
        "alpha_z": 1.0,
        "bid": 1.91,
        "ask": 2.09,
        "curr_price": 2.00,
        "spread_divergence": 0.0,
    }

    decision = evaluate_entry_liquidity(
        bid=ctx["bid"],
        ask=ctx["ask"],
        curr_price=ctx["curr_price"],
        alpha_z=ctx["alpha_z"],
        spread_divergence=ctx["spread_divergence"],
        cfg=cfg,
    )

    assert decision["ok"] is False
    assert decision["reason"] == "spread_too_wide"
    assert core._check_entry_liquidity_guard(ctx) is False


def test_shared_entry_min_price_uses_same_effective_floor_everywhere() -> None:
    _bootstrap_imports()
    from entry_risk_rules import get_entry_min_option_price  # noqa: E402
    from strategy_config0 import StrategyConfig  # noqa: E402

    cfg = StrategyConfig()
    cfg.MIN_OPTION_PRICE = 2.0

    assert get_entry_min_option_price(cfg) == 2.0
    assert get_entry_min_option_price(2.0) == 2.0
    assert get_entry_min_option_price(0.0) == 0.05


def test_shared_entry_liquidity_rejects_low_price_before_execution() -> None:
    _bootstrap_imports()
    from entry_risk_rules import evaluate_entry_liquidity  # noqa: E402
    from strategy_config0 import StrategyConfig  # noqa: E402

    cfg = StrategyConfig()
    cfg.MIN_OPTION_PRICE = 2.0
    decision = evaluate_entry_liquidity(
        bid=1.80,
        ask=1.90,
        curr_price=1.85,
        alpha_z=1.0,
        spread_divergence=0.0,
        cfg=cfg,
    )

    assert decision["ok"] is False
    assert decision["reason"] == "min_option_price"
    assert "min_option_price=2.000" in decision["detail"]


def main() -> None:
    test_shared_entry_liquidity_matches_v0_strategy_guard()
    test_shared_entry_min_price_uses_same_effective_floor_everywhere()
    test_shared_entry_liquidity_rejects_low_price_before_execution()
    print("[OK] entry risk rules tests passed")


if __name__ == "__main__":
    main()

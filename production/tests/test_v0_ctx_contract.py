#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import logging
import math
import sys
import types
from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest.mock import patch


def _bootstrap_imports() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    production_dir = repo_root / "production"
    baseline_dir = production_dir / "baseline"
    history_replay_dir = production_dir / "history_replay"
    dao_dir = baseline_dir / "DAO"
    sys.path.insert(0, str(production_dir))
    sys.path.insert(0, str(baseline_dir))
    sys.path.insert(0, str(history_replay_dir))
    sys.path.insert(0, str(dao_dir))


def _load_execution_engine_module():
    sys.modules.setdefault("torch", types.ModuleType("torch"))
    scipy_mod = types.ModuleType("scipy")
    scipy_stats_mod = types.ModuleType("scipy.stats")
    scipy_stats_mod.norm = object()
    scipy_mod.stats = scipy_stats_mod
    sys.modules.setdefault("scipy", scipy_mod)
    sys.modules.setdefault("scipy.stats", scipy_stats_mod)
    ibkr_stub = types.ModuleType("ibkr_connector_v8")
    ibkr_stub.IBKRConnectorFinal = object
    sys.modules["ibkr_connector_v8"] = ibkr_stub
    with patch.object(logging, "FileHandler", lambda *_args, **_kwargs: logging.NullHandler()):
        import execution_engine_v8 as ee  # noqa: E402
    return ee


def _state(**overrides):
    base = dict(
        position=0,
        last_valid_iv=0.30,
        last_opt_price=0.0,
        entry_price=0.0,
        entry_stock=0.0,
        entry_ts=0.0,
        max_roi=0.0,
        entry_spy_roc=0.0,
        entry_index_trend=0,
        last_spread_pct=0.0,
        cooldown_until=0.0,
        last_alpha_z=0.0,
        warmup_complete=False,
        correction_mode="NORMAL",
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _build_engine(ee, state, *, mode="realtime", latest_quote=None):
    engine = SimpleNamespace(
        mode=mode,
        states={"NVDA": state},
        latest_execution_quote_by_symbol={"NVDA": latest_quote or {}},
        global_cooldown_until=0.0,
    )
    engine._get_fair_market_price = MethodType(ee.ExecutionEngineV8._get_fair_market_price, engine)
    engine._execution_quote_freshness = MethodType(ee.ExecutionEngineV8._execution_quote_freshness, engine)
    engine._calc_trading_minutes = MethodType(ee.ExecutionEngineV8._calc_trading_minutes, engine)
    engine._build_strategy_ctx = MethodType(ee.ExecutionEngineV8._build_strategy_ctx, engine)
    return engine


def test_v0_ctx_for_flat_position_uses_alpha_direction_and_feed_mid() -> None:
    _bootstrap_imports()
    ee = _load_execution_engine_module()
    st = _state(position=0, last_valid_iv=0.20)
    engine = _build_engine(ee, st, mode="realtime")

    item = {
        "symbol": "NVDA",
        "stock_price": 100.0,
        "alpha": 1.20,
        "cs_alpha_z": 1.10,
        "vol_z": 0.40,
        "roc_5m": 0.003,
        "event_prob": 0.05,
        "macd": 0.02,
        "macd_slope": 0.01,
        "snap_roc": 0.001,
        "is_ready": True,
        "correction_mode": "NORMAL",
    }
    opt_data = {
        "has_feed": True,
        "call_bid": 2.00,
        "call_ask": 2.20,
        "call_price": 2.10,
        "call_iv": 0.40,
        "put_bid": 1.80,
        "put_ask": 2.00,
        "put_price": 1.90,
        "put_iv": 0.50,
    }
    frame = {
        "index_trend": 1,
        "global_regime_reversal_cnt": 4,
        "global_is_volatile_regime": False,
        "global_regime_band": "calm",
        "global_regime_score": 0.20,
    }

    ctx, market_opt_price, ctx_curr_price, ctx_bid, ctx_ask = engine._build_strategy_ctx(
        item,
        opt_data,
        frame,
        item_time := __import__("datetime").datetime(2026, 4, 25, 10, 0, 0),
        1_777_100_000.0,
        0.002,
        0.001,
    )

    assert ctx["position"] == 0
    assert ctx["holding"] is None
    assert ctx_bid == 2.00 and ctx_ask == 2.20, "空仓时应按 alpha>0 选择 call 盘口"
    assert math.isclose(market_opt_price, 2.10, rel_tol=0, abs_tol=1e-9)
    assert math.isclose(ctx_curr_price, 2.10, rel_tol=0, abs_tol=1e-9), "有完整盘口时应取 fair price(mid)"
    assert math.isclose(ctx["stock_iv"], 0.45, rel_tol=0, abs_tol=1e-9), "空仓时应取 call/put IV 的均值"
    assert ctx["time"] == item_time
    assert ctx["held_mins"] == 0.0
    assert ctx["spread_divergence"] == 0.0, "空仓路径当前不会计算 spread_divergence"
    assert st.warmup_complete is True
    assert math.isclose(st.last_alpha_z, 1.20, rel_tol=0, abs_tol=1e-9)


def test_v0_ctx_for_open_position_prefers_fresh_execution_quote_and_updates_roi() -> None:
    _bootstrap_imports()
    ee = _load_execution_engine_module()
    st = _state(
        position=1,
        last_valid_iv=0.35,
        last_opt_price=2.10,
        entry_price=2.00,
        entry_stock=100.0,
        entry_ts=1_777_100_000.0,
        max_roi=0.10,
        entry_spy_roc=0.01,
        entry_index_trend=1,
        last_spread_pct=0.02,
    )
    latest_quote = {
        "ts": 1_777_100_600.0,
        "wall_ts": 10_000.0,
        "call_price": 2.50,
        "call_bid": 2.45,
        "call_ask": 2.55,
    }
    engine = _build_engine(ee, st, mode="realtime", latest_quote=latest_quote)

    item = {
        "symbol": "NVDA",
        "stock_price": 110.0,
        "alpha": 0.80,
        "cs_alpha_z": 0.75,
        "vol_z": 0.50,
        "roc_5m": 0.004,
        "event_prob": 0.02,
        "macd": 0.03,
        "macd_slope": 0.02,
        "snap_roc": 0.002,
        "is_ready": True,
        "correction_mode": "NORMAL",
    }
    opt_data = {
        "has_feed": False,
        "call_bid": 0.0,
        "call_ask": 0.0,
        "call_price": 0.0,
        "call_iv": 0.0,
        "put_bid": 0.0,
        "put_ask": 0.0,
        "put_price": 0.0,
        "put_iv": 0.0,
    }
    frame = {
        "index_trend": 1,
        "global_regime_reversal_cnt": 2,
        "global_is_volatile_regime": False,
        "global_regime_band": "calm",
        "global_regime_score": 0.10,
    }

    with patch("time.time", return_value=10_001.0):
        ctx, market_opt_price, ctx_curr_price, ctx_bid, ctx_ask = engine._build_strategy_ctx(
            item,
            opt_data,
            frame,
            __import__("datetime").datetime(2026, 4, 25, 10, 10, 0),
            1_777_100_600.0,
            0.003,
            0.002,
        )

    assert math.isclose(market_opt_price, 2.50, rel_tol=0, abs_tol=1e-9)
    assert math.isclose(ctx_curr_price, 2.50, rel_tol=0, abs_tol=1e-9), "持仓 realtime 应优先用新鲜 1s execution quote"
    assert math.isclose(ctx_bid, 2.45, rel_tol=0, abs_tol=1e-9)
    assert math.isclose(ctx_ask, 2.55, rel_tol=0, abs_tol=1e-9)
    assert math.isclose(ctx["held_mins"], 10.0, rel_tol=0, abs_tol=1e-9)
    assert math.isclose(ctx["holding"]["max_roi"], 0.25, rel_tol=0, abs_tol=1e-9), "应把 max_roi 更新到当前更高收益"
    assert math.isclose(ctx["spread_divergence"], 0.02, rel_tol=0, abs_tol=1e-9)
    assert math.isclose(st.max_roi, 0.25, rel_tol=0, abs_tol=1e-9)
    assert math.isclose(st.last_spread_pct, 0.04, rel_tol=0, abs_tol=1e-9)
    assert math.isclose(st.last_opt_price, 2.50, rel_tol=0, abs_tol=1e-9)


def main() -> None:
    test_v0_ctx_for_flat_position_uses_alpha_direction_and_feed_mid()
    test_v0_ctx_for_open_position_prefers_fresh_execution_quote_and_updates_roi()
    print("[OK] v0 ctx contract tests passed")


if __name__ == "__main__":
    main()

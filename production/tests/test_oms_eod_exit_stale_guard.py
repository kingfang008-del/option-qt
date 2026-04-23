#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import logging
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
    ibkr_stub = types.ModuleType("ibkr_connector_v8")
    ibkr_stub.IBKRConnectorFinal = object
    sys.modules["ibkr_connector_v8"] = ibkr_stub
    with patch.object(logging, "FileHandler", lambda *_args, **_kwargs: logging.NullHandler()):
        import execution_engine_v8 as ee  # noqa: E402
    return ee


def _build_guard_harness(ee):
    engine = SimpleNamespace(
        mode="realtime",
        latest_execution_quote_by_symbol={
            "NVDA": {
                "ts": 1000.0,
                "wall_ts": 1.0,
            }
        },
        _last_stale_quote_block_warn_ts={},
    )
    engine._execution_quote_freshness = MethodType(ee.ExecutionEngineV8._execution_quote_freshness, engine)
    engine._should_block_strategy_on_stale_quote = MethodType(
        ee.ExecutionEngineV8._should_block_strategy_on_stale_quote,
        engine,
    )
    return engine


def test_eod_exit_not_blocked_when_alpha_frame_has_valid_quote() -> None:
    _bootstrap_imports()
    ee = _load_execution_engine_module()
    engine = _build_guard_harness(ee)

    with patch("time.time", return_value=10_000.0):
        blocked = engine._should_block_strategy_on_stale_quote(
            "NVDA",
            1_800.0,
            "SELL",
            "EOD_CLEAR",
            frame_has_quote=True,
        )

    assert blocked is False


def test_sell_exit_still_blocked_when_no_frame_quote_and_execution_quote_stale() -> None:
    _bootstrap_imports()
    ee = _load_execution_engine_module()
    engine = _build_guard_harness(ee)

    with patch("time.time", return_value=10_000.0):
        blocked = engine._should_block_strategy_on_stale_quote(
            "NVDA",
            1_800.0,
            "SELL",
            "EOD_CLEAR",
            frame_has_quote=False,
        )

    assert blocked is True

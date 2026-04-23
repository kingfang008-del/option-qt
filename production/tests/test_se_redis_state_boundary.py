#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Regression tests for the SE/OMS Redis state boundary.

Redis may carry ALPHA_FRAME/SYNC events and dashboard projections, but it must
not be used by SignalEngine as a trading-state source.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[2]
BASELINE_DIR = REPO_ROOT / "production" / "baseline"
sys.path.insert(0, str(BASELINE_DIR))


def _class_methods(path: Path, class_name: str) -> set[str]:
    src = path.read_text(encoding="utf-8")
    mod = ast.parse(src, filename=str(path))
    for node in mod.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return {item.name for item in node.body if isinstance(item, ast.FunctionDef)}
    raise AssertionError(f"class {class_name} not found in {path}")


def test_signal_engine_has_no_redis_oms_state_sync_method():
    methods = _class_methods(BASELINE_DIR / "signal_engine_v8.py", "SignalEngineV8")

    assert "_sync_state_from_oms" not in methods
    assert "_sync_circuit_breaker_from_redis" not in methods
    assert "_sync_symbol_cooldowns_from_redis" not in methods


def test_se_startup_cleanup_never_touches_redis_or_pg_state():
    import startup_state_hygiene

    def _boom():
        raise AssertionError("SE cleanup must not initialize Redis")

    with patch.dict("os.environ", {"RUN_MODE": "REALTIME_DRY"}, clear=False):
        with patch.object(startup_state_hygiene, "_make_redis_client", side_effect=_boom):
            result = startup_state_hygiene.run_startup_cleanup(role="se", dry_run=False)

    assert result["skipped"] is True
    assert result["reason"] == "se_no_trading_state_cleanup"
    assert result["redis_cleared"] == 0
    assert result["pg_cleared"] == 0

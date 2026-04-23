#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from pathlib import Path


BASELINE_DIR = Path(__file__).resolve().parents[1] / "baseline"
sys.path.insert(0, str(BASELINE_DIR))

from orchestrator_execution import OrchestratorExecution  # noqa: E402


def test_non_urgent_profit_exit_downgrades_configured_mkt_to_lmt():
    assert OrchestratorExecution._resolve_exit_order_type("MKT", "LADDER_TIGHT:roi=18%") == "LMT"
    assert OrchestratorExecution._resolve_exit_order_type("MKT", "TAKE_PROFIT") == "LMT"


def test_time_and_spread_stops_are_not_market_order_urgent():
    assert not OrchestratorExecution._is_urgent_exit_reason("TIME_STOP:31m")
    assert not OrchestratorExecution._is_urgent_exit_reason("SPREAD_STOP:22%")
    assert OrchestratorExecution._resolve_exit_order_type("MKT", "SPREAD_STOP:22%") == "LMT"


def test_hard_risk_exits_can_use_market_when_configured():
    assert OrchestratorExecution._resolve_exit_order_type("MKT", "HARD_STOP:-16%") == "MKT"
    assert OrchestratorExecution._resolve_exit_order_type("MKT", "STOCK_STOP_VOLATILE:-0.4%") == "MKT"
    assert OrchestratorExecution._resolve_exit_order_type("MKT", "FLIP") == "MKT"
    assert OrchestratorExecution._resolve_exit_order_type("MKT", "EOD_CLEAR") == "MKT"


def test_lmt_config_remains_lmt_even_for_urgent_exits():
    assert OrchestratorExecution._resolve_exit_order_type("LMT", "HARD_STOP:-16%") == "LMT"
    assert OrchestratorExecution._resolve_exit_order_type("LMT", "FORCE_CLOSE") == "LMT"


def test_force_flag_is_urgent_when_mkt_configured():
    assert OrchestratorExecution._resolve_exit_order_type("MKT", "manual_close", is_force=True) == "MKT"

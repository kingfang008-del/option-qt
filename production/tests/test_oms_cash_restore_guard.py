#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from pathlib import Path


BASELINE_DIR = Path(__file__).resolve().parents[1] / "baseline"
sys.path.insert(0, str(BASELINE_DIR))

from orchestrator_state_manager import (  # noqa: E402
    safe_state_partition_name,
    sanitize_restored_mock_cash,
    tag_state_snapshot_rows,
)


def test_realtime_dry_rejects_implausible_legacy_cash_restore():
    cash, reason = sanitize_restored_mock_cash(
        500000.0,
        initial_cash=50000.0,
        is_realtime_dry=True,
        max_multiplier=3.0,
    )

    assert cash == 50000.0
    assert reason == "above_dry_restore_cap:500000.00>150000.00"


def test_realtime_dry_accepts_plausible_cash_restore():
    cash, reason = sanitize_restored_mock_cash(
        62000.0,
        initial_cash=50000.0,
        is_realtime_dry=True,
        max_multiplier=3.0,
    )

    assert cash == 62000.0
    assert reason is None


def test_non_dry_does_not_cap_restored_cash():
    cash, reason = sanitize_restored_mock_cash(
        500000.0,
        initial_cash=50000.0,
        is_realtime_dry=False,
        max_multiplier=3.0,
    )

    assert cash == 500000.0
    assert reason is None


def test_state_partition_name_is_safe_and_stable():
    name1 = safe_state_partition_name("oms_REALTIME_DRY/live db0")
    name2 = safe_state_partition_name("oms_REALTIME_DRY/live db0")

    assert name1 == name2
    assert name1.startswith("symbol_state_")
    assert "/" not in name1
    assert " " not in name1
    assert len(name1) <= 63


def test_state_snapshot_rows_are_tagged_with_namespace_and_mode():
    rows = {
        "AAPL": {"position": 1},
        "_GLOBAL_STATE_": {"mock_cash": 50000.0},
    }

    tagged = tag_state_snapshot_rows(
        rows,
        namespace="oms_realtime_dry_live_db0",
        run_mode="REALTIME_DRY",
    )

    assert tagged["AAPL"]["state_namespace"] == "oms_realtime_dry_live_db0"
    assert tagged["AAPL"]["mode"] == "REALTIME_DRY"
    assert tagged["_GLOBAL_STATE_"]["state_namespace"] == "oms_realtime_dry_live_db0"
    assert tagged["_GLOBAL_STATE_"]["mode"] == "REALTIME_DRY"

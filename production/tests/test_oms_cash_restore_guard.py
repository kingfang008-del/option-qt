#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from pathlib import Path


BASELINE_DIR = Path(__file__).resolve().parents[1] / "baseline"
sys.path.insert(0, str(BASELINE_DIR))

from orchestrator_state_manager import sanitize_restored_mock_cash  # noqa: E402


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

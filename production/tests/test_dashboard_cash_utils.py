#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import sys
from pathlib import Path


BASELINE_DIR = Path(__file__).resolve().parents[1] / "baseline"
sys.path.insert(0, str(BASELINE_DIR))

from dashboard_cash_utils import (  # noqa: E402
    parse_oms_ledger_hash,
    parse_oms_live_cash_payload,
    select_remaining_cash,
)


def test_realtime_dry_rejects_ledger_from_other_mode():
    cash, source = parse_oms_ledger_hash(
        {
            b"cash": b"500000.00",
            b"updated_at": b"1000.0",
            b"mode": b"BACKTEST",
        },
        now_ts=1001.0,
        expected_modes=["REALTIME_DRY"],
    )

    assert cash is None
    assert source is None


def test_realtime_dry_rejects_legacy_live_cash_without_mode():
    raw = json.dumps({"cash": 500000.0, "ts": 1000.0})

    cash, source = parse_oms_live_cash_payload(
        raw,
        now_ts=1001.0,
        expected_modes=["REALTIME_DRY"],
    )

    assert cash is None
    assert source is None


def test_realtime_dry_uses_fresh_matching_oms_ledger():
    cash, source = parse_oms_ledger_hash(
        {
            "cash": "49213.25",
            "updated_at": "1000.0",
            "mode": "REALTIME_DRY",
        },
        now_ts=1001.0,
        expected_modes=["REALTIME_DRY"],
    )

    assert cash == 49213.25
    assert source == "OMS ledger"


def test_realtime_dry_never_uses_trade_log_cash_fallback():
    cash, source = select_remaining_cash(
        live_cash=None,
        live_cash_source=None,
        log_cash=500000.0,
        latest_cash=None,
        run_mode="REALTIME_DRY",
    )

    assert cash == 0.0
    assert source == "No fresh OMS cash"


def test_non_realtime_can_use_trade_log_cash_fallback():
    cash, source = select_remaining_cash(
        live_cash=None,
        live_cash_source=None,
        log_cash=123456.0,
        latest_cash=None,
        run_mode="LIVEREPLAY",
    )

    assert cash == 123456.0
    assert source == "Trade Log"

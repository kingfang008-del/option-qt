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


def test_alpha_frame_adapter_maps_call_and_put_decision_quotes() -> None:
    _bootstrap_imports()
    from Domain import alpha_frame_from_legacy  # noqa: E402

    payload = {
        "source": "alpha_engine_v8",
        "action": "ALPHA_FRAME",
        "ts": 1_710_000_060,
        "frame_id": "frame-42",
        "symbols": ["NVDA", "TSLA"],
        "items": [
            {
                "symbol": "NVDA",
                "batch_idx": 0,
                "stock_price": 910.0,
                "alpha": 1.4,
                "cs_alpha_z": 1.3,
                "vol_z": 0.7,
                "roc_5m": 0.02,
                "macd": 0.3,
                "macd_slope": 0.01,
                "snap_roc": 0.004,
                "event_prob": 0.25,
                "is_ready": True,
                "last_valid_iv": 0.42,
                "correction_mode": "NORMAL",
                "alpha_label_ts": 1_710_000_000,
                "alpha_available_ts": 1_710_000_060,
                "opt_data": {
                    "has_feed": True,
                    "call_price": 12.5,
                    "call_bid": 12.4,
                    "call_ask": 12.6,
                    "call_bid_size": 15,
                    "call_ask_size": 18,
                    "call_id": "NVDA_CALL",
                    "call_k": 900,
                    "call_iv": 0.41,
                    "call_vol": 1200,
                    "put_price": 10.0,
                    "put_bid": 9.9,
                    "put_ask": 10.1,
                    "put_id": "NVDA_PUT",
                },
            },
            {
                "symbol": "TSLA",
                "batch_idx": 1,
                "stock_price": 170.0,
                "alpha": -0.9,
                "cs_alpha_z": -0.8,
                "vol_z": 0.4,
                "roc_5m": -0.01,
                "macd": -0.2,
                "macd_slope": -0.02,
                "snap_roc": -0.003,
                "event_prob": 0.10,
                "is_ready": True,
                "last_valid_iv": 0.38,
                "correction_mode": "NORMAL",
                "alpha_label_ts": 1_710_000_000,
                "alpha_available_ts": 1_710_000_060,
                "opt_data": {
                    "has_feed": True,
                    "call_price": 4.0,
                    "call_bid": 3.9,
                    "call_ask": 4.1,
                    "call_id": "TSLA_CALL",
                    "put_price": 6.5,
                    "put_bid": 6.4,
                    "put_ask": 6.6,
                    "put_bid_size": 20,
                    "put_ask_size": 25,
                    "put_id": "TSLA_PUT",
                    "put_k": 165,
                    "put_iv": 0.37,
                    "put_vol": 900,
                },
            },
        ],
        "index_trend": 1,
        "global_regime_band": "calm",
        "is_zombie_market": False,
    }

    frame = alpha_frame_from_legacy(payload)

    assert frame.validate() == []
    assert frame.frame_id == "frame-42"
    assert frame.market_regime == "calm"
    assert frame.items[0].decision_quote is not None
    assert frame.items[0].decision_quote.contract_id == "NVDA_CALL"
    assert frame.items[0].decision_quote.metadata["option_right"] == "call"
    assert frame.items[1].decision_quote.contract_id == "TSLA_PUT"
    assert frame.items[1].decision_quote.metadata["option_right"] == "put"


def test_execution_quote_adapter_uses_cached_option_side() -> None:
    _bootstrap_imports()
    from Domain import execution_quote_from_legacy_payload  # noqa: E402

    payload = {
        "ts": 1_710_000_070.0,
        "call_price": 12.5,
        "call_bid": 12.4,
        "call_ask": 12.6,
        "put_price": 8.1,
        "put_bid": 8.0,
        "put_ask": 8.2,
    }

    quote = execution_quote_from_legacy_payload(
        "NVDA",
        payload,
        legacy_position=-1,
    )

    assert quote.validate() == []
    assert quote.last_price == 8.1
    assert quote.best_bid == 8.0
    assert quote.best_ask == 8.2
    assert quote.metadata["option_right"] == "put"


def test_position_adapter_keeps_put_as_long_option_position() -> None:
    _bootstrap_imports()
    from Domain import PositionSide, position_snapshot_from_legacy_state  # noqa: E402

    payload = {
        "symbol": "MSTR",
        "position": -1,
        "qty": 3,
        "entry_price": 5.5,
        "entry_ts": 1_710_000_120.0,
        "contract_id": "MSTR_PUT",
        "strike_price": 1500,
        "expiry_date": "2026-06-19T00:00:00",
        "opt_type": "put",
        "last_valid_iv": 0.55,
        "open_fill_confirmed": True,
    }

    position = position_snapshot_from_legacy_state(payload, explicit_kind=None)

    assert position.validate() == []
    assert position.side == PositionSide.LONG
    assert position.quantity == 3
    assert position.metadata["option_right"] == "put"
    assert position.metadata["strike_price"] == 1500


def test_execution_window_adapter_accepts_quote_mapping() -> None:
    _bootstrap_imports()
    from Domain import execution_window_from_legacy  # noqa: E402

    frame_payload = {
        "source": "alpha_engine_v8",
        "action": "ALPHA_FRAME",
        "ts": 1_710_000_060,
        "frame_id": "frame-77",
        "symbols": ["AAPL"],
        "items": [
            {
                "symbol": "AAPL",
                "batch_idx": 0,
                "stock_price": 200.0,
                "alpha": 0.4,
                "alpha_label_ts": 1_710_000_000,
                "alpha_available_ts": 1_710_000_060,
                "opt_data": {
                    "best_bid": 199.9,
                    "best_ask": 200.1,
                    "last_price": 200.0,
                    "ts": 1_710_000_060.0,
                },
            }
        ],
        "global_regime_band": "trend",
    }
    quotes_payload = {
        "AAPL": {
            "ts": 1_710_000_061.0,
            "best_bid": 199.95,
            "best_ask": 200.05,
            "last_price": 200.0,
            "source": "replay_feed",
        }
    }

    window = execution_window_from_legacy(frame_payload, quotes_payload=quotes_payload)

    assert window.validate() == []
    assert window.alpha_frame.frame_id == "frame-77"
    assert len(window.quotes_1s) == 1
    assert window.quotes_1s[0].symbol == "AAPL"

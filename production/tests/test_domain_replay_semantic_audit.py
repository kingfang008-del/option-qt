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


class _State:
    def __init__(self, symbol: str, position: int = 1) -> None:
        self.symbol = symbol
        self.position = position
        self.qty = 2
        self.entry_price = 9.5
        self.entry_stock = 900.0
        self.entry_ts = 1_710_000_065.0
        self.entry_spy_roc = 0.0
        self.entry_index_trend = 0
        self.entry_alpha_z = 1.0
        self.entry_iv = 0.4
        self.max_roi = 0.0
        self.cooldown_until = 0.0
        self.contract_id = "NVDA_CALL"
        self.strike_price = 900.0
        self.expiry_date = None
        self.last_valid_iv = 0.4
        self.opt_type = "call" if position >= 0 else "put"
        self.warmup_complete = True
        self.correction_mode = "NORMAL"
        self.prev_macd_hist = 0.0
        self.last_spread_pct = 0.0
        self.entry_slot_reserved = True
        self.open_fill_confirmed = True

    def to_dict(self):
        return {
            "symbol": self.symbol,
            "position": self.position,
            "qty": self.qty,
            "entry_price": self.entry_price,
            "entry_stock": self.entry_stock,
            "entry_ts": self.entry_ts,
            "entry_spy_roc": self.entry_spy_roc,
            "entry_index_trend": self.entry_index_trend,
            "entry_alpha_z": self.entry_alpha_z,
            "entry_iv": self.entry_iv,
            "max_roi": self.max_roi,
            "cooldown_until": self.cooldown_until,
            "contract_id": self.contract_id,
            "strike_price": self.strike_price,
            "expiry_date": self.expiry_date,
            "last_valid_iv": self.last_valid_iv,
            "opt_type": self.opt_type,
            "warmup_complete": self.warmup_complete,
            "correction_mode": self.correction_mode,
            "prev_macd_hist": self.prev_macd_hist,
            "last_spread_pct": self.last_spread_pct,
            "entry_slot_reserved": self.entry_slot_reserved,
            "open_fill_confirmed": self.open_fill_confirmed,
        }


class _ExecEngine:
    def __init__(self) -> None:
        self.latest_execution_quote_by_symbol = {
            "NVDA": {
                "ts": 1_710_000_061.0,
                "call_price": 10.1,
                "call_bid": 10.0,
                "call_ask": 10.2,
            }
        }
        self.states = {"NVDA": _State("NVDA", position=1)}


def test_replay_semantic_auditor_accepts_valid_window() -> None:
    _bootstrap_imports()
    from Domain import ReplaySemanticAuditor  # noqa: E402

    auditor = ReplaySemanticAuditor(enabled=True, strict=True, log_every=1)
    signal_packet = {
        "frame_id": "1710000060",
        "symbols": ["NVDA"],
        "stock_price": [900.0],
        "precalc_alpha": [1.0],
        "fast_vol": [0.2],
        "alpha_label_ts": [1_710_000_000.0],
        "alpha_available_ts": [1_710_000_060.0],
        "feed_call_price": [10.0],
        "feed_put_price": [8.0],
        "feed_call_bid": [9.9],
        "feed_call_ask": [10.1],
        "feed_put_bid": [7.9],
        "feed_put_ask": [8.1],
        "feed_call_bid_size": [100.0],
        "feed_call_ask_size": [100.0],
        "feed_put_bid_size": [100.0],
        "feed_put_ask_size": [100.0],
        "feed_call_k": [900.0],
        "feed_put_k": [890.0],
        "feed_call_iv": [0.4],
        "feed_put_iv": [0.35],
        "feed_call_vol": [1.0],
        "feed_put_vol": [1.0],
        "feed_call_id": ["NVDA_CALL"],
        "feed_put_id": ["NVDA_PUT"],
        "ts": 1_710_000_060.0,
    }
    quote_packet = {
        "symbols": ["NVDA"],
        "ts": 1_710_000_061.0,
        "precalc_alpha": [1.0],
        "feed_call_price": [10.1],
        "feed_put_price": [8.1],
        "feed_call_bid": [10.0],
        "feed_call_ask": [10.2],
        "feed_put_bid": [8.0],
        "feed_put_ask": [8.2],
    }

    auditor.audit_pre_window(1_710_000_060, signal_packet)
    auditor.audit_quote_packet(quote_packet)
    auditor.audit_post_window(1_710_000_060, _ExecEngine(), quote_packet)

    stats = auditor.stats()
    assert stats["pre_window_ok"] == 1
    assert stats["quote_ok"] == 1
    assert stats["post_window_ok"] == 1


def test_replay_semantic_auditor_strict_raises_on_crossed_book() -> None:
    _bootstrap_imports()
    from Domain import ReplaySemanticAuditor  # noqa: E402

    auditor = ReplaySemanticAuditor(enabled=True, strict=True, log_every=1)
    bad_quote_packet = {
        "symbols": ["NVDA"],
        "ts": 1_710_000_061.0,
        "precalc_alpha": [1.0],
        "feed_call_price": [10.1],
        "feed_put_price": [8.1],
        "feed_call_bid": [10.3],
        "feed_call_ask": [10.2],
        "feed_put_bid": [8.0],
        "feed_put_ask": [8.2],
    }

    try:
        auditor.audit_quote_packet(bad_quote_packet)
    except AssertionError as exc:
        assert "crossed book" not in str(exc) or "best_ask must be >= best_bid" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("strict mode should raise on invalid quote packet")

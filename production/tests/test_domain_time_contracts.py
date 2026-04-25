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


def test_execution_window_roundtrip_and_validation() -> None:
    _bootstrap_imports()
    from Domain import (  # noqa: E402
        AlphaFrame,
        AlphaFrameItem,
        ExecutionQuote1s,
        ExecutionWindow,
        InstrumentKind,
        InstrumentTraits,
        QuoteSourceKind,
    )

    traits = InstrumentTraits.option("NVDA")
    item = AlphaFrameItem(
        symbol="NVDA",
        instrument_traits=traits,
        alpha=1.25,
        alpha_label_ts=1_710_000_000,
        alpha_available_ts=1_710_000_060,
        batch_idx=0,
        frame_id="frame-1",
        reference_price=905.0,
        event_prob=0.35,
    )
    frame = AlphaFrame.from_items(
        minute_ts=1_710_000_060,
        items=[item],
        frame_id="frame-1",
        index_trend=1,
        market_regime="calm",
    )
    quotes = [
        ExecutionQuote1s(
            symbol="NVDA",
            instrument_kind=InstrumentKind.OPTION,
            ts=1_710_000_060.0,
            last_price=12.5,
            best_bid=12.4,
            best_ask=12.6,
            bid_size=10,
            ask_size=12,
            source_kind=QuoteSourceKind.REPLAY_FEED,
        ),
        ExecutionQuote1s(
            symbol="NVDA",
            instrument_kind=InstrumentKind.OPTION,
            ts=1_710_000_061.0,
            last_price=12.55,
            best_bid=12.45,
            best_ask=12.65,
            bid_size=8,
            ask_size=10,
            source_kind=QuoteSourceKind.REPLAY_FEED,
        ),
    ]
    window = ExecutionWindow.from_frame(
        minute_ts=1_710_000_060,
        alpha_frame=frame,
        quotes_1s=quotes,
    )

    assert traits.validate() == []
    assert frame.validate() == []
    assert window.validate() == []
    assert "quotes=2" in window.summary()

    restored = ExecutionWindow.from_dict(window.to_dict())
    assert restored.validate() == []
    assert restored.alpha_frame.frame_id == "frame-1"
    assert restored.quotes_1s[0].best_bid == 12.4


def test_alpha_frame_rejects_duplicate_symbols() -> None:
    _bootstrap_imports()
    from Domain import AlphaFrame, AlphaFrameItem, InstrumentTraits  # noqa: E402

    traits = InstrumentTraits.stock("AAPL")
    item_a = AlphaFrameItem(
        symbol="AAPL",
        instrument_traits=traits,
        alpha=0.5,
        alpha_label_ts=100,
        alpha_available_ts=160,
    )
    item_b = AlphaFrameItem(
        symbol="AAPL",
        instrument_traits=traits,
        alpha=0.8,
        alpha_label_ts=100,
        alpha_available_ts=160,
    )
    frame = AlphaFrame.from_items(160, [item_a, item_b], frame_id="dup")
    errors = frame.validate()

    assert any("duplicated" in msg for msg in errors)


def test_decision_quote_rejects_crossed_book() -> None:
    _bootstrap_imports()
    from Domain import DecisionQuoteSnapshot, InstrumentKind  # noqa: E402

    quote = DecisionQuoteSnapshot(
        symbol="BTC-USD-SWAP",
        instrument_kind=InstrumentKind.PERPETUAL,
        quote_ts=1_800_000_000.0,
        last_price=62000.0,
        best_bid=62010.0,
        best_ask=62005.0,
    )
    errors = quote.validate()

    assert any("best_ask must be >= best_bid" in msg for msg in errors)


def test_perpetual_position_snapshot_supports_short() -> None:
    _bootstrap_imports()
    from Domain import InstrumentTraits, PositionSide, PositionSnapshot  # noqa: E402

    traits = InstrumentTraits.perpetual("BTC-USD-SWAP", venue="binance")
    position = PositionSnapshot(
        symbol="BTC-USD-SWAP",
        instrument_traits=traits,
        side=PositionSide.SHORT,
        quantity=0.25,
        avg_entry_price=62000.0,
        entry_ts=1_800_000_005.0,
        entry_frame_id="btc-frame-1",
    )

    assert traits.validate() == []
    assert traits.supports_short is True
    assert position.validate() == []
    assert position.signed_quantity == -0.25

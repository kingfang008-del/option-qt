from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional

from .contracts import (
    AlphaFrame,
    AlphaFrameItem,
    DecisionQuoteSnapshot,
    ExecutionQuote1s,
    ExecutionWindow,
    InstrumentKind,
    InstrumentTraits,
    PositionSide,
    PositionSnapshot,
    QuoteSourceKind,
)


def _coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _coerce_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return default


def _coerce_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    text = str(value)
    return text if text else default


def _normalize_option_right(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in ("call", "c", "1", "long_call"):
        return "call"
    if text in ("put", "p", "-1", "long_put"):
        return "put"
    return None


def _infer_instrument_kind(
    payload: Optional[Mapping[str, Any]] = None,
    explicit_kind: Optional[InstrumentKind] = None,
) -> InstrumentKind:
    if explicit_kind is not None:
        return explicit_kind
    if not payload:
        return InstrumentKind.UNKNOWN
    if any(key in payload for key in ("call_price", "put_price", "call_bid", "put_bid", "call_id", "put_id")):
        return InstrumentKind.OPTION
    if any(key in payload for key in ("strike_price", "expiry_date", "opt_type", "entry_iv", "last_valid_iv")):
        return InstrumentKind.OPTION
    if any(key in payload for key in ("mark_price", "index_price", "funding_rate")):
        return InstrumentKind.PERPETUAL
    if any(key in payload for key in ("best_bid", "best_ask", "last_price")):
        return InstrumentKind.STOCK
    return InstrumentKind.UNKNOWN


def _infer_quote_source_kind(
    payload: Optional[Mapping[str, Any]] = None,
    explicit_kind: Optional[QuoteSourceKind] = None,
) -> QuoteSourceKind:
    if explicit_kind is not None:
        return explicit_kind
    if not payload:
        return QuoteSourceKind.UNKNOWN
    source_kind = payload.get("source_kind")
    if source_kind:
        try:
            return QuoteSourceKind(source_kind)
        except Exception:
            pass
    source = str(payload.get("source", "") or "").lower()
    if "replay" in source:
        return QuoteSourceKind.REPLAY_FEED
    if "live" in source:
        return QuoteSourceKind.LIVE_FEED
    return QuoteSourceKind.UNKNOWN


def _infer_quote_ts(payload: Optional[Mapping[str, Any]], fallback_ts: float = 0.0) -> float:
    if not payload:
        return float(fallback_ts or 0.0)
    for key in ("quote_ts", "ts", "event_ts", "wall_ts"):
        if key in payload:
            value = _coerce_float(payload.get(key), 0.0)
            if value > 0.0:
                return value
    return float(fallback_ts or 0.0)


def _infer_option_right(
    payload: Optional[Mapping[str, Any]] = None,
    *,
    explicit_right: Optional[str] = None,
    legacy_position: Optional[int] = None,
    alpha: Optional[float] = None,
) -> Optional[str]:
    if explicit_right:
        return _normalize_option_right(explicit_right)
    if payload:
        for key in ("option_right", "opt_type", "right"):
            right = _normalize_option_right(payload.get(key))
            if right:
                return right
    if legacy_position == 1:
        return "call"
    if legacy_position == -1:
        return "put"
    if alpha is not None:
        if float(alpha) > 0.0:
            return "call"
        if float(alpha) < 0.0:
            return "put"
    if payload:
        if _coerce_float(payload.get("call_price"), 0.0) > 0.0 or _coerce_str(payload.get("call_id")):
            return "call"
        if _coerce_float(payload.get("put_price"), 0.0) > 0.0 or _coerce_str(payload.get("put_id")):
            return "put"
    return None


def _extract_side_payload(
    payload: Mapping[str, Any],
    option_right: Optional[str],
) -> Dict[str, Any]:
    if option_right == "call":
        return {
            "last_price": _coerce_float(payload.get("call_price"), 0.0),
            "best_bid": _coerce_float(payload.get("call_bid"), 0.0),
            "best_ask": _coerce_float(payload.get("call_ask"), 0.0),
            "bid_size": _coerce_float(payload.get("call_bid_size"), 0.0),
            "ask_size": _coerce_float(payload.get("call_ask_size"), 0.0),
            "contract_id": _coerce_str(payload.get("call_id")),
            "strike": _coerce_float(payload.get("call_k"), 0.0),
            "iv": _coerce_float(payload.get("call_iv"), 0.0),
            "volume": _coerce_float(payload.get("call_vol"), 0.0),
        }
    if option_right == "put":
        return {
            "last_price": _coerce_float(payload.get("put_price"), 0.0),
            "best_bid": _coerce_float(payload.get("put_bid"), 0.0),
            "best_ask": _coerce_float(payload.get("put_ask"), 0.0),
            "bid_size": _coerce_float(payload.get("put_bid_size"), 0.0),
            "ask_size": _coerce_float(payload.get("put_ask_size"), 0.0),
            "contract_id": _coerce_str(payload.get("put_id")),
            "strike": _coerce_float(payload.get("put_k"), 0.0),
            "iv": _coerce_float(payload.get("put_iv"), 0.0),
            "volume": _coerce_float(payload.get("put_vol"), 0.0),
        }
    return {
        "last_price": _coerce_float(payload.get("last_price"), 0.0),
        "best_bid": _coerce_float(payload.get("best_bid"), 0.0),
        "best_ask": _coerce_float(payload.get("best_ask"), 0.0),
        "bid_size": _coerce_float(payload.get("bid_size"), 0.0),
        "ask_size": _coerce_float(payload.get("ask_size"), 0.0),
        "contract_id": _coerce_str(payload.get("contract_id")),
        "strike": _coerce_float(payload.get("strike"), 0.0),
        "iv": _coerce_float(payload.get("iv"), 0.0),
        "volume": _coerce_float(payload.get("volume"), 0.0),
    }


def _collect_extra(payload: Mapping[str, Any], known_keys: Iterable[str]) -> Dict[str, Any]:
    known = set(known_keys)
    return {
        str(key): value
        for key, value in payload.items()
        if key not in known and value not in (None, "")
    }


def instrument_traits_from_legacy(
    symbol: str,
    *,
    payload: Optional[Mapping[str, Any]] = None,
    explicit_kind: Optional[InstrumentKind] = None,
    overrides: Optional[Mapping[str, Any]] = None,
) -> InstrumentTraits:
    kind = _infer_instrument_kind(payload, explicit_kind)
    changes = dict(overrides or {})
    if kind == InstrumentKind.OPTION:
        return InstrumentTraits.option(symbol, **changes)
    if kind == InstrumentKind.PERPETUAL:
        return InstrumentTraits.perpetual(symbol, **changes)
    if kind == InstrumentKind.STOCK:
        return InstrumentTraits.stock(symbol, **changes)
    base = {
        "symbol": symbol,
        "instrument_kind": kind.value,
        **changes,
    }
    return InstrumentTraits.from_dict(base)


def decision_quote_from_legacy_payload(
    symbol: str,
    payload: Mapping[str, Any],
    *,
    explicit_kind: Optional[InstrumentKind] = None,
    explicit_source_kind: Optional[QuoteSourceKind] = None,
    explicit_option_right: Optional[str] = None,
    legacy_position: Optional[int] = None,
    alpha: Optional[float] = None,
    fallback_ts: float = 0.0,
) -> DecisionQuoteSnapshot:
    instrument_kind = _infer_instrument_kind(payload, explicit_kind)
    source_kind = _infer_quote_source_kind(payload, explicit_source_kind)
    option_right = _infer_option_right(
        payload,
        explicit_right=explicit_option_right,
        legacy_position=legacy_position,
        alpha=alpha,
    )
    side_payload = _extract_side_payload(payload, option_right if instrument_kind == InstrumentKind.OPTION else None)
    metadata = {
        "option_right": option_right,
        "has_feed": bool(payload.get("has_feed", False)),
        "legacy_payload_kind": "decision_quote",
        **_collect_extra(
            payload,
            known_keys={
                "quote_ts",
                "ts",
                "event_ts",
                "wall_ts",
                "source",
                "source_kind",
                "has_feed",
                "call_price",
                "put_price",
                "call_bid",
                "call_ask",
                "put_bid",
                "put_ask",
                "call_bid_size",
                "call_ask_size",
                "put_bid_size",
                "put_ask_size",
                "call_id",
                "put_id",
                "call_k",
                "put_k",
                "call_iv",
                "put_iv",
                "call_vol",
                "put_vol",
                "last_price",
                "best_bid",
                "best_ask",
                "bid_size",
                "ask_size",
                "mark_price",
                "index_price",
                "contract_id",
                "venue",
                "volume",
                "strike",
                "iv",
            },
        ),
    }
    if side_payload["strike"] > 0.0:
        metadata["strike"] = side_payload["strike"]
    if side_payload["iv"] > 0.0:
        metadata["iv"] = side_payload["iv"]
    if side_payload["volume"] > 0.0:
        metadata["volume"] = side_payload["volume"]
    return DecisionQuoteSnapshot(
        symbol=symbol,
        instrument_kind=instrument_kind,
        quote_ts=_infer_quote_ts(payload, fallback_ts),
        last_price=side_payload["last_price"],
        best_bid=side_payload["best_bid"],
        best_ask=side_payload["best_ask"],
        bid_size=side_payload["bid_size"],
        ask_size=side_payload["ask_size"],
        mark_price=(
            None if payload.get("mark_price") in (None, "") else _coerce_float(payload.get("mark_price"), 0.0)
        ),
        index_price=(
            None if payload.get("index_price") in (None, "") else _coerce_float(payload.get("index_price"), 0.0)
        ),
        contract_id=side_payload["contract_id"],
        venue=_coerce_str(payload.get("venue")),
        source_kind=source_kind,
        metadata=metadata,
    )


def execution_quote_from_legacy_payload(
    symbol: str,
    payload: Mapping[str, Any],
    *,
    explicit_kind: Optional[InstrumentKind] = None,
    explicit_source_kind: Optional[QuoteSourceKind] = None,
    explicit_option_right: Optional[str] = None,
    legacy_position: Optional[int] = None,
    alpha: Optional[float] = None,
    fallback_ts: float = 0.0,
) -> ExecutionQuote1s:
    instrument_kind = _infer_instrument_kind(payload, explicit_kind)
    source_kind = _infer_quote_source_kind(payload, explicit_source_kind)
    option_right = _infer_option_right(
        payload,
        explicit_right=explicit_option_right,
        legacy_position=legacy_position,
        alpha=alpha,
    )
    side_payload = _extract_side_payload(payload, option_right if instrument_kind == InstrumentKind.OPTION else None)
    metadata = {
        "option_right": option_right,
        "legacy_payload_kind": "execution_quote",
        **_collect_extra(
            payload,
            known_keys={
                "quote_ts",
                "ts",
                "event_ts",
                "wall_ts",
                "source",
                "source_kind",
                "call_price",
                "put_price",
                "call_bid",
                "call_ask",
                "put_bid",
                "put_ask",
                "call_bid_size",
                "call_ask_size",
                "put_bid_size",
                "put_ask_size",
                "call_id",
                "put_id",
                "last_price",
                "best_bid",
                "best_ask",
                "bid_size",
                "ask_size",
                "mark_price",
                "index_price",
                "contract_id",
                "venue",
                "sequence_no",
                "exchange_latency_ms",
            },
        ),
    }
    return ExecutionQuote1s(
        symbol=symbol,
        instrument_kind=instrument_kind,
        ts=_infer_quote_ts(payload, fallback_ts),
        last_price=side_payload["last_price"],
        best_bid=side_payload["best_bid"],
        best_ask=side_payload["best_ask"],
        bid_size=side_payload["bid_size"],
        ask_size=side_payload["ask_size"],
        mark_price=(
            None if payload.get("mark_price") in (None, "") else _coerce_float(payload.get("mark_price"), 0.0)
        ),
        index_price=(
            None if payload.get("index_price") in (None, "") else _coerce_float(payload.get("index_price"), 0.0)
        ),
        contract_id=side_payload["contract_id"],
        venue=_coerce_str(payload.get("venue")),
        source_kind=source_kind,
        sequence_no=payload.get("sequence_no"),
        exchange_latency_ms=(
            None
            if payload.get("exchange_latency_ms") in (None, "")
            else _coerce_float(payload.get("exchange_latency_ms"), 0.0)
        ),
        metadata=metadata,
    )


def alpha_frame_item_from_legacy(
    payload: Mapping[str, Any],
    *,
    frame_id: str = "",
    explicit_kind: Optional[InstrumentKind] = None,
    explicit_option_right: Optional[str] = None,
) -> AlphaFrameItem:
    symbol = _coerce_str(payload.get("symbol"))
    alpha = _coerce_float(payload.get("alpha"), 0.0)
    opt_data = payload.get("opt_data") or {}
    instrument_traits = instrument_traits_from_legacy(
        symbol,
        payload=opt_data if opt_data else payload,
        explicit_kind=explicit_kind,
    )
    alpha_label_ts = _coerce_int(payload.get("alpha_label_ts"), 0)
    alpha_available_ts = _coerce_int(payload.get("alpha_available_ts"), 0)
    option_right = _infer_option_right(
        opt_data if isinstance(opt_data, Mapping) else None,
        explicit_right=explicit_option_right,
        alpha=alpha,
    )
    decision_quote = None
    if isinstance(opt_data, Mapping) and opt_data:
        decision_quote = decision_quote_from_legacy_payload(
            symbol,
            opt_data,
            explicit_kind=instrument_traits.instrument_kind,
            explicit_option_right=option_right,
            alpha=alpha,
            fallback_ts=float(alpha_available_ts or 0),
        )
    tags = _collect_extra(
        payload,
        known_keys={
            "symbol",
            "batch_idx",
            "stock_price",
            "alpha",
            "cs_alpha_z",
            "vol_z",
            "roc_5m",
            "macd",
            "macd_slope",
            "snap_roc",
            "event_prob",
            "is_ready",
            "last_valid_iv",
            "correction_mode",
            "alpha_label_ts",
            "alpha_available_ts",
            "opt_data",
        },
    )
    if option_right:
        tags["option_right"] = option_right
    last_valid_iv = _coerce_float(payload.get("last_valid_iv"), 0.0)
    if last_valid_iv > 0.0:
        tags["last_valid_iv"] = last_valid_iv
    return AlphaFrameItem(
        symbol=symbol,
        instrument_traits=instrument_traits,
        alpha=alpha,
        alpha_label_ts=alpha_label_ts,
        alpha_available_ts=alpha_available_ts,
        batch_idx=_coerce_int(payload.get("batch_idx"), -1),
        frame_id=_coerce_str(payload.get("frame_id"), frame_id) or frame_id,
        reference_price=_coerce_float(payload.get("stock_price"), 0.0),
        cs_alpha_z=_coerce_float(payload.get("cs_alpha_z"), 0.0),
        vol_z=_coerce_float(payload.get("vol_z"), 0.0),
        roc_5m=_coerce_float(payload.get("roc_5m"), 0.0),
        macd=_coerce_float(payload.get("macd"), 0.0),
        macd_slope=_coerce_float(payload.get("macd_slope"), 0.0),
        snap_roc=_coerce_float(payload.get("snap_roc"), 0.0),
        event_prob=_coerce_float(payload.get("event_prob"), 0.0),
        is_ready=bool(payload.get("is_ready", False)),
        correction_mode=_coerce_str(payload.get("correction_mode"), "NORMAL"),
        decision_quote=decision_quote,
        tags=tags,
    )


def alpha_frame_from_legacy(
    payload: Mapping[str, Any],
    *,
    explicit_kind: Optional[InstrumentKind] = None,
) -> AlphaFrame:
    minute_ts = _coerce_int(payload.get("ts"), 0)
    frame_id = _coerce_str(payload.get("frame_id"))
    items_payload = payload.get("items") or []
    items = [
        alpha_frame_item_from_legacy(item, frame_id=frame_id, explicit_kind=explicit_kind)
        for item in items_payload
        if isinstance(item, Mapping)
    ]
    metadata = {
        "source": _coerce_str(payload.get("source")),
        "symbols": list(payload.get("symbols") or []),
        "spy_roc_5min": list(payload.get("spy_roc_5min") or []),
        "qqq_roc_5min": list(payload.get("qqq_roc_5min") or []),
        "global_regime_reversal_cnt": _coerce_int(payload.get("global_regime_reversal_cnt"), 0),
        "global_is_volatile_regime": bool(payload.get("global_is_volatile_regime", False)),
        "global_regime_score": _coerce_float(payload.get("global_regime_score"), 0.0),
    }
    metadata.update(
        _collect_extra(
            payload,
            known_keys={
                "source",
                "action",
                "ts",
                "frame_id",
                "symbols",
                "items",
                "index_trend",
                "spy_roc_5min",
                "qqq_roc_5min",
                "is_zombie_market",
                "global_regime_reversal_cnt",
                "global_is_volatile_regime",
                "global_regime_band",
                "global_regime_score",
            },
        )
    )
    return AlphaFrame(
        frame_id=frame_id,
        minute_ts=minute_ts,
        alpha_label_ts=minute_ts - 60 if minute_ts > 0 else 0,
        alpha_available_ts=minute_ts,
        items=items,
        index_trend=_coerce_int(payload.get("index_trend"), 0),
        market_regime=_coerce_str(payload.get("global_regime_band"), "unknown"),
        is_zombie_market=bool(payload.get("is_zombie_market", False)),
        metadata=metadata,
    )


def execution_window_from_legacy(
    alpha_frame_payload: Mapping[str, Any],
    *,
    quotes_payload: Optional[Any] = None,
    explicit_kind: Optional[InstrumentKind] = None,
) -> ExecutionWindow:
    frame = alpha_frame_from_legacy(alpha_frame_payload, explicit_kind=explicit_kind)
    quotes: List[ExecutionQuote1s] = []
    if isinstance(quotes_payload, Mapping):
        for symbol, quote_payload in quotes_payload.items():
            if isinstance(quote_payload, Mapping):
                quotes.append(
                    execution_quote_from_legacy_payload(
                        str(symbol),
                        quote_payload,
                        explicit_kind=explicit_kind,
                        fallback_ts=float(frame.minute_ts),
                    )
                )
    elif isinstance(quotes_payload, list):
        for quote_payload in quotes_payload:
            if not isinstance(quote_payload, Mapping):
                continue
            symbol = _coerce_str(quote_payload.get("symbol"))
            if not symbol:
                continue
            quotes.append(
                execution_quote_from_legacy_payload(
                    symbol,
                    quote_payload,
                    explicit_kind=explicit_kind,
                    fallback_ts=float(frame.minute_ts),
                )
            )
    quotes.sort(key=lambda item: item.ts)
    return ExecutionWindow(
        minute_ts=frame.minute_ts,
        alpha_label_ts=frame.alpha_label_ts,
        alpha_available_ts=frame.alpha_available_ts,
        alpha_frame=frame,
        quotes_1s=quotes,
    )


def position_snapshot_from_legacy_state(
    payload: Mapping[str, Any],
    *,
    explicit_kind: Optional[InstrumentKind] = None,
) -> PositionSnapshot:
    symbol = _coerce_str(payload.get("symbol"))
    instrument_traits = instrument_traits_from_legacy(symbol, payload=payload, explicit_kind=explicit_kind)
    legacy_position = _coerce_int(payload.get("position"), 0)
    option_right = _infer_option_right(payload, legacy_position=legacy_position)

    if instrument_traits.instrument_kind == InstrumentKind.OPTION:
        side = PositionSide.FLAT if legacy_position == 0 else PositionSide.LONG
    else:
        if legacy_position > 0:
            side = PositionSide.LONG
        elif legacy_position < 0:
            side = PositionSide.SHORT
        else:
            side = PositionSide.FLAT

    metadata = {
        "legacy_position": legacy_position,
        "entry_stock": _coerce_float(payload.get("entry_stock"), 0.0),
        "entry_alpha_z": _coerce_float(payload.get("entry_alpha_z"), 0.0),
        "entry_iv": _coerce_float(payload.get("entry_iv"), 0.0),
        "last_valid_iv": _coerce_float(payload.get("last_valid_iv"), 0.0),
        "max_roi": _coerce_float(payload.get("max_roi"), 0.0),
        "open_fill_confirmed": bool(payload.get("open_fill_confirmed", False)),
        "warmup_complete": bool(payload.get("warmup_complete", False)),
        "correction_mode": _coerce_str(payload.get("correction_mode"), "NORMAL"),
        "option_right": option_right,
    }
    strike_price = _coerce_float(payload.get("strike_price"), 0.0)
    if strike_price > 0.0:
        metadata["strike_price"] = strike_price
    expiry_date = payload.get("expiry_date")
    if expiry_date not in (None, ""):
        metadata["expiry_date"] = expiry_date

    return PositionSnapshot(
        symbol=symbol,
        instrument_traits=instrument_traits,
        side=side,
        quantity=abs(_coerce_float(payload.get("qty"), 0.0)),
        avg_entry_price=_coerce_float(payload.get("entry_price"), 0.0),
        entry_ts=_coerce_float(payload.get("entry_ts"), 0.0),
        contract_id=_coerce_str(payload.get("contract_id")),
        entry_frame_id=_coerce_str(payload.get("entry_frame_id")),
        entry_quote_ts=(
            None
            if payload.get("entry_quote_ts") in (None, "")
            else _coerce_float(payload.get("entry_quote_ts"), 0.0)
        ),
        metadata=metadata,
    )


__all__ = [
    "alpha_frame_from_legacy",
    "alpha_frame_item_from_legacy",
    "decision_quote_from_legacy_payload",
    "execution_quote_from_legacy_payload",
    "execution_window_from_legacy",
    "instrument_traits_from_legacy",
    "position_snapshot_from_legacy_state",
]

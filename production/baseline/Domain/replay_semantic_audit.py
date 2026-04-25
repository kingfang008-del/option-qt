from __future__ import annotations

import logging
import math
import os
from typing import Any, Dict, Iterable, List, Mapping, Optional

from .adapters import (
    alpha_frame_from_legacy,
    execution_quote_from_legacy_payload,
    position_snapshot_from_legacy_state,
)

logger = logging.getLogger("ReplaySemanticAuditor")


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _safe_float(seq: Any, idx: int, default: float = 0.0) -> float:
    try:
        if seq is None:
            return default
        return float(seq[idx]) if idx < len(seq) else default
    except Exception:
        return default


def _safe_str(seq: Any, idx: int, default: str = "") -> str:
    try:
        if seq is None:
            return default
        return str(seq[idx]) if idx < len(seq) and seq[idx] is not None else default
    except Exception:
        return default


class ReplaySemanticAuditor:
    """Optional semantic assertions for replay-side Domain validation."""

    def __init__(
        self,
        *,
        enabled: Optional[bool] = None,
        strict: Optional[bool] = None,
        log_every: Optional[int] = None,
    ) -> None:
        self.enabled = _env_flag("DOMAIN_REPLAY_ASSERT_ENABLE", False) if enabled is None else bool(enabled)
        self.strict = _env_flag("DOMAIN_REPLAY_ASSERT_STRICT", False) if strict is None else bool(strict)
        self.log_every = max(
            1,
            int(os.environ.get("DOMAIN_REPLAY_ASSERT_LOG_EVERY", "20") if log_every is None else log_every),
        )
        self._counts = {
            "pre_window_ok": 0,
            "pre_window_error": 0,
            "quote_ok": 0,
            "quote_error": 0,
            "post_window_ok": 0,
            "post_window_error": 0,
        }

    def _record(self, key_prefix: str, errors: List[str], detail: str) -> None:
        if errors:
            self._counts[f"{key_prefix}_error"] += 1
            message = f"[ReplaySemanticAudit] {key_prefix} invalid errors={len(errors)} detail={detail} first={errors[0]}"
            if self.strict:
                raise AssertionError(message)
            logger.warning(message)
            return
        self._counts[f"{key_prefix}_ok"] += 1
        if self._counts[f"{key_prefix}_ok"] == 1 or self._counts[f"{key_prefix}_ok"] % self.log_every == 0:
            logger.info(
                f"[ReplaySemanticAudit] {key_prefix} ok ok={self._counts[f'{key_prefix}_ok']} "
                f"error={self._counts[f'{key_prefix}_error']} detail={detail}"
            )

    def stats(self) -> Dict[str, int]:
        return dict(self._counts)

    def _build_alpha_frame_payload(self, signal_packet: Mapping[str, Any], minute_ts: int) -> Dict[str, Any]:
        symbols = list(signal_packet.get("symbols") or [])
        items: List[Dict[str, Any]] = []
        for idx, sym in enumerate(symbols):
            call_price = _safe_float(signal_packet.get("feed_call_price"), idx, 0.0)
            put_price = _safe_float(signal_packet.get("feed_put_price"), idx, 0.0)
            call_bid = _safe_float(signal_packet.get("feed_call_bid"), idx, 0.0)
            call_ask = _safe_float(signal_packet.get("feed_call_ask"), idx, 0.0)
            put_bid = _safe_float(signal_packet.get("feed_put_bid"), idx, 0.0)
            put_ask = _safe_float(signal_packet.get("feed_put_ask"), idx, 0.0)
            opt_data = {
                "ts": float(minute_ts),
                "has_feed": any(v > 0.0 for v in (call_price, put_price, call_bid, call_ask, put_bid, put_ask)),
                "call_price": call_price,
                "put_price": put_price,
                "call_bid": call_bid,
                "call_ask": call_ask,
                "put_bid": put_bid,
                "put_ask": put_ask,
                "call_bid_size": _safe_float(signal_packet.get("feed_call_bid_size"), idx, 0.0),
                "call_ask_size": _safe_float(signal_packet.get("feed_call_ask_size"), idx, 0.0),
                "put_bid_size": _safe_float(signal_packet.get("feed_put_bid_size"), idx, 0.0),
                "put_ask_size": _safe_float(signal_packet.get("feed_put_ask_size"), idx, 0.0),
                "call_k": _safe_float(signal_packet.get("feed_call_k"), idx, 0.0),
                "put_k": _safe_float(signal_packet.get("feed_put_k"), idx, 0.0),
                "call_iv": _safe_float(signal_packet.get("feed_call_iv"), idx, 0.0),
                "put_iv": _safe_float(signal_packet.get("feed_put_iv"), idx, 0.0),
                "call_vol": _safe_float(signal_packet.get("feed_call_vol"), idx, 0.0),
                "put_vol": _safe_float(signal_packet.get("feed_put_vol"), idx, 0.0),
                "call_id": _safe_str(signal_packet.get("feed_call_id"), idx, ""),
                "put_id": _safe_str(signal_packet.get("feed_put_id"), idx, ""),
            }
            items.append(
                {
                    "symbol": str(sym),
                    "batch_idx": idx,
                    "stock_price": _safe_float(signal_packet.get("stock_price"), idx, 0.0),
                    "alpha": _safe_float(signal_packet.get("precalc_alpha"), idx, 0.0),
                    "cs_alpha_z": _safe_float(signal_packet.get("precalc_alpha"), idx, 0.0),
                    "vol_z": _safe_float(signal_packet.get("fast_vol"), idx, 0.0),
                    "roc_5m": _safe_float(signal_packet.get("spy_roc_5min"), idx, 0.0),
                    "macd": 0.0,
                    "macd_slope": 0.0,
                    "snap_roc": 0.0,
                    "event_prob": 0.0,
                    "is_ready": True,
                    "correction_mode": "NORMAL",
                    "alpha_label_ts": _safe_float(signal_packet.get("alpha_label_ts"), idx, float(minute_ts - 60)),
                    "alpha_available_ts": _safe_float(signal_packet.get("alpha_available_ts"), idx, float(minute_ts)),
                    "opt_data": opt_data,
                }
            )
        return {
            "action": "ALPHA_FRAME",
            "source": "replay_semantic_audit",
            "ts": float(minute_ts),
            "frame_id": str(signal_packet.get("frame_id", int(minute_ts))),
            "symbols": symbols,
            "items": items,
            "index_trend": 0,
            "global_regime_band": "unknown",
            "is_zombie_market": False,
        }

    def audit_pre_window(self, minute_ts: int, signal_packet: Mapping[str, Any]) -> None:
        if not self.enabled:
            return
        payload = self._build_alpha_frame_payload(signal_packet, minute_ts)
        frame = alpha_frame_from_legacy(payload)
        errors = frame.validate()
        signal_symbols = {str(sym) for sym in (signal_packet.get("symbols") or [])}
        frame_symbols = {item.symbol for item in frame.items}
        if frame.minute_ts != int(minute_ts):
            errors.append(f"frame.minute_ts mismatch {frame.minute_ts} != {int(minute_ts)}")
        if frame_symbols != signal_symbols:
            errors.append(f"symbol set mismatch frame={sorted(frame_symbols)} signal={sorted(signal_symbols)}")
        for item in frame.items:
            quote = item.decision_quote
            if quote is not None and quote.quote_ts > float(item.alpha_available_ts):
                errors.append(f"{item.symbol} decision_quote ts exceeds alpha_available_ts")
            if quote is not None:
                option_right = str(quote.metadata.get("option_right") or "")
                if item.alpha > 0.0 and option_right == "put":
                    errors.append(f"{item.symbol} positive alpha mapped to put quote")
                if item.alpha < 0.0 and option_right == "call":
                    errors.append(f"{item.symbol} negative alpha mapped to call quote")
        self._record("pre_window", errors, f"minute_ts={minute_ts} items={len(frame.items)}")

    def audit_quote_packet(self, quote_packet: Mapping[str, Any]) -> None:
        if not self.enabled:
            return
        symbols = list(quote_packet.get("symbols") or [])
        errors: List[str] = []
        for idx, sym in enumerate(symbols):
            payload = {
                "ts": float(quote_packet.get("ts", 0.0) or 0.0),
                "call_price": _safe_float(quote_packet.get("feed_call_price"), idx, 0.0),
                "put_price": _safe_float(quote_packet.get("feed_put_price"), idx, 0.0),
                "call_bid": _safe_float(quote_packet.get("feed_call_bid"), idx, 0.0),
                "call_ask": _safe_float(quote_packet.get("feed_call_ask"), idx, 0.0),
                "put_bid": _safe_float(quote_packet.get("feed_put_bid"), idx, 0.0),
                "put_ask": _safe_float(quote_packet.get("feed_put_ask"), idx, 0.0),
            }
            alpha = _safe_float(quote_packet.get("precalc_alpha"), idx, 0.0)
            quote = execution_quote_from_legacy_payload(str(sym), payload, alpha=alpha, fallback_ts=float(payload["ts"]))
            quote_errors = quote.validate()
            errors.extend([f"{sym}.{msg}" for msg in quote_errors])
        self._record(
            "quote",
            errors,
            f"ts={float(quote_packet.get('ts', 0.0) or 0.0):.3f} symbols={len(symbols)}",
        )

    def audit_post_window(self, minute_ts: int, exec_engine: Any, quote_packet: Mapping[str, Any]) -> None:
        if not self.enabled:
            return
        errors: List[str] = []
        symbols = list(quote_packet.get("symbols") or [])
        packet_ts = float(quote_packet.get("ts", 0.0) or 0.0)
        for idx, sym in enumerate(symbols):
            cached = dict(getattr(exec_engine, "latest_execution_quote_by_symbol", {}).get(sym) or {})
            if cached:
                cached_quote = execution_quote_from_legacy_payload(
                    str(sym),
                    cached,
                    legacy_position=int(getattr(getattr(exec_engine, "states", {}).get(sym), "position", 0) or 0),
                    fallback_ts=float(cached.get("ts", 0.0) or 0.0),
                )
                cached_errors = cached_quote.validate()
                errors.extend([f"cache[{sym}].{msg}" for msg in cached_errors])
                if packet_ts > 0.0 and not math.isclose(float(cached_quote.ts), packet_ts, rel_tol=0.0, abs_tol=1e-9):
                    errors.append(f"cache[{sym}].ts mismatch {cached_quote.ts} != {packet_ts}")
            state = getattr(exec_engine, "states", {}).get(sym)
            if state is not None:
                row = dict(state.to_dict())
                row["symbol"] = str(sym)
                position = position_snapshot_from_legacy_state(row)
                pos_errors = position.validate()
                errors.extend([f"state[{sym}].{msg}" for msg in pos_errors])
        self._record("post_window", errors, f"minute_ts={minute_ts} symbols={len(symbols)}")


__all__ = ["ReplaySemanticAuditor"]

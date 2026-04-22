#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Pure helpers for Dashboard Remaining Cash source selection.

The dashboard must not infer realtime cash from trade logs. In live/dry-live
mode, the OMS Redis ledger is the only authoritative source; otherwise stale
backtest/replay rows can make Remaining Cash jump without any real trade.
"""

import json
import time
from typing import Any, Mapping, Optional, Tuple


def _decode(value: Any) -> Any:
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", errors="ignore")
    return value


def _hash_get(mapping: Mapping[Any, Any], key: str) -> Any:
    if not mapping:
        return None
    return mapping.get(key, mapping.get(key.encode("utf-8")))


def _expected_mode_set(expected_modes=None) -> set:
    return {str(mode).upper() for mode in (expected_modes or []) if str(mode)}


def _mode_matches(mode: Any, expected_modes=None) -> bool:
    expected = _expected_mode_set(expected_modes)
    if not expected:
        return True
    mode_text = str(_decode(mode) or "").upper()
    return bool(mode_text) and mode_text in expected


def parse_oms_ledger_hash(
    ledger: Mapping[Any, Any],
    *,
    now_ts: Optional[float] = None,
    max_age_sec: float = 120.0,
    allow_stale: bool = False,
    expected_modes=None,
) -> Tuple[Optional[float], Optional[str]]:
    """Return fresh OMS ledger cash if it is positive and mode-matched."""
    if not ledger:
        return None, None
    if not _mode_matches(_hash_get(ledger, "mode"), expected_modes):
        return None, None
    try:
        cash = float(_decode(_hash_get(ledger, "cash")) or 0.0)
        ts = float(_decode(_hash_get(ledger, "updated_at")) or 0.0)
    except Exception:
        return None, None
    if cash <= 0 or ts <= 0:
        return None, None
    now_ts = time.time() if now_ts is None else float(now_ts)
    if not allow_stale and now_ts - ts > float(max_age_sec):
        return None, None
    return cash, "OMS ledger"


def parse_oms_live_cash_payload(
    raw: Any,
    *,
    now_ts: Optional[float] = None,
    max_age_sec: float = 120.0,
    allow_stale: bool = False,
    expected_modes=None,
) -> Tuple[Optional[float], Optional[str]]:
    """Return legacy oms:live_positions cash only when mode is explicit."""
    if not raw:
        return None, None
    try:
        data = json.loads(_decode(raw))
    except Exception:
        return None, None
    # When a dashboard expects a run mode, legacy payloads without mode are not
    # safe enough for realtime cash display.
    if not _mode_matches(data.get("mode"), expected_modes):
        return None, None
    try:
        cash = float(data.get("cash", 0.0) or 0.0)
        ts = float(data.get("ts", 0.0) or 0.0)
    except Exception:
        return None, None
    if cash <= 0 or ts <= 0:
        return None, None
    now_ts = time.time() if now_ts is None else float(now_ts)
    if not allow_stale and now_ts - ts > float(max_age_sec):
        return None, None
    return cash, "OMS live"


def select_remaining_cash(
    *,
    live_cash: Optional[float],
    live_cash_source: Optional[str],
    log_cash: Optional[float],
    latest_cash: Optional[float],
    run_mode: str,
) -> Tuple[float, str]:
    """Choose the dashboard Remaining Cash source.

    In realtime modes, never fall back to trade-log cash. Trade logs are audit
    events, not an authoritative ledger projection, and may include replay/dry
    rows from a different process or stale dashboard query window.
    """
    if live_cash is not None:
        return float(live_cash), live_cash_source or "OMS live"

    mode = str(run_mode or "").upper()
    if not mode.startswith("REALTIME") and log_cash is not None:
        return float(log_cash), "Trade Log"

    if latest_cash is not None:
        return float(latest_cash), "PG _GLOBAL_STATE_"

    return 0.0, "No fresh OMS cash"

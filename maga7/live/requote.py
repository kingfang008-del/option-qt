"""Dynamic LMT requote pricing for Mag7 OMS (borrowed from baseline_qqq spirit).

Research replay stays on static fill_frac. Live/Paper may cancel-replace
toward the touch with hard caps. Pure functions only — no IBKR I/O here.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RequoteConfig:
    enabled: bool = True
    entry_timeout_sec: float = 3.0
    exit_timeout_sec: float = 2.0
    max_entry_requotes: int = 3
    max_exit_requotes: int = 3
    # Entry: escalate effective spread fraction from base toward max, never cross ask.
    entry_frac_step: float = 0.05
    entry_max_slippage_pct: float = 0.08
    entry_step_cap_pct: float = 0.03
    # Exit: step toward/at bid; urgent may go slightly through bid.
    exit_step: float = 0.01
    exit_urgent_min_bid_ratio: float = 0.90
    exit_urgent_max_abs_discount: float = 0.15
    cancel_settle_sec: float = 0.50


def requote_config_from_trade(trade: dict[str, Any] | None) -> RequoteConfig:
    trade = trade or {}
    risk = dict(trade.get("risk") or {})
    rq = dict(risk.get("requote") or trade.get("requote") or {})
    return RequoteConfig(
        enabled=bool(rq.get("enabled", True)),
        entry_timeout_sec=float(rq.get("entry_timeout_sec", 3.0)),
        exit_timeout_sec=float(rq.get("exit_timeout_sec", 2.0)),
        max_entry_requotes=int(rq.get("max_entry_requotes", 3)),
        max_exit_requotes=int(rq.get("max_exit_requotes", 3)),
        entry_frac_step=float(rq.get("entry_frac_step", 0.05)),
        entry_max_slippage_pct=float(rq.get("entry_max_slippage_pct", 0.08)),
        entry_step_cap_pct=float(rq.get("entry_step_cap_pct", 0.03)),
        exit_step=float(rq.get("exit_step", 0.01)),
        exit_urgent_min_bid_ratio=float(rq.get("exit_urgent_min_bid_ratio", 0.90)),
        exit_urgent_max_abs_discount=float(rq.get("exit_urgent_max_abs_discount", 0.15)),
        cancel_settle_sec=float(rq.get("cancel_settle_sec", 0.50)),
    )


def _tick_floor(px: float) -> float:
    return math.floor(float(px) * 100.0 + 1e-9) / 100.0


def _tick_ceil(px: float) -> float:
    return math.ceil(float(px) * 100.0 - 1e-9) / 100.0


def entry_requote_limit(
    *,
    bid: float,
    ask: float,
    attempt_no: int,
    base_frac: float,
    max_frac: float,
    ref_price: float,
    prev_limit: float,
    cfg: RequoteConfig,
) -> float | None:
    """Next BUY limit. attempt_no>=1. None = stop chasing (cap / bad quote)."""
    if not (math.isfinite(bid) and math.isfinite(ask) and ask >= bid > 0):
        return None
    attempt = max(int(attempt_no), 0)
    frac = min(float(base_frac) + float(cfg.entry_frac_step) * attempt, float(max_frac))
    frac = max(0.0, min(frac, 1.0))
    raw = bid + (ask - bid) * frac
    ask_tick = _tick_ceil(ask)
    bid_tick = _tick_floor(bid)
    ask_minus = max(round(ask_tick - 0.01, 2), round(bid_tick, 2))
    candidate = _tick_floor(raw)
    candidate = max(round(bid_tick, 2), min(candidate, ask_minus))
    if candidate < 0.05:
        return None
    ref = float(ref_price or 0.0)
    if ref > 0:
        cap = ref * (1.0 + float(cfg.entry_max_slippage_pct))
        if candidate > cap + 1e-9:
            return None
        candidate = min(candidate, cap)
    prev = float(prev_limit or 0.0)
    if prev > 0:
        step_cap = prev * (1.0 + float(cfg.entry_step_cap_pct))
        candidate = min(candidate, step_cap)
    return round(float(candidate), 2)


def exit_requote_limit(
    *,
    bid: float,
    ask: float,
    attempt_no: int,
    prev_limit: float,
    urgent: bool,
    cfg: RequoteConfig,
) -> float | None:
    """Next SELL limit. attempt_no>=1. Normal: stick to bid; urgent: may go through."""
    if not (math.isfinite(bid) and bid > 0):
        return None
    attempt = max(int(attempt_no), 1)
    step = float(cfg.exit_step) * max(attempt - 1, 0)
    if urgent:
        floor = max(
            bid * float(cfg.exit_urgent_min_bid_ratio),
            bid - float(cfg.exit_urgent_max_abs_discount),
            0.01,
        )
        candidate = max(bid - float(cfg.exit_step) - step, floor)
    else:
        # Move from previous passive limit down to bid; never below bid.
        prev = float(prev_limit or 0.0)
        if prev > bid:
            candidate = max(prev - float(cfg.exit_step), bid)
        else:
            candidate = bid
        candidate = max(candidate, bid)
    if ask > 0 and candidate > ask:
        candidate = ask
    if candidate < 0.01:
        return None
    return round(float(candidate), 2)


def is_urgent_exit_reason(reason: str) -> bool:
    reason_u = str(reason or "").upper()
    if reason_u in {
        "SL",
        "DAY_CIRCUIT",
        "EOD",
        "EXIT_CHASE_CAP",
        "GAP_FLATTEN",
        "ADVERSE_FILL_FLATTEN",
        "TRADE_TOX",
        "TRADE_TOX_RECONNECT",
        "HOLD_SHOCK",
        "PROFIT_PROTECT",
    }:
        return True
    if reason_u.startswith("SL") or reason_u.startswith("TRADE_TOX"):
        return True
    return False

"""Live risk guards for Mag7 OMS — staleness, spread, gap jump, adverse fill, chase.

These are hard gates for Shadow/Paper/Live. Research replay does not use them.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RiskConfig:
    max_stock_staleness_sec: float = 2.0
    max_option_staleness_sec: float = 5.0
    max_spread_pct: float = 0.25
    max_entry_mid_jump_pct: float = 0.15
    max_exit_mid_jump_pct: float = 0.20
    max_gap_hold_ticks: int = 3
    max_fill_spread_frac: float = 0.95
    max_exit_chase: int = 3
    day_circuit_force_flatten: bool = True
    halt_entries_on_gap: bool = True


def risk_config_from_trade(trade: dict[str, Any] | None, connector_cfg: Any = None) -> RiskConfig:
    trade = trade or {}
    risk = dict(trade.get("risk") or {})
    stock_stale = risk.get("max_stock_staleness_sec")
    opt_stale = risk.get("max_option_staleness_sec")
    if stock_stale is None and connector_cfg is not None:
        stock_stale = getattr(connector_cfg, "max_stock_staleness_sec", 2.0)
    if opt_stale is None and connector_cfg is not None:
        opt_stale = getattr(connector_cfg, "max_option_staleness_sec", 5.0)
    return RiskConfig(
        max_stock_staleness_sec=float(stock_stale if stock_stale is not None else 2.0),
        max_option_staleness_sec=float(opt_stale if opt_stale is not None else 5.0),
        max_spread_pct=float(risk.get("max_spread_pct", 0.25)),
        max_entry_mid_jump_pct=float(risk.get("max_entry_mid_jump_pct", 0.15)),
        max_exit_mid_jump_pct=float(risk.get("max_exit_mid_jump_pct", 0.20)),
        max_gap_hold_ticks=int(risk.get("max_gap_hold_ticks", 3)),
        max_fill_spread_frac=float(risk.get("max_fill_spread_frac", 0.95)),
        max_exit_chase=int(risk.get("max_exit_chase", 3)),
        day_circuit_force_flatten=bool(risk.get("day_circuit_force_flatten", True)),
        halt_entries_on_gap=bool(risk.get("halt_entries_on_gap", True)),
    )


def quote_mid(bid: float, ask: float) -> float | None:
    if not (math.isfinite(bid) and math.isfinite(ask) and ask >= bid > 0):
        return None
    return 0.5 * (bid + ask)


def is_fresh(ts: float | None, *, now: float, max_age_sec: float) -> bool:
    if ts is None or not math.isfinite(float(ts)):
        return False
    return (now - float(ts)) <= float(max_age_sec)


def spread_pct(bid: float, ask: float) -> float | None:
    mid = quote_mid(bid, ask)
    if mid is None or mid <= 0:
        return None
    return (ask - bid) / mid


def spread_ok(bid: float, ask: float, *, max_spread_pct: float) -> tuple[bool, str]:
    pct = spread_pct(bid, ask)
    if pct is None:
        return False, "bad_quote"
    if pct > float(max_spread_pct):
        return False, "spread_too_wide"
    return True, "ok"


def mid_jump_pct(prev_mid: float | None, mid: float) -> float | None:
    if prev_mid is None or not math.isfinite(prev_mid) or prev_mid <= 0:
        return None
    if not math.isfinite(mid) or mid <= 0:
        return None
    return abs(mid - prev_mid) / prev_mid


def entry_quote_ok(
    *,
    bid: float,
    ask: float,
    prev_mid: float | None,
    cfg: RiskConfig,
) -> tuple[bool, str, float | None]:
    """Validate option quote for entry. Returns (ok, reason, mid)."""
    mid = quote_mid(bid, ask)
    if mid is None:
        return False, "bad_quote", None
    ok, reason = spread_ok(bid, ask, max_spread_pct=cfg.max_spread_pct)
    if not ok:
        return False, reason, mid
    jump = mid_jump_pct(prev_mid, mid)
    if jump is not None and jump > cfg.max_entry_mid_jump_pct:
        return False, "entry_mid_jump", mid
    return True, "ok", mid


def observe_exit_mid(
    *,
    last_good_mid: float,
    mid: float,
    gap_hold_count: int,
    cfg: RiskConfig,
) -> tuple[float, int, str]:
    """Update last_good_mid / gap hold. Returns (new_last_good, new_hold, status).

    status: init | stable | gap | gap_force
    """
    if last_good_mid <= 0 or not math.isfinite(last_good_mid):
        return float(mid), 0, "init"
    jump = mid_jump_pct(last_good_mid, mid)
    if jump is None:
        return last_good_mid, gap_hold_count + 1, "gap"
    if jump <= cfg.max_exit_mid_jump_pct:
        return float(mid), 0, "stable"
    hold = gap_hold_count + 1
    if hold >= cfg.max_gap_hold_ticks:
        return last_good_mid, hold, "gap_force"
    return last_good_mid, hold, "gap"


def fill_adverse(
    *,
    bid: float,
    ask: float,
    fill_px: float,
    side: str,
    cfg: RiskConfig,
) -> tuple[bool, str, float]:
    """True if fill is adversely worse than allowed spread fraction."""
    mid = quote_mid(bid, ask)
    spread = ask - bid if mid is not None else 0.0
    side_u = side.upper()
    if spread <= 0 or mid is None:
        return True, "bad_quote_for_audit", float("nan")
    if side_u in {"BUY", "BOT", "LONG"}:
        frac = (fill_px - bid) / spread
        # Hard: never accept fill through the ask by more than a tiny epsilon.
        if fill_px > ask * 1.001:
            return True, "fill_above_ask", float(frac)
        if frac > cfg.max_fill_spread_frac:
            return True, "fill_spread_frac", float(frac)
    else:
        frac = (ask - fill_px) / spread
        if fill_px < bid * 0.999:
            return True, "fill_below_bid", float(frac)
        if frac > cfg.max_fill_spread_frac:
            return True, "fill_spread_frac", float(frac)
    return False, "ok", float(frac)

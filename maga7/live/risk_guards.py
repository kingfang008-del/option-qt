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
    # Data-integrity entry gates (anti suicide opens on first dirty print / broken feed).
    halt_entries_on_feed_unhealthy: bool = True
    require_live_market_data: bool = True
    require_entry_quote_stable_ticks: int = 2
    max_universe_stale_frac: float = 0.34
    universe_stale_mult: float = 2.5
    entry_cooldown_after_gap_sec: float = 120.0
    max_future_skew_sec: float = 1.0
    max_signal_age_sec: float = 0.0


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
        halt_entries_on_feed_unhealthy=bool(
            risk.get("halt_entries_on_feed_unhealthy", True)
        ),
        require_live_market_data=bool(risk.get("require_live_market_data", True)),
        require_entry_quote_stable_ticks=max(
            1, int(risk.get("require_entry_quote_stable_ticks", 2))
        ),
        max_universe_stale_frac=float(risk.get("max_universe_stale_frac", 0.34)),
        universe_stale_mult=float(risk.get("universe_stale_mult", 2.5)),
        entry_cooldown_after_gap_sec=float(
            risk.get("entry_cooldown_after_gap_sec", 120.0)
        ),
        max_future_skew_sec=max(0.0, float(risk.get("max_future_skew_sec", 1.0))),
        max_signal_age_sec=max(0.0, float(risk.get("max_signal_age_sec", 0.0))),
    )


def quote_mid(bid: float, ask: float) -> float | None:
    if not (math.isfinite(bid) and math.isfinite(ask) and ask >= bid > 0):
        return None
    return 0.5 * (bid + ask)


def is_fresh(
    ts: float | None,
    *,
    now: float,
    max_age_sec: float,
    max_future_skew_sec: float = 1.0,
) -> bool:
    if ts is None or not math.isfinite(float(ts)):
        return False
    age = float(now) - float(ts)
    return -float(max_future_skew_sec) <= age <= float(max_age_sec)


def spread_pct(bid: float, ask: float) -> float | None:
    mid = quote_mid(bid, ask)
    if mid is None or mid <= 0:
        return None
    return (ask - bid) / mid


def quote_spread_fields(
    bid: float,
    ask: float,
    *,
    fill_px: float | None = None,
    side: str | None = None,
) -> dict[str, float | None]:
    """Snapshot bid/ask/spread for OPEN/CLOSE trade records (Dash / audits)."""
    bid_f = float(bid) if bid is not None else 0.0
    ask_f = float(ask) if ask is not None else 0.0
    mid = quote_mid(bid_f, ask_f)
    abs_spread = (ask_f - bid_f) if mid is not None else None
    pct = spread_pct(bid_f, ask_f)
    fill_frac: float | None = None
    if (
        fill_px is not None
        and side is not None
        and mid is not None
        and abs_spread is not None
        and abs_spread > 0
    ):
        px = float(fill_px)
        side_u = str(side).upper()
        if side_u == "BUY":
            fill_frac = (px - bid_f) / abs_spread
        elif side_u == "SELL":
            fill_frac = (ask_f - px) / abs_spread
    return {
        "bid": bid_f if mid is not None else None,
        "ask": ask_f if mid is not None else None,
        "spread": abs_spread,
        "spread_pct": pct,
        "fill_spread_frac": fill_frac,
    }


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
    max_spread_pct: float | None = None,
    min_mid: float | None = None,
) -> tuple[bool, str, float | None]:
    """Validate option quote for entry. Returns (ok, reason, mid)."""
    mid = quote_mid(bid, ask)
    if mid is None:
        return False, "bad_quote", None
    if min_mid is not None and mid < float(min_mid):
        return False, "entry_mid_too_low", mid
    spread_cap = (
        float(cfg.max_spread_pct)
        if max_spread_pct is None
        else min(float(cfg.max_spread_pct), float(max_spread_pct))
    )
    ok, reason = spread_ok(bid, ask, max_spread_pct=spread_cap)
    if not ok:
        return False, reason, mid
    jump = mid_jump_pct(prev_mid, mid)
    if jump is not None and jump > cfg.max_entry_mid_jump_pct:
        return False, "entry_mid_jump", mid
    return True, "ok", mid


def signal_quote_lag_ok(
    *,
    signal_ts: float,
    quote_ts: float,
    max_lag_sec: float | None,
) -> tuple[bool, str, float | None]:
    """Match offline entry semantics: first usable quote must follow signal promptly."""
    if max_lag_sec is None or float(max_lag_sec) <= 0:
        return True, "ok", None
    if not (
        math.isfinite(float(signal_ts))
        and float(signal_ts) > 0
        and math.isfinite(float(quote_ts))
        and float(quote_ts) > 0
    ):
        return False, "option_quote_timestamp_missing", None
    lag = float(quote_ts) - float(signal_ts)
    if lag < 0:
        return False, "option_quote_before_signal", lag
    if lag > float(max_lag_sec):
        return False, "option_quote_lag_exceeded", lag
    return True, "ok", lag


def entry_feed_ok(
    *,
    connected: bool,
    data_mode: str | None,
    stock_lags_sec: dict[str, float | None],
    cfg: RiskConfig,
    now: float | None = None,
    last_gap_event_ts: float = 0.0,
) -> tuple[bool, str]:
    """Hard gate: refuse new entries when the live tape looks broken.

    Covers IB disconnect, non-LIVE MD (incl. shadow/dry), universe staleness,
    and a cooldown after GAP_FLATTEN so we don't reopen into a sick book.
    """
    _ = now  # reserved for future absolute-clock checks
    if not cfg.halt_entries_on_feed_unhealthy:
        return True, "ok"
    if not connected:
        return False, "ibkr_disconnected"
    if cfg.require_live_market_data and str(data_mode or "").upper() != "LIVE":
        mode = str(data_mode or "unknown").lower() or "unknown"
        return False, f"market_data_{mode}"
    cooldown = float(cfg.entry_cooldown_after_gap_sec)
    if cooldown > 0 and last_gap_event_ts > 0 and now is not None:
        if (float(now) - float(last_gap_event_ts)) < cooldown:
            return False, "gap_cooldown"
    symbols = [str(s).upper() for s in stock_lags_sec.keys()]
    if len(symbols) >= 3:
        max_lag = max(
            float(cfg.max_stock_staleness_sec) * float(cfg.universe_stale_mult),
            float(cfg.max_stock_staleness_sec) + 1.0,
        )
        stale_n = 0
        for sym in symbols:
            lag = stock_lags_sec.get(sym)
            if lag is None or not math.isfinite(float(lag)) or float(lag) > max_lag:
                stale_n += 1
        frac = stale_n / float(len(symbols))
        if frac > float(cfg.max_universe_stale_frac):
            return False, "universe_stale"
    return True, "ok"


def next_entry_quote_stable_ticks(
    *,
    prev_stable: int,
    quote_ok: bool,
    prev_mid: float | None,
    require_ticks: int,
    prev_quote_ts: float | None = None,
    quote_ts: float | None = None,
) -> tuple[int, bool, str]:
    """Accumulate consecutive good option prints before allowing entry.

    First print after missing/stale (prev_mid is None) never opens — it only
    warms the streak. Returns (new_stable, ready, reason).
    """
    need = max(1, int(require_ticks))
    if not quote_ok:
        return 0, False, "bad_quote"
    if (
        prev_quote_ts is not None
        and quote_ts is not None
        and (
            not math.isfinite(float(prev_quote_ts))
            or not math.isfinite(float(quote_ts))
            or float(quote_ts) <= float(prev_quote_ts)
        )
    ):
        return int(prev_stable), False, "option_quote_not_advanced"
    if max(1, int(require_ticks)) <= 1:
        return max(1, int(prev_stable)), True, "ok"
    if prev_mid is None:
        return 1, False, "option_quote_warmup"
    stable = int(prev_stable) + 1
    if stable < need:
        return stable, False, "option_quote_warmup"
    return stable, True, "ok"


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

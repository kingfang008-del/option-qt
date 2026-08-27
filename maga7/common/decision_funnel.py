"""Frozen candidate → Top2 decision funnel (V2 architecture).

This module is the single source of truth for:
  - Smooth ∨ Impulse candidate generation
  - per-symbol/dir first-fire merge
  - day portfolio Top2 selection
  - reject → replacement seat chain

Do not change frozen defaults without bumping FUNNEL_VERSION.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd

from maga7.common.smooth_trend import (
    ImpulseLaunchConfig,
    SmoothLaunch,
    SmoothLaunchConfig,
    SmoothStockTradeConfig,
    detect_impulse_launches_day,
    detect_smooth_launches_day,
    merge_dual_sleeve_launches,
)

FUNNEL_VERSION = "top2_smooth_impulse_v1"

# Frozen production research defaults (V2 Phase 0).
FROZEN_SMOOTH = SmoothLaunchConfig(
    scan_start="09:45",
    scan_end="11:30",
    min_look_ret=0.002,
    cooldown_minutes=60,
)
FROZEN_IMPULSE = ImpulseLaunchConfig(
    scan_start="09:45",
    scan_end="11:30",
    min_look_ret=0.004,
    cooldown_minutes=30,
)
FROZEN_TRADE = SmoothStockTradeConfig(
    first_per_symbol_dir=True,
    prefer_smooth_over_impulse=True,
    max_positions=2,
    break_max_adverse=0.012,
    max_hold_minutes=180,
    break_min_up_frac=0.35,
)

SYMS_MAG7 = ["NVDA", "TSLA", "AAPL", "AMZN", "META", "MSFT", "AMD", "GOOGL"]


@dataclass(frozen=True)
class FunnelConfig:
    smooth: SmoothLaunchConfig = FROZEN_SMOOTH
    impulse: ImpulseLaunchConfig = FROZEN_IMPULSE
    max_positions: int = 2
    first_per_symbol_dir: bool = True
    prefer_smooth: bool = True
    directions: tuple[str, ...] = ("UP", "DN")
    version: str = FUNNEL_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "max_positions": self.max_positions,
            "first_per_symbol_dir": self.first_per_symbol_dir,
            "prefer_smooth": self.prefer_smooth,
            "directions": list(self.directions),
            "smooth": asdict(self.smooth),
            "impulse": asdict(self.impulse),
        }


def launch_to_candidate(ln: SmoothLaunch, sleeve: str) -> dict[str, Any]:
    return {
        "date": str(ln.date),
        "symbol": str(ln.symbol).upper(),
        "direction": str(ln.direction).upper(),
        "sleeve": sleeve,
        "detect_ts": pd.Timestamp(ln.detect_ts),
        "price": float(ln.price),
        "score": float(ln.score),
        "look_ret": float(ln.look_ret),
        "path_eff": float(ln.path_eff),
        "up_frac": float(ln.up_frac),
        "max_dd": float(ln.max_dd),
        "from_extreme": float(ln.from_extreme),
        "funnel_version": FUNNEL_VERSION,
    }


def collect_day_candidates(
    stock_by_symbol: dict[str, pd.DataFrame],
    *,
    date: str,
    cfg: FunnelConfig | None = None,
) -> list[dict[str, Any]]:
    """All merged candidates for one session date across symbols."""
    cfg = cfg or FunnelConfig()
    out: list[dict[str, Any]] = []
    for sym, raw in stock_by_symbol.items():
        if raw is None or raw.empty:
            continue
        day = raw[raw["date"].astype(str) == str(date)]
        if day.empty:
            continue
        smooth = detect_smooth_launches_day(
            day, symbol=sym, date=date, cfg=cfg.smooth, directions=cfg.directions
        )
        impulse = detect_impulse_launches_day(
            day, symbol=sym, date=date, cfg=cfg.impulse, directions=cfg.directions
        )
        merged = merge_dual_sleeve_launches(
            smooth,
            impulse,
            first_per_symbol_dir=cfg.first_per_symbol_dir,
            prefer_smooth=cfg.prefer_smooth,
        )
        out.extend(launch_to_candidate(ln, sleeve) for ln, sleeve in merged)
    out.sort(key=lambda r: (pd.Timestamp(r["detect_ts"]), -float(r["score"])))
    return out


def select_top_seats(
    candidates: list[dict[str, Any]],
    *,
    max_positions: int = 2,
) -> list[dict[str, Any]]:
    """Select TopN seats: earliest unique symbols, one seat per symbol."""
    picked: list[dict[str, Any]] = []
    seen: set[str] = set()
    for c in candidates:
        sym = str(c["symbol"]).upper()
        if sym in seen:
            continue
        if len(picked) >= max_positions:
            break
        seat = dict(c)
        seat["seat_rank"] = len(picked) + 1
        seat["is_selected"] = True
        picked.append(seat)
        seen.add(sym)
    return picked


def build_replacement_chain(
    candidates: list[dict[str, Any]],
    seat: dict[str, Any],
    *,
    max_positions: int = 2,
    max_alts: int = 8,
) -> list[dict[str, Any]]:
    """Candidates that would fill this seat if the seat candidate is rejected.

    Re-simulates TopN selection after removing the seat's symbol from the pool.
    Returns ordered alternates that newly occupy a seat (or would occupy the
    freed capacity).
    """
    banned = {str(seat["symbol"]).upper()}
    remaining = [c for c in candidates if str(c["symbol"]).upper() not in banned]
    # Exact: re-run selection excluding the rejected symbol.
    alt_selected = select_top_seats(remaining, max_positions=max_positions)
    # Alternates = newly selected symbols that were not in original selection
    # except we only care about replacements relative to rejecting this seat.
    original = select_top_seats(candidates, max_positions=max_positions)
    original_syms = {str(x["symbol"]).upper() for x in original}
    alts: list[dict[str, Any]] = []
    for i, a in enumerate(alt_selected):
        sym = str(a["symbol"]).upper()
        if sym in original_syms and sym != str(seat["symbol"]).upper():
            # Still selected for another seat — not a replacement for this seat.
            continue
        if sym == str(seat["symbol"]).upper():
            continue
        # A true replacement is any newly selected symbol after reject,
        # or the symbol that takes the freed slot.
        row = dict(a)
        row["alt_rank"] = len(alts) + 1
        row["replaces_seat_rank"] = int(seat["seat_rank"])
        row["is_selected"] = False
        alts.append(row)
        if len(alts) >= max_alts:
            break
    # If exact reselection didn't surface new names (edge cases), fall back to
    # chronological next unique symbols after the seat.
    if not alts:
        blocked = original_syms - {str(seat["symbol"]).upper()}
        for c in candidates:
            if pd.Timestamp(c["detect_ts"]) < pd.Timestamp(seat["detect_ts"]):
                continue
            sym = str(c["symbol"]).upper()
            if sym == str(seat["symbol"]).upper() or sym in blocked:
                continue
            row = dict(c)
            row["alt_rank"] = len(alts) + 1
            row["replaces_seat_rank"] = int(seat["seat_rank"])
            row["is_selected"] = False
            alts.append(row)
            blocked.add(sym)
            if len(alts) >= max_alts:
                break
    return alts


def day_decision_seats(
    stock_by_symbol: dict[str, pd.DataFrame],
    *,
    date: str,
    cfg: FunnelConfig | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return (selected_seats_with_alts, all_candidates)."""
    cfg = cfg or FunnelConfig()
    cands = collect_day_candidates(stock_by_symbol, date=date, cfg=cfg)
    seats = select_top_seats(cands, max_positions=cfg.max_positions)
    for seat in seats:
        seat["replacement_chain"] = build_replacement_chain(
            cands, seat, max_positions=cfg.max_positions
        )
    return seats, cands

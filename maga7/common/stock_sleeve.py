"""Independent stock sleeve — research only, never mutates options baseline.

Entry: multi-factor first time in top2 → long (UP) / short (DN) underlying.
Sizing / concurrency / exits are sleeve-local; PnL is stock close-to-close.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

from maga7.common.macro_unusual import prepare_day
from maga7.common.multifactor_rank import (
    FactorSnap,
    MultiFactorConfig,
    iter_first_top2_entries,
)

ExitMode = Literal["eod", "window_end", "hold_30", "hold_60", "hold_120"]


@dataclass(frozen=True)
class StockSleeveConfig:
    position_frac: float = 0.25
    concurrent: int = 4
    max_trades_per_day: int = 4
    # Separate UP/DN caps so early DN fills do not block a late UP mover (META).
    max_up: int = 2
    max_dn: int = 2
    # If side full, replace weakest open score when a stronger first-top2 arrives.
    displace: bool = True
    window_start: str = "10:30"
    window_end: str = "14:00"
    exit_mode: ExitMode = "eod"
    stable_bars: int = 1
    step_minutes: int = 1
    directions: tuple[str, ...] = ("UP", "DN")
    # one-way cost as fraction of notional (round-trip = 2x applied on ret)
    cost_bps: float = 1.0
    start_equity: float = 100.0


def _price_at(day: pd.DataFrame, tod: str) -> tuple[pd.Timestamp | None, float | None]:
    upto = day[day["tod"] <= tod]
    if upto.empty:
        return None, None
    row = upto.iloc[-1]
    return pd.Timestamp(row["_ts"]), float(row["close"])


def _exit_bar(
    day: pd.DataFrame,
    *,
    entry_ts: pd.Timestamp,
    cfg: StockSleeveConfig,
) -> tuple[pd.Timestamp | None, float | None, str]:
    if day.empty:
        return None, None, "no_day"
    mode = str(cfg.exit_mode)
    if mode == "window_end":
        upto = day[day["tod"] <= cfg.window_end]
        if upto.empty:
            return None, None, "no_window_end"
        row = upto.iloc[-1]
        return pd.Timestamp(row["_ts"]), float(row["close"]), "window_end"
    if mode.startswith("hold_"):
        mins = int(mode.split("_", 1)[1])
        target = entry_ts + pd.Timedelta(minutes=mins)
        after = day[day["_ts"] >= target]
        if after.empty:
            row = day.iloc[-1]
            return pd.Timestamp(row["_ts"]), float(row["close"]), f"hold_{mins}_eod"
        row = after.iloc[0]
        return pd.Timestamp(row["_ts"]), float(row["close"]), f"hold_{mins}"
    # eod
    row = day.iloc[-1]
    return pd.Timestamp(row["_ts"]), float(row["close"]), "eod"


def simulate_stock_entry(
    stock_by: dict[str, pd.DataFrame],
    snap: FactorSnap,
    *,
    cfg: StockSleeveConfig,
    day_cache: dict[str, pd.DataFrame] | None = None,
) -> dict[str, Any] | None:
    day_cache = day_cache if day_cache is not None else {}
    sym = snap.symbol
    if sym not in day_cache:
        sdf = stock_by.get(sym)
        if sdf is None:
            return None
        day_cache[sym] = prepare_day(sdf, snap.date)
    day = day_cache[sym]
    if day.empty:
        return None
    entry_ts, entry_px = _price_at(day, snap.tod)
    if entry_ts is None or entry_px is None or entry_px <= 0:
        return None
    exit_ts, exit_px, exit_reason = _exit_bar(day, entry_ts=entry_ts, cfg=cfg)
    if exit_ts is None or exit_px is None or exit_px <= 0:
        return None
    if exit_ts < entry_ts:
        return None
    raw = exit_px / entry_px - 1.0
    signed = raw if snap.direction == "UP" else -raw
    cost = 2.0 * (float(cfg.cost_bps) / 1e4)
    ret = float(signed - cost)
    return {
        "date": str(snap.date),
        "symbol": sym,
        "dir": snap.direction,
        "entry_ts": entry_ts,
        "exit_ts": exit_ts,
        "entry_px": float(entry_px),
        "exit_px": float(exit_px),
        "entry_tod": str(snap.tod),
        "fp": float(snap.fp),
        "score": float(snap.score),
        "rank": int(snap.rank),
        "raw_stock_ret": float(raw),
        "ret": ret,
        "exit_reason": exit_reason,
        "route": "stock_sleeve",
        "event_source": "mf_top2",
    }


def collect_day_entries(
    stock_by: dict[str, pd.DataFrame],
    *,
    symbols: list[str],
    dates: list[str],
    mf_cfg: MultiFactorConfig,
    sleeve_cfg: StockSleeveConfig,
    tod_median_by_sym_date: dict[tuple[str, str], dict[str, float]],
) -> dict[str, list[FactorSnap]]:
    """Precompute causal first-top2 fills per date — exit-agnostic.

    Per-direction caps. If a side is full and ``displace=True`` (default), a new
    first-top2 name replaces the weakest open score on that side (causal).
    """
    by_date: dict[str, list[FactorSnap]] = {}
    displace = bool(getattr(sleeve_cfg, "displace", True))
    for date in dates:
        med = {s: tod_median_by_sym_date.get((s, str(date)), {}) for s in symbols}
        snaps = iter_first_top2_entries(
            stock_by,
            date=str(date),
            symbols=symbols,
            cfg=mf_cfg,
            tod_median_by_sym=med,
            step_minutes=int(sleeve_cfg.step_minutes),
            directions=tuple(sleeve_cfg.directions),  # type: ignore[arg-type]
            stable_bars=int(sleeve_cfg.stable_bars),
        )
        open_up: list[FactorSnap] = []
        open_dn: list[FactorSnap] = []
        # One symbol ≤ one side per day (avoid META DN then META UP).
        sym_side: dict[str, str] = {}
        for snap in snaps:
            prev = sym_side.get(snap.symbol)
            if prev is not None and prev != snap.direction:
                # Later opposite top2 flips the symbol (drop earlier side).
                if prev == "UP":
                    open_up = [s for s in open_up if s.symbol != snap.symbol]
                else:
                    open_dn = [s for s in open_dn if s.symbol != snap.symbol]
                sym_side.pop(snap.symbol, None)
            side = open_up if snap.direction == "UP" else open_dn
            cap = int(sleeve_cfg.max_up if snap.direction == "UP" else sleeve_cfg.max_dn)
            if any(s.symbol == snap.symbol for s in side):
                continue
            if len(side) < cap:
                side.append(snap)
                sym_side[snap.symbol] = snap.direction
                continue
            if not displace or not side:
                continue
            weak_i = min(range(len(side)), key=lambda i: side[i].score)
            if float(snap.score) > float(side[weak_i].score):
                dropped = side[weak_i]
                sym_side.pop(dropped.symbol, None)
                side[weak_i] = snap
                sym_side[snap.symbol] = snap.direction
        picked = sorted(open_up + open_dn, key=lambda s: (s.asof, s.symbol))
        if len(picked) > int(sleeve_cfg.max_trades_per_day):
            picked = picked[: int(sleeve_cfg.max_trades_per_day)]
        if len(picked) > int(sleeve_cfg.concurrent):
            picked = picked[: int(sleeve_cfg.concurrent)]
        by_date[str(date)] = picked
    return by_date


def replay_stock_sleeve(
    stock_by: dict[str, pd.DataFrame],
    *,
    symbols: list[str],
    dates: list[str],
    mf_cfg: MultiFactorConfig,
    sleeve_cfg: StockSleeveConfig,
    tod_median_by_sym_date: dict[tuple[str, str], dict[str, float]],
    entries_by_date: dict[str, list[FactorSnap]] | None = None,
) -> dict[str, Any]:
    """Causal stock-sleeve replay. Independent equity path."""
    if entries_by_date is None:
        entries_by_date = collect_day_entries(
            stock_by,
            symbols=symbols,
            dates=dates,
            mf_cfg=mf_cfg,
            sleeve_cfg=sleeve_cfg,
            tod_median_by_sym_date=tod_median_by_sym_date,
        )

    equity = float(sleeve_cfg.start_equity)
    peak = equity
    maxdd = 0.0
    trades: list[dict[str, Any]] = []
    daily_rows: list[dict[str, Any]] = []

    for date in dates:
        day_start = equity
        day_cache: dict[str, pd.DataFrame] = {}
        day_trades: list[dict[str, Any]] = []
        for snap in entries_by_date.get(str(date), []):
            row = simulate_stock_entry(
                stock_by, snap, cfg=sleeve_cfg, day_cache=day_cache
            )
            if row is None:
                continue
            row["size_frac"] = float(sleeve_cfg.position_frac)
            row["equity_before"] = float(day_start)
            day_trades.append(row)

        day_ret = 0.0
        for row in day_trades:
            contrib = float(sleeve_cfg.position_frac) * float(row["ret"])
            day_ret += contrib
            row["contrib"] = contrib
            trades.append(row)
        equity = day_start * (1.0 + day_ret)
        peak = max(peak, equity)
        dd = equity / peak - 1.0 if peak > 0 else 0.0
        maxdd = min(maxdd, dd)
        daily_rows.append(
            {
                "date": str(date),
                "day_ret": float(day_ret),
                "equity": float(equity),
                "n_trades": int(len(day_trades)),
            }
        )

    tdf = pd.DataFrame(trades)
    ddf = pd.DataFrame(daily_rows)
    total = float(equity / float(sleeve_cfg.start_equity) - 1.0)
    return {
        "trades": tdf,
        "daily": ddf,
        "summary": {
            "total_ret": total,
            "end_equity": float(equity),
            "maxdd": float(maxdd),
            "n_trades": int(len(tdf)),
            "n_days": int(len(ddf)),
            "n_trade_days": int((ddf["n_trades"] > 0).sum()) if len(ddf) else 0,
            "position_frac": float(sleeve_cfg.position_frac),
            "exit_mode": str(sleeve_cfg.exit_mode),
            "stable_bars": int(sleeve_cfg.stable_bars),
            "win_rate": float((tdf["ret"] > 0).mean()) if len(tdf) else None,
            "avg_ret": float(tdf["ret"].mean()) if len(tdf) else None,
        },
    }

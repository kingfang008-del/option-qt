"""Macro / unusual-activity candidates — research only, default off.

Causal 1m view meant to approximate "obvious K-line + volume" movers that
earliest Rule-A + TopK often miss. Emits **candidates only**; does not replace
freeze TopK or mutate research_baseline.

Fire (UP / DN):
  1. Relative volume: session cum$ / same-TOD median of prior ``lookback_days``
     ≥ ``vol_ratio_min``.
  2. Structure: close vs day open + session VWAP.
  3. Momentum: |from_prev| ≥ ``fp_min`` with sign.
  4. Optional hold bars.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

NY = "America/New_York"
Direction = Literal["UP", "DN"]


@dataclass(frozen=True)
class MacroUnusualConfig:
    lookback_days: int = 10
    vol_ratio_min: float = 1.20
    fp_min: float = 0.01
    hold_bars: int = 0
    window_start: str = "10:30"
    window_end: str = "14:00"
    require_above_open: bool = True
    require_above_vwap: bool = True
    only_up: bool = False
    min_cum_dvol: float = 5e8


@dataclass(frozen=True)
class MacroCandidate:
    symbol: str
    date: str
    direction: Direction
    sig_ts: pd.Timestamp
    from_prev: float
    vol_ratio: float
    cum_dvol: float
    score: float
    reason: str
    above_open: bool
    above_vwap: bool


def prepare_day(df: pd.DataFrame, date: str) -> pd.DataFrame:
    day = df[df["date"].astype(str) == str(date)].copy()
    if day.empty:
        return day
    ts = pd.to_datetime(day["timestamp"])
    if getattr(ts.dt, "tz", None) is None:
        ts = ts.dt.tz_localize(NY)
    else:
        ts = ts.dt.tz_convert(NY)
    day = day.assign(_ts=ts).sort_values("_ts")
    day["tod"] = day["_ts"].dt.strftime("%H:%M")
    day["dvol"] = day["close"].astype(float) * day["volume"].astype(float)
    day["cum_dvol"] = day["dvol"].cumsum()
    tp = (
        day["high"].astype(float) + day["low"].astype(float) + day["close"].astype(float)
    ) / 3.0
    vol = day["volume"].astype(float)
    day["vwap"] = (tp * vol).cumsum() / vol.cumsum().replace(0, np.nan)
    day["day_open"] = float(day.iloc[0]["open"])
    return day.reset_index(drop=True)


def build_tod_median_curve(
    stock_df: pd.DataFrame,
    *,
    before_date: str,
    lookback_days: int,
) -> dict[str, float]:
    """tod -> median cum_dvol across prior lookback sessions."""
    prev_dates = sorted(d for d in stock_df["date"].astype(str).unique() if d < str(before_date))
    prev_dates = prev_dates[-int(lookback_days) :]
    by_tod: dict[str, list[float]] = {}
    for d in prev_dates:
        day = prepare_day(stock_df, d)
        if day.empty:
            continue
        for _, row in day.iterrows():
            by_tod.setdefault(str(row["tod"]), []).append(float(row["cum_dvol"]))
    return {tod: float(np.median(vs)) for tod, vs in by_tod.items() if vs}


def _structure_ok(
    row: pd.Series, *, direction: Direction, cfg: MacroUnusualConfig
) -> tuple[bool, bool, bool]:
    px = float(row["close"])
    day_open = float(row["day_open"])
    vwap = float(row["vwap"]) if np.isfinite(row["vwap"]) else float("nan")
    above_open = px >= day_open
    above_vwap = bool(np.isfinite(vwap) and px >= vwap)
    ok = True
    if cfg.require_above_open:
        ok = ok and (above_open if direction == "UP" else px <= day_open)
    if cfg.require_above_vwap:
        ok = ok and (
            above_vwap if direction == "UP" else (np.isfinite(vwap) and px <= vwap)
        )
    return ok, above_open, above_vwap


def first_macro_fire(
    stock_df: pd.DataFrame,
    *,
    date: str,
    symbol: str,
    direction: Direction,
    cfg: MacroUnusualConfig | None = None,
    tod_median: dict[str, float] | None = None,
) -> MacroCandidate | None:
    cfg = cfg or MacroUnusualConfig()
    day = prepare_day(stock_df, date)
    if day.empty:
        return None
    if tod_median is None:
        tod_median = build_tod_median_curve(
            stock_df, before_date=date, lookback_days=cfg.lookback_days
        )
    win = day[(day["tod"] >= cfg.window_start) & (day["tod"] <= cfg.window_end)].reset_index(
        drop=True
    )
    if win.empty:
        return None

    for i in range(len(win)):
        row = win.iloc[i]
        fp = float(row["from_prev"]) if np.isfinite(row.get("from_prev", np.nan)) else float("nan")
        if not np.isfinite(fp):
            continue
        if direction == "UP" and fp < float(cfg.fp_min):
            continue
        if direction == "DN" and fp > -float(cfg.fp_min):
            continue
        struct_ok, above_open, above_vwap = _structure_ok(row, direction=direction, cfg=cfg)
        if not struct_ok:
            continue
        cum = float(row["cum_dvol"])
        if cum < float(cfg.min_cum_dvol):
            continue
        med = tod_median.get(str(row["tod"]))
        if med is None or med <= 0:
            continue
        ratio = cum / med
        if ratio < float(cfg.vol_ratio_min):
            continue
        hold_ok = True
        for j in range(1, int(cfg.hold_bars) + 1):
            if i + j >= len(win):
                hold_ok = False
                break
            nrow = win.iloc[i + j]
            nfp = float(nrow["from_prev"]) if np.isfinite(nrow.get("from_prev", np.nan)) else float("nan")
            if direction == "UP" and (not np.isfinite(nfp) or nfp < float(cfg.fp_min)):
                hold_ok = False
                break
            if direction == "DN" and (not np.isfinite(nfp) or nfp > -float(cfg.fp_min)):
                hold_ok = False
                break
            sok, _, _ = _structure_ok(nrow, direction=direction, cfg=cfg)
            if not sok:
                hold_ok = False
                break
        if not hold_ok:
            continue
        return MacroCandidate(
            symbol=str(symbol),
            date=str(date),
            direction=direction,
            sig_ts=pd.Timestamp(row["_ts"]),
            from_prev=fp,
            vol_ratio=float(ratio),
            cum_dvol=cum,
            score=float(ratio) * abs(fp),
            reason="macro_vol_structure",
            above_open=above_open,
            above_vwap=above_vwap,
        )
    return None


def scan_macro_day(
    stock_by: dict[str, pd.DataFrame],
    *,
    date: str,
    symbols: list[str],
    cfg: MacroUnusualConfig | None = None,
    tod_median_by_sym: dict[str, dict[str, float]] | None = None,
) -> list[MacroCandidate]:
    cfg = cfg or MacroUnusualConfig()
    out: list[MacroCandidate] = []
    dirs: list[Direction] = ["UP"] if cfg.only_up else ["UP", "DN"]
    for sym in symbols:
        sdf = stock_by.get(sym)
        if sdf is None or getattr(sdf, "empty", True):
            continue
        med = None if tod_median_by_sym is None else tod_median_by_sym.get(sym)
        for d in dirs:
            hit = first_macro_fire(
                sdf, date=date, symbol=sym, direction=d, cfg=cfg, tod_median=med
            )
            if hit is not None:
                out.append(hit)
    out.sort(key=lambda c: (-c.score, c.sig_ts, c.symbol))
    return out


def cfg_from_dict(blob: dict[str, Any] | None) -> MacroUnusualConfig:
    b = blob or {}
    return MacroUnusualConfig(
        lookback_days=int(b.get("lookback_days", 10) or 10),
        vol_ratio_min=float(b.get("vol_ratio_min", 1.20) or 1.20),
        fp_min=float(b.get("fp_min", 0.01) or 0.01),
        hold_bars=int(b.get("hold_bars", 0) or 0),
        window_start=str(b.get("window_start", "10:30")),
        window_end=str(b.get("window_end", "14:00")),
        require_above_open=bool(b.get("require_above_open", True)),
        require_above_vwap=bool(b.get("require_above_vwap", True)),
        only_up=bool(b.get("only_up", False)),
        min_cum_dvol=float(b.get("min_cum_dvol", 5e8) or 5e8),
    )

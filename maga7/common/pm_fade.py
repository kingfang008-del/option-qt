"""Afternoon fade sleeve (research): 5-minute extension fade, 1DTE+.

Causal rule at clocks 14:30 / 14:45 / 15:00 / 15:15 (and optional 5m grid):
  ext = close(clock) / close(14:00) - 1
  if |ext| >= ext_min → fade direction = opposite of ext
  optional confirm: last confirm_minutes return already against ext

Distinct from CORE Rule-A (continuation) and AM launch_slope (impulse).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd

NY = "America/New_York"


@dataclass(frozen=True)
class PmFadeConfig:
    enabled: bool = False
    session_start: str = "14:00"
    session_end: str = "15:30"
    anchor: str = "14:00"
    # evaluation clocks (HH:MM); empty → every step_minutes from anchor+step
    clocks: tuple[str, ...] = ("14:30", "14:45", "15:00", "15:15")
    step_minutes: int = 5
    ext_min: float = 0.008
    confirm_minutes: int = 5
    require_confirm: bool = True
    hold_minutes: int = 15
    flatten_by: str = "15:45"
    prefer_dte: int = 1
    allowed_dte: tuple[int, ...] = (1, 2)
    position_frac: float = 0.10

    @classmethod
    def from_profile(cls, profile: dict[str, Any] | None) -> "PmFadeConfig":
        raw = (profile or {}).get("pm_fade")
        if not isinstance(raw, dict):
            return cls(enabled=False)
        clocks = raw.get("clocks")
        if isinstance(clocks, str):
            clocks_t = tuple(x.strip() for x in clocks.split(",") if x.strip())
        elif isinstance(clocks, (list, tuple)):
            clocks_t = tuple(str(x) for x in clocks)
        else:
            clocks_t = ("14:30", "14:45", "15:00", "15:15")
        dte = raw.get("allowed_dte") or [1, 2]
        return cls(
            enabled=bool(raw.get("enabled", False)),
            session_start=str(raw.get("session_start") or "14:00"),
            session_end=str(raw.get("session_end") or "15:30"),
            anchor=str(raw.get("anchor") or "14:00"),
            clocks=clocks_t,
            step_minutes=int(raw.get("step_minutes", 5) or 5),
            ext_min=float(raw.get("ext_min", 0.008) or 0.008),
            confirm_minutes=int(raw.get("confirm_minutes", 5) or 5),
            require_confirm=bool(raw.get("require_confirm", True)),
            hold_minutes=int(raw.get("hold_minutes", 15) or 15),
            flatten_by=str(raw.get("flatten_by") or "15:45"),
            prefer_dte=int(raw.get("prefer_dte", 1) or 1),
            allowed_dte=tuple(int(x) for x in dte),
            position_frac=float(raw.get("position_frac", 0.10) or 0.10),
        )


def _tod_minutes(hhmm: str) -> int:
    hh, mm = str(hhmm).split(":")
    return int(hh) * 60 + int(mm)


def _px_at(day: pd.DataFrame, tod_m: int) -> float | None:
    if day is None or day.empty:
        return None
    w = day[day["_tod"] <= tod_m]
    if w.empty:
        return None
    try:
        return float(w.iloc[-1]["close"])
    except (TypeError, ValueError, IndexError):
        return None


def prepare_day(df: pd.DataFrame, date: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    g = df[df["date"].astype(str) == str(date)].copy()
    if g.empty:
        return g
    g["timestamp"] = pd.to_datetime(g["timestamp"])
    if g["timestamp"].dt.tz is None:
        g["timestamp"] = g["timestamp"].dt.tz_localize(NY)
    else:
        g["timestamp"] = g["timestamp"].dt.tz_convert(NY)
    g["_tod"] = g["timestamp"].dt.hour * 60 + g["timestamp"].dt.minute
    return g.sort_values("timestamp")


def iter_pm_fade_signals(
    day: pd.DataFrame,
    *,
    date: str,
    symbol: str,
    cfg: PmFadeConfig,
) -> list[dict[str, Any]]:
    """Return fade signal dicts for one symbol-day (causal, 1m bars)."""
    if day is None or day.empty:
        return []
    if "_tod" not in day.columns:
        day = prepare_day(day, date)
        if day.empty:
            return []
    anchor_m = _tod_minutes(cfg.anchor)
    end_m = _tod_minutes(cfg.session_end)
    px_anchor = _px_at(day, anchor_m)
    if px_anchor is None or px_anchor <= 0:
        return []
    clocks = list(cfg.clocks)
    if not clocks:
        step = max(1, int(cfg.step_minutes))
        clocks = []
        t = anchor_m + step
        while t <= end_m:
            clocks.append(f"{t // 60:02d}:{t % 60:02d}")
            t += step
    out: list[dict[str, Any]] = []
    for ck in clocks:
        ck_m = _tod_minutes(ck)
        if ck_m < anchor_m or ck_m > end_m:
            continue
        px = _px_at(day, ck_m)
        if px is None:
            continue
        ext = float(px / px_anchor - 1.0)
        if abs(ext) + 1e-12 < float(cfg.ext_min):
            continue
        fade_dir = "DN" if ext > 0 else "UP"
        confirm_ok = True
        conf_ret = None
        if cfg.require_confirm and cfg.confirm_minutes > 0:
            px_prev = _px_at(day, ck_m - int(cfg.confirm_minutes))
            if px_prev is None or px_prev <= 0:
                confirm_ok = False
            else:
                conf_ret = float(px / px_prev - 1.0)
                # last N minutes already moving against the extension
                confirm_ok = (ext > 0 and conf_ret < 0) or (ext < 0 and conf_ret > 0)
        if not confirm_ok:
            continue
        # signal timestamp = last bar ≤ clock
        w = day[day["_tod"] <= ck_m]
        ts = w.iloc[-1]["timestamp"]
        out.append(
            {
                "date": str(date),
                "symbol": str(symbol),
                "dir": fade_dir,
                "ts": ts,
                "clock": ck,
                "ext_from_anchor": ext,
                "confirm_ret": conf_ret,
                "anchor_px": px_anchor,
                "signal_px": px,
                "hold_minutes": int(cfg.hold_minutes),
                "flatten_by": cfg.flatten_by,
                "ext_min": float(cfg.ext_min),
                "prefer_dte": int(cfg.prefer_dte),
            }
        )
    return out


def scan_pm_fade_day(
    stock_by: dict[str, pd.DataFrame],
    *,
    date: str,
    symbols: Iterable[str],
    cfg: PmFadeConfig,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for sym in symbols:
        day = prepare_day(stock_by.get(str(sym)), str(date))
        rows.extend(iter_pm_fade_signals(day, date=str(date), symbol=str(sym), cfg=cfg))
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).sort_values("ts")
    # one signal per symbol-dir-day (earliest clock)
    df = df.drop_duplicates(["date", "symbol", "dir"], keep="first")
    return df.reset_index(drop=True)

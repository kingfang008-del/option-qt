"""Hold-period Watchdog: flatten on extreme index shock against the trade.

Design intent
-------------
Entry Watchdog (L1/L2) only gates **new** entries. During the T+30 hold the
option rails (TP/SL / hold_extend) still run, but a market-wide flip
(e.g. sudden QQQ plunge after an UP entry) is not covered.

This module adds a **narrow** mid-hold flatten:
  - Measure QQQ move from entry-time level to now (causal 1m, with bar delay).
  - UP trade: flatten if QQQ dropped by ≥ ``qqq_adverse_from_entry``.
  - DN trade: flatten if QQQ rose by ≥ ``qqq_adverse_from_entry``.

Keep thresholds large (≈0.8–1.5%) so routine chop does not fire. Optional
``require_option_mtm_max`` only cuts when option MTM is also weak.

Default: **off** on research_baseline until ablation clears.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class HoldWatchdogConfig:
    enabled: bool = False
    qqq_adverse_from_entry: float = 0.008
    min_hold_seconds: int = 60
    require_option_mtm_max: float | None = None


def hold_watchdog_from_trade(trade: dict[str, Any] | None) -> HoldWatchdogConfig:
    raw = (trade or {}).get("hold_watchdog") or {}
    if not isinstance(raw, dict):
        return HoldWatchdogConfig(enabled=False)
    mtm_raw = raw.get("require_option_mtm_max", None)
    mtm = float(mtm_raw) if mtm_raw is not None else None
    return HoldWatchdogConfig(
        enabled=bool(raw.get("enabled", False)),
        qqq_adverse_from_entry=float(raw.get("qqq_adverse_from_entry", 0.008) or 0.008),
        min_hold_seconds=int(raw.get("min_hold_seconds", 60) or 60),
        require_option_mtm_max=mtm,
    )


def _ensure_ts_ns(day: pd.DataFrame) -> pd.DataFrame:
    if day is None or day.empty:
        return day
    if "_ts_ns" in day.columns and "_close" in day.columns:
        return day
    out = day.copy()
    ts = pd.to_datetime(out["timestamp"])
    if getattr(ts.dt, "tz", None) is None:
        ts = ts.dt.tz_localize("America/New_York")
    else:
        ts = ts.dt.tz_convert("America/New_York")
    out["_ts_ns"] = [int(pd.Timestamp(x).value) for x in ts]
    out["_close"] = out["close"].astype(float)
    return out.sort_values("_ts_ns").reset_index(drop=True)


def qqq_close_at(qqq_day: pd.DataFrame | None, asof_ts: pd.Timestamp) -> float | None:
    """Last QQQ close at/before ``asof_ts``."""
    if qqq_day is None or getattr(qqq_day, "empty", True):
        return None
    day = _ensure_ts_ns(qqq_day)
    if day is None or day.empty:
        return None
    asof = pd.Timestamp(asof_ts)
    if asof.tzinfo is None:
        asof = asof.tz_localize("America/New_York")
    else:
        asof = asof.tz_convert("America/New_York")
    ts_ns = day["_ts_ns"].to_numpy(dtype=np.int64)
    i = int(np.searchsorted(ts_ns, int(asof.value), side="right") - 1)
    if i < 0:
        return None
    px = float(day["_close"].iloc[i])
    return px if np.isfinite(px) and px > 0 else None


def qqq_adverse_from_entry(
    qqq_day: pd.DataFrame | None,
    *,
    entry_ts: pd.Timestamp,
    now_ts: pd.Timestamp,
    direction: str,
    thresh: float,
    bar_delay_seconds: int = 0,
) -> tuple[bool, float | None]:
    """True if QQQ moved against ``direction`` by ≥ ``thresh`` since entry.

    Returns ``(fired, signed_qqq_ret)`` where signed_ret is +favorable to trade.
    """
    if thresh is None or float(thresh) <= 0:
        return False, None
    delay = pd.Timedelta(seconds=int(bar_delay_seconds or 0))
    entry_vis = pd.Timestamp(entry_ts) - delay
    now_vis = pd.Timestamp(now_ts) - delay
    px0 = qqq_close_at(qqq_day, entry_vis)
    px1 = qqq_close_at(qqq_day, now_vis)
    if px0 is None or px1 is None or px0 <= 0:
        return False, None
    raw = px1 / px0 - 1.0
    d = str(direction).upper()
    # Favorable signed ret: UP wants QQQ up; DN wants QQQ down.
    signed = raw if d == "UP" else -raw
    fired = signed <= -float(thresh)
    return bool(fired), float(signed)

"""Session-open extension gate (from_open) for chase / overbought entries.

``from_open = close_asof / day_open - 1``. Unlike ``from_prev`` (vs prior close),
this catches names that already ran hard *within* the session before Rule-A fire.

Modes:
  - ``block``: reject entry when over threshold
  - ``scale``: keep seat, multiply ``size_frac`` by ``scale``

Default ``same_sign_only=True``: only fire when ``sign(from_open)`` matches trade
direction (UP after a large green open extension, DN after a large red dump).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class FromOpenGateConfig:
    enabled: bool = False
    max_abs: float = 0.04
    mode: str = "block"  # block | scale
    scale: float = 0.5
    same_sign_only: bool = True


def parse_from_open_gate(raw: Any) -> FromOpenGateConfig:
    if not isinstance(raw, dict):
        return FromOpenGateConfig(enabled=False)
    mode = str(raw.get("mode") or "block").strip().lower()
    if mode in {"reject", "hard", "skip"}:
        mode = "block"
    if mode in {"soft", "size", "half"}:
        mode = "scale"
    if mode not in {"block", "scale"}:
        mode = "block"
    scale = float(raw.get("scale", 0.5) or 0.5)
    scale = max(0.0, min(1.0, scale))
    return FromOpenGateConfig(
        enabled=bool(raw.get("enabled", False)),
        max_abs=float(raw.get("max_abs", 0.04) or 0.04),
        mode=mode,
        scale=scale,
        same_sign_only=bool(raw.get("same_sign_only", True)),
    )


def session_from_open(
    stock_df: pd.DataFrame | None,
    *,
    date: str,
    asof_ts: pd.Timestamp,
) -> float | None:
    """Causal ``close(asof) / day_open - 1``. Missing bars → None (fail-open)."""
    if stock_df is None or stock_df.empty:
        return None
    day = stock_df[stock_df["date"].astype(str) == str(date)]
    if day.empty:
        return None
    day = day.sort_values("timestamp")
    open_col = "open" if "open" in day.columns else None
    if open_col is None:
        return None
    try:
        day_open = float(day.iloc[0][open_col])
    except (TypeError, ValueError, IndexError):
        return None
    if not (day_open > 0.0):
        return None
    asof = pd.Timestamp(asof_ts)
    if asof.tzinfo is None and day["timestamp"].dt.tz is not None:
        asof = asof.tz_localize(day["timestamp"].dt.tz)
    elif asof.tzinfo is not None and day["timestamp"].dt.tz is None:
        asof = asof.tz_localize(None)
    else:
        try:
            asof = asof.tz_convert(day["timestamp"].dt.tz)
        except (TypeError, AttributeError, ValueError):
            pass
    upto = day[day["timestamp"] <= asof]
    if upto.empty:
        return None
    try:
        px = float(upto.iloc[-1]["close"])
    except (TypeError, ValueError, IndexError):
        return None
    if not (px > 0.0):
        return None
    return float(px / day_open - 1.0)


def resolve_from_open_gate(
    cfg: FromOpenGateConfig,
    *,
    from_open: float | None,
    direction: str,
) -> tuple[str, float, float | None]:
    """Return ``(action, size_mult, from_open)``.

    ``action`` is ``allow`` | ``block`` | ``scale``.
    """
    if not cfg.enabled:
        return "allow", 1.0, from_open
    if from_open is None:
        return "allow", 1.0, None
    try:
        fo = float(from_open)
    except (TypeError, ValueError):
        return "allow", 1.0, None
    d = str(direction or "").strip().upper()
    if cfg.same_sign_only:
        if d == "UP" and fo <= 0.0:
            return "allow", 1.0, fo
        if d == "DN" and fo >= 0.0:
            return "allow", 1.0, fo
    if abs(fo) <= float(cfg.max_abs) + 1e-12:
        return "allow", 1.0, fo
    if cfg.mode == "scale":
        return "scale", float(cfg.scale), fo
    return "block", 0.0, fo

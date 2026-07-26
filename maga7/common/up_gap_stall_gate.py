"""UP overnight-gap early stalled-extension gate (causal).

Targets 06-11 TSLA: overnight UP gap (~1.75%) with near-zero session extension
at the **feature** clock (fo≈0) while still range-chasing — not a continuation
into fill (entry fo/chase already drift).

Rule (default): UP ∧ fav_gap≥min_gap ∧ |fo|≤max_abs_fo ∧ chase≥min_chase
∧ minutes_from_open≤max_sess_min → block.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from maga7.common.overnight_gap_gate import overnight_gap
from maga7.common.range_stall_gate import session_chase_and_pre5


@dataclass(frozen=True)
class UpGapStallGateConfig:
    enabled: bool = False
    min_fav_gap: float = 0.015
    max_abs_from_open: float = 0.001
    min_chase: float = 0.9
    max_sess_min: float = 40.0
    mode: str = "block"  # block | scale
    scale: float = 0.5
    on_missing: str = "allow"  # allow | block


@dataclass(frozen=True)
class UpGapStallDecision:
    allow: bool
    size_scale: float
    reason: str
    fav_gap: float | None = None
    from_open: float | None = None
    chase: float | None = None
    sess_min: float | None = None


def parse_up_gap_stall_gate(raw: Any) -> UpGapStallGateConfig:
    if not isinstance(raw, dict):
        return UpGapStallGateConfig(enabled=False)
    mode = str(raw.get("mode") or "block").strip().lower()
    if mode in {"reject", "hard", "skip"}:
        mode = "block"
    if mode in {"soft", "size", "half", "degrade"}:
        mode = "scale"
    if mode not in {"block", "scale"}:
        mode = "block"
    on_miss = str(raw.get("on_missing") or "allow").strip().lower()
    if on_miss not in {"allow", "block"}:
        on_miss = "allow"
    return UpGapStallGateConfig(
        enabled=bool(raw.get("enabled", False)),
        min_fav_gap=float(raw.get("min_fav_gap", 0.015) or 0.015),
        max_abs_from_open=float(raw.get("max_abs_from_open", 0.001) or 0.001),
        min_chase=float(raw.get("min_chase", 0.9) or 0.9),
        max_sess_min=float(raw.get("max_sess_min", 40.0) or 40.0),
        mode=mode,
        scale=max(0.0, min(1.0, float(raw.get("scale", 0.5) or 0.5))),
        on_missing=on_miss,
    )


def _minutes_from_open(asof_ts: pd.Timestamp) -> float | None:
    asof = pd.Timestamp(asof_ts)
    try:
        if asof.tzinfo is None:
            asof = asof.tz_localize("America/New_York")
        else:
            asof = asof.tz_convert("America/New_York")
    except (TypeError, ValueError):
        return None
    open_ts = asof.normalize() + pd.Timedelta(hours=9, minutes=30)
    return float((asof - open_ts).total_seconds() / 60.0)


def resolve_up_gap_stall_gate(
    cfg: UpGapStallGateConfig,
    *,
    stock_df: pd.DataFrame | None,
    date: str,
    asof_ts: pd.Timestamp,
    direction: str,
) -> UpGapStallDecision:
    if not cfg.enabled:
        return UpGapStallDecision(True, 1.0, "off")
    d = str(direction or "").upper()
    if d != "UP":
        return UpGapStallDecision(True, 1.0, "dir_skip")
    gap = overnight_gap(stock_df, date=str(date))
    chase, _pre5, from_open = session_chase_and_pre5(
        stock_df, date=str(date), asof_ts=asof_ts, direction=d, pre_seconds=300
    )
    sess_min = _minutes_from_open(asof_ts)
    if gap is None or from_open is None or chase is None or sess_min is None:
        if cfg.on_missing == "block":
            return UpGapStallDecision(False, 0.0, "missing")
        return UpGapStallDecision(True, 1.0, "missing_allow")
    fav_gap = float(gap)  # UP favorable = up gap
    if fav_gap + 1e-12 < float(cfg.min_fav_gap):
        return UpGapStallDecision(
            True,
            1.0,
            "gap_short",
            fav_gap=fav_gap,
            from_open=float(from_open),
            chase=float(chase),
            sess_min=sess_min,
        )
    if abs(float(from_open)) - 1e-12 > float(cfg.max_abs_from_open):
        return UpGapStallDecision(
            True,
            1.0,
            "fo_moved",
            fav_gap=fav_gap,
            from_open=float(from_open),
            chase=float(chase),
            sess_min=sess_min,
        )
    if chase + 1e-12 < float(cfg.min_chase):
        return UpGapStallDecision(
            True,
            1.0,
            "chase_low",
            fav_gap=fav_gap,
            from_open=float(from_open),
            chase=float(chase),
            sess_min=sess_min,
        )
    if sess_min - 1e-12 > float(cfg.max_sess_min):
        return UpGapStallDecision(
            True,
            1.0,
            "sess_late",
            fav_gap=fav_gap,
            from_open=float(from_open),
            chase=float(chase),
            sess_min=sess_min,
        )
    reason = (
        f"block_up_gap>={cfg.min_fav_gap:g}&|fo|<={cfg.max_abs_from_open:g}"
        f"&chase>={cfg.min_chase:g}&sess<={cfg.max_sess_min:g}"
    )
    if cfg.mode == "scale":
        return UpGapStallDecision(
            True,
            float(cfg.scale),
            reason.replace("block_", "degrade_", 1),
            fav_gap=fav_gap,
            from_open=float(from_open),
            chase=float(chase),
            sess_min=sess_min,
        )
    return UpGapStallDecision(
        False,
        0.0,
        reason,
        fav_gap=fav_gap,
        from_open=float(from_open),
        chase=float(chase),
        sess_min=sess_min,
    )

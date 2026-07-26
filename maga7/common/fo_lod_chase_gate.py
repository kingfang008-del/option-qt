"""Large from-open + LOD/HOD chase block (causal).

Targets late continuation entries after the move is already extended and price
sits on the session extreme — e.g. 2026-07-24 TSLA DN put bought at fo≈−3.5%
with chase≈1.0 (on LOD). Buying 0DTE there is chase, not impulse.

Default (DN / LOD): fav_from_open≥min ∧ chase≥min_chase ∧ dist_ext≤max_dist_ext
where ``dist_ext`` is ``(px-lo)/open`` for DN and ``(hi-px)/open`` for UP.
Measure at feature clock (same family as dn/up_gap_stall).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from maga7.common.range_stall_gate import session_chase_and_pre5


@dataclass(frozen=True)
class FoLodChaseGateConfig:
    enabled: bool = False
    min_fav_from_open: float = 0.03
    min_chase: float = 0.9
    max_dist_ext: float = 0.003
    dirs: tuple[str, ...] | None = ("DN",)
    mode: str = "block"  # block | scale
    scale: float = 0.5
    on_missing: str = "allow"  # allow | block


@dataclass(frozen=True)
class FoLodChaseDecision:
    allow: bool
    size_scale: float
    reason: str
    fav_from_open: float | None = None
    chase: float | None = None
    dist_ext: float | None = None


def parse_fo_lod_chase_gate(raw: Any) -> FoLodChaseGateConfig:
    if not isinstance(raw, dict):
        return FoLodChaseGateConfig(enabled=False)
    mode = str(raw.get("mode") or "block").strip().lower()
    if mode in {"reject", "hard", "skip"}:
        mode = "block"
    if mode in {"soft", "size", "half", "degrade"}:
        mode = "scale"
    if mode not in {"block", "scale"}:
        mode = "block"
    dirs_raw = raw.get("dirs") or raw.get("directions")
    if raw.get("dn_only") in (True, 1, "1", "true", "True", "yes"):
        dirs_raw = ["DN"]
    if raw.get("up_only") in (True, 1, "1", "true", "True", "yes"):
        dirs_raw = ["UP"]
    dirs: tuple[str, ...] | None = ("DN",)
    if isinstance(dirs_raw, str):
        dirs = tuple(x.strip().upper() for x in dirs_raw.split(",") if x.strip())
    elif isinstance(dirs_raw, (list, tuple)):
        dirs = tuple(str(x).strip().upper() for x in dirs_raw if str(x).strip())
    if dirs == ():
        dirs = None
    on_miss = str(raw.get("on_missing") or "allow").strip().lower()
    if on_miss not in {"allow", "block"}:
        on_miss = "allow"
    return FoLodChaseGateConfig(
        enabled=bool(raw.get("enabled", False)),
        min_fav_from_open=float(raw.get("min_fav_from_open", 0.03) or 0.03),
        min_chase=float(raw.get("min_chase", 0.9) or 0.9),
        max_dist_ext=float(raw.get("max_dist_ext", 0.003) or 0.003),
        dirs=dirs,
        mode=mode,
        scale=max(0.0, min(1.0, float(raw.get("scale", 0.5) or 0.5))),
        on_missing=on_miss,
    )


def _dist_to_extreme(
    stock_df: pd.DataFrame | None,
    *,
    date: str,
    asof_ts: pd.Timestamp,
    direction: str,
) -> float | None:
    if stock_df is None or stock_df.empty:
        return None
    if "date" not in stock_df.columns or "timestamp" not in stock_df.columns:
        return None
    day = stock_df[stock_df["date"].astype(str) == str(date)].sort_values("timestamp")
    if day.empty:
        return None
    asof = pd.Timestamp(asof_ts)
    tz = day["timestamp"].dt.tz
    if asof.tzinfo is None and tz is not None:
        asof = asof.tz_localize(tz)
    elif asof.tzinfo is not None and tz is None:
        asof = asof.tz_localize(None)
    else:
        try:
            asof = asof.tz_convert(tz)
        except (TypeError, ValueError):
            pass
    before = day[day["timestamp"] <= asof]
    if before.empty:
        return None
    try:
        op = float(day.iloc[0]["open"])
        px = float(before.iloc[-1]["close"])
        hi = float(before["high"].max()) if "high" in before.columns else px
        lo = float(before["low"].min()) if "low" in before.columns else px
    except (TypeError, ValueError, IndexError):
        return None
    if not (op > 0 and px > 0):
        return None
    d = str(direction or "").upper()
    if d == "DN":
        return float((px - lo) / op)
    if d == "UP":
        return float((hi - px) / op)
    return None


def resolve_fo_lod_chase_gate(
    cfg: FoLodChaseGateConfig,
    *,
    stock_df: pd.DataFrame | None,
    date: str,
    asof_ts: pd.Timestamp,
    direction: str,
) -> FoLodChaseDecision:
    if not cfg.enabled:
        return FoLodChaseDecision(True, 1.0, "off")
    d = str(direction or "").upper()
    if cfg.dirs is not None and d not in set(cfg.dirs):
        return FoLodChaseDecision(True, 1.0, "dir_skip")
    chase, _pre5, from_open = session_chase_and_pre5(
        stock_df, date=str(date), asof_ts=asof_ts, direction=d, pre_seconds=300
    )
    dist = _dist_to_extreme(stock_df, date=str(date), asof_ts=asof_ts, direction=d)
    if chase is None or from_open is None or dist is None:
        if cfg.on_missing == "block":
            return FoLodChaseDecision(False, 0.0, "missing")
        return FoLodChaseDecision(True, 1.0, "missing_allow")
    fav = float(from_open) if d == "UP" else float(-from_open)
    if fav + 1e-12 < float(cfg.min_fav_from_open):
        return FoLodChaseDecision(
            True, 1.0, "fo_short", fav_from_open=fav, chase=float(chase), dist_ext=float(dist)
        )
    if chase + 1e-12 < float(cfg.min_chase):
        return FoLodChaseDecision(
            True, 1.0, "chase_low", fav_from_open=fav, chase=float(chase), dist_ext=float(dist)
        )
    if dist - 1e-12 > float(cfg.max_dist_ext):
        return FoLodChaseDecision(
            True, 1.0, "off_extreme", fav_from_open=fav, chase=float(chase), dist_ext=float(dist)
        )
    reason = (
        f"block_fo>={cfg.min_fav_from_open:g}&chase>={cfg.min_chase:g}"
        f"&dist_ext<={cfg.max_dist_ext:g}"
    )
    if cfg.mode == "scale":
        return FoLodChaseDecision(
            True,
            float(cfg.scale),
            reason.replace("block_", "degrade_", 1),
            fav_from_open=fav,
            chase=float(chase),
            dist_ext=float(dist),
        )
    return FoLodChaseDecision(
        False, 0.0, reason, fav_from_open=fav, chase=float(chase), dist_ext=float(dist)
    )

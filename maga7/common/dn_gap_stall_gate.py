"""DN overnight-gap mid-extension stall gate (causal).

Targets 02-17 GOOGL: large DN gap (~1.86%) with only partial session
extension (from_open in a mid band) and broad peer align — not a full
continuation (06-22 GOOGL fo≈gap) and not a fade-to-open winner.

Rule (default): DN ∧ fav_gap≥min_gap ∧ fo∈[min_fo, max_fo] ∧ peer≥min_peer → block.
Optional ``min_chase`` measured at the same asof clock.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from maga7.common.overnight_gap_gate import overnight_gap
from maga7.common.range_stall_gate import session_chase_and_pre5


@dataclass(frozen=True)
class DnGapStallGateConfig:
    enabled: bool = False
    min_fav_gap: float = 0.018
    min_fav_from_open: float = 0.008
    max_fav_from_open: float = 0.014
    min_peer: int = 6
    min_chase: float | None = None
    mode: str = "block"  # block | scale
    scale: float = 0.5
    on_missing: str = "allow"  # allow | block


@dataclass(frozen=True)
class DnGapStallDecision:
    allow: bool
    size_scale: float
    reason: str
    fav_gap: float | None = None
    fav_from_open: float | None = None
    chase: float | None = None


def parse_dn_gap_stall_gate(raw: Any) -> DnGapStallGateConfig:
    if not isinstance(raw, dict):
        return DnGapStallGateConfig(enabled=False)
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
    chase = raw.get("min_chase")
    return DnGapStallGateConfig(
        enabled=bool(raw.get("enabled", False)),
        min_fav_gap=float(raw.get("min_fav_gap", 0.018) or 0.018),
        min_fav_from_open=float(raw.get("min_fav_from_open", 0.008) or 0.008),
        max_fav_from_open=float(raw.get("max_fav_from_open", 0.014) or 0.014),
        min_peer=int(raw.get("min_peer", 6) or 6),
        min_chase=float(chase) if chase is not None else None,
        mode=mode,
        scale=max(0.0, min(1.0, float(raw.get("scale", 0.5) or 0.5))),
        on_missing=on_miss,
    )


def resolve_dn_gap_stall_gate(
    cfg: DnGapStallGateConfig,
    *,
    stock_df: pd.DataFrame | None,
    date: str,
    asof_ts: pd.Timestamp,
    direction: str,
    peer_n: int | None = None,
) -> DnGapStallDecision:
    if not cfg.enabled:
        return DnGapStallDecision(True, 1.0, "off")
    d = str(direction or "").upper()
    if d != "DN":
        return DnGapStallDecision(True, 1.0, "dir_skip")
    gap = overnight_gap(stock_df, date=str(date))
    chase, _pre5, from_open = session_chase_and_pre5(
        stock_df, date=str(date), asof_ts=asof_ts, direction=d, pre_seconds=300
    )
    if gap is None or from_open is None:
        if cfg.on_missing == "block":
            return DnGapStallDecision(False, 0.0, "missing")
        return DnGapStallDecision(True, 1.0, "missing_allow")
    fav_gap = float(-gap)  # DN favorable = down gap
    fav_fo = float(-from_open)
    if peer_n is None or int(peer_n) < int(cfg.min_peer):
        return DnGapStallDecision(
            True, 1.0, "peer_low", fav_gap=fav_gap, fav_from_open=fav_fo, chase=chase
        )
    if fav_gap + 1e-12 < float(cfg.min_fav_gap):
        return DnGapStallDecision(
            True, 1.0, "gap_short", fav_gap=fav_gap, fav_from_open=fav_fo, chase=chase
        )
    if fav_fo + 1e-12 < float(cfg.min_fav_from_open):
        return DnGapStallDecision(
            True, 1.0, "fo_low", fav_gap=fav_gap, fav_from_open=fav_fo, chase=chase
        )
    if fav_fo - 1e-12 > float(cfg.max_fav_from_open):
        return DnGapStallDecision(
            True, 1.0, "fo_high", fav_gap=fav_gap, fav_from_open=fav_fo, chase=chase
        )
    if cfg.min_chase is not None:
        if chase is None or chase + 1e-12 < float(cfg.min_chase):
            return DnGapStallDecision(
                True, 1.0, "chase_low", fav_gap=fav_gap, fav_from_open=fav_fo, chase=chase
            )
    reason = (
        f"block_dn_gap>={cfg.min_fav_gap:g}&fo∈[{cfg.min_fav_from_open:g},"
        f"{cfg.max_fav_from_open:g}]&peer>={cfg.min_peer}"
    )
    if cfg.mode == "scale":
        return DnGapStallDecision(
            True,
            float(cfg.scale),
            reason.replace("block_", "degrade_", 1),
            fav_gap=fav_gap,
            fav_from_open=fav_fo,
            chase=chase,
        )
    return DnGapStallDecision(
        False, 0.0, reason, fav_gap=fav_gap, fav_from_open=fav_fo, chase=chase
    )

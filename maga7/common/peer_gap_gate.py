"""Weak-peer + overnight gap stall gate (causal).

Probe (spine after overnight_gap BLOCK): SL/TOX days 04-08 AAPL and 02-18 NVDA
share ``peer_align==3`` (minimum) and fav overnight gap ≈ +2%. Broader gap/ffo
rules kill META 07-01 / NVDA winners; peer-cap keeps the cut surgical.

Rule: ``peer_align <= max_peer`` AND ``fav_gap >= min_fav_gap`` → block/scale.
Optional ``max_fav_from_open``: also require stalled extension from day open.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from maga7.common.overnight_gap_gate import overnight_gap


@dataclass(frozen=True)
class PeerGapGateConfig:
    enabled: bool = False
    min_fav_gap: float = 0.015
    max_peer: int = 3
    mode: str = "block"  # block | scale
    scale: float = 0.5
    dirs: tuple[str, ...] | None = None
    max_fav_from_open: float | None = None
    on_missing_gap: str = "allow"  # allow | block
    on_missing_peer: str = "allow"


@dataclass(frozen=True)
class PeerGapDecision:
    allow: bool
    size_scale: float
    reason: str
    gap: float | None = None
    fav_gap: float | None = None
    peer_n: int | None = None
    fav_from_open: float | None = None


def parse_peer_gap_gate(raw: Any) -> PeerGapGateConfig:
    if not isinstance(raw, dict):
        return PeerGapGateConfig(enabled=False)
    mode = str(raw.get("mode") or "block").strip().lower()
    if mode in {"reject", "hard", "skip"}:
        mode = "block"
    if mode in {"soft", "size", "half", "degrade"}:
        mode = "scale"
    if mode not in {"block", "scale"}:
        mode = "block"
    dirs_raw = raw.get("dirs") or raw.get("directions")
    if raw.get("up_only") in (True, 1, "1", "true", "True", "yes"):
        dirs_raw = ["UP"]
    dirs: tuple[str, ...] | None = None
    if isinstance(dirs_raw, str):
        dirs = tuple(x.strip().upper() for x in dirs_raw.split(",") if x.strip())
    elif isinstance(dirs_raw, (list, tuple)):
        dirs = tuple(str(x).strip().upper() for x in dirs_raw if str(x).strip())
    if dirs == ():
        dirs = None
    mfo = raw.get("max_fav_from_open")
    mfo_f = float(mfo) if mfo is not None else None
    on_miss_g = str(raw.get("on_missing_gap") or "allow").strip().lower()
    if on_miss_g not in {"allow", "block"}:
        on_miss_g = "allow"
    on_miss_p = str(raw.get("on_missing_peer") or "allow").strip().lower()
    if on_miss_p not in {"allow", "block"}:
        on_miss_p = "allow"
    return PeerGapGateConfig(
        enabled=bool(raw.get("enabled", False)),
        min_fav_gap=float(raw.get("min_fav_gap", raw.get("max_fav_gap", 0.015)) or 0.015),
        max_peer=int(raw.get("max_peer", raw.get("peer_max", 3)) or 3),
        mode=mode,
        scale=max(0.0, min(1.0, float(raw.get("scale", 0.5) or 0.5))),
        dirs=dirs,
        max_fav_from_open=mfo_f,
        on_missing_gap=on_miss_g,
        on_missing_peer=on_miss_p,
    )


def resolve_peer_gap_gate(
    cfg: PeerGapGateConfig,
    *,
    stock_df: Any,
    date: str,
    direction: str,
    peer_n: int | None,
    from_open: float | None = None,
) -> PeerGapDecision:
    if not cfg.enabled:
        return PeerGapDecision(True, 1.0, "off")
    d = str(direction or "").upper()
    if cfg.dirs is not None and d not in set(cfg.dirs):
        return PeerGapDecision(True, 1.0, "dir_skip", peer_n=peer_n)
    if peer_n is None:
        if cfg.on_missing_peer == "block":
            return PeerGapDecision(False, 0.0, "missing_peer")
        return PeerGapDecision(True, 1.0, "missing_peer_allow")
    if int(peer_n) > int(cfg.max_peer):
        return PeerGapDecision(True, 1.0, "peer_strong", peer_n=int(peer_n))
    gap = overnight_gap(stock_df, date=str(date))
    if gap is None:
        if cfg.on_missing_gap == "block":
            return PeerGapDecision(False, 0.0, "missing_gap", peer_n=int(peer_n))
        return PeerGapDecision(True, 1.0, "missing_gap_allow", peer_n=int(peer_n))
    fav = float(gap) if d == "UP" else float(-gap)
    if fav + 1e-12 < float(cfg.min_fav_gap):
        return PeerGapDecision(
            True, 1.0, "gap_small", gap=float(gap), fav_gap=fav, peer_n=int(peer_n)
        )
    fav_fo = None
    if from_open is not None:
        try:
            fo = float(from_open)
            fav_fo = fo if d == "UP" else -fo
        except (TypeError, ValueError):
            fav_fo = None
    if cfg.max_fav_from_open is not None:
        if fav_fo is None:
            return PeerGapDecision(
                True,
                1.0,
                "ffo_missing_allow",
                gap=float(gap),
                fav_gap=fav,
                peer_n=int(peer_n),
            )
        if fav_fo - 1e-12 > float(cfg.max_fav_from_open):
            return PeerGapDecision(
                True,
                1.0,
                "ffo_extended",
                gap=float(gap),
                fav_gap=fav,
                peer_n=int(peer_n),
                fav_from_open=fav_fo,
            )
    reason = f"peer<={cfg.max_peer}&gap>={cfg.min_fav_gap:g}"
    if cfg.mode == "scale":
        return PeerGapDecision(
            True,
            float(cfg.scale),
            f"degrade_{reason}",
            gap=float(gap),
            fav_gap=fav,
            peer_n=int(peer_n),
            fav_from_open=fav_fo,
        )
    return PeerGapDecision(
        False,
        0.0,
        f"block_{reason}",
        gap=float(gap),
        fav_gap=fav,
        peer_n=int(peer_n),
        fav_from_open=fav_fo,
    )

"""Minimal entry iceberg sizing for Mag7 OMS.

Splits a target contract qty into clips based on top-of-book ask_size
(or a static notional fallback when size is missing).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class IcebergConfig:
    enabled: bool = True
    # Max share of displayed ask_size used for one clip (0.5 = half the offer).
    ask_size_frac: float = 0.5
    min_chunk_qty: int = 1
    max_chunks: int = 5
    # When ask_size missing / zero: cap each clip by this $ notional.
    fallback_notional: float = 8_000.0
    # Hard stop: do not place further clips after this many seconds.
    max_total_sec: float = 15.0


def iceberg_config_from_trade(trade_cfg: dict[str, Any] | None) -> IcebergConfig:
    trade = trade_cfg or {}
    raw = trade.get("iceberg") or (trade.get("risk") or {}).get("iceberg") or {}
    if not isinstance(raw, dict):
        raw = {}
    return IcebergConfig(
        enabled=bool(raw.get("enabled", True)),
        ask_size_frac=float(raw.get("ask_size_frac", 0.5)),
        min_chunk_qty=max(1, int(raw.get("min_chunk_qty", 1))),
        max_chunks=max(1, int(raw.get("max_chunks", 5))),
        fallback_notional=max(100.0, float(raw.get("fallback_notional", 8_000.0))),
        max_total_sec=max(1.0, float(raw.get("max_total_sec", 15.0))),
    )


def max_chunk_qty(
    *,
    mid: float,
    ask_size: float | None,
    cfg: IcebergConfig,
) -> int:
    """Largest single clip in contracts given top-of-book."""
    mid_f = float(mid or 0.0)
    if mid_f <= 0:
        return max(1, int(cfg.min_chunk_qty))
    by_size = 0
    if ask_size is not None and float(ask_size) > 0:
        by_size = int(max(1.0, float(ask_size) * float(cfg.ask_size_frac)))
    by_notional = int(max(1.0, float(cfg.fallback_notional) // (mid_f * 100.0)))
    if by_size > 0:
        clip = min(by_size, by_notional) if by_notional > 0 else by_size
    else:
        clip = by_notional
    return max(int(cfg.min_chunk_qty), int(clip))


def plan_entry_chunks(
    total_qty: int,
    *,
    mid: float,
    ask_size: float | None = None,
    cfg: IcebergConfig | None = None,
) -> list[int]:
    """Return positive chunk qtys that sum to ``total_qty`` (len <= max_chunks)."""
    cfg = cfg or IcebergConfig()
    qty = max(0, int(total_qty))
    if qty <= 0:
        return []
    if not cfg.enabled:
        return [qty]
    clip = max_chunk_qty(mid=mid, ask_size=ask_size, cfg=cfg)
    if qty <= clip:
        return [qty]
    chunks: list[int] = []
    left = qty
    while left > 0 and len(chunks) < int(cfg.max_chunks):
        # Last allowed chunk takes the remainder (may exceed clip once).
        if len(chunks) == int(cfg.max_chunks) - 1:
            chunks.append(left)
            left = 0
            break
        take = min(clip, left)
        chunks.append(int(take))
        left -= int(take)
    if left > 0:
        chunks[-1] = int(chunks[-1] + left)
    return [c for c in chunks if c > 0]


def encode_chunk_queue(chunks: list[int]) -> str:
    return ",".join(str(int(c)) for c in chunks if int(c) > 0)


def decode_chunk_queue(raw: str | None) -> list[int]:
    if not raw:
        return []
    out: list[int] = []
    for part in str(raw).split(","):
        part = part.strip()
        if not part:
            continue
        try:
            q = int(part)
        except ValueError:
            continue
        if q > 0:
            out.append(q)
    return out

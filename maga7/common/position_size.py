"""Position sizing helpers (shared by offline / stream / OMS)."""
from __future__ import annotations

from typing import Any


def count_others_open(
    open_until: dict[str, Any] | None,
    *,
    symbol: str | None,
    entry_ts: Any,
) -> int:
    if entry_ts is None or not open_until:
        return 0
    n = 0
    for s, until in open_until.items():
        if symbol is not None and s == symbol:
            continue
        if until is not None and until > entry_ts:
            n += 1
    return n


def resolve_size_frac(
    trade: dict[str, Any] | None,
    *,
    top_k: int = 2,
    open_until: dict[str, Any] | None = None,
    symbol: str | None = None,
    entry_ts: Any = None,
) -> tuple[float, str, int, bool, str]:
    """Return ``(size_frac, mode, n_concurrent, allow, reason)``.

    ``position_sizing``:
      - ``topk``: always ``position_frac / top_k`` (legacy)
      - ``concurrent`` / ``live`` (default):
          * alone → full ``position_frac`` (25%)
          * 1 other still open → ``position_frac / max_concurrent`` (12.5%)
          * already ``max_concurrent`` others open → **reject** (no 3rd leg;
            never stack two full 25% into 50%)

    ``max_concurrent_positions`` defaults to ``top_k`` (usually 2).
    """
    trade = trade or {}
    pos = float(trade.get("position_frac", 0.25))
    mode = str(trade.get("position_sizing") or trade.get("sizing_mode") or "concurrent").strip().lower()
    max_conc = int(trade.get("max_concurrent_positions") or top_k or 2)
    max_conc = max(max_conc, 1)

    if mode in ("topk", "top_k", "split_topk"):
        k = max(int(top_k), 1)
        return pos / k, "topk", k, True, "topk"

    n_others = count_others_open(open_until, symbol=symbol, entry_ts=entry_ts)
    n_conc = n_others + 1

    # Already at cap (e.g. 2 open) → do not open another.
    if n_others >= max_conc:
        return 0.0, "concurrent", n_conc, False, "max_concurrent"

    if n_others == 0:
        # Solo: full sleeve. Never foresight a later fill.
        return pos, "concurrent", 1, True, "solo_full"

    # One (or more below cap) already open: split sleeve, never another full 25%.
    # With max_conc=2 this is always pos/2 when n_others==1.
    size = pos / max_conc
    return size, "concurrent", n_conc, True, "split_cap"

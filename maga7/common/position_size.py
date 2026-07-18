"""Position sizing helpers (shared by offline / stream / OMS)."""
from __future__ import annotations

import math
from typing import Any


def coerce_size_scale(raw: Any, default: float = 1.0) -> float:
    """Clamp a multiplicative size scale into ``[0, 1]``."""
    if raw is None:
        return float(default)
    try:
        s = float(raw)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(s):
        return float(default)
    return max(0.0, min(float(s), 1.0))


def apply_size_scale(size_frac: float, scale: Any = 1.0) -> float:
    """Apply regime/watchdog (or other) scale on top of concurrent sleeve."""
    return float(size_frac) * coerce_size_scale(scale, default=1.0)


def regime_scale_from_meta(meta: dict[str, Any] | None) -> float:
    meta = meta or {}
    return coerce_size_scale(meta.get("regime_size_scale", 1.0), default=1.0)


def post_win_cooldown_action(
    trade: dict[str, Any] | None,
    *,
    prev_day_ret: float | None = None,
    cooldown_left: int = 0,
) -> tuple[str, float]:
    """After a large winning session, cool down following session(s).

    ``trade.post_win_cooldown_mode``:
      - ``off`` / missing: no-op → ``("", 1.0)``
      - ``skip``: skip all entries → ``("skip", 0.0)``
      - ``scale``: multiply size by ``post_win_cooldown_scale`` → ``("scale", scale)``

    Active when ``cooldown_left > 0`` (preferred), or when
    ``prev_day_ret >= post_win_cooldown_day_ret`` (default 0.10) for a
    one-shot next-day cool-down.
    """
    trade = trade or {}
    mode = str(trade.get("post_win_cooldown_mode") or "off").strip().lower()
    if mode in {"", "off", "none", "false", "0"}:
        return "", 1.0
    active = int(cooldown_left or 0) > 0
    if not active:
        thr = trade.get("post_win_cooldown_day_ret", 0.10)
        if thr is None or prev_day_ret is None:
            return "", 1.0
        if float(prev_day_ret) < float(thr):
            return "", 1.0
    if mode == "skip":
        return "skip", 0.0
    if mode in {"scale", "half", "reduce"}:
        scale = float(trade.get("post_win_cooldown_scale", 0.5))
        scale = max(0.0, min(scale, 1.0))
        return "scale", scale
    return "", 1.0


def post_win_cooldown_sessions(trade: dict[str, Any] | None) -> int:
    trade = trade or {}
    return max(1, int(trade.get("post_win_cooldown_sessions", 1) or 1))


def block_same_dir_after_win_enabled(trade: dict[str, Any] | None) -> bool:
    """Per-symbol same-direction block the session after a big win.

    Unlike account-level ``post_win_cooldown_*``, this does not change global
    sizing — only suppresses repeating yesterday's winning symbol+direction
    (e.g. TSLA UP TP yesterday → no TSLA UP today).
    """
    trade = trade or {}
    raw = trade.get("block_same_dir_after_win", False)
    if isinstance(raw, str):
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    return bool(raw)


def is_symbol_dir_big_win(
    *,
    ret: float,
    reason: str | None,
    trade: dict[str, Any] | None,
) -> bool:
    trade = trade or {}
    if str(reason or "").upper() == "TP":
        return True
    thr = trade.get("block_same_dir_after_win_ret", 0.50)
    if thr is None:
        return False
    try:
        return float(ret) >= float(thr)
    except (TypeError, ValueError):
        return False


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

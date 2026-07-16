"""Re-entry policy helpers (shared by offline / stream / scanner)."""
from __future__ import annotations

from typing import Any


def resolve_only_win_reenter(trade: dict[str, Any] | None) -> bool:
    """Whether subsequent entries require the previous trade to have won.

    Prefer explicit ``reentry_mode``:
      - cooldown_only | cooldown | always  → False (entry clock = signal+cooldown)
      - only_win | win_only                → True
    Fallback: legacy ``only_reenter_after_win`` bool.
    """
    trade = trade or {}
    mode = str(trade.get("reentry_mode") or "").strip().lower()
    if mode in ("cooldown_only", "cooldown", "always", "signal_cooldown"):
        return False
    if mode in ("only_win", "win_only", "only_reenter_after_win"):
        return True
    return bool(trade.get("only_reenter_after_win", False))

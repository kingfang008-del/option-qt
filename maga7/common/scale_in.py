"""Optional split entry + pullback add-on with secondary factor confirm.

Research knob (default off). First tranche enters at signal fill; if option MTM
pulls back by ``pullback_ret`` and factors still align, add the second tranche
at the then-current buy fill. Both tranches share the same exit path/rails.

Equity impact is encoded in ``SimResult.ret`` so ``size_frac * ret`` stays valid:
  - added:   ret = first_frac*r1 + add_frac*r2
  - no add:  ret = first_frac*r1   (undeployed add_frac earns 0)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ScaleInConfig:
    enabled: bool = False
    first_frac: float = 0.5
    add_frac: float = 0.5
    pullback_ret: float = 0.30
    confirm_mode: str = "mf"  # mf | mf_streak | always | never
    min_hold_seconds: int = 120
    max_wait_seconds: int | None = None


def scale_in_from_trade(trade: dict[str, Any] | None) -> ScaleInConfig:
    raw = (trade or {}).get("scale_in") or {}
    if not isinstance(raw, dict):
        return ScaleInConfig(enabled=False)
    max_wait = raw.get("max_wait_seconds", None)
    return ScaleInConfig(
        enabled=bool(raw.get("enabled", False)),
        first_frac=float(raw.get("first_frac", 0.5) or 0.5),
        add_frac=float(raw.get("add_frac", 0.5) or 0.5),
        pullback_ret=float(raw.get("pullback_ret", 0.30) or 0.30),
        confirm_mode=str(raw.get("confirm_mode", "mf") or "mf").strip().lower(),
        min_hold_seconds=int(raw.get("min_hold_seconds", 120) or 120),
        max_wait_seconds=int(max_wait) if max_wait is not None else None,
    )


def confirm_scale_in(
    *,
    mode: str,
    direction: str,
    mf: float | None,
    streak_up: int = 0,
    streak_dn: int = 0,
) -> bool:
    """Secondary factor check at pullback touch."""
    m = str(mode or "mf").strip().lower()
    d = str(direction or "").upper()
    if m in {"never", "off", "none", "half_only"}:
        return False
    if m in {"always", "any"}:
        return True
    if mf is None or not (mf == mf):  # NaN
        return False
    if d == "UP":
        mf_ok = mf > 0
        streak_ok = int(streak_up) > 0
    elif d == "DN":
        mf_ok = mf < 0
        streak_ok = int(streak_dn) > 0
    else:
        return False
    if m in {"mf"}:
        return bool(mf_ok)
    if m in {"mf_streak", "both", "streak_mf"}:
        return bool(mf_ok and streak_ok)
    if m in {"streak"}:
        return bool(streak_ok)
    return bool(mf_ok)


def blend_scale_in_ret(
    *,
    entry1: float,
    entry2: float | None,
    exit_px: float,
    first_frac: float,
    add_frac: float,
) -> tuple[float, float, bool]:
    """Return ``(blended_ret, deployed_frac, added)``."""
    f1 = max(0.0, float(first_frac))
    f2 = max(0.0, float(add_frac))
    r1 = float(exit_px) / float(entry1) - 1.0
    if entry2 is not None and float(entry2) > 0 and f2 > 0:
        r2 = float(exit_px) / float(entry2) - 1.0
        return f1 * r1 + f2 * r2, f1 + f2, True
    return f1 * r1, f1, False

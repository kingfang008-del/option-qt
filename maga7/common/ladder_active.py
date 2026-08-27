"""Second-level active ladder exits for 0DTE/1DTE research (not freeze).

Never "set and forget": hard max hold, stepped TP/SL, profit-stall, optional mf flip.
Partial scale-out is deferred — V0 full-flatten rails only.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class TpRail:
    ret: float
    action: str = "exit"  # exit | trail
    trail_dd: float = 0.05


@dataclass(frozen=True)
class SlRail:
    ret: float  # negative MTM threshold, e.g. -0.12


@dataclass
class LadderActiveConfig:
    enabled: bool = False
    max_hold_seconds: int = 300
    # Sorted ascending by ret (more negative first for SL).
    sl_rails: tuple[SlRail, ...] = (SlRail(-0.12), SlRail(-0.20))
    # Sorted ascending by ret (looser TP first).
    tp_rails: tuple[TpRail, ...] = (
        TpRail(0.12, "trail", 0.04),
        TpRail(0.20, "exit", 0.0),
    )
    stall_min_peak: float = 0.08
    stall_seconds: int = 20
    mf_flip: bool = True
    mf_grace_seconds: int = 15
    # If True, classic tp_mult/sl_mult still apply as outer rails.
    keep_outer_rails: bool = True
    # always | mixed_wash_up | prevention — day-level gate (replay / research).
    when: str = "always"


def ladder_day_should_arm(
    cfg: LadderActiveConfig,
    *,
    date: str,
    stock_by: dict[str, Any],
    qqq_df: Any,
    symbols: list[str],
    asof: str = "10:30",
    washout_breadth_min: int = 3,
    wash_drop_min: float = 0.008,
    frac_above_min: float = 0.35,
    frac_above_max: float = 0.70,
) -> bool:
    """Whether ladder exits arm for this session date."""
    if not cfg.enabled:
        return False
    when = str(cfg.when or "always").strip().lower()
    if when in {"", "always", "on", "all"}:
        return True
    if when in {"mixed_wash_up", "prevention", "up_toxic", "toxic_up"}:
        from maga7.common.predictive_prevention import evaluate_prevention_rule

        hit = evaluate_prevention_rule(
            date=str(date),
            stock_by=stock_by,
            qqq_df=qqq_df,
            symbols=list(symbols),
            asof=str(asof),
            rule="mixed_wash_up",
            prefer_risk_off=True,
            washout_breadth_min=int(washout_breadth_min),
            wash_drop_min=float(wash_drop_min),
            frac_above_min=float(frac_above_min),
            frac_above_max=float(frac_above_max),
        )
        return hit is not None
    return True


def ladder_active_from_trade(trade: dict[str, Any] | None) -> LadderActiveConfig:
    trade = trade or {}
    raw = trade.get("ladder_active")
    mode = str(trade.get("exit_mode") or "").lower()
    mode_hit = any(
        x in mode.replace(",", "+").replace("|", "+")
        for x in ("ladder_active", "sec_active", "hft_ladder", "active_ladder")
    )
    if raw is None and not mode_hit:
        return LadderActiveConfig(enabled=False)
    if isinstance(raw, bool):
        return LadderActiveConfig(enabled=bool(raw) or mode_hit)
    if not isinstance(raw, dict):
        raw = {}
    enabled = bool(raw.get("enabled", True if mode_hit else False))

    sl_rails: list[SlRail] = []
    for item in raw.get("sl_rails") or ():
        if isinstance(item, dict):
            sl_rails.append(SlRail(ret=float(item.get("ret", -0.12))))
        elif isinstance(item, (int, float)):
            sl_rails.append(SlRail(ret=float(item)))
    if not sl_rails:
        sl_rails = [SlRail(-0.12), SlRail(-0.20)]
    sl_rails_t = tuple(sorted(sl_rails, key=lambda r: r.ret))  # -0.20 then -0.12

    tp_rails: list[TpRail] = []
    for item in raw.get("tp_rails") or ():
        if isinstance(item, dict):
            act = str(item.get("action") or "exit").strip().lower()
            if act not in {"exit", "trail", "lock"}:
                act = "exit"
            if act == "lock":
                act = "trail"
            tp_rails.append(
                TpRail(
                    ret=float(item.get("ret", 0.15)),
                    action=act,
                    trail_dd=float(item.get("trail_dd", 0.05) or 0.05),
                )
            )
        elif isinstance(item, (int, float)):
            tp_rails.append(TpRail(ret=float(item), action="exit"))
    if not tp_rails:
        tp_rails = [TpRail(0.12, "trail", 0.04), TpRail(0.20, "exit")]
    tp_rails_t = tuple(sorted(tp_rails, key=lambda r: r.ret))

    stall = raw.get("profit_stall") if isinstance(raw.get("profit_stall"), dict) else {}
    when = str(raw.get("when") or "always").strip().lower() or "always"
    return LadderActiveConfig(
        enabled=enabled,
        max_hold_seconds=int(raw.get("max_hold_seconds", 300) or 300),
        sl_rails=sl_rails_t,
        tp_rails=tp_rails_t,
        stall_min_peak=float(
            stall.get("min_peak", raw.get("stall_min_peak", 0.08)) or 0.08
        ),
        stall_seconds=int(stall.get("stall_seconds", raw.get("stall_seconds", 20)) or 20),
        mf_flip=bool(raw.get("mf_flip", True)),
        mf_grace_seconds=int(raw.get("mf_grace_seconds", 15) or 15),
        keep_outer_rails=bool(raw.get("keep_outer_rails", True)),
        when=when,
    )

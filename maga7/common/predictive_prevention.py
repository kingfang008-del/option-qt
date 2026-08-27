"""Predictive morning prevention — day-type risk_off / expert before entries.

This is **not** a consecutive-loss circuit. It scores causal morning features
(at Watchdog ``asof``, default 10:30) and maps to expert overlays via
``RegimeWatchdog`` prevention lane (``mixed_wash_up`` → ``up_toxic`` /
``up_toxic_block``).

See ``docs/predictive_prevention.md``.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd

from maga7.common.watchdog import RegimeWatchdog, WatchdogState, eval_router_rule


@dataclass(frozen=True)
class PreventionDecision:
    """Snapshot for logs / Dash / session ``prevention.json``."""

    enabled: bool
    date: str
    asof: str
    state: str
    expert: str | None
    reason: str
    route_tag: str
    prefer_risk_off: bool
    rule: str | None
    note: str = "predictive; not a post-loss circuit"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def prevention_from_watchdog(
    watchdog: RegimeWatchdog | None,
    *,
    date: str,
) -> PreventionDecision:
    """Read the active Watchdog day decision as a prevention snapshot."""
    if watchdog is None or not getattr(watchdog, "cfg", None) or not watchdog.cfg.enabled:
        return PreventionDecision(
            enabled=False,
            date=str(date),
            asof="10:30",
            state="off",
            expert=None,
            reason="watchdog_off",
            route_tag="baseline",
            prefer_risk_off=False,
            rule=None,
        )
    cfg = watchdog.cfg
    d = watchdog.decision_at()
    prefer = bool((cfg.prevention_router_cfg or {}).get("prefer_risk_off", False))
    return PreventionDecision(
        enabled=bool(cfg.prevention_enabled),
        date=str(date),
        asof=str(cfg.asof),
        state=str(d.state.value if isinstance(d.state, WatchdogState) else d.state),
        expert=d.expert,
        reason=str(d.reason or ""),
        route_tag=str((d.overlay.route_tag if d.overlay else None) or "baseline"),
        prefer_risk_off=prefer,
        rule=str(cfg.prevention_rule) if cfg.prevention_enabled else None,
    )


def evaluate_prevention_rule(
    *,
    date: str,
    stock_by: dict[str, pd.DataFrame],
    qqq_df: pd.DataFrame | None,
    symbols: list[str],
    asof: str = "10:30",
    rule: str = "mixed_wash_up",
    prefer_risk_off: bool = True,
    washout_breadth_min: int = 3,
    wash_drop_min: float = 0.008,
    frac_above_min: float = 0.35,
    frac_above_max: float = 0.70,
) -> str | None:
    """Standalone causal rule eval (no Watchdog object required)."""
    return eval_router_rule(
        rule,
        date=str(date),
        stock_by=stock_by,
        qqq_df=qqq_df,
        symbols=list(symbols),
        asof_hhmm=str(asof),
        router_cfg={
            "prefer_risk_off": bool(prefer_risk_off),
            "washout_breadth_min": int(washout_breadth_min),
            "wash_drop_min": float(wash_drop_min),
            "frac_above_min": float(frac_above_min),
            "frac_above_max": float(frac_above_max),
            "risk_off_expert": "up_toxic_block",
            "soft_expert": "up_toxic",
        },
    )

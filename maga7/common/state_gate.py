"""Day-state gate: veto / scale entries — never emit Call/Put.

Classifies the morning into ``trend`` | ``mixed_wash`` | ``reclaim_trap`` |
``unknown`` using causal Watchdog-style rules, then blocks or scales directions.

This is the research replacement for "tune flow thresholds": flow still produces
candidates; state_gate only answers whether the current regime historically
deserves risk.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd

from maga7.common.predictive_prevention import evaluate_prevention_rule


@dataclass(frozen=True)
class StateGateConfig:
    enabled: bool = False
    asof: str = "10:30"
    # mixed_wash_up → typically toxic for naked UP 0DTE
    mixed_wash_breadth_min: int = 5
    mixed_wash_drop_min: float = 0.008
    mixed_wash_frac_above_min: float = 0.35
    mixed_wash_frac_above_max: float = 0.70
    mixed_wash_action: str = "block_up"  # block_up | scale_up | pass
    mixed_wash_scale: float = 0.25
    # reclaim after wash → trap risk; scale rather than hard block by default
    reclaim_rule: str = "washout_and_reclaim"
    reclaim_action: str = "scale"  # block | scale | pass
    reclaim_scale: float = 0.5
    reclaim_wash_drop_min: float = 0.015
    reclaim_breadth_min: int = 5

    @classmethod
    def from_profile(cls, profile: dict[str, Any] | None) -> "StateGateConfig":
        raw = (profile or {}).get("state_gate")
        if not isinstance(raw, dict):
            return cls(enabled=False)
        mw = raw.get("mixed_wash") if isinstance(raw.get("mixed_wash"), dict) else {}
        rc = raw.get("reclaim_trap") if isinstance(raw.get("reclaim_trap"), dict) else {}
        return cls(
            enabled=bool(raw.get("enabled", False)),
            asof=str(raw.get("asof") or "10:30"),
            mixed_wash_breadth_min=int(mw.get("washout_breadth_min", raw.get("washout_breadth_min", 5)) or 5),
            mixed_wash_drop_min=float(mw.get("wash_drop_min", raw.get("wash_drop_min", 0.008)) or 0.008),
            mixed_wash_frac_above_min=float(mw.get("frac_above_min", 0.35) or 0.35),
            mixed_wash_frac_above_max=float(mw.get("frac_above_max", 0.70) or 0.70),
            mixed_wash_action=str(mw.get("action", "block_up") or "block_up").strip().lower(),
            mixed_wash_scale=float(mw.get("scale", 0.25) or 0.25),
            reclaim_rule=str(rc.get("rule", "washout_and_reclaim") or "washout_and_reclaim"),
            reclaim_action=str(rc.get("action", "scale") or "scale").strip().lower(),
            reclaim_scale=float(rc.get("scale", 0.5) or 0.5),
            reclaim_wash_drop_min=float(rc.get("wash_drop_min", 0.015) or 0.015),
            reclaim_breadth_min=int(rc.get("washout_breadth_min", 5) or 5),
        )


@dataclass
class StateGateDayDecision:
    enabled: bool
    date: str
    asof: str
    state: str  # trend | mixed_wash | reclaim_trap | unknown
    reason: str
    block_directions: list[str] = field(default_factory=list)
    direction_size_scale: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StateGateEntryDecision:
    allow: bool
    size_scale: float
    state: str
    reason: str


class StateGate:
    """Causal morning state classifier + entry veto."""

    def __init__(self, cfg: StateGateConfig):
        self.cfg = cfg
        self._day: StateGateDayDecision | None = None

    @classmethod
    def from_profile(cls, profile: dict[str, Any] | None) -> "StateGate":
        return cls(StateGateConfig.from_profile(profile))

    def begin_day(
        self,
        date: str,
        *,
        stock_by: dict[str, pd.DataFrame],
        qqq_df: pd.DataFrame | None,
        symbols: list[str],
    ) -> StateGateDayDecision:
        if not self.cfg.enabled:
            dec = StateGateDayDecision(
                enabled=False,
                date=str(date),
                asof=self.cfg.asof,
                state="off",
                reason="disabled",
            )
            self._day = dec
            return dec

        # 1) mixed wash (stricter breadth than failed prevention default=3)
        mw_hit = evaluate_prevention_rule(
            date=str(date),
            stock_by=stock_by,
            qqq_df=qqq_df,
            symbols=list(symbols),
            asof=str(self.cfg.asof),
            rule="mixed_wash_up",
            prefer_risk_off=True,
            washout_breadth_min=int(self.cfg.mixed_wash_breadth_min),
            wash_drop_min=float(self.cfg.mixed_wash_drop_min),
            frac_above_min=float(self.cfg.mixed_wash_frac_above_min),
            frac_above_max=float(self.cfg.mixed_wash_frac_above_max),
        )
        block: list[str] = []
        scales: dict[str, float] = {}
        if mw_hit is not None:
            action = self.cfg.mixed_wash_action
            if action in {"block_up", "block", "hard"}:
                block.append("UP")
            elif action in {"scale_up", "scale"}:
                scales["UP"] = float(self.cfg.mixed_wash_scale)
            dec = StateGateDayDecision(
                enabled=True,
                date=str(date),
                asof=self.cfg.asof,
                state="mixed_wash",
                reason=str(mw_hit),
                block_directions=block,
                direction_size_scale=scales,
            )
            self._day = dec
            return dec

        # 2) reclaim trap (wash then bounce) — scale by default
        rc_hit = evaluate_prevention_rule(
            date=str(date),
            stock_by=stock_by,
            qqq_df=qqq_df,
            symbols=list(symbols),
            asof=str(self.cfg.asof),
            rule=str(self.cfg.reclaim_rule),
            prefer_risk_off=True,
            washout_breadth_min=int(self.cfg.reclaim_breadth_min),
            wash_drop_min=float(self.cfg.reclaim_wash_drop_min),
            frac_above_min=0.0,
            frac_above_max=1.0,
        )
        if rc_hit is not None:
            action = self.cfg.reclaim_action
            if action in {"block", "hard"}:
                block = ["UP", "DN"]
            elif action in {"scale", "scale_all"}:
                scales = {"UP": float(self.cfg.reclaim_scale), "DN": float(self.cfg.reclaim_scale)}
            elif action in {"scale_up"}:
                scales = {"UP": float(self.cfg.reclaim_scale)}
            dec = StateGateDayDecision(
                enabled=True,
                date=str(date),
                asof=self.cfg.asof,
                state="reclaim_trap",
                reason=str(rc_hit),
                block_directions=block,
                direction_size_scale=scales,
            )
            self._day = dec
            return dec

        dec = StateGateDayDecision(
            enabled=True,
            date=str(date),
            asof=self.cfg.asof,
            state="trend",
            reason="no_toxic_morning_flags",
        )
        self._day = dec
        return dec

    def decide_entry(self, direction: str) -> StateGateEntryDecision:
        day = self._day
        if day is None or not day.enabled:
            return StateGateEntryDecision(True, 1.0, "off", "disabled")
        dir_u = str(direction or "").upper()
        if dir_u in {d.upper() for d in day.block_directions}:
            return StateGateEntryDecision(False, 0.0, day.state, f"block_{dir_u}_{day.state}")
        scale = float(day.direction_size_scale.get(dir_u, 1.0))
        if scale <= 0:
            return StateGateEntryDecision(False, 0.0, day.state, f"scale0_{dir_u}_{day.state}")
        return StateGateEntryDecision(True, scale, day.state, day.reason)


def load_state_gate(profile: dict[str, Any] | None) -> StateGate:
    return StateGate.from_profile(profile)

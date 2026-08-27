"""Causal CORE session risk budget (C6) — realized equity DD, not climate.

C2 failed because 10:30 VIXY/breadth look the same in weak and strong windows.
This layer answers remaining risk from **path accounting**:

    current_dd = equity / peak - 1
    if current_dd <= trigger: size *= scale

Same rule in every climate. It fires more when the book is already in a hole
(weak window) and almost never at strong-window highs.

Does **not** emit direction, does **not** block Rule-A, does **not** add morph
BLOCKs, does **not** use calendar labels as a live gate.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SessionRiskBudgetConfig:
    enabled: bool = False
    mode: str = "dd_step"  # dd_step | dd_linear
    dd_trigger: float = -0.05
    scale: float = 0.5
    dd_span: float = 0.10  # linear: size=0 at current_dd = -dd_span
    min_scale: float = 0.0


def parse_session_risk_budget(raw: Any) -> SessionRiskBudgetConfig:
    if not isinstance(raw, dict):
        return SessionRiskBudgetConfig(enabled=False)
    trig = raw.get("dd_trigger", -0.05)
    span = raw.get("dd_span", raw.get("dd_budget", 0.10))
    return SessionRiskBudgetConfig(
        enabled=bool(raw.get("enabled", False)),
        mode=str(raw.get("mode") or "dd_step").strip().lower(),
        dd_trigger=float(trig if trig is not None else -0.05),
        scale=max(0.0, min(1.0, float(raw.get("scale", 0.5) or 0.5))),
        dd_span=max(1e-6, float(span if span is not None else 0.10)),
        min_scale=max(0.0, min(1.0, float(raw.get("min_scale", 0.0) or 0.0))),
    )


def current_drawdown(equity: float, peak: float) -> float:
    eq = float(equity)
    pk = float(peak)
    if pk <= 0 or eq <= 0:
        return 0.0
    return eq / pk - 1.0


def resolve_session_risk_budget(
    cfg: SessionRiskBudgetConfig,
    *,
    current_dd: float | None,
) -> tuple[float, str]:
    """Return ``(size_mult, reason)`` from realized drawdown at decision time."""
    if not cfg.enabled:
        return 1.0, "budget_off"
    if current_dd is None:
        return 1.0, "budget_missing_passthrough"
    dd = float(current_dd)
    if not (dd == dd):  # NaN
        return 1.0, "budget_missing_passthrough"
    mode = cfg.mode
    if mode in {"dd_linear", "linear"}:
        # 0 DD → 1.0; -dd_span → 0 (then clamp to min_scale).
        raw = 1.0 + dd / float(cfg.dd_span)
        sc = max(float(cfg.min_scale), min(1.0, float(raw)))
        if sc < 1.0 - 1e-12:
            return sc, f"budget_linear:{dd:.3f}:{sc:.2f}"
        return 1.0, "budget_ok"
    # dd_step (default): full size until trigger, then constant scale.
    if dd <= float(cfg.dd_trigger) + 1e-15:
        sc = float(cfg.scale)
        if sc < 1.0 - 1e-12:
            return sc, f"budget_dd:{dd:.3f}:{sc:.2f}"
    return 1.0, "budget_ok"

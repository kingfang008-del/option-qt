"""Δ-aware time-stop: flatten when underlying fails to confirm.

If after ``check_seconds`` the signed stock move from fill is still below
``min_stock_move`` and option MTM is not positive, exit with ``DELTA_STOP``.
Refuses to pay further Θ/Vega when Δ is not working.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class DeltaTimeStopConfig:
    enabled: bool = False
    check_seconds: int = 300
    max_seconds: int | None = 900
    min_stock_move: float = 0.0015
    opt_mtm_max: float = 0.0


def delta_time_stop_from_trade(trade: dict[str, Any] | None) -> DeltaTimeStopConfig:
    trade = trade or {}
    raw = trade.get("delta_time_stop")
    if raw is None:
        return DeltaTimeStopConfig(enabled=False)
    if isinstance(raw, bool):
        return DeltaTimeStopConfig(enabled=bool(raw))
    if not isinstance(raw, dict):
        return DeltaTimeStopConfig(enabled=False)
    max_raw = raw.get("max_seconds", 900)
    max_sec = None if max_raw is None else int(max_raw)
    return DeltaTimeStopConfig(
        enabled=bool(raw.get("enabled", False)),
        check_seconds=int(raw.get("check_seconds", 300) or 300),
        max_seconds=max_sec,
        min_stock_move=float(raw.get("min_stock_move", 0.0015) or 0.0015),
        opt_mtm_max=float(raw.get("opt_mtm_max", 0.0) if raw.get("opt_mtm_max") is not None else 0.0),
    )


def morning_r5_scale_from_trade(trade: dict[str, Any] | None) -> dict[str, Any] | None:
    """Parse ``trade.morning_r5_scale``; None if disabled."""
    trade = trade or {}
    raw = trade.get("morning_r5_scale")
    if not isinstance(raw, dict) or not bool(raw.get("enabled", False)):
        return None
    return dict(raw)


@dataclass
class AdverseSoftConfig:
    """Causal 5m deep-adverse soft gate (default OFF).

    After ``check_seconds``, if running signed stock MAE <= ``-adverse_mae``
    and option MTM <= ``opt_mtm_max``:

    - ``tox_tighten``: tighten trade_toxic cut/mfe (no hard flatten)
    - ``soft_exit``: flatten with ``ADVERSE_SOFT``

    Narrowing knobs (all default loose / off except opt red):
    - ``max_opt_mfe``: require quote peak MTM < this (None = off)
    - ``require_still_adverse``: current signed stock still <= ``still_adverse_max``
      (filters wash-then-recover winners that only printed a deep MAE)
    """

    enabled: bool = False
    mode: str = "tox_tighten"  # tox_tighten | soft_exit
    check_seconds: int = 300
    adverse_mae: float = 0.0015
    opt_mtm_max: float = 0.0
    tight_cut_ret: float = 0.15
    tight_mfe_bypass: float = 0.03
    extend_max_cut: bool = True
    max_opt_mfe: float | None = None
    require_still_adverse: bool = False
    still_adverse_max: float = -0.0010


def adverse_soft_from_trade(trade: dict[str, Any] | None) -> AdverseSoftConfig:
    trade = trade or {}
    raw = trade.get("adverse_soft")
    if raw is None:
        return AdverseSoftConfig(enabled=False)
    if isinstance(raw, bool):
        return AdverseSoftConfig(enabled=bool(raw))
    if not isinstance(raw, dict):
        return AdverseSoftConfig(enabled=False)
    mode = str(raw.get("mode", "tox_tighten") or "tox_tighten").strip().lower()
    if mode not in {"tox_tighten", "soft_exit", "tighten", "exit"}:
        mode = "tox_tighten"
    if mode == "tighten":
        mode = "tox_tighten"
    if mode == "exit":
        mode = "soft_exit"
    mfe_raw = raw.get("max_opt_mfe", None)
    max_opt_mfe = None if mfe_raw in (None, "", False) else float(mfe_raw)
    return AdverseSoftConfig(
        enabled=bool(raw.get("enabled", False)),
        mode=mode,
        check_seconds=int(raw.get("check_seconds", 300) or 300),
        adverse_mae=float(raw.get("adverse_mae", 0.0015) or 0.0015),
        opt_mtm_max=float(raw.get("opt_mtm_max", 0.0) if raw.get("opt_mtm_max") is not None else 0.0),
        tight_cut_ret=float(raw.get("tight_cut_ret", 0.15) or 0.15),
        tight_mfe_bypass=float(raw.get("tight_mfe_bypass", 0.03) or 0.03),
        extend_max_cut=bool(raw.get("extend_max_cut", True)),
        max_opt_mfe=max_opt_mfe,
        require_still_adverse=bool(raw.get("require_still_adverse", False)),
        still_adverse_max=float(
            raw.get("still_adverse_max", -0.0010) if raw.get("still_adverse_max") is not None else -0.0010
        ),
    )


@dataclass
class StockRevExitConfig:
    """Flatten when underlying has reversed against the trade.

    After ``min_hold_minutes``, if signed stock return from fill is
    ``<= stock_max`` and option MTM ``<= opt_mtm_max``, exit ``STOCK_REV``.
    Path evidence for "why still long" — not a fixed clock.

    ``when``: day-level arm gate — ``always`` | ``mixed_wash_up`` (same detector
    as conditional ladder / prevention). Off days keep peer3 clock/TP rails.

    ``routes``: optional allow-list (e.g. ``("hunt",)``). ``None`` = all routes.
    """

    enabled: bool = False
    min_hold_minutes: float = 15.0
    stock_max: float = 0.0
    opt_mtm_max: float = 0.10
    when: str = "always"
    routes: tuple[str, ...] | None = None
    # Optional overrides for mixed_wash_up day gate (None → caller/profile default).
    washout_breadth_min: int | None = None
    wash_drop_min: float | None = None
    frac_above_min: float | None = None
    frac_above_max: float | None = None


def _parse_stock_rev_routes(raw: Any) -> tuple[str, ...] | None:
    if raw in (None, "", False):
        return None
    if isinstance(raw, str):
        parts = tuple(x.strip().lower() for x in raw.split(",") if x.strip())
        return parts or None
    if isinstance(raw, (list, tuple)):
        parts = tuple(str(x).strip().lower() for x in raw if str(x).strip())
        return parts or None
    return None


def stock_rev_applies_to_route(
    cfg: StockRevExitConfig | None, route: str | None
) -> bool:
    """Whether STOCK_REV is armed for this trade route (baseline/hunt/…)."""
    if cfg is None or not bool(getattr(cfg, "enabled", False)):
        return False
    routes = getattr(cfg, "routes", None)
    if routes is None:
        return True
    r = str(route or "baseline").strip().lower() or "baseline"
    return r in {str(x).strip().lower() for x in routes}


def stock_rev_exit_from_trade(trade: dict[str, Any] | None) -> StockRevExitConfig:
    trade = trade or {}
    raw = trade.get("stock_rev_exit")
    if raw is None:
        return StockRevExitConfig(enabled=False)
    if isinstance(raw, bool):
        return StockRevExitConfig(enabled=bool(raw))
    if not isinstance(raw, dict):
        return StockRevExitConfig(enabled=False)
    when = str(raw.get("when", "always") or "always").strip().lower()
    if when in {"", "on", "all"}:
        when = "always"
    routes = _parse_stock_rev_routes(raw.get("routes"))
    if raw.get("hunt_only") in (True, 1, "1", "true", "True", "yes"):
        routes = ("hunt",)

    def _opt_int(key: str) -> int | None:
        v = raw.get(key)
        return None if v in (None, "", False) else int(v)

    def _opt_float(key: str) -> float | None:
        v = raw.get(key)
        return None if v in (None, "", False) else float(v)

    return StockRevExitConfig(
        enabled=bool(raw.get("enabled", False)),
        min_hold_minutes=float(raw.get("min_hold_minutes", 15.0) or 15.0),
        stock_max=float(raw.get("stock_max", 0.0) if raw.get("stock_max") is not None else 0.0),
        opt_mtm_max=float(
            raw.get("opt_mtm_max", 0.10) if raw.get("opt_mtm_max") is not None else 0.10
        ),
        when=when,
        routes=routes,
        washout_breadth_min=_opt_int("washout_breadth_min"),
        wash_drop_min=_opt_float("wash_drop_min"),
        frac_above_min=_opt_float("frac_above_min"),
        frac_above_max=_opt_float("frac_above_max"),
    )


def stock_rev_day_should_arm(
    cfg: StockRevExitConfig,
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
    """Whether STOCK_REV arms for this session date."""
    if not cfg.enabled:
        return False
    when = str(cfg.when or "always").strip().lower()
    if when in {"", "always", "on", "all"}:
        return True
    if when in {"mixed_wash_up", "prevention", "up_toxic", "toxic_up"}:
        from maga7.common.predictive_prevention import evaluate_prevention_rule

        bmin = (
            int(cfg.washout_breadth_min)
            if cfg.washout_breadth_min is not None
            else int(washout_breadth_min)
        )
        wdrop = (
            float(cfg.wash_drop_min)
            if cfg.wash_drop_min is not None
            else float(wash_drop_min)
        )
        fmin = (
            float(cfg.frac_above_min)
            if cfg.frac_above_min is not None
            else float(frac_above_min)
        )
        fmax = (
            float(cfg.frac_above_max)
            if cfg.frac_above_max is not None
            else float(frac_above_max)
        )
        hit = evaluate_prevention_rule(
            date=str(date),
            stock_by=stock_by,
            qqq_df=qqq_df,
            symbols=list(symbols),
            asof=str(asof),
            rule="mixed_wash_up",
            prefer_risk_off=True,
            washout_breadth_min=bmin,
            wash_drop_min=wdrop,
            frac_above_min=fmin,
            frac_above_max=fmax,
        )
        return hit is not None
    return True


@dataclass
class RoiTimeStopConfig:
    """V0-style option-ROI progress gates (not stock displacement).

    At each rail ``mins``, if option MTM ``cur_ret < min_roi``, exit
    ``ROI_TIME{mins}``. Default OFF.
    """

    enabled: bool = False
    # (minutes_from_fill, min_roi) checked in ascending minutes order.
    rails: tuple[tuple[float, float], ...] = ((15.0, 0.05), (30.0, 0.05))


def roi_time_stop_from_trade(trade: dict[str, Any] | None) -> RoiTimeStopConfig:
    trade = trade or {}
    raw = trade.get("roi_time_stop")
    if raw is None:
        return RoiTimeStopConfig(enabled=False)
    if isinstance(raw, bool):
        return RoiTimeStopConfig(enabled=bool(raw))
    if not isinstance(raw, dict):
        return RoiTimeStopConfig(enabled=False)
    rails_raw = raw.get("rails")
    rails: list[tuple[float, float]] = []
    if isinstance(rails_raw, (list, tuple)) and rails_raw:
        for item in rails_raw:
            if isinstance(item, dict):
                rails.append((float(item.get("mins", 15)), float(item.get("min_roi", 0.05))))
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                rails.append((float(item[0]), float(item[1])))
    else:
        # V0-shaped mid + late shortcuts
        mid_m = raw.get("mid_mins", 15)
        mid_r = raw.get("mid_min_roi", 0.05)
        late_m = raw.get("late_mins")
        late_r = raw.get("late_min_roi", 0.05)
        if mid_m is not None:
            rails.append((float(mid_m), float(mid_r if mid_r is not None else 0.0)))
        if late_m is not None:
            rails.append((float(late_m), float(late_r if late_r is not None else 0.0)))
    if not rails:
        rails = [(15.0, 0.05), (30.0, 0.05)]
    rails_sorted = tuple(sorted({(float(m), float(r)) for m, r in rails}, key=lambda x: x[0]))
    return RoiTimeStopConfig(
        enabled=bool(raw.get("enabled", False)),
        rails=rails_sorted,
    )

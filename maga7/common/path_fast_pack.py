"""Conditional fast exit pack — tighten path rails without global scalp.

Day gates (``when``):
  - ``always`` / ``mixed_wash_up`` — stock breadth wash (existing)
  - ``qqq_opt_chop`` — QQQ locked option-surface chop @ asof (Greeks / micro)
  - ``wash_or_opt_chop`` / ``wash_and_opt_chop`` — combine wash + surface

Research only — not freeze.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PathFastPackConfig:
    enabled: bool = False
    when: str = "mixed_wash_up"
    hold_minutes: int = 20
    trail_activate: float = 0.15
    trail_dd: float = 0.08
    stock_rev_min_hold_minutes: float = 5.0
    stock_rev_stock_max: float = 0.0
    stock_rev_opt_mtm_max: float = 0.05
    washout_breadth_min: int | None = 3
    wash_drop_min: float | None = None
    frac_above_min: float | None = None
    frac_above_max: float | None = None
    asof: str | None = None
    # QQQ option-surface chop gate (bucketed_v7)
    opt_symbol: str = "QQQ"
    opt_lookback_days: int = 40
    opt_imbalance_max: float = -0.05
    opt_chop_pctile_min: float = 0.70
    # imb_or_chop (legacy) | imb_only | imb_and_chop
    # imb_only cuts +imb FAs but misses Jul20-style wash days with +imb.
    opt_gate: str = "imb_only"
    opt_bucketed_root: str | None = None
    # Post-wash refine (arm diagnosis 2026-07-21): cut false-alarm wash days,
    # keep Jul20. Applied only after mixed_wash_up hits.
    wash_refine: bool = False
    wash_refine_chop_max: float = 1.85
    wash_refine_med_stock_ret_max: float = 0.003
    wash_refine_pcr_max: float = 2.0
    wash_refine_iv_mom_max: float = 0.03
    wash_refine_n_down_max: int = 4
    wash_refine_am_start: str = "09:35"


_SURFACE_CACHE: dict[str, pd.DataFrame] = {}
_ASOF_CACHE: dict[tuple[str, str, str], dict[str, float] | None] = {}


def path_fast_pack_from_trade(trade: dict[str, Any] | None) -> PathFastPackConfig:
    trade = trade or {}
    raw = trade.get("path_fast_pack")
    if raw is None:
        return PathFastPackConfig(enabled=False)
    if isinstance(raw, bool):
        return PathFastPackConfig(enabled=bool(raw))
    if not isinstance(raw, dict):
        return PathFastPackConfig(enabled=False)
    when = str(raw.get("when", "mixed_wash_up") or "mixed_wash_up").strip().lower()
    if when in {"", "on", "all"}:
        when = "always"

    def _opt_int(key: str) -> int | None:
        v = raw.get(key)
        return None if v in (None, "", False) else int(v)

    def _opt_float(key: str) -> float | None:
        v = raw.get(key)
        return None if v in (None, "", False) else float(v)

    return PathFastPackConfig(
        enabled=bool(raw.get("enabled", False)),
        when=when,
        hold_minutes=int(raw.get("hold_minutes", 20) or 20),
        trail_activate=float(
            raw.get("trail_activate", 0.15) if raw.get("trail_activate") is not None else 0.15
        ),
        trail_dd=float(raw.get("trail_dd", 0.08) if raw.get("trail_dd") is not None else 0.08),
        stock_rev_min_hold_minutes=float(
            raw.get("stock_rev_min_hold_minutes", 5.0) or 5.0
        ),
        stock_rev_stock_max=float(
            raw.get("stock_rev_stock_max", 0.0)
            if raw.get("stock_rev_stock_max") is not None
            else 0.0
        ),
        stock_rev_opt_mtm_max=float(
            raw.get("stock_rev_opt_mtm_max", 0.05)
            if raw.get("stock_rev_opt_mtm_max") is not None
            else 0.05
        ),
        washout_breadth_min=_opt_int("washout_breadth_min"),
        wash_drop_min=_opt_float("wash_drop_min"),
        frac_above_min=_opt_float("frac_above_min"),
        frac_above_max=_opt_float("frac_above_max"),
        asof=(str(raw["asof"]) if raw.get("asof") not in (None, "") else None),
        opt_symbol=str(raw.get("opt_symbol") or "QQQ").upper(),
        opt_lookback_days=int(raw.get("opt_lookback_days", 40) or 40),
        opt_imbalance_max=float(
            raw.get("opt_imbalance_max", -0.05)
            if raw.get("opt_imbalance_max") is not None
            else -0.05
        ),
        opt_chop_pctile_min=float(
            raw.get("opt_chop_pctile_min", 0.70)
            if raw.get("opt_chop_pctile_min") is not None
            else 0.70
        ),
        opt_gate=str(raw.get("opt_gate") or "imb_only").strip().lower(),
        opt_bucketed_root=(
            str(raw["opt_bucketed_root"]) if raw.get("opt_bucketed_root") not in (None, "") else None
        ),
        wash_refine=bool(raw.get("wash_refine", False)),
        wash_refine_chop_max=float(
            raw.get("wash_refine_chop_max", 1.85)
            if raw.get("wash_refine_chop_max") is not None
            else 1.85
        ),
        wash_refine_med_stock_ret_max=float(
            raw.get("wash_refine_med_stock_ret_max", 0.003)
            if raw.get("wash_refine_med_stock_ret_max") is not None
            else 0.003
        ),
        wash_refine_pcr_max=float(
            raw.get("wash_refine_pcr_max", 2.0)
            if raw.get("wash_refine_pcr_max") is not None
            else 2.0
        ),
        wash_refine_iv_mom_max=float(
            raw.get("wash_refine_iv_mom_max", 0.03)
            if raw.get("wash_refine_iv_mom_max") is not None
            else 0.03
        ),
        wash_refine_n_down_max=int(raw.get("wash_refine_n_down_max", 4) or 4),
        wash_refine_am_start=str(raw.get("wash_refine_am_start") or "09:35"),
    )


def _session_ret_to_asof(
    df: Any, *, date: str, t0: str, asof: str
) -> float | None:
    if df is None or getattr(df, "empty", True):
        return None
    d = df
    if "date" in d.columns:
        d = d[d["date"].astype(str) == str(date)]
    if d is None or d.empty or "timestamp" not in d.columns or "close" not in d.columns:
        return None
    ts = pd.to_datetime(d["timestamp"], utc=True, errors="coerce")
    try:
        ts = ts.dt.tz_convert("America/New_York")
    except Exception:
        pass
    hhmm = ts.dt.strftime("%H:%M")
    a = d.loc[hhmm >= str(t0)].sort_values("timestamp")
    b = d.loc[hhmm <= str(asof)].sort_values("timestamp")
    if a.empty or b.empty:
        return None
    p0 = float(a.iloc[0]["close"])
    p1 = float(b.iloc[-1]["close"])
    if p0 <= 0:
        return None
    return float(p1 / p0 - 1.0)


def _wash_refine_ok(
    cfg: PathFastPackConfig,
    *,
    date: str,
    stock_by: dict[str, Any],
    symbols: list[str],
    asof: str,
) -> bool:
    """Reject false-alarm wash mornings; keep Jul20-like mild wash."""
    if not cfg.wash_refine:
        return True
    asof_hhmm = str(cfg.asof or asof)
    t0 = str(cfg.wash_refine_am_start or "09:35")
    rets: list[float] = []
    n_down = 0
    for sym in symbols:
        r = _session_ret_to_asof(stock_by.get(sym), date=str(date), t0=t0, asof=asof_hhmm)
        if r is None or not np.isfinite(r):
            continue
        rets.append(float(r))
        if r <= -0.008:
            n_down += 1
    if n_down > int(cfg.wash_refine_n_down_max):
        return False
    if rets:
        med = float(np.nanmedian(np.asarray(rets, dtype=float)))
        if med >= float(cfg.wash_refine_med_stock_ret_max):
            return False

    snap = _asof_snap(cfg, str(date), asof_hhmm)
    if snap:
        chop = _local_chop_score(snap)
        if chop >= float(cfg.wash_refine_chop_max):
            return False
        pcr = float(snap.get("options_pcr_volume") or 0.0)
        if np.isfinite(pcr) and pcr >= float(cfg.wash_refine_pcr_max):
            return False
        iv_mom = float(snap.get("options_iv_momentum") or 0.0)
        if np.isfinite(iv_mom) and iv_mom >= float(cfg.wash_refine_iv_mom_max):
            return False
    return True


def _wash_hit(
    cfg: PathFastPackConfig,
    *,
    date: str,
    stock_by: dict[str, Any],
    qqq_df: Any,
    symbols: list[str],
    asof: str,
    washout_breadth_min: int,
    wash_drop_min: float,
    frac_above_min: float,
    frac_above_max: float,
) -> bool:
    from maga7.common.predictive_prevention import evaluate_prevention_rule

    hit = evaluate_prevention_rule(
        date=str(date),
        stock_by=stock_by,
        qqq_df=qqq_df,
        symbols=list(symbols),
        asof=str(cfg.asof or asof),
        rule="mixed_wash_up",
        prefer_risk_off=True,
        washout_breadth_min=int(
            cfg.washout_breadth_min
            if cfg.washout_breadth_min is not None
            else washout_breadth_min
        ),
        wash_drop_min=float(
            cfg.wash_drop_min if cfg.wash_drop_min is not None else wash_drop_min
        ),
        frac_above_min=float(
            cfg.frac_above_min if cfg.frac_above_min is not None else frac_above_min
        ),
        frac_above_max=float(
            cfg.frac_above_max if cfg.frac_above_max is not None else frac_above_max
        ),
    )
    if hit is None:
        return False
    return _wash_refine_ok(
        cfg,
        date=str(date),
        stock_by=stock_by,
        symbols=list(symbols),
        asof=str(asof),
    )


def _surface_frame(cfg: PathFastPackConfig) -> pd.DataFrame:
    from maga7.common.option_surface import DEFAULT_BUCKETED_ROOT, load_surface_range

    root = (
        Path(cfg.opt_bucketed_root).expanduser()
        if cfg.opt_bucketed_root
        else DEFAULT_BUCKETED_ROOT
    )
    key = f"{cfg.opt_symbol}:{root}"
    if key not in _SURFACE_CACHE:
        # Wide window; filter by date in asof helpers.
        _SURFACE_CACHE[key] = load_surface_range(
            cfg.opt_symbol, "2026-01-01", "2026-12-31", root=root
        )
    return _SURFACE_CACHE[key]


def _asof_snap(cfg: PathFastPackConfig, date: str, asof: str) -> dict[str, float] | None:
    from maga7.common.option_surface import surface_asof

    key = (cfg.opt_symbol, str(date), str(cfg.asof or asof))
    if key in _ASOF_CACHE:
        return _ASOF_CACHE[key]
    df = _surface_frame(cfg)
    snap = surface_asof(df, date=str(date), asof=str(cfg.asof or asof))
    _ASOF_CACHE[key] = snap
    return snap


def _local_chop_score(snap: dict[str, float]) -> float:
    from maga7.common.option_surface import opt_chop_score

    s = opt_chop_score(snap)
    return float(s) if s is not None else 0.0


def _opt_chop_pctile_hit(cfg: PathFastPackConfig, *, date: str, asof: str, snap: dict) -> bool:
    """Causal chop-score percentile vs prior lookback 10:30 snaps."""
    df = _surface_frame(cfg)
    if df is None or df.empty:
        return False
    asof_hhmm = str(cfg.asof or asof)
    hist = df[df["timestamp"].dt.strftime("%H:%M") == asof_hhmm].copy()
    if hist.empty:
        return False
    hist = hist[hist["date"].astype(str) < str(date)].tail(int(cfg.opt_lookback_days))
    if len(hist) < 10:
        return _local_chop_score(snap) >= 0.8
    scores = []
    for _, row in hist.iterrows():
        s = {
            "options_vw_spread": float(row.get("options_vw_spread") or 0.0),
            "options_iv_divergence": float(row.get("options_iv_divergence") or 0.0),
            "options_gamma_accel": float(row.get("options_gamma_accel") or 0.0),
            "options_vw_imbalance": float(row.get("options_vw_imbalance") or 0.0),
        }
        scores.append(_local_chop_score(s))
    today = _local_chop_score(snap)
    arr = np.asarray(scores, dtype=float)
    pct = float((arr <= today).mean()) if len(arr) else 0.0
    return pct >= float(cfg.opt_chop_pctile_min)


def _opt_chop_hit(cfg: PathFastPackConfig, *, date: str, asof: str) -> bool:
    """QQQ surface gate. Default ``imb_only`` (see arm diagnosis 2026-07-21)."""
    snap = _asof_snap(cfg, date, asof)
    if not snap:
        return False
    imb = float(snap.get("options_vw_imbalance") or 0.0)
    imb_ok = np.isfinite(imb) and imb <= float(cfg.opt_imbalance_max)
    gate = str(cfg.opt_gate or "imb_only").strip().lower()

    if gate in {"imb_only", "imbalance", "imb"}:
        return bool(imb_ok)

    if gate in {"imb_and_chop", "and"}:
        return bool(imb_ok) and _opt_chop_pctile_hit(
            cfg, date=date, asof=asof, snap=snap
        )

    # legacy: imb OR chop percentile (over-arms on +imb trend days)
    if imb_ok:
        return True
    return _opt_chop_pctile_hit(cfg, date=date, asof=asof, snap=snap)


def path_fast_pack_day_should_arm(
    cfg: PathFastPackConfig,
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
    if not cfg.enabled:
        return False
    when = str(cfg.when or "always").strip().lower()
    if when in {"", "always", "on", "all"}:
        return True

    wash_kw = dict(
        date=str(date),
        stock_by=stock_by,
        qqq_df=qqq_df,
        symbols=list(symbols),
        asof=str(cfg.asof or asof),
        washout_breadth_min=washout_breadth_min,
        wash_drop_min=wash_drop_min,
        frac_above_min=frac_above_min,
        frac_above_max=frac_above_max,
    )

    if when in {"mixed_wash_up", "prevention", "up_toxic", "toxic_up"}:
        return _wash_hit(cfg, **wash_kw)

    if when in {"qqq_opt_chop", "opt_chop", "qqq_surface_chop"}:
        return _opt_chop_hit(cfg, date=str(date), asof=str(cfg.asof or asof))

    if when in {"wash_or_opt_chop", "wash_or_qqq_opt"}:
        return _wash_hit(cfg, **wash_kw) or _opt_chop_hit(
            cfg, date=str(date), asof=str(cfg.asof or asof)
        )

    if when in {"wash_and_opt_chop", "wash_and_qqq_opt"}:
        return _wash_hit(cfg, **wash_kw) and _opt_chop_hit(
            cfg, date=str(date), asof=str(cfg.asof or asof)
        )

    return False


def apply_path_fast_pack_overrides(
    *,
    hold_minutes: int,
    trail_activate: float | None,
    trail_dd: float | None,
    stock_rev: Any,
    pack: PathFastPackConfig,
) -> dict[str, Any]:
    """Return kwargs to merge into simulate_trade / live exit params when armed."""
    from maga7.common.delta_time_stop import StockRevExitConfig

    srev = stock_rev
    if isinstance(srev, StockRevExitConfig):
        srev = StockRevExitConfig(
            enabled=True,
            min_hold_minutes=float(pack.stock_rev_min_hold_minutes),
            stock_max=float(pack.stock_rev_stock_max),
            opt_mtm_max=float(pack.stock_rev_opt_mtm_max),
            when="always",
            routes=getattr(srev, "routes", None),
            washout_breadth_min=srev.washout_breadth_min,
            wash_drop_min=srev.wash_drop_min,
            frac_above_min=srev.frac_above_min,
            frac_above_max=srev.frac_above_max,
        )
    elif isinstance(srev, dict) or srev is None:
        srev = StockRevExitConfig(
            enabled=True,
            min_hold_minutes=float(pack.stock_rev_min_hold_minutes),
            stock_max=float(pack.stock_rev_stock_max),
            opt_mtm_max=float(pack.stock_rev_opt_mtm_max),
            when="always",
        )
    return {
        "hold_minutes": int(pack.hold_minutes),
        "trail_activate": float(pack.trail_activate),
        "trail_dd": float(pack.trail_dd),
        "stock_rev_exit": srev,
        "hold_extend_minutes": None,
    }

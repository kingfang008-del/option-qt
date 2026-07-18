"""Causal trend-purity score → position size scale.

v1 ``momentum``: |from_prev| + peer breadth (tended to rank early fires high).
v2 ``efficiency``: intraday path noise / range efficiency (targets chop vs clean trend).

All inputs are as-of signal ``feature_ts`` (no EOD lookahead).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def path_efficiency_features(
    stock_day: pd.DataFrame | None,
    *,
    asof_ts,
    direction: str,
    window: int = 20,
) -> dict[str, float | None]:
    """Intraday path / noise features ending at ``asof_ts``.

    Returns keys (values in ~[0,1] when defined, else None):
      - ``path_eff``: |net return| / sum(|bar ret|)  (1=straight line, ~0=chop)
      - ``range_eff``: |close-open| / (high-low) over window
      - ``dir_frac``: fraction of bars closing in trade direction
      - ``adverse``: max adverse excursion / (|net| + eps), mapped later to purity
    """
    out: dict[str, float | None] = {
        "path_eff": None,
        "range_eff": None,
        "dir_frac": None,
        "adverse": None,
    }
    if stock_day is None or getattr(stock_day, "empty", True):
        return out
    if "close" not in stock_day.columns or "timestamp" not in stock_day.columns:
        return out
    asof = pd.Timestamp(asof_ts)
    if asof.tzinfo is None:
        asof = asof.tz_localize("America/New_York")
    else:
        asof = asof.tz_convert("America/New_York")
    day = stock_day[stock_day["timestamp"] <= asof].sort_values("timestamp")
    w = max(int(window), 3)
    if len(day) < w:
        return out
    g = day.tail(w)
    close = g["close"].astype(float).to_numpy()
    if not np.isfinite(close).all() or close[0] <= 0 or close[-1] <= 0:
        return out
    rets = np.diff(close) / close[:-1]
    if len(rets) == 0:
        return out
    net = float(close[-1] / close[0] - 1.0)
    path_den = float(np.abs(rets).sum())
    if path_den > 1e-12:
        out["path_eff"] = float(np.clip(abs(net) / path_den, 0.0, 1.0))
    else:
        out["path_eff"] = 0.0

    o = float(g.iloc[0]["open"]) if "open" in g.columns else float(close[0])
    h = float(g["high"].astype(float).max()) if "high" in g.columns else float(np.max(close))
    l = float(g["low"].astype(float).min()) if "low" in g.columns else float(np.min(close))
    c = float(close[-1])
    rng = h - l
    if rng > 1e-12 and np.isfinite(o):
        out["range_eff"] = float(np.clip(abs(c - o) / rng, 0.0, 1.0))
    else:
        out["range_eff"] = None

    dir_u = str(direction).upper()
    if "open" in g.columns:
        bar_o = g["open"].astype(float).to_numpy()
        bar_c = close
        if dir_u == "UP":
            fav = bar_c > bar_o
        elif dir_u == "DN":
            fav = bar_c < bar_o
        else:
            fav = np.zeros(len(bar_c), dtype=bool)
        out["dir_frac"] = float(np.mean(fav)) if len(fav) else None
    else:
        if dir_u == "UP":
            fav = rets > 0
        elif dir_u == "DN":
            fav = rets < 0
        else:
            fav = np.zeros(len(rets), dtype=bool)
        out["dir_frac"] = float(np.mean(fav)) if len(fav) else None

    # Max adverse excursion vs start of window (positive = against trade).
    if dir_u == "UP":
        dd = (close[0] - close) / close[0]
        adv = float(np.nanmax(dd)) if len(dd) else 0.0
    elif dir_u == "DN":
        dd = (close - close[0]) / close[0]
        adv = float(np.nanmax(dd)) if len(dd) else 0.0
    else:
        adv = 0.0
    out["adverse"] = float(max(adv, 0.0))
    return out


def trend_purity_score(
    *,
    direction: str,
    from_prev: float | None,
    peer_n: int | None,
    peer_min: int = 3,
    peer_universe: int = 8,
    mf10: float | None = None,
    streak: int | None = None,
    streak_min: int = 8,
    qqq_from_prev: float | None = None,
    si: float | None = None,
    fp_ref: float = 0.025,
    features: str = "momentum",
    path_eff: float | None = None,
    range_eff: float | None = None,
    dir_frac: float | None = None,
    adverse: float | None = None,
    adverse_ref: float = 0.005,
) -> tuple[float, dict[str, float]]:
    """Return ``(score ∈ ~[0,1], components)``.

    ``features``:
      - ``momentum``: original |fp|+peer blend (research control)
      - ``efficiency``: path/range/dir_frac − adverse (preferred)
      - ``hybrid``: 50/50 momentum + efficiency
    """
    dir_u = str(direction).upper()
    parts: dict[str, float] = {}

    try:
        afp = abs(float(from_prev)) if from_prev is not None and np.isfinite(float(from_prev)) else 0.0
    except (TypeError, ValueError):
        afp = 0.0
    ref = max(float(fp_ref), 1e-6)
    parts["fp"] = float(min(1.0, afp / ref))

    pn = int(peer_n or 0)
    pmin = max(int(peer_min), 0)
    univ = max(int(peer_universe), pmin + 1)
    parts["peer"] = float(np.clip((pn - pmin) / max(univ - pmin, 1), 0.0, 1.0))

    qfp = qqq_from_prev
    if qfp is None or not np.isfinite(float(qfp)):
        parts["qqq"] = 0.5
    else:
        qfp_f = float(qfp)
        if dir_u == "UP":
            parts["qqq"] = 1.0 if qfp_f > 0 else 0.0
        elif dir_u == "DN":
            parts["qqq"] = 1.0 if qfp_f < 0 else 0.0
        else:
            parts["qqq"] = 0.0

    if mf10 is None or not np.isfinite(float(mf10)):
        parts["mf"] = 0.5
    else:
        mf = float(mf10)
        if dir_u == "UP":
            parts["mf"] = 1.0 if mf > 0 else 0.0
        elif dir_u == "DN":
            parts["mf"] = 1.0 if mf < 0 else 0.0
        else:
            parts["mf"] = 0.0

    if streak is not None and int(streak_min) > 0:
        parts["streak"] = float(np.clip((int(streak) - int(streak_min)) / 10.0, 0.0, 1.0))
    else:
        parts["streak"] = 0.5

    if si is None or not np.isfinite(float(si)):
        parts["si"] = 0.5
    else:
        si_f = float(si)
        if dir_u == "UP":
            parts["si"] = float(np.clip((si_f + 1.0) / 2.0, 0.0, 1.0))
        elif dir_u == "DN":
            parts["si"] = float(np.clip((1.0 - si_f) / 2.0, 0.0, 1.0))
        else:
            parts["si"] = 0.0

    # Efficiency block (defaults neutral if missing).
    parts["path_eff"] = float(path_eff) if path_eff is not None and np.isfinite(path_eff) else 0.5
    parts["range_eff"] = float(range_eff) if range_eff is not None and np.isfinite(range_eff) else 0.5
    parts["dir_frac"] = float(dir_frac) if dir_frac is not None and np.isfinite(dir_frac) else 0.5
    if adverse is None or not np.isfinite(float(adverse)):
        parts["adverse_ok"] = 0.5
    else:
        # 0 adverse → 1.0; adverse_ref (e.g. 50bp) → 0.0
        parts["adverse_ok"] = float(
            np.clip(1.0 - float(adverse) / max(float(adverse_ref), 1e-6), 0.0, 1.0)
        )

    mom = (
        0.35 * parts["fp"]
        + 0.25 * parts["peer"]
        + 0.15 * parts["qqq"]
        + 0.10 * parts["mf"]
        + 0.05 * parts["streak"]
        + 0.10 * parts["si"]
    )
    eff = (
        0.35 * parts["path_eff"]
        + 0.25 * parts["range_eff"]
        + 0.20 * parts["dir_frac"]
        + 0.15 * parts["adverse_ok"]
        + 0.05 * parts["qqq"]
    )
    feat = str(features or "momentum").strip().lower()
    if feat in {"efficiency", "eff", "noise", "path"}:
        score = eff
    elif feat in {"hybrid", "mix", "both"}:
        score = 0.5 * mom + 0.5 * eff
    else:
        score = mom
    parts["mom"] = float(mom)
    parts["eff"] = float(eff)
    return float(score), parts


def trend_purity_size_scale(
    score: float,
    trade: dict[str, Any] | None = None,
) -> tuple[float, str]:
    """Map purity score → multiplicative size scale in ``[0, 1]``.

    ``trade.trend_purity_mode``:
      - ``continuous`` (default): scale = clip(score / high, min_scale, 1)
      - ``tier``: <low → min_scale; <high → mid_scale; else 1
      - ``skip_low``: <skip_below → 0; else continuous
    """
    trade = trade or {}
    mode = str(trade.get("trend_purity_mode") or "continuous").strip().lower()
    high = float(trade.get("trend_purity_high", 0.70) or 0.70)
    low = float(trade.get("trend_purity_low", 0.40) or 0.40)
    min_scale = float(trade.get("trend_purity_min_scale", 0.35) or 0.35)
    mid_scale = float(trade.get("trend_purity_mid_scale", 0.55) or 0.55)
    skip_below = trade.get("trend_purity_skip_below")
    high = max(high, 1e-6)
    min_scale = float(np.clip(min_scale, 0.0, 1.0))
    mid_scale = float(np.clip(mid_scale, 0.0, 1.0))
    s = float(score)

    if mode in {"skip_low", "skip"}:
        thr = float(skip_below) if skip_below is not None else low
        if s < thr:
            return 0.0, "purity_skip"
        scale = float(np.clip(s / high, min_scale, 1.0))
        return scale, "purity_cont"

    if mode in {"tier", "buckets"}:
        if s < low:
            return min_scale, "purity_tier_low"
        if s < high:
            return mid_scale, "purity_tier_mid"
        return 1.0, "purity_tier_high"

    scale = float(np.clip(s / high, min_scale, 1.0))
    return scale, "purity_cont"

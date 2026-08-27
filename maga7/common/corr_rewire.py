"""Causal Mag7–QQQ correlation rewiring (day / entry risk scale).

Adapted from Vibe-Trading ``academic_corr_rewire`` / ``correlation-regime``
Mode 4: compare an event-window correlation to the calm window immediately
before it. Here the primary series is equal-weight Mag7 1m returns vs QQQ
(not a full cross-sectional RANK zoo).

Default OFF — research gate for loss reduction / MaxDD, not a timing alpha.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

NY = "America/New_York"

DEFAULT_MAG7 = ("NVDA", "TSLA", "AAPL", "AMZN", "META", "MSFT", "AMD", "GOOGL")


def _to_ny(ts) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        return t.tz_localize(NY)
    return t.tz_convert(NY)


def _rth_returns(df: pd.DataFrame | None, *, asof: pd.Timestamp) -> pd.Series:
    """1m close-to-close returns up to ``asof`` (RTH 09:30+)."""
    if df is None or df.empty or "close" not in df.columns:
        return pd.Series(dtype=float)
    g = df.copy()
    g["timestamp"] = pd.to_datetime(g["timestamp"])
    if g["timestamp"].dt.tz is None:
        g["timestamp"] = g["timestamp"].dt.tz_localize(NY)
    else:
        g["timestamp"] = g["timestamp"].dt.tz_convert(NY)
    g = g[g["timestamp"] <= asof].sort_values("timestamp")
    if g.empty:
        return pd.Series(dtype=float)
    hm = g["timestamp"].dt.hour * 60 + g["timestamp"].dt.minute
    g = g[hm >= 9 * 60 + 30]
    if len(g) < 3:
        return pd.Series(dtype=float)
    px = pd.to_numeric(g["close"], errors="coerce")
    ret = px.pct_change()
    out = pd.Series(ret.to_numpy(), index=pd.DatetimeIndex(g["timestamp"]), dtype=float)
    return out.replace([np.inf, -np.inf], np.nan).dropna()


def _align_ew_mag7(
    stock_by: dict[str, pd.DataFrame],
    symbols: list[str],
    *,
    asof: pd.Timestamp,
) -> tuple[pd.Series, pd.Series]:
    """Equal-weight Mag7 returns and QQQ returns, inner-joined on timestamp."""
    series: dict[str, pd.Series] = {}
    for sym in symbols:
        r = _rth_returns(stock_by.get(sym), asof=asof)
        if not r.empty:
            series[str(sym).upper()] = r
    q = _rth_returns(stock_by.get("QQQ"), asof=asof)
    if not series or q.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    panel = pd.DataFrame(series)
    # require at least half the names present on a bar
    min_n = max(2, int(np.ceil(0.5 * len(series))))
    ew = panel.mean(axis=1, skipna=True)
    n_obs = panel.notna().sum(axis=1)
    ew = ew.where(n_obs >= min_n)
    joined = pd.concat([ew.rename("mag7"), q.rename("qqq")], axis=1, join="inner").dropna()
    if joined.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    return joined["mag7"], joined["qqq"]


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float | None:
    if len(a) < 5:
        return None
    if float(np.std(a)) < 1e-12 or float(np.std(b)) < 1e-12:
        return None
    c = float(np.corrcoef(a, b)[0, 1])
    if not np.isfinite(c):
        return None
    return c


def _edge_density(panel: pd.DataFrame, *, edge_threshold: float = 0.5) -> float | None:
    """Fraction of Mag7 pairs with |ρ| >= threshold in the window."""
    if panel.shape[1] < 2 or len(panel) < 10:
        return None
    corr = panel.corr().to_numpy()
    n = corr.shape[0]
    vals = []
    for i in range(n):
        for j in range(i + 1, n):
            v = corr[i, j]
            if np.isfinite(v):
                vals.append(abs(float(v)))
    if not vals:
        return None
    thr = float(edge_threshold)
    return float(sum(1 for v in vals if v >= thr) / len(vals))


@dataclass
class CorrRewireSnapshot:
    rho_event: float | None
    rho_calm: float | None
    rewire: float | None
    edge_density: float | None
    n_event: int
    n_calm: int
    trigger: bool
    reason: str
    size_scale: float = 1.0


def corr_rewire_asof(
    stock_by: dict[str, pd.DataFrame],
    *,
    asof_ts,
    symbols: list[str] | tuple[str, ...] | None = None,
    event_bars: int = 60,
    calm_bars: int = 180,
    min_bars: int = 30,
    edge_threshold: float = 0.5,
    rewire_min: float | None = 0.25,
    rho_event_max: float | None = None,
    rho_event_min: float | None = None,
    edge_density_min: float | None = None,
    action: str = "scale",
    scale: float = 0.5,
) -> CorrRewireSnapshot:
    """Causal Mag7–QQQ corr rewire at ``asof_ts``.

    Triggers (any):
      - ``rewire = |ρ_event - ρ_calm| >= rewire_min``
      - ``ρ_event <= rho_event_min`` (decorrelated)
      - ``ρ_event >= rho_event_max`` (optional fuse / chase)
      - Mag7 pairwise ``edge_density <= edge_density_min``
    """
    asof = _to_ny(asof_ts)
    syms = [str(s).upper() for s in (symbols or DEFAULT_MAG7)]
    mag7, qqq = _align_ew_mag7(stock_by, syms, asof=asof)
    empty = CorrRewireSnapshot(
        rho_event=None,
        rho_calm=None,
        rewire=None,
        edge_density=None,
        n_event=0,
        n_calm=0,
        trigger=False,
        reason="insufficient",
        size_scale=1.0,
    )
    if len(mag7) < int(min_bars) + 5:
        return empty

    e_n = int(event_bars)
    c_n = int(calm_bars)
    if len(mag7) < e_n:
        # use whatever we have for event; calm may be short
        e_n = max(int(min_bars), len(mag7) // 3)
    event_m = mag7.iloc[-e_n:]
    event_q = qqq.iloc[-e_n:]
    calm_end = len(mag7) - e_n
    if calm_end < int(min_bars):
        rho_e = _safe_corr(event_m.to_numpy(), event_q.to_numpy())
        return CorrRewireSnapshot(
            rho_event=rho_e,
            rho_calm=None,
            rewire=None,
            edge_density=None,
            n_event=len(event_m),
            n_calm=0,
            trigger=False,
            reason="no_calm",
            size_scale=1.0,
        )
    calm_m = mag7.iloc[max(0, calm_end - c_n) : calm_end]
    calm_q = qqq.iloc[max(0, calm_end - c_n) : calm_end]
    rho_e = _safe_corr(event_m.to_numpy(), event_q.to_numpy())
    rho_c = _safe_corr(calm_m.to_numpy(), calm_q.to_numpy())
    rewire = None
    if rho_e is not None and rho_c is not None:
        rewire = abs(float(rho_e) - float(rho_c))

    # Mag7 pairwise density on event window
    panel_cols = {}
    for sym in syms:
        r = _rth_returns(stock_by.get(sym), asof=asof)
        if r.empty:
            continue
        panel_cols[sym] = r.iloc[-e_n:]
    edge = None
    if len(panel_cols) >= 3:
        panel = pd.DataFrame(panel_cols).dropna(how="all")
        # align length
        panel = panel.tail(e_n)
        edge = _edge_density(panel, edge_threshold=float(edge_threshold))

    reasons: list[str] = []
    if rewire_min is not None and rewire is not None and rewire >= float(rewire_min):
        reasons.append(f"rewire>={rewire_min:g}")
    if rho_event_min is not None and rho_e is not None and rho_e <= float(rho_event_min):
        reasons.append(f"rho_e<={rho_event_min:g}")
    if rho_event_max is not None and rho_e is not None and rho_e >= float(rho_event_max):
        reasons.append(f"rho_e>={rho_event_max:g}")
    if edge_density_min is not None and edge is not None and edge <= float(edge_density_min):
        reasons.append(f"edge<={edge_density_min:g}")

    trigger = bool(reasons)
    act = str(action or "scale").strip().lower()
    sc = float(scale)
    sc = max(0.0, min(sc, 1.0))
    if not trigger:
        return CorrRewireSnapshot(
            rho_event=rho_e,
            rho_calm=rho_c,
            rewire=rewire,
            edge_density=edge,
            n_event=len(event_m),
            n_calm=len(calm_m),
            trigger=False,
            reason="ok",
            size_scale=1.0,
        )
    if act == "block":
        return CorrRewireSnapshot(
            rho_event=rho_e,
            rho_calm=rho_c,
            rewire=rewire,
            edge_density=edge,
            n_event=len(event_m),
            n_calm=len(calm_m),
            trigger=True,
            reason="+".join(reasons),
            size_scale=0.0,
        )
    return CorrRewireSnapshot(
        rho_event=rho_e,
        rho_calm=rho_c,
        rewire=rewire,
        edge_density=edge,
        n_event=len(event_m),
        n_calm=len(calm_m),
        trigger=True,
        reason="+".join(reasons),
        size_scale=sc,
    )


def corr_rewire_from_trade(trade: dict[str, Any] | None) -> dict[str, Any] | None:
    """Parse ``trade.corr_rewire`` config; None if disabled."""
    trade = trade or {}
    raw = trade.get("corr_rewire")
    if raw is None:
        return None
    if isinstance(raw, bool):
        return {"enabled": bool(raw)} if raw else None
    if not isinstance(raw, dict):
        return None
    if not bool(raw.get("enabled", False)):
        return None
    return dict(raw)

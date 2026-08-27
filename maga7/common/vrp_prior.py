"""VRP-lite soft prior for long-option books (buyer-side, not short-vol).

Classic VRP sells rich IV; Mag7 buys options, so we only use IV−RV as a
**size / skip prior** when implied is statistically rich vs recent realized.

Causal: QQQ surface ATM IV @ asof + trailing close-to-close RV (no same-day future).

RV prefers ``stock_1s_root`` (``/mnt/s990/data/raw_1s/stocks``) — not left-labeled
``spnq_train`` 1m cache.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from maga7.common.option_surface import (
    DEFAULT_BUCKETED_ROOT,
    load_surface_range,
    surface_asof,
)

NY = "America/New_York"
DEFAULT_STOCK_1S_ROOT = Path("/mnt/s990/data/raw_1s/stocks")


@dataclass(frozen=True)
class VrpSizeScaleConfig:
    enabled: bool = False
    asof: str = "10:30"
    rv_lookback_days: int = 5
    iv_col: str = "options_struc_atm_iv"
    # Action when VRP is rich
    mode: str = "scale"  # scale | skip
    scale: float = 0.5
    # Rich if VRP >= rolling pctile (causal, lookback exclusive of today)
    rich_pctile: float = 0.70
    # Absolute floor (annualized decimal); None = pctile only
    rich_min: float | None = 0.0
    surface_symbol: str = "QQQ"
    surface_root: str | None = None
    stock_1s_root: str | None = None
    missing: str = "passthrough"  # passthrough | skip


def parse_vrp_size_scale(raw: Any) -> VrpSizeScaleConfig:
    if not isinstance(raw, dict):
        return VrpSizeScaleConfig(enabled=False)
    rich_min = raw.get("rich_min", 0.0)
    return VrpSizeScaleConfig(
        enabled=bool(raw.get("enabled", False)),
        asof=str(raw.get("asof") or "10:30"),
        rv_lookback_days=max(2, int(raw.get("rv_lookback_days", 5) or 5)),
        iv_col=str(raw.get("iv_col") or "options_struc_atm_iv"),
        mode=str(raw.get("mode") or "scale").strip().lower(),
        scale=float(raw.get("scale", 0.5) or 0.5),
        rich_pctile=float(raw.get("rich_pctile", 0.70) or 0.70),
        rich_min=(None if rich_min is None else float(rich_min)),
        surface_symbol=str(raw.get("surface_symbol") or "QQQ").upper(),
        surface_root=(str(raw["surface_root"]) if raw.get("surface_root") else None),
        stock_1s_root=(str(raw["stock_1s_root"]) if raw.get("stock_1s_root") else None),
        missing=str(raw.get("missing") or "passthrough").strip().lower(),
    )


def _daily_closes_from_1s(
    stock_1s_root: Path | str | None,
    *,
    symbol: str,
    start: str,
    end: str,
) -> pd.Series:
    """RTH last print per session from authoritative 1s stock root."""
    if stock_1s_root is None:
        return pd.Series(dtype=float)
    root = Path(stock_1s_root).expanduser()
    if not root.is_dir():
        return pd.Series(dtype=float)
    from maga7.common.bar_agg import load_stock_1s_day
    from maga7.common.stock_1s import session_dates

    closes: dict[str, float] = {}
    for date in session_dates(str(start), str(end)):
        day = load_stock_1s_day(root, symbol, date)
        if day is None or day.empty or "close" not in day.columns:
            continue
        ts = pd.to_datetime(day["timestamp"])
        if getattr(ts.dt, "tz", None) is None:
            ts = ts.dt.tz_localize(NY, ambiguous="infer")
        else:
            ts = ts.dt.tz_convert(NY)
        # RTH prints only (exclude pre/post for close-to-close RV)
        t = ts.dt.time
        m = (t >= pd.Timestamp("09:30").time()) & (t < pd.Timestamp("16:00").time())
        sub = day.loc[m]
        if sub.empty:
            sub = day
        px = float(sub["close"].iloc[-1])
        if np.isfinite(px) and px > 0:
            closes[str(date)] = px
    if not closes:
        return pd.Series(dtype=float)
    return pd.Series(closes).sort_index().astype(float)


def _daily_closes(qqq_df: pd.DataFrame | None) -> pd.Series:
    """Fallback: last bar close per date from an in-memory frame (1s-agg or cache)."""
    if qqq_df is None or getattr(qqq_df, "empty", True):
        return pd.Series(dtype=float)
    day = qqq_df.copy()
    if "date" not in day.columns:
        return pd.Series(dtype=float)
    ts = pd.to_datetime(day["timestamp"])
    if getattr(ts.dt, "tz", None) is None:
        ts = ts.dt.tz_localize(NY, ambiguous="infer")
    else:
        ts = ts.dt.tz_convert(NY)
    day = day.assign(_ts=ts).sort_values("_ts")
    g = day.groupby(day["date"].astype(str), sort=True)["close"].last()
    return g.astype(float)


def trailing_rv_ann(
    closes: pd.Series,
    *,
    date: str,
    lookback_days: int,
) -> float | None:
    """Close-to-close RV ending **before** ``date`` (causal)."""
    if closes is None or closes.empty:
        return None
    idx = [str(x) for x in closes.index]
    # exclusive of today
    hist = [d for d in idx if d < str(date)]
    if len(hist) < int(lookback_days) + 1:
        return None
    use = hist[-(int(lookback_days) + 1) :]
    px = closes.reindex(use).astype(float).dropna()
    if len(px) < int(lookback_days) + 1:
        return None
    rets = px.pct_change().dropna()
    if len(rets) < int(lookback_days):
        return None
    vol = float(rets.std(ddof=1)) * float(np.sqrt(252.0))
    return vol if np.isfinite(vol) and vol > 0 else None


def build_vrp_day_table(
    *,
    qqq_df: pd.DataFrame | None,
    start: str,
    end: str,
    cfg: VrpSizeScaleConfig,
    warm_days: int = 40,
    stock_1s_root: Path | str | None = None,
) -> pd.DataFrame:
    """Per-day IV, RV, VRP, causal rich flag.

    Loads ~``warm_days`` of surface/history before ``start`` so the rolling
    VRP percentile is usable from the first evaluation day.

    RV source priority: ``stock_1s_root`` / cfg.stock_1s_root → ``qqq_df``.
    """
    root = Path(cfg.surface_root).expanduser() if cfg.surface_root else DEFAULT_BUCKETED_ROOT
    warm_pad = str(pd.Timestamp(start) - pd.Timedelta(days=int(warm_days) * 2 + 14))[:10]
    s1s = stock_1s_root or cfg.stock_1s_root or DEFAULT_STOCK_1S_ROOT
    closes = _daily_closes_from_1s(
        s1s, symbol=cfg.surface_symbol, start=warm_pad, end=str(end)
    )
    if closes.empty:
        closes = _daily_closes(qqq_df)
    warm_start = str(start)
    if len(closes):
        before = [d for d in closes.index.astype(str) if d < str(start)]
        if before:
            warm_start = before[max(0, len(before) - int(warm_days))]
    else:
        warm_start = warm_pad
    surf = load_surface_range(cfg.surface_symbol, warm_start, end, root=root)
    dates = sorted({str(d) for d in (surf["date"].astype(str).unique() if len(surf) else [])})
    if len(closes):
        dates = sorted(
            set(dates)
            | {d for d in closes.index.astype(str) if warm_start <= d <= end}
        )

    rows: list[dict[str, Any]] = []
    vrp_hist: list[float] = []
    for date in dates:
        if date < warm_start or date > end:
            continue
        snap = surface_asof(surf, date=date, asof=cfg.asof) if len(surf) else None
        iv = None
        if snap is not None:
            raw_iv = snap.get(cfg.iv_col)
            if raw_iv is None:
                raw_iv = snap.get("options_vw_iv")
            try:
                iv = float(raw_iv) if raw_iv is not None else None
            except (TypeError, ValueError):
                iv = None
            if iv is not None and (not np.isfinite(iv) or iv <= 0):
                iv = None
        rv = trailing_rv_ann(closes, date=date, lookback_days=cfg.rv_lookback_days)
        vrp = (float(iv) - float(rv)) if (iv is not None and rv is not None) else None
        rich = False
        thr = None
        # need a few history points before pctile is meaningful
        if vrp is not None and len(vrp_hist) >= 5:
            thr = float(np.quantile(vrp_hist, float(cfg.rich_pctile)))
            rich = bool(vrp >= thr)
            if cfg.rich_min is not None:
                rich = bool(rich and vrp >= float(cfg.rich_min))
        if start <= date <= end:
            rows.append(
                {
                    "date": date,
                    "iv": iv,
                    "rv": rv,
                    "vrp": vrp,
                    "rich_thr": thr,
                    "rich": rich,
                    "n_hist": len(vrp_hist),
                }
            )
        if vrp is not None:
            vrp_hist.append(float(vrp))
    return pd.DataFrame(rows)


def resolve_vrp_size_scale(
    cfg: VrpSizeScaleConfig,
    *,
    date: str,
    day_table: pd.DataFrame | None,
) -> tuple[float, str]:
    """Return ``(scale, reason)``. scale=0 means skip entry."""
    if not cfg.enabled:
        return 1.0, "off"
    if day_table is None or day_table.empty:
        if cfg.missing == "skip":
            return 0.0, "vrp_missing_skip"
        return 1.0, "vrp_missing_passthrough"
    row = day_table[day_table["date"].astype(str) == str(date)]
    if row.empty:
        if cfg.missing == "skip":
            return 0.0, "vrp_missing_skip"
        return 1.0, "vrp_missing_passthrough"
    r = row.iloc[-1]
    if r.get("vrp") is None or (isinstance(r.get("vrp"), float) and not np.isfinite(r["vrp"])):
        if cfg.missing == "skip":
            return 0.0, "vrp_nan_skip"
        return 1.0, "vrp_nan_passthrough"
    if not bool(r.get("rich")):
        return 1.0, "vrp_ok"
    if cfg.mode in {"skip", "block", "hard"}:
        return 0.0, f"vrp_rich_skip:{float(r['vrp']):.3f}"
    scale = max(0.0, min(1.0, float(cfg.scale)))
    return scale, f"vrp_rich_scale:{scale:.2f}:{float(r['vrp']):.3f}"

"""Load locked option-surface aggregates (bucketed_v7) for regime / hold research.

Source pipeline:
  option_cac_day_vectorized.py → quote_options_monthly_iv
  options_locked_feature.py    → quote_options_bucketed_v7/{SYM}

Default root: ``~/train_data/quote_options_bucketed_v7``.
Causal snapshots use asof timestamp (default 10:30 NY).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_BUCKETED_ROOT = Path.home() / "train_data" / "quote_options_bucketed_v7"

SURFACE_COLS = [
    "options_vw_iv",
    "options_vw_delta",
    "options_vw_gamma",
    "options_vw_vega",
    "options_vw_theta",
    "options_vw_vanna",
    "options_vw_charm",
    "options_vw_spread",
    "options_vw_imbalance",
    "options_pcr_volume",
    "options_flow_skew",
    "options_iv_momentum",
    "options_gamma_accel",
    "options_iv_divergence",
    "options_struc_atm_iv",
    "options_struc_skew",
    "options_struc_term",
]


def bucketed_root_from_paths(paths: dict[str, Any] | None = None) -> Path:
    if paths and paths.get("option_bucketed_root"):
        return Path(paths["option_bucketed_root"]).expanduser()
    return DEFAULT_BUCKETED_ROOT


def load_surface_month(symbol: str, year_month: str, *, root: Path | None = None) -> pd.DataFrame:
    root = root or DEFAULT_BUCKETED_ROOT
    path = Path(root) / str(symbol).upper() / f"{year_month}.parquet"
    if not path.is_file():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    if "timestamp" not in df.columns:
        return pd.DataFrame()
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df["timestamp"] = df["timestamp"].dt.tz_convert("America/New_York")
    df["date"] = df["timestamp"].dt.strftime("%Y-%m-%d")
    return df


def load_surface_range(
    symbol: str,
    start: str,
    end: str,
    *,
    root: Path | None = None,
) -> pd.DataFrame:
    root = root or DEFAULT_BUCKETED_ROOT
    months = pd.period_range(start=start[:7], end=end[:7], freq="M").astype(str)
    frames = [load_surface_month(symbol, m, root=root) for m in months]
    frames = [f for f in frames if f is not None and not f.empty]
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    m = (df["date"] >= str(start)[:10]) & (df["date"] <= str(end)[:10])
    return df.loc[m].sort_values("timestamp").reset_index(drop=True)


def surface_asof(
    df: pd.DataFrame,
    *,
    date: str,
    asof: str = "10:30",
) -> dict[str, float] | None:
    """Last bar on ``date`` at or before ``asof`` (HH:MM NY)."""
    if df is None or df.empty:
        return None
    day = df[df["date"].astype(str) == str(date)]
    if day.empty:
        return None
    asof_ts = pd.Timestamp(f"{date} {asof}", tz="America/New_York")
    up = day[day["timestamp"] <= asof_ts]
    if up.empty:
        return None
    row = up.iloc[-1]
    out: dict[str, float] = {"date": str(date), "asof": str(asof)}
    for c in SURFACE_COLS:
        if c not in row.index:
            continue
        v = row[c]
        try:
            fv = float(v)
        except Exception:
            continue
        out[c] = fv if np.isfinite(fv) else float("nan")
    return out


def opt_chop_score(snap: dict[str, float] | None) -> float | None:
    """Unit-free local score from one snapshot (not cross-sectional rank).

    Higher → more hostile microstructure for slow option holds.
    """
    if not snap:
        return None
    spread = float(snap.get("options_vw_spread") or 0.0)
    iv_div = abs(float(snap.get("options_iv_divergence") or 0.0))
    g_acc = abs(float(snap.get("options_gamma_accel") or 0.0))
    imb = float(snap.get("options_vw_imbalance") or 0.0)
    # Negative imbalance (offer-heavy) adds chop for long-call sleeves.
    return float(spread * 50.0 + iv_div * 10.0 + g_acc * 5.0 + max(0.0, -imb) * 2.0)

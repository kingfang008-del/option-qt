"""Second-level launch-slope features (research; NOT Rule-A / freeze).

Detects the steepest short-window price impulse — the "first candle" of a
spike — using causal rolling returns on 1s OHLCV. Optional money-flow confirm
via ``attach_sec_mf_features``.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from maga7.common.sec_mf import attach_sec_mf_features

NY = "America/New_York"


def _ret_k(close: np.ndarray, k: int) -> np.ndarray:
    """close[t]/close[t-k]-1; NaN for t < k."""
    n = len(close)
    k = max(1, int(k))
    out = np.full(n, np.nan, dtype=np.float64)
    if n > k:
        with np.errstate(divide="ignore", invalid="ignore"):
            out[k:] = close[k:] / close[:-k] - 1.0
    return out


def _rolling_max(x: np.ndarray, win: int) -> np.ndarray:
    """Causal rolling max over last ``win`` samples (includes current)."""
    win = max(1, int(win))
    n = len(x)
    out = np.full(n, np.nan, dtype=np.float64)
    if n == 0:
        return out
    # O(n*win) is fine for RTH morning (~2k bars).
    for i in range(n):
        lo = max(0, i + 1 - win)
        sl = x[lo : i + 1]
        sl = sl[np.isfinite(sl)]
        if sl.size:
            out[i] = float(sl.max())
    return out


def _rolling_min(x: np.ndarray, win: int) -> np.ndarray:
    win = max(1, int(win))
    n = len(x)
    out = np.full(n, np.nan, dtype=np.float64)
    if n == 0:
        return out
    for i in range(n):
        lo = max(0, i + 1 - win)
        sl = x[lo : i + 1]
        sl = sl[np.isfinite(sl)]
        if sl.size:
            out[i] = float(sl.min())
    return out


def attach_launch_slope_features(
    df: pd.DataFrame,
    *,
    slope_sec: int = 5,
    peak_lookback_sec: int = 60,
    vol_ma_sec: int = 300,
    prev_close: float | None = None,
    mf_window_sec: int | None = 60,
) -> pd.DataFrame:
    """Causal short-window return + local-peak flags + optional sec mf.

    Columns of interest:
      - ``ret_k``: k-second close return
      - ``slope_abs``: |ret_k|
      - ``is_local_max_up``: ret_k == rolling_max(ret_k) over peak_lookback (UP launch)
      - ``is_local_min_dn``: ret_k == rolling_min(ret_k) over peak_lookback (DN launch)
      - ``vol_z``, ``from_prev``, optional ``mf`` / streaks
    """
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.sort_values("timestamp").drop_duplicates("timestamp").copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"])
    if getattr(out["timestamp"].dt, "tz", None) is None:
        out["timestamp"] = out["timestamp"].dt.tz_localize(NY)
    else:
        out["timestamp"] = out["timestamp"].dt.tz_convert(NY)

    c = out["close"].to_numpy(dtype=np.float64)
    v = out["volume"].to_numpy(dtype=np.float64) if "volume" in out.columns else np.zeros(len(out))
    o = (
        out["open"].to_numpy(dtype=np.float64)
        if "open" in out.columns
        else c.copy()
    )

    k = max(1, int(slope_sec))
    ret = _ret_k(c, k)
    rmax = _rolling_max(ret, peak_lookback_sec)
    rmin = _rolling_min(ret, peak_lookback_sec)
    # equality with tiny tolerance for float noise
    eps = 1e-12
    is_max = np.isfinite(ret) & np.isfinite(rmax) & (ret >= rmax - eps) & (ret > 0)
    is_min = np.isfinite(ret) & np.isfinite(rmin) & (ret <= rmin + eps) & (ret < 0)

    # volume z vs recent mean
    vol_ma = np.full(len(v), np.nan, dtype=np.float64)
    w = max(5, int(vol_ma_sec))
    csum = np.cumsum(np.concatenate([[0.0], v]))
    for i in range(len(v)):
        lo = max(0, i + 1 - w)
        n = i + 1 - lo
        if n >= max(5, w // 10):
            vol_ma[i] = (csum[i + 1] - csum[lo]) / n
    with np.errstate(divide="ignore", invalid="ignore"):
        vol_z = v / vol_ma

    if prev_close is None or not np.isfinite(prev_close) or prev_close <= 0:
        prev_close = float(o[0]) if len(o) else float("nan")
    from_prev = c / float(prev_close) - 1.0

    out["slope_sec"] = int(k)
    out["ret_k"] = ret
    out["slope_abs"] = np.abs(ret)
    out["is_local_max_up"] = is_max
    out["is_local_min_dn"] = is_min
    out["vol_z"] = vol_z
    out["from_prev"] = from_prev

    if mf_window_sec is not None and int(mf_window_sec) > 0:
        cols = ["timestamp", "close"]
        for cname in ("open", "high", "low", "volume"):
            if cname in out.columns:
                cols.append(cname)
        mf_src = out[cols].copy()
        if "open" not in mf_src.columns:
            mf_src["open"] = mf_src["close"]
        if "high" not in mf_src.columns:
            mf_src["high"] = mf_src["close"]
        if "low" not in mf_src.columns:
            mf_src["low"] = mf_src["close"]
        if "volume" not in mf_src.columns:
            mf_src["volume"] = 0.0
        mf_df = attach_sec_mf_features(
            mf_src,
            mf_window_sec=int(mf_window_sec),
            vol_ma_sec=max(300, int(mf_window_sec) * 2),
            prev_close=float(prev_close),
        )
        if not mf_df.empty:
            out["mf"] = mf_df["mf"].to_numpy()
            out["streak_up"] = mf_df["streak_up"].to_numpy()
            out["streak_dn"] = mf_df["streak_dn"].to_numpy()
            out["net$"] = mf_df["net$"].to_numpy()
    return out


def launch_edges(
    feat: pd.DataFrame,
    *,
    direction: str,
    abs_ret_min: float,
    require_local_peak: bool = True,
) -> np.ndarray:
    """Indices where a launch impulse first crosses ``abs_ret_min``.

    UP: ret_k >= abs_ret_min (and optional local max)
    DN: ret_k <= -abs_ret_min (and optional local min)
    Rising-edge only (first touch after being below threshold).
    """
    if feat is None or feat.empty:
        return np.array([], dtype=np.int64)
    ret = feat["ret_k"].to_numpy(dtype=np.float64)
    thr = abs(float(abs_ret_min))
    d = str(direction).upper()
    if d == "UP":
        hit = np.isfinite(ret) & (ret >= thr)
        if require_local_peak and "is_local_max_up" in feat.columns:
            hit = hit & feat["is_local_max_up"].to_numpy(dtype=bool)
    elif d == "DN":
        hit = np.isfinite(ret) & (ret <= -thr)
        if require_local_peak and "is_local_min_dn" in feat.columns:
            hit = hit & feat["is_local_min_dn"].to_numpy(dtype=bool)
    else:
        return np.array([], dtype=np.int64)
    prev = np.concatenate([[False], hit[:-1]])
    return np.flatnonzero(hit & ~prev)


def launch_edges_multi(
    feat: pd.DataFrame,
    *,
    abs_ret_mins: Iterable[float],
    require_local_peak: bool = True,
) -> dict[tuple[str, float], np.ndarray]:
    """Map (dir, abs_ret_min) → edge indices."""
    out: dict[tuple[str, float], np.ndarray] = {}
    for thr in abs_ret_mins:
        for d in ("UP", "DN"):
            out[(d, float(thr))] = launch_edges(
                feat, direction=d, abs_ret_min=float(thr), require_local_peak=require_local_peak
            )
    return out

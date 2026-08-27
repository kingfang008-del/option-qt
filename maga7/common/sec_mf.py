"""Second-level money-flow features for morning research (NOT Rule-A / 1m clock).

Research-only: same net$ proxy as ``attach_mf_features``, but rolling windows
are in **seconds**. Do not wire into freeze scanner without a separate acceptance.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

NY = "America/New_York"


def _rolling_sum(x: np.ndarray, win: int) -> np.ndarray:
    win = max(1, int(win))
    if len(x) == 0:
        return x.astype(np.float64)
    c = np.cumsum(np.concatenate([[0.0], np.asarray(x, dtype=np.float64)]))
    out = np.full(len(x), np.nan, dtype=np.float64)
    if len(x) >= win:
        out[win - 1 :] = c[win:] - c[:-win]
    return out


def _rolling_mean(x: np.ndarray, win: int, *, min_periods: int | None = None) -> np.ndarray:
    win = max(1, int(win))
    mp = int(min_periods if min_periods is not None else max(1, win // 4))
    if len(x) == 0:
        return x.astype(np.float64)
    c = np.cumsum(np.concatenate([[0.0], np.asarray(x, dtype=np.float64)]))
    out = np.full(len(x), np.nan, dtype=np.float64)
    idx = np.arange(len(x))
    left = np.maximum(0, idx + 1 - win)
    n = idx + 1 - left
    s = c[idx + 1] - c[left]
    m = s / n
    m[n < mp] = np.nan
    return m


def _streak_sign(mf: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Consecutive seconds with mf>0 / mf<0 (NaN breaks streak)."""
    n = len(mf)
    up = np.zeros(n, dtype=np.int32)
    dn = np.zeros(n, dtype=np.int32)
    for i in range(n):
        v = mf[i]
        if not np.isfinite(v) or v == 0:
            continue
        if v > 0:
            up[i] = (up[i - 1] + 1) if i and up[i - 1] else 1
        else:
            dn[i] = (dn[i - 1] + 1) if i and dn[i - 1] else 1
    return up, dn


def attach_sec_mf_features(
    df: pd.DataFrame,
    *,
    mf_window_sec: int = 100,
    vol_ma_sec: int = 300,
    prev_close: float | None = None,
) -> pd.DataFrame:
    """Causal second-level mf / streak / from_prev / vol_z on 1s OHLCV."""
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.sort_values("timestamp").drop_duplicates("timestamp").copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"])
    if getattr(out["timestamp"].dt, "tz", None) is None:
        out["timestamp"] = out["timestamp"].dt.tz_localize(NY)
    else:
        out["timestamp"] = out["timestamp"].dt.tz_convert(NY)

    o = out["open"].to_numpy(dtype=np.float64) if "open" in out.columns else out["close"].to_numpy(dtype=np.float64)
    h = out["high"].to_numpy(dtype=np.float64) if "high" in out.columns else out["close"].to_numpy(dtype=np.float64)
    l = out["low"].to_numpy(dtype=np.float64) if "low" in out.columns else out["close"].to_numpy(dtype=np.float64)
    c = out["close"].to_numpy(dtype=np.float64)
    v = out["volume"].to_numpy(dtype=np.float64) if "volume" in out.columns else np.zeros(len(out))

    hl = h - l
    hl = np.where(hl == 0.0, np.nan, hl)
    buy = np.nan_to_num((c - l) / hl, nan=0.5) * v
    sell = np.nan_to_num((h - c) / hl, nan=0.5) * v
    net = (buy - sell) * c
    mf = _rolling_sum(net, mf_window_sec)
    su, sd = _streak_sign(mf)
    vol_ma = _rolling_mean(v, vol_ma_sec, min_periods=max(5, vol_ma_sec // 10))
    with np.errstate(divide="ignore", invalid="ignore"):
        vol_z = v / vol_ma

    if prev_close is None or not np.isfinite(prev_close) or prev_close <= 0:
        # fallback: first open of the loaded slice
        prev_close = float(o[0]) if len(o) else float("nan")
    from_prev = c / float(prev_close) - 1.0

    out["net$"] = net
    out["mf"] = mf
    out["streak_up"] = su
    out["streak_dn"] = sd
    out["vol_z"] = vol_z
    out["from_prev"] = from_prev
    out["mf_window_sec"] = int(mf_window_sec)
    return out


def forward_returns(close: np.ndarray, horizon_sec: int) -> np.ndarray:
    """close[t+H]/close[t]-1; NaN if not enough future bars (assumes ~1s grid)."""
    n = len(close)
    h = max(1, int(horizon_sec))
    out = np.full(n, np.nan, dtype=np.float64)
    if n > h:
        out[: n - h] = close[h:] / close[: n - h] - 1.0
    return out

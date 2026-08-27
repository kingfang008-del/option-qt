"""Causal second-level stock features for session entry research.

Inspired by ``preprocess/ask_bid/feature_merge_option_raw`` (volume_ratio,
vwap_diff, return_divergence, …) but computed on **1s closes/volumes** with
``last print ≤ t`` indexing — no left-labeled 1m bars.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from maga7.common.sec_mf import _rolling_mean, _rolling_sum, _streak_sign

NY = "America/New_York"


def prepare_day_arrays(day: pd.DataFrame) -> dict[str, np.ndarray]:
    """RTH 1s day → sorted arrays + causal rolling helpers."""
    d = day.sort_values("timestamp").drop_duplicates("timestamp").copy()
    ts = pd.to_datetime(d["timestamp"])
    if getattr(ts.dt, "tz", None) is None:
        ts = ts.dt.tz_localize(NY)
    else:
        ts = ts.dt.tz_convert(NY)
    o = d["open"].to_numpy(dtype=np.float64) if "open" in d.columns else d["close"].to_numpy(dtype=np.float64)
    h = d["high"].to_numpy(dtype=np.float64) if "high" in d.columns else d["close"].to_numpy(dtype=np.float64)
    l = d["low"].to_numpy(dtype=np.float64) if "low" in d.columns else d["close"].to_numpy(dtype=np.float64)
    c = d["close"].to_numpy(dtype=np.float64)
    v = d["volume"].to_numpy(dtype=np.float64) if "volume" in d.columns else np.zeros(len(d))
    v = np.nan_to_num(v, nan=0.0)
    c = np.where(np.isfinite(c) & (c > 0), c, np.nan)
    # session VWAP (causal cum)
    pv = np.nan_to_num(c, nan=0.0) * v
    cum_pv = np.cumsum(pv)
    cum_v = np.cumsum(v)
    with np.errstate(divide="ignore", invalid="ignore"):
        sess_vwap = np.where(cum_v > 0, cum_pv / cum_v, c)
    # money-flow proxy (same as sec_mf)
    hl = h - l
    hl = np.where(hl == 0.0, np.nan, hl)
    buy = np.nan_to_num((c - l) / hl, nan=0.5) * v
    sell = np.nan_to_num((h - c) / hl, nan=0.5) * v
    net = (buy - sell) * np.nan_to_num(c, nan=0.0)
    mf100 = _rolling_sum(net, 100)
    mf300 = _rolling_sum(net, 300)
    su100, sd100 = _streak_sign(mf100)
    vol_ma300 = _rolling_mean(v, 300, min_periods=30)
    with np.errstate(divide="ignore", invalid="ignore"):
        vol_z = v / vol_ma300
    # volume_ratio ≈ (sum vol last 60s) / (60 * mean vol last 300s)
    vol60 = np.nan_to_num(_rolling_sum(v, 60), nan=0.0)
    vol_mean300 = _rolling_mean(v, 300, min_periods=60)
    with np.errstate(divide="ignore", invalid="ignore"):
        volume_ratio_60 = np.where(
            np.isfinite(vol_mean300) & (vol_mean300 > 0),
            vol60 / (vol_mean300 * 60.0),
            np.nan,
        )
    return {
        "ts_ns": ts.astype("int64").to_numpy(),
        "ts_ny": ts.to_numpy(),
        "hm": (ts.dt.hour * 60 + ts.dt.minute).to_numpy(dtype=np.int32),
        "sec_of_day": (
            ts.dt.hour * 3600 + ts.dt.minute * 60 + ts.dt.second
        ).to_numpy(dtype=np.int32),
        "open": o,
        "high": h,
        "low": l,
        "close": c,
        "volume": v,
        "sess_vwap": sess_vwap,
        "pv": pv,
        "cum_pv": cum_pv,
        "cum_v": cum_v,
        "net": net,
        "mf100": mf100,
        "mf300": mf300,
        "streak_up": su100.astype(np.float64),
        "streak_dn": sd100.astype(np.float64),
        "vol_z": vol_z,
        "volume_ratio_60": volume_ratio_60,
        "vol60": vol60,
        "day_open": float(o[np.isfinite(o)][0]) if np.isfinite(o).any() else float("nan"),
    }


def rolling_vwap_at(
    arr: dict[str, np.ndarray],
    i: int,
    win_sec: int,
    *,
    min_vol: float = 0.0,
) -> float:
    """Causal trailing VWAP ending at index ``i`` over ``win_sec`` seconds.

    Uses prints with ``ts in (t_i - win_sec, t_i]``. Returns NaN if volume is
    insufficient or the first print in the window is >5s after the window start.
    """
    ts_ns = arr["ts_ns"]
    if i < 0 or i >= len(ts_ns):
        return float("nan")
    w = max(1, int(win_sec))
    t_ns = int(ts_ns[i])
    start_ns = t_ns - w * 1_000_000_000
    j = int(np.searchsorted(ts_ns, start_ns, side="right"))  # first ts > start
    if j > i:
        return float("nan")
    # First print in window should be near the window open.
    if abs(int(ts_ns[j]) - start_ns) > 5_000_000_000:
        return float("nan")
    cum_pv = arr["cum_pv"]
    cum_v = arr["cum_v"]
    pv0 = float(cum_pv[j - 1]) if j > 0 else 0.0
    v0 = float(cum_v[j - 1]) if j > 0 else 0.0
    vol = float(cum_v[i]) - v0
    if not np.isfinite(vol) or vol <= float(min_vol):
        return float("nan")
    pv = float(cum_pv[i]) - pv0
    if not np.isfinite(pv):
        return float("nan")
    return pv / vol


def rolling_vwap_series(
    arr: dict[str, np.ndarray],
    win_sec: int,
    *,
    min_vol: float = 0.0,
) -> np.ndarray:
    """Vectorized trailing VWAP for every index (NaN where unavailable)."""
    ts_ns = arr["ts_ns"]
    n = len(ts_ns)
    out = np.full(n, np.nan, dtype=np.float64)
    if n == 0:
        return out
    w = max(1, int(win_sec))
    cum_pv = arr["cum_pv"]
    cum_v = arr["cum_v"]
    starts = ts_ns - np.int64(w) * np.int64(1_000_000_000)
    j = np.searchsorted(ts_ns, starts, side="right")  # first ts > start
    ok = (j <= np.arange(n)) & (j < n)
    jj = np.where(ok, j, 0)
    ok &= np.abs(ts_ns[jj].astype(np.int64) - starts) <= np.int64(5_000_000_000)
    pv0 = np.zeros(n, dtype=np.float64)
    v0 = np.zeros(n, dtype=np.float64)
    prev = ok & (j > 0)
    if prev.any():
        jp = j[prev] - 1
        pv0[prev] = cum_pv[jp]
        v0[prev] = cum_v[jp]
    vol = cum_v - v0
    pv = cum_pv - pv0
    ok &= np.isfinite(vol) & (vol > float(min_vol)) & np.isfinite(pv)
    out[ok] = pv[ok] / vol[ok]
    return out


def _idx_at(ts_ns: np.ndarray, t_ns: int) -> int:
    return int(np.searchsorted(ts_ns, t_ns, side="right") - 1)


def _idx_lookback(ts_ns: np.ndarray, t_ns: int, lookback_sec: int) -> int:
    t0 = t_ns - int(lookback_sec) * 1_000_000_000
    return int(np.searchsorted(ts_ns, t0, side="right") - 1)


def features_at(arr: dict[str, np.ndarray], t: pd.Timestamp) -> dict[str, Any] | None:
    """Causal feature snapshot at ``t`` (last print ≤ t)."""
    ts_ns = arr["ts_ns"]
    if len(ts_ns) < 30:
        return None
    t_ns = int(pd.Timestamp(t).tz_convert(NY).value if pd.Timestamp(t).tzinfo else pd.Timestamp(t, tz=NY).value)
    i = _idx_at(ts_ns, t_ns)
    if i < 0:
        return None
    # require fresh print
    if abs(int(ts_ns[i]) - t_ns) > 5_000_000_000:
        return None
    c = arr["close"]
    vwap = arr["sess_vwap"]
    px = float(c[i])
    if not np.isfinite(px) or px <= 0:
        return None
    out: dict[str, Any] = {
        "px": px,
        "vwap": float(vwap[i]) if np.isfinite(vwap[i]) else np.nan,
        "from_open": px / arr["day_open"] - 1.0 if arr["day_open"] > 0 else np.nan,
        "vol_z": float(arr["vol_z"][i]),
        "volume_ratio_60": float(arr["volume_ratio_60"][i]),
        "mf100": float(arr["mf100"][i]),
        "mf300": float(arr["mf300"][i]),
        "streak_up": int(arr["streak_up"][i]),
        "streak_dn": int(arr["streak_dn"][i]),
    }
    out["vwap_diff"] = (px / out["vwap"] - 1.0) if out["vwap"] and out["vwap"] > 0 else np.nan

    for w in (15, 30, 60, 120):
        j = _idx_lookback(ts_ns, t_ns, w)
        if j < 0 or abs(int(ts_ns[j]) - (t_ns - w * 1_000_000_000)) > 5_000_000_000:
            out[f"ret_{w}"] = np.nan
            out[f"vwap_ret_{w}"] = np.nan
            out[f"range_{w}"] = np.nan
            out[f"ret_div_{w}"] = np.nan
            continue
        a = float(c[j])
        out[f"ret_{w}"] = px / a - 1.0 if a > 0 else np.nan
        va, vb = float(vwap[j]), float(vwap[i])
        out[f"vwap_ret_{w}"] = vb / va - 1.0 if va > 0 and vb > 0 else np.nan
        hi = float(np.nanmax(arr["high"][j : i + 1]))
        lo = float(np.nanmin(arr["low"][j : i + 1]))
        out[f"range_{w}"] = (hi - lo) / px if px > 0 else np.nan
        r = out[f"ret_{w}"]
        vr = out[f"vwap_ret_{w}"]
        out[f"ret_div_{w}"] = (r - vr) if np.isfinite(r) and np.isfinite(vr) else np.nan

    return out

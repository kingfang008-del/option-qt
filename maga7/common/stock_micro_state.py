"""Causal stock micro-state: local slope (velocity) + accel + SNR.

Fits underlying close (not option premium). Dual-window confluence:
short-window slope rising-edge AND long-window slope same sign, with
SNR = |slope| / residual_sigma high enough to reject whipsaw.

PARKED alternative: ``smooth_regress_1s`` (OLS smooth-regime only) — do not
use as primary research path.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

NY = "America/New_York"


@dataclass(frozen=True)
class MicroStateConfig:
    short_sec: int = 20
    long_sec: int = 60
    stride_sec: int = 5
    min_snr: float = 2.0
    min_slope_bp_per_min: float = 1.5
    require_accel: bool = True
    scan_start: str = "09:30"
    scan_end: str = "10:15"
    cooldown_sec: int = 120


def _hhmm(s: str) -> tuple[int, int]:
    h, m = str(s).split(":")
    return int(h), int(m)


def prepare_day(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.sort_values("timestamp").drop_duplicates("timestamp").copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"])
    if getattr(out["timestamp"].dt, "tz", None) is None:
        out["timestamp"] = out["timestamp"].dt.tz_localize(NY)
    else:
        out["timestamp"] = out["timestamp"].dt.tz_convert(NY)
    return out.reset_index(drop=True)


def _ols_slope_resid(c: np.ndarray) -> tuple[float, float, float, float] | None:
    """Return (slope_per_sec, resid_std, r2, tstat) on close vs index."""
    n = len(c)
    if n < 6:
        return None
    c = np.asarray(c, dtype=np.float64)
    if not np.isfinite(c).all() or np.any(c <= 0):
        return None
    t = np.arange(n, dtype=np.float64)
    t_m, c_m = t.mean(), c.mean()
    dt, dc = t - t_m, c - c_m
    den = float(np.dot(dt, dt))
    if den < 1e-12:
        return None
    b = float(np.dot(dt, dc) / den)
    resid = c - (c_m - b * t_m + b * t)
    ss_res = float(np.dot(resid, resid))
    ss_tot = float(np.dot(dc, dc))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-18 else 0.0
    resid_std = float(np.sqrt(ss_res / max(n - 2, 1)))
    tstat = abs(b) * float(np.sqrt(den)) / resid_std if resid_std > 1e-12 else 1e6
    return b, resid_std, float(r2), float(tstat)


def window_state(c: np.ndarray) -> dict[str, float] | None:
    fit = _ols_slope_resid(c)
    if fit is None:
        return None
    b, resid_std, r2, tstat = fit
    px = float(c[-1])
    if px <= 0:
        return None
    slope_bp_per_min = (b / px) * 60.0 * 1e4
    resid_bp = (resid_std / px) * 1e4
    return {
        "slope_bp_per_min": float(slope_bp_per_min),
        "resid_bp": float(resid_bp),
        "snr": float(tstat),  # slope t-stat as SNR proxy
        "r2": float(r2),
        "px": px,
    }


def detect_micro_edges(
    stock_day: pd.DataFrame,
    *,
    symbol: str,
    date: str,
    cfg: MicroStateConfig | None = None,
) -> list[dict]:
    """Rising-edge micro momentum with long-window confluence + SNR gate."""
    cfg = cfg or MicroStateConfig()
    day = prepare_day(stock_day)
    if day.empty or "close" not in day.columns:
        return []
    ts = pd.DatetimeIndex(day["timestamp"])
    close = day["close"].to_numpy(dtype=np.float64)
    ws, wl = int(cfg.short_sec), int(cfg.long_sec)
    stride = max(1, int(cfg.stride_sec))
    sh, sm = _hhmm(cfg.scan_start)
    eh, em = _hhmm(cfg.scan_end)
    t_lo = pd.Timestamp(f"{date} {sh:02d}:{sm:02d}:00", tz=NY)
    t_hi = pd.Timestamp(f"{date} {eh:02d}:{em:02d}:00", tz=NY)

    events: list[dict] = []
    last_fire: dict[str, pd.Timestamp] = {}
    prev_slope_s: float | None = None
    i = max(ws, wl) - 1
    while i < len(close):
        t = ts[i]
        if t < t_lo:
            i += stride
            continue
        if t >= t_hi:
            break
        short = window_state(close[i - ws + 1 : i + 1])
        long = window_state(close[i - wl + 1 : i + 1])
        if short is None or long is None:
            prev_slope_s = None
            i += stride
            continue
        s = short["slope_bp_per_min"]
        L = long["slope_bp_per_min"]
        accel = 0.0 if prev_slope_s is None else float(s - prev_slope_s)

        if prev_slope_s is not None:
            for direction in ("UP", "DN"):
                if direction == "UP":
                    edge = prev_slope_s <= 0 < s
                    same_long = L > 0
                    accel_ok = (accel > 0) if cfg.require_accel else True
                else:
                    edge = prev_slope_s >= 0 > s
                    same_long = L < 0
                    accel_ok = (accel < 0) if cfg.require_accel else True
                if not (edge and same_long and accel_ok):
                    continue
                if abs(s) < cfg.min_slope_bp_per_min:
                    continue
                if short["snr"] < cfg.min_snr:
                    continue
                prev = last_fire.get(direction)
                if prev is not None and (t - prev).total_seconds() < cfg.cooldown_sec:
                    continue
                events.append(
                    {
                        "date": str(date),
                        "symbol": str(symbol).upper(),
                        "dir": direction,
                        "ts": t,
                        "entry_px": short["px"],
                        "short_sec": ws,
                        "long_sec": wl,
                        "slope_s": float(s),
                        "slope_l": float(L),
                        "accel": float(accel),
                        "snr": float(short["snr"]),
                        "resid_bp": float(short["resid_bp"]),
                        "r2_s": float(short["r2"]),
                    }
                )
                last_fire[direction] = t
        prev_slope_s = s
        i += stride
    return events

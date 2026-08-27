"""Causal 1s local-regression smooth-regime detector — PARKED.

Superseded by ``stock_micro_state`` (velocity/accel + SNR + dual-window) +
``scan_micro_state_quote_scalp``. Do not treat this module as the primary path;
option-premium curve fitting is explicitly out of scope.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

NY = "America/New_York"


@dataclass(frozen=True)
class SmoothRegressConfig:
    win_sec: int = 60
    stride_sec: int = 10
    min_r2: float = 0.70
    max_resid_bp: float = 3.0  # residual std / price * 1e4
    min_slope_bp_per_min: float = 2.0  # |b| in bp per minute
    min_path_eff: float = 0.35
    scan_start: str = "09:45"
    scan_end: str = "15:30"
    cooldown_sec: int = 300
    require_rising_edge: bool = True


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


def _path_eff(c: np.ndarray) -> float:
    if len(c) < 3 or c[0] <= 0:
        return 0.0
    net = abs(c[-1] / c[0] - 1.0)
    rets = np.diff(c) / c[:-1]
    sumabs = float(np.abs(rets).sum()) or 1e-12
    return float(net / sumabs)


def fit_window(c: np.ndarray) -> dict[str, float] | None:
    """OLS close = a + b·t on indices 0..n-1. Causal window only."""
    n = len(c)
    if n < 8:
        return None
    c = np.asarray(c, dtype=np.float64)
    if not np.isfinite(c).all() or np.any(c <= 0):
        return None
    t = np.arange(n, dtype=np.float64)
    t_mean = t.mean()
    c_mean = c.mean()
    dt = t - t_mean
    dc = c - c_mean
    den = float(np.dot(dt, dt))
    if den < 1e-12:
        return None
    b = float(np.dot(dt, dc) / den)
    a = float(c_mean - b * t_mean)
    pred = a + b * t
    resid = c - pred
    ss_res = float(np.dot(resid, resid))
    ss_tot = float(np.dot(dc, dc))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-18 else 0.0
    resid_std = float(np.sqrt(ss_res / max(n - 2, 1)))
    px = float(c[-1])
    # slope in bp per minute (assuming 1 sample = 1 second)
    slope_bp_per_min = (b / px) * 60.0 * 1e4 if px > 0 else 0.0
    resid_bp = (resid_std / px) * 1e4 if px > 0 else 1e9
    return {
        "r2": float(r2),
        "slope": float(b),
        "slope_bp_per_min": float(slope_bp_per_min),
        "resid_bp": float(resid_bp),
        "path_eff": _path_eff(c),
        "ret_w": float(c[-1] / c[0] - 1.0),
        "px": px,
    }


def is_smooth(feat: dict[str, float], cfg: SmoothRegressConfig) -> bool:
    if feat["r2"] < cfg.min_r2:
        return False
    if feat["resid_bp"] > cfg.max_resid_bp:
        return False
    if abs(feat["slope_bp_per_min"]) < cfg.min_slope_bp_per_min:
        return False
    if feat["path_eff"] < cfg.min_path_eff:
        return False
    return True


def detect_day_edges(
    stock_day: pd.DataFrame,
    *,
    symbol: str,
    date: str,
    cfg: SmoothRegressConfig | None = None,
) -> list[dict]:
    """Return rising-edge smooth-regime entries for one symbol-day."""
    cfg = cfg or SmoothRegressConfig()
    day = prepare_day(stock_day)
    if day.empty or "close" not in day.columns:
        return []
    ts = pd.DatetimeIndex(day["timestamp"])
    close = day["close"].to_numpy(dtype=np.float64)
    w = max(8, int(cfg.win_sec))
    stride = max(1, int(cfg.stride_sec))
    sh, sm = _hhmm(cfg.scan_start)
    eh, em = _hhmm(cfg.scan_end)
    t_lo = pd.Timestamp(f"{date} {sh:02d}:{sm:02d}:00", tz=NY)
    t_hi = pd.Timestamp(f"{date} {eh:02d}:{em:02d}:00", tz=NY)

    events: list[dict] = []
    last_fire: dict[str, pd.Timestamp] = {}
    prev_smooth = False

    i = w - 1
    while i < len(close):
        t = ts[i]
        if t < t_lo:
            i += stride
            continue
        if t >= t_hi:
            break
        feat = fit_window(close[i - w + 1 : i + 1])
        if feat is None:
            prev_smooth = False
            i += stride
            continue
        smooth = is_smooth(feat, cfg)
        rising = smooth and (not prev_smooth if cfg.require_rising_edge else smooth)
        prev_smooth = smooth
        if not rising:
            i += stride
            continue
        direction = "UP" if feat["slope_bp_per_min"] > 0 else "DN"
        prev = last_fire.get(direction)
        if prev is not None and (t - prev).total_seconds() < cfg.cooldown_sec:
            i += stride
            continue
        events.append(
            {
                "date": str(date),
                "symbol": str(symbol).upper(),
                "dir": direction,
                "ts": t,
                "entry_px": feat["px"],
                "win_sec": int(w),
                "r2": feat["r2"],
                "slope_bp_per_min": feat["slope_bp_per_min"],
                "resid_bp": feat["resid_bp"],
                "path_eff": feat["path_eff"],
                "ret_w": feat["ret_w"],
            }
        )
        last_fire[direction] = t
        i += stride
    return events


def detect_day_edges_grid(
    stock_day: pd.DataFrame,
    *,
    symbol: str,
    date: str,
    win_secs: Iterable[int],
    min_r2s: Iterable[float],
    base: SmoothRegressConfig | None = None,
) -> list[dict]:
    """Run a small detector grid; tag each event with its cell params."""
    base = base or SmoothRegressConfig()
    out: list[dict] = []
    for w in win_secs:
        for r2 in min_r2s:
            cfg = SmoothRegressConfig(
                win_sec=int(w),
                stride_sec=base.stride_sec,
                min_r2=float(r2),
                max_resid_bp=base.max_resid_bp,
                min_slope_bp_per_min=base.min_slope_bp_per_min,
                min_path_eff=base.min_path_eff,
                scan_start=base.scan_start,
                scan_end=base.scan_end,
                cooldown_sec=base.cooldown_sec,
                require_rising_edge=base.require_rising_edge,
            )
            for ev in detect_day_edges(stock_day, symbol=symbol, date=date, cfg=cfg):
                ev = dict(ev)
                ev["min_r2"] = float(r2)
                ev["cell"] = f"w{int(w)}_r2{r2:.2f}"
                out.append(ev)
    return out

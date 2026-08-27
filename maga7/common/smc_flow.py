"""SMC / ICT-lite structure + order-flow *proxies* on stock 1s OHLCV.

No aggressor tape / CVD / footprint in current Polygon second aggs.
Proxies used here:
  - Structure: swing sweep (liquidity grab) and BOS (close through swing)
  - Displacement: short-window signed return
  - Flow: down-tick volume share (Δclose-signed) and CLV money-flow (sec_mf)

Research-only. Do not wire into freeze scanner without dual-window quote PASS.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from maga7.common.sec_mf import attach_sec_mf_features

NY = "America/New_York"


@dataclass(frozen=True)
class SmcFlowArm:
    morph: str  # sweep_rev_dn | bos_disp_dn
    direction: str
    arm_i: int
    stock_ret_disp: float
    dn_vol_share: float
    mf: float
    streak_dn: int
    swing_level: float


def _to_ny_series(ts: pd.Series) -> pd.Series:
    t = pd.to_datetime(ts)
    if getattr(t.dt, "tz", None) is None:
        return t.dt.tz_localize(NY)
    return t.dt.tz_convert(NY)


def prepare_smc_flow_day(bars: pd.DataFrame | None, *, mf_window_sec: int = 100) -> dict[str, Any] | None:
    """Build causal arrays for one symbol-day. Returns None if unusable."""
    if bars is None or bars.empty:
        return None
    need = {"timestamp", "open", "high", "low", "close", "volume"}
    if not need.issubset(set(bars.columns)):
        return None
    feat = attach_sec_mf_features(bars, mf_window_sec=int(mf_window_sec))
    if feat is None or feat.empty:
        return None
    feat = feat.sort_values("timestamp").drop_duplicates("timestamp")
    ts = _to_ny_series(feat["timestamp"])
    ts_ns = ts.map(lambda x: int(pd.Timestamp(x).value)).to_numpy(dtype=np.int64)
    o = pd.to_numeric(feat["open"], errors="coerce").to_numpy(dtype=np.float64)
    h = pd.to_numeric(feat["high"], errors="coerce").to_numpy(dtype=np.float64)
    l = pd.to_numeric(feat["low"], errors="coerce").to_numpy(dtype=np.float64)
    c = pd.to_numeric(feat["close"], errors="coerce").to_numpy(dtype=np.float64)
    v = pd.to_numeric(feat["volume"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    mf = pd.to_numeric(feat["mf"], errors="coerce").to_numpy(dtype=np.float64)
    streak_dn = feat["streak_dn"].to_numpy(dtype=np.int32)
    vol_z = (
        pd.to_numeric(feat["vol_z"], errors="coerce").to_numpy(dtype=np.float64)
        if "vol_z" in feat.columns
        else np.full(len(c), np.nan, dtype=np.float64)
    )
    n = len(c)
    if n < 60:
        return None
    # Per-second signed volume (proxy for aggression): attribute bar volume to Δclose.
    d = np.diff(c, prepend=c[0])
    up_v = np.where(d > 0, v, 0.0)
    dn_v = np.where(d < 0, v, 0.0)
    return {
        "ts_ns": ts_ns,
        "open": o,
        "high": h,
        "low": l,
        "close": c,
        "volume": v,
        "up_v": up_v,
        "dn_v": dn_v,
        "mf": mf,
        "streak_dn": streak_dn,
        "vol_z": vol_z,
        "n": n,
    }


def _idx_at_or_before(ts_ns: np.ndarray, t_ns: int) -> int | None:
    i = int(np.searchsorted(ts_ns, t_ns, side="right") - 1)
    if i < 0:
        return None
    return i


def dn_vol_share_at(
    arrays: dict[str, Any],
    *,
    i: int,
    window_sec: int,
    min_seconds: int = 20,
) -> float | None:
    """Down-tick volume / (up+down) over (i-window, i]."""
    ts_ns = arrays["ts_ns"]
    up_v = arrays["up_v"]
    dn_v = arrays["dn_v"]
    if i < 1:
        return None
    now_ns = int(ts_ns[i])
    start_ns = now_ns - int(window_sec) * 1_000_000_000
    left = int(np.searchsorted(ts_ns, start_ns, side="left"))
    if i - left < 2:
        return None
    span = (ts_ns[i] - ts_ns[left]) / 1e9
    if span < float(min_seconds):
        return None
    u = float(up_v[left : i + 1].sum())
    d = float(dn_v[left : i + 1].sum())
    denom = u + d
    if denom <= 0:
        return None
    return d / denom


def detect_smc_flow_dn(
    arrays: dict[str, Any],
    *,
    i: int,
    morph: str,
    swing_sec: int,
    disp_sec: int,
    disp_thr: float,
    flow_sec: int,
    min_dn_vol_share: float | None,
    min_streak_dn: int,
    require_mf_neg: bool,
) -> SmcFlowArm | None:
    """Causal DN arm at bar index ``i`` (inclusive)."""
    n = int(arrays["n"])
    c = arrays["close"]
    h = arrays["high"]
    l = arrays["low"]
    ts_ns = arrays["ts_ns"]
    if i < max(int(swing_sec), int(disp_sec), int(flow_sec), 30) or i >= n:
        return None

    # Prior structure excludes the last ``disp_sec`` bars (displacement / reclaim window).
    gap = max(5, int(disp_sec))
    left = max(0, i - int(swing_sec))
    right_excl = max(left + 2, i - gap + 1)
    if right_excl <= left + 1:
        return None
    prior_hi = float(np.nanmax(h[left:right_excl]))
    prior_lo = float(np.nanmin(l[left:right_excl]))
    if not (np.isfinite(prior_hi) and np.isfinite(prior_lo) and prior_hi > prior_lo):
        return None

    # Displacement: return over last disp_sec (approx by timestamp lookback).
    t0_ns = int(ts_ns[i]) - int(disp_sec) * 1_000_000_000
    j0 = _idx_at_or_before(ts_ns, t0_ns)
    if j0 is None or j0 >= i:
        return None
    c0 = float(c[j0])
    c1 = float(c[i])
    if not (np.isfinite(c0) and np.isfinite(c1) and c0 > 0):
        return None
    ret = c1 / c0 - 1.0
    if ret > -float(disp_thr):
        return None

    morph_u = str(morph).strip().lower()
    swing_level = float("nan")
    if morph_u == "sweep_rev_dn":
        # Liquidity grab above prior swing high inside the gap window, then close back below.
        win_hi = float(np.nanmax(h[right_excl : i + 1]))
        if not (np.isfinite(win_hi) and win_hi > prior_hi and c1 < prior_hi):
            return None
        swing_level = prior_hi
    elif morph_u == "bos_disp_dn":
        # Break of structure: close through prior swing low with displacement.
        if not (c1 < prior_lo):
            return None
        swing_level = prior_lo
    else:
        return None

    share = dn_vol_share_at(arrays, i=i, window_sec=int(flow_sec))
    if min_dn_vol_share is not None:
        if share is None or share < float(min_dn_vol_share):
            return None
    mf_v = float(arrays["mf"][i]) if np.isfinite(arrays["mf"][i]) else float("nan")
    sd = int(arrays["streak_dn"][i])
    if require_mf_neg and not (np.isfinite(mf_v) and mf_v < 0):
        return None
    if int(min_streak_dn) > 0 and sd < int(min_streak_dn):
        return None

    return SmcFlowArm(
        morph=morph_u,
        direction="DN",
        arm_i=int(i),
        stock_ret_disp=float(ret),
        dn_vol_share=float(share) if share is not None else float("nan"),
        mf=mf_v,
        streak_dn=sd,
        swing_level=float(swing_level),
    )


def first_smc_flow_dn_in_window(
    arrays: dict[str, Any],
    *,
    t_start,
    t_end,
    morph: str,
    swing_sec: int,
    disp_sec: int,
    disp_thr: float,
    flow_sec: int,
    min_dn_vol_share: float | None,
    min_streak_dn: int,
    require_mf_neg: bool,
    stride_sec: int = 15,
) -> tuple[pd.Timestamp, SmcFlowArm] | None:
    """Stride scan; return first arm in [t_start, t_end)."""
    ts_ns = arrays["ts_ns"]
    t0 = pd.Timestamp(t_start)
    if t0.tzinfo is None:
        t0 = t0.tz_localize(NY)
    else:
        t0 = t0.tz_convert(NY)
    t1 = pd.Timestamp(t_end)
    if t1.tzinfo is None:
        t1 = t1.tz_localize(NY)
    else:
        t1 = t1.tz_convert(NY)
    stride = pd.Timedelta(seconds=int(stride_sec))
    t = t0
    while t < t1:
        i = _idx_at_or_before(ts_ns, int(t.value))
        if i is not None:
            arm = detect_smc_flow_dn(
                arrays,
                i=i,
                morph=morph,
                swing_sec=swing_sec,
                disp_sec=disp_sec,
                disp_thr=disp_thr,
                flow_sec=flow_sec,
                min_dn_vol_share=min_dn_vol_share,
                min_streak_dn=min_streak_dn,
                require_mf_neg=require_mf_neg,
            )
            if arm is not None:
                return t, arm
        t += stride
    return None

"""Underlying-level option flow proxies from OPRA trade prints.

Supports:
  - 1s aggregates (``v``): ``{root}/{SYM}/{SYM}_{date}.parquet`` with o/h/l/c/v
  - tick prints (``size``): ``S3_DATA_KIND=tick`` layout with price/size/sip_timestamp

No aggressor side — proxies are put/call volume share and put volume z-score.
Research-only until trades + quote FillSpec dual PASS.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

NY = "America/New_York"
_CP_RE = re.compile(r"^[A-Z]+(\d{6})([CP])(\d{8})$")
DEFAULT_TICK_ROOT = Path("/mnt/s990/new_option_data_s3_tick")


@dataclass(frozen=True)
class OptionFlowArm:
    direction: str
    arm_i: int
    put_share: float
    put_vol_z: float
    put_v: float
    call_v: float
    stock_ret_lb: float


def option_right(ticker: str) -> str | None:
    s = str(ticker).replace("O:", "").strip().upper()
    m = _CP_RE.match(s)
    if not m:
        return None
    return m.group(2)


def load_option_tick_day(tick_root: Path | str | None, symbol: str, date: str) -> pd.DataFrame | None:
    """Load one symbol-day of option tick prints (or None)."""
    if tick_root is None:
        return None
    root = Path(tick_root).expanduser()
    p = root / symbol / f"{symbol}_{date}.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    if df.empty or "timestamp" not in df.columns:
        return None
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    if getattr(df["timestamp"].dt, "tz", None) is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize(NY)
    else:
        df["timestamp"] = df["timestamp"].dt.tz_convert(NY)
    return df


def tick_dates(tick_root: Path | str, symbol: str | None = None) -> list[str]:
    """Sorted YYYY-MM-DD dates present under tick root."""
    root = Path(tick_root).expanduser()
    if not root.exists():
        return []
    dates: set[str] = set()
    syms = [symbol] if symbol else [p.name for p in root.iterdir() if p.is_dir()]
    for sym in syms:
        for f in (root / sym).glob(f"{sym}_*.parquet"):
            dates.add(f.stem.split("_", 1)[1])
    return sorted(dates)


def prepare_option_flow_day(tday: pd.DataFrame | None) -> dict[str, Any] | None:
    """Aggregate all contracts → 1s put/call volume series (causal features).

    Accepts 1s ``v`` or tick ``size`` (summed into each second bucket).
    """
    if tday is None or tday.empty:
        return None
    if "ticker" not in tday.columns or "timestamp" not in tday.columns:
        return None
    if "v" in tday.columns:
        vol_col = "v"
        source = "1s_agg"
    elif "size" in tday.columns:
        vol_col = "size"
        source = "tick"
    else:
        return None
    df = tday.copy()
    rights = df["ticker"].astype(str).map(option_right)
    df = df.assign(_right=rights)
    df = df[df["_right"].isin(["P", "C"])]
    if df.empty:
        return None
    ts = pd.to_datetime(df["timestamp"])
    if getattr(ts.dt, "tz", None) is None:
        ts = ts.dt.tz_localize(NY)
    else:
        ts = ts.dt.tz_convert(NY)
    df = df.assign(timestamp=ts, sec=ts.dt.floor("s"))
    v = pd.to_numeric(df[vol_col], errors="coerce").fillna(0.0)
    df = df.assign(v=v)
    g = (
        df.groupby(["sec", "_right"], sort=True)["v"]
        .sum()
        .unstack(fill_value=0.0)
        .rename(columns={"P": "put_v", "C": "call_v"})
    )
    if "put_v" not in g.columns:
        g["put_v"] = 0.0
    if "call_v" not in g.columns:
        g["call_v"] = 0.0
    g = g[["put_v", "call_v"]].sort_index()
    # densify to continuous seconds between first/last print
    if g.empty:
        return None
    full = pd.date_range(g.index.min(), g.index.max(), freq="s", tz=NY)
    g = g.reindex(full, fill_value=0.0)
    put = g["put_v"].to_numpy(dtype=np.float64)
    call = g["call_v"].to_numpy(dtype=np.float64)
    ts_ns = g.index.map(lambda x: int(pd.Timestamp(x).value)).to_numpy(dtype=np.int64)
    return {
        "ts_ns": ts_ns,
        "put_v": put,
        "call_v": call,
        "n": len(put),
        "source": source,
    }


def _idx_at_or_before(ts_ns: np.ndarray, t_ns: int) -> int | None:
    i = int(np.searchsorted(ts_ns, t_ns, side="right") - 1)
    return i if i >= 0 else None


def _window_sums(
    arrays: dict[str, Any], *, i: int, window_sec: int
) -> tuple[float, float] | None:
    ts_ns = arrays["ts_ns"]
    if i < 0 or i >= int(arrays["n"]):
        return None
    start_ns = int(ts_ns[i]) - int(window_sec) * 1_000_000_000
    left = int(np.searchsorted(ts_ns, start_ns, side="left"))
    if i - left < 2:
        return None
    pv = float(arrays["put_v"][left : i + 1].sum())
    cv = float(arrays["call_v"][left : i + 1].sum())
    return pv, cv


def put_flow_features_at(
    arrays: dict[str, Any],
    *,
    i: int,
    window_sec: int,
    baseline_sec: int = 300,
) -> tuple[float, float, float, float] | None:
    """Return (put_share, put_vol_z, put_v, call_v) over window ending at i."""
    w = _window_sums(arrays, i=i, window_sec=window_sec)
    if w is None:
        return None
    pv, cv = w
    denom = pv + cv
    if denom <= 0:
        return None
    share = pv / denom
    # baseline mean put volume per second over longer lookback
    base = _window_sums(arrays, i=i, window_sec=max(int(baseline_sec), int(window_sec) * 2))
    if base is None:
        return None
    bp, _bc = base
    base_win = max(int(baseline_sec), int(window_sec) * 2)
    base_rate = bp / float(base_win)
    win_rate = pv / float(max(1, int(window_sec)))
    if base_rate <= 1e-9:
        z = 0.0 if win_rate <= 0 else 10.0
    else:
        z = win_rate / base_rate
    return float(share), float(z), float(pv), float(cv)


def detect_put_flow_dn(
    flow: dict[str, Any],
    *,
    i: int,
    window_sec: int,
    min_put_share: float,
    min_put_vol_z: float,
    min_put_v: float,
    stock_ret_lb: float | None,
    max_stock_ret: float | None,
) -> OptionFlowArm | None:
    """DN arm: put-dominated print flow (+ optional stock dump confirm)."""
    feat = put_flow_features_at(flow, i=i, window_sec=int(window_sec))
    if feat is None:
        return None
    share, z, pv, cv = feat
    if share < float(min_put_share):
        return None
    if z < float(min_put_vol_z):
        return None
    if pv < float(min_put_v):
        return None
    sr = float(stock_ret_lb) if stock_ret_lb is not None and np.isfinite(stock_ret_lb) else float("nan")
    if max_stock_ret is not None:
        if not np.isfinite(sr) or sr > float(max_stock_ret):
            return None
    return OptionFlowArm(
        direction="DN",
        arm_i=int(i),
        put_share=float(share),
        put_vol_z=float(z),
        put_v=float(pv),
        call_v=float(cv),
        stock_ret_lb=sr,
    )


def _stock_ret_at(
    stock_ts_ns: np.ndarray | None,
    stock_px: np.ndarray | None,
    *,
    t_ns: int,
    stock_lb_sec: int,
) -> float | None:
    if stock_ts_ns is None or stock_px is None or not len(stock_ts_ns):
        return None
    j1 = _idx_at_or_before(stock_ts_ns, t_ns)
    j0 = _idx_at_or_before(stock_ts_ns, t_ns - int(stock_lb_sec) * 1_000_000_000)
    if j0 is None or j1 is None or j1 <= j0:
        return None
    a, b = float(stock_px[j0]), float(stock_px[j1])
    if a > 0 and b > 0 and np.isfinite(a) and np.isfinite(b):
        return b / a - 1.0
    return None


def iter_put_flow_dn_in_window(
    flow: dict[str, Any],
    *,
    t_start,
    t_end,
    window_sec: int,
    min_put_share: float,
    min_put_vol_z: float,
    min_put_v: float,
    stock_ts_ns: np.ndarray | None,
    stock_px: np.ndarray | None,
    stock_lb_sec: int,
    max_stock_ret: float | None,
    stride_sec: int = 5,
    rearm_gap_sec: int = 60,
    max_arms: int | None = None,
    fire_mode: str = "rising",
    pulse_z_delta: float = 0.5,
    pulse_v_mult: float = 1.25,
) -> list[tuple[pd.Timestamp, OptionFlowArm]]:
    """Second-stride opportunistic arms.

    fire_mode:
      - hold: fire while gate on (subject to rearm_gap) — dilutes edge
      - rising: fire only on False→True gate edge (new episode)
      - pulse: rising, or still-on but put_vol_z / put_v makes a new impulse
      - first: at most one arm (legacy)
    """
    mode = str(fire_mode or "rising").strip().lower()
    if mode not in {"hold", "rising", "pulse", "first"}:
        mode = "rising"
    if mode == "first":
        max_arms = 1
        rearm_gap_sec = 10**9

    ts_ns = flow["ts_ns"]
    t0 = pd.Timestamp(t_start)
    t0 = t0.tz_localize(NY) if t0.tzinfo is None else t0.tz_convert(NY)
    t1 = pd.Timestamp(t_end)
    t1 = t1.tz_localize(NY) if t1.tzinfo is None else t1.tz_convert(NY)
    stride = pd.Timedelta(seconds=max(1, int(stride_sec)))
    gap = pd.Timedelta(seconds=max(0, int(rearm_gap_sec)))
    out: list[tuple[pd.Timestamp, OptionFlowArm]] = []
    t = t0
    next_ok = t0
    prev_on = False
    last_z = float("-inf")
    last_v = 0.0
    while t < t1:
        i = _idx_at_or_before(ts_ns, int(t.value))
        arm = None
        if i is not None:
            sr = _stock_ret_at(
                stock_ts_ns,
                stock_px,
                t_ns=int(t.value),
                stock_lb_sec=int(stock_lb_sec),
            )
            arm = detect_put_flow_dn(
                flow,
                i=i,
                window_sec=window_sec,
                min_put_share=min_put_share,
                min_put_vol_z=min_put_vol_z,
                min_put_v=min_put_v,
                stock_ret_lb=sr,
                max_stock_ret=max_stock_ret,
            )
        on = arm is not None
        fire = False
        if on and t >= next_ok:
            if mode in {"hold", "first"}:
                fire = True
            elif mode == "rising":
                fire = not prev_on
            else:  # pulse
                if not prev_on:
                    fire = True
                else:
                    assert arm is not None
                    fire = (
                        arm.put_vol_z >= last_z + float(pulse_z_delta)
                        or arm.put_v >= last_v * float(pulse_v_mult)
                    )
        if fire and arm is not None:
            out.append((t, arm))
            next_ok = t + gap
            last_z = float(arm.put_vol_z)
            last_v = float(arm.put_v)
            if max_arms is not None and len(out) >= int(max_arms):
                break
        prev_on = on
        t += stride
    return out


def first_put_flow_dn_in_window(
    flow: dict[str, Any],
    *,
    t_start,
    t_end,
    window_sec: int,
    min_put_share: float,
    min_put_vol_z: float,
    min_put_v: float,
    stock_ts_ns: np.ndarray | None,
    stock_px: np.ndarray | None,
    stock_lb_sec: int,
    max_stock_ret: float | None,
    stride_sec: int = 15,
) -> tuple[pd.Timestamp, OptionFlowArm] | None:
    hits = iter_put_flow_dn_in_window(
        flow,
        t_start=t_start,
        t_end=t_end,
        window_sec=window_sec,
        min_put_share=min_put_share,
        min_put_vol_z=min_put_vol_z,
        min_put_v=min_put_v,
        stock_ts_ns=stock_ts_ns,
        stock_px=stock_px,
        stock_lb_sec=stock_lb_sec,
        max_stock_ret=max_stock_ret,
        stride_sec=stride_sec,
        fire_mode="first",
    )
    return hits[0] if hits else None

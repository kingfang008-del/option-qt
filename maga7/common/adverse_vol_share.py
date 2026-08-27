"""Short-window adverse volume share on stock 1s bars.

Share = volume on seconds the underlying prints against the trade
       / (adverse + favorable volume). Flat seconds excluded.

Higher share ⇒ selling (or buying) pressure aligned against the option.
Default OFF — probe AUC ~0.67 vs pure vol-dryup ~0.50.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

NY = "America/New_York"


@dataclass
class AdverseVolShareConfig:
    enabled: bool = False
    mode: str = "tox_tighten"  # tox_tighten | soft_exit
    check_seconds: int = 180
    window_seconds: int = 180
    min_share: float = 0.55
    opt_mtm_max: float = 0.0
    tight_cut_ret: float = 0.15
    tight_mfe_bypass: float = 0.03
    extend_max_cut: bool = False
    require_stock_adverse: bool = False
    stock_adverse_max: float = -0.0005


@dataclass
class EntryAdvVolConfig:
    """Pre-fill adverse volume share gate (default OFF).

    At the intended entry clock (+ optional ``lag_seconds``), if share over
    ``window_seconds`` >= ``max_share``, either ``block`` the signal or ``scale`` size.
    """

    enabled: bool = False
    action: str = "scale"  # block | scale
    window_seconds: int = 120
    max_share: float = 0.55
    scale: float = 0.5
    on_missing: str = "allow"  # allow | block
    tod_start: str | None = None
    tod_end: str | None = None
    dirs: tuple[str, ...] | None = None  # None = both; e.g. ("UP",)
    lag_seconds: int = 0  # evaluate share at ts+lag (also delays entry if >0 via replay)


def entry_adv_vol_from_trade(trade: dict[str, Any] | None) -> EntryAdvVolConfig:
    trade = trade or {}
    raw = trade.get("entry_adv_vol") or trade.get("entry_adv_vol_share")
    if raw is None:
        return EntryAdvVolConfig(enabled=False)
    if isinstance(raw, bool):
        return EntryAdvVolConfig(enabled=bool(raw))
    if not isinstance(raw, dict):
        return EntryAdvVolConfig(enabled=False)
    action = str(raw.get("action", "scale") or "scale").strip().lower()
    if action in {"size", "half"}:
        action = "scale"
    if action not in {"block", "scale"}:
        action = "scale"
    on_miss = str(raw.get("on_missing", "allow") or "allow").strip().lower()
    if on_miss not in {"allow", "block"}:
        on_miss = "allow"
    sc = float(raw.get("scale", 0.5) or 0.5)
    sc = max(0.0, min(sc, 1.0))
    tod0 = raw.get("tod_start")
    tod1 = raw.get("tod_end")
    dirs_raw = raw.get("dirs") or raw.get("directions")
    dirs: tuple[str, ...] | None = None
    if isinstance(dirs_raw, str):
        dirs = tuple(x.strip().upper() for x in dirs_raw.split(",") if x.strip())
    elif isinstance(dirs_raw, (list, tuple)):
        dirs = tuple(str(x).strip().upper() for x in dirs_raw if str(x).strip())
    if dirs == ():
        dirs = None
    lag = max(0, int(raw.get("lag_seconds", 0) or 0))
    return EntryAdvVolConfig(
        enabled=bool(raw.get("enabled", False)),
        action=action,
        window_seconds=int(raw.get("window_seconds", 120) or 120),
        max_share=float(raw.get("max_share", 0.55) or 0.55),
        scale=sc,
        on_missing=on_miss,
        tod_start=str(tod0) if tod0 not in (None, "", False) else None,
        tod_end=str(tod1) if tod1 not in (None, "", False) else None,
        dirs=dirs,
        lag_seconds=lag,
    )


def adverse_vol_share_from_trade(trade: dict[str, Any] | None) -> AdverseVolShareConfig:
    trade = trade or {}
    raw = trade.get("adverse_vol_share")
    if raw is None:
        return AdverseVolShareConfig(enabled=False)
    if isinstance(raw, bool):
        return AdverseVolShareConfig(enabled=bool(raw))
    if not isinstance(raw, dict):
        return AdverseVolShareConfig(enabled=False)
    mode = str(raw.get("mode", "tox_tighten") or "tox_tighten").strip().lower()
    if mode in {"tighten"}:
        mode = "tox_tighten"
    if mode in {"exit"}:
        mode = "soft_exit"
    if mode not in {"tox_tighten", "soft_exit"}:
        mode = "tox_tighten"
    return AdverseVolShareConfig(
        enabled=bool(raw.get("enabled", False)),
        mode=mode,
        check_seconds=int(raw.get("check_seconds", 180) or 180),
        window_seconds=int(raw.get("window_seconds", 180) or 180),
        min_share=float(raw.get("min_share", 0.55) or 0.55),
        opt_mtm_max=float(raw.get("opt_mtm_max", 0.0) if raw.get("opt_mtm_max") is not None else 0.0),
        tight_cut_ret=float(raw.get("tight_cut_ret", 0.15) or 0.15),
        tight_mfe_bypass=float(raw.get("tight_mfe_bypass", 0.03) or 0.03),
        extend_max_cut=bool(raw.get("extend_max_cut", False)),
        require_stock_adverse=bool(raw.get("require_stock_adverse", False)),
        stock_adverse_max=float(
            raw.get("stock_adverse_max", -0.0005)
            if raw.get("stock_adverse_max") is not None
            else -0.0005
        ),
    )


def _to_ny(ts) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        return t.tz_localize(NY)
    return t.tz_convert(NY)


def prepare_stock_1s_arrays(bars: pd.DataFrame | None) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Return (ts_ns, close, volume) sorted, or None."""
    if bars is None or bars.empty:
        return None
    if "close" not in bars.columns or "volume" not in bars.columns:
        return None
    df = bars
    if "timestamp" not in df.columns:
        return None
    ts = pd.to_datetime(df["timestamp"])
    if getattr(ts.dt, "tz", None) is None:
        ts = ts.dt.tz_localize(NY)
    else:
        ts = ts.dt.tz_convert(NY)
    ts_ns_all = ts.map(lambda x: int(pd.Timestamp(x).value)).to_numpy(dtype=np.int64)
    order = np.argsort(ts_ns_all)
    ts_ns = ts_ns_all[order]
    close = pd.to_numeric(df["close"], errors="coerce").to_numpy(dtype=float)[order]
    vol = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0).to_numpy(dtype=float)[order]
    return ts_ns, close, vol


def adverse_vol_share_asof(
    arrays: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
    *,
    now_ts,
    window_seconds: int,
    direction: str,
    min_seconds: int = 30,
) -> float | None:
    """Causal adverse volume share over ``(now - window, now]``."""
    if arrays is None:
        return None
    ts_ns, close, vol = arrays
    if ts_ns.size < 2:
        return None
    now = _to_ny(now_ts)
    now_ns = int(now.value)
    start_ns = now_ns - int(window_seconds) * 1_000_000_000
    right = int(np.searchsorted(ts_ns, now_ns, side="right"))
    left = int(np.searchsorted(ts_ns, start_ns, side="left"))
    if right - left < 2:
        return None
    c = close[left:right]
    v = vol[left:right]
    if c.size < 2:
        return None
    # Drop leading nan closes
    ok = np.isfinite(c) & np.isfinite(v)
    if ok.sum() < 2:
        return None
    c = c[ok]
    v = v[ok]
    d = np.diff(c)
    # align volume with the bar that printed the move (use end-of-interval vol)
    v_move = v[1:]
    up = str(direction).upper() == "UP"
    # UP trade: adverse = down ticks; DN trade: adverse = up ticks
    adv_mask = d < 0 if up else d > 0
    fav_mask = d > 0 if up else d < 0
    adv = float(v_move[adv_mask].sum())
    fav = float(v_move[fav_mask].sum())
    denom = adv + fav
    if denom <= 0:
        return None
    # require enough clock coverage
    span_sec = (ts_ns[right - 1] - ts_ns[left]) / 1e9
    if span_sec < float(min_seconds):
        return None
    return adv / denom

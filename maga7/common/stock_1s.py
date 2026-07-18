"""Stock second-level fact source helpers for Mag7.

Authoritative root: ``paths.stock_1s_root`` → ``/mnt/s990/data/raw_1s/stocks``.
Pre-aggregated ``stock_root`` (spnq_train 1m) is a research cache only — parity
and live paths must build features from 1s via this module.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pandas as pd

from maga7.common.bar_agg import aggregate_1s_to_1m, load_stock_1s_day
from maga7.common.signals import attach_mf_features, resolve_mf_fast_window

NY = "America/New_York"


def session_dates(start: str, end: str) -> list[str]:
    """NYSE business days in [start, end] (approx via pandas bdate_range)."""
    return [d.strftime("%Y-%m-%d") for d in pd.bdate_range(start, end)]


def load_symbol_1s_bars(
    stock_1s_root: Path | str,
    symbol: str,
    dates: list[str],
) -> pd.DataFrame:
    """Load and aggregate 1s → left-labeled RTH 1m bars for ``dates``."""
    root = Path(stock_1s_root)
    frames: list[pd.DataFrame] = []
    for date in dates:
        raw = load_stock_1s_day(root, symbol, date)
        if raw.empty:
            continue
        bars = aggregate_1s_to_1m(raw, symbol=symbol, rth_only=True)
        if bars.empty:
            continue
        bars = bars.copy()
        bars["date"] = date
        bars["symbol"] = symbol
        frames.append(bars)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["timestamp"] = pd.to_datetime(out["timestamp"])
    if getattr(out["timestamp"].dt, "tz", None) is None:
        out["timestamp"] = out["timestamp"].dt.tz_localize(NY)
    else:
        out["timestamp"] = out["timestamp"].dt.tz_convert(NY)
    out = out.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    out["date"] = out["timestamp"].dt.strftime("%Y-%m-%d")
    out["tod"] = out["timestamp"].dt.strftime("%H:%M")
    return out


def build_stock_by_from_1s(
    profile: dict[str, Any],
    *,
    dates: list[str] | None = None,
    include_refs: bool = True,
) -> dict[str, pd.DataFrame]:
    """Feature frames for Mag7 symbols (+ optional QQQ/VIXY) from stock 1s only."""
    stock_1s = Path(profile["_paths"]["stock_1s_root"])
    if not stock_1s.is_dir():
        raise FileNotFoundError(f"stock_1s_root missing: {stock_1s}")
    sig = profile.get("signal") or {}
    start = profile["date_range"]["start"]
    end = profile["date_range"]["end"]
    dates = dates or session_dates(start, end)
    symbols = list(profile["symbols"])
    if include_refs:
        for ref in ("QQQ", "VIXY"):
            if ref not in symbols:
                symbols.append(ref)
    out: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        raw = load_symbol_1s_bars(stock_1s, sym, dates)
        if raw.empty:
            continue
        out[sym] = attach_mf_features(
            raw,
            mf_window=int(sig.get("mf_window", 10)),
            vol_ma_window=int(sig.get("vol_ma_window", 20)),
            mf_fast_window=resolve_mf_fast_window(sig),
        )
    return out


def coverage_report(
    stock_by: dict[str, pd.DataFrame],
    *,
    dates: list[str],
    symbols: list[str],
) -> dict[str, Any]:
    """Per-symbol day counts and missing dates vs requested sessions."""
    cov: dict[str, Any] = {}
    for sym in symbols:
        df = stock_by.get(sym)
        have = set(df["date"].astype(str).unique()) if df is not None and not df.empty else set()
        missing = [d for d in dates if d not in have]
        cov[sym] = {
            "n_days": len(have),
            "missing_dates": missing,
        }
    return {"sessions": len(dates), "symbols": cov}


def regime_gate_from_1s(
    profile: dict[str, Any],
    stock_by: dict[str, pd.DataFrame],
):
    """Build Mag7RegimeGate from 1s-derived QQQ/VIXY frames.

    Falls back to ``stock_root`` month files only if QQQ 1s is entirely missing
    (should not happen for July parity when QQQ 1s is present).
    """
    from maga7.common.regime import Mag7RegimeGate, _vixy_z_series
    from maga7.common.replay import month_list
    from qqq_btc.common.regime_features import add_vix_regime_features

    reg = profile.get("regime") or {}
    if not reg.get("enabled"):
        return None
    start = profile["date_range"]["start"]
    end = profile["date_range"]["end"]
    qqq = stock_by.get("QQQ")
    vixy = stock_by.get("VIXY")
    if qqq is None or qqq.empty:
        return Mag7RegimeGate.from_profile(profile, months=month_list(start, end))

    frames = []
    q = qqq[["timestamp", "date", "close", "from_prev", "mf10"]].rename(
        columns={"close": "qqq_close", "from_prev": "qqq_from_prev", "mf10": "qqq_mf10"}
    )
    frames.append(q.set_index("timestamp"))
    if vixy is not None and not vixy.empty:
        v = vixy[["timestamp", "close"]].rename(columns={"close": "vixy_close"})
        v = v.set_index("timestamp").sort_index()
        v["vixy_z"] = _vixy_z_series(v["vixy_close"])
        tmp = v.reset_index().copy()
        tmp["vix_proxy_close"] = tmp["vixy_close"]
        tmp = add_vix_regime_features(
            tmp,
            vix_col="vix_proxy_close",
            window=int(reg.get("vix_reversal_window", 30)),
            threshold=float(reg.get("vix_reversal_pct", 0.0015)),
        )
        v = tmp.set_index("timestamp")[["vixy_close", "vixy_z", "vix_reversal_count_30m"]]
        frames.append(v)
    frame = frames[0]
    for extra in frames[1:]:
        frame = frame.join(extra, how="outer")
    frame = frame.sort_index()
    if "date" not in frame.columns:
        frame["date"] = pd.Index(frame.index).tz_convert(NY).strftime("%Y-%m-%d")
    cols = [c for c in frame.columns if c != "date"]
    frame[cols] = frame.groupby("date", sort=False)[cols].ffill()
    # Deepcopy: overlays mutate cfg; must not share profile["regime"] across gates/runs.
    return Mag7RegimeGate(frame=frame, cfg=copy.deepcopy(reg))

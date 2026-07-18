"""Opening washout + fractal-high break (ORB) — research expert signals.

Causal, 1m bars only. Default **off** in freeze; intended as a Router expert
for 09:30–10:00 V-reversal longs, separate from Rule-A (10:30+).

State ``open_washout``:
  - Session open → within ``wash_window_end``, price prints a low at least
    ``wash_drop_min`` below the open (and optionally after ``wash_min_bars``).

Signal ``orb_fractal_break`` (UP only in v1):
  - During the first unilateral selloff after open, lock ``fractal_high`` =
    high of the last bar of that selloff (user: 9:35 last-down-bar high).
  - After the selloff ends, first bar whose **close** strictly exceeds
    ``fractal_high`` (optional hold: next bar also closes above) → UP fire.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

NY = "America/New_York"


def _ensure_ny_ts(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    ts = pd.to_datetime(out["timestamp"])
    if getattr(ts.dt, "tz", None) is None:
        ts = ts.dt.tz_localize("UTC").dt.tz_convert(NY)
    else:
        ts = ts.dt.tz_convert(NY)
    out["_ts"] = ts
    return out


def _rth_slice(
    day: pd.DataFrame,
    *,
    start_hhmm: str = "09:30",
    end_hhmm: str = "10:00",
) -> pd.DataFrame:
    if day.empty:
        return day
    d0 = day["_ts"].iloc[0]
    date = d0.strftime("%Y-%m-%d")
    t0 = pd.Timestamp(f"{date} {start_hhmm}", tz=NY)
    t1 = pd.Timestamp(f"{date} {end_hhmm}", tz=NY)
    return day[(day["_ts"] >= t0) & (day["_ts"] <= t1)].sort_values("_ts")


@dataclass(frozen=True)
class OrbOpenConfig:
    wash_window_end: str = "10:00"
    wash_drop_min: float = 0.003  # |low/open - 1| for washout state
    wash_min_bars: int = 3
    selloff_min_bars: int = 3
    selloff_end_bounce_bars: int = 1  # bars without new low → selloff ends
    confirm_close_above: bool = True
    hold_confirm_bars: int = 0  # 0 = fire on break bar; 1 = need next bar hold
    signal_deadline: str = "10:00"
    only_up: bool = True

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "OrbOpenConfig":
        raw = raw or {}
        return cls(
            wash_window_end=str(raw.get("wash_window_end", "10:00")),
            wash_drop_min=float(raw.get("wash_drop_min", 0.003) or 0.003),
            wash_min_bars=int(raw.get("wash_min_bars", 3) or 3),
            selloff_min_bars=int(raw.get("selloff_min_bars", 3) or 3),
            selloff_end_bounce_bars=int(raw.get("selloff_end_bounce_bars", 1) or 1),
            confirm_close_above=bool(raw.get("confirm_close_above", True)),
            hold_confirm_bars=int(raw.get("hold_confirm_bars", 0) or 0),
            signal_deadline=str(raw.get("signal_deadline", "10:00")),
            only_up=bool(raw.get("only_up", True)),
        )


@dataclass
class OrbSignal:
    symbol: str
    date: str
    direction: str
    sig_ts: pd.Timestamp
    open_px: float
    wash_low: float
    wash_drop: float
    fractal_high: float
    selloff_end_ts: pd.Timestamp
    break_px: float
    reason: str = "orb_fractal_break"


def detect_open_washout(
    day_df: pd.DataFrame,
    *,
    cfg: OrbOpenConfig | None = None,
) -> dict[str, Any] | None:
    """Return washout stats if open→wash_window printed a deep enough low."""
    cfg = cfg or OrbOpenConfig()
    if day_df is None or day_df.empty:
        return None
    day = _ensure_ny_ts(day_df)
    win = _rth_slice(day, start_hhmm="09:30", end_hhmm=cfg.wash_window_end)
    if len(win) < max(cfg.wash_min_bars, 2):
        return None
    open_px = float(win.iloc[0]["open"] if "open" in win.columns else win.iloc[0]["close"])
    if not np.isfinite(open_px) or open_px <= 0:
        return None
    lows = pd.to_numeric(win["low"] if "low" in win.columns else win["close"], errors="coerce")
    wash_low = float(lows.min())
    if not np.isfinite(wash_low) or wash_low <= 0:
        return None
    drop = 1.0 - wash_low / open_px
    if drop < float(cfg.wash_drop_min):
        return None
    low_i = int(lows.to_numpy().argmin())
    return {
        "open_px": open_px,
        "wash_low": wash_low,
        "wash_drop": float(drop),
        "wash_low_ts": win.iloc[low_i]["_ts"],
        "n_bars": int(len(win)),
    }


def detect_orb_fractal_break(
    day_df: pd.DataFrame,
    *,
    symbol: str,
    date: str,
    cfg: OrbOpenConfig | None = None,
) -> OrbSignal | None:
    """First UP fractal-high break after open washout selloff (causal)."""
    cfg = cfg or OrbOpenConfig()
    wash = detect_open_washout(day_df, cfg=cfg)
    if wash is None:
        return None
    day = _ensure_ny_ts(day_df)
    win = _rth_slice(day, start_hhmm="09:30", end_hhmm=cfg.signal_deadline)
    if len(win) < cfg.selloff_min_bars + 2:
        return None

    opens = pd.to_numeric(win["open"] if "open" in win.columns else win["close"], errors="coerce").to_numpy()
    highs = pd.to_numeric(win["high"] if "high" in win.columns else win["close"], errors="coerce").to_numpy()
    lows = pd.to_numeric(win["low"] if "low" in win.columns else win["close"], errors="coerce").to_numpy()
    closes = pd.to_numeric(win["close"], errors="coerce").to_numpy()
    ts = win["_ts"].to_numpy()
    open_px = float(wash["open_px"])

    # --- first unilateral selloff: bars printing new lows while below open ---
    selloff_idx: list[int] = []
    running_low = open_px
    i = 0
    while i < len(win):
        if not np.isfinite(lows[i]) or not np.isfinite(closes[i]):
            i += 1
            continue
        # start selloff once we break below open
        if not selloff_idx:
            if lows[i] < open_px * (1.0 - 1e-12):
                selloff_idx.append(i)
                running_low = float(lows[i])
            i += 1
            continue
        # extend while making new lows (or close lower) and still under open
        if lows[i] < running_low - 1e-12 and closes[i] < open_px:
            selloff_idx.append(i)
            running_low = float(lows[i])
            i += 1
            continue
        break

    if len(selloff_idx) < int(cfg.selloff_min_bars):
        return None

    last_down = selloff_idx[-1]
    fractal_high = float(highs[last_down])
    if not np.isfinite(fractal_high) or fractal_high <= 0:
        return None
    # require selloff deep enough vs open (reuse wash_drop_min on path low)
    path_low = float(np.nanmin(lows[selloff_idx[0] : last_down + 1]))
    if 1.0 - path_low / open_px < float(cfg.wash_drop_min):
        return None

    selloff_end_ts = pd.Timestamp(ts[last_down]).tz_convert(NY) if pd.Timestamp(ts[last_down]).tzinfo else pd.Timestamp(ts[last_down]).tz_localize(NY)

    # --- watch for break after selloff end ---
    pending_break_i: int | None = None
    for j in range(last_down + 1, len(win)):
        if not np.isfinite(closes[j]) or not np.isfinite(highs[j]):
            continue
        broke = closes[j] > fractal_high if cfg.confirm_close_above else highs[j] > fractal_high
        if not broke:
            pending_break_i = None
            continue
        if int(cfg.hold_confirm_bars) <= 0:
            return OrbSignal(
                symbol=str(symbol),
                date=str(date),
                direction="UP",
                sig_ts=pd.Timestamp(ts[j]).tz_convert(NY)
                if pd.Timestamp(ts[j]).tzinfo
                else pd.Timestamp(ts[j]).tz_localize(NY),
                open_px=open_px,
                wash_low=float(wash["wash_low"]),
                wash_drop=float(wash["wash_drop"]),
                fractal_high=fractal_high,
                selloff_end_ts=selloff_end_ts,
                break_px=float(closes[j]),
            )
        if pending_break_i is None:
            pending_break_i = j
            continue
        # hold: need hold_confirm_bars consecutive closes above
        if j - pending_break_i >= int(cfg.hold_confirm_bars) and closes[j] > fractal_high:
            return OrbSignal(
                symbol=str(symbol),
                date=str(date),
                direction="UP",
                sig_ts=pd.Timestamp(ts[j]).tz_convert(NY)
                if pd.Timestamp(ts[j]).tzinfo
                else pd.Timestamp(ts[j]).tz_localize(NY),
                open_px=open_px,
                wash_low=float(wash["wash_low"]),
                wash_drop=float(wash["wash_drop"]),
                fractal_high=fractal_high,
                selloff_end_ts=selloff_end_ts,
                break_px=float(closes[j]),
                reason="orb_fractal_break_hold",
            )
    return None


def scan_orb_day(
    stock_by: dict[str, pd.DataFrame],
    *,
    date: str,
    symbols: list[str],
    cfg: OrbOpenConfig | None = None,
) -> list[OrbSignal]:
    """Per-symbol first ORB fire on ``date`` (if any)."""
    cfg = cfg or OrbOpenConfig()
    out: list[OrbSignal] = []
    for sym in symbols:
        sdf = stock_by.get(sym)
        if sdf is None or sdf.empty:
            continue
        day = sdf[sdf["date"].astype(str) == str(date)]
        if day.empty:
            continue
        sig = detect_orb_fractal_break(day, symbol=sym, date=date, cfg=cfg)
        if sig is not None:
            out.append(sig)
    out.sort(key=lambda s: (s.sig_ts, s.symbol))
    return out


@dataclass(frozen=True)
class WashoutReclaimConfig:
    """Per-symbol washout → reclaim-open hunter (research).

    Distinct from ``orb_fractal`` (fractal-high break) and from Halt's
    Mag7-breadth ``washout_and_reclaim`` (which *blocks* entries).
    """

    wash_window_end: str = "10:00"
    wash_drop_min: float = 0.01
    wash_min_bars: int = 3
    signal_deadline: str = "10:15"
    hold_confirm_bars: int = 0
    reclaim_level: str = "open"  # open | mid (50% of wash)
    reclaim_buffer_pct: float = 0.0  # require close > reclaim_px * (1 + buf)
    only_up: bool = True

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "WashoutReclaimConfig":
        raw = raw or {}
        return cls(
            wash_window_end=str(raw.get("wash_window_end", "10:00")),
            wash_drop_min=float(raw.get("wash_drop_min", 0.01) or 0.01),
            wash_min_bars=int(raw.get("wash_min_bars", 3) or 3),
            signal_deadline=str(raw.get("signal_deadline", "10:15")),
            hold_confirm_bars=int(raw.get("hold_confirm_bars", 0) or 0),
            reclaim_level=str(raw.get("reclaim_level", "open") or "open").strip().lower(),
            reclaim_buffer_pct=float(raw.get("reclaim_buffer_pct", 0.0) or 0.0),
            only_up=bool(raw.get("only_up", True)),
        )


def detect_washout_reclaim(
    day_df: pd.DataFrame,
    *,
    symbol: str,
    date: str,
    cfg: WashoutReclaimConfig | None = None,
) -> OrbSignal | None:
    """UP when session prints deep washout then close reclaims open (or mid).

    Causal: wash low must print before the reclaim bar; fire on first close
    through reclaim level after that low (optional hold bars).
    """
    cfg = cfg or WashoutReclaimConfig()
    orb_wash = OrbOpenConfig(
        wash_window_end=cfg.wash_window_end,
        wash_drop_min=float(cfg.wash_drop_min),
        wash_min_bars=int(cfg.wash_min_bars),
        signal_deadline=cfg.signal_deadline,
    )
    wash = detect_open_washout(day_df, cfg=orb_wash)
    if wash is None:
        return None
    day = _ensure_ny_ts(day_df)
    win = _rth_slice(day, start_hhmm="09:30", end_hhmm=cfg.signal_deadline)
    if len(win) < max(cfg.wash_min_bars, 2) + 1:
        return None

    open_px = float(wash["open_px"])
    wash_low = float(wash["wash_low"])
    wash_low_ts = pd.Timestamp(wash["wash_low_ts"])
    if getattr(wash_low_ts, "tzinfo", None) is None:
        wash_low_ts = wash_low_ts.tz_localize(NY)
    else:
        wash_low_ts = wash_low_ts.tz_convert(NY)

    level = str(cfg.reclaim_level or "open").strip().lower()
    if level in {"mid", "half", "0.5", "50"}:
        reclaim_px = wash_low + 0.5 * (open_px - wash_low)
    else:
        reclaim_px = open_px
    if not np.isfinite(reclaim_px) or reclaim_px <= 0:
        return None
    buf = max(0.0, float(cfg.reclaim_buffer_pct or 0.0))
    reclaim_thr = reclaim_px * (1.0 + buf)

    closes = pd.to_numeric(win["close"], errors="coerce").to_numpy()
    ts = win["_ts"].to_numpy()
    pending: int | None = None
    for j in range(len(win)):
        t_j = pd.Timestamp(ts[j])
        if getattr(t_j, "tzinfo", None) is None:
            t_j = t_j.tz_localize(NY)
        else:
            t_j = t_j.tz_convert(NY)
        if t_j <= wash_low_ts:
            continue
        c = float(closes[j]) if np.isfinite(closes[j]) else float("nan")
        if not np.isfinite(c) or c <= reclaim_thr:
            pending = None
            continue
        if int(cfg.hold_confirm_bars) <= 0:
            return OrbSignal(
                symbol=str(symbol),
                date=str(date),
                direction="UP",
                sig_ts=t_j,
                open_px=open_px,
                wash_low=wash_low,
                wash_drop=float(wash["wash_drop"]),
                fractal_high=float(reclaim_thr),
                selloff_end_ts=wash_low_ts,
                break_px=c,
                reason=f"washout_reclaim_{level}",
            )
        if pending is None:
            pending = j
            continue
        if j - pending >= int(cfg.hold_confirm_bars) and c > reclaim_thr:
            return OrbSignal(
                symbol=str(symbol),
                date=str(date),
                direction="UP",
                sig_ts=t_j,
                open_px=open_px,
                wash_low=wash_low,
                wash_drop=float(wash["wash_drop"]),
                fractal_high=float(reclaim_thr),
                selloff_end_ts=wash_low_ts,
                break_px=c,
                reason=f"washout_reclaim_{level}_hold",
            )
    return None


def scan_washout_reclaim_day(
    stock_by: dict[str, pd.DataFrame],
    *,
    date: str,
    symbols: list[str],
    cfg: WashoutReclaimConfig | None = None,
) -> list[OrbSignal]:
    """Per-symbol first washout→reclaim fire on ``date``."""
    cfg = cfg or WashoutReclaimConfig()
    out: list[OrbSignal] = []
    for sym in symbols:
        sdf = stock_by.get(sym)
        if sdf is None or sdf.empty:
            continue
        day = sdf[sdf["date"].astype(str) == str(date)]
        if day.empty:
            continue
        sig = detect_washout_reclaim(day, symbol=sym, date=date, cfg=cfg)
        if sig is not None:
            out.append(sig)
    out.sort(key=lambda s: (s.sig_ts, s.symbol))
    return out


def count_open_washout(
    stock_by: dict[str, pd.DataFrame],
    *,
    date: str,
    symbols: list[str],
    cfg: OrbOpenConfig | None = None,
) -> tuple[int, list[str]]:
    """How many symbols printed ``open_washout`` on ``date`` (causal, 1m).

    Returns ``(n, [symbols...])``. Used as a Router **gate** state — not an entry.
    """
    cfg = cfg or OrbOpenConfig()
    hit: list[str] = []
    for sym in symbols:
        sdf = stock_by.get(sym)
        if sdf is None or sdf.empty:
            continue
        day = sdf[sdf["date"].astype(str) == str(date)]
        if day.empty:
            continue
        if detect_open_washout(day, cfg=cfg) is not None:
            hit.append(str(sym))
    return len(hit), hit

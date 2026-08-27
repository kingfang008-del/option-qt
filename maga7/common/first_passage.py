"""ATR-normalized first-passage labels for Top2 Entry Validator (V2)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

NY = "America/New_York"


def _to_ny(ts) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        return t.tz_localize(NY)
    return t.tz_convert(NY)


def _prepare_day(df: pd.DataFrame, date: str) -> pd.DataFrame:
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"])
    if getattr(out["timestamp"].dt, "tz", None) is None:
        out["timestamp"] = out["timestamp"].dt.tz_localize(NY)
    else:
        out["timestamp"] = out["timestamp"].dt.tz_convert(NY)
    if "date" not in out.columns:
        out["date"] = out["timestamp"].dt.strftime("%Y-%m-%d")
    day = out[out["date"].astype(str) == str(date)].sort_values("timestamp")
    return day.reset_index(drop=True)


@dataclass(frozen=True)
class FirstPassageConfig:
    horizon_minutes: int = 90
    # Fixed percentage baselines (~1.0% MFE / 0.5% MAE)
    good_mfe_pct: float = 0.010
    toxic_mae_pct: float = 0.005
    # Primary ATR: causal daily ATR (prior sessions). Mag7 daily ATR% ~1.5–3%;
    # 0.5x / 0.25x maps near the pct baseline and leaves ambiguous mass.
    atr_days: int = 14
    good_mfe_atr: float = 0.50
    toxic_mae_atr: float = 0.25
    # Intraday fallback if daily ATR unavailable
    atr_window: int = 60
    min_atr_pct: float = 0.005  # floor ~0.5% daily-scale


def _ensure_ts(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"])
    if getattr(out["timestamp"].dt, "tz", None) is None:
        out["timestamp"] = out["timestamp"].dt.tz_localize(NY)
    else:
        out["timestamp"] = out["timestamp"].dt.tz_convert(NY)
    if "date" not in out.columns:
        out["date"] = out["timestamp"].dt.strftime("%Y-%m-%d")
    return out


def causal_intraday_atr_pct(
    hist: pd.DataFrame,
    *,
    asof_ts,
    window: int = 60,
) -> float | None:
    """Causal 1m ATR% using bars strictly <= asof (fallback)."""
    hist = _ensure_ts(hist)
    asof = _to_ny(asof_ts)
    upto = hist[hist["timestamp"] <= asof]
    if len(upto) < 10:
        return None
    hi = pd.to_numeric(upto.get("high", upto["close"]), errors="coerce")
    lo = pd.to_numeric(upto.get("low", upto["close"]), errors="coerce")
    cl = pd.to_numeric(upto["close"], errors="coerce")
    prev = cl.shift(1)
    tr = pd.concat([(hi - lo).abs(), (hi - prev).abs(), (lo - prev).abs()], axis=1).max(axis=1)
    use_n = min(int(window), int(tr.notna().sum()))
    if use_n < 10:
        return None
    atr = float(tr.tail(use_n).mean())
    px = float(cl.iloc[-1])
    if not np.isfinite(atr) or not np.isfinite(px) or px <= 0 or atr <= 0:
        return None
    return max(atr / px, 1e-6)


def causal_daily_atr_pct(
    hist: pd.DataFrame,
    *,
    asof_ts,
    n_days: int = 14,
) -> float | None:
    """Causal daily ATR as fraction of last prior close.

    Uses only fully completed sessions strictly before asof's date.
    """
    hist = _ensure_ts(hist)
    asof = _to_ny(asof_ts)
    asof_date = asof.strftime("%Y-%m-%d")
    prior = hist[hist["date"].astype(str) < asof_date]
    if prior.empty:
        return None
    rows = []
    for d, g in prior.groupby(prior["date"].astype(str), sort=True):
        hi = float(pd.to_numeric(g.get("high", g["close"]), errors="coerce").max())
        lo = float(pd.to_numeric(g.get("low", g["close"]), errors="coerce").min())
        cl = float(pd.to_numeric(g["close"], errors="coerce").iloc[-1])
        if not np.isfinite(hi) or not np.isfinite(lo) or not np.isfinite(cl) or cl <= 0:
            continue
        rows.append({"date": d, "high": hi, "low": lo, "close": cl})
    if len(rows) < 3:
        return None
    daily = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    daily["prev_close"] = daily["close"].shift(1)
    tr = pd.concat(
        [
            (daily["high"] - daily["low"]).abs(),
            (daily["high"] - daily["prev_close"]).abs(),
            (daily["low"] - daily["prev_close"]).abs(),
        ],
        axis=1,
    ).max(axis=1)
    use = tr.dropna().tail(int(n_days))
    if len(use) < 3:
        return None
    atr = float(use.mean())
    px = float(daily["close"].iloc[-1])
    if not np.isfinite(atr) or not np.isfinite(px) or px <= 0 or atr <= 0:
        return None
    return max(atr / px, 1e-6)


def causal_atr_pct(
    hist: pd.DataFrame,
    *,
    asof_ts,
    n_days: int = 14,
    window: int = 60,
) -> float | None:
    """Prefer daily ATR%; fall back to intraday ATR%."""
    daily = causal_daily_atr_pct(hist, asof_ts=asof_ts, n_days=n_days)
    if daily is not None:
        return daily
    return causal_intraday_atr_pct(hist, asof_ts=asof_ts, window=window)


def first_passage_label(
    day: pd.DataFrame,
    *,
    entry_ts,
    direction: str,
    date: str | None = None,
    cfg: FirstPassageConfig | None = None,
    atr_hist: pd.DataFrame | None = None,
) -> dict[str, Any] | None:
    """Label ClearTrue / ClearFalse / Ambiguous via first barrier touch.

    Pass ``atr_hist`` (same symbol, including prior days) for stable morning ATR.
    """
    cfg = cfg or FirstPassageConfig()
    d = str(direction).upper()
    if date is not None:
        day = _prepare_day(day, date)
    else:
        day = _prepare_day(day, str(_to_ny(entry_ts).strftime("%Y-%m-%d")))
    et = _to_ny(entry_ts)
    atr_src = atr_hist if atr_hist is not None and not atr_hist.empty else day
    atr_pct = causal_atr_pct(
        atr_src, asof_ts=et, n_days=cfg.atr_days, window=cfg.atr_window
    )
    if atr_pct is None:
        atr_pct = cfg.min_atr_pct
    atr_pct = max(float(atr_pct), cfg.min_atr_pct)

    mfe_thr_pct = float(cfg.good_mfe_pct)
    mae_thr_pct = float(cfg.toxic_mae_pct)
    mfe_thr_atr = float(cfg.good_mfe_atr) * atr_pct
    mae_thr_atr = float(cfg.toxic_mae_atr) * atr_pct

    after = day[(day["timestamp"] >= et) & (day["timestamp"] <= et + pd.Timedelta(minutes=cfg.horizon_minutes))]
    if len(after) < 5:
        return None
    px0 = float(after.iloc[0]["close"])
    if px0 <= 0:
        return None

    has_hl = "high" in after.columns and "low" in after.columns
    hit_mfe_pct = hit_mae_pct = None
    hit_mfe_atr = hit_mae_atr = None
    mfe_path = 0.0
    mae_path = 0.0

    for _, row in after.iterrows():
        if has_hl:
            hi = float(pd.to_numeric(row["high"], errors="coerce"))
            lo = float(pd.to_numeric(row["low"], errors="coerce"))
        else:
            hi = lo = float(row["close"])
        if d == "UP":
            fav = hi / px0 - 1.0
            adv = -(lo / px0 - 1.0)
        else:
            fav = 1.0 - lo / px0
            adv = hi / px0 - 1.0
        mfe_path = max(mfe_path, fav)
        mae_path = max(mae_path, adv)
        ts = pd.Timestamp(row["timestamp"])
        if hit_mfe_pct is None and fav >= mfe_thr_pct:
            hit_mfe_pct = ts
        if hit_mae_pct is None and adv >= mae_thr_pct:
            hit_mae_pct = ts
        if hit_mfe_atr is None and fav >= mfe_thr_atr:
            hit_mfe_atr = ts
        if hit_mae_atr is None and adv >= mae_thr_atr:
            hit_mae_atr = ts
        # early stop if both schemes decided
        if hit_mfe_pct is not None and hit_mae_pct is not None and hit_mfe_atr is not None and hit_mae_atr is not None:
            break

    def _ternary(hit_mfe, hit_mae) -> str:
        if hit_mfe is None and hit_mae is None:
            return "ambiguous"
        if hit_mfe is not None and (hit_mae is None or hit_mfe <= hit_mae):
            return "clear_true"
        if hit_mae is not None and (hit_mfe is None or hit_mae < hit_mfe):
            return "clear_false"
        return "ambiguous"

    label_pct = _ternary(hit_mfe_pct, hit_mae_pct)
    label_atr = _ternary(hit_mfe_atr, hit_mae_atr)
    return {
        "entry_px": px0,
        "atr_pct": atr_pct,
        "mfe_path": float(mfe_path),
        "mae_path": float(mae_path),
        "mfe_thr_pct": mfe_thr_pct,
        "mae_thr_pct": mae_thr_pct,
        "mfe_thr_atr": mfe_thr_atr,
        "mae_thr_atr": mae_thr_atr,
        "label_pct": label_pct,
        "label_atr": label_atr,
        "y_clear_true_pct": int(label_pct == "clear_true"),
        "y_clear_false_pct": int(label_pct == "clear_false"),
        "y_clear_true_atr": int(label_atr == "clear_true"),
        "y_clear_false_atr": int(label_atr == "clear_false"),
        "y_train_pct": (
            1
            if label_pct == "clear_true"
            else (0 if label_pct == "clear_false" else None)
        ),
        "y_train_atr": (
            1
            if label_atr == "clear_true"
            else (0 if label_atr == "clear_false" else None)
        ),
        "t_mfe_pct": str(hit_mfe_pct) if hit_mfe_pct is not None else None,
        "t_mae_pct": str(hit_mae_pct) if hit_mae_pct is not None else None,
        "t_mfe_atr": str(hit_mfe_atr) if hit_mfe_atr is not None else None,
        "t_mae_atr": str(hit_mae_atr) if hit_mae_atr is not None else None,
        "horizon_minutes": int(cfg.horizon_minutes),
    }

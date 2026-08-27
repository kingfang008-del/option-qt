"""Stock 1s hold-window path typology for whipsaw vs bleed.

Signed return is direction-aware: +favorable / -adverse.
Does not require trade-tape prints — 1s OHLCV close path is enough
for Mag7's minute-scale holds.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

NY = "America/New_York"

# Thresholds in fraction (1bp = 1e-4)
SHALLOW_MAE = 0.0003  # -3bp
DEEP_MAE = 0.0015  # -15bp
EXIT_OK = 0.0003  # recovered to >= -3bp
RECOVER_FRAC = 0.60


@dataclass(frozen=True)
class PathHorizon:
    signed_ret: float | None
    mae: float | None
    mfe: float | None
    n_bars: int


@dataclass
class HoldPathMetrics:
    entry_px: float | None
    exit_px: float | None
    hold_sec: float | None
    signed_exit: float | None
    mae: float | None
    mfe: float | None
    t_mae_sec: float | None
    recover_frac: float | None
    h1: PathHorizon
    h5: PathHorizon
    h15: PathHorizon
    subtype: str
    vol_adverse: float | None = None
    vol_recover: float | None = None

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        for key in ("h1", "h5", "h15"):
            hz = d.pop(key)
            prefix = key
            d[f"{prefix}_signed"] = hz["signed_ret"]
            d[f"{prefix}_mae"] = hz["mae"]
            d[f"{prefix}_mfe"] = hz["mfe"]
            d[f"{prefix}_n"] = hz["n_bars"]
        return d


def _to_ny(ts) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        return t.tz_localize(NY)
    return t.tz_convert(NY)


def signed_stock_ret(px: float, entry_px: float, direction: str) -> float:
    """+favorable for trade direction."""
    raw = (float(px) / float(entry_px)) - 1.0
    return raw if str(direction).upper() == "UP" else -raw


def classify_whipsaw_subtype(
    mae: float | None,
    signed_exit: float | None,
    *,
    shallow_mae: float = SHALLOW_MAE,
    deep_mae: float = DEEP_MAE,
    exit_ok: float = EXIT_OK,
    recover_frac_min: float = RECOVER_FRAC,
) -> str:
    """Map hold path into coarse subtypes.

    - no_adverse: never meaningfully against
    - shallow_wash_recover: small adverse then mostly back
    - deep_adverse_recover: deep wash then stock recovers (option often still dead)
    - deep_adverse_persist: deep wash and still adverse at exit
    - shallow_adverse_persist: small adverse, no recovery
    """
    if mae is None or signed_exit is None or not np.isfinite(mae) or not np.isfinite(signed_exit):
        return "missing"
    if mae > -shallow_mae:
        return "no_adverse"
    depth = abs(float(mae))
    recovered = (float(signed_exit) - float(mae)) / depth if depth > 0 else 0.0
    exit_ok_flag = float(signed_exit) >= -exit_ok
    if mae <= -deep_mae:
        if exit_ok_flag or recovered >= recover_frac_min:
            return "deep_adverse_recover"
        return "deep_adverse_persist"
    # shallow band
    if exit_ok_flag or recovered >= recover_frac_min:
        return "shallow_wash_recover"
    return "shallow_adverse_persist"


def _horizon(signed: np.ndarray, mask: np.ndarray) -> PathHorizon:
    if not mask.any():
        return PathHorizon(None, None, None, 0)
    s = signed[mask]
    return PathHorizon(
        signed_ret=float(s[-1]),
        mae=float(np.min(s)),
        mfe=float(np.max(s)),
        n_bars=int(s.size),
    )


def analyze_hold_path(
    bars: pd.DataFrame,
    *,
    entry_ts,
    exit_ts,
    direction: str,
    price_col: str = "close",
) -> HoldPathMetrics:
    """Compute signed path metrics on a 1s (or denser) OHLCV frame."""
    empty_h = PathHorizon(None, None, None, 0)
    if bars is None or bars.empty or price_col not in bars.columns:
        return HoldPathMetrics(
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            empty_h,
            empty_h,
            empty_h,
            "missing",
        )

    et = _to_ny(entry_ts)
    xt = _to_ny(exit_ts)
    df = bars.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    if getattr(df["timestamp"].dt, "tz", None) is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize(NY)
    else:
        df["timestamp"] = df["timestamp"].dt.tz_convert(NY)
    win = df[(df["timestamp"] >= et) & (df["timestamp"] <= xt)].sort_values("timestamp")
    if win.empty:
        # allow 1s snap just before entry
        pre = df[df["timestamp"] <= et].tail(1)
        if pre.empty:
            return HoldPathMetrics(
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                empty_h,
                empty_h,
                empty_h,
                "missing",
            )
        win = pre

    px = win[price_col].astype(float).to_numpy()
    ts = win["timestamp"].to_numpy()
    entry_px = float(px[0])
    if entry_px <= 0:
        return HoldPathMetrics(
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            empty_h,
            empty_h,
            empty_h,
            "missing",
        )

    signed = np.array([signed_stock_ret(p, entry_px, direction) for p in px], dtype=float)
    mae = float(np.min(signed))
    mfe = float(np.max(signed))
    i_mae = int(np.argmin(signed))
    t_mae = float((pd.Timestamp(ts[i_mae]) - et).total_seconds())
    signed_exit = float(signed[-1])
    hold_sec = float((pd.Timestamp(ts[-1]) - et).total_seconds())
    depth = abs(mae)
    recover_frac = ((signed_exit - mae) / depth) if depth > 1e-12 else None

    elapsed = np.array([(pd.Timestamp(t) - et).total_seconds() for t in ts], dtype=float)
    h1 = _horizon(signed, elapsed <= 60.0)
    h5 = _horizon(signed, elapsed <= 300.0)
    h15 = _horizon(signed, elapsed <= 900.0)

    vol_adv = vol_rec = None
    if "volume" in win.columns and len(px) >= 2:
        vol = win["volume"].astype(float).to_numpy()
        # volume on bars while making new adverse lows vs after mae
        vol_adv = float(np.nansum(vol[: i_mae + 1]))
        vol_rec = float(np.nansum(vol[i_mae + 1 :])) if i_mae + 1 < len(vol) else 0.0

    subtype = classify_whipsaw_subtype(mae, signed_exit)
    return HoldPathMetrics(
        entry_px=entry_px,
        exit_px=float(px[-1]),
        hold_sec=hold_sec,
        signed_exit=signed_exit,
        mae=mae,
        mfe=mfe,
        t_mae_sec=t_mae,
        recover_frac=recover_frac,
        h1=h1,
        h5=h5,
        h15=h15,
        subtype=subtype,
        vol_adverse=vol_adv,
        vol_recover=vol_rec,
    )

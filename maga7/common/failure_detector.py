"""Layer 4 Failure Detector — post-entry structure failure (stock research).

Outputs HOLD / EXIT (REDUCE reserved). Evaluates only information available
after entry; does not delay entry. Sleeve-specific defaults: Impulse tighter.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

from maga7.common.smooth_trend import SmoothStockTradeConfig, _hhmm_to_min, _prepare_day, _window_stats

NY = "America/New_York"
Action = Literal["HOLD", "REDUCE", "EXIT"]


@dataclass(frozen=True)
class FailureDetectorConfig:
    """Rule-based failure arms (Phase 3 v1)."""

    enabled: bool = True
    # Only arm after this many minutes (avoid noise in first bar)
    min_hold_minutes: float = 1.0
    # Evaluate failure primarily in early window; after that trail owns exit
    max_eval_minutes: float = 15.0
    # Absolute adverse from entry (signed) → EXIT
    early_mae_cut: float = 0.004
    # Giveback from MFE peak within early window
    early_giveback: float = 0.003
    # Break below/above structure from pre-entry lookback extreme
    structure_lookback: int = 10
    structure_break_buf: float = 0.0005
    # Path collapse: last N bars up_frac + still underwater
    path_lookback: int = 5
    path_min_up_frac: float = 0.35
    # Lose session open / VWAP while underwater (off by default — too aggressive)
    lose_open: bool = False
    lose_vwap: bool = False
    # Sleeve overrides applied by factory
    sleeve: str = "smooth"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def failure_cfg_for_sleeve(sleeve: str) -> FailureDetectorConfig:
    s = str(sleeve).lower()
    if s == "impulse":
        return FailureDetectorConfig(
            sleeve="impulse",
            min_hold_minutes=1.0,
            max_eval_minutes=10.0,
            early_mae_cut=0.0035,
            early_giveback=0.0025,
            path_min_up_frac=0.30,
            structure_lookback=8,
        )
    return FailureDetectorConfig(
        sleeve="smooth",
        min_hold_minutes=2.0,
        max_eval_minutes=15.0,
        early_mae_cut=0.005,
        early_giveback=0.004,
        path_min_up_frac=0.35,
        structure_lookback=12,
    )


def _session_open_vwap(day: pd.DataFrame, asof) -> tuple[float | None, float | None]:
    day = day.copy()
    day["timestamp"] = pd.to_datetime(day["timestamp"])
    if getattr(day["timestamp"].dt, "tz", None) is None:
        day["timestamp"] = day["timestamp"].dt.tz_localize(NY)
    else:
        day["timestamp"] = day["timestamp"].dt.tz_convert(NY)
    asof = pd.Timestamp(asof)
    if asof.tzinfo is None:
        asof = asof.tz_localize(NY)
    else:
        asof = asof.tz_convert(NY)
    upto = day[day["timestamp"] <= asof]
    if upto.empty:
        return None, None
    open_px = float(upto.iloc[0]["open"] if "open" in upto.columns else upto.iloc[0]["close"])
    cl = pd.to_numeric(upto["close"], errors="coerce")
    vol = pd.to_numeric(upto["volume"], errors="coerce") if "volume" in upto.columns else pd.Series(1.0, index=upto.index)
    vol = vol.fillna(0.0).clip(lower=0.0)
    if float(vol.sum()) > 0:
        vwap = float((cl * vol).sum() / vol.sum())
    else:
        vwap = float(cl.mean())
    return open_px, vwap


def evaluate_failure(
    *,
    direction: str,
    entry_px: float,
    entry_ts,
    now_ts,
    now_px: float,
    peak: float,
    trough: float,
    closes_since_entry: np.ndarray,
    structure_extreme: float | None,
    open_px: float | None,
    vwap: float | None,
    cfg: FailureDetectorConfig,
) -> tuple[Action, str | None]:
    """Causal failure check at one bar. Returns (action, reason)."""
    if not cfg.enabled:
        return "HOLD", None
    et = pd.Timestamp(entry_ts)
    nt = pd.Timestamp(now_ts)
    if et.tzinfo is None:
        et = et.tz_localize(NY)
    if nt.tzinfo is None:
        nt = nt.tz_localize(NY)
    held = (nt - et).total_seconds() / 60.0
    if held < cfg.min_hold_minutes or held > cfg.max_eval_minutes:
        return "HOLD", None

    d = str(direction).upper()
    if d == "UP":
        signed = now_px / entry_px - 1.0
        mae = max(0.0, 1.0 - now_px / entry_px) if now_px < entry_px else 0.0
        # also track worst adverse from entry using trough of path
        if len(closes_since_entry):
            lo = float(np.min(closes_since_entry))
            mae = max(mae, max(0.0, 1.0 - lo / entry_px))
        mfe = max(0.0, peak / entry_px - 1.0)
        giveback = (peak / now_px - 1.0) if peak > 0 and now_px > 0 else 0.0
    else:
        signed = 1.0 - now_px / entry_px
        mae = max(0.0, now_px / entry_px - 1.0) if now_px > entry_px else 0.0
        if len(closes_since_entry):
            hi = float(np.max(closes_since_entry))
            mae = max(mae, max(0.0, hi / entry_px - 1.0))
        mfe = max(0.0, 1.0 - trough / entry_px)
        giveback = (now_px / trough - 1.0) if trough > 0 else 0.0

    if mae >= cfg.early_mae_cut:
        return "EXIT", "FD_EARLY_MAE"

    if mfe >= cfg.early_mae_cut * 0.5 and giveback >= cfg.early_giveback and signed < 0.001:
        return "EXIT", "FD_GIVEBACK"

    if structure_extreme is not None and np.isfinite(structure_extreme):
        if d == "UP" and now_px < structure_extreme * (1.0 - cfg.structure_break_buf):
            return "EXIT", "FD_STRUCTURE"
        if d == "DN" and now_px > structure_extreme * (1.0 + cfg.structure_break_buf):
            return "EXIT", "FD_STRUCTURE"

    if len(closes_since_entry) >= cfg.path_lookback + 1:
        w = closes_since_entry[-(cfg.path_lookback + 1) :]
        st = _window_stats(w, direction=d)
        if st is not None and st["up_frac"] < cfg.path_min_up_frac and signed < 0.0:
            return "EXIT", "FD_PATH"

    if signed < 0.0:
        if cfg.lose_open and open_px is not None and open_px > 0:
            if d == "UP" and now_px < open_px:
                return "EXIT", "FD_LOSE_OPEN"
            if d == "DN" and now_px > open_px:
                return "EXIT", "FD_LOSE_OPEN"
        if cfg.lose_vwap and vwap is not None and vwap > 0:
            if d == "UP" and now_px < vwap:
                return "EXIT", "FD_LOSE_VWAP"
            if d == "DN" and now_px > vwap:
                return "EXIT", "FD_LOSE_VWAP"

    return "HOLD", None


def _pre_entry_structure(
    day: pd.DataFrame,
    *,
    entry_ts,
    direction: str,
    lookback: int,
) -> float | None:
    day = day.copy()
    day["timestamp"] = pd.to_datetime(day["timestamp"])
    if getattr(day["timestamp"].dt, "tz", None) is None:
        day["timestamp"] = day["timestamp"].dt.tz_localize(NY)
    else:
        day["timestamp"] = day["timestamp"].dt.tz_convert(NY)
    et = pd.Timestamp(entry_ts)
    if et.tzinfo is None:
        et = et.tz_localize(NY)
    else:
        et = et.tz_convert(NY)
    before = day[day["timestamp"] < et].tail(int(lookback))
    if before.empty:
        return None
    if str(direction).upper() == "UP":
        lo = pd.to_numeric(before.get("low", before["close"]), errors="coerce")
        return float(lo.min())
    hi = pd.to_numeric(before.get("high", before["close"]), errors="coerce")
    return float(hi.max())


def simulate_stock_with_failure(
    day: pd.DataFrame,
    *,
    entry_ts,
    direction: str,
    trade_cfg: SmoothStockTradeConfig | None = None,
    fd_cfg: FailureDetectorConfig | None = None,
    date: str | None = None,
    sleeve: str = "smooth",
    bar_seconds: int = 60,
    trail_arm_minutes: float | None = None,
) -> dict[str, Any] | None:
    """Baseline trail/smooth exits + optional early Failure Detector EXIT.

    ``bar_seconds``: bar duration (60 for 1m, 1 for 1s). Lookback *bars* for
    SMOOTH_BREAK scale with this; trail arming uses wall-clock minutes.
    """
    trade_cfg = trade_cfg or SmoothStockTradeConfig()
    fd_cfg = fd_cfg or failure_cfg_for_sleeve(sleeve)
    if date is not None:
        day = _prepare_day(day, date)
    else:
        day = day.copy()
        day["timestamp"] = pd.to_datetime(day["timestamp"])
        if getattr(day["timestamp"].dt, "tz", None) is None:
            day["timestamp"] = day["timestamp"].dt.tz_localize(NY)
        else:
            day["timestamp"] = day["timestamp"].dt.tz_convert(NY)
        day = day.sort_values("timestamp")

    entry_ts = pd.Timestamp(entry_ts)
    if entry_ts.tzinfo is None:
        entry_ts = entry_ts.tz_localize(NY)
    else:
        entry_ts = entry_ts.tz_convert(NY)
    after = day[day["timestamp"] >= entry_ts]
    if after.empty:
        return None
    entry_px = float(after.iloc[0]["close"])
    if entry_px <= 0:
        return None

    need_book = bool(fd_cfg.enabled and (fd_cfg.lose_open or fd_cfg.lose_vwap))
    need_structure = bool(fd_cfg.enabled and fd_cfg.structure_lookback and fd_cfg.structure_lookback < 900)
    need_path = bool(fd_cfg.enabled and fd_cfg.path_min_up_frac >= 0)
    need_giveback = bool(fd_cfg.enabled and fd_cfg.early_giveback < 1.0)
    mae_only = bool(
        fd_cfg.enabled
        and not need_book
        and not need_structure
        and not need_path
        and not need_giveback
    )

    structure = (
        _pre_entry_structure(
            day, entry_ts=entry_ts, direction=direction, lookback=fd_cfg.structure_lookback
        )
        if need_structure
        else None
    )
    closes = after["close"].astype(float).to_numpy()
    n = len(closes)
    # wall-clock held minutes from timestamps (supports irregular 1s gaps)
    tsec = (
        pd.to_datetime(after["timestamp"], utc=True)
        .astype("int64")
        .to_numpy(dtype=np.int64)
        / 1e9
    )
    entry_sec = float(pd.Timestamp(entry_ts).timestamp())
    held_min = (tsec - entry_sec) / 60.0
    hm = (
        pd.to_datetime(after["timestamp"]).dt.hour.to_numpy() * 60
        + pd.to_datetime(after["timestamp"]).dt.minute.to_numpy()
    )
    eod_m = _hhmm_to_min(trade_cfg.eod_hhmm)
    arm_min = float(trail_arm_minutes if trail_arm_minutes is not None else trade_cfg.break_lookback)
    # SMOOTH_BREAK window in bars: ~break_lookback minutes of bars
    lb_bars = max(3, int(round(trade_cfg.break_lookback * 60 / max(int(bar_seconds), 1))))
    # stride for dense 1s: evaluate trail/smooth every ~5s
    stride = 1 if bar_seconds >= 30 else max(1, int(round(5 / max(int(bar_seconds), 1))))

    d = str(direction).upper()
    if d == "UP":
        peak = np.maximum.accumulate(closes)
        trough = np.minimum.accumulate(closes)
        mae_from_entry = np.maximum(0.0, 1.0 - closes / entry_px)
        trail_adverse = np.where(closes > 0, peak / closes - 1.0, 0.0)
        signed = closes / entry_px - 1.0
    else:
        peak = np.maximum.accumulate(closes)
        trough = np.minimum.accumulate(closes)
        mae_from_entry = np.maximum(0.0, closes / entry_px - 1.0)
        trail_adverse = np.where(trough > 0, closes / trough - 1.0, 0.0)
        signed = 1.0 - closes / entry_px

    exit_i = n - 1
    reason = "EOD"
    fd_fired = False

    # --- Fast FD MAE-only first touch ---
    if mae_only:
        in_win = (held_min >= fd_cfg.min_hold_minutes) & (held_min <= fd_cfg.max_eval_minutes)
        hit = np.where(in_win & (mae_from_entry >= fd_cfg.early_mae_cut))[0]
        if len(hit):
            exit_i = int(hit[0])
            reason = "FD_EARLY_MAE"
            fd_fired = True

    # Baseline exits (and full FD if not mae_only), earliest wins
    if not fd_fired:
        open_px = float(day.iloc[0]["open"] if "open" in day.columns else day.iloc[0]["close"])
        # precompute session vwap only if needed
        vwap_series = None
        if need_book and fd_cfg.lose_vwap:
            vol = (
                pd.to_numeric(day.get("volume", 1.0), errors="coerce").fillna(0.0).clip(lower=0.0).to_numpy()
            )
            cl_all = pd.to_numeric(day["close"], errors="coerce").to_numpy(dtype=float)
            csum_pv = np.cumsum(cl_all * vol)
            csum_v = np.cumsum(vol)
            vwap_all = np.where(csum_v > 0, csum_pv / np.maximum(csum_v, 1e-12), cl_all)
            # map after-index → day index via search
            day_ts = pd.to_datetime(day["timestamp"])
            after_ts = pd.to_datetime(after["timestamp"])
            pos = day_ts.searchsorted(after_ts, side="right") - 1
            pos = np.clip(pos, 0, len(vwap_all) - 1)
            vwap_series = vwap_all[pos]

        for i in range(stride, n, stride):
            if held_min[i] >= trade_cfg.max_hold_minutes:
                exit_i, reason = i, "TIME"
                break
            if int(hm[i]) >= eod_m:
                exit_i, reason = i, "EOD"
                break

            if fd_cfg.enabled and not mae_only:
                px = float(closes[i])
                op = open_px if need_book else None
                vw = float(vwap_series[i]) if vwap_series is not None else None
                action, fd_reason = evaluate_failure(
                    direction=direction,
                    entry_px=entry_px,
                    entry_ts=entry_ts,
                    now_ts=after["timestamp"].iloc[i],
                    now_px=px,
                    peak=float(peak[i]),
                    trough=float(trough[i]),
                    closes_since_entry=closes[: i + 1],
                    structure_extreme=structure,
                    open_px=op,
                    vwap=vw,
                    cfg=fd_cfg,
                )
                if action == "EXIT" and fd_reason:
                    exit_i, reason = i, fd_reason
                    fd_fired = True
                    break

            if trail_adverse[i] >= trade_cfg.break_max_adverse and held_min[i] >= arm_min:
                exit_i, reason = i, "TRAIL_BREAK"
                break
            if i >= lb_bars:
                w = closes[i - lb_bars : i + 1]
                st = _window_stats(w, direction=direction)
                if st is not None and st["up_frac"] < trade_cfg.break_min_up_frac and signed[i] < 0.001:
                    exit_i, reason = i, "SMOOTH_BREAK"
                    break
    else:
        # FD already fired; still allow earlier TIME/EOD only if before FD (shouldn't)
        pass

    # If mae_only FD fired, also check whether TIME/EOD/trail would have been earlier — no, FD wins.
    # But if FD did not fire, loop above set exit. If mae_only and no hit, run baseline loop:
    if mae_only and not fd_fired:
        for i in range(stride, n, stride):
            if held_min[i] >= trade_cfg.max_hold_minutes:
                exit_i, reason = i, "TIME"
                break
            if int(hm[i]) >= eod_m:
                exit_i, reason = i, "EOD"
                break
            if trail_adverse[i] >= trade_cfg.break_max_adverse and held_min[i] >= arm_min:
                exit_i, reason = i, "TRAIL_BREAK"
                break
            if i >= lb_bars:
                w = closes[i - lb_bars : i + 1]
                st = _window_stats(w, direction=direction)
                if st is not None and st["up_frac"] < trade_cfg.break_min_up_frac and signed[i] < 0.001:
                    exit_i, reason = i, "SMOOTH_BREAK"
                    break

    exit_px = float(closes[exit_i])
    exit_ts = pd.Timestamp(after["timestamp"].iloc[exit_i])
    raw = exit_px / entry_px - 1.0
    signed_ret = raw if d == "UP" else -raw
    cost = 2.0 * (float(trade_cfg.cost_bps) / 1e4)
    path = closes[: exit_i + 1]
    if d == "UP":
        mfe = float(path.max() / entry_px - 1.0)
        mae = float(1.0 - path.min() / entry_px)
    else:
        mfe = float(1.0 - path.min() / entry_px)
        mae = float(path.max() / entry_px - 1.0)
    return {
        "entry_ts": entry_ts,
        "exit_ts": exit_ts,
        "entry_px": entry_px,
        "exit_px": exit_px,
        "raw_stock_ret": float(raw),
        "ret": float(signed_ret - cost),
        "exit_reason": reason,
        "hold_minutes": float((exit_ts - entry_ts).total_seconds() / 60.0),
        "fd_fired": bool(fd_fired),
        "mfe": mfe,
        "mae": mae,
        "sleeve_cfg": fd_cfg.sleeve,
    }

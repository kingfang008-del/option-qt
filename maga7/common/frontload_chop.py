"""Causal FRONTLOAD_CHOP day label: H1 move done by 10:30, then chop risk.

Decision clock is **10:30 ET** — only uses prints/bars with timestamp ≤ 10:30.
Intended for CORE de-weight / block-new after the open book is already priced.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd

NY = "America/New_York"


@dataclass(frozen=True)
class FrontloadChopConfig:
    enabled: bool = True
    decision_tod: str = "10:30"
    mid_tod: str = "10:00"  # split H1 into open→mid vs mid→decision
    # Mag7 |open→decision| median threshold
    min_med_abs_h1: float = 0.008
    # count of names with |h1| ≥ this
    min_name_abs_h1: float = 0.008
    min_n_large: int = 4
    # late-H1 quiet: median |1m ret| over [quiet_start, decision)
    quiet_start_tod: str = "10:15"
    max_quiet_abs_1m: float = 0.00085
    require_quiet: bool = True
    # deceleration: median |open→10:00| / |10:00→10:30| ≥ ratio (impulse then stall)
    require_decel: bool = True
    min_decel_ratio: float = 1.85
    min_med_abs_first: float = 0.006  # need real first-half move
    # action when labeled
    mode: str = "scale"  # scale | block
    size_scale: float = 0.5
    # Sub-state overlay: only act on frontload days when regime looks weak/choppy.
    # overlay="always" → year-round AND (v2). overlay="weak" → require weak_substate.
    overlay: str = "always"  # always | weak
    overlay_combine: str = "or"  # or | and among enabled predicates
    overlay_vixy_z_min: float | None = None  # fire if vixy_z >= thr
    overlay_max_abs_qqq_fp: float | None = None  # fire if |qqq_from_prev| <= thr
    overlay_on_flip: bool = False  # fire if qqq_day_flipped


def _opt_float(v: Any) -> float | None:
    if v is None or v == "" or (isinstance(v, str) and v.strip().lower() in {"none", "null"}):
        return None
    return float(v)


def parse_frontload_chop(raw: Any) -> FrontloadChopConfig:
    if not raw or not isinstance(raw, Mapping):
        return FrontloadChopConfig(enabled=False)
    d = dict(raw)
    mode = str(d.get("mode") or "scale").strip().lower()
    if mode not in {"scale", "block", "size"}:
        mode = "scale"
    if mode == "size":
        mode = "scale"
    overlay = str(d.get("overlay") or "always").strip().lower()
    if overlay not in {"always", "weak"}:
        overlay = "always"
    combine = str(d.get("overlay_combine") or "or").strip().lower()
    if combine not in {"or", "and"}:
        combine = "or"
    return FrontloadChopConfig(
        enabled=bool(d.get("enabled", True)),
        decision_tod=str(d.get("decision_tod") or "10:30"),
        mid_tod=str(d.get("mid_tod") or "10:00"),
        min_med_abs_h1=float(d.get("min_med_abs_h1", 0.008) or 0.008),
        min_name_abs_h1=float(d.get("min_name_abs_h1", 0.008) or 0.008),
        min_n_large=int(d.get("min_n_large", 4) or 4),
        quiet_start_tod=str(d.get("quiet_start_tod") or "10:15"),
        max_quiet_abs_1m=float(d.get("max_quiet_abs_1m", 0.00085) or 0.00085),
        require_quiet=bool(d.get("require_quiet", True)),
        require_decel=bool(d.get("require_decel", True)),
        min_decel_ratio=float(d.get("min_decel_ratio", 1.85) or 1.85),
        min_med_abs_first=float(d.get("min_med_abs_first", 0.006) or 0.006),
        mode=mode,
        size_scale=float(d.get("size_scale", 0.5) or 0.5),
        overlay=overlay,
        overlay_combine=combine,
        overlay_vixy_z_min=_opt_float(d.get("overlay_vixy_z_min")),
        overlay_max_abs_qqq_fp=_opt_float(d.get("overlay_max_abs_qqq_fp")),
        overlay_on_flip=bool(d.get("overlay_on_flip", False)),
    )


def weak_substate_ok(dec: Any, cfg: FrontloadChopConfig) -> bool:
    """True if entry-time regime looks weak/choppy enough to apply FRONTLOAD action."""
    preds: list[bool] = []
    if cfg.overlay_vixy_z_min is not None:
        vz = getattr(dec, "vixy_z", None)
        preds.append(
            vz is not None and np.isfinite(float(vz)) and float(vz) >= float(cfg.overlay_vixy_z_min)
        )
    if cfg.overlay_max_abs_qqq_fp is not None:
        fp = getattr(dec, "qqq_from_prev", None)
        preds.append(
            fp is not None
            and np.isfinite(float(fp))
            and abs(float(fp)) <= float(cfg.overlay_max_abs_qqq_fp)
        )
    if cfg.overlay_on_flip:
        preds.append(bool(getattr(dec, "qqq_day_flipped", False)))
    if not preds:
        # weak overlay with no predicates configured → never fire (fail closed)
        return False
    if cfg.overlay_combine == "and":
        return all(preds)
    return any(preds)


def _hm(tod: str) -> int:
    h, m = str(tod).split(":")
    return int(h) * 60 + int(m)


def _day_frame(day: pd.DataFrame) -> pd.DataFrame:
    d = day.sort_values("timestamp").copy()
    ts = pd.to_datetime(d["timestamp"])
    if getattr(ts.dt, "tz", None) is None:
        ts = ts.dt.tz_localize(NY)
    else:
        ts = ts.dt.tz_convert(NY)
    d["ts"] = ts
    d["hm"] = d["ts"].dt.hour * 60 + d["ts"].dt.minute
    return d


def _open_px(d: pd.DataFrame) -> float | None:
    if d.empty:
        return None
    if "open" in d.columns and np.isfinite(float(d.iloc[0]["open"])):
        o = float(d.iloc[0]["open"])
        if o > 0:
            return o
    c = float(d.iloc[0]["close"])
    return c if c > 0 else None


def _px_at_hm(rth: pd.DataFrame, hm: int) -> float | None:
    sub = rth[rth["hm"] <= hm]
    if sub.empty:
        return None
    px = float(sub.iloc[-1]["close"])
    return px if px > 0 else None


def symbol_h1_stats(
    day: pd.DataFrame,
    *,
    decision_tod: str = "10:30",
    mid_tod: str = "10:00",
    quiet_start_tod: str = "10:15",
) -> dict[str, float] | None:
    """Causal open→decision ret, first/second half, late-window quiet."""
    d = _day_frame(day)
    if d.empty:
        return None
    dec = _hm(decision_tod)
    mid = _hm(mid_tod)
    q0 = _hm(quiet_start_tod)
    rth = d[(d["hm"] >= 9 * 60 + 30) & (d["hm"] <= dec)]
    if rth.empty:
        return None
    o = _open_px(rth)
    if o is None:
        return None
    px_mid = _px_at_hm(rth, mid)
    px_dec = float(rth.iloc[-1]["close"])
    ret_h1 = px_dec / o - 1.0
    ret_first = (px_mid / o - 1.0) if px_mid is not None else float("nan")
    ret_second = (px_dec / px_mid - 1.0) if px_mid is not None else float("nan")
    quiet = rth[(rth["hm"] >= q0) & (rth["hm"] < dec)]
    if len(quiet) >= 3:
        px = quiet["close"].astype(float).to_numpy()
        # prefer 1m stride if 1s bars
        step = 60 if len(px) > 120 else 1
        samp = px[::step]
        if len(samp) >= 3:
            r = np.diff(samp) / samp[:-1]
            quiet_abs = float(np.nanmedian(np.abs(r)))
        else:
            quiet_abs = float("nan")
    else:
        quiet_abs = float("nan")
    hi = float(rth["high"].max()) if "high" in rth.columns else float(rth["close"].max())
    lo = float(rth["low"].min()) if "low" in rth.columns else float(rth["close"].min())
    abs_first = abs(ret_first) if np.isfinite(ret_first) else float("nan")
    abs_second = abs(ret_second) if np.isfinite(ret_second) else float("nan")
    decel = (
        abs_first / max(abs_second, 1e-6)
        if np.isfinite(abs_first) and np.isfinite(abs_second)
        else float("nan")
    )
    return {
        "ret_h1": float(ret_h1),
        "abs_h1": float(abs(ret_h1)),
        "ret_first": float(ret_first) if np.isfinite(ret_first) else float("nan"),
        "ret_second": float(ret_second) if np.isfinite(ret_second) else float("nan"),
        "abs_first": float(abs_first) if np.isfinite(abs_first) else float("nan"),
        "abs_second": float(abs_second) if np.isfinite(abs_second) else float("nan"),
        "decel_ratio": float(decel) if np.isfinite(decel) else float("nan"),
        "range_h1": float((hi - lo) / o) if o else float("nan"),
        "quiet_abs_1m": float(quiet_abs),
    }


def label_frontload_day(
    stock_by_date: Mapping[str, pd.DataFrame],
    *,
    symbols: list[str],
    cfg: FrontloadChopConfig | None = None,
) -> dict[str, Any]:
    """Label one session from per-symbol day frames (1s or 1m)."""
    cfg = cfg or FrontloadChopConfig()
    rows: list[dict[str, Any]] = []
    for sym in symbols:
        day = stock_by_date.get(sym)
        if day is None or getattr(day, "empty", True):
            continue
        st = symbol_h1_stats(
            day,
            decision_tod=cfg.decision_tod,
            mid_tod=cfg.mid_tod,
            quiet_start_tod=cfg.quiet_start_tod,
        )
        if st is None:
            continue
        rows.append({"symbol": str(sym).upper(), **st})
    if not rows:
        return {
            "is_frontload": False,
            "reason": "no_symbols",
            "n_names": 0,
            "n_large": 0,
            "med_abs_h1": None,
            "med_quiet_abs_1m": None,
            "med_abs_first": None,
            "med_decel_ratio": None,
        }
    df = pd.DataFrame(rows)
    med_abs = float(df["abs_h1"].median())
    n_large = int((df["abs_h1"] >= float(cfg.min_name_abs_h1)).sum())
    med_q = float(df["quiet_abs_1m"].median()) if df["quiet_abs_1m"].notna().any() else float("nan")
    med_first = (
        float(df["abs_first"].median()) if "abs_first" in df.columns and df["abs_first"].notna().any() else float("nan")
    )
    med_decel = (
        float(df["decel_ratio"].median())
        if "decel_ratio" in df.columns and df["decel_ratio"].notna().any()
        else float("nan")
    )
    quiet_ok = (not cfg.require_quiet) or (
        np.isfinite(med_q) and med_q <= float(cfg.max_quiet_abs_1m)
    )
    decel_ok = (not cfg.require_decel) or (
        np.isfinite(med_decel)
        and med_decel >= float(cfg.min_decel_ratio)
        and np.isfinite(med_first)
        and med_first >= float(cfg.min_med_abs_first)
    )
    base_ok = med_abs >= float(cfg.min_med_abs_h1) and n_large >= int(cfg.min_n_large)
    is_fl = bool(base_ok and quiet_ok and decel_ok)
    if is_fl:
        reason = "frontload_chop"
    elif base_ok and quiet_ok and not decel_ok:
        reason = "h1_quiet_but_no_decel"
    elif base_ok and decel_ok and not quiet_ok:
        reason = "h1_decel_but_not_quiet"
    elif base_ok:
        reason = "h1_large_but_filters"
    else:
        reason = "not_frontload"
    return {
        "is_frontload": is_fl,
        "reason": reason,
        "n_names": int(len(df)),
        "n_large": n_large,
        "med_abs_h1": med_abs,
        "med_quiet_abs_1m": None if not np.isfinite(med_q) else med_q,
        "med_abs_first": None if not np.isfinite(med_first) else med_first,
        "med_decel_ratio": None if not np.isfinite(med_decel) else med_decel,
        "med_ret_h1": float(df["ret_h1"].median()),
        "names": df.to_dict(orient="records"),
    }


def build_frontload_day_table(
    stock_by: Mapping[str, pd.DataFrame],
    *,
    dates: list[str],
    symbols: list[str],
    cfg: FrontloadChopConfig | None = None,
) -> pd.DataFrame:
    """Scan dates → one row per day label (causal @ decision_tod)."""
    cfg = cfg or FrontloadChopConfig()
    out_rows: list[dict[str, Any]] = []
    for date in dates:
        by_sym: dict[str, pd.DataFrame] = {}
        for sym in symbols:
            df = stock_by.get(sym)
            if df is None or df.empty or "date" not in df.columns:
                continue
            sub = df[df["date"].astype(str) == str(date)]
            if not sub.empty:
                by_sym[str(sym).upper()] = sub
        lab = label_frontload_day(by_sym, symbols=list(by_sym.keys()), cfg=cfg)
        out_rows.append({"date": str(date), **{k: v for k, v in lab.items() if k != "names"}})
    return pd.DataFrame(out_rows)


@dataclass
class FrontloadChopGate:
    """Wrap Mag7RegimeGate: on frontload days scale or block new CORE entries."""

    inner: Any
    day_flags: dict[str, bool]
    fl_cfg: FrontloadChopConfig
    n_scale: int = 0
    n_block: int = 0
    n_overlay_skip: int = 0

    @property
    def cfg(self) -> dict[str, Any]:
        # Watchdog mutates inner.cfg in place.
        return self.inner.cfg

    @cfg.setter
    def cfg(self, value: dict[str, Any]) -> None:
        self.inner.cfg = value

    def check(self, direction: str, ts: pd.Timestamp) -> Any:
        from maga7.common.regime import RegimeDecision

        dec = self.inner.check(direction, ts)
        if not self.fl_cfg.enabled:
            return dec
        t = pd.Timestamp(ts)
        if t.tzinfo is None:
            t = t.tz_localize(NY)
        else:
            t = t.tz_convert(NY)
        date = str(t.strftime("%Y-%m-%d"))
        if not self.day_flags.get(date, False):
            return dec
        if not dec.allow:
            return dec
        if self.fl_cfg.overlay == "weak" and not weak_substate_ok(dec, self.fl_cfg):
            self.n_overlay_skip += 1
            return dec
        if self.fl_cfg.mode == "block":
            self.n_block += 1
            return RegimeDecision(
                allow=False,
                reason="frontload_chop",
                qqq_from_prev=dec.qqq_from_prev,
                qqq_mf10=dec.qqq_mf10,
                vix_reversal=dec.vix_reversal,
                vixy_z=dec.vixy_z,
                size_scale=0.0,
                qqq_day_flipped=dec.qqq_day_flipped,
            )
        self.n_scale += 1
        scale = float(dec.size_scale) * float(self.fl_cfg.size_scale)
        return RegimeDecision(
            allow=True,
            reason=f"{dec.reason}+frontload_scale",
            qqq_from_prev=dec.qqq_from_prev,
            qqq_mf10=dec.qqq_mf10,
            vix_reversal=dec.vix_reversal,
            vixy_z=dec.vixy_z,
            size_scale=scale,
            qqq_day_flipped=dec.qqq_day_flipped,
        )


__all__ = [
    "FrontloadChopConfig",
    "FrontloadChopGate",
    "build_frontload_day_table",
    "label_frontload_day",
    "parse_frontload_chop",
    "symbol_h1_stats",
    "weak_substate_ok",
]

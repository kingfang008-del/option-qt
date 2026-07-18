"""Mag7 regime gates — QQQ alignment + VIXY chop/put gate (QQQ-inspired, rule-only).

Does **not** use TFT/FCS scores. Optional veto before Mag7 entries/reentries.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from maga7.common.signals import attach_mf_features, load_stock_month_files
from qqq_btc.common.regime_features import add_vix_regime_features

NY = "America/New_York"


def _vixy_z_series(close: pd.Series, *, window: int = 60, min_periods: int = 20) -> pd.Series:
    """Causal rolling z-score of VIXY close (same formula family as QQQ put_gate / vix_level)."""
    x = pd.to_numeric(close, errors="coerce")
    mu = x.rolling(window, min_periods=min_periods).mean()
    sd = x.rolling(window, min_periods=min_periods).std(ddof=1)
    return (x - mu) / (sd + 1e-6)


def _signed(x: float | None, eps: float = 0.0) -> int:
    if x is None or not np.isfinite(x):
        return 0
    if abs(float(x)) <= float(eps):
        return 0
    return 1 if float(x) > 0 else -1


def build_regime_frame(
    stock_root,
    months: list[str],
    *,
    start: str,
    end: str,
    vix_reversal_window: int = 30,
    vix_reversal_pct: float = 0.0015,
) -> pd.DataFrame:
    """Merge QQQ from_prev + VIXY reversal count + vixy_z on 1m timestamps."""
    qqq = load_stock_month_files(stock_root, "QQQ", months)
    vixy = load_stock_month_files(stock_root, "VIXY", months)
    if qqq.empty and vixy.empty:
        return pd.DataFrame()

    frames = []
    if not qqq.empty:
        qqq = qqq[(qqq["date"] >= start) & (qqq["date"] <= end)].copy()
        qqq = attach_mf_features(qqq, mf_window=10, vol_ma_window=20)
        keep = ["timestamp", "date", "close", "from_prev", "mf10"]
        if "open" in qqq.columns:
            keep.append("open")
        q = qqq[keep].rename(
            columns={
                "close": "qqq_close",
                "open": "qqq_open",
                "from_prev": "qqq_from_prev",
                "mf10": "qqq_mf10",
            }
        )
        frames.append(q.set_index("timestamp"))

    if not vixy.empty:
        vixy = vixy[(vixy["date"] >= start) & (vixy["date"] <= end)].copy()
        v = vixy[["timestamp", "close"]].rename(columns={"close": "vixy_close"})
        v = v.set_index("timestamp").sort_index()
        v["vixy_z"] = _vixy_z_series(v["vixy_close"])
        # reuse QQQ regime_features on VIXY close
        tmp = v.reset_index().copy()
        tmp["vix_proxy_close"] = tmp["vixy_close"]
        tmp = add_vix_regime_features(
            tmp,
            vix_col="vix_proxy_close",
            window=vix_reversal_window,
            threshold=vix_reversal_pct,
        )
        v = tmp.set_index("timestamp")[["vixy_close", "vixy_z", "vix_reversal_count_30m"]]
        frames.append(v)

    if not frames:
        return pd.DataFrame()
    out = frames[0]
    for f in frames[1:]:
        out = out.join(f, how="outer")
    out = out.sort_index()
    if "date" not in out.columns:
        out["date"] = pd.Index(out.index).tz_convert(NY).strftime("%Y-%m-%d")
    # forward-fill within day for QQQ/VIXY clock gaps
    cols = [c for c in out.columns if c != "date"]
    out[cols] = out.groupby("date", sort=False)[cols].ffill()
    return out


@dataclass
class RegimeDecision:
    allow: bool
    reason: str
    qqq_from_prev: float | None = None
    qqq_mf10: float | None = None
    vix_reversal: float | None = None
    vixy_z: float | None = None
    size_scale: float = 1.0
    qqq_day_flipped: bool = False


@dataclass
class Mag7RegimeGate:
    """Causal lookup: given Mag7 direction + timestamp → allow/deny (optional size scale)."""

    frame: pd.DataFrame
    cfg: dict[str, Any]
    _day_qqq_end: dict[str, float] = field(default_factory=dict, init=False, repr=False)
    _day_qqq_open: dict[str, float] = field(default_factory=dict, init=False, repr=False)
    _day_order: list[str] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        self._build_day_qqq_ends()

    def _build_day_qqq_ends(self) -> None:
        """Last finite qqq_from_prev per session (for day-over-day flip)."""
        self._day_qqq_end = {}
        self._day_qqq_open = {}
        self._day_order = []
        if self.frame is None or self.frame.empty:
            return
        if "date" not in self.frame.columns:
            return
        for date, g in self.frame.groupby("date", sort=True):
            d = str(date)
            self._day_order.append(d)
            if "qqq_from_prev" in g.columns:
                s = pd.to_numeric(g["qqq_from_prev"], errors="coerce").dropna()
                if not s.empty:
                    self._day_qqq_end[d] = float(s.iloc[-1])
            # Session open: prefer first finite qqq_open, else first qqq_close.
            if "qqq_open" in g.columns:
                o = pd.to_numeric(g["qqq_open"], errors="coerce").dropna()
                if not o.empty:
                    self._day_qqq_open[d] = float(o.iloc[0])
            if d not in self._day_qqq_open and "qqq_close" in g.columns:
                c = pd.to_numeric(g["qqq_close"], errors="coerce").dropna()
                if not c.empty:
                    self._day_qqq_open[d] = float(c.iloc[0])

    def _prior_day_qqq(self, date: str) -> float | None:
        if date not in self._day_qqq_end:
            # still allow lookup of prior among known days
            dates = self._day_order
        else:
            dates = self._day_order
        try:
            i = dates.index(date)
        except ValueError:
            # date not in map — find last prior
            priors = [d for d in dates if d < date]
            if not priors:
                return None
            return self._day_qqq_end.get(priors[-1])
        if i <= 0:
            return None
        return self._day_qqq_end.get(dates[i - 1])

    @classmethod
    def from_profile(cls, profile: dict[str, Any], months: list[str] | None = None) -> "Mag7RegimeGate | None":
        reg = profile.get("regime") or {}
        if not reg.get("enabled"):
            return None
        paths = profile["_paths"]
        start = profile["date_range"]["start"]
        end = profile["date_range"]["end"]
        from maga7.common.replay import month_list

        months = months or month_list(start, end)
        frame = build_regime_frame(
            paths["stock_root"],
            months,
            start=start,
            end=end,
            vix_reversal_window=int(reg.get("vix_reversal_window", 30)),
            vix_reversal_pct=float(reg.get("vix_reversal_pct", 0.0015)),
        )
        if frame.empty:
            return None
        # Deepcopy: Watchdog/router overlays mutate cfg in-place; must not alias profile["regime"].
        return cls(frame=frame, cfg=copy.deepcopy(reg))

    def _row_at(self, ts: pd.Timestamp) -> pd.Series | None:
        if self.frame.empty:
            return None
        t = pd.Timestamp(ts)
        if t.tzinfo is None:
            t = t.tz_localize(NY)
        else:
            t = t.tz_convert(NY)
        # last bar at or before ts
        idx = self.frame.index
        pos = idx.searchsorted(t, side="right") - 1
        if pos < 0:
            return None
        return self.frame.iloc[pos]

    def check(self, direction: str, ts: pd.Timestamp) -> RegimeDecision:
        row = self._row_at(ts)
        block_missing = bool(self.cfg.get("block_on_missing", False))
        if row is None:
            return RegimeDecision(allow=not block_missing, reason="regime_missing")

        qfp = row.get("qqq_from_prev")
        qmf = row.get("qqq_mf10")
        vrev = row.get("vix_reversal_count_30m")
        vz = row.get("vixy_z")
        qfp_f = float(qfp) if qfp is not None and np.isfinite(qfp) else None
        qmf_f = float(qmf) if qmf is not None and np.isfinite(qmf) else None
        vrev_f = float(vrev) if vrev is not None and np.isfinite(vrev) else None
        vz_f = float(vz) if vz is not None and np.isfinite(vz) else None

        date = None
        if "date" in row.index and pd.notna(row.get("date")):
            date = str(row.get("date"))
        else:
            t = pd.Timestamp(ts)
            if t.tzinfo is None:
                t = t.tz_localize(NY)
            else:
                t = t.tz_convert(NY)
            date = t.strftime("%Y-%m-%d")

        flip_eps = float(self.cfg.get("qqq_day_flip_eps", self.cfg.get("qqq_from_prev_eps", 0.0)) or 0.0)
        cur_s = _signed(qfp_f, flip_eps)
        prior = self._prior_day_qqq(date) if date else None
        prior_s = _signed(prior, flip_eps)
        day_flipped = bool(cur_s != 0 and prior_s != 0 and cur_s != prior_s)

        def _dec(
            allow: bool,
            reason: str,
            *,
            size_scale: float = 1.0,
        ) -> RegimeDecision:
            return RegimeDecision(
                allow,
                reason,
                qqq_from_prev=qfp_f,
                qqq_mf10=qmf_f,
                vix_reversal=vrev_f,
                vixy_z=vz_f,
                size_scale=float(size_scale),
                qqq_day_flipped=day_flipped,
            )

        # 1) VIX chop gate (QQQ regime_vix_reversal)
        vmax = self.cfg.get("vix_reversal_max")
        if vmax is not None and vrev_f is not None and vrev_f > float(vmax):
            return _dec(False, "vix_reversal")

        # 2) QQQ from_prev align
        if bool(self.cfg.get("qqq_align", True)):
            eps = float(self.cfg.get("qqq_from_prev_eps", 0.0))
            if qfp_f is None:
                if block_missing:
                    return _dec(False, "qqq_missing")
            elif direction == "UP" and qfp_f < -eps:
                return _dec(False, "qqq_align_up")
            elif direction == "DN" and qfp_f > eps:
                return _dec(False, "qqq_align_dn")

        # 2b) DN blocked when QQQ already above today's open (gap-fill rebound days)
        if bool(self.cfg.get("block_dn_if_qqq_above_open", False)) and direction == "DN":
            q_open = self._day_qqq_open.get(date) if date else None
            q_px = row.get("qqq_close")
            q_px_f = float(q_px) if q_px is not None and np.isfinite(q_px) else None
            if q_open is not None and q_px_f is not None and q_px_f > float(q_open):
                return _dec(False, "qqq_above_open_dn")

        # 2b') Router / expert: hard-block listed directions (default off)
        block_dirs = self.cfg.get("block_directions") or ()
        if isinstance(block_dirs, str):
            block_dirs = [x.strip() for x in block_dirs.split(",") if x.strip()]
        block_dirs_u = {str(x).upper() for x in block_dirs}
        if direction in block_dirs_u:
            return _dec(False, f"block_dir_{direction.lower()}")

        # 3) QQQ mf10 same-sign align (stronger than from_prev alone)
        if bool(self.cfg.get("qqq_mf10_align", False)):
            if qmf_f is None:
                if block_missing:
                    return _dec(False, "qqq_mf10_missing")
            elif direction == "UP" and qmf_f <= 0:
                return _dec(False, "qqq_mf10_up")
            elif direction == "DN" and qmf_f >= 0:
                return _dec(False, "qqq_mf10_dn")

        # 4) Put / DN requires elevated VIXY z
        put_min = self.cfg.get("put_vixy_z_min")
        if put_min is not None and direction == "DN":
            if vz_f is None:
                if block_missing:
                    return _dec(False, "vixy_z_missing")
            elif vz_f < float(put_min):
                return _dec(False, "put_vixy_z")

        # 5) Day-over-day QQQ from_prev sign flip → block or scale
        #    ``qqq_day_flip_mode``: off|block|scale
        #    legacy bool ``block_after_qqq_day_flip`` → block
        flip_mode = str(self.cfg.get("qqq_day_flip_mode") or "").strip().lower()
        if not flip_mode:
            if bool(self.cfg.get("block_after_qqq_day_flip", False)):
                flip_mode = "block"
            else:
                flip_mode = "off"
        if flip_mode in {"block", "skip"} and day_flipped:
            return _dec(False, "qqq_day_flip")
        scale = 1.0
        reason = "ok"
        if flip_mode in {"scale", "half", "reduce"} and day_flipped:
            scale = float(self.cfg.get("qqq_day_flip_scale", 0.5))
            scale = max(0.0, min(scale, 1.0))
            if scale <= 0.0:
                return _dec(False, "qqq_day_flip")
            reason = "qqq_day_flip_scale"

        # 2c) Softer: scale DN size when QQQ above open (default off; prefer over hard block)
        scale_dn_qqq = self.cfg.get("scale_dn_if_qqq_above_open")
        if scale_dn_qqq is not None and direction == "DN":
            q_open = self._day_qqq_open.get(date) if date else None
            q_px = row.get("qqq_close")
            q_px_f = float(q_px) if q_px is not None and np.isfinite(q_px) else None
            if q_open is not None and q_px_f is not None and q_px_f > float(q_open):
                sc = max(0.0, min(float(scale_dn_qqq), 1.0))
                if sc <= 0.0:
                    return _dec(False, "qqq_above_open_dn")
                if sc < 1.0:
                    scale = float(scale) * sc
                    reason = (
                        "qqq_above_open_dn_scale"
                        if reason == "ok"
                        else f"{reason}+qqq_above_open_dn_scale"
                    )

        # 2d) Router / expert: always scale a direction (default off)
        dir_scales = self.cfg.get("direction_size_scale") or {}
        if isinstance(dir_scales, dict) and direction in dir_scales:
            sc = max(0.0, min(float(dir_scales[direction]), 1.0))
            if sc <= 0.0:
                return _dec(False, f"dir_scale_{direction.lower()}")
            if sc < 1.0:
                scale = float(scale) * sc
                reason = (
                    f"dir_scale_{direction.lower()}"
                    if reason == "ok"
                    else f"{reason}+dir_scale_{direction.lower()}"
                )

        return _dec(True, reason, size_scale=float(scale))

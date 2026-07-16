"""Mag7 regime gates — QQQ alignment + VIXY chop/put gate (QQQ-inspired, rule-only).

Does **not** use TFT/FCS scores. Optional veto before Mag7 entries/reentries.
"""
from __future__ import annotations

from dataclasses import dataclass
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
        q = qqq[["timestamp", "date", "close", "from_prev", "mf10"]].rename(
            columns={"close": "qqq_close", "from_prev": "qqq_from_prev", "mf10": "qqq_mf10"}
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


@dataclass
class Mag7RegimeGate:
    """Causal lookup: given Mag7 direction + timestamp → allow/deny."""

    frame: pd.DataFrame
    cfg: dict[str, Any]

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
        return cls(frame=frame, cfg=reg)

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

        def _dec(allow: bool, reason: str) -> RegimeDecision:
            return RegimeDecision(
                allow,
                reason,
                qqq_from_prev=qfp_f,
                qqq_mf10=qmf_f,
                vix_reversal=vrev_f,
                vixy_z=vz_f,
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

        return _dec(True, "ok")


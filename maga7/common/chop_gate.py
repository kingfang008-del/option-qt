"""Calm-range / mixed-tape day gate (chop overlay).

Distinct from Watchdog toxic days (washout / reclaim): those are violent opens.
Chop here means **index already ranging by 10:30** while Mag7 breadth is mixed —
false-break fuel for Rule-A 0DTE.

Causal features at ``asof`` (default 10:30):
  - ``q_am``: QQQ close/open - 1
  - ``q_rng``: QQQ (high-low)/open
  - ``frac_above``: share of Mag7 closes > open
  - ``med_abs``: median |stock from_open|

Default rule ``stock_noise`` (RTH bars only, no premarket):
  ``|q_am| <= q_am_max`` AND ``med_abs >= med_abs_min`` AND ``frac in [lo, hi]``
  Optional ``q_rng_min`` (0 = ignore).

Modes: ``scale`` (soft size) | ``block`` (skip baseline entries that day).
Does **not** emit Call/Put; parallel to ``state_gate``.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ChopGateConfig:
    enabled: bool = False
    asof: str = "10:30"
    rule: str = "stock_noise"
    mode: str = "scale"  # scale | block
    scale: float = 0.5
    q_am_max: float = 0.005
    # 0 disables the QQQ range leg (RTH 09:30–asof range is often too quiet).
    q_rng_min: float = 0.0
    frac_above_lo: float = 0.35
    frac_above_hi: float = 0.50
    med_abs_min: float | None = 0.01
    med_abs_max: float | None = None
    symbols: list[str] | None = None

    @classmethod
    def from_profile(cls, profile: dict[str, Any] | None) -> "ChopGateConfig":
        raw = (profile or {}).get("chop_gate")
        if not isinstance(raw, dict):
            return cls(enabled=False)
        mode = str(raw.get("mode") or "scale").strip().lower()
        if mode in {"soft", "size", "half"}:
            mode = "scale"
        if mode in {"hard", "reject", "skip"}:
            mode = "block"
        if mode not in {"scale", "block"}:
            mode = "scale"
        scale = float(raw.get("scale", 0.5) or 0.5)
        scale = max(0.0, min(1.0, scale))
        syms = raw.get("symbols")
        if isinstance(syms, str):
            syms = [x.strip().upper() for x in syms.split(",") if x.strip()]
        elif isinstance(syms, list):
            syms = [str(x).upper() for x in syms]
        else:
            syms = None
        rule = str(raw.get("rule") or "stock_noise").strip().lower()
        # Legacy alias from first draft (premarket-calibrated wide_mix).
        if rule in {"wide_mix", "mix", "default"}:
            rule = "stock_noise"
        med_min = raw.get("med_abs_min", 0.01)
        med_max = raw.get("med_abs_max", None)
        return cls(
            enabled=bool(raw.get("enabled", False)),
            asof=str(raw.get("asof") or "10:30"),
            rule=rule,
            mode=mode,
            scale=scale,
            q_am_max=float(raw.get("q_am_max", 0.005) or 0.005),
            q_rng_min=float(raw.get("q_rng_min", 0.0) or 0.0),
            frac_above_lo=float(raw.get("frac_above_lo", 0.35) or 0.35),
            frac_above_hi=float(raw.get("frac_above_hi", 0.50) or 0.50),
            med_abs_min=float(med_min) if med_min is not None else None,
            med_abs_max=float(med_max) if med_max is not None else None,
            symbols=syms,
        )


@dataclass
class ChopGateDayDecision:
    enabled: bool
    date: str
    asof: str
    state: str  # off | trend | chop | unknown
    reason: str
    features: dict[str, float | None] = field(default_factory=dict)
    block_all: bool = False
    size_scale: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ChopGateEntryDecision:
    allow: bool
    size_scale: float
    state: str
    reason: str


def _asof_ts(date: str, asof: str) -> pd.Timestamp:
    hh, mm = str(asof).split(":")
    return pd.Timestamp(f"{date} {int(hh):02d}:{int(mm):02d}:00", tz="America/New_York")


def _day_upto(df: pd.DataFrame | None, date: str, asof: pd.Timestamp) -> pd.DataFrame:
    if df is None or getattr(df, "empty", True):
        return pd.DataFrame()
    g = df[df["date"].astype(str) == str(date)]
    if g.empty:
        return g
    ts = pd.to_datetime(g["timestamp"])
    if ts.dt.tz is None:
        # compare naive
        asof_cmp = asof.tz_localize(None) if asof.tzinfo is not None else asof
        return g[ts <= asof_cmp].sort_values("timestamp")
    return g[ts <= asof].sort_values("timestamp")


def compute_chop_features(
    *,
    date: str,
    stock_by: dict[str, pd.DataFrame],
    qqq_df: pd.DataFrame | None,
    symbols: list[str],
    asof: str = "10:30",
) -> dict[str, float | None]:
    """Causal AM tape features; missing → None fields (fail-open later)."""
    asof_t = _asof_ts(str(date), asof)
    q = _day_upto(qqq_df if qqq_df is not None else stock_by.get("QQQ"), str(date), asof_t)
    out: dict[str, float | None] = {
        "q_am": None,
        "q_rng": None,
        "frac_above": None,
        "med_abs": None,
        "med_ret": None,
        "n_names": None,
    }
    if not q.empty and "open" in q.columns and "close" in q.columns:
        try:
            qo = float(q.iloc[0]["open"])
            qc = float(q.iloc[-1]["close"])
            if qo > 0:
                out["q_am"] = float(qc / qo - 1.0)
                if "high" in q.columns and "low" in q.columns:
                    qh = float(pd.to_numeric(q["high"], errors="coerce").max())
                    ql = float(pd.to_numeric(q["low"], errors="coerce").min())
                    out["q_rng"] = float((qh - ql) / qo)
        except (TypeError, ValueError, IndexError):
            pass
    rets: list[float] = []
    for sym in symbols:
        g = _day_upto(stock_by.get(str(sym)), str(date), asof_t)
        if g.empty or "open" not in g.columns or "close" not in g.columns:
            continue
        try:
            o = float(g.iloc[0]["open"])
            c = float(g.iloc[-1]["close"])
        except (TypeError, ValueError, IndexError):
            continue
        if o > 0:
            rets.append(float(c / o - 1.0))
    if rets:
        arr = np.asarray(rets, dtype=float)
        out["n_names"] = float(len(arr))
        out["frac_above"] = float(np.mean(arr > 0.0))
        out["med_abs"] = float(np.median(np.abs(arr)))
        out["med_ret"] = float(np.median(arr))
    return out


def is_chop_day(cfg: ChopGateConfig, feats: dict[str, float | None]) -> tuple[bool, str]:
    q_am = feats.get("q_am")
    q_rng = feats.get("q_rng")
    frac = feats.get("frac_above")
    med_abs = feats.get("med_abs")
    if q_am is None or frac is None:
        return False, "missing_features"
    if abs(float(q_am)) > float(cfg.q_am_max) + 1e-12:
        return False, "q_am_trend"
    if float(cfg.q_rng_min) > 0.0:
        if q_rng is None or float(q_rng) + 1e-12 < float(cfg.q_rng_min):
            return False, "q_rng_narrow"
    if not (float(cfg.frac_above_lo) - 1e-12 <= float(frac) <= float(cfg.frac_above_hi) + 1e-12):
        return False, "frac_consensus"
    if cfg.med_abs_min is not None:
        if med_abs is None or float(med_abs) + 1e-12 < float(cfg.med_abs_min):
            return False, "med_abs_low"
    if cfg.med_abs_max is not None:
        if med_abs is None or float(med_abs) > float(cfg.med_abs_max) + 1e-12:
            return False, "med_abs_high"
    rule = str(cfg.rule or "stock_noise")
    parts = [f"|q_am|<={cfg.q_am_max:g}", f"frac∈[{cfg.frac_above_lo:g},{cfg.frac_above_hi:g}]"]
    if cfg.med_abs_min is not None:
        parts.append(f"med_abs>={cfg.med_abs_min:g}")
    if float(cfg.q_rng_min) > 0.0:
        parts.append(f"q_rng>={cfg.q_rng_min:g}")
    return True, f"{rule}:" + "&".join(parts)


class ChopGate:
    def __init__(self, cfg: ChopGateConfig):
        self.cfg = cfg
        self._day: ChopGateDayDecision | None = None

    @classmethod
    def from_profile(cls, profile: dict[str, Any] | None) -> "ChopGate":
        return cls(ChopGateConfig.from_profile(profile))

    def begin_day(
        self,
        date: str,
        *,
        stock_by: dict[str, pd.DataFrame],
        qqq_df: pd.DataFrame | None,
        symbols: list[str],
    ) -> ChopGateDayDecision:
        if not self.cfg.enabled:
            dec = ChopGateDayDecision(
                enabled=False,
                date=str(date),
                asof=self.cfg.asof,
                state="off",
                reason="disabled",
            )
            self._day = dec
            return dec
        syms = list(self.cfg.symbols) if self.cfg.symbols else list(symbols)
        feats = compute_chop_features(
            date=str(date),
            stock_by=stock_by,
            qqq_df=qqq_df,
            symbols=syms,
            asof=self.cfg.asof,
        )
        hit, reason = is_chop_day(self.cfg, feats)
        if not hit:
            dec = ChopGateDayDecision(
                enabled=True,
                date=str(date),
                asof=self.cfg.asof,
                state="trend" if reason != "missing_features" else "unknown",
                reason=reason,
                features=feats,
                block_all=False,
                size_scale=1.0,
            )
            self._day = dec
            return dec
        if self.cfg.mode == "block":
            dec = ChopGateDayDecision(
                enabled=True,
                date=str(date),
                asof=self.cfg.asof,
                state="chop",
                reason=reason,
                features=feats,
                block_all=True,
                size_scale=0.0,
            )
        else:
            dec = ChopGateDayDecision(
                enabled=True,
                date=str(date),
                asof=self.cfg.asof,
                state="chop",
                reason=reason,
                features=feats,
                block_all=False,
                size_scale=float(self.cfg.scale),
            )
        self._day = dec
        return dec

    def decide_entry(self, direction: str | None = None) -> ChopGateEntryDecision:
        _ = direction  # direction-agnostic day overlay
        day = self._day
        if day is None or not day.enabled:
            return ChopGateEntryDecision(True, 1.0, "off", "disabled")
        if day.block_all or day.state == "chop" and self.cfg.mode == "block":
            return ChopGateEntryDecision(False, 0.0, day.state, f"block_{day.reason}")
        if day.state == "chop" and float(day.size_scale) < 1.0 - 1e-12:
            return ChopGateEntryDecision(True, float(day.size_scale), day.state, day.reason)
        return ChopGateEntryDecision(True, 1.0, day.state, day.reason)


def load_chop_gate(profile: dict[str, Any] | None) -> ChopGate:
    return ChopGate.from_profile(profile)

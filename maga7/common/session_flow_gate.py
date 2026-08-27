"""Proactive entry gate: QQQ+VIXY chop regime + session cumulative money-flow leaders.

Motivation (Jul10–23): index chops while single names trend. Sliding mf10/peer
align miss the day's capital leader; post-loss size cuts are too late.

Causal pieces:
  A) Index chop at ``asof`` (default 10:30):
       |QQQ from_open| <= q_am_max
       and (optional) |VIXY from_open| <= vixy_am_max
  B) Session cumflow leader at signal time:
       ``cum = cumsum(net$)`` from RTH open (already on attach_mf_features)
       require symbol in Top-K by |cum| among Mag7 and sign(cum) matches dir

``when``:
  - ``chop_only``: enforce B only on chop days; trend days pass
  - ``always``: always enforce B
  - ``chop_block``: chop days block all baseline (no leader escape)

Modes for B:
  - ``block`` / ``scale``: non-leader reject or downsize
  - ``boost``: leaders get ``boost`` (>1) size; non-leaders get ``non_leader_scale`` (default 1.0)
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

NY = "America/New_York"


@dataclass(frozen=True)
class SessionFlowGateConfig:
    enabled: bool = False
    asof: str = "10:30"
    when: str = "chop_only"  # chop_only | always | chop_block
    mode: str = "block"  # block | scale | boost
    scale: float = 0.5  # non-leader scale when mode=scale
    boost: float = 1.5  # leader size when mode=boost
    non_leader_scale: float = 1.0  # non-leader size when mode=boost
    q_am_max: float = 0.005
    vixy_am_max: float | None = 0.015  # None = ignore VIXY leg
    top_k: int = 2
    require_sign_align: bool = True
    min_abs_cum: float = 0.0
    symbols: list[str] | None = None

    @classmethod
    def from_profile(cls, profile: dict[str, Any] | None) -> "SessionFlowGateConfig":
        raw = (profile or {}).get("session_flow_gate")
        if not isinstance(raw, dict):
            return cls(enabled=False)
        mode = str(raw.get("mode") or "block").strip().lower()
        if mode in {"soft", "size", "half"}:
            mode = "scale"
        if mode in {"hard", "reject", "skip"}:
            mode = "block"
        if mode in {"leader_boost", "upsize", "boost_leader"}:
            mode = "boost"
        if mode not in {"block", "scale", "boost"}:
            mode = "block"
        when = str(raw.get("when") or "chop_only").strip().lower()
        if when in {"chop", "on_chop"}:
            when = "chop_only"
        if when in {"all", "always_on"}:
            when = "always"
        if when in {"block_chop", "chop_halt"}:
            when = "chop_block"
        if when not in {"chop_only", "always", "chop_block"}:
            when = "chop_only"
        scale = max(0.0, min(1.0, float(raw.get("scale", 0.5) or 0.5)))
        boost = max(1.0, min(3.0, float(raw.get("boost", 1.5) or 1.5)))
        nls = max(0.0, min(1.0, float(raw.get("non_leader_scale", 1.0) or 1.0)))
        syms = raw.get("symbols")
        if isinstance(syms, str):
            syms = [x.strip().upper() for x in syms.split(",") if x.strip()]
        elif isinstance(syms, list):
            syms = [str(x).upper() for x in syms]
        else:
            syms = None
        vixy_am = raw.get("vixy_am_max", 0.015)
        return cls(
            enabled=bool(raw.get("enabled", False)),
            asof=str(raw.get("asof") or "10:30"),
            when=when,
            mode=mode,
            scale=scale,
            boost=boost,
            non_leader_scale=nls,
            q_am_max=float(raw.get("q_am_max", 0.005) or 0.005),
            vixy_am_max=float(vixy_am) if vixy_am is not None else None,
            top_k=max(1, int(raw.get("top_k", 2) or 2)),
            require_sign_align=bool(raw.get("require_sign_align", True)),
            min_abs_cum=float(raw.get("min_abs_cum", 0.0) or 0.0),
            symbols=syms,
        )


@dataclass
class SessionFlowDayDecision:
    enabled: bool
    date: str
    asof: str
    state: str  # off | trend | chop | unknown
    reason: str
    features: dict[str, float | None] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SessionFlowEntryDecision:
    allow: bool
    size_scale: float
    state: str
    reason: str
    cum: float | None = None
    rank: int | None = None


def _asof_ts(date: str, asof: str) -> pd.Timestamp:
    hh, mm = str(asof).split(":")
    return pd.Timestamp(f"{date} {int(hh):02d}:{int(mm):02d}:00", tz=NY)


def _day_upto(df: pd.DataFrame | None, date: str, asof: pd.Timestamp) -> pd.DataFrame:
    if df is None or getattr(df, "empty", True):
        return pd.DataFrame()
    g = df[df["date"].astype(str) == str(date)]
    if g.empty:
        return g
    ts = pd.to_datetime(g["timestamp"])
    if getattr(ts.dt, "tz", None) is None:
        asof_cmp = asof.tz_localize(None) if asof.tzinfo is not None else asof
        return g[ts <= asof_cmp].sort_values("timestamp")
    return g[ts <= asof].sort_values("timestamp")


def _from_open(df: pd.DataFrame | None, date: str, asof: pd.Timestamp) -> float | None:
    g = _day_upto(df, date, asof)
    if g.empty or "open" not in g.columns or "close" not in g.columns:
        return None
    try:
        o = float(g.iloc[0]["open"])
        c = float(g.iloc[-1]["close"])
    except (TypeError, ValueError, IndexError):
        return None
    if o <= 0 or not np.isfinite(o) or not np.isfinite(c):
        return None
    return float(c / o - 1.0)


def session_cum_at(
    stock_df: pd.DataFrame | None,
    *,
    date: str,
    asof_ts: pd.Timestamp,
) -> float | None:
    """Causal session cumulative net$ at/before asof (uses ``cum`` or builds it)."""
    g = _day_upto(stock_df, date, asof_ts)
    if g.empty:
        return None
    if "cum" in g.columns and np.isfinite(g.iloc[-1]["cum"]):
        return float(g.iloc[-1]["cum"])
    if "net$" not in g.columns:
        return None
    s = pd.to_numeric(g["net$"], errors="coerce").fillna(0.0).cumsum()
    if s.empty or not np.isfinite(s.iloc[-1]):
        return None
    return float(s.iloc[-1])


def compute_index_chop_features(
    *,
    date: str,
    stock_by: dict[str, pd.DataFrame],
    qqq_df: pd.DataFrame | None,
    vixy_df: pd.DataFrame | None,
    asof: str = "10:30",
) -> dict[str, float | None]:
    asof_t = _asof_ts(str(date), asof)
    q = qqq_df if qqq_df is not None else stock_by.get("QQQ")
    v = vixy_df if vixy_df is not None else stock_by.get("VIXY")
    return {
        "q_am": _from_open(q, str(date), asof_t),
        "vixy_am": _from_open(v, str(date), asof_t),
    }


def is_index_chop(cfg: SessionFlowGateConfig, feats: dict[str, float | None]) -> tuple[bool, str]:
    q_am = feats.get("q_am")
    if q_am is None or not np.isfinite(q_am):
        return False, "missing_qqq"
    if abs(float(q_am)) > float(cfg.q_am_max) + 1e-12:
        return False, "q_am_trend"
    if cfg.vixy_am_max is not None:
        v_am = feats.get("vixy_am")
        # Fail-soft when VIXY cache missing (partial months); QQQ leg still binds.
        if v_am is not None and np.isfinite(v_am):
            if abs(float(v_am)) > float(cfg.vixy_am_max) + 1e-12:
                return False, "vixy_am_move"
    return True, f"chop_|q_am|<={cfg.q_am_max:g}" + (
        f"&|vixy_am|<={cfg.vixy_am_max:g}" if cfg.vixy_am_max is not None else ""
    )


def cumflow_ranks(
    *,
    date: str,
    asof_ts: pd.Timestamp,
    stock_by: dict[str, pd.DataFrame],
    symbols: list[str],
) -> dict[str, tuple[float, int]]:
    """symbol → (cum, rank_by_|cum| 1=largest)."""
    vals: list[tuple[str, float]] = []
    for sym in symbols:
        c = session_cum_at(stock_by.get(str(sym)), date=str(date), asof_ts=asof_ts)
        if c is None or not np.isfinite(c):
            continue
        vals.append((str(sym).upper(), float(c)))
    vals.sort(key=lambda x: abs(x[1]), reverse=True)
    out: dict[str, tuple[float, int]] = {}
    for i, (sym, c) in enumerate(vals, start=1):
        out[sym] = (c, i)
    return out


class SessionFlowGate:
    def __init__(self, cfg: SessionFlowGateConfig):
        self.cfg = cfg
        self._day: SessionFlowDayDecision | None = None
        self._stock_by: dict[str, pd.DataFrame] | None = None
        self._symbols: list[str] = []

    @classmethod
    def from_profile(cls, profile: dict[str, Any] | None) -> "SessionFlowGate":
        return cls(SessionFlowGateConfig.from_profile(profile))

    def begin_day(
        self,
        date: str,
        *,
        stock_by: dict[str, pd.DataFrame],
        qqq_df: pd.DataFrame | None,
        vixy_df: pd.DataFrame | None,
        symbols: list[str],
    ) -> SessionFlowDayDecision:
        self._stock_by = stock_by
        self._symbols = list(self.cfg.symbols) if self.cfg.symbols else list(symbols)
        if not self.cfg.enabled:
            dec = SessionFlowDayDecision(
                enabled=False, date=str(date), asof=self.cfg.asof, state="off", reason="disabled"
            )
            self._day = dec
            return dec
        feats = compute_index_chop_features(
            date=str(date),
            stock_by=stock_by,
            qqq_df=qqq_df,
            vixy_df=vixy_df,
            asof=self.cfg.asof,
        )
        hit, reason = is_index_chop(self.cfg, feats)
        if hit:
            state, rsn = "chop", reason
        elif reason.startswith("missing"):
            state, rsn = "unknown", reason
        else:
            state, rsn = "trend", reason
        dec = SessionFlowDayDecision(
            enabled=True,
            date=str(date),
            asof=self.cfg.asof,
            state=state,
            reason=rsn,
            features=feats,
        )
        self._day = dec
        return dec

    def decide_entry(
        self,
        *,
        symbol: str,
        direction: str,
        asof_ts: pd.Timestamp,
    ) -> SessionFlowEntryDecision:
        day = self._day
        if day is None or not day.enabled or not self.cfg.enabled:
            return SessionFlowEntryDecision(True, 1.0, "off", "disabled")

        when = self.cfg.when
        if when == "chop_block" and day.state == "chop":
            return SessionFlowEntryDecision(False, 0.0, "chop", f"chop_block:{day.reason}")

        enforce_leader = when == "always" or (when == "chop_only" and day.state == "chop")
        if not enforce_leader:
            return SessionFlowEntryDecision(True, 1.0, day.state, f"pass_{day.reason}")

        if self._stock_by is None:
            return SessionFlowEntryDecision(True, 1.0, day.state, "fail_open_no_book")

        ranks = cumflow_ranks(
            date=day.date,
            asof_ts=asof_ts,
            stock_by=self._stock_by,
            symbols=self._symbols,
        )
        sym = str(symbol).upper()
        if sym not in ranks:
            if self.cfg.mode == "boost":
                return SessionFlowEntryDecision(
                    True, float(self.cfg.non_leader_scale), day.state, "boost_miss_cum"
                )
            if self.cfg.mode == "scale":
                return SessionFlowEntryDecision(
                    True, float(self.cfg.scale), day.state, "scale_missing_cum"
                )
            return SessionFlowEntryDecision(False, 0.0, day.state, "block_missing_cum")

        cum, rank = ranks[sym]
        if float(self.cfg.min_abs_cum) > 0 and abs(cum) < float(self.cfg.min_abs_cum):
            if self.cfg.mode == "boost":
                return SessionFlowEntryDecision(
                    True,
                    float(self.cfg.non_leader_scale),
                    day.state,
                    "boost_cum_small",
                    cum=cum,
                    rank=rank,
                )
            if self.cfg.mode == "scale":
                return SessionFlowEntryDecision(
                    True, float(self.cfg.scale), day.state, "scale_cum_small", cum=cum, rank=rank
                )
            return SessionFlowEntryDecision(
                False, 0.0, day.state, "block_cum_small", cum=cum, rank=rank
            )

        d = str(direction or "").upper()
        if self.cfg.require_sign_align:
            if d == "UP" and cum <= 0:
                ok_sign = False
            elif d == "DN" and cum >= 0:
                ok_sign = False
            else:
                ok_sign = True
        else:
            ok_sign = True

        ok_rank = int(rank) <= int(self.cfg.top_k)
        if ok_rank and ok_sign:
            if self.cfg.mode == "boost":
                return SessionFlowEntryDecision(
                    True,
                    float(self.cfg.boost),
                    day.state,
                    f"boost_k{self.cfg.top_k}",
                    cum=cum,
                    rank=rank,
                )
            return SessionFlowEntryDecision(
                True, 1.0, day.state, f"leader_k{self.cfg.top_k}", cum=cum, rank=rank
            )

        why = []
        if not ok_rank:
            why.append(f"rank{rank}>k{self.cfg.top_k}")
        if not ok_sign:
            why.append("sign_mismatch")
        reason = "&".join(why)
        if self.cfg.mode == "boost":
            return SessionFlowEntryDecision(
                True,
                float(self.cfg.non_leader_scale),
                day.state,
                f"noleader_{reason}",
                cum=cum,
                rank=rank,
            )
        if self.cfg.mode == "scale":
            return SessionFlowEntryDecision(
                True, float(self.cfg.scale), day.state, f"scale_{reason}", cum=cum, rank=rank
            )
        return SessionFlowEntryDecision(
            False, 0.0, day.state, f"block_{reason}", cum=cum, rank=rank
        )


def load_session_flow_gate(profile: dict[str, Any] | None) -> SessionFlowGate:
    return SessionFlowGate.from_profile(profile)

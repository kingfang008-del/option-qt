"""AM v2 executable satellite — config + online launch detector (Step4 shadow).

Frozen research recipe (Step2b/3 + blind-lift under quote FillSpec):
  launch k=3 |ret|≥0.25% cd=300 · 10:00–11:30 · TP15/SL25/h900
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from maga7.common.replay import to_ny

NY = "America/New_York"

DEFAULT_AM_V2: dict[str, Any] = {
    "enabled": False,
    "execute_mode": "shadow",
    "wired": False,
    "event_calendar_block": False,
    "window_start": "10:00",
    "window_end": "11:30",
    "flatten_before": "11:45",
    "slope_sec": 3,
    "abs_ret_min": 0.0025,
    "cooldown_sec": 300,
    "peak_lookback_sec": 60,
    "dirs": ["UP", "DN"],
    "tp": 0.15,
    "sl": 0.25,
    "max_hold_sec": 900,
    "max_lag_sec": 5.0,
    "max_spread_pct": 0.15,
    "min_mid": 0.05,
    "entry_frac": 0.75,
    "exit_frac": 0.75,
    "position_frac": 0.10,
    "moneyness": "ATM",
    "prefer_dte": 0,
    "allowed_dte": [0, 1, 2],
}


def load_am_v2_cfg(profile: dict[str, Any] | None) -> dict[str, Any]:
    raw = dict(DEFAULT_AM_V2)
    block = (profile or {}).get("am_v2")
    if isinstance(block, dict):
        raw.update(block)
    dirs = raw.get("dirs") or ["UP", "DN"]
    raw["dirs"] = [str(d).upper() for d in dirs]
    return raw


def am_v2_enabled(profile: dict[str, Any] | None) -> bool:
    cfg = load_am_v2_cfg(profile)
    return bool(cfg.get("enabled"))


def _hhmm_minutes(hhmm: str) -> int:
    a, b = str(hhmm).split(":")
    return int(a) * 60 + int(b)


def in_am_v2_window(ts: pd.Timestamp, cfg: dict[str, Any]) -> bool:
    t = to_ny(ts)
    hm = t.hour * 60 + t.minute
    return _hhmm_minutes(cfg["window_start"]) <= hm < _hhmm_minutes(cfg["window_end"])


@dataclass
class AmV2Alert:
    date: str
    symbol: str
    dir: str
    ts: pd.Timestamp
    px: float
    ret_k: float


@dataclass
class AmV2LaunchTracker:
    """Causal rising-edge launch detector matching ``launch_edges`` offline."""

    slope_sec: int = 3
    abs_ret_min: float = 0.002
    cooldown_sec: float = 300.0
    peak_lookback_sec: int = 60
    dirs: tuple[str, ...] = ("UP", "DN")
    closes: deque[float] = field(default_factory=deque)
    rets: deque[float] = field(default_factory=deque)
    last_fire: pd.Timestamp | None = None
    prev_hit_up: bool = False
    prev_hit_dn: bool = False

    def __post_init__(self) -> None:
        # Need slope_sec lag + peak window of ret_k.
        maxlen = max(int(self.slope_sec), 1) + max(int(self.peak_lookback_sec), 1) + 2
        self.closes = deque(self.closes, maxlen=maxlen)
        self.rets = deque(self.rets, maxlen=maxlen)

    def reset(self) -> None:
        self.closes.clear()
        self.rets.clear()
        self.last_fire = None
        self.prev_hit_up = False
        self.prev_hit_dn = False

    def on_close(self, ts: pd.Timestamp, close: float) -> AmV2Alert | None:
        ts = to_ny(ts)
        c = float(close)
        if not (c > 0):
            return None
        self.closes.append(c)
        k = max(1, int(self.slope_sec))
        if len(self.closes) <= k:
            self.rets.append(float("nan"))
            self.prev_hit_up = False
            self.prev_hit_dn = False
            return None
        ret = self.closes[-1] / self.closes[-1 - k] - 1.0
        self.rets.append(float(ret))
        thr = abs(float(self.abs_ret_min))
        win = max(1, int(self.peak_lookback_sec))
        recent = [r for r in list(self.rets)[-win:] if r == r]  # finite
        rmax = max(recent) if recent else float("nan")
        rmin = min(recent) if recent else float("nan")

        hit_up = ret >= thr and ret == rmax
        hit_dn = ret <= -thr and ret == rmin
        edge_up = hit_up and not self.prev_hit_up
        edge_dn = hit_dn and not self.prev_hit_dn
        self.prev_hit_up = hit_up
        self.prev_hit_dn = hit_dn

        direction: str | None = None
        if edge_up and "UP" in self.dirs:
            direction = "UP"
        elif edge_dn and "DN" in self.dirs:
            direction = "DN"
        if direction is None:
            return None
        if self.last_fire is not None:
            if (ts - self.last_fire).total_seconds() < float(self.cooldown_sec):
                return None
        self.last_fire = ts
        return AmV2Alert(
            date=ts.strftime("%Y-%m-%d"),
            symbol="",  # filled by scanner
            dir=direction,
            ts=ts,
            px=c,
            ret_k=float(ret),
        )

    def snapshot_state(self) -> dict[str, Any]:
        return {
            "closes": list(self.closes),
            "rets": list(self.rets),
            "last_fire": self.last_fire.isoformat() if self.last_fire is not None else None,
            "prev_hit_up": bool(self.prev_hit_up),
            "prev_hit_dn": bool(self.prev_hit_dn),
            "slope_sec": int(self.slope_sec),
            "abs_ret_min": float(self.abs_ret_min),
            "cooldown_sec": float(self.cooldown_sec),
            "peak_lookback_sec": int(self.peak_lookback_sec),
            "dirs": list(self.dirs),
        }

    def restore_state(self, state: dict[str, Any] | None) -> None:
        if not isinstance(state, dict):
            return
        self.reset()
        for c in state.get("closes") or []:
            try:
                self.closes.append(float(c))
            except Exception:
                pass
        for r in state.get("rets") or []:
            try:
                self.rets.append(float(r))
            except Exception:
                pass
        lf = state.get("last_fire")
        self.last_fire = to_ny(pd.Timestamp(lf)) if lf else None
        self.prev_hit_up = bool(state.get("prev_hit_up", False))
        self.prev_hit_dn = bool(state.get("prev_hit_dn", False))


def tracker_from_cfg(cfg: dict[str, Any]) -> AmV2LaunchTracker:
    return AmV2LaunchTracker(
        slope_sec=int(cfg.get("slope_sec", 3) or 3),
        abs_ret_min=float(cfg.get("abs_ret_min", 0.002) or 0.002),
        cooldown_sec=float(cfg.get("cooldown_sec", 300) or 300),
        peak_lookback_sec=int(cfg.get("peak_lookback_sec", 60) or 60),
        dirs=tuple(cfg.get("dirs") or ("UP", "DN")),
    )

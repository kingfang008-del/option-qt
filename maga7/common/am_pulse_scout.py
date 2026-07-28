"""AM pulse scout + tradable sleeve helpers (A 09:30–10:30; B 10:30–11:30).

Alert path: emits ``AM_SCOUT_ALERT`` (ops / dash).
Sleeve path: Mag7Scanner A/B drains → satellite OMS (shadow by default).

Arms (each arm ≤ max_alerts_per_symbol per symbol×day; FO preferred on same bar):
  FO   — |fav_from_open| ≥ min_fav_from_open
  LB   — |ret over lookback_bars| ≥ min_lookback_ret (1m bars)

Live/stream: feed completed 1m bars via ``AmPulseScout.on_bar``.
Offline: ``scan_day`` / ``tools/scan_am_pulse_scout``.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd

NY = "America/New_York"
EVENT = "AM_SCOUT_ALERT"
EVENT_SOURCE = "am_pulse_sleeve"

# Live champion (quote dual PASS at lag5 / sp≤15%).
DEFAULT_LIVE: dict[str, Any] = {
    "enabled": False,
    "execute_mode": "shadow",  # shadow | live | off
    "arm": "FO",
    "dirs": ["DN", "UP"],
    "window_start": "09:30",
    "window_end": "10:30",
    "flatten_before": "10:45",
    "min_fav_from_open": 0.008,
    "lookback_bars": 2,
    "min_lookback_ret": 0.99,  # FO-only live default
    "tp": 0.15,
    "sl": 0.20,
    "max_hold_sec": 900,
    "max_spread_pct": 0.15,
    "max_lag_sec": 5.0,
    "min_mid": 0.05,
    "entry_frac": 0.75,
    "exit_frac": 0.75,
    "position_frac": 0.10,
    "moneyness": "ATM",
}


@dataclass(frozen=True)
class AmPulseScoutConfig:
    enabled: bool = True
    window_start: str = "09:30"
    window_end: str = "10:30"  # exclusive — CORE ownership from 10:30
    min_fav_from_open: float = 0.01
    # 0 = off. Blocks FO when |from_open| already exceeds this (chase).
    max_fav_from_open: float = 0.0
    lookback_bars: int = 2  # 1m bars → ~2 minutes
    min_lookback_ret: float = 0.008
    min_chase: float | None = None  # optional FO arm tighten
    dirs: tuple[str, ...] = ("DN", "UP")
    max_alerts_per_symbol: int = 1
    symbols: tuple[str, ...] | None = None
    # If True, only latch day_open from the 09:30 RTH bar (or seed_day_open).
    # Prevents late-start / restart from treating a mid-session open as RTH open.
    rth_open_only: bool = True


@dataclass(frozen=True)
class AmScoutAlert:
    event: str
    date: str
    symbol: str
    dir: str
    ts: str
    arm: str
    fav_from_open: float
    chase: float | None
    dist_ext: float | None
    lookback_ret: float | None
    day_open: float
    px: float
    session_hi: float
    session_lo: float
    source: str = "am_pulse_scout"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def parse_am_pulse_scout(raw: Any) -> AmPulseScoutConfig:
    if not isinstance(raw, dict):
        return AmPulseScoutConfig(enabled=False)
    dirs_raw = raw.get("dirs") or raw.get("directions") or ["DN", "UP"]
    if isinstance(dirs_raw, str):
        dirs = tuple(x.strip().upper() for x in dirs_raw.split(",") if x.strip())
    else:
        dirs = tuple(str(x).strip().upper() for x in dirs_raw if str(x).strip())
    if not dirs:
        dirs = ("DN", "UP")
    syms_raw = raw.get("symbols")
    syms: tuple[str, ...] | None = None
    if isinstance(syms_raw, str):
        syms = tuple(x.strip().upper() for x in syms_raw.split(",") if x.strip())
    elif isinstance(syms_raw, (list, tuple)):
        syms = tuple(str(x).strip().upper() for x in syms_raw if str(x).strip())
    chase = raw.get("min_chase")
    max_fo_raw = raw.get("max_fav_from_open", 0.0)
    try:
        max_fo = float(max_fo_raw) if max_fo_raw is not None else 0.0
    except (TypeError, ValueError):
        max_fo = 0.0
    rth_only = raw.get("rth_open_only", True)
    return AmPulseScoutConfig(
        enabled=bool(raw.get("enabled", True)),
        window_start=str(raw.get("window_start") or "09:30"),
        window_end=str(raw.get("window_end") or "10:30"),
        min_fav_from_open=float(raw.get("min_fav_from_open", 0.01) or 0.01),
        max_fav_from_open=max(0.0, float(max_fo) or 0.0),
        lookback_bars=max(1, int(raw.get("lookback_bars", 2) or 2)),
        min_lookback_ret=float(raw.get("min_lookback_ret", 0.008) or 0.008),
        min_chase=float(chase) if chase is not None else None,
        dirs=dirs,
        max_alerts_per_symbol=max(1, int(raw.get("max_alerts_per_symbol", 1) or 1)),
        symbols=syms,
        rth_open_only=bool(True if rth_only is None else rth_only),
    )


def load_am_pulse_lane_cfg(
    profile: dict[str, Any] | None = None,
    lane: str = "am_pulse",
) -> dict[str, Any]:
    """Merge one AM pulse lane block onto live champion defaults."""
    cfg = dict(DEFAULT_LIVE)
    if not isinstance(profile, dict):
        return cfg
    block = profile.get(str(lane))
    if not isinstance(block, dict):
        return cfg
    cfg.update(block)
    # Normalize enums / lists
    dirs = cfg.get("dirs") or ["DN", "UP"]
    if isinstance(dirs, str):
        cfg["dirs"] = [x.strip().upper() for x in dirs.split(",") if x.strip()]
    else:
        cfg["dirs"] = [str(x).strip().upper() for x in dirs if str(x).strip()]
    arm = str(cfg.get("arm") or "FO").upper()
    cfg["arm"] = arm if arm in {"FO", "LB", "BOTH"} else "FO"
    mode = str(cfg.get("execute_mode") or "shadow").strip().lower()
    if mode in {"true", "1", "yes", "execute"}:
        mode = "live"
    if mode in {"false", "0", "no", "audit"}:
        mode = "off"
    if mode not in {"shadow", "live", "off"}:
        mode = "shadow"
    cfg["execute_mode"] = mode
    return cfg


def load_am_pulse_cfg(profile: dict[str, Any] | None = None) -> dict[str, Any]:
    """Backward-compatible loader for the original ``am_pulse`` lane."""
    return load_am_pulse_lane_cfg(profile, "am_pulse")


def am_pulse_lane_enabled(
    profile: dict[str, Any] | None,
    lane: str = "am_pulse",
) -> bool:
    if not isinstance(profile, dict):
        return False
    block = profile.get(str(lane))
    if not isinstance(block, dict):
        return False
    return bool(block.get("enabled", False))


def am_pulse_enabled(profile: dict[str, Any] | None) -> bool:
    """Backward-compatible enabled check for the original lane."""
    return am_pulse_lane_enabled(profile, "am_pulse")


def scout_config_from_live(cfg: dict[str, Any]) -> AmPulseScoutConfig:
    """Build detector config from live champion (FO-only unless arm allows LB)."""
    arm = str(cfg.get("arm") or "FO").upper()
    fo = float(cfg.get("min_fav_from_open", 0.008) or 0.008)
    lb = float(cfg.get("min_lookback_ret", 0.008) or 0.008)
    if arm == "FO":
        lb = 0.99
    elif arm == "LB":
        fo = 0.99
    dirs = cfg.get("dirs") or ["DN"]
    if isinstance(dirs, str):
        dirs_t = tuple(x.strip().upper() for x in dirs.split(",") if x.strip())
    else:
        dirs_t = tuple(str(x).strip().upper() for x in dirs if str(x).strip())
    max_fo_raw = cfg.get("max_fav_from_open", 0.0)
    try:
        max_fo = float(max_fo_raw) if max_fo_raw is not None else 0.0
    except (TypeError, ValueError):
        max_fo = 0.0
    rth_only = cfg.get("rth_open_only", True)
    return AmPulseScoutConfig(
        enabled=True,
        window_start=str(cfg.get("window_start") or "09:30"),
        window_end=str(cfg.get("window_end") or "10:30"),
        min_fav_from_open=fo,
        max_fav_from_open=max(0.0, float(max_fo) or 0.0),
        lookback_bars=max(1, int(cfg.get("lookback_bars", 2) or 2)),
        min_lookback_ret=lb,
        dirs=dirs_t or ("DN", "UP"),
        max_alerts_per_symbol=1,
        symbols=None,
        rth_open_only=bool(True if rth_only is None else rth_only),
    )


def _hhmm_to_min(hhmm: str) -> int:
    parts = str(hhmm).strip().split(":")
    return int(parts[0]) * 60 + int(parts[1])


def _in_window(ts: pd.Timestamp, start: str, end: str) -> bool:
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize(NY)
    else:
        t = t.tz_convert(NY)
    hm = t.hour * 60 + t.minute
    return _hhmm_to_min(start) <= hm < _hhmm_to_min(end)


@dataclass
class AmPulseScout:
    """Stateful per-day scout. Call ``begin_day`` then ``on_bar`` for each 1m close."""

    cfg: AmPulseScoutConfig = field(default_factory=AmPulseScoutConfig)
    _date: str | None = None
    _alerts_n: dict[tuple[str, str], int] = field(default_factory=dict)
    _closes: dict[str, list[float]] = field(default_factory=dict)
    _day_open: dict[str, float] = field(default_factory=dict)
    _hi: dict[str, float] = field(default_factory=dict)
    _lo: dict[str, float] = field(default_factory=dict)

    def begin_day(self, date: str) -> None:
        self._date = str(date)
        self._alerts_n.clear()
        self._closes.clear()
        self._day_open.clear()
        self._hi.clear()
        self._lo.clear()

    def seed_day_open(self, symbol: str, day_open: float) -> None:
        """Latch RTH open from an external source (scanner state / official open).

        Never overwrites an already-latched open for the day.
        """
        sym = str(symbol).upper()
        px = float(day_open)
        if sym and px > 0 and sym not in self._day_open:
            self._day_open[sym] = px

    def _arm_budget(self, sym: str, arm: str) -> bool:
        return int(self._alerts_n.get((sym, arm), 0)) < int(self.cfg.max_alerts_per_symbol)

    @staticmethod
    def _is_rth_open_bar(ts: pd.Timestamp) -> bool:
        t = pd.Timestamp(ts)
        if t.tzinfo is None:
            t = t.tz_localize(NY)
        else:
            t = t.tz_convert(NY)
        return int(t.hour) == 9 and int(t.minute) == 30

    def snapshot_state(self) -> dict[str, Any]:
        """Return JSON-safe detector state for restart continuity."""
        return {
            "date": self._date,
            "alerts_n": [
                {"symbol": sym, "arm": arm, "n": int(n)}
                for (sym, arm), n in sorted(self._alerts_n.items())
            ],
            "closes": {sym: list(values) for sym, values in self._closes.items()},
            "day_open": dict(self._day_open),
            "hi": dict(self._hi),
            "lo": dict(self._lo),
        }

    def restore_state(self, payload: dict[str, Any] | None) -> None:
        """Restore a prior ``snapshot_state`` payload."""
        raw = payload if isinstance(payload, dict) else {}
        self._date = str(raw.get("date")) if raw.get("date") else None
        self._alerts_n = {
            (str(row.get("symbol") or "").upper(), str(row.get("arm") or "").upper()): int(
                row.get("n") or 0
            )
            for row in (raw.get("alerts_n") or [])
            if row.get("symbol") and row.get("arm")
        }
        self._closes = {
            str(sym).upper(): [float(value) for value in (values or [])]
            for sym, values in (raw.get("closes") or {}).items()
        }
        self._day_open = {
            str(sym).upper(): float(value)
            for sym, value in (raw.get("day_open") or {}).items()
        }
        self._hi = {
            str(sym).upper(): float(value)
            for sym, value in (raw.get("hi") or {}).items()
        }
        self._lo = {
            str(sym).upper(): float(value)
            for sym, value in (raw.get("lo") or {}).items()
        }

    def on_bar(
        self,
        *,
        symbol: str,
        ts: pd.Timestamp,
        open_: float,
        high: float,
        low: float,
        close: float,
    ) -> AmScoutAlert | None:
        if not self.cfg.enabled:
            return None
        sym = str(symbol).upper()
        if self.cfg.symbols is not None and sym not in set(self.cfg.symbols):
            return None
        if self._date is None:
            return None
        if not (open_ > 0 and close > 0 and high > 0 and low > 0):
            return None

        # Latch RTH open only once: prefer 09:30 bar, or first bar if rth_open_only=False.
        if sym not in self._day_open:
            allow_latch = (not bool(self.cfg.rth_open_only)) or self._is_rth_open_bar(ts)
            if allow_latch:
                self._day_open[sym] = float(open_)
                self._hi[sym] = float(high)
                self._lo[sym] = float(low)
                self._closes[sym] = []
        if sym not in self._day_open:
            # No official open yet (late start without seed) — do not invent one.
            return None
        if sym not in self._hi:
            self._hi[sym] = float(high)
            self._lo[sym] = float(low)
            self._closes[sym] = []
        self._hi[sym] = max(float(self._hi[sym]), float(high))
        self._lo[sym] = min(float(self._lo[sym]), float(low))
        self._closes[sym].append(float(close))

        day_open = float(self._day_open[sym])
        px = float(close)
        hi = float(self._hi[sym])
        lo = float(self._lo[sym])
        from_open = px / day_open - 1.0
        closes = self._closes[sym]
        lb_ret = None
        if len(closes) > int(self.cfg.lookback_bars):
            p0 = closes[-(int(self.cfg.lookback_bars) + 1)]
            if p0 > 0:
                lb_ret = px / p0 - 1.0
        # Keep RTH-open/session state before a delayed lane starts, but never
        # consume its independent trigger budget outside that lane's window.
        if not _in_window(ts, self.cfg.window_start, self.cfg.window_end):
            return None

        def _chase_dist(direction: str) -> tuple[float, float]:
            if hi > lo:
                rng = (px - lo) / (hi - lo)
            else:
                rng = 0.5
            if direction == "UP":
                return float(rng), float((hi - px) / day_open)
            return float(1.0 - rng), float((px - lo) / day_open)

        max_fo = float(self.cfg.max_fav_from_open or 0.0)
        fo_ok = abs(from_open) + 1e-12 >= float(self.cfg.min_fav_from_open)
        if max_fo > 0:
            fo_ok = fo_ok and abs(from_open) - 1e-12 <= max_fo

        # Prefer FO arm (opening extension), else lookback impulse.
        alert: AmScoutAlert | None = None
        if self._arm_budget(sym, "FO") and fo_ok:
            d = "UP" if from_open >= 0 else "DN"
            if d in set(self.cfg.dirs):
                chase, dist = _chase_dist(d)
                if self.cfg.min_chase is None or chase + 1e-12 >= float(self.cfg.min_chase):
                    alert = AmScoutAlert(
                        event=EVENT,
                        date=str(self._date),
                        symbol=sym,
                        dir=d,
                        ts=str(pd.Timestamp(ts)),
                        arm="FO",
                        fav_from_open=float(abs(from_open)),
                        chase=float(chase),
                        dist_ext=float(dist),
                        lookback_ret=float(lb_ret) if lb_ret is not None else None,
                        day_open=day_open,
                        px=px,
                        session_hi=hi,
                        session_lo=lo,
                    )
        if alert is None and self._arm_budget(sym, "LB") and lb_ret is not None:
            if abs(lb_ret) + 1e-12 >= float(self.cfg.min_lookback_ret):
                d = "UP" if lb_ret >= 0 else "DN"
                if d in set(self.cfg.dirs):
                    chase, dist = _chase_dist(d)
                    alert = AmScoutAlert(
                        event=EVENT,
                        date=str(self._date),
                        symbol=sym,
                        dir=d,
                        ts=str(pd.Timestamp(ts)),
                        arm="LB",
                        fav_from_open=float(abs(from_open)),
                        chase=float(chase),
                        dist_ext=float(dist),
                        lookback_ret=float(lb_ret),
                        day_open=day_open,
                        px=px,
                        session_hi=hi,
                        session_lo=lo,
                    )
        if alert is None:
            return None
        key = (sym, alert.arm)
        self._alerts_n[key] = int(self._alerts_n.get(key, 0)) + 1
        return alert


def scan_day(
    stock_day: pd.DataFrame,
    *,
    date: str,
    symbol: str,
    cfg: AmPulseScoutConfig | None = None,
) -> list[AmScoutAlert]:
    """Scan one symbol-day of 1m (or denser) OHLCV; returns alerts in time order."""
    cfg = cfg or AmPulseScoutConfig()
    if stock_day is None or stock_day.empty:
        return []
    day = stock_day.copy()
    if "date" in day.columns:
        day = day[day["date"].astype(str) == str(date)]
    if day.empty:
        return []
    day = day.sort_values("timestamp")
    scout = AmPulseScout(cfg=cfg)
    scout.begin_day(str(date))
    out: list[AmScoutAlert] = []
    for r in day.itertuples(index=False):
        a = scout.on_bar(
            symbol=symbol,
            ts=pd.Timestamp(r.timestamp),
            open_=float(r.open),
            high=float(r.high),
            low=float(r.low),
            close=float(r.close),
        )
        if a is not None:
            out.append(a)
    return out

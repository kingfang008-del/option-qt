"""Own-path smooth trend launch detector (research).

Detects the *start* of a smooth grind on a single symbol — not cross-section
rank #1. Tuned against the 2026-07-20 MSFT case (inflection ~10:03, causal
confirm ~10:12).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd

NY = "America/New_York"


@dataclass(frozen=True)
class SmoothLaunchConfig:
    lookback_minutes: int = 10
    trough_lookback_minutes: int = 20
    min_look_ret: float = 0.0015  # +15bp over lookback (signed)
    min_path_eff: float = 0.40
    min_up_frac: float = 0.60
    max_dd: float = -0.004  # shallower than -40bp inside lookback
    min_from_trough: float = 0.002  # +20bp off recent trough/peak
    # session scan window (NY)
    scan_start: str = "09:45"
    scan_end: str = "15:00"
    # cooldown after a fire (minutes) so one trend ≠ many fires
    cooldown_minutes: int = 30
    min_bars: int = 5


@dataclass
class SmoothLaunch:
    date: str
    symbol: str
    direction: str  # UP | DN
    detect_ts: pd.Timestamp
    price: float
    look_ret: float
    path_eff: float
    up_frac: float
    max_dd: float
    from_extreme: float
    score: float

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["detect_ts"] = str(self.detect_ts)
        return d


def _hhmm_to_min(hhmm: str) -> int:
    h, m = str(hhmm).split(":")
    return int(h) * 60 + int(m)


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


def _window_stats(closes: np.ndarray, *, direction: str) -> dict[str, float] | None:
    if closes is None or len(closes) < 3:
        return None
    c = np.asarray(closes, dtype=float)
    if direction == "UP":
        signed = np.diff(c) / c[:-1]
        net = c[-1] / c[0] - 1.0
        peak = np.maximum.accumulate(c)
        dd = float(np.min(c / peak - 1.0))
        from_extreme = c[-1] / float(np.min(c)) - 1.0
        up_frac = float((signed > 0).mean())
    else:
        signed = -(np.diff(c) / c[:-1])
        net = 1.0 - c[-1] / c[0]
        trough = np.minimum.accumulate(c)
        # adverse = bounce against the short
        dd = float(np.min(1.0 - c / trough))  # negative-ish if rises from trough
        # for DN, "dd" = max adverse rebound from running low of short thesis
        run_low = np.minimum.accumulate(c)
        adverse = c / run_low - 1.0
        dd = -float(np.max(adverse))  # more negative = worse
        from_extreme = 1.0 - c[-1] / float(np.max(c))
        up_frac = float((signed > 0).mean())  # fraction of bars moving DN
    sumabs = float(np.abs(np.diff(c) / c[:-1]).sum()) or 1e-12
    path_eff = float(abs(net) / sumabs)
    return {
        "look_ret": float(net),
        "path_eff": path_eff,
        "up_frac": up_frac,
        "max_dd": float(dd),
        "from_extreme": float(from_extreme),
    }


def launch_score(st: dict[str, float]) -> float:
    return (
        100.0 * st["look_ret"]
        + 2.0 * st["path_eff"]
        + 1.0 * st["up_frac"]
        - 50.0 * abs(min(0.0, st["max_dd"]))
    )


def detect_smooth_launches_day(
    stock_day: pd.DataFrame,
    *,
    symbol: str,
    date: str,
    cfg: SmoothLaunchConfig | None = None,
    directions: Iterable[str] = ("UP", "DN"),
) -> list[SmoothLaunch]:
    """Causal minute scan: emit when own-path smooth launch conditions first hold."""
    cfg = cfg or SmoothLaunchConfig()
    day = _prepare_day(stock_day, date)
    if day.empty or "close" not in day.columns:
        return []
    closes = day["close"].astype(float).to_numpy()
    ts = day["timestamp"]
    start_m = _hhmm_to_min(cfg.scan_start)
    end_m = _hhmm_to_min(cfg.scan_end)
    lb = int(cfg.lookback_minutes)
    tb = int(cfg.trough_lookback_minutes)
    out: list[SmoothLaunch] = []
    last_fire: dict[str, pd.Timestamp] = {}

    for i in range(len(day)):
        t = ts.iloc[i]
        hm = int(t.hour) * 60 + int(t.minute)
        if hm < start_m or hm > end_m:
            continue
        if i < max(lb, cfg.min_bars):
            continue
        i0 = max(0, i - lb)
        i_ext0 = max(0, i - tb)
        win = closes[i0 : i + 1]
        ext = closes[i_ext0 : i + 1]
        for direction in directions:
            d = str(direction).upper()
            if d not in {"UP", "DN"}:
                continue
            prev = last_fire.get(d)
            if prev is not None and t < prev + pd.Timedelta(minutes=cfg.cooldown_minutes):
                continue
            st = _window_stats(win, direction=d)
            if st is None:
                continue
            # from trough/peak over longer extreme window
            if d == "UP":
                from_ext = float(closes[i] / float(np.min(ext)) - 1.0)
            else:
                from_ext = float(1.0 - closes[i] / float(np.max(ext)))
            st["from_extreme"] = from_ext
            if st["look_ret"] < cfg.min_look_ret:
                continue
            if st["path_eff"] < cfg.min_path_eff:
                continue
            if st["up_frac"] < cfg.min_up_frac:
                continue
            if st["max_dd"] < cfg.max_dd:  # more negative than allowed
                continue
            if from_ext < cfg.min_from_trough:
                continue
            sc = launch_score(st)
            out.append(
                SmoothLaunch(
                    date=str(date),
                    symbol=str(symbol).upper(),
                    direction=d,
                    detect_ts=pd.Timestamp(t),
                    price=float(closes[i]),
                    look_ret=st["look_ret"],
                    path_eff=st["path_eff"],
                    up_frac=st["up_frac"],
                    max_dd=st["max_dd"],
                    from_extreme=from_ext,
                    score=float(sc),
                )
            )
            last_fire[d] = pd.Timestamp(t)
    return out


@dataclass
class DayMove:
    """Oracle significant session move for hit evaluation."""

    date: str
    symbol: str
    direction: str
    start_ts: pd.Timestamp
    end_ts: pd.Timestamp
    start_px: float
    end_px: float
    move_ret: float
    max_adverse: float


def extract_significant_moves(
    stock_day: pd.DataFrame,
    *,
    symbol: str,
    date: str,
    min_move: float = 0.015,
    min_leg_minutes: int = 20,
) -> list[DayMove]:
    """Label major UP/DN legs: open→session peak / trough with enough travel.

    UP: from session open (or early trough before peak) to peak if peak-open >= min.
    DN: from session open (or early peak before trough) to trough if open-trough >= min.
    Also adds a secondary leg if afternoon extremes extend further.
    """
    day = _prepare_day(stock_day, date)
    if len(day) < min_leg_minutes + 5:
        return []
    ts = day["timestamp"]
    c = day["close"].astype(float).to_numpy()
    moves: list[DayMove] = []

    # Primary UP: earliest bar at running min before the session max
    i_peak = int(np.argmax(c))
    peak_ret = c[i_peak] / c[0] - 1.0
    if peak_ret >= min_move and i_peak >= min_leg_minutes:
        i_start = int(np.argmin(c[: i_peak + 1]))
        # adverse after start toward peak
        leg = c[i_start : i_peak + 1]
        peak_run = np.maximum.accumulate(leg)
        mae = float(np.min(leg / peak_run - 1.0))
        moves.append(
            DayMove(
                date=str(date),
                symbol=str(symbol).upper(),
                direction="UP",
                start_ts=pd.Timestamp(ts.iloc[i_start]),
                end_ts=pd.Timestamp(ts.iloc[i_peak]),
                start_px=float(c[i_start]),
                end_px=float(c[i_peak]),
                move_ret=float(c[i_peak] / c[i_start] - 1.0),
                max_adverse=mae,
            )
        )

    # Primary DN
    i_trough = int(np.argmin(c))
    trough_ret = 1.0 - c[i_trough] / c[0]
    if trough_ret >= min_move and i_trough >= min_leg_minutes:
        i_start = int(np.argmax(c[: i_trough + 1]))
        leg = c[i_start : i_trough + 1]
        run_low = np.minimum.accumulate(leg)
        mae = -float(np.max(leg / run_low - 1.0))
        moves.append(
            DayMove(
                date=str(date),
                symbol=str(symbol).upper(),
                direction="DN",
                start_ts=pd.Timestamp(ts.iloc[i_start]),
                end_ts=pd.Timestamp(ts.iloc[i_trough]),
                start_px=float(c[i_start]),
                end_px=float(c[i_trough]),
                move_ret=float(1.0 - c[i_trough] / c[i_start]),
                max_adverse=mae,
            )
        )
    return moves


def match_launches_to_moves(
    launches: list[SmoothLaunch],
    moves: list[DayMove],
    *,
    max_late_minutes: int = 45,
    max_early_minutes: int = 15,
    min_capture_frac: float = 0.40,
) -> list[dict[str, Any]]:
    """Hit if a same-direction launch fires near move start and before most of the move."""
    rows: list[dict[str, Any]] = []
    used: set[int] = set()
    for mi, mv in enumerate(moves):
        best = None
        best_i = None
        for li, ln in enumerate(launches):
            if li in used:
                continue
            if ln.direction != mv.direction:
                continue
            if ln.symbol != mv.symbol or ln.date != mv.date:
                continue
            dt_min = (ln.detect_ts - mv.start_ts).total_seconds() / 60.0
            if dt_min < -max_early_minutes or dt_min > max_late_minutes:
                continue
            # capture: remaining move from detect price to end
            if mv.direction == "UP":
                if ln.price <= 0:
                    continue
                capt = mv.end_px / ln.price - 1.0
            else:
                capt = 1.0 - mv.end_px / ln.price
            frac = capt / mv.move_ret if mv.move_ret > 1e-9 else 0.0
            if frac < min_capture_frac:
                continue
            cand = {
                "hit": True,
                "move": mv,
                "launch": ln,
                "delay_min": dt_min,
                "capture_ret": capt,
                "capture_frac": frac,
            }
            if best is None or abs(dt_min) < abs(best["delay_min"]):
                best = cand
                best_i = li
        if best is not None and best_i is not None:
            used.add(best_i)
            rows.append(best)
        else:
            rows.append(
                {
                    "hit": False,
                    "move": mv,
                    "launch": None,
                    "delay_min": None,
                    "capture_ret": None,
                    "capture_frac": None,
                }
            )
    return rows


@dataclass(frozen=True)
class ImpulseLaunchConfig:
    lookback_minutes: int = 5
    min_look_ret: float = 0.004  # +40bp vertical
    scan_start: str = "09:45"
    scan_end: str = "11:30"
    cooldown_minutes: int = 30


def detect_impulse_launches_day(
    stock_day: pd.DataFrame,
    *,
    symbol: str,
    date: str,
    cfg: ImpulseLaunchConfig | None = None,
    directions: Iterable[str] = ("UP", "DN"),
) -> list[SmoothLaunch]:
    """Parallel sleeve: short-window vertical displacement (gap/impulse)."""
    cfg = cfg or ImpulseLaunchConfig()
    day = _prepare_day(stock_day, date)
    if day.empty:
        return []
    closes = day["close"].astype(float).to_numpy()
    ts = day["timestamp"]
    start_m = _hhmm_to_min(cfg.scan_start)
    end_m = _hhmm_to_min(cfg.scan_end)
    lb = int(cfg.lookback_minutes)
    out: list[SmoothLaunch] = []
    last_fire: dict[str, pd.Timestamp] = {}
    for i in range(lb, len(day)):
        t = ts.iloc[i]
        hm = int(t.hour) * 60 + int(t.minute)
        if hm < start_m or hm > end_m:
            continue
        win = closes[i - lb : i + 1]
        for direction in directions:
            d = str(direction).upper()
            prev = last_fire.get(d)
            if prev is not None and t < prev + pd.Timedelta(minutes=cfg.cooldown_minutes):
                continue
            st = _window_stats(win, direction=d)
            if st is None or st["look_ret"] < cfg.min_look_ret:
                continue
            # impulse allows lower path_eff; still require directional majority
            if st["up_frac"] < 0.5:
                continue
            out.append(
                SmoothLaunch(
                    date=str(date),
                    symbol=str(symbol).upper(),
                    direction=d,
                    detect_ts=pd.Timestamp(t),
                    price=float(closes[i]),
                    look_ret=st["look_ret"],
                    path_eff=st["path_eff"],
                    up_frac=st["up_frac"],
                    max_dd=st["max_dd"],
                    from_extreme=st["from_extreme"],
                    score=launch_score(st) + 1.0,  # slight impulse tag boost
                )
            )
            last_fire[d] = pd.Timestamp(t)
    return out


@dataclass(frozen=True)
class SmoothStockTradeConfig:
    """Stock execution for smooth/impulse sleeves (research)."""

    cost_bps: float = 1.0
    # exits
    max_hold_minutes: int = 120
    # smooth-break: last N bars adverse or path_eff collapse
    break_lookback: int = 10
    break_min_up_frac: float = 0.40  # exit if favoring bars drop below this
    break_max_adverse: float = 0.004  # exit if adverse move ≥ 40bp from entry peak/trough
    eod_hhmm: str = "15:55"
    # portfolio
    max_positions: int = 2
    first_per_symbol_dir: bool = True
    prefer_smooth_over_impulse: bool = True


def _simulate_stock_path(
    day: pd.DataFrame,
    *,
    entry_ts: pd.Timestamp,
    direction: str,
    cfg: SmoothStockTradeConfig,
    date: str | None = None,
) -> dict[str, Any] | None:
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
    closes = after["close"].astype(float).to_numpy()
    times = after["timestamp"]
    eod_m = _hhmm_to_min(cfg.eod_hhmm)
    peak = entry_px
    trough = entry_px
    exit_i = len(closes) - 1
    reason = "eod"
    for i in range(1, len(closes)):
        t = times.iloc[i]
        px = float(closes[i])
        hm = int(t.hour) * 60 + int(t.minute)
        held_min = (t - entry_ts).total_seconds() / 60.0
        if direction == "UP":
            peak = max(peak, px)
            adverse = peak / px - 1.0 if px > 0 else 0.0  # giveback from peak
            signed_from_entry = px / entry_px - 1.0
        else:
            trough = min(trough, px)
            adverse = px / trough - 1.0 if trough > 0 else 0.0
            signed_from_entry = 1.0 - px / entry_px

        if held_min >= cfg.max_hold_minutes:
            exit_i, reason = i, "TIME"
            break
        if hm >= eod_m:
            exit_i, reason = i, "EOD"
            break
        # trailing adverse from favorable extreme
        if adverse >= cfg.break_max_adverse and held_min >= cfg.break_lookback:
            exit_i, reason = i, "TRAIL_BREAK"
            break
        # micro-structure break: last lookback bars lose direction
        if i >= cfg.break_lookback:
            w = closes[i - cfg.break_lookback : i + 1]
            st = _window_stats(w, direction=direction)
            if st is not None and st["up_frac"] < cfg.break_min_up_frac and signed_from_entry < 0.001:
                exit_i, reason = i, "SMOOTH_BREAK"
                break

    exit_px = float(closes[exit_i])
    exit_ts = pd.Timestamp(times.iloc[exit_i])
    raw = exit_px / entry_px - 1.0
    signed = raw if direction == "UP" else -raw
    cost = 2.0 * (float(cfg.cost_bps) / 1e4)
    return {
        "entry_ts": entry_ts,
        "exit_ts": exit_ts,
        "entry_px": entry_px,
        "exit_px": exit_px,
        "raw_stock_ret": float(raw),
        "ret": float(signed - cost),
        "exit_reason": reason,
        "hold_minutes": float((exit_ts - entry_ts).total_seconds() / 60.0),
    }


def merge_dual_sleeve_launches(
    smooth: list[SmoothLaunch],
    impulse: list[SmoothLaunch],
    *,
    first_per_symbol_dir: bool = True,
    prefer_smooth: bool = True,
) -> list[tuple[SmoothLaunch, str]]:
    """Merge sleeves; tag source. Optionally keep first per symbol/dir."""
    tagged: list[tuple[SmoothLaunch, str]] = []
    if prefer_smooth:
        tagged.extend((ln, "smooth") for ln in smooth)
        tagged.extend((ln, "impulse") for ln in impulse)
    else:
        tagged.extend((ln, "impulse") for ln in impulse)
        tagged.extend((ln, "smooth") for ln in smooth)
    tagged.sort(key=lambda x: (x[0].detect_ts, 0 if x[1] == "smooth" else 1))
    if not first_per_symbol_dir:
        return tagged
    seen: set[tuple[str, str]] = set()
    out: list[tuple[SmoothLaunch, str]] = []
    for ln, src in tagged:
        key = (ln.symbol, ln.direction)
        if key in seen:
            continue
        # if impulse fires first then smooth later same dir — keep first only
        seen.add(key)
        out.append((ln, src))
    return out


def replay_smooth_impulse_stock_day(
    stock_day: pd.DataFrame,
    *,
    symbol: str,
    date: str,
    smooth_cfg: SmoothLaunchConfig | None = None,
    impulse_cfg: ImpulseLaunchConfig | None = None,
    trade_cfg: SmoothStockTradeConfig | None = None,
) -> list[dict[str, Any]]:
    """One symbol-day: detect → merge → simulate stock trades (no portfolio cap)."""
    smooth_cfg = smooth_cfg or SmoothLaunchConfig(scan_end="11:30", min_look_ret=0.002)
    impulse_cfg = impulse_cfg or ImpulseLaunchConfig()
    trade_cfg = trade_cfg or SmoothStockTradeConfig()
    smooth = detect_smooth_launches_day(
        stock_day, symbol=symbol, date=date, cfg=smooth_cfg, directions=("UP", "DN")
    )
    impulse = detect_impulse_launches_day(
        stock_day, symbol=symbol, date=date, cfg=impulse_cfg, directions=("UP", "DN")
    )
    merged = merge_dual_sleeve_launches(
        smooth,
        impulse,
        first_per_symbol_dir=trade_cfg.first_per_symbol_dir,
        prefer_smooth=trade_cfg.prefer_smooth_over_impulse,
    )
    rows: list[dict[str, Any]] = []
    for ln, src in merged:
        sim = _simulate_stock_path(
            stock_day,
            entry_ts=ln.detect_ts,
            direction=ln.direction,
            cfg=trade_cfg,
            date=date,
        )
        if sim is None:
            continue
        rows.append(
            {
                "date": date,
                "symbol": symbol,
                "direction": ln.direction,
                "sleeve": src,
                "detect_ts": str(ln.detect_ts),
                "score": ln.score,
                "look_ret": ln.look_ret,
                "path_eff": ln.path_eff,
                **{k: (str(v) if isinstance(v, pd.Timestamp) else v) for k, v in sim.items()},
            }
        )
    return rows


def apply_day_portfolio_cap(
    trades: list[dict[str, Any]],
    *,
    max_positions: int = 2,
) -> list[dict[str, Any]]:
    """Keep earliest max_positions trades per date (cross-symbol)."""
    if not trades:
        return []
    by_date: dict[str, list[dict[str, Any]]] = {}
    for t in trades:
        by_date.setdefault(str(t["date"]), []).append(t)
    out: list[dict[str, Any]] = []
    for date, rows in sorted(by_date.items()):
        rows = sorted(rows, key=lambda r: (str(r["detect_ts"]), -float(r.get("score") or 0)))
        # one position per symbol
        picked: list[dict[str, Any]] = []
        seen_sym: set[str] = set()
        for r in rows:
            if r["symbol"] in seen_sym:
                continue
            if len(picked) >= max_positions:
                break
            picked.append(r)
            seen_sym.add(r["symbol"])
        out.extend(picked)
    return out

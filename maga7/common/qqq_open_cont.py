"""QQQ 0DTE open_cont expert (09:45 continuation) — executable sleeve.

Champion (quote dual PASS + trades dual PASS incl. Jul):
  clock 09:45 · |from_open|≥0.2% · tp10% / sl25% · FillSpec 0.75
  entry gates: spread≤15% · lag≤2s · mid≥0.05

Satellite to Mag7 research_baseline (does not rewrite Rule-A).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from maga7.common.bar_agg import load_stock_1s_day
from maga7.common.fills import FillSpec
from maga7.common.option_quote_tpsl import entry_quote_row, simulate_quote_tpsl
from maga7.common.option_trade_tpsl import simulate_trade_tpsl
from maga7.common.option_trades import load_option_trades
from maga7.common.replay import to_ny
from maga7.tools.run_morning_sec_qqq_dte1 import _load_atm_path
from maga7.tools.scan_morning_sec_edge import _morning_slice
from maga7.tools.scan_session_horizon_foresight import _paths_by_ticker

NY = "America/New_York"
_OCC = re.compile(
    r"^(?P<root>[A-Z]+)(?P<yy>\d{2})(?P<mm>\d{2})(?P<dd>\d{2})"
    r"(?P<cp>[CP])(?P<strike>\d{8})$"
)

DEFAULT_CHAMPION: dict[str, Any] = {
    "clock": "09:45",
    "from_open_min": 0.002,
    "tp": 0.10,
    "sl": 0.25,
    "max_hold_sec": 900,
    "max_spread_pct": 0.15,
    "max_lag_sec": 2.0,
    "min_mid": 0.05,
    "entry_frac": 0.75,
    "exit_frac": 0.75,
    "position_frac": 0.10,
    "slip": 0.01,
}


@dataclass(frozen=True)
class OpenContSignal:
    date: str
    direction: str
    from_open: float
    entry_ts: pd.Timestamp
    spot: float
    open_px: float


def load_champion(profile: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = dict(DEFAULT_CHAMPION)
    if isinstance(profile, dict):
        block = profile.get("qqq_open_cont")
        if isinstance(block, dict):
            cfg.update({k: block[k] for k in cfg if k in block})
            # pass-through non-champion keys used by live wire
            for k in ("enabled", "quote_1s_root", "book", "note"):
                if k in block:
                    cfg[k] = block[k]
    return cfg


def open_cont_enabled(profile: dict[str, Any] | None) -> bool:
    if not isinstance(profile, dict):
        return False
    block = profile.get("qqq_open_cont")
    if not isinstance(block, dict):
        return False
    return bool(block.get("enabled", False))


def signal_from_open_spot(
    *,
    date: str,
    open_px: float,
    spot: float,
    entry_ts: pd.Timestamp,
    from_open_min: float = 0.002,
) -> OpenContSignal | None:
    """Build signal from RTH open + spot at clock (live / shadow in-memory path)."""
    if not (np.isfinite(open_px) and open_px > 0 and np.isfinite(spot) and spot > 0):
        return None
    from_open = float(spot / float(open_px) - 1.0)
    if abs(from_open) < float(from_open_min):
        return None
    direction = "UP" if from_open > 0 else "DN"
    return OpenContSignal(
        date=str(date),
        direction=direction,
        from_open=from_open,
        entry_ts=to_ny(entry_ts),
        spot=float(spot),
        open_px=float(open_px),
    )


def resolve_atm_ticker(
    quote_root: Path | str | None,
    date: str,
    direction: str,
) -> tuple[str | None, float | None]:
    """QQQ 0DTE ATM ticker from bucketed day file (shadow / dry)."""
    if quote_root is None:
        return None, None
    root = Path(quote_root)
    if not root.is_dir():
        return None, None
    _path, ticker, strike = _load_atm_path(root, date, str(direction).upper())
    return ticker, strike


def signal_at_clock(
    stock_1s_root: Path,
    date: str,
    *,
    clock: str = "09:45",
    from_open_min: float = 0.002,
) -> OpenContSignal | None:
    day = load_stock_1s_day(stock_1s_root, "QQQ", date)
    buf = _morning_slice(day, start="09:30", end="16:00")
    if buf is None or buf.empty:
        return None
    ts = pd.DatetimeIndex(pd.to_datetime(buf["timestamp"]))
    if ts.tz is None:
        ts = ts.tz_localize(NY, ambiguous="infer")
    else:
        ts = ts.tz_convert(NY)
    close = buf["close"].astype(float).to_numpy()
    open_px = float(close[0])
    if not (np.isfinite(open_px) and open_px > 0):
        return None
    t0 = pd.Timestamp(f"{date} {clock}", tz=NY)
    i = int(ts.searchsorted(t0, side="left"))
    if i >= len(close):
        return None
    return signal_from_open_spot(
        date=date,
        open_px=open_px,
        spot=float(close[i]),
        entry_ts=ts[i],
        from_open_min=from_open_min,
    )


def _atm_from_trades(
    trade_paths: dict[str, tuple[np.ndarray, np.ndarray]],
    *,
    date: str,
    direction: str,
    spot: float,
) -> tuple[str | None, float | None]:
    if not trade_paths or not np.isfinite(spot) or spot <= 0:
        return None, None
    ymd = date.replace("-", "")[2:]
    want_cp = "C" if direction == "UP" else "P"
    best_t: str | None = None
    best_k: float | None = None
    best_abs = float("inf")
    for raw in trade_paths:
        key = str(raw).replace("O:", "")
        m = _OCC.match(key)
        if m is None or m.group("root") != "QQQ":
            continue
        exp = f"{m.group('yy')}{m.group('mm')}{m.group('dd')}"
        if exp != ymd or m.group("cp") != want_cp:
            continue
        k = float(m.group("strike")) / 1000.0
        ad = abs(k - spot)
        if ad < best_abs:
            best_abs = ad
            best_k = k
            best_t = str(raw)
    return best_t, best_k


def simulate_day(
    *,
    date: str,
    stock_1s_root: Path,
    quote_root: Path | None,
    trades_root: Path | None,
    champion: dict[str, Any] | None = None,
    book: str = "auto",
) -> dict[str, Any] | None:
    """One causal day. ``book``: auto|quote|trades."""
    cfg = dict(DEFAULT_CHAMPION)
    if champion:
        cfg.update({k: champion[k] for k in cfg if k in champion})
    sig = signal_at_clock(
        stock_1s_root,
        date,
        clock=str(cfg["clock"]),
        from_open_min=float(cfg["from_open_min"]),
    )
    if sig is None:
        return None

    fill = FillSpec(
        entry_frac=float(cfg["entry_frac"]), exit_frac=float(cfg["exit_frac"])
    )
    used = str(book)

    if used in {"auto", "quote"} and quote_root is not None:
        qpath, ticker, _ = _load_atm_path(Path(quote_root), date, sig.direction)
        if qpath is not None and not getattr(qpath, "empty", True):
            probe = entry_quote_row(
                qpath,
                sig.entry_ts,
                max_lag_sec=float(cfg["max_lag_sec"]),
                max_spread_pct=float(cfg["max_spread_pct"]),
                min_mid=float(cfg["min_mid"]),
            )
            if probe is not None:
                sim = simulate_quote_tpsl(
                    qpath,
                    sig.entry_ts,
                    tp=float(cfg["tp"]),
                    sl=float(cfg["sl"]),
                    max_hold_sec=int(cfg["max_hold_sec"]),
                    fill=fill,
                    max_lag_sec=float(cfg["max_lag_sec"]),
                    max_spread_pct=float(cfg["max_spread_pct"]),
                    min_mid=float(cfg["min_mid"]),
                )
                if sim is not None and np.isfinite(sim.get("ret", np.nan)):
                    return {
                        "date": date,
                        "symbol": "QQQ",
                        "dir": sig.direction,
                        "from_open": sig.from_open,
                        "entry_ts": str(sim["entry_ts"]),
                        "exit_ts": str(sim["exit_ts"]),
                        "ticker": ticker,
                        "ret": float(sim["ret"]),
                        "exit_reason": sim["reason"],
                        "hold_sec": sim["hold_sec"],
                        "book": "quote",
                        "event_source": "qqq_open_cont",
                        "size": float(cfg["position_frac"]),
                        "pnl_frac": float(sim["ret"]) * float(cfg["position_frac"]),
                    }
        if used == "quote":
            return None

    if trades_root is None:
        return None
    tday = load_option_trades(trades_root, "QQQ", date)
    if tday is None or tday.empty:
        return None
    trade_paths = _paths_by_ticker(tday)
    ticker = None
    if quote_root is not None:
        _, ticker, _ = _load_atm_path(Path(quote_root), date, sig.direction)
        if ticker:
            key = str(ticker).replace("O:", "")
            if key not in trade_paths and str(ticker) not in trade_paths:
                ticker = None
            else:
                ticker = key if key in trade_paths else str(ticker)
    if not ticker:
        ticker, _ = _atm_from_trades(
            trade_paths, date=date, direction=sig.direction, spot=sig.spot
        )
    if not ticker or ticker not in trade_paths:
        return None
    pts, plast = trade_paths[ticker]
    sim_t = simulate_trade_tpsl(
        pts,
        plast,
        sig.entry_ts,
        tp=float(cfg["tp"]),
        sl=float(cfg["sl"]),
        max_hold_sec=int(cfg["max_hold_sec"]),
        slip=float(cfg["slip"]),
    )
    if sim_t is None or not np.isfinite(sim_t.get("ret", np.nan)):
        return None
    et = to_ny(sig.entry_ts)
    return {
        "date": date,
        "symbol": "QQQ",
        "dir": sig.direction,
        "from_open": sig.from_open,
        "entry_ts": str(et),
        "exit_ts": str(et + pd.Timedelta(seconds=float(sim_t["hold_sec"]))),
        "ticker": ticker,
        "ret": float(sim_t["ret"]),
        "exit_reason": sim_t["reason"],
        "hold_sec": sim_t["hold_sec"],
        "book": "trades",
        "event_source": "qqq_open_cont",
        "size": float(cfg["position_frac"]),
        "pnl_frac": float(sim_t["ret"]) * float(cfg["position_frac"]),
    }

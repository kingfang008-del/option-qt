"""Signal-time ATM contract selection (research).

Causal rule: only use day_iv bars with timestamp <= signal time.
Prefer trading-DTE 0, then 1, then 2 (same policy as day lock).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

NY = "America/New_York"


@dataclass(frozen=True)
class ContractPick:
    ticker: str
    strike: float
    dte: int
    spot: float
    snap_ts: pd.Timestamp
    source: str  # signal_atm | day_lock_fallback


def to_ny(ts) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        return t.tz_localize(NY)
    return t.tz_convert(NY)


def trading_dte(exp_date, trade_date) -> int:
    """NYSE-session DTE (0=same session), including exchange holidays."""
    trade = pd.Timestamp(trade_date).normalize()
    expiry = pd.Timestamp(exp_date).normalize()
    if expiry < trade:
        return -1
    if expiry == trade:
        return 0
    from preprocess.download.dte_utils import trading_sessions_between

    return int(trading_sessions_between(trade, expiry))


def _normalize_ticker(t: str) -> str:
    return str(t).replace("O:", "")


class DayIvChainCache:
    """Lazy per-(symbol,date) day_iv loader."""

    def __init__(self, day_iv_root: Path):
        self.root = Path(day_iv_root)
        self._cache: dict[tuple[str, str], pd.DataFrame | None] = {}

    def get(self, symbol: str, date: str) -> pd.DataFrame | None:
        key = (symbol, date)
        if key in self._cache:
            return self._cache[key]
        p = self.root / symbol / f"{symbol}_{date}.parquet"
        if not p.exists():
            self._cache[key] = None
            return None
        df = pd.read_parquet(p)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        if df["timestamp"].dt.tz is None:
            df["timestamp"] = df["timestamp"].dt.tz_localize(NY)
        else:
            df["timestamp"] = df["timestamp"].dt.tz_convert(NY)
        df["ticker_norm"] = df["ticker"].astype(str).map(_normalize_ticker)
        self._cache[key] = df
        return df


def pick_signal_atm(
    chain: pd.DataFrame | None,
    *,
    date: str,
    direction: str,
    sig_ts,
    spot: float,
    prefer_dte: int = 0,
    allowed_dte: list[int] | tuple[int, ...] = (0, 1, 2),
) -> ContractPick | None:
    """Pick nearest ATM (by |K-spot|) on preferred short DTE at/before sig_ts."""
    if chain is None or chain.empty or not np.isfinite(spot) or spot <= 0:
        return None
    sig_ts = to_ny(sig_ts)
    sub = chain[chain["timestamp"] <= sig_ts]
    if sub.empty:
        return None
    last = sub.sort_values("timestamp").groupby("ticker_norm", as_index=False).last()
    side = "c" if str(direction).upper() == "UP" else "p"
    last = last[last["contract_type"].astype(str).str.lower().str.startswith(side)].copy()
    if last.empty:
        return None
    last["exp"] = pd.to_datetime(last["expiration_date"]).dt.date
    trade_d = pd.Timestamp(date).date()
    last["dte"] = last["exp"].map(lambda e: trading_dte(e, trade_d))
    allowed = set(int(x) for x in allowed_dte)
    last = last[last["dte"].isin(allowed)]
    if last.empty:
        return None

    order = [int(prefer_dte)] + [int(x) for x in allowed_dte if int(x) != int(prefer_dte)]
    for dte in order:
        cand = last[last["dte"] == dte]
        if cand.empty:
            continue
        cand = cand.copy()
        cand["abs_k"] = (cand["strike_price"].astype(float) - float(spot)).abs()
        # Prefer closer strike; break ties with higher volume / tighter abs delta if present
        sort_cols = ["abs_k"]
        asc = [True]
        if "volume" in cand.columns:
            sort_cols.append("volume")
            asc.append(False)
        if "delta" in cand.columns:
            cand["abs_delta_gap"] = (cand["delta"].astype(float).abs() - 0.50).abs()
            sort_cols.append("abs_delta_gap")
            asc.append(True)
        cand = cand.sort_values(sort_cols, ascending=asc)
        row = cand.iloc[0]
        return ContractPick(
            ticker=_normalize_ticker(row["ticker_norm"]),
            strike=float(row["strike_price"]),
            dte=int(row["dte"]),
            spot=float(spot),
            snap_ts=to_ny(row["timestamp"]),
            source="signal_atm",
        )
    return None


def day_iv_path_as_quotes(
    chain: pd.DataFrame | None,
    ticker: str,
    *,
    half_spread_frac: float = 0.01,
) -> pd.DataFrame | None:
    """Build synthetic bid/ask path from day_iv close for a ticker.

    mid = close; bid = mid*(1-s); ask = mid*(1+s). Research fallback when 1s
    quotes for the signal-selected contract are missing.
    """
    if chain is None or chain.empty:
        return None
    t = _normalize_ticker(ticker)
    sub = chain[chain["ticker_norm"] == t].sort_values("timestamp")
    if sub.empty:
        return None
    mid = sub["close"].astype(float)
    ok = mid.notna() & (mid > 0)
    sub = sub.loc[ok].copy()
    if sub.empty:
        return None
    mid = sub["close"].astype(float)
    s = float(half_spread_frac)
    out = pd.DataFrame(
        {
            "timestamp": list(sub["timestamp"]),
            "ticker": t,
            "bid": (mid * (1.0 - s)).to_numpy(),
            "ask": (mid * (1.0 + s)).to_numpy(),
            "mid_price": mid.to_numpy(),
            "price": mid.to_numpy(),
        }
    )
    return out.drop_duplicates("timestamp", keep="last")


def resolve_contract(
    *,
    mode: str,
    chain: pd.DataFrame | None,
    date: str,
    direction: str,
    sig_ts,
    spot: float | None,
    day_lock_ticker: str | None,
    prefer_dte: int = 0,
    allowed_dte: list[int] | tuple[int, ...] = (0, 1, 2),
    fallback_day_lock: bool = True,
) -> ContractPick | None:
    """Resolve trade ticker under day_lock or signal_atm modes."""
    mode = str(mode or "day_lock").lower()
    if mode in ("day_lock", "prelock", "open_lock"):
        if not day_lock_ticker:
            return None
        return ContractPick(
            ticker=_normalize_ticker(day_lock_ticker),
            strike=float("nan"),
            dte=-1,
            spot=float(spot) if spot is not None else float("nan"),
            snap_ts=to_ny(sig_ts),
            source="day_lock",
        )

    if spot is None or not np.isfinite(spot):
        if fallback_day_lock and day_lock_ticker:
            return ContractPick(
                ticker=_normalize_ticker(day_lock_ticker),
                strike=float("nan"),
                dte=-1,
                spot=float("nan"),
                snap_ts=to_ny(sig_ts),
                source="day_lock_fallback",
            )
        return None

    pick = pick_signal_atm(
        chain,
        date=date,
        direction=direction,
        sig_ts=sig_ts,
        spot=float(spot),
        prefer_dte=prefer_dte,
        allowed_dte=allowed_dte,
    )
    if pick is not None:
        return pick
    if fallback_day_lock and day_lock_ticker:
        return ContractPick(
            ticker=_normalize_ticker(day_lock_ticker),
            strike=float("nan"),
            dte=-1,
            spot=float(spot),
            snap_ts=to_ny(sig_ts),
            source="day_lock_fallback",
        )
    return None


def lock_policy_from_profile(profile: dict[str, Any]) -> tuple[int, list[int]]:
    lock = profile.get("lock") or {}
    prefer = int(lock.get("prefer_dte", 0))
    allowed = [int(x) for x in (lock.get("allowed_dte") or [0, 1, 2])]
    return prefer, allowed

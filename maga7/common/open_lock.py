"""Causal open-window lock from day_iv (09:30 snapshot).

Unlike step1_build_target_map_old (all-day |delta| = look-ahead), this locks
ATM/OTM at the first RTH bar using spot + strike distance — live-feasible.

Map rows: one (bucket_id, front_dte) per contract so 0/1/2 DTE can coexist.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from maga7.common.contract_select import DayIvChainCache, trading_dte, to_ny

NY = "America/New_York"

# Classic 4-bucket: 0 Put ATM, 1 Put OTM1, 2 Call ATM, 3 Call OTM1
# Ladder OTM2+: 4 Put OTM2, 5 Call OTM2, 6 Put OTM3, 7 Call OTM3, ...
BUCKET_SPECS = [
    (0, "p", "ATM", 0),
    (1, "p", "OTM", 1),
    (2, "c", "ATM", 0),
    (3, "c", "OTM", 1),
]


def ladder_bucket_id(side: str, rung: int) -> int:
    """rung 0=ATM, 1=OTM1, 2=OTM2, ..."""
    side = side.lower()[0]
    r = int(rung)
    if r <= 0:
        return 2 if side == "c" else 0
    if r == 1:
        return 3 if side == "c" else 1
    # OTM2+ : even put, odd call starting at 4/5
    return (4 + 2 * (r - 2)) if side == "p" else (5 + 2 * (r - 2))


def direction_ladder_buckets(direction: str, otm_rungs: int = 2) -> list[int]:
    """Bucket ids locked for one trade direction (ATM + OTM1..OTMk)."""
    side = "c" if str(direction).upper() == "UP" else "p"
    n = max(0, int(otm_rungs))
    return [ladder_bucket_id(side, r) for r in range(0, n + 1)]


def bucket_specs_for_rungs(otm_rungs: int = 1) -> list[tuple[int, str, str, int]]:
    """(bucket_id, side, money_tag, rung) for ATM + OTM1..OTMk on both sides."""
    n = max(0, int(otm_rungs))
    specs: list[tuple[int, str, str, int]] = []
    for side in ("p", "c"):
        for r in range(0, n + 1):
            tag = "ATM" if r == 0 else "OTM"
            specs.append((ladder_bucket_id(side, r), side, tag, r))
    return specs


def strike_from_occ(ticker: str) -> float:
    t = str(ticker).replace("O:", "")
    m = re.search(r"[CP](\d{8})$", t)
    return float(int(m.group(1)) / 1000.0) if m else float("nan")


def is_clearly_otm(
    direction: str,
    spot: float,
    strike: float,
    *,
    thresh: float = 0.01,
) -> bool:
    """True if option is clearly OTM vs spot.

    UP/Call: OTM when strike > spot (spot below K)
    DN/Put:  OTM when strike < spot (spot above K)
    """
    if not np.isfinite(spot) or not np.isfinite(strike) or spot <= 0:
        return False
    gap = (strike - spot) / spot
    if str(direction).upper() == "UP":
        return gap >= float(thresh)  # K clearly above spot
    return (-gap) >= float(thresh)  # K clearly below spot


def _open_snapshot(chain: pd.DataFrame, date: str) -> pd.DataFrame | None:
    if chain is None or chain.empty:
        return None
    day_start = to_ny(f"{date} 09:30:00")
    sub = chain[chain["timestamp"] >= day_start]
    if sub.empty:
        sub = chain
    # first available minute
    t0 = sub["timestamp"].min()
    snap = sub[sub["timestamp"] == t0].copy()
    return snap if not snap.empty else None


def _side_candidates(snap: pd.DataFrame, side: str) -> pd.DataFrame:
    side = side.lower()[0]
    return snap[snap["contract_type"].astype(str).str.lower().str.startswith(side)].copy()


def _pick_atm(snap: pd.DataFrame, *, side: str, spot: float) -> pd.Series | None:
    cand = _side_candidates(snap, side)
    if cand.empty:
        return None
    cand = cand.assign(dist=(cand["strike_price"].astype(float) - float(spot)).abs())
    vol_key = ["dist", "volume"] if "volume" in cand.columns else ["dist"]
    return cand.sort_values(vol_key).iloc[0]


def _pick_otm_rung(
    snap: pd.DataFrame,
    *,
    side: str,
    spot: float,
    otm_rung: int,
    exclude_strikes: set[float] | None = None,
) -> pd.Series | None:
    """n-th distinct OTM strike strictly beyond spot (and beyond ATM).

    Call OTM: K > spot, ascending K (1=nearest OTM, 2=next, ...)
    Put  OTM: K < spot, descending K
    Strikes in exclude_strikes are skipped so ATM/OTM1 never collide.
    """
    if int(otm_rung) <= 0:
        return _pick_atm(snap, side=side, spot=spot)
    cand = _side_candidates(snap, side)
    if cand.empty:
        return None
    k = cand["strike_price"].astype(float)
    side = side.lower()[0]
    if side == "c":
        otm = cand[k > float(spot)].copy()  # strict: never same as ATM-at-or-below-spot edge
    else:
        otm = cand[k < float(spot)].copy()
    if otm.empty:
        return None
    excl = {float(x) for x in (exclude_strikes or set()) if np.isfinite(x)}
    if excl:
        otm = otm[~otm["strike_price"].astype(float).isin(excl)]
    if otm.empty:
        return None
    # nearest OTM first
    if side == "c":
        otm = otm.assign(otm_rank=otm["strike_price"].astype(float))
        otm = otm.sort_values(
            ["otm_rank", "volume"] if "volume" in otm.columns else ["otm_rank"]
        )
    else:
        otm = otm.assign(otm_rank=-otm["strike_price"].astype(float))
        otm = otm.sort_values(
            ["otm_rank", "volume"] if "volume" in otm.columns else ["otm_rank"]
        )
    # unique strikes in order
    seen: set[float] = set()
    ordered: list[pd.Series] = []
    for _, row in otm.iterrows():
        kk = float(row["strike_price"])
        if kk in seen:
            continue
        seen.add(kk)
        ordered.append(row)
    idx = int(otm_rung) - 1
    if idx < 0 or idx >= len(ordered):
        return None
    return ordered[idx]


def _pick_strike_side(
    snap: pd.DataFrame,
    *,
    side: str,
    spot: float,
    moneyness: str,
    otm_rung: int = 1,
    exclude_strikes: set[float] | None = None,
) -> pd.Series | None:
    """Pick ATM or the n-th strict OTM (otm_rung=1 → nearest OTM ≠ ATM)."""
    if moneyness.upper() == "ATM" or int(otm_rung) <= 0:
        return _pick_atm(snap, side=side, spot=spot)
    return _pick_otm_rung(
        snap,
        side=side,
        spot=spot,
        otm_rung=otm_rung,
        exclude_strikes=exclude_strikes,
    )


def _snap_from_option_1m(
    opt_1m_root: Path,
    symbol: str,
    date: str,
    *,
    spot: float,
    open_ts_cutoff: str = "09:45",
) -> pd.DataFrame | None:
    """Build a synthetic open snapshot from option 1m prints (covers 0DTE early)."""
    p = Path(opt_1m_root) / symbol / f"{symbol}_{date}.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    if "timestamp" not in df.columns:
        return None
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    if df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize(NY)
    else:
        df["timestamp"] = df["timestamp"].dt.tz_convert(NY)
    cutoff = to_ny(f"{date} {open_ts_cutoff}:00")
    day_start = to_ny(f"{date} 09:30:00")
    df = df[(df["timestamp"] >= day_start) & (df["timestamp"] <= cutoff)]
    if df.empty:
        return None
    df["ticker_norm"] = df["ticker"].astype(str).str.replace("O:", "", regex=False)

    def _side_exp(t: str):
        m = re.search(r"^[A-Z]+(\d{6})([CP])(\d{8})$", t)
        if not m:
            return None
        exp = pd.Timestamp("20" + m.group(1)).date()
        side = m.group(2).lower()
        strike = int(m.group(3)) / 1000.0
        return exp, side, strike

    rows = []
    for t, g in df.groupby("ticker_norm"):
        parsed = _side_exp(str(t))
        if parsed is None:
            continue
        exp, side, strike = parsed
        first = g.sort_values("timestamp").iloc[0]
        rows.append(
            {
                "ticker": t,
                "ticker_norm": t,
                "timestamp": first["timestamp"],
                "contract_type": side,
                "strike_price": strike,
                "expiration_date": exp,
                "stock_close": spot,
                "volume": float(first["v"]) if "v" in g.columns else float(first.get("volume", 0) or 0),
                "close": float(first["c"]) if "c" in g.columns else float(first.get("close", 0) or 0),
            }
        )
    if not rows:
        return None
    return pd.DataFrame(rows)


def lock_symbol_day_open(
    chain: pd.DataFrame | None,
    *,
    symbol: str,
    date: str,
    allowed_dte: Iterable[int] = (0, 1, 2),
    spot_override: float | None = None,
    option_1m_root: Path | None = None,
    otm_rungs: int = 1,
) -> list[dict[str, Any]]:
    snap = _open_snapshot(chain, date)
    spot = float(spot_override) if spot_override is not None else None
    if spot is None and snap is not None and "stock_close" in snap.columns:
        spot = float(snap["stock_close"].median())
    if spot is None or not np.isfinite(spot) or spot <= 0:
        return []

    # Merge day_iv open snap with option_1m early prints (fills missing 0DTE).
    frames = []
    if snap is not None and not snap.empty:
        frames.append(snap)
    if option_1m_root is not None:
        m1 = _snap_from_option_1m(option_1m_root, symbol, date, spot=spot)
        if m1 is not None and not m1.empty:
            frames.append(m1)
    if not frames:
        return []
    # Prefer day_iv rows when both exist (has volume/greeks continuity); keep union of tickers
    merged = pd.concat(frames, ignore_index=True, sort=False)
    if "ticker_norm" not in merged.columns:
        merged["ticker_norm"] = merged["ticker"].astype(str).str.replace("O:", "", regex=False)
    merged = merged.drop_duplicates("ticker_norm", keep="first")

    merged = merged.copy()
    merged["exp"] = pd.to_datetime(merged["expiration_date"]).dt.date
    trade_d = pd.Timestamp(date).date()
    merged["dte"] = merged["exp"].map(lambda e: trading_dte(e, trade_d))

    rows: list[dict[str, Any]] = []
    specs = bucket_specs_for_rungs(otm_rungs)
    for dte in [int(x) for x in allowed_dte]:
        sub = merged[merged["dte"] == dte]
        if sub.empty:
            continue
        # Pick ATM first per side; each OTM rung = next distinct OTM after exclusions.
        picked_by_side: dict[str, set[float]] = {"p": set(), "c": set()}
        for bucket_id, side, money, rung in sorted(specs, key=lambda x: (x[1], x[3])):
            excl = set(picked_by_side.get(side, set()))
            if rung <= 0:
                picked = _pick_atm(sub, side=side, spot=spot)
            else:
                # Always take the *next* available OTM after already-picked strikes.
                picked = _pick_otm_rung(
                    sub,
                    side=side,
                    spot=spot,
                    otm_rung=1,
                    exclude_strikes=excl,
                )
            if picked is None:
                continue
            ticker = str(picked.get("ticker_norm") or picked["ticker"]).replace("O:", "")
            k = float(picked["strike_price"])
            picked_by_side.setdefault(side, set()).add(k)
            rung_tag = "ATM" if rung <= 0 else f"OTM{rung}"
            rows.append(
                {
                    "date_str": date,
                    "symbol": symbol,
                    "contract_symbol": f"O:{ticker}",
                    "bucket_id": int(bucket_id),
                    "front_dte": int(dte),
                    "dte_mode": "trading",
                    "lock_spot": spot,
                    "lock_ts": str(picked["timestamp"]),
                    "strike": k,
                    "ladder_rung": int(rung),
                    "tag": f"open_{dte}dte_{rung_tag}_{side}",
                }
            )
    return rows


def build_open_lock_map(
    *,
    day_iv_root: Path,
    symbols: list[str],
    dates: list[str] | None = None,
    start: str | None = None,
    end: str | None = None,
    allowed_dte: Iterable[int] = (0, 1, 2),
    stock_by: dict[str, pd.DataFrame] | None = None,
    option_1m_root: Path | None = None,
    otm_rungs: int = 1,
) -> pd.DataFrame:
    cache = DayIvChainCache(day_iv_root)
    rows: list[dict[str, Any]] = []
    for sym in symbols:
        if dates is None:
            files = sorted(
                p
                for p in (Path(day_iv_root) / sym).glob(f"{sym}_*.parquet")
                if "_high_features" not in p.name
            )
            sym_dates = [p.stem.split("_", 1)[1] for p in files]
            if start:
                sym_dates = [d for d in sym_dates if d >= start]
            if end:
                sym_dates = [d for d in sym_dates if d <= end]
        else:
            sym_dates = list(dates)
        for date in sym_dates:
            spot = None
            if stock_by and sym in stock_by:
                sdf = stock_by[sym]
                day = sdf[sdf["date"] == date].sort_values("timestamp")
                if not day.empty and "open" in day.columns:
                    spot = float(day.iloc[0]["open"])
            rows.extend(
                lock_symbol_day_open(
                    cache.get(sym, date),
                    symbol=sym,
                    date=date,
                    allowed_dte=allowed_dte,
                    spot_override=spot,
                    option_1m_root=option_1m_root,
                    otm_rungs=otm_rungs,
                )
            )
    return pd.DataFrame(rows)


def load_multidte_lock_index(
    lock_path: Path,
) -> dict[tuple[str, str], dict[int, dict[int, str]]]:
    """(symbol, date) -> front_dte -> bucket_id -> ticker (no O:)."""
    lock = pd.read_parquet(lock_path)
    lock["contract"] = lock["contract_symbol"].astype(str).str.replace("O:", "", regex=False)
    out: dict[tuple[str, str], dict[int, dict[int, str]]] = {}
    for (sym, date), g in lock.groupby(["symbol", "date_str"]):
        by_dte: dict[int, dict[int, str]] = {}
        for r in g.itertuples():
            dte = int(r.front_dte)
            by_dte.setdefault(dte, {})[int(r.bucket_id)] = str(r.contract)
        out[(str(sym), str(date))] = by_dte
    return out


def resolve_open_lock_contract(
    by_dte: dict[int, dict[int, str]] | None,
    *,
    direction: str,
    moneyness: str,
    spot: float | None,
    prefer_dte: int = 0,
    allowed_dte: Iterable[int] = (0, 1, 2),
    clear_otm_thresh: float | None = 0.01,
    bucket_map: dict[tuple[str, str], int] | None = None,
    ladder: bool = False,
    otm_rungs: int = 2,
) -> tuple[str | None, int | None, str]:
    """Pick contract from multi-DTE open lock.

    Modes:
      - classic: fixed bucket (ATM/OTM) + optional clear-OTM 0DTE ban
      - ladder: among ATM+OTM1..OTMk for the direction, pick strike closest to spot
    """
    if not by_dte:
        return None, None, "no_lock"
    if bucket_map is None:
        from maga7.common.replay import BUCKET_MAP

        bucket_map = BUCKET_MAP
    allowed = [int(x) for x in allowed_dte]
    order = [int(prefer_dte)] + [d for d in allowed if d != int(prefer_dte)]

    if ladder and spot is not None and np.isfinite(float(spot)) and float(spot) > 0:
        bids = direction_ladder_buckets(direction, otm_rungs=otm_rungs)
        for dte in order:
            buckets = by_dte.get(dte) or {}
            cands: list[tuple[str, float]] = []
            for bid in bids:
                ticker = buckets.get(bid)
                if not ticker:
                    continue
                k = strike_from_occ(ticker)
                if not np.isfinite(k):
                    continue
                cands.append((ticker, float(k)))
            if not cands:
                continue
            best = min(cands, key=lambda tk: abs(tk[1] - float(spot)))
            ticker, strike = best
            if (
                clear_otm_thresh is not None
                and dte == 0
                and is_clearly_otm(direction, float(spot), strike, thresh=float(clear_otm_thresh))
            ):
                # entire 0DTE ladder is still clearly OTM vs spot → skip to 1DTE+
                continue
            reason = "open_ladder"
            if dte > 0 and clear_otm_thresh is not None:
                # annotate if 0DTE ATM would have been clear-OTM
                atm_bid = bucket_map.get((direction, "ATM"))
                z0 = (by_dte.get(0) or {}).get(atm_bid) if atm_bid is not None else None
                if z0 and is_clearly_otm(
                    direction, float(spot), strike_from_occ(z0), thresh=float(clear_otm_thresh)
                ):
                    reason = "open_ladder_skip0_clear_otm"
            return ticker, dte, reason
        return None, None, "exhausted"

    bid = bucket_map[(direction, moneyness)]
    for dte in order:
        buckets = by_dte.get(dte) or {}
        ticker = buckets.get(bid)
        if not ticker:
            continue
        strike = strike_from_occ(ticker)
        if (
            clear_otm_thresh is not None
            and dte == 0
            and spot is not None
            and is_clearly_otm(direction, float(spot), strike, thresh=float(clear_otm_thresh))
        ):
            continue  # ban 0DTE when clearly OTM
        reason = "open_lock"
        if dte > 0 and clear_otm_thresh is not None and spot is not None:
            z0 = (by_dte.get(0) or {}).get(bid)
            if z0 and is_clearly_otm(direction, float(spot), strike_from_occ(z0), thresh=float(clear_otm_thresh)):
                reason = "open_lock_skip0_clear_otm"
        return ticker, dte, reason
    return None, None, "exhausted"

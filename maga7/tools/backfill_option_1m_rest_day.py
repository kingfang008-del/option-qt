#!/usr/bin/env python3
"""Download one day of Mag7 option 1m bars via Polygon REST (when S3 flatfiles lag).

Writes S3-compatible parquet:
  {option_1m_root}/{SYM}/{SYM}_{date}.parquet
  columns: ticker, v, o, c, h, l, t, n, timestamp
"""
from __future__ import annotations

import argparse
import concurrent.futures
import os
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
from polygon import RESTClient
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from preprocess.download.dte_utils import trading_sessions_between

DEFAULT_SYMBOLS = ["NVDA", "TSLA", "AAPL", "AMZN", "META", "MSFT", "AMD", "GOOGL"]
DEFAULT_OUT = Path("/mnt/s990/new_option_data_s3")
NY = "America/New_York"


def _api_key() -> str:
    for k in ("MASSIVE_API_KEY", "POLYGON_API_KEY", "POLYGON_KEY"):
        v = os.environ.get(k, "").strip()
        if v:
            return v
    raise SystemExit("need MASSIVE_API_KEY or POLYGON_API_KEY")


def _spot_from_stock_1s(stock_1s_root: Path, symbol: str, day: str) -> float | None:
    p = stock_1s_root / symbol / f"{symbol}_{day}.parquet"
    if not p.is_file():
        return None
    df = pd.read_parquet(p)
    if df.empty:
        return None
    ts = pd.to_datetime(df["timestamp"] if "timestamp" in df.columns else df["ts"], utc=True)
    if getattr(ts.dt, "tz", None) is not None:
        ts = ts.dt.tz_convert(NY)
    else:
        ts = ts.dt.tz_localize("UTC").dt.tz_convert(NY)
    df = df.assign(timestamp=ts).sort_values("timestamp")
    day_df = df[df["timestamp"].dt.strftime("%Y-%m-%d") == day]
    if day_df.empty:
        day_df = df
    col = "open" if "open" in day_df.columns else "close"
    return float(day_df.iloc[0][col])


def _candidate_expiries(trade_day: str, max_dte: int = 2) -> list[str]:
    """Calendar window covering trading DTE 0..max_dte (weekends/holidays padded)."""
    d0 = date.fromisoformat(trade_day)
    out = []
    for i in range(0, max_dte + 14):
        d = d0 + timedelta(days=i)
        if d.weekday() >= 5:
            continue
        out.append(d.isoformat())
        if len(out) >= max_dte + 6:
            break
    return out


def _list_near_contracts(
    client: RESTClient,
    *,
    symbol: str,
    trade_day: str,
    spot: float,
    strike_pct: float,
    allowed_dte: tuple[int, ...],
) -> list[str]:
    lo, hi = spot * (1 - strike_pct), spot * (1 + strike_pct)
    tickers: set[str] = set()
    trade_ts = pd.Timestamp(trade_day)
    for exp in _candidate_expiries(trade_day, max_dte=max(allowed_dte)):
        dte = trading_sessions_between(trade_ts, pd.Timestamp(exp))
        if dte not in allowed_dte:
            continue
        contracts = []
        # Past expiries need expired=True; future need False/omit.
        for expired_flag in (True, False):
            try:
                contracts = list(
                    client.list_options_contracts(
                        underlying_ticker=symbol,
                        expiration_date=exp,
                        expired=expired_flag,
                        limit=1000,
                    )
                )
            except Exception:
                contracts = []
            if contracts:
                break
        for c in contracts:
            k = float(getattr(c, "strike_price", 0) or 0)
            if k <= 0 or k < lo or k > hi:
                continue
            t = str(c.ticker)
            if not t.startswith("O:"):
                t = f"O:{t}"
            tickers.add(t)
    return sorted(tickers)


def _fetch_one(args: tuple[str, str, str]) -> pd.DataFrame | None:
    api_key, ticker, day = args
    client = RESTClient(api_key)
    try:
        aggs = list(
            client.list_aggs(
                ticker=ticker,
                multiplier=1,
                timespan="minute",
                from_=day,
                to=day,
                limit=50000,
            )
        )
    except Exception:
        return None
    if not aggs:
        return None
    rows = []
    for a in aggs:
        ts = pd.Timestamp(a.timestamp, unit="ms", tz="UTC").tz_convert(NY)
        rows.append(
            {
                "ticker": ticker,
                "v": float(getattr(a, "volume", 0) or 0),
                "o": float(getattr(a, "open", 0) or 0),
                "c": float(getattr(a, "close", 0) or 0),
                "h": float(getattr(a, "high", 0) or 0),
                "l": float(getattr(a, "low", 0) or 0),
                "t": int(a.timestamp),
                "n": int(getattr(a, "transactions", 0) or 0),
                "timestamp": ts,
            }
        )
    return pd.DataFrame(rows)


def download_symbol_day(
    *,
    symbol: str,
    day: str,
    out_root: Path,
    stock_1s_root: Path,
    api_key: str,
    strike_pct: float,
    max_workers: int,
    allowed_dte: tuple[int, ...],
) -> dict:
    spot = _spot_from_stock_1s(stock_1s_root, symbol, day)
    if spot is None or spot <= 0:
        return {"symbol": symbol, "ok": False, "reason": "no_spot"}
    client = RESTClient(api_key)
    tickers = _list_near_contracts(
        client,
        symbol=symbol,
        trade_day=day,
        spot=spot,
        strike_pct=strike_pct,
        allowed_dte=allowed_dte,
    )
    if not tickers:
        return {"symbol": symbol, "ok": False, "reason": "no_contracts", "spot": spot}
    jobs = [(api_key, t, day) for t in tickers]
    frames: list[pd.DataFrame] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        for df in tqdm(
            ex.map(_fetch_one, jobs),
            total=len(jobs),
            desc=f"{symbol} option_1m",
            leave=False,
        ):
            if df is not None and not df.empty:
                frames.append(df)
    if not frames:
        return {
            "symbol": symbol,
            "ok": False,
            "reason": "no_aggs",
            "spot": spot,
            "n_tickers": len(tickers),
        }
    out = pd.concat(frames, ignore_index=True)
    out_dir = out_root / symbol
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{symbol}_{day}.parquet"
    out.to_parquet(path, index=False)
    return {
        "symbol": symbol,
        "ok": True,
        "spot": spot,
        "n_tickers": len(tickers),
        "n_tickers_with_data": int(out["ticker"].nunique()),
        "n_rows": int(len(out)),
        "path": str(path),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--date", required=True, help="YYYY-MM-DD")
    ap.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    ap.add_argument("--option-1m-root", default=str(DEFAULT_OUT))
    ap.add_argument("--stock-1s-root", default="/mnt/s990/data/raw_1s/stocks")
    ap.add_argument("--strike-pct", type=float, default=0.12, help="keep strikes within ±pct of spot")
    ap.add_argument("--max-workers", type=int, default=24)
    ap.add_argument("--allowed-dte", default="0,1,2")
    args = ap.parse_args()

    api_key = _api_key()
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    allowed = tuple(int(x) for x in args.allowed_dte.split(",") if x.strip())
    out_root = Path(args.option_1m_root).expanduser()
    stock_root = Path(args.stock_1s_root).expanduser()
    reports = []
    for sym in symbols:
        rep = download_symbol_day(
            symbol=sym,
            day=args.date,
            out_root=out_root,
            stock_1s_root=stock_root,
            api_key=api_key,
            strike_pct=float(args.strike_pct),
            max_workers=int(args.max_workers),
            allowed_dte=allowed,
        )
        print(rep, flush=True)
        reports.append(rep)
    n_ok = sum(1 for r in reports if r.get("ok"))
    print({"ok": n_ok, "total": len(reports)}, flush=True)
    if n_ok == 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build QQQ 0DTE ladder map from the options contracts API.

The existing day-IV source can miss many 0DTE near-ATM contracts because IV/Greek
calculation drops difficult same-day quotes.  For short-DTE microstructure work,
the universe should come from the contract chain itself, anchored by the
underlying spot near the open.
"""
from __future__ import annotations

import argparse
import ast
import json
import logging
import math
import os
from pathlib import Path
from typing import Any

import pandas as pd
from polygon import RESTClient
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("build_0dte_api_ladder_map")


def load_legacy_api_key() -> str:
    legacy_path = Path(__file__).with_name("step2_polygon_second_sniper_v1.py")
    try:
        tree = ast.parse(legacy_path.read_text(encoding="utf-8"))
        for node in tree.body:
            if not isinstance(node, ast.Assign):
                continue
            if not any(isinstance(t, ast.Name) and t.id == "API_KEY" for t in node.targets):
                continue
            val = node.value
            if isinstance(val, ast.Call) and len(val.args) >= 2 and isinstance(val.args[1], ast.Constant):
                return str(val.args[1].value)
            if isinstance(val, ast.Constant):
                return str(val.value)
    except Exception:
        return ""
    return ""


API_KEY = os.environ.get("POLYGON_API_KEY") or load_legacy_api_key()
NY = "America/New_York"


def stock_lock_price(
    symbol: str,
    date_str: str,
    minute: str = "09:40",
    client: RESTClient | None = None,
) -> float | None:
    month = date_str[:7]
    p = Path.home() / f"train_data/spnq_train_resampled/{symbol}/regular/09:30-16:00/1min/{month}.parquet"
    if p.exists():
        df = pd.read_parquet(p, columns=["timestamp", "close"])
        ts = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(NY)
        target = pd.Timestamp(f"{date_str} {minute}:00", tz=NY)
        sub = df[(ts.dt.date.astype(str) == date_str)].copy()
        if not sub.empty:
            sub["_ts"] = ts[ts.dt.date.astype(str) == date_str].values
            sub = sub.sort_values("_ts")
            before = sub[sub["_ts"] <= target.tz_localize(None)]
            row = before.iloc[-1] if not before.empty else sub.iloc[0]
            return float(row["close"])
    if client is None:
        return None
    # Fallback when local 1min parquet is missing (e.g. newly started month).
    try:
        bars = list(
            client.list_aggs(
                ticker=symbol,
                multiplier=1,
                timespan="minute",
                from_=date_str,
                to=date_str,
                limit=50000,
            )
        )
    except Exception as exc:
        logger.warning("polygon minute lock fallback failed %s %s: %s", symbol, date_str, exc)
        return None
    if not bars:
        return None
    rows = []
    for b in bars:
        ts = pd.Timestamp(b.timestamp, unit="ms", tz="UTC").tz_convert(NY)
        rows.append({"_ts": ts, "close": float(b.close)})
    sub = pd.DataFrame(rows).sort_values("_ts")
    target = pd.Timestamp(f"{date_str} {minute}:00", tz=NY)
    before = sub[sub["_ts"] <= target]
    row = before.iloc[-1] if not before.empty else sub.iloc[0]
    return float(row["close"])


def trading_dates(symbol: str, start: str, end: str, client: RESTClient | None = None) -> list[str]:
    start_m, end_m = start[:7], end[:7]
    root = Path.home() / f"train_data/spnq_train_resampled/{symbol}/regular/09:30-16:00/1min"
    dates: set[str] = set()
    for f in sorted(root.glob("*.parquet")):
        mon = f.stem
        if mon < start_m or mon > end_m:
            continue
        df = pd.read_parquet(f, columns=["timestamp"])
        ts = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(NY)
        for d in ts.dt.date.astype(str).unique():
            if start <= d <= end:
                dates.add(d)
    if dates:
        return sorted(dates)
    if client is None:
        return []
    # Fallback: Polygon daily bars for months without local 1min files.
    try:
        bars = list(
            client.list_aggs(
                ticker=symbol,
                multiplier=1,
                timespan="day",
                from_=start,
                to=end,
                limit=50000,
            )
        )
    except Exception as exc:
        logger.warning("polygon trading_dates fallback failed %s: %s", symbol, exc)
        return []
    for b in bars:
        d = pd.Timestamp(b.timestamp, unit="ms", tz="UTC").tz_convert(NY).strftime("%Y-%m-%d")
        if start <= d <= end:
            dates.add(d)
    return sorted(dates)


def get_contract_rows(client: RESTClient, symbol: str, date_str: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for c in client.list_options_contracts(
        underlying_ticker=symbol,
        expiration_date=date_str,
        expired="true",
        limit=1000,
        sort="strike_price",
        order="asc",
    ):
        rows.append(
            {
                "ticker": getattr(c, "ticker", ""),
                "contract_type": str(getattr(c, "contract_type", "")).lower(),
                "expiration_date": str(getattr(c, "expiration_date", date_str)),
                "strike_price": float(getattr(c, "strike_price", 0.0)),
            }
        )
    return pd.DataFrame(rows)


def select_side(chain: pd.DataFrame, spot: float, side: str, n: int) -> pd.DataFrame:
    typ = "put" if side == "PUT" else "call"
    sub = chain[chain["contract_type"] == typ].copy()
    if sub.empty:
        return sub
    if side == "PUT":
        preferred = sub[sub["strike_price"] <= spot].sort_values("strike_price", ascending=False).head(n)
        fallback = sub[sub["strike_price"] > spot].sort_values("strike_price", ascending=True)
    else:
        preferred = sub[sub["strike_price"] >= spot].sort_values("strike_price", ascending=True).head(n)
        fallback = sub[sub["strike_price"] < spot].sort_values("strike_price", ascending=False)
    if len(preferred) < n:
        preferred = pd.concat([preferred, fallback.head(n - len(preferred))], ignore_index=True)
    return preferred.head(n).copy()


def build_day(client: RESTClient, symbol: str, date_str: str, n_per_side: int, lock_minute: str) -> list[dict]:
    spot = stock_lock_price(symbol, date_str, lock_minute, client=client)
    if spot is None:
        return []
    chain = get_contract_rows(client, symbol, date_str)
    if chain.empty:
        return []
    rows = []
    for side, offset in [("PUT", 0), ("CALL", n_per_side)]:
        sel = select_side(chain, spot, side, n_per_side)
        if len(sel) < n_per_side:
            return []
        for i, (_, r) in enumerate(sel.iterrows()):
            strike = float(r["strike_price"])
            rows.append(
                {
                    "date_str": date_str,
                    "contract_symbol": str(r["ticker"]),
                    "bucket_id": offset + i,
                    "symbol": symbol,
                    "tag": f"{side}_K{i:02d}",
                    "side": side,
                    "target_abs_delta": float("nan"),
                    "target_dte": 0,
                    "expiration": date_str,
                    "strike": strike,
                    "stock_close_at_lock": spot,
                    "premium_at_lock": float("nan"),
                    "premium_pct_at_lock": float("nan"),
                    "delta_at_lock": float("nan"),
                    "abs_delta_at_lock": float("nan"),
                    "moneyness_at_lock": math.log(strike / spot) if strike > 0 and spot > 0 else float("nan"),
                    "volume_at_lock": float("nan"),
                    "lock_timestamp": pd.Timestamp(f"{date_str} {lock_minute}:00", tz=NY).isoformat(),
                    "selected_dte": 0,
                }
            )
    return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="QQQ")
    p.add_argument("--start-date", default="2026-01-01")
    p.add_argument("--end-date", default="2026-06-30")
    p.add_argument("--n-per-side", type=int, default=4)
    p.add_argument("--lock-minute", default="09:40")
    p.add_argument("--output", default=str(Path.home() / "train_data/locked_targets_map_0dte_api_ladder.parquet"))
    p.add_argument("--report", default="qqq_btc/results/locked_targets_map_0dte_api_ladder_report.json")
    p.add_argument("--api-key", default=API_KEY)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.api_key:
        raise SystemExit("POLYGON_API_KEY is not set and no legacy API key found")
    client = RESTClient(args.api_key)
    dates = trading_dates(args.symbol, args.start_date, args.end_date, client=client)
    all_rows = []
    missing = []
    for d in tqdm(dates, desc="0dte-api-map"):
        rows = build_day(client, args.symbol, d, args.n_per_side, args.lock_minute)
        if rows:
            all_rows.extend(rows)
        else:
            missing.append(d)
    if not all_rows:
        raise SystemExit("no contracts selected")
    out = pd.DataFrame(all_rows).sort_values(["symbol", "date_str", "bucket_id"]).reset_index(drop=True)
    out_path = Path(args.output).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False, compression="zstd")
    summary = {
        "symbol": args.symbol,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "n_per_side": args.n_per_side,
        "trading_dates": len(dates),
        "selected_days": int(out["date_str"].nunique()),
        "rows": int(len(out)),
        "missing_days": missing,
        "output": str(out_path),
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

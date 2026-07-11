#!/usr/bin/env python3
"""Build MAG7 short-DTE (trading dte∈{0,1,2}) ATM ladder locked maps via Polygon.

Unlike QQQ daily-0DTE maps, this selects — for each trade_date — the expiry whose
trading DTE equals the requested bucket, then locks near-ATM puts/calls.

Default research window starts 2026-02-01 (MAG7 Mon/Wed expiries ~Feb 2026).
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

from preprocess.download.build_0dte_api_ladder_map import (
    get_contract_rows,
    select_side,
    stock_lock_price,
    trading_dates,
)
from preprocess.download.dte_utils import trading_sessions_between

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("build_mag7_short_dte_api_ladder_map")

NY = "America/New_York"


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


def trading_dates_union(symbol: str, start: str, end: str, client: RESTClient) -> list[str]:
    """Local 1min dates plus Polygon daily bars (local stock bars may lag)."""
    local = set(trading_dates(symbol, start, end, client=None))
    poly = set(trading_dates(symbol, start, end, client=client))
    # If local returned something, trading_dates skips polygon — force polygon path.
    if not poly or local == poly:
        # Explicit polygon daily pull when local is a strict subset of the window.
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
            for b in bars:
                d = pd.Timestamp(b.timestamp, unit="ms", tz="UTC").tz_convert(NY).strftime("%Y-%m-%d")
                if start <= d <= end:
                    poly.add(d)
        except Exception as exc:
            logger.warning("polygon date union failed %s: %s", symbol, exc)
    return sorted(local | poly)


def candidate_expiries(trade_date: str, horizon_cal_days: int = 14) -> list[str]:
    start = pd.Timestamp(trade_date)
    end = start + pd.Timedelta(days=horizon_cal_days)
    return [d.strftime("%Y-%m-%d") for d in pd.bdate_range(start, end)]


def resolve_expiry_for_dte(
    client: RESTClient,
    symbol: str,
    trade_date: str,
    target_dte: int,
    expiry_cache: dict[str, bool],
) -> str | None:
    """Return the nearest listed expiry with trading_dte == target_dte."""
    qd = pd.Timestamp(trade_date)
    hits: list[str] = []
    for exp in candidate_expiries(trade_date):
        if exp not in expiry_cache:
            try:
                hit = next(
                    client.list_options_contracts(
                        underlying_ticker=symbol,
                        expiration_date=exp,
                        expired="true",
                        limit=1,
                    ),
                    None,
                )
            except Exception:
                hit = None
            expiry_cache[exp] = hit is not None
        if not expiry_cache[exp]:
            continue
        if trading_sessions_between(qd, pd.Timestamp(exp)) == target_dte:
            hits.append(exp)
    return hits[0] if hits else None


def build_day_bucket(
    client: RESTClient,
    symbol: str,
    date_str: str,
    target_dte: int,
    n_per_side: int,
    lock_minute: str,
    expiry_cache: dict[str, bool],
) -> list[dict[str, Any]]:
    spot = stock_lock_price(symbol, date_str, lock_minute, client=client)
    if spot is None:
        return []
    exp = resolve_expiry_for_dte(client, symbol, date_str, target_dte, expiry_cache)
    if exp is None:
        return []
    chain = get_contract_rows(client, symbol, exp)
    if chain.empty:
        return []
    rows: list[dict[str, Any]] = []
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
                    "tag": f"DTE{target_dte}_{side}_K{i:02d}",
                    "side": side,
                    "target_abs_delta": float("nan"),
                    "target_dte": int(target_dte),
                    "selected_dte": int(target_dte),
                    "expiration": exp,
                    "strike": strike,
                    "stock_close_at_lock": spot,
                    "premium_at_lock": float("nan"),
                    "premium_pct_at_lock": float("nan"),
                    "delta_at_lock": float("nan"),
                    "abs_delta_at_lock": float("nan"),
                    "moneyness_at_lock": math.log(strike / spot) if strike > 0 and spot > 0 else float("nan"),
                    "volume_at_lock": float("nan"),
                    "lock_timestamp": pd.Timestamp(f"{date_str} {lock_minute}:00", tz=NY).isoformat(),
                }
            )
    return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--symbols", default="NVDA,TSLA")
    p.add_argument("--start-date", default="2026-02-02", help="First MAG7 Monday expiry / short-DTE window start")
    p.add_argument("--end-date", default="2026-07-09")
    p.add_argument("--dtes", default="0,1,2")
    p.add_argument("--n-per-side", type=int, default=4)
    p.add_argument("--lock-minute", default="09:40")
    p.add_argument(
        "--output",
        default=str(Path.home() / "train_data/locked_targets_map_mag7_short_dte_api_ladder.parquet"),
    )
    p.add_argument(
        "--report",
        default="stock_options/results/locked_targets_map_mag7_short_dte_api_ladder_report.json",
    )
    p.add_argument("--api-key", default=API_KEY)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.api_key:
        raise SystemExit("POLYGON_API_KEY is not set and no legacy API key found")
    client = RESTClient(args.api_key)
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    dtes = [int(x) for x in args.dtes.split(",") if x.strip() != ""]

    all_rows: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    expiry_cache: dict[str, dict[str, bool]] = {}

    for symbol in symbols:
        dates = trading_dates_union(symbol, args.start_date, args.end_date, client=client)
        logger.info("%s trading days: %d (%s .. %s)", symbol, len(dates), dates[0] if dates else None, dates[-1] if dates else None)
        expiry_cache.setdefault(symbol, {})
        for dte in dtes:
            for d in tqdm(dates, desc=f"{symbol}-dte{dte}"):
                rows = build_day_bucket(
                    client,
                    symbol,
                    d,
                    dte,
                    args.n_per_side,
                    args.lock_minute,
                    expiry_cache[symbol],
                )
                if rows:
                    all_rows.extend(rows)
                else:
                    missing.append({"symbol": symbol, "date_str": d, "target_dte": dte})

    if not all_rows:
        raise SystemExit("no contracts selected")

    out = pd.DataFrame(all_rows).sort_values(["symbol", "date_str", "selected_dte", "bucket_id"]).reset_index(drop=True)
    out_path = Path(args.output).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False, compression="zstd")

    by = (
        out.groupby(["symbol", "selected_dte"])["date_str"]
        .nunique()
        .reset_index(name="n_days")
        .to_dict(orient="records")
    )
    summary = {
        "symbols": symbols,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "dtes": dtes,
        "n_per_side": args.n_per_side,
        "rows": int(len(out)),
        "selected_day_buckets": by,
        "missing_day_buckets": len(missing),
        "missing_sample": missing[:50],
        "note": "Default start 2026-02-01: MAG7 Mon/Wed expiries roughly begin then.",
        "output": str(out_path),
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

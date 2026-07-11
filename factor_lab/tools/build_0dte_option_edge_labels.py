#!/usr/bin/env python3
"""Build high-frequency 0DTE option-edge labels from 1m bid/ask quotes.

The label is a strict long-option execution return:
  entry at current ask, exit at future bid, minus round-trip commission.

Output is one row per minute/contract/bucket with multiple horizons.
This is intentionally model-free and is used to verify whether tradable
short-horizon option edge exists before training another predictor.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


DEFAULT_HORIZONS = (1, 3, 5, 10)
NY = "America/New_York"


def parse_side(ticker: object) -> str | None:
    """Parse OCC side from a QQQ option ticker."""
    m = re.search(r"\d{6}([CP])\d{8}$", str(ticker).replace("O:", ""))
    if not m:
        return None
    return "CALL" if m.group(1) == "C" else "PUT"


def to_ny_timestamp(series: pd.Series) -> pd.Series:
    ts = pd.to_datetime(series, errors="coerce")
    if ts.dt.tz is None:
        return ts.dt.tz_localize("UTC").dt.tz_convert(NY)
    return ts.dt.tz_convert(NY)


def future_stack(values: pd.Series, horizon: int) -> pd.DataFrame:
    """Future rows t+1..t+h for MFE/MAE style stats."""
    return pd.concat([values.shift(-i) for i in range(1, horizon + 1)], axis=1)


def add_group_labels(group: pd.DataFrame, horizons: list[int], commission_per_contract: float) -> pd.DataFrame:
    g = group.sort_values("timestamp").copy()
    ask = pd.to_numeric(g["ask"], errors="coerce").replace(0.0, np.nan)
    bid = pd.to_numeric(g["bid"], errors="coerce").replace(0.0, np.nan)
    round_trip_cost = 2.0 * float(commission_per_contract)
    cost_frac = round_trip_cost / (ask * 100.0)

    for h in horizons:
        future_bid = bid.shift(-h)
        window = future_stack(bid, h)
        g[f"ret_{h}m"] = future_bid / ask - 1.0 - cost_frac
        g[f"mfe_{h}m"] = window.max(axis=1) / ask - 1.0 - cost_frac
        g[f"mae_{h}m"] = window.min(axis=1) / ask - 1.0 - cost_frac
    return g


def process_day(
    path: Path,
    *,
    symbol: str,
    horizons: list[int],
    commission_per_contract: float,
    max_spread_pct: float,
) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if df.empty:
        return pd.DataFrame()

    required = {"timestamp", "ticker", "bucket_id", "bid", "ask"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")

    out = df.copy()
    out["timestamp"] = to_ny_timestamp(out["timestamp"]).dt.floor("min")
    out["date_str"] = path.stem.replace(f"{symbol}_", "")
    out["ticker"] = out["ticker"].astype(str).str.replace("O:", "", regex=False)
    out["side"] = out["ticker"].map(parse_side)
    out["bucket_id"] = pd.to_numeric(out["bucket_id"], errors="coerce").astype("Int64")
    for c in ("bid", "ask", "close", "open", "high", "low", "volume", "spread_pct", "bid_size", "ask_size"):
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    if "spread_pct" not in out.columns:
        mid = (out["bid"] + out["ask"]) / 2.0
        out["spread_pct"] = (out["ask"] - out["bid"]) / mid.replace(0.0, np.nan)

    out = out[
        out["side"].isin(["CALL", "PUT"])
        & out["bucket_id"].notna()
        & (out["bid"] > 0)
        & (out["ask"] > 0)
        & (out["spread_pct"] <= max_spread_pct)
    ].copy()
    if out.empty:
        return pd.DataFrame()

    out = out.sort_values(["ticker", "timestamp"]).drop_duplicates(["ticker", "timestamp"], keep="last")
    parts = [
        add_group_labels(g, horizons, commission_per_contract)
        for _, g in out.groupby("ticker", sort=False)
    ]
    labeled = pd.concat(parts, ignore_index=True).sort_values(["timestamp", "bucket_id", "ticker"])

    keep = [
        "date_str",
        "timestamp",
        "ticker",
        "bucket_id",
        "side",
        "bid",
        "ask",
        "spread_pct",
    ]
    optional = [c for c in ("close", "volume", "bid_size", "ask_size", "volume_imbalance") if c in labeled.columns]
    label_cols = [c for h in horizons for c in (f"ret_{h}m", f"mfe_{h}m", f"mae_{h}m")]
    return labeled[keep + optional + label_cols].reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build 0DTE short-horizon option edge labels")
    parser.add_argument("--option-1m-root", default="/mnt/s990/data/raw_1m/dte0_options")
    parser.add_argument("--symbol", default="QQQ")
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--output-dir", default=str(Path.home() / "train_data/option_edge_labels_0dte"))
    parser.add_argument("--horizons", default="1,3,5,10", help="comma-separated minute horizons")
    parser.add_argument("--commission-per-contract", type=float, default=0.65)
    parser.add_argument("--max-spread-pct", type=float, default=0.30)
    parser.add_argument("--single-file", action="store_true", help="also write one combined parquet")
    args = parser.parse_args()

    horizons = [int(x) for x in args.horizons.split(",") if x.strip()]
    in_dir = Path(args.option_1m_root).expanduser() / args.symbol
    out_dir = Path(args.output_dir).expanduser() / args.symbol
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(in_dir.glob(f"{args.symbol}_*.parquet"))
    files = [
        p for p in files
        if args.start_date <= p.stem.replace(f"{args.symbol}_", "") <= args.end_date
    ]
    if not files:
        raise SystemExit(f"no files under {in_dir} for {args.start_date}..{args.end_date}")

    report = {
        "symbol": args.symbol,
        "input": str(in_dir),
        "output": str(out_dir),
        "start_date": args.start_date,
        "end_date": args.end_date,
        "horizons": horizons,
        "days": 0,
        "rows": 0,
        "by_day": {},
    }
    frames: list[pd.DataFrame] = []
    for fp in tqdm(files, desc="0DTE labels"):
        date_str = fp.stem.replace(f"{args.symbol}_", "")
        day = process_day(
            fp,
            symbol=args.symbol,
            horizons=horizons,
            commission_per_contract=args.commission_per_contract,
            max_spread_pct=args.max_spread_pct,
        )
        if day.empty:
            report["by_day"][date_str] = {"rows": 0}
            continue
        day.to_parquet(out_dir / fp.name, index=False, compression="zstd")
        report["days"] += 1
        report["rows"] += int(len(day))
        report["by_day"][date_str] = {
            "rows": int(len(day)),
            "minutes": int(day["timestamp"].nunique()),
            "buckets": sorted(int(x) for x in day["bucket_id"].dropna().unique()),
            "sides": day["side"].value_counts().to_dict(),
        }
        if args.single_file:
            frames.append(day)

    if args.single_file and frames:
        combined = pd.concat(frames, ignore_index=True).sort_values(["timestamp", "bucket_id", "ticker"])
        combined.to_parquet(
            Path(args.output_dir).expanduser() / f"{args.symbol}_{args.start_date}_{args.end_date}_labels.parquet",
            index=False,
            compression="zstd",
        )

    (Path(args.output_dir).expanduser() / f"{args.symbol}_{args.start_date}_{args.end_date}_report.json").write_text(
        json.dumps(report, indent=2, default=str),
        encoding="utf-8",
    )
    print(json.dumps({k: v for k, v in report.items() if k != "by_day"}, indent=2, default=str))


if __name__ == "__main__":
    main()

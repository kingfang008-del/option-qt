#!/usr/bin/env python3
"""Backfill 1s underlying bars for dates present in a locked map."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd
from polygon import RESTClient
from tqdm import tqdm

from preprocess.download.build_0dte_api_ladder_map import load_legacy_api_key
from preprocess.download.step2_polygon_second_sniper_v1 import load_stock_price_map

API_KEY = os.environ.get("POLYGON_API_KEY") or load_legacy_api_key()
STOCK_ROOT = Path("/mnt/s990/data/raw_1s/stocks")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--locked-map",
        default=str(Path.home() / "train_data/locked_targets_map_mag7_short_dte_api_ladder.parquet"),
    )
    p.add_argument("--symbols", default="NVDA,TSLA")
    p.add_argument("--start-date", default="2026-02-02")
    p.add_argument("--end-date", default="2026-07-09")
    p.add_argument("--report", default="stock_options/results/mag7_stock_1s_backfill_report.json")
    p.add_argument("--api-key", default=API_KEY)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.api_key:
        raise SystemExit("POLYGON_API_KEY missing")
    client = RESTClient(args.api_key)
    df = pd.read_parquet(args.locked_map)
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    summary = {}
    for symbol in symbols:
        sub = df[df["symbol"].astype(str).str.upper().eq(symbol)].copy()
        sub = sub[(sub["date_str"] >= args.start_date) & (sub["date_str"] <= args.end_date)]
        dates = sorted(sub["date_str"].astype(str).unique())
        existed, downloaded, empty = [], [], []
        for d in tqdm(dates, desc=f"{symbol}-stock1s"):
            path = STOCK_ROOT / symbol / f"{symbol}_{d}.parquet"
            if path.exists() and path.stat().st_size > 0:
                existed.append(d)
                continue
            m = load_stock_price_map(client, symbol, d)
            if path.exists() and path.stat().st_size > 0:
                downloaded.append(d)
            elif m:
                # fallback wrote nothing but map non-empty from 1min — still count as soft
                downloaded.append(d)
            else:
                empty.append(d)
        summary[symbol] = {
            "n_dates": len(dates),
            "already_existed": len(existed),
            "downloaded_or_filled": len(downloaded),
            "empty": len(empty),
            "empty_sample": empty[:20],
            "first": dates[0] if dates else None,
            "last": dates[-1] if dates else None,
        }
        print(json.dumps(summary[symbol], indent=2))
    out = Path(args.report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"config": {k: v for k, v in vars(args).items() if k != "api_key"}, "symbols": summary}, indent=2), encoding="utf-8")
    print(f"report -> {out}")


if __name__ == "__main__":
    main()

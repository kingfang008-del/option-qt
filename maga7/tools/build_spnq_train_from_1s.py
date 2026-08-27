#!/usr/bin/env python3
"""Rebuild Mag7 research 1m cache (spnq_train) from stock 1s fact source.

Source: ``/mnt/s990/data/raw_1s/stocks/{SYM}/{SYM}_{date}.parquet``
Sink:   ``~/train_data/spnq_train/{SYM}/{YYYY-MM}.parquet``
        (``load_stock_month_files`` layout)

Bars: left-labeled RTH 1m via ``aggregate_1s_to_1m`` (Mag7 clock contract).
This overwrites the research cache only — live/scanner still prefer raw 1s.

Example:
  PYTHONPATH=. python -m maga7.tools.build_spnq_train_from_1s \\
    --start-date 2026-01-01 --end-date 2026-07-23 --force
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from maga7.common.bar_agg import aggregate_1s_to_1m, load_stock_1s_day
from maga7.common.stock_1s import session_dates

NY = "America/New_York"
DEFAULT_1S = Path("/mnt/s990/data/raw_1s/stocks")
DEFAULT_OUT = Path.home() / "train_data" / "spnq_train"

# Mag7 spine + refs + common peers used by washout / hunt / router.
DEFAULT_SYMBOLS = (
    "NVDA",
    "TSLA",
    "AAPL",
    "AMZN",
    "META",
    "MSFT",
    "AMD",
    "GOOGL",
    "MU",
    "AVGO",
    "QQQ",
    "VIXY",
)


def _dates_for_symbol(root: Path, symbol: str, start: str, end: str) -> list[str]:
    want = set(session_dates(start, end))
    have: list[str] = []
    ddir = root / symbol
    if not ddir.is_dir():
        return []
    for p in sorted(ddir.glob(f"{symbol}_*.parquet")):
        d = p.stem.split("_", 1)[-1]
        if d in want:
            have.append(d)
    return have


def _agg_fast(raw: pd.DataFrame) -> pd.DataFrame:
    """Pandas left/closed-left 1m — same intent as aggregate_1s_to_1m, faster."""
    if raw is None or raw.empty:
        return pd.DataFrame()
    x = raw.copy()
    x["timestamp"] = pd.to_datetime(x["timestamp"])
    if getattr(x["timestamp"].dt, "tz", None) is None:
        x["timestamp"] = x["timestamp"].dt.tz_localize(NY)
    else:
        x["timestamp"] = x["timestamp"].dt.tz_convert(NY)
    t = x["timestamp"].dt.time
    x = x[(t >= pd.Timestamp("09:30").time()) & (t < pd.Timestamp("16:00").time())]
    if x.empty:
        return pd.DataFrame()
    for col in ("open", "high", "low"):
        if col not in x.columns:
            x[col] = x["close"]
    if "volume" not in x.columns:
        x["volume"] = 0.0
    x = x.sort_values("timestamp").set_index("timestamp")
    out = (
        x.resample("1min", label="left", closed="left")
        .agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
        )
        .dropna(subset=["close"])
        .reset_index()
    )
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stock-1s-root", default=str(DEFAULT_1S))
    ap.add_argument("--out-root", default=str(DEFAULT_OUT))
    ap.add_argument("--start-date", default="2026-01-01")
    ap.add_argument("--end-date", default="2026-07-23")
    ap.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    ap.add_argument(
        "--engine",
        choices=("fast", "strict"),
        default="fast",
        help="fast=pandas resample; strict=MinuteBarAggregator loop",
    )
    ap.add_argument("--force", action="store_true", help="overwrite existing month files")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    root = Path(args.stock_1s_root).expanduser()
    out_root = Path(args.out_root).expanduser()
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    if not root.is_dir():
        print(f"ERROR missing stock_1s_root: {root}", flush=True)
        return 2

    print(
        f"build spnq_train from 1s → {out_root}\n"
        f"  source={root}\n"
        f"  range={args.start_date}..{args.end_date} engine={args.engine} "
        f"symbols={symbols}",
        flush=True,
    )
    if not args.dry_run:
        out_root.mkdir(parents=True, exist_ok=True)

    meta: dict = {
        "stock_1s_root": str(root),
        "out_root": str(out_root),
        "start": args.start_date,
        "end": args.end_date,
        "engine": args.engine,
        "symbols": {},
    }

    for sym in symbols:
        dates = _dates_for_symbol(root, sym, args.start_date, args.end_date)
        print(f"[{sym}] 1s days={len(dates)}", flush=True)
        if not dates:
            meta["symbols"][sym] = {"n_days": 0, "months": []}
            continue
        by_month: dict[str, list[pd.DataFrame]] = defaultdict(list)
        n_ok = 0
        for i, date in enumerate(dates):
            raw = load_stock_1s_day(root, sym, date)
            if raw is None or raw.empty:
                continue
            if args.engine == "strict":
                bars = aggregate_1s_to_1m(raw, symbol=sym, rth_only=True)
            else:
                bars = _agg_fast(raw)
            if bars is None or bars.empty:
                continue
            bars = bars.copy()
            if "vwap" not in bars.columns:
                bars["vwap"] = bars["close"]
            if "transactions" not in bars.columns:
                bars["transactions"] = (bars["volume"] > 0).astype(int)
            ym = date[:7]
            by_month[ym].append(bars)
            n_ok += 1
            if (i + 1) % 40 == 0:
                print(f"  … {sym} {i+1}/{len(dates)}", flush=True)

        written: list[str] = []
        for ym, frames in sorted(by_month.items()):
            month_df = (
                pd.concat(frames, ignore_index=True)
                .sort_values("timestamp")
                .drop_duplicates("timestamp")
                .reset_index(drop=True)
            )
            dest = out_root / sym / f"{ym}.parquet"
            if dest.exists() and not args.force:
                print(f"  skip exists {dest}", flush=True)
                written.append(ym)
                continue
            if args.dry_run:
                print(f"  dry {dest} rows={len(month_df)}", flush=True)
                written.append(ym)
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            month_df.to_parquet(dest, index=False)
            print(f"  wrote {dest} rows={len(month_df)}", flush=True)
            written.append(ym)

        meta["symbols"][sym] = {"n_days": n_ok, "months": written}

    if not args.dry_run:
        (out_root / "BUILD_FROM_1S.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )
    print(json.dumps(meta, indent=2), flush=True)
    print(f"done → {out_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

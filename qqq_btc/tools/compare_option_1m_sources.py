#!/usr/bin/env python3
"""对比两路 1m 期权 parquet（Massive vs Databento 等）。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def compare_day(day: str, left_root: Path, right_root: Path, symbol: str = "QQQ") -> dict:
    pl = left_root / symbol / f"{symbol}_{day}.parquet"
    pr = right_root / symbol / f"{symbol}_{day}.parquet"
    out = {"day": day, "left_exists": pl.exists(), "right_exists": pr.exists()}
    if not (pl.exists() and pr.exists()):
        return out

    a = pd.read_parquet(pl)
    b = pd.read_parquet(pr)
    a["timestamp"] = pd.to_datetime(a["timestamp"])
    b["timestamp"] = pd.to_datetime(b["timestamp"])
    m = a.merge(b, on=["timestamp", "bucket_id", "ticker"], suffixes=("_l", "_r"))
    out.update(
        {
            "left_rows": int(len(a)),
            "right_rows": int(len(b)),
            "merge_rows": int(len(m)),
            "ticker_match": sorted(a["ticker"].unique()) == sorted(b["ticker"].unique()),
            "left_tickers": sorted(a["ticker"].unique()),
            "right_tickers": sorted(b["ticker"].unique()),
        }
    )
    for col in ("bid", "ask", "close"):
        lc, rc = f"{col}_l", f"{col}_r"
        if lc not in m.columns:
            continue
        diff = (m[lc] - m[rc]).abs()
        out[f"{col}_exact_rate"] = float((m[lc] == m[rc]).mean()) if len(m) else 0.0
        out[f"{col}_mean_abs_diff"] = float(diff.mean()) if len(m) else None
        out[f"{col}_max_abs_diff"] = float(diff.max()) if len(m) else None
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--left-root", required=True, help="例如 Massive 重下 1m")
    p.add_argument("--right-root", required=True, help="例如 options_databento 1m")
    p.add_argument("--left-label", default="left")
    p.add_argument("--right-label", default="right")
    p.add_argument("--month-prefix", default="2026-0")
    p.add_argument("--symbol", default="QQQ")
    p.add_argument("--output", required=True)
    args = p.parse_args()

    left_root = Path(args.left_root)
    right_root = Path(args.right_root)
    days = sorted(
        {f.stem.split("_", 1)[1] for f in left_root.glob(f"{args.symbol}/*.parquet")}
        | {f.stem.split("_", 1)[1] for f in right_root.glob(f"{args.symbol}/*.parquet")}
    )
    days = [d for d in days if d.startswith(args.month_prefix)]

    rows = [compare_day(d, left_root, right_root, args.symbol) for d in days]
    summary = {
        "left": args.left_label,
        "right": args.right_label,
        "days_total": len(days),
        "days_both_exist": sum(1 for r in rows if r.get("merge_rows")),
        "days_ticker_match": sum(1 for r in rows if r.get("ticker_match")),
        "avg_bid_exact_rate": float(
            pd.Series([r.get("bid_exact_rate") for r in rows if "bid_exact_rate" in r]).mean()
        ),
        "days": rows,
    }
    Path(args.output).write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in summary if k != "days"}, indent=2))


if __name__ == "__main__":
    main()

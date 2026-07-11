#!/usr/bin/env python3
"""对比 dte1 vs options_databento 1m 盘口差异（V4 replay 归因）。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def compare_day(day: str, db_root: Path, d1_root: Path, symbol: str = "QQQ") -> dict:
    p_db = db_root / symbol / f"{symbol}_{day}.parquet"
    p_d1 = d1_root / symbol / f"{symbol}_{day}.parquet"
    out = {"day": day, "db_exists": p_db.exists(), "d1_exists": p_d1.exists()}
    if not (p_db.exists() and p_d1.exists()):
        return out

    db = pd.read_parquet(p_db)
    d1 = pd.read_parquet(p_d1)
    db["timestamp"] = pd.to_datetime(db["timestamp"])
    d1["timestamp"] = pd.to_datetime(d1["timestamp"])

    m = db.merge(d1, on=["timestamp", "bucket_id", "ticker"], suffixes=("_db", "_d1"))
    out["merge_rows"] = int(len(m))
    out["db_rows"] = int(len(db))
    out["d1_rows"] = int(len(d1))

    for leg in ("bid", "ask"):
        a, b = f"{leg}_db", f"{leg}_d1"
        diff = (m[a] - m[b]).abs()
        out[f"{leg}_exact_rate"] = float((m[a] == m[b]).mean())
        out[f"{leg}_mean_abs_diff"] = float(diff.mean())
        out[f"{leg}_max_abs_diff"] = float(diff.max())

    # dte1 多出来的分钟（可能为 forward-fill）
    for b in range(4):
        ts_db = set(db[db["bucket_id"] == b]["timestamp"])
        ts_d1 = set(d1[d1["bucket_id"] == b]["timestamp"])
        out[f"bucket{b}_extra_d1_minutes"] = int(len(ts_d1 - ts_db))

    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--month", default="2026-06")
    p.add_argument("--db-root", default="/mnt/s990/data/raw_1m/options_databento")
    p.add_argument("--d1-root", default="/mnt/s990/data/raw_1m/dte1_options")
    p.add_argument("--output", default="/mnt/s990/data/v4_original_jul5/manifest/dte1_vs_databento_quote_diag.json")
    args = p.parse_args()

    db_root = Path(args.db_root)
    d1_root = Path(args.d1_root)
    days = sorted(
        {f.stem.split("_", 1)[1] for f in db_root.glob("QQQ/QQQ_*.parquet")}
        | {f.stem.split("_", 1)[1] for f in d1_root.glob("QQQ/QQQ_*.parquet")}
    )
    days = [d for d in days if d.startswith(args.month)]

    rows = [compare_day(d, db_root, d1_root) for d in days]
    summary = {
        "month": args.month,
        "days_compared": sum(1 for r in rows if r.get("merge_rows")),
        "days_db_only": [r["day"] for r in rows if r["db_exists"] and not r["d1_exists"]],
        "days_d1_only": [r["day"] for r in rows if r["d1_exists"] and not r["db_exists"]],
        "avg_bid_exact_rate": float(
            pd.Series([r.get("bid_exact_rate") for r in rows if "bid_exact_rate" in r]).mean()
        ),
        "total_extra_d1_minutes": int(
            sum(r.get(f"bucket{b}_extra_d1_minutes", 0) for r in rows for b in range(4))
        ),
        "root_cause": (
            "dte1 1m 在 databento 无 quote 的分钟仍保留 bucket 行（forward-fill/stale BBO），"
            "merge_asof 会用到陈旧 bid/ask，导致 replay 与 V4 原始 databento 路径偏离。"
        ),
        "days": rows,
    }
    Path(args.output).write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in summary if k != "days"}, indent=2))


if __name__ == "__main__":
    main()

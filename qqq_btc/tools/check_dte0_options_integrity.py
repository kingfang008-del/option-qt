#!/usr/bin/env python3
"""Check raw 1s dte0_options completeness and true 0DTE consistency."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

REQUIRED_COLS = {
    "ts", "timestamp", "ticker", "bucket_id", "underlying",
    "bid", "ask", "bid_size", "ask_size", "price", "strike",
}
EXPECTED_BUCKETS = {0, 1, 2, 3}
OCC_EXP_RE = re.compile(r"(\d{6})[CP]\d{8}$")


def occ_expiry_date(ticker: str) -> pd.Timestamp | None:
    m = OCC_EXP_RE.search(str(ticker).replace("O:", ""))
    if not m:
        return None
    code = m.group(1)
    return pd.Timestamp(f"20{code[:2]}-{code[2:4]}-{code[4:6]}")


def check_day(path: Path) -> dict:
    date_str = path.stem.split("_", 1)[-1]
    trade_date = pd.Timestamp(date_str).date()
    out = {
        "date": date_str,
        "path": str(path),
        "ok": True,
        "issues": [],
    }
    try:
        df = pd.read_parquet(path)
    except Exception as exc:
        out["ok"] = False
        out["issues"].append(f"read_error:{exc}")
        return out

    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        out["ok"] = False
        out["issues"].append(f"missing_cols:{sorted(missing)}")

    if df.empty:
        out["ok"] = False
        out["issues"].append("empty_file")
        return out

    out["rows"] = int(len(df))
    out["tickers"] = int(df["ticker"].nunique())
    buckets = set(pd.to_numeric(df["bucket_id"], errors="coerce").dropna().astype(int).unique())
    out["buckets"] = [int(x) for x in sorted(buckets)]
    missing_buckets = sorted(EXPECTED_BUCKETS - buckets)
    if missing_buckets:
        out["ok"] = False
        out["issues"].append(f"missing_buckets:{missing_buckets}")

    exp_dates = df["ticker"].map(occ_expiry_date)
    cal_dte = (pd.to_datetime(exp_dates) - pd.Timestamp(trade_date)).dt.days
    out["calendar_dte"] = int(cal_dte.dropna().iloc[0]) if cal_dte.notna().any() else None
    dte_counts = cal_dte.value_counts(dropna=False)
    out["dte_mix"] = {str(int(k) if pd.notna(k) else "nan"): int(v) for k, v in dte_counts.items()}

    if (cal_dte != 0).any():
        bad = cal_dte[cal_dte != 0]
        out["ok"] = False
        out["issues"].append(f"non_zero_dte_rows:{int(bad.count())}")
        out["sample_ticker"] = str(df.loc[bad.index[0], "ticker"])

    bid = pd.to_numeric(df["bid"], errors="coerce")
    ask = pd.to_numeric(df["ask"], errors="coerce")
    spread = (ask - bid) / ((bid + ask) / 2 + 1e-9)
    out["spread_pct_median"] = float(spread.median())
    out["spread_pct_p95"] = float(spread.quantile(0.95))

    if out["rows"] < 1000:
        out["issues"].append(f"low_rows:{out['rows']}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default="/mnt/s990/data/raw_1s/dte0_options/QQQ")
    parser.add_argument("--locked-map", default=str(Path.home() / "train_data/locked_targets_map_0dte.parquet"))
    parser.add_argument("--output", default="qqq_btc/results/dte0_options_integrity.json")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    files = sorted(raw_dir.glob("QQQ_*.parquet"))
    results = [check_day(p) for p in files]

    expected_days: set[str] = set()
    if Path(args.locked_map).exists():
        m = pd.read_parquet(args.locked_map)
        expected_days = set(m["date_str"].astype(str).unique())

    have_days = {r["date"] for r in results}
    missing_days = sorted(expected_days - have_days) if expected_days else []
    extra_days = sorted(have_days - expected_days) if expected_days else []

    bad_dte = [r for r in results if r.get("calendar_dte") not in (None, 0)]
    bad_files = [r for r in results if not r.get("ok", False)]
    low_rows = [r for r in results if r.get("rows", 0) < 1000]

    summary = {
        "raw_dir": str(raw_dir),
        "n_files": len(results),
        "expected_days": len(expected_days),
        "missing_days_count": len(missing_days),
        "missing_days_sample": missing_days[:20],
        "extra_days_count": len(extra_days),
        "bad_dte_days": len(bad_dte),
        "bad_dte_sample": bad_dte[:10],
        "bad_files_count": len(bad_files),
        "bad_files_sample": bad_files[:10],
        "low_rows_days": len(low_rows),
        "date_range": [results[0]["date"], results[-1]["date"]] if results else None,
        "median_rows": int(pd.Series([r.get("rows", 0) for r in results]).median()) if results else 0,
        "pass": len(bad_dte) == 0 and len(bad_files) == 0 and len(missing_days) == 0,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "summary": summary,
                "bad_days": bad_dte[:50],
                "bad_files": bad_files[:50],
                "missing_days": missing_days,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2))
    print(f"report -> {out_path}")


if __name__ == "__main__":
    main()

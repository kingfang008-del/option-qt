#!/usr/bin/env python3
"""Compare locked target map vs downloaded 1s options; emit map for missing contracts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def normalize_contract(symbol: str) -> str:
    return str(symbol).replace("O:", "")


def compare_locked_map(
    locked_map: pd.DataFrame,
    raw_dir: Path,
    symbol: str,
    min_rows: int,
) -> tuple[pd.DataFrame, dict]:
    required = {"date_str", "contract_symbol", "bucket_id", "symbol"}
    missing_cols = required - set(locked_map.columns)
    if missing_cols:
        raise ValueError(f"locked map missing columns: {sorted(missing_cols)}")

    sym_map = locked_map[locked_map["symbol"] == symbol].copy()
    if sym_map.empty:
        raise ValueError(f"no rows for symbol={symbol} in locked map")

    missing_rows: list[dict] = []
    file_cache: dict[str, pd.DataFrame | None] = {}

    for _, row in sym_map.iterrows():
        date_str = str(row["date_str"])
        bucket_id = int(row["bucket_id"])
        expected = normalize_contract(row["contract_symbol"])
        fp = raw_dir / f"{symbol}_{date_str}.parquet"

        if not fp.exists():
            missing_rows.append(
                {
                    **row.to_dict(),
                    "missing_reason": "missing_file",
                    "existing_ticker": None,
                    "existing_rows": 0,
                }
            )
            continue

        if date_str not in file_cache:
            try:
                file_cache[date_str] = pd.read_parquet(fp, columns=["ticker", "bucket_id"])
            except Exception as exc:
                file_cache[date_str] = None
                missing_rows.append(
                    {
                        **row.to_dict(),
                        "missing_reason": f"read_error:{exc}",
                        "existing_ticker": None,
                        "existing_rows": 0,
                    }
                )
                continue

        df = file_cache[date_str]
        if df is None or df.empty:
            missing_rows.append(
                {
                    **row.to_dict(),
                    "missing_reason": "empty_file",
                    "existing_ticker": None,
                    "existing_rows": 0,
                }
            )
            continue

        sub = df[df["bucket_id"].astype(int) == bucket_id]
        if sub.empty:
            missing_rows.append(
                {
                    **row.to_dict(),
                    "missing_reason": "missing_bucket",
                    "existing_ticker": None,
                    "existing_rows": 0,
                }
            )
            continue

        got_ticker = str(sub["ticker"].iloc[0])
        got_rows = int(len(sub))
        if got_ticker != expected:
            missing_rows.append(
                {
                    **row.to_dict(),
                    "missing_reason": "wrong_ticker",
                    "existing_ticker": got_ticker,
                    "existing_rows": got_rows,
                }
            )
        elif got_rows < min_rows:
            missing_rows.append(
                {
                    **row.to_dict(),
                    "missing_reason": f"low_rows:{got_rows}",
                    "existing_ticker": got_ticker,
                    "existing_rows": got_rows,
                }
            )

    miss_df = pd.DataFrame(missing_rows)
    if not miss_df.empty:
        download_map = miss_df.drop(columns=["missing_reason", "existing_ticker", "existing_rows"], errors="ignore")
        # step2 expects same schema as locked map
        download_map = download_map[sym_map.columns.tolist()]
    else:
        download_map = sym_map.iloc[0:0].copy()

    have_days = {p.stem.split("_", 1)[-1] for p in raw_dir.glob(f"{symbol}_*.parquet")}
    expected_days = set(sym_map["date_str"].astype(str).unique())
    summary = {
        "symbol": symbol,
        "raw_dir": str(raw_dir),
        "locked_rows": int(len(sym_map)),
        "expected_days": len(expected_days),
        "downloaded_days": len(have_days),
        "missing_file_days": sorted(expected_days - have_days),
        "missing_contract_rows": int(len(miss_df)),
        "missing_unique_days": int(miss_df["date_str"].nunique()) if not miss_df.empty else 0,
        "missing_unique_contracts": int(miss_df["contract_symbol"].nunique()) if not miss_df.empty else 0,
        "by_reason": miss_df["missing_reason"].value_counts().to_dict() if not miss_df.empty else {},
        "by_bucket": miss_df.groupby("bucket_id").size().astype(int).to_dict() if not miss_df.empty else {},
        "download_map_rows": int(len(download_map)),
    }
    return download_map, summary, miss_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Build target map for missing 1s option downloads")
    parser.add_argument(
        "--locked-map",
        default=str(Path.home() / "train_data/locked_targets_map_0dte.parquet"),
    )
    parser.add_argument("--raw-dir", default="/mnt/s990/data/raw_1s/dte0_options/QQQ")
    parser.add_argument("--symbol", default="QQQ")
    parser.add_argument(
        "--output-map",
        default=str(Path.home() / "train_data/locked_targets_map_0dte_missing.parquet"),
    )
    parser.add_argument(
        "--output-report",
        default="qqq_btc/results/dte0_options_missing_diff.json",
    )
    parser.add_argument("--min-rows", type=int, default=100, help="treat bucket rows below this as missing")
    args = parser.parse_args()

    locked_map = pd.read_parquet(args.locked_map)
    download_map, summary, miss_df = compare_locked_map(
        locked_map=locked_map,
        raw_dir=Path(args.raw_dir),
        symbol=args.symbol,
        min_rows=args.min_rows,
    )

    out_map = Path(args.output_map)
    out_map.parent.mkdir(parents=True, exist_ok=True)
    if download_map.empty:
        if out_map.exists():
            out_map.unlink()
    else:
        download_map.to_parquet(out_map, index=False)

    report = {
        "summary": summary,
        "missing_rows_sample": miss_df.head(50).to_dict(orient="records") if not miss_df.empty else [],
        "output_map": str(out_map) if not download_map.empty else None,
    }
    out_report = Path(args.output_report)
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_report.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"download map -> {out_map} ({len(download_map)} rows)")
    print(f"report -> {out_report}")


if __name__ == "__main__":
    main()

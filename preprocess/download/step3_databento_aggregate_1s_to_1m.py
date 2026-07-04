"""
Databento 1s 期权 quote → 1m 聚合

输入: /mnt/s990/data/raw_1s/options_databento/{SYMBOL}/{SYMBOL}_{date}.parquet
输出: /mnt/s990/data/raw_1m/options_databento/{SYMBOL}/{SYMBOL}_{date}.parquet

用法:
  python step3_databento_aggregate_1s_to_1m.py
  python step3_databento_aggregate_1s_to_1m.py --symbol QQQ --date-from 2025-01-01 --force
"""

from __future__ import annotations

import argparse
import concurrent.futures
import glob
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

INPUT_DIR = "/mnt/s990/data/raw_1s/options_databento"
OUTPUT_DIR = "/mnt/s990/data/raw_1m/options_databento"
MAX_WORKERS = 16

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("Databento_1s_to_1m")


def aggregate_one_day(file_path: str, output_dir: str, force: bool) -> str:
    try:
        parts = Path(file_path).parts
        symbol = parts[-2]
        filename = parts[-1]
        date_str = filename.replace(f"{symbol}_", "").replace(".parquet", "")

        out_dir = os.path.join(output_dir, symbol)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, filename)

        if os.path.exists(out_path) and not force:
            return f"⏩ {symbol} {date_str} exists"

        df = pd.read_parquet(file_path)
        if df.empty:
            return f"⚠️ {symbol} {date_str}: empty input"

        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(
                "America/New_York"
            )
        elif "ts" in df.columns:
            df["timestamp"] = pd.to_datetime(df["ts"], unit="s", utc=True).dt.tz_convert(
                "America/New_York"
            )
        else:
            return f"❌ {symbol} {date_str}: no timestamp/ts column"

        px_col = "price" if "price" in df.columns else "mid_price"
        if px_col not in df.columns:
            df["mid_price"] = (df["bid"] + df["ask"]) / 2.0
            px_col = "mid_price"

        df = df.sort_values(["ticker", "timestamp"])
        df = df.set_index("timestamp")

        agg_spec: dict = {
            px_col: ["first", "max", "min", "last"],
            "bid": "last",
            "ask": "last",
            "bid_size": "last",
            "ask_size": "last",
            "bucket_id": "last",
            "strike": "last",
            "underlying": "last",
        }
        for col in ("tag", "source_schema"):
            if col in df.columns:
                agg_spec[col] = "last"

        agg_df = df.groupby(
            ["ticker", pd.Grouper(freq="1min", label="right", closed="left")],
            sort=False,
        ).agg(agg_spec)

        agg_df.columns = [
            "open",
            "high",
            "low",
            "close",
            "bid",
            "ask",
            "bid_size",
            "ask_size",
            "bucket_id",
            "strike",
            "underlying",
        ] + [c for c in ("tag", "source_schema") if c in df.columns]

        agg_df = agg_df.dropna(subset=["close"]).reset_index()

        if agg_df.empty:
            return f"⚠️ {symbol} {date_str}: no rows after resample"

        agg_df["volume"] = agg_df["bid_size"] + agg_df["ask_size"]
        agg_df["spread_pct"] = (agg_df["ask"] - agg_df["bid"]) / (agg_df["close"] + 1e-6)
        agg_df["volume_imbalance"] = (
            agg_df["bid_size"] - agg_df["ask_size"]
        ) / (agg_df["volume"] + 1e-6)
        agg_df["ts"] = agg_df["timestamp"].astype("int64") / 1e9

        out_cols = [
            "ts",
            "timestamp",
            "ticker",
            "tag",
            "bucket_id",
            "underlying",
            "strike",
            "open",
            "high",
            "low",
            "close",
            "bid",
            "ask",
            "bid_size",
            "ask_size",
            "volume",
            "spread_pct",
            "volume_imbalance",
            "source_schema",
        ]
        final_df = agg_df[[c for c in out_cols if c in agg_df.columns]]

        final_df.to_parquet(out_path, engine="pyarrow", index=False, compression="zstd")
        n_tickers = final_df["ticker"].nunique()
        return f"🎯 {symbol} {date_str}: {len(final_df)} bars, {n_tickers} contracts"

    except Exception as exc:
        return f"❌ {file_path}: {exc}"


def collect_files(
    input_dir: str,
    symbol: str | None,
    date_from: str | None,
    date_to: str | None,
) -> list[str]:
    pattern = os.path.join(input_dir, "*", "*.parquet")
    files = sorted(glob.glob(pattern))
    if symbol:
        sym = symbol.upper()
        files = [f for f in files if f"/{sym}/" in f.replace("\\", "/")]
    if date_from or date_to:
        filtered = []
        for f in files:
            name = Path(f).stem
            date_str = name.split("_", 1)[-1]
            if date_from and date_str < date_from:
                continue
            if date_to and date_str > date_to:
                continue
            filtered.append(f)
        files = filtered
    return files


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate Databento 1s options to 1m")
    p.add_argument("--input-dir", default=INPUT_DIR)
    p.add_argument("--output-dir", default=OUTPUT_DIR)
    p.add_argument("--symbol", default=None, help="e.g. QQQ")
    p.add_argument("--date-from", default=None, help="YYYY-MM-DD")
    p.add_argument("--date-to", default=None, help="YYYY-MM-DD")
    p.add_argument("--max-workers", type=int, default=MAX_WORKERS)
    p.add_argument("--force", action="store_true", help="overwrite existing 1m files")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not os.path.isdir(args.input_dir):
        logger.error("输入目录不存在: %s", args.input_dir)
        return

    files = collect_files(args.input_dir, args.symbol, args.date_from, args.date_to)
    if not files:
        logger.warning("未找到待处理文件: %s", args.input_dir)
        return

    logger.info(
        "输入: %s | 输出: %s | 文件数: %d | workers: %d",
        args.input_dir,
        args.output_dir,
        len(files),
        args.max_workers,
    )

    ok = skip = warn = err = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.max_workers) as pool:
        futures = [
            pool.submit(aggregate_one_day, fp, args.output_dir, args.force)
            for fp in files
        ]
        for msg in tqdm(
            (f.result() for f in concurrent.futures.as_completed(futures)),
            total=len(futures),
            desc="1s→1m",
        ):
            if msg.startswith("🎯"):
                ok += 1
            elif msg.startswith("⏩"):
                skip += 1
            elif msg.startswith("⚠️"):
                warn += 1
                logger.warning(msg)
            else:
                err += 1
                logger.error(msg)

    logger.info("完成: success=%d skip=%d warn=%d error=%d", ok, skip, warn, err)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
QQQ 0DTE/1DTE 合约锁定雷达 — 生成 locked_targets_map_0dte.parquet

用法:
  cd New_Pro/preprocess
  OPTION_ANCHOR_PROFILE=qqq_0dte python step1_build_target_map_0dte.py \\
      --start-date 2022-09-01 --end-date 2026-03-01
"""
from __future__ import annotations

import argparse
import concurrent.futures
import logging
import multiprocessing
import os
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from anchor_contract_utils import get_daily_locked_contracts, load_anchor_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

ANCHOR_CFG = load_anchor_config()


def process_single_file(args):
    file_path, sym, cfg = args
    if "high_features" in file_path.name:
        return None, None

    try:
        df = pd.read_parquet(file_path)
        if df.empty:
            return None, None

        rename_map = {
            "expiration_date": "expiration",
            "strike_price": "strike",
            "ticker": "contract_symbol",
        }
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

        for col in ["timestamp", "expiration"]:
            df[col] = pd.to_datetime(df[col], errors="coerce")
            if df[col].dt.tz is None:
                df[col] = df[col].dt.tz_localize("UTC").dt.tz_convert("America/New_York")
            else:
                df[col] = df[col].dt.tz_convert("America/New_York")

        df["dte"] = (
            df["expiration"].dt.normalize() - df["timestamp"].dt.normalize()
        ).dt.days.fillna(-1).astype(int)

        locked_df = get_daily_locked_contracts(df, cfg)
        if locked_df is not None and not locked_df.empty:
            locked_df["symbol"] = sym
            return locked_df, None
    except Exception as e:
        return None, f"🚨 [报错] 处理文件 {file_path.name} 时发生错误: {e}"

    return None, None


def main():
    parser = argparse.ArgumentParser(description="Build QQQ 0DTE locked target map")
    parser.add_argument("--start-date", type=str, default=None)
    parser.add_argument("--end-date", type=str, default=None)
    parser.add_argument("--config", type=str, default=None, help="anchor_qqq_0dte.json path")
    args = parser.parse_args()

    cfg = load_anchor_config(Path(args.config)) if args.config else ANCHOR_CFG
    paths = cfg.get("_paths_resolved") or {}
    raw_dir = paths.get("raw_iv_dir") or Path.home() / "train_data/nq_options_day_iv"
    output_file = paths.get("locked_targets_output") or Path.home() / "train_data/locked_targets_map_0dte.parquet"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    symbols = cfg.get("symbols") or ["QQQ"]
    try:
        sys.path.insert(0, str(_SCRIPT_DIR.parent / "baseline_qqq"))
        from config import TARGET_SYMBOLS

        symbols = [s for s in symbols if s in TARGET_SYMBOLS] or list(TARGET_SYMBOLS)
    except ImportError:
        pass

    logger.info(
        f"📡 QQQ 0DTE 雷达 | profile={cfg.get('profile')} | "
        f"DTE prefer={cfg.get('front_prefer_dte')} | symbols={symbols}"
    )
    logger.info(f"   输入: {raw_dir}")
    logger.info(f"   输出: {output_file}")

    tasks = []
    for sym in symbols:
        src_dir = raw_dir / sym
        if not src_dir.exists():
            logger.warning(f"跳过 {sym}: 目录不存在 {src_dir}")
            continue
        for p in src_dir.glob(f"{sym}_*.parquet"):
            try:
                file_date_str = p.stem.split("_")[-1]
                if args.start_date and file_date_str < args.start_date:
                    continue
                if args.end_date and file_date_str > args.end_date:
                    continue
            except Exception:
                pass
            tasks.append((p, sym, cfg))

    if not tasks:
        logger.error("❌ 未找到任何待处理的 Parquet 文件。")
        return

    workers = max(1, multiprocessing.cpu_count() - 2)
    logger.info(f"🚀 任务数 {len(tasks)}，并发 {workers}")

    all_targets = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        for result_df, error_msg in tqdm(
            executor.map(process_single_file, tasks),
            total=len(tasks),
            desc="⚡ 0DTE 锁定",
        ):
            if error_msg:
                tqdm.write(error_msg)
            if result_df is not None:
                all_targets.append(result_df)

    if all_targets:
        final_map = pd.concat(all_targets, ignore_index=True)
        final_map.to_parquet(output_file, compression="zstd", index=False)
        n_days = final_map["date_str"].nunique()
        logger.info(
            f"🎉 完成 | {len(final_map):,} 条锁定记录 | {n_days} 个交易日 | → {output_file}"
        )
        dte_dist = final_map.groupby("date_str")["front_dte"].first().value_counts().sort_index()
        logger.info(f"   Front DTE 分布:\n{dte_dist.to_string()}")
    else:
        logger.error("❌ 未找到任何符合条件的合约。")


if __name__ == "__main__":
    main()

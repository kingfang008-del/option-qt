#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
合约锁定雷达 — 按 anchor profile 生成 locked_targets_map.parquet。

默认 profile=qqq_0dte (0/1/2 DTE, 4-bucket)。Legacy 9DTE 用 --profile legacy_9dte。

用法:
  cd preprocess/download

  # QQQ 0DTE/1DTE (推荐,与 qqq_btc 对齐)
  python step1_build_target_map.py --profile qqq_0dte \\
      --start-date 2022-09-01 --end-date 2026-03-01

  # 旧 ~9DTE + 次月 6-bucket
  python step1_build_target_map.py --profile legacy_9dte \\
      --start-date 2022-03-01 --end-date 2025-06-30

  # 自定义 JSON + 多标的
  python step1_build_target_map.py --config ../CONFIG/anchor_qqq_0dte.json \\
      --symbols QQQ,SPY --raw-dir ~/train_data/nq_options_day_iv \\
      --output ~/train_data/locked_targets_map_0dte.parquet
"""
from __future__ import annotations

import argparse
import concurrent.futures
import logging
import multiprocessing
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from anchor_utils import (
    BUILTIN_PROFILES,
    get_daily_locked_contracts,
    resolve_anchor_config,
    resolve_paths,
    resolve_symbols,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def process_single_file(args):
    """Worker: 单日 parquet → 锁定 bucket 合约列表。"""
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
    parser = argparse.ArgumentParser(
        description="Build locked target map (anchor profile: qqq_0dte / legacy_9dte / custom JSON)",
    )
    parser.add_argument(
        "--profile",
        choices=list(BUILTIN_PROFILES.keys()),
        default=None,
        help=f"内置锚点 profile (默认 qqq_0dte)。可选: {', '.join(BUILTIN_PROFILES)}",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="自定义 anchor JSON 路径(覆盖 --profile)",
    )
    parser.add_argument("--start-date", type=str, default=None, help="YYYY-MM-DD")
    parser.add_argument("--end-date", type=str, default=None, help="YYYY-MM-DD")
    parser.add_argument("--raw-dir", type=str, default=None, help="nq_options_day_iv 根目录")
    parser.add_argument("--output", type=str, default=None, help="输出 parquet 路径")
    parser.add_argument("--symbols", type=str, default=None, help="逗号分隔,如 QQQ,SPY")
    args = parser.parse_args()

    cfg = resolve_anchor_config(profile=args.profile, config_path=args.config)
    raw_dir, output_file = resolve_paths(cfg, args.raw_dir, args.output)
    symbols = resolve_symbols(cfg, args.symbols)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    n_buckets = 6 if cfg.get("use_next_buckets") else 4
    logger.info(
        "📡 合约锁定雷达 | profile=%s | prefer_dte=%s | allowed=%s | buckets=%d | symbols=%s",
        cfg.get("profile"),
        cfg.get("front_prefer_dte"),
        cfg.get("front_allowed_dte"),
        n_buckets,
        symbols,
    )
    logger.info("   输入: %s", raw_dir)
    logger.info("   输出: %s", output_file)

    tasks = []
    for sym in symbols:
        src_dir = raw_dir / sym
        if not src_dir.exists():
            logger.warning("跳过 %s: 目录不存在 %s", sym, src_dir)
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
    logger.info("🚀 任务数 %d，并发 %d", len(tasks), workers)

    all_targets = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        for result_df, error_msg in tqdm(
            executor.map(process_single_file, tasks),
            total=len(tasks),
            desc=f"⚡ {cfg.get('profile', 'anchor')}",
        ):
            if error_msg:
                tqdm.write(error_msg)
            if result_df is not None:
                all_targets.append(result_df)

    if not all_targets:
        logger.error("❌ 未找到任何符合条件的合约。")
        return

    final_map = pd.concat(all_targets, ignore_index=True)
    final_map.to_parquet(output_file, compression="zstd", index=False)
    n_days = final_map["date_str"].nunique()
    logger.info(
        "🎉 完成 | %s 条锁定 | %d 交易日 | → %s",
        f"{len(final_map):,}",
        n_days,
        output_file,
    )
    if "front_dte" in final_map.columns:
        dte_dist = final_map.groupby("date_str")["front_dte"].first().value_counts().sort_index()
        logger.info("   Front DTE 分布:\n%s", dte_dist.to_string())


if __name__ == "__main__":
    main()

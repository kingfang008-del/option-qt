#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
合约锁定雷达 — 按 anchor profile 生成 locked_targets_map.parquet。

默认 profile=qqq_0dte (strict 0 DTE only;无 0DTE 则跳过当天)。
1DTE 管线用 --profile qqq_1dte。Legacy 9DTE 用 --profile legacy_9dte。

用法:
  cd preprocess/download

  # QQQ strict 0DTE (V4 主路径)
  python step1_build_target_map.py --profile qqq_0dte \\
      --start-date 2022-09-01 --end-date 2026-03-01

  # QQQ strict 1DTE (独立 locked map / 独立训练链)
  python step1_build_target_map.py --profile qqq_1dte \\
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
from dte_utils import compute_dte_series

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

NY = "America/New_York"


def _normalize_contract_symbol(value: object) -> str:
    """Normalize OCC symbols for existence checks while preserving output format."""
    return str(value).replace("O:", "")


def _active_bucket_ids(cfg: dict) -> set[int]:
    n_buckets = 6 if cfg.get("use_next_buckets") else 4
    return set(range(n_buckets))


def _validate_locked_contracts(
    locked_df: pd.DataFrame,
    source_df: pd.DataFrame,
    cfg: dict,
) -> pd.DataFrame:
    """
    Keep only locked rows that are backed by real same-day source contracts.

    For 6-bucket profiles, a day is usable only if all active buckets are present.
    This prevents step2 from receiving synthetic/non-tradable contracts or partial
    daily maps that later show up as missing quote buckets.
    """
    if locked_df.empty:
        return locked_df

    validated = locked_df.copy()
    if bool(cfg.get("validate_contract_exists", True)):
        source = source_df.copy()
        if "date_str" not in source.columns:
            source["date_str"] = source["timestamp"].dt.date.astype(str)
        available_by_day = (
            source.groupby("date_str")["contract_symbol"]
            .apply(lambda s: {_normalize_contract_symbol(x) for x in s.dropna().unique()})
            .to_dict()
        )

        exists_mask = []
        for _, row in validated.iterrows():
            day_contracts = available_by_day.get(str(row["date_str"]), set())
            exists_mask.append(_normalize_contract_symbol(row["contract_symbol"]) in day_contracts)
        validated = validated.loc[exists_mask].copy()

    require_complete = bool(cfg.get("require_complete_buckets", cfg.get("use_next_buckets", False)))
    if require_complete and not validated.empty:
        expected = _active_bucket_ids(cfg)
        keep_parts = []
        for _, day_group in validated.groupby("date_str"):
            day_buckets = set(day_group["bucket_id"].astype(int))
            if day_buckets != expected:
                continue
            # One row per bucket. If duplicates ever appear, keep the first stable choice.
            day_group = day_group.sort_values("bucket_id").drop_duplicates("bucket_id", keep="first")
            if set(day_group["bucket_id"].astype(int)) == expected:
                keep_parts.append(day_group)
        if not keep_parts:
            return validated.iloc[0:0].copy()
        validated = pd.concat(keep_parts, ignore_index=True)

    return validated


def _to_ny_quote_time(series: pd.Series) -> pd.Series:
    """Quote timestamp: naive → UTC, tz-aware → NY."""
    ts = pd.to_datetime(series, errors="coerce")
    if ts.dt.tz is None:
        return ts.dt.tz_localize("UTC").dt.tz_convert(NY)
    return ts.dt.tz_convert(NY)


def _to_ny_expiration(series: pd.Series) -> pd.Series:
    """Expiration calendar date: naive midnight must NOT be treated as UTC."""
    exp = pd.to_datetime(series, errors="coerce")
    if exp.dt.tz is None:
        return exp.dt.tz_localize(NY, ambiguous="infer")
    return exp.dt.tz_convert(NY)


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

        df["timestamp"] = _to_ny_quote_time(df["timestamp"])
        if "expiration" in df.columns:
            df["expiration"] = _to_ny_expiration(df["expiration"])

        use_trading_dte = bool(cfg.get("use_trading_dte", False))
        df["dte"] = compute_dte_series(
            df["timestamp"],
            df["expiration"],
            use_trading_dte=use_trading_dte,
        )

        locked_df = get_daily_locked_contracts(df, cfg)
        if locked_df is not None and not locked_df.empty:
            locked_df = _validate_locked_contracts(locked_df, df, cfg)
        if locked_df is not None and not locked_df.empty:
            locked_df["symbol"] = sym
            return locked_df, None
    except Exception as e:
        return None, f"🚨 [报错] 处理文件 {file_path.name} 时发生错误: {e}"

    return None, None


def main():
    parser = argparse.ArgumentParser(
        description="Build locked target map (profile: qqq_0dte / qqq_1dte / legacy_9dte / custom JSON)",
    )
    parser.add_argument(
        "--profile",
        default=None,
        help=(
            "锚点 profile 名，加载 CONFIG/anchor_{profile}.json。"
            f"内置: {', '.join(BUILTIN_PROFILES)}"
        ),
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

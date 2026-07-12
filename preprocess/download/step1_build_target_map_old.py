#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
历史锁约雷达（V4 / standard_old_v2 血缘）— main 首版 step1 逻辑冻结版。

来源:
  git 64b0532 (main 首次引入) preprocess/download/step1_build_target_map.py
  + 同 commit qqq_btc/qqq/anchor.get_daily_locked_contracts

与现网 step1_build_target_map.py 的关键差异（刻意保留）:
  1) DTE: expiration naive 先按 UTC 再转 NY（会偏一天）→ prefer=0 时大量锁到真实 trading-1DTE
  2) 选约快照: 用全天所有 bar 的供应商原始 |delta|，无开盘窗、无重算 delta、无 put-call 合成
  3) 配置: front_allowed=[0,1,2] prefer=0（0/1/2 自动退档）

用法:
  cd preprocess/download

  # 复现 standard_old_v2 锁约口径（默认）
  python step1_build_target_map_old.py \\
      --start-date 2023-03-28 --end-date 2026-06-30

  # 明确用 trading-1DTE（修掉 UTC bug，但保留全天原始 delta 选约）
  python step1_build_target_map_old.py --dte-mode trading \\
      --config ../CONFIG/anchor_qqq_1dte_4bucket.json \\
      --output ~/train_data/locked_targets_map_old_style_trading_1dte.parquet

说明:
  「更好」指对齐 V4 / old_v2 训练分布与 IC，不是指实盘无前视。
  现网 step1 对实盘/无前视更正确；对离线复现 V4 血缘则应使用本脚本。
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import multiprocessing
import os
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import pandas as pd
from tqdm import tqdm

_SCRIPT_DIR = Path(__file__).resolve().parent
_PREPROCESS_ROOT = _SCRIPT_DIR.parent
_REPO_ROOT = _PREPROCESS_ROOT.parent
_CONFIG_DIR = _PREPROCESS_ROOT / "CONFIG"

if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

NY = "America/New_York"

_FRONT_4_BUCKET_TARGETS = [
    (0, True, False, 0.50),
    (1, True, False, 0.25),
    (2, True, True, 0.50),
    (3, True, True, 0.25),
]

_LEGACY_6_BUCKET_TARGETS = [
    (0, True, False, 0.50),
    (1, True, False, 0.25),
    (2, True, True, 0.50),
    (3, True, True, 0.25),
    (4, False, False, 0.50),
    (5, False, True, 0.50),
]

DEFAULT_CONFIG = _CONFIG_DIR / "anchor_qqq_old_v2.json"


def _expand_path(raw: str) -> Path:
    return Path(os.path.expanduser(raw)).expanduser()


def load_anchor_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["_config_path"] = str(path)
    paths = cfg.get("paths") or {}
    cfg["_paths_resolved"] = {k: _expand_path(v) for k, v in paths.items()}
    return cfg


def select_front_dte(available_dtes: Sequence[int], cfg: dict) -> Optional[int]:
    """main 首版: allowed 内 prefer，否则 fallback 到 >= min 的最小 dte。"""
    allowed = set(cfg.get("front_allowed_dte") or [0, 1, 2])
    prefer = int(cfg.get("front_prefer_dte", 0))
    dte_min = int(cfg.get("front_min_dte", 0))
    dte_max = int(cfg.get("front_max_dte", 2))

    candidates = sorted(
        {int(d) for d in available_dtes if dte_min <= int(d) <= dte_max and int(d) in allowed}
    )
    if not candidates:
        fallbacks = sorted({int(d) for d in available_dtes if int(d) >= dte_min})
        if not fallbacks:
            return None
        return fallbacks[0]

    if prefer in candidates:
        return prefer
    return min(candidates, key=lambda d: (abs(d - prefer), d))


def bucket_targets(cfg: dict) -> List[Tuple[int, bool, bool, float]]:
    if cfg.get("use_next_buckets", False):
        return list(_LEGACY_6_BUCKET_TARGETS)
    return list(_FRONT_4_BUCKET_TARGETS)


def get_daily_locked_contracts_old(df: pd.DataFrame, cfg: dict) -> Optional[pd.DataFrame]:
    """
    main 首版锁约（冻结）:
    - 全天所有分钟行参与 |delta| 距离排序（非开盘窗、非每合约最后一笔）
    - 使用供应商原始 delta.abs()，不重算
    - 无 put-call parity 合成
    """
    work = df.copy()
    work["date_str"] = work["timestamp"].dt.date.astype(str)
    work["abs_delta"] = work["delta"].abs()

    dte_min = int(cfg.get("front_min_dte", 0))
    dte_max = max(int(cfg.get("front_max_dte", 2)), 90 if cfg.get("use_next_buckets") else 2)
    candidates = work[(work["dte"] >= dte_min) & (work["dte"] <= dte_max)].copy()
    if candidates.empty:
        return None

    locked_map = []
    targets = bucket_targets(cfg)
    delta_tol = float(cfg.get("delta_tolerance", 0.15))

    for date_val, daily_group in candidates.groupby("date_str"):
        available_dtes = daily_group["dte"].unique()
        selected_front_dte = select_front_dte(available_dtes, cfg)
        if selected_front_dte is None:
            continue

        selected_next_dte = selected_front_dte
        if cfg.get("use_next_buckets", False):
            next_target = selected_front_dte + 28
            min_next = selected_front_dte + 20
            max_next = selected_front_dte + 50
            next_options = [d for d in available_dtes if min_next <= d <= max_next]
            if next_options:
                selected_next_dte = min(next_options, key=lambda x: abs(x - next_target))
            else:
                fallbacks = [d for d in available_dtes if d > selected_front_dte + 15]
                if fallbacks:
                    selected_next_dte = min(fallbacks)

        for b_id, is_front, is_call, target_delta in targets:
            target_dte = selected_front_dte if is_front else selected_next_dte
            type_str = "Call" if is_call else "Put"
            mask = (daily_group["dte"] == target_dte) & (
                daily_group["contract_type"].astype(str).str.upper().str.startswith(type_str[0])
            )
            subset = daily_group[mask].copy()
            if subset.empty:
                continue

            subset["delta_dist"] = (subset["abs_delta"] - target_delta).abs()
            delta_candidates = subset[subset["delta_dist"] < delta_tol]
            if delta_candidates.empty:
                best_ticker = subset.sort_values("delta_dist").iloc[0]["contract_symbol"]
            else:
                best_ticker = delta_candidates.sort_values("delta_dist").iloc[0]["contract_symbol"]

            locked_map.append(
                {
                    "date_str": date_val,
                    "contract_symbol": best_ticker,
                    "bucket_id": b_id,
                    "front_dte": int(selected_front_dte),
                }
            )

    if not locked_map:
        return None
    return pd.DataFrame(locked_map)


def _apply_dte(df: pd.DataFrame, dte_mode: str) -> pd.DataFrame:
    """
    dte_mode:
      legacy — 首版原样：naive expiration 当 UTC（复现 old_v2 的关键 bug）
      calendar — naive expiration 当 NY 本地日（修 bug 后的日历 DTE）
      trading — 交易日 DTE（intentional 1DTE 口径）
    """
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    out["expiration"] = pd.to_datetime(out["expiration"], errors="coerce")

    if dte_mode == "legacy":
        for col in ["timestamp", "expiration"]:
            if out[col].dt.tz is None:
                out[col] = out[col].dt.tz_localize("UTC").dt.tz_convert(NY)
            else:
                out[col] = out[col].dt.tz_convert(NY)
        out["dte"] = (
            out["expiration"].dt.normalize() - out["timestamp"].dt.normalize()
        ).dt.days.fillna(-1).astype(int)
        return out

    # timestamp: naive → UTC → NY（报价时间戳惯例）
    if out["timestamp"].dt.tz is None:
        out["timestamp"] = out["timestamp"].dt.tz_localize("UTC").dt.tz_convert(NY)
    else:
        out["timestamp"] = out["timestamp"].dt.tz_convert(NY)

    # expiration: naive → NY（日历日，不把午夜当 UTC）
    if out["expiration"].dt.tz is None:
        out["expiration"] = out["expiration"].dt.tz_localize(NY, ambiguous="infer")
    else:
        out["expiration"] = out["expiration"].dt.tz_convert(NY)

    if dte_mode == "calendar":
        out["dte"] = (
            out["expiration"].dt.normalize() - out["timestamp"].dt.normalize()
        ).dt.days.fillna(-1).astype(int)
        return out

    if dte_mode == "trading":
        from dte_utils import compute_dte_series

        out["dte"] = compute_dte_series(
            out["timestamp"], out["expiration"], use_trading_dte=True
        )
        return out

    raise ValueError(f"unknown dte_mode={dte_mode!r}; use legacy|calendar|trading")


def process_single_file(args):
    """Worker: 单日 parquet → 锁定 bucket 合约列表。"""
    file_path, sym, cfg, dte_mode = args
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
        df = _apply_dte(df, dte_mode)

        locked_df = get_daily_locked_contracts_old(df, cfg)
        if locked_df is not None and not locked_df.empty:
            locked_df["symbol"] = sym
            locked_df["dte_mode"] = dte_mode
            return locked_df, None
    except Exception as e:
        return None, f"[error] {file_path.name}: {e}"

    return None, None


def main():
    parser = argparse.ArgumentParser(
        description="Build locked target map with V4/old_v2 (main first) lock logic",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG),
        help=f"anchor JSON (default: {DEFAULT_CONFIG})",
    )
    parser.add_argument(
        "--dte-mode",
        choices=["legacy", "calendar", "trading"],
        default="legacy",
        help="legacy=首版 UTC bug（默认，对齐 old_v2）；trading=交易日 DTE；calendar=修 TZ 后日历 DTE",
    )
    parser.add_argument("--start-date", type=str, default=None, help="YYYY-MM-DD")
    parser.add_argument("--end-date", type=str, default=None, help="YYYY-MM-DD")
    parser.add_argument("--raw-dir", type=str, default=None, help="nq_options_day_iv 根目录")
    parser.add_argument("--output", type=str, default=None, help="输出 parquet 路径")
    parser.add_argument("--symbols", type=str, default=None, help="逗号分隔,如 QQQ")
    args = parser.parse_args()

    cfg_path = Path(args.config).expanduser()
    if not cfg_path.is_absolute():
        alt = _SCRIPT_DIR / cfg_path
        cfg_path = alt if alt.exists() else (_CONFIG_DIR / cfg_path.name if (_CONFIG_DIR / cfg_path.name).exists() else cfg_path)
    cfg = load_anchor_config(cfg_path)

    paths = cfg.get("_paths_resolved") or {}
    raw_dir = Path(args.raw_dir).expanduser() if args.raw_dir else paths.get("raw_iv_dir")
    output_file = Path(args.output).expanduser() if args.output else paths.get("locked_targets_output")
    if raw_dir is None:
        raw_dir = Path.home() / "train_data/nq_options_day_iv"
    if output_file is None:
        output_file = Path.home() / "train_data/locked_targets_map_old_v2.parquet"
    raw_dir, output_file = Path(raw_dir), Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    else:
        symbols = [str(s).upper() for s in (cfg.get("symbols") or ["QQQ"])]

    n_buckets = 6 if cfg.get("use_next_buckets") else 4
    logger.info(
        "OLD lock radar | profile=%s | dte_mode=%s | prefer=%s | allowed=%s | buckets=%d",
        cfg.get("profile"),
        args.dte_mode,
        cfg.get("front_prefer_dte"),
        cfg.get("front_allowed_dte"),
        n_buckets,
    )
    logger.info("  config: %s", cfg_path)
    logger.info("  input:  %s", raw_dir)
    logger.info("  output: %s", output_file)

    tasks = []
    for sym in symbols:
        src_dir = raw_dir / sym
        if not src_dir.exists():
            logger.warning("skip %s: missing %s", sym, src_dir)
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
            tasks.append((p, sym, cfg, args.dte_mode))

    if not tasks:
        logger.error("no parquet tasks found")
        return

    workers = max(1, multiprocessing.cpu_count() - 2)
    logger.info("tasks=%d workers=%d", len(tasks), workers)

    all_targets = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        for result_df, error_msg in tqdm(
            executor.map(process_single_file, tasks),
            total=len(tasks),
            desc=f"old/{args.dte_mode}",
        ):
            if error_msg:
                tqdm.write(error_msg)
            if result_df is not None:
                all_targets.append(result_df)

    if not all_targets:
        logger.error("no contracts locked")
        return

    final_map = pd.concat(all_targets, ignore_index=True)
    final_map.to_parquet(output_file, compression="zstd", index=False)
    n_days = final_map["date_str"].nunique()
    logger.info("done | %s rows | %d days | → %s", f"{len(final_map):,}", n_days, output_file)
    if "front_dte" in final_map.columns:
        dte_dist = final_map.groupby("date_str")["front_dte"].first().value_counts().sort_index()
        logger.info("front_dte dist:\n%s", dte_dist.to_string())


if __name__ == "__main__":
    main()

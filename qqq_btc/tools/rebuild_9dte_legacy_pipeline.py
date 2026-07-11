#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Legacy 9DTE 快速训练流水线 — 数据源 /mnt/s990/data/raw_1s/options/QQQ (1012 日, 6-bucket)。

用法:
  python qqq_btc/tools/rebuild_9dte_legacy_pipeline.py
  python qqq_btc/tools/rebuild_9dte_legacy_pipeline.py --step post
  python qqq_btc/tools/rebuild_9dte_legacy_pipeline.py --step train
"""
from __future__ import annotations

import argparse
import json
import logging
import multiprocessing
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from glob import glob
from pathlib import Path

import pandas as pd
from tqdm import tqdm

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

logger = logging.getLogger("rebuild_9dte")

RAW_1S = Path("/mnt/s990/data/raw_1s/options")
RAW_1M = Path("/mnt/s990/data/raw_1m/options")
DAY_IV = Path.home() / "train_data/quote_options_day_iv_9dte_legacy"
MONTHLY_IV = Path.home() / "train_data/quote_options_monthly_iv_9dte_legacy"
BUCKETED = Path.home() / "train_data/quote_options_bucketed_9dte_legacy"
RAW_FEAT = Path.home() / "train_data/quote_features_raw_9dte_legacy"
TRAIN_DIR = Path.home() / "train_data/quote_features_train_9dte_legacy"
VAL_DIR = Path.home() / "train_data/quote_features_val_9dte_legacy"
TEST_DIR = Path.home() / "train_data/quote_features_test_9dte_legacy"
LMDB_ROOT = Path.home() / "train_data/lmdb"
ANCHOR = _REPO / "qqq_btc/CONFIG/anchor_qqq_9dte_legacy.json"
FEATURE_CONFIG = _REPO / "qqq_btc/CONFIG/slow_feature_qqq_v2.json"
SYMBOL_MAP = _REPO / "qqq_btc/CONFIG/symbol_map.json"
CKPT_DIR = _REPO / "checkpoints_qqq_9dte_legacy"
PYTHON = os.environ.get("REBUILD_PYTHON", sys.executable)

TRAIN_RANGE = (pd.Timestamp("2023-03-01"), pd.Timestamp("2025-12-31"))
VAL_RANGE = (pd.Timestamp("2026-01-01"), pd.Timestamp("2026-03-31"))
TEST_RANGE = (pd.Timestamp("2026-04-01"), pd.Timestamp("2026-06-30"))


def step_aggregate(sym: str = "QQQ", force: bool = False) -> None:
    cmd = [
        PYTHON,
        str(_REPO / "preprocess/download/step3_databento_aggregate_1s_to_1m.py"),
        "--input-dir", str(RAW_1S),
        "--output-dir", str(RAW_1M),
        "--symbol", sym,
    ]
    if force:
        cmd.append("--force")
    logger.info("=== aggregate 1s→1m: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(_REPO))


def step_day_iv(sym: str = "QQQ", force: bool = False) -> None:
    sys.path.insert(0, str(_REPO / "preprocess/ask_bid"))
    from option_cac_day_vectorized_day import OptionIVCalculator, init_worker_rfr, compute_single_day_file

    day_files = sorted((RAW_1M / sym).glob(f"{sym}_*.parquet"))
    if not day_files:
        raise SystemExit(f"no 1m files under {RAW_1M / sym}")

    iv_dir = DAY_IV / sym / "standard"
    iv_dir.mkdir(parents=True, exist_ok=True)
    if force:
        for p in day_files:
            out_p = iv_dir / p.name
            if out_p.exists():
                out_p.unlink()

    calc = OptionIVCalculator(
        db_path=str(Path.home() / "notebook/stocks.db"),
        data_root=str(Path.home() / "train_data/spnq_train_resampled"),
        option_root=str(RAW_1M),
        iv_option_root=str(DAY_IV),
    )
    underlying = calc._get_underlying_df(sym)
    if underlying is None or underlying.empty:
        raise SystemExit(f"no underlying for {sym}")

    dates = pd.to_datetime([p.stem.split("_")[-1] for p in day_files])
    calc._load_risk_free_rates(dates)
    rfr = calc._audit_and_prepare_rfr(dates)
    init_worker_rfr(rfr)

    tasks = [(str(p), sym, underlying, str(iv_dir)) for p in day_files]
    logger.info("=== day IV %s: %d files ===", sym, len(tasks))
    ok = 0
    with ThreadPoolExecutor(max_workers=20) as ex:
        futs = {ex.submit(compute_single_day_file, t): t for t in tasks}
        for fut in tqdm(as_completed(futs), total=len(futs), desc="day_iv"):
            if fut.result():
                ok += 1
    logger.info("day IV done: %d/%d", ok, len(tasks))


def step_monthly_iv(sym: str = "QQQ") -> None:
    sys.path.insert(0, str(_REPO / "preprocess/ask_bid"))
    from iv_day2month import process_single_symbol

    files = glob(str(DAY_IV / "**" / "*.parquet"), recursive=True)
    sym_files = [f for f in files if Path(f).name.startswith(f"{sym}_")]
    msg = process_single_symbol((sym, sym_files, str(MONTHLY_IV)))
    logger.info("monthly_iv: %s", msg)


def step_bucketed(sym: str = "QQQ") -> None:
    sys.path.insert(0, str(_REPO / "preprocess/ask_bid"))
    from options_locked_feature import process_single_file

    BUCKETED.mkdir(parents=True, exist_ok=True)
    src_dir = MONTHLY_IV / sym / "standard"
    tasks = [(p, BUCKETED, sym) for p in sorted(src_dir.glob("*.parquet"))]
    logger.info("=== bucketed: %d day files ===", len(tasks))
    with ProcessPoolExecutor(max_workers=16) as pool:
        futs = [pool.submit(process_single_file, t) for t in tasks]
        for fut in tqdm(as_completed(futs), total=len(futs), desc="bucketed"):
            res = fut.result()
            if res:
                logger.warning(res)


def step_feature_merge() -> None:
    import preprocess.ask_bid.feature_merge_option_raw as fm

    logger.info("=== feature_merge -> %s", RAW_FEAT)
    fm.OPTION_MONTHLY_DIR = MONTHLY_IV
    fm.AGG_OPTION_MONTHLY_DIR = BUCKETED
    fm.OUTPUT_FEATURES_DIR = RAW_FEAT
    fm.CONFIG_FILE = str(FEATURE_CONFIG)
    fm.OVERWRITE_EXISTING = True

    with open(FEATURE_CONFIG, "r") as f:
        config = json.load(f)
    fm.generate_vix_level_global(config)
    fm.main()
    fm.update_vol_vix_abs(config)
    fm.update_cat_features_in_files(config)
    fm.update_new_labels_in_files(config)


def _dest_for_month(file_path: Path) -> Path | None:
    ym = pd.Timestamp(file_path.stem + "-01")
    rel = file_path.relative_to(RAW_FEAT / "QQQ")
    if TRAIN_RANGE[0] <= ym <= TRAIN_RANGE[1]:
        return TRAIN_DIR / "QQQ" / rel
    if VAL_RANGE[0] <= ym <= VAL_RANGE[1]:
        return VAL_DIR / "QQQ" / rel
    if TEST_RANGE[0] <= ym <= TEST_RANGE[1]:
        return TEST_DIR / "QQQ" / rel
    return None


def step_split() -> None:
    logger.info("=== split -> train/val/test *_9dte_legacy")
    files = sorted((RAW_FEAT / "QQQ").glob("**/*.parquet"))
    ok = 0
    for fp in tqdm(files, desc="split"):
        dest = _dest_for_month(fp)
        if dest is None:
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(fp, dest)
        ok += 1
    logger.info("copied %d files", ok)


def step_rolling_norm() -> None:
    from preprocess.ask_bid.apply_rolling_norm_standalone import (
        find_leaf_directories,
        load_target_features,
        process_single_directory,
    )

    norm_cols = load_target_features(FEATURE_CONFIG)
    logger.info("=== rolling_norm on *_9dte_legacy (%d cols)", len(norm_cols))
    for stage_root in (TRAIN_DIR, VAL_DIR, TEST_DIR):
        if not stage_root.exists():
            logger.warning("skip missing %s", stage_root)
            continue
        targets = find_leaf_directories(stage_root)
        tasks = [(d, norm_cols) for d in targets]
        with ProcessPoolExecutor(max_workers=max(1, os.cpu_count() - 2)) as pool:
            results = list(tqdm(pool.map(process_single_directory, tasks), total=len(tasks), desc=stage_root.name))
        errs = [r for r in results if r and str(r).startswith("ERROR")]
        if errs:
            raise RuntimeError(f"rolling_norm errors in {stage_root}: {errs[:3]}")


def step_lmdb(sym: str = "QQQ") -> None:
    for stage in ("train", "val", "test"):
        feat = Path.home() / f"train_data/quote_features_{stage}_9dte_legacy"
        out = LMDB_ROOT / f"{stage}_qqq_9dte_legacy.lmdb"
        subprocess.run(
            [
                PYTHON, str(_REPO / "qqq_btc/tools/build_lmdb.py"),
                "--feature-root", str(feat),
                "--config", str(FEATURE_CONFIG),
                "--symbol-map", str(SYMBOL_MAP),
                "--output", str(out),
                "--symbols", sym,
            ],
            check=True,
        )


def step_train(epochs: int = 20) -> None:
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    log_path = CKPT_DIR / "train.log"
    cmd = [
        PYTHON, "-m", "qqq_btc.model.train",
        "--mode", "pretrain",
        "--config", str(FEATURE_CONFIG),
        "--data-root", str(LMDB_ROOT),
        "--train-lmdb", "train_qqq_9dte_legacy.lmdb",
        "--val-lmdbs", "val_qqq_9dte_legacy.lmdb",
        "--checkpoint-dir", str(CKPT_DIR),
        "--epochs", str(epochs),
        "--device", "auto",
    ]
    logger.info("=== train: %s", " ".join(cmd))
    with open(log_path, "w", encoding="utf-8") as logf:
        subprocess.run(cmd, check=True, stdout=logf, stderr=subprocess.STDOUT)
    logger.info("train log -> %s", log_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Legacy 9DTE pipeline from /mnt/s990/data/raw_1s/options/QQQ")
    parser.add_argument(
        "--step",
        default="all",
        choices=["all", "aggregate", "day_iv", "monthly", "bucketed", "merge", "split", "norm", "lmdb", "train", "post"],
    )
    parser.add_argument("--force", action="store_true", help="overwrite 1m/day IV")
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    steps = {
        "aggregate": lambda: step_aggregate(force=args.force),
        "day_iv": lambda: step_day_iv(force=args.force),
        "monthly": step_monthly_iv,
        "bucketed": step_bucketed,
        "merge": step_feature_merge,
        "split": step_split,
        "norm": step_rolling_norm,
        "lmdb": step_lmdb,
        "train": lambda: step_train(epochs=args.epochs),
    }
    order = ["aggregate", "day_iv", "monthly", "bucketed", "merge", "split", "norm", "lmdb", "train"]
    if args.step == "all":
        run = order
    elif args.step == "post":
        run = ["split", "norm", "lmdb", "train"]
    else:
        run = [args.step]

    for name in run:
        steps[name]()

    logger.info("done 9dte legacy pipeline step=%s", args.step)


if __name__ == "__main__":
    try:
        multiprocessing.set_start_method("fork")
    except RuntimeError:
        pass
    main()

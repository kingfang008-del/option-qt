#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Weekly-DTE single-stock pipeline entry point.

This is a thin orchestration layer. It keeps stock option paths/configs separate
from qqq_btc while reusing mature builders such as target-map generation,
LMDB creation, and model training.
"""
from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from stock_options.common.weekly_dte_config import ANCHOR_CONFIG, SYMBOL_MAP_PATH

logger = logging.getLogger("stock_weekly_dte")
PYTHON = os.environ.get("REBUILD_PYTHON", sys.executable)


def load_profile(symbol: str) -> Any:
    module = importlib.import_module(f"stock_options.{symbol.lower()}.config_weekly_dte")
    return module.CONFIG


def run_cmd(cmd: list[str]) -> None:
    logger.info("run: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(_REPO))


def step_show(symbol: str) -> None:
    cfg = load_profile(symbol)
    logger.info("profile bundle:\n%s", json.dumps(_jsonable(cfg.as_dict()), indent=2))


def step_target_map(symbol: str, start_date: str | None, end_date: str | None) -> None:
    cmd = [
        PYTHON,
        str(_REPO / "preprocess/download/step1_build_target_map.py"),
        "--config",
        str(ANCHOR_CONFIG),
        "--symbols",
        symbol.upper(),
    ]
    if start_date:
        cmd += ["--start-date", start_date]
    if end_date:
        cmd += ["--end-date", end_date]
    run_cmd(cmd)


def step_feature_merge(
    symbol: str,
    start_date: str | None,
    end_date: str | None,
    workers: int,
    force: bool,
) -> None:
    cmd = [
        PYTHON,
        str(_REPO / "stock_options/tools/feature_merge_weekly_dte.py"),
        "--symbol",
        symbol.upper(),
        "--step",
        "all",
        "--workers",
        str(workers),
    ]
    if start_date:
        cmd += ["--start-date", start_date]
    if end_date:
        cmd += ["--end-date", end_date]
    if force:
        cmd.append("--overwrite")
        cmd.append("--force-enrich")
    run_cmd(cmd)


def _filter_month_file(src: Path, dst: Path, cfg: Any, force: bool) -> int:
    if dst.exists() and not force:
        return len(pd.read_parquet(dst, columns=["timestamp"]))

    df = pd.read_parquet(src)
    ts = pd.to_datetime(df["timestamp"], errors="coerce")
    mask = ts.dt.dayofweek.isin(cfg.train_weekdays)

    # If feature_merge has a DTE column, enforce the weekly-DTE bucket directly.
    for dte_col in ("dte", "front_dte", "option_dte"):
        if dte_col in df.columns:
            mask &= df[dte_col].isin(cfg.allowed_dte)
            break

    sub = df.loc[mask].copy()
    if sub.empty:
        logger.warning("skip %s: no rows for weekdays=%s dte=%s", src, cfg.train_weekdays, cfg.allowed_dte)
        return 0

    dst.parent.mkdir(parents=True, exist_ok=True)
    sub.to_parquet(dst, index=False)
    logger.info("filter %s -> %s | %d/%d rows", src.name, dst.name, len(sub), len(df))
    return len(sub)


def step_filter(symbol: str, force: bool = False) -> None:
    cfg = load_profile(symbol)
    stages = {
        "train": cfg.feature_train_root,
        "val": cfg.feature_val_root,
        "test": cfg.feature_test_root,
    }
    for stage, src_root in stages.items():
        dst_root = src_root.parent / f"{src_root.name}_weekday_dte"
        total = 0
        for res in ("1min", "5min"):
            src_dir = src_root / cfg.symbol / "regular" / "09:30-16:00" / res
            if not src_dir.exists():
                logger.warning("missing feature dir: %s", src_dir)
                continue
            for src in sorted(src_dir.glob("*.parquet")):
                dst = dst_root / cfg.symbol / "regular" / "09:30-16:00" / res / src.name
                total += _filter_month_file(src, dst, cfg, force)
        logger.info("stage=%s filtered_rows=%d dst=%s", stage, total, dst_root)


def step_lmdb(symbol: str) -> None:
    cfg = load_profile(symbol)
    run_cmd(
        [
            PYTHON,
            str(_REPO / "stock_options/tools/feature_merge_weekly_dte.py"),
            "--symbol",
            cfg.symbol,
            "--step",
            "config",
        ]
    )
    with open(SYMBOL_MAP_PATH, encoding="utf-8") as f:
        if cfg.symbol not in json.load(f):
            raise KeyError(f"{cfg.symbol} missing from {SYMBOL_MAP_PATH}")

    stage_roots = {
        "train": cfg.feature_train_root,
        "val": cfg.feature_val_root,
        "test": cfg.feature_test_root,
    }
    stage_out = {
        "train": cfg.lmdb_train,
        "val": cfg.lmdb_val,
        "test": cfg.lmdb_test,
    }
    for stage, root in stage_roots.items():
        filtered = root.parent / f"{root.name}_weekday_dte"
        feature_root = filtered if filtered.exists() else root
        run_cmd(
            [
                PYTHON,
                str(_REPO / "qqq_btc/tools/build_lmdb.py"),
                "--feature-root",
                str(feature_root),
                "--config",
                str(cfg.feature_config),
                "--symbol-map",
                str(SYMBOL_MAP_PATH),
                "--output",
                str(stage_out[stage]),
                "--symbols",
                cfg.symbol,
            ]
        )


def step_train(symbol: str, epochs: int) -> None:
    cfg = load_profile(symbol)
    cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    run_cmd(
        [
            PYTHON,
            "-m",
            "qqq_btc.model.train",
            "--mode",
            "pretrain",
            "--config",
            str(cfg.feature_config),
            "--data-root",
            str(cfg.lmdb_train.parent),
            "--train-lmdb",
            cfg.lmdb_train.name,
            "--val-lmdbs",
            cfg.lmdb_val.name,
            "--checkpoint-dir",
            str(cfg.checkpoint_dir),
            "--epochs",
            str(epochs),
            "--device",
            "auto",
        ]
    )


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, tuple):
        return list(obj)
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    return obj


def main() -> None:
    parser = argparse.ArgumentParser(description="Stock weekly-DTE pipeline")
    parser.add_argument("--symbol", default="NVDA", choices=["NVDA", "TSLA"])
    parser.add_argument(
        "--step",
        default="show",
        choices=["show", "target-map", "feature-merge", "filter", "lmdb", "train", "all"],
    )
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    order = ["target-map", "feature-merge", "filter", "lmdb", "train"] if args.step == "all" else [args.step]
    for step in order:
        if step == "show":
            step_show(args.symbol)
        elif step == "target-map":
            step_target_map(args.symbol, args.start_date, args.end_date)
        elif step == "feature-merge":
            step_feature_merge(args.symbol, args.start_date, args.end_date, args.workers, args.force)
        elif step == "filter":
            step_filter(args.symbol, force=args.force)
        elif step == "lmdb":
            step_lmdb(args.symbol)
        elif step == "train":
            step_train(args.symbol, epochs=args.epochs)


if __name__ == "__main__":
    main()


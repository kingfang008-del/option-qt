#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Non-0DTE stock TFT pipeline entry (isolated; does not modify QQQ/0DTE code).

  feature-build → split → label → lmdb → train (qqq_btc.model.train)
"""
from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from stock_options.common.non0dte_config import FEATURE_CONFIG_PATH, SYMBOL_MAP_PATH

logger = logging.getLogger("stock_non0dte_tft")
PYTHON = os.environ.get("REBUILD_PYTHON", sys.executable)


def load_profile(symbol: str) -> Any:
    return importlib.import_module(f"stock_options.{symbol.lower()}.config_non0dte").CONFIG


def run_cmd(cmd: list[str]) -> None:
    logger.info("run: %s", " ".join(cmd))
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{_REPO}{os.pathsep}{env.get('PYTHONPATH', '')}"
    env["FEATURE_CONFIG"] = str(FEATURE_CONFIG_PATH)
    subprocess.run(cmd, check=True, cwd=str(_REPO), env=env)


def step_show(symbol: str) -> None:
    cfg = load_profile(symbol)
    def j(o):
        if isinstance(o, Path):
            return str(o)
        if isinstance(o, dict):
            return {k: j(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [j(v) for v in o]
        return o
    print(json.dumps(j(cfg.as_dict()), indent=2))


def step_features(symbol: str, workers: int, force: bool) -> None:
    cmd = [
        PYTHON,
        str(_REPO / "stock_options/tools/feature_build_stock_non0dte.py"),
        "--symbol",
        symbol.upper(),
        "--step",
        "all",
        "--workers",
        str(workers),
    ]
    if force:
        cmd += ["--force", "--overwrite"]
    run_cmd(cmd)


def _month_in_range(month: str, start: str, end: str) -> bool:
    # month is YYYY-MM; compare against date bounds inclusively by month overlap
    m0 = f"{month}-01"
    m1 = (pd.Timestamp(month) + pd.offsets.MonthEnd(0)).strftime("%Y-%m-%d")
    return not (m1 < start or m0 > end)


def step_split(symbol: str, force: bool = False) -> None:
    """Split feature_raw months into train/val/test under build_root (1min + 5min)."""
    cfg = load_profile(symbol)
    src_root = cfg.feature_raw_root
    if not src_root.exists():
        raise SystemExit(f"missing feature_raw: {src_root}")

    ranges = {
        "train": (cfg.research_start, cfg.split_train_end, cfg.feature_train_root),
        "val": (cfg.split_train_end, cfg.split_val_end, cfg.feature_val_root),
        "test": (cfg.split_val_end, cfg.split_test_end, cfg.feature_test_root),
    }
    files = sorted(src_root.glob(f"{cfg.symbol}/regular/09:30-16:00/*/*.parquet"))
    if not files:
        files = sorted(src_root.glob("*/*/*/*/*.parquet"))
        files = [f for f in files if cfg.symbol in str(f)]
    if not files:
        raise SystemExit(f"no feature files under {src_root}")

    for stage, (start, end, dst_root) in ranges.items():
        n = 0
        for src in files:
            df = pd.read_parquet(src)
            ts = pd.to_datetime(df["timestamp"], errors="coerce")
            if getattr(ts.dt, "tz", None) is not None:
                d = ts.dt.tz_convert("America/New_York").dt.strftime("%Y-%m-%d")
            else:
                d = ts.dt.strftime("%Y-%m-%d")
            if stage == "train":
                mask = (d >= start) & (d <= end)
            else:
                mask = (d > start) & (d <= end)
            if "stock_dte" in df.columns:
                mask &= df["stock_dte"] >= 1
            if "stock_is_0dte" in df.columns:
                mask &= df["stock_is_0dte"] < 0.5
            sub = df.loc[mask].copy()
            if sub.empty:
                continue
            try:
                rel = src.relative_to(src_root)
            except ValueError:
                rel = Path(cfg.symbol) / "regular" / "09:30-16:00" / "1min" / src.name
            dst = dst_root / rel
            if dst.exists() and not force:
                n += len(sub)
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            sub.to_parquet(dst, index=False)
            n += len(sub)
        logger.info("split %s rows=%d -> %s", stage, n, dst_root)


def step_label(symbol: str) -> None:
    cfg = load_profile(symbol)
    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    for stage, root in {
        "train": cfg.feature_train_root,
        "val": cfg.feature_val_root,
        "test": cfg.feature_test_root,
    }.items():
        minute = root / cfg.symbol / "regular" / "09:30-16:00" / "1min"
        if not minute.exists():
            logger.warning("skip label %s missing %s", stage, minute)
            continue
        run_cmd(
            [
                PYTHON,
                str(_REPO / "qqq_btc/tools/label_pipeline.py"),
                "--input",
                str(minute),
                "--output",
                str(minute),
                "--symbol",
                cfg.symbol,
                "--anchor-config",
                str(cfg.anchor_config),
                "--report",
                str(cfg.results_dir / f"label_report_{stage}.json"),
            ]
        )


def step_lmdb(symbol: str) -> None:
    cfg = load_profile(symbol)
    with open(SYMBOL_MAP_PATH, encoding="utf-8") as f:
        if cfg.symbol not in json.load(f):
            raise KeyError(cfg.symbol)
    for stage, root, out in [
        ("train", cfg.feature_train_root, cfg.lmdb_train),
        ("val", cfg.feature_val_root, cfg.lmdb_val),
        ("test", cfg.feature_test_root, cfg.lmdb_test),
    ]:
        if not root.exists():
            logger.warning("skip lmdb %s", stage)
            continue
        run_cmd(
            [
                PYTHON,
                str(_REPO / "qqq_btc/tools/build_lmdb.py"),
                "--feature-root",
                str(root),
                "--config",
                str(cfg.feature_config),
                "--symbol-map",
                str(SYMBOL_MAP_PATH),
                "--output",
                str(out),
                "--symbols",
                cfg.symbol,
                "--window-step",
                "1",
            ]
        )


def step_train(symbol: str, epochs: int, device: str) -> None:
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
            device,
        ]
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Stock non-0DTE TFT pipeline")
    p.add_argument("--symbol", default="NVDA", choices=["NVDA", "TSLA"])
    p.add_argument(
        "--step",
        required=True,
        choices=["show", "features", "split", "label", "lmdb", "train", "all"],
    )
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--device", default="auto")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    order = ["features", "split", "label", "lmdb", "train"] if args.step == "all" else [args.step]
    for step in order:
        if step == "show":
            step_show(args.symbol)
        elif step == "features":
            step_features(args.symbol, args.workers, args.force)
        elif step == "split":
            step_split(args.symbol, force=args.force)
        elif step == "label":
            step_label(args.symbol)
        elif step == "lmdb":
            step_lmdb(args.symbol)
        elif step == "train":
            step_train(args.symbol, args.epochs, args.device)


if __name__ == "__main__":
    main()

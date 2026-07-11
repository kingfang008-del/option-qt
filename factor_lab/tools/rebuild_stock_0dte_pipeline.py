#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
个股 0DTE (周频 / Phase-1 仅周五) 训练管线。

与 QQQ/9DTE 隔离:
  - anchor: anchor_stock_0dte_weekly.json (DTE=0, weekly expiry)
  - config: qqq_btc/stock/config_0dte.py
  - symbol_map: symbol_map_stock.json

Phase-1 策略 (数据有限):
  train_weekdays=deploy_weekdays=(4,)  → 仅周五样本进 LMDB
  周四模型不能推周五合约: 因 DTE/gamma/theta 分布不同, 必须同 regime 训练+部署

用法:
  python qqq_btc/tools/rebuild_stock_0dte_pipeline.py --symbol NVDA --step filter
  python qqq_btc/tools/rebuild_stock_0dte_pipeline.py --symbol NVDA --step lmdb
  python qqq_btc/tools/rebuild_stock_0dte_pipeline.py --symbol NVDA --step train
"""
from __future__ import annotations

import argparse
import json
import logging
import multiprocessing
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from qqq_btc.stock.config_0dte import for_symbol, TRAIN_WEEKDAYS

logger = logging.getLogger("rebuild_stock_0dte")
PYTHON = os.environ.get("REBUILD_PYTHON", sys.executable)
LMDB_ROOT = Path.home() / "train_data/lmdb"


def _weekday_name(d: int) -> str:
    return ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")[d]


def filter_feature_dir(
    src_root: Path,
    dst_root: Path,
    symbol: str,
    weekdays: tuple[int, ...],
    *,
    force: bool = False,
) -> int:
    """按 weekday 过滤 feature parquet (1min + 5min), 写入 dst_root。"""
    src_root = src_root.expanduser()
    dst_root = dst_root.expanduser()
    sym = symbol.upper()
    kept_rows = 0
    for res in ("1min", "5min"):
        src_dir = src_root / sym / "regular" / "09:30-16:00" / res
        if not src_dir.exists():
            logger.warning("missing %s", src_dir)
            continue
        for src in sorted(src_dir.glob("*.parquet")):
            dst = dst_root / sym / "regular" / "09:30-16:00" / res / src.name
            if dst.exists() and not force:
                kept_rows += len(pd.read_parquet(dst, columns=["timestamp"]))
                continue
            df = pd.read_parquet(src)
            ts = pd.to_datetime(df["timestamp"])
            sub = df.loc[ts.dt.dayofweek.isin(weekdays)].copy()
            if sub.empty:
                logger.warning("skip %s: no rows for weekdays=%s", src.name, weekdays)
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            sub.to_parquet(dst, index=False)
            kept_rows += len(sub)
            logger.info(
                "filter %s -> %s | %d/%d rows (%s)",
                src.name, dst.name, len(sub), len(df),
                ",".join(_weekday_name(d) for d in weekdays),
            )
    return kept_rows


def step_filter(symbol: str, weekdays: tuple[int, ...], force: bool = False) -> None:
    cfg = for_symbol(symbol)
    for stage in ("train", "val", "test"):
        src = cfg[f"feature_{stage}_root"]
        dst = src.parent / f"{src.name}_fri"
        n = filter_feature_dir(src, dst, symbol, weekdays, force=force)
        logger.info("stage=%s filtered_rows=%d dst=%s", stage, n, dst)


def step_lmdb(symbol: str, weekdays: tuple[int, ...]) -> None:
    cfg = for_symbol(symbol)
    suf = symbol.lower()
    sym_map = cfg["feature_config"].parent / "symbol_map_stock.json"
    with open(sym_map, encoding="utf-8") as f:
        if symbol.upper() not in json.load(f):
            raise KeyError(f"{symbol} missing from {sym_map}")

    for stage in ("train", "val", "test"):
        feat = cfg[f"feature_{stage}_root"]
        fri = feat.parent / f"{feat.name}_fri"
        root = fri if fri.exists() else feat
        out = LMDB_ROOT / f"{stage}_{suf}_stock_0dte_fri.lmdb"
        subprocess.run(
            [
                PYTHON, str(_REPO / "qqq_btc/tools/build_lmdb.py"),
                "--feature-root", str(root),
                "--config", str(cfg["feature_config"]),
                "--symbol-map", str(sym_map),
                "--output", str(out),
                "--symbols", symbol.upper(),
            ],
            check=True,
        )
        logger.info("lmdb %s -> %s (weekdays=%s)", stage, out, weekdays)


def step_train(symbol: str, epochs: int = 20) -> None:
    cfg = for_symbol(symbol)
    suf = symbol.lower()
    ckpt = cfg["checkpoint"]
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        PYTHON, "-m", "qqq_btc.model.train",
        "--mode", "pretrain",
        "--config", str(cfg["feature_config"]),
        "--data-root", str(LMDB_ROOT),
        "--train-lmdb", f"train_{suf}_stock_0dte_fri.lmdb",
        "--val-lmdbs", f"val_{suf}_stock_0dte_fri.lmdb",
        "--checkpoint-dir", str(ckpt.parent),
        "--epochs", str(epochs),
        "--device", "auto",
    ]
    log_path = ckpt.parent / "train.log"
    logger.info("train: %s", " ".join(cmd))
    with open(log_path, "w", encoding="utf-8") as logf:
        subprocess.run(cmd, check=True, stdout=logf, stderr=subprocess.STDOUT)


def main() -> None:
    parser = argparse.ArgumentParser(description="Single-stock 0DTE Friday-only pipeline")
    parser.add_argument("--symbol", default="NVDA")
    parser.add_argument(
        "--weekdays", default="4",
        help="Comma-separated dayofweek (0=Mon, 4=Fri). Default: 4 (Friday only)",
    )
    parser.add_argument("--step", default="all", choices=["all", "filter", "lmdb", "train"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    weekdays = tuple(int(x.strip()) for x in args.weekdays.split(",") if x.strip())
    steps = {
        "filter": lambda: step_filter(args.symbol, weekdays, force=args.force),
        "lmdb": lambda: step_lmdb(args.symbol, weekdays),
        "train": lambda: step_train(args.symbol, epochs=args.epochs),
    }
    order = ["filter", "lmdb", "train"] if args.step == "all" else [args.step]
    for name in order:
        steps[name]()
    logger.info("done stock_0dte symbol=%s weekdays=%s", args.symbol, weekdays)


if __name__ == "__main__":
    try:
        multiprocessing.set_start_method("fork")
    except RuntimeError:
        pass
    main()

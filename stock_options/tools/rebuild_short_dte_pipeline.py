#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Short-DTE weekday modeling via V4 dual-stream TFT.

Mainline for MAG7 NVDA/TSLA:
  locked map → (IV/bucket/merge) → weekday/DTE enrich → split/norm/label
  → LMDB → ``python -m qqq_btc.model.train`` (same TFT backbone as QQQ V4)

State-gate rule tools remain optional diagnostics; they are NOT the model path.
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

from stock_options.common.short_dte_config import (
    ANCHOR_CONFIG,
    FEATURE_CONFIG_PATH,
    RESEARCH_START,
    SYMBOL_MAP_PATH,
)

logger = logging.getLogger("stock_short_dte_tft")
PYTHON = os.environ.get("REBUILD_PYTHON", sys.executable)


def load_profile(symbol: str) -> Any:
    module = importlib.import_module(f"stock_options.{symbol.lower()}.config_short_dte")
    return module.CONFIG


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return obj


def run_cmd(cmd: list[str]) -> None:
    logger.info("run: %s", " ".join(cmd))
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{_REPO}{os.pathsep}{env.get('PYTHONPATH', '')}"
    env["FEATURE_CONFIG"] = str(FEATURE_CONFIG_PATH)
    subprocess.run(cmd, check=True, cwd=str(_REPO), env=env)


def step_show(symbol: str) -> None:
    cfg = load_profile(symbol)
    print(json.dumps(_jsonable(cfg.as_dict()), indent=2))


def step_feature_config(symbol: str) -> None:
    run_cmd(
        [
            PYTHON,
            str(_REPO / "stock_options/tools/feature_merge_short_dte.py"),
            "--symbol",
            symbol.upper(),
            "--step",
            "config",
        ]
    )


def step_weekday_report(symbols: list[str], probe_micro: bool) -> None:
    cmd = [
        PYTHON,
        str(_REPO / "stock_options/tools/report_mag7_short_dte_weekday_coverage.py"),
        "--symbols",
        ",".join(symbols),
        "--start-date",
        RESEARCH_START,
    ]
    if probe_micro:
        cmd.append("--probe-micro")
    run_cmd(cmd)


def step_feature_merge(
    symbol: str,
    start_date: str | None,
    end_date: str | None,
    workers: int,
    force: bool,
) -> None:
    cfg = load_profile(symbol)
    cmd = [
        PYTHON,
        str(_REPO / "stock_options/tools/feature_merge_short_dte.py"),
        "--symbol",
        cfg.symbol,
        "--step",
        "all",
        "--start-date",
        start_date or RESEARCH_START,
        "--end-date",
        end_date or cfg.split_test_end,
        "--workers",
        str(workers),
    ]
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
    if "stock_dte" in df.columns:
        mask &= df["stock_dte"].isin(cfg.allowed_dte)
    elif "stock_is_short_dte" in df.columns:
        mask &= df["stock_is_short_dte"] > 0.5

    sub = df.loc[mask].copy()
    if sub.empty:
        logger.warning("skip %s: empty after weekday/dte filter", src)
        return 0
    dst.parent.mkdir(parents=True, exist_ok=True)
    sub.to_parquet(dst, index=False)
    logger.info("filter %s -> %s | %d/%d", src.name, dst, len(sub), len(df))
    return len(sub)


def step_filter(symbol: str, force: bool = False) -> None:
    """Keep only short-DTE weekday rows for TFT LMDB."""
    cfg = load_profile(symbol)
    stages = {
        "train": cfg.feature_train_root,
        "val": cfg.feature_val_root,
        "test": cfg.feature_test_root,
    }
    for stage, root in stages.items():
        # Prefer raw tree if split trees are not built yet.
        src_root = root if root.exists() else cfg.feature_raw_root
        if not src_root.exists():
            logger.warning("missing feature root for %s: %s", stage, src_root)
            continue
        dst_root = root.parent / f"{root.name}_weekday_dte"
        total = 0
        for res_dir in src_root.glob("*/*/09:30-16:00/*"):
            if not res_dir.is_dir():
                continue
            for src in sorted(res_dir.glob("*.parquet")):
                # Mirror QQQ layout: SYMBOL/regular/09:30-16:00/1min/YYYY-MM.parquet
                rel = src.relative_to(src_root) if src_root == cfg.feature_raw_root else src.relative_to(root)
                dst = dst_root / rel
                total += _filter_month_file(src, dst, cfg, force)
        logger.info("stage=%s filtered_rows=%d dst=%s", stage, total, dst_root)


def step_label(symbol: str) -> None:
    cfg = load_profile(symbol)
    for stage, root in {
        "train": cfg.feature_train_root,
        "val": cfg.feature_val_root,
        "test": cfg.feature_test_root,
    }.items():
        filtered = root.parent / f"{root.name}_weekday_dte"
        feature_root = filtered if filtered.exists() else root
        minute_dir = feature_root / cfg.symbol / "regular" / "09:30-16:00" / "1min"
        if not minute_dir.exists():
            logger.warning("skip label %s: missing %s", stage, minute_dir)
            continue
        run_cmd(
            [
                PYTHON,
                str(_REPO / "qqq_btc/tools/label_pipeline.py"),
                "--input",
                str(minute_dir),
                "--output",
                str(minute_dir),
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
    step_feature_config(symbol)
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
        if not feature_root.exists():
            logger.warning("skip lmdb %s: missing %s", stage, feature_root)
            continue
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
                "--window-step",
                "1",
            ]
        )


def step_train(symbol: str, epochs: int, device: str) -> None:
    cfg = load_profile(symbol)
    cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    cfg.results_dir.mkdir(parents=True, exist_ok=True)
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


def step_target_map(symbols: list[str], start_date: str | None, end_date: str | None) -> None:
    cmd = [
        PYTHON,
        str(_REPO / "preprocess/download/build_mag7_short_dte_api_ladder_map.py"),
        "--symbols",
        ",".join(symbols),
        "--start-date",
        start_date or RESEARCH_START,
    ]
    if end_date:
        cmd += ["--end-date", end_date]
    run_cmd(cmd)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MAG7 short-DTE V4 TFT pipeline")
    p.add_argument("--symbol", default="NVDA", choices=["NVDA", "TSLA"])
    p.add_argument("--symbols", default="NVDA,TSLA")
    p.add_argument(
        "--step",
        required=True,
        choices=[
            "show",
            "feature-config",
            "weekday-report",
            "target-map",
            "feature-merge",
            "filter",
            "label",
            "lmdb",
            "train",
            "tft-light",  # config + weekday report only
        ],
    )
    p.add_argument("--start-date", default=None)
    p.add_argument("--end-date", default=None)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--device", default="auto")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--force", action="store_true")
    p.add_argument("--probe-micro", action="store_true")
    p.add_argument("--anchor", default=str(ANCHOR_CONFIG))
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

    if args.step == "show":
        step_show(args.symbol)
    elif args.step == "feature-config":
        step_feature_config(args.symbol)
    elif args.step == "weekday-report":
        step_weekday_report(symbols, args.probe_micro)
    elif args.step == "target-map":
        step_target_map(symbols, args.start_date, args.end_date)
    elif args.step == "feature-merge":
        step_feature_merge(args.symbol, args.start_date, args.end_date, args.workers, args.force)
    elif args.step == "filter":
        step_filter(args.symbol, force=args.force)
    elif args.step == "label":
        step_label(args.symbol)
    elif args.step == "lmdb":
        step_lmdb(args.symbol)
    elif args.step == "train":
        step_train(args.symbol, args.epochs, args.device)
    elif args.step == "tft-light":
        step_show(args.symbol)
        step_feature_config(args.symbol)
        step_weekday_report(symbols, args.probe_micro)
    else:
        raise SystemExit(f"unknown step {args.step}")


if __name__ == "__main__":
    main()

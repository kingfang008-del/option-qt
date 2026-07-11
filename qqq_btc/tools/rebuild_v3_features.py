#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
从 options_databento_v3 重建 QQQ 特征 → split → norm → label → eval。

用法:
  python qqq_btc/tools/rebuild_v3_features.py --step all
  python qqq_btc/tools/rebuild_v3_features.py --step eval
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
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from glob import glob
from pathlib import Path

import pandas as pd
from tqdm import tqdm

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

logger = logging.getLogger("rebuild_v3")

V3_1M = Path("/mnt/s990/data/raw_1m/options_databento_v3")
DAY_IV = Path.home() / "train_data/quote_options_day_iv_v3"
MONTHLY_IV = Path.home() / "train_data/quote_options_monthly_iv_v3"
BUCKETED_IV = Path.home() / "train_data/quote_options_bucketed_v7_v3"
RAW_FEAT = Path.home() / "train_data/quote_features_raw_v3"
TRAIN_DIR = Path.home() / "train_data/quote_features_train_v3"
VAL_DIR = Path.home() / "train_data/quote_features_val_v3"
TEST_DIR = Path.home() / "train_data/quote_features_test_v3"
ANCHOR_V3 = _REPO / "qqq_btc/CONFIG/anchor_qqq_0dte_v3.json"
FEATURE_CONFIG = _REPO / "qqq_btc/CONFIG/slow_feature_qqq_v2.json"
V4_CONFIG_FALLBACK = _REPO / "qqq_btc/CONFIG/slow_feature_qqq_v4.json"
SYMBOL_MAP = _REPO / "qqq_btc/CONFIG/symbol_map.json"
LMDB_ROOT = Path.home() / "train_data/lmdb"
CKPT_DIR = _REPO / "checkpoints_qqq_v3_0dte"
IC_REPORT = _REPO / "qqq_btc/results/v3_0dte_ic_sanity.json"
CHECKPOINT = CKPT_DIR / "best.pth"
EVAL_OUT = Path("/tmp/qqq_btc_test_eval_v3_0dte")
PYTHON = os.environ.get("REBUILD_PYTHON", sys.executable)

TRAIN_RANGE = (pd.Timestamp("2023-03-01"), pd.Timestamp("2025-12-31"))
VAL_RANGE = (pd.Timestamp("2026-01-01"), pd.Timestamp("2026-03-31"))
TEST_RANGE = (pd.Timestamp("2026-04-01"), pd.Timestamp("2026-06-30"))


def ensure_anchor_config() -> None:
    src = _REPO / "qqq_btc/CONFIG/anchor_qqq_0dte.json"
    cfg = json.loads(src.read_text(encoding="utf-8"))
    cfg["paths"]["day_iv_dir"] = str(DAY_IV)
    cfg["description"] = "QQQ 0DTE anchor — day_iv from options_databento_v3"
    ANCHOR_V3.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("wrote anchor config -> %s", ANCHOR_V3)


def step_day_iv() -> None:
    from preprocess.ask_bid.option_cac_day_vectorized_day import OptionIVCalculator

    logger.info("=== step_day_iv: %s -> %s", V3_1M, DAY_IV)
    calc = OptionIVCalculator(
        db_path=str(Path.home() / "notebook/stocks.db"),
        option_root=str(V3_1M),
        data_root=str(Path.home() / "train_data/spnq_train_resampled"),
        iv_option_root=str(DAY_IV),
    )
    calc.run(max_concurrent_stocks=1)


def step_monthly_iv() -> None:
    from preprocess.ask_bid.iv_day2month import process_single_symbol, get_target_symbols

    logger.info("=== step_monthly_iv: %s -> %s", DAY_IV, MONTHLY_IV)
    symbols = get_target_symbols(str(Path.home() / "notebook/stocks.db"))
    files = glob(str(DAY_IV / "**" / "*.parquet"), recursive=True)
    sym_files: dict[str, list[str]] = defaultdict(list)
    for fp in files:
        name = Path(fp).name
        sym = name.rsplit("_", 1)[0]
        if sym in symbols:
            sym_files[sym].append(fp)
    for sym, flist in sym_files.items():
        msg = process_single_symbol((sym, flist, str(MONTHLY_IV)))
        logger.info(msg)


def step_bucketed_iv() -> None:
    from preprocess.ask_bid.options_locked_feature import process_single_file

    logger.info("=== step_bucketed_iv: %s -> %s", MONTHLY_IV, BUCKETED_IV)
    BUCKETED_IV.mkdir(parents=True, exist_ok=True)
    src_dir = MONTHLY_IV / "QQQ" / "standard"
    tasks = [(p, BUCKETED_IV, "QQQ") for p in sorted(src_dir.glob("*.parquet"))]
    with ProcessPoolExecutor(max_workers=16) as pool:
        futs = [pool.submit(process_single_file, t) for t in tasks]
        for fut in tqdm(as_completed(futs), total=len(futs), desc="bucketed_iv"):
            res = fut.result()
            if res:
                logger.warning(res)


def step_feature_merge() -> None:
    import preprocess.ask_bid.feature_merge_option_raw as fm

    logger.info("=== step_feature_merge -> %s", RAW_FEAT)
    fm.OPTION_MONTHLY_DIR = MONTHLY_IV
    fm.AGG_OPTION_MONTHLY_DIR = BUCKETED_IV
    fm.OUTPUT_FEATURES_DIR = RAW_FEAT
    fm.OVERWRITE_EXISTING = True
    fm.main()


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
    logger.info("=== step_split: %s -> train/val/test *_v3", RAW_FEAT)
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

    cfg = _feature_config()
    os.environ["FEATURE_CONFIG"] = str(cfg)
    norm_cols = load_target_features(cfg)
    logger.info("=== step_rolling_norm (%d cols) cfg=%s", len(norm_cols), cfg.name)

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


def step_labels() -> None:
    logger.info("=== step_labels on train/val/test *_v3")
    py = sys.executable
    for stage_dir in (TRAIN_DIR, VAL_DIR, TEST_DIR):
        inp = stage_dir / "QQQ/regular/09:30-16:00/1min"
        if not inp.exists():
            logger.warning("skip labels: %s", inp)
            continue
        subprocess.run(
            [
                py,
                str(_REPO / "qqq_btc/tools/label_pipeline.py"),
                "--input",
                str(inp),
                "--output",
                str(inp),
                "--symbol",
                "QQQ",
                "--anchor-config",
                str(ANCHOR_V3),
                "--report",
                f"/tmp/label_report_v3_{stage_dir.name}.json",
            ],
            check=True,
        )


def _feature_config() -> Path:
    if FEATURE_CONFIG.exists():
        return FEATURE_CONFIG
    return V4_CONFIG_FALLBACK


def step_lmdb(sym: str = "QQQ") -> None:
    cfg = _feature_config()
    for stage in ("train", "val", "test"):
        feat = Path.home() / f"train_data/quote_features_{stage}_v3"
        out = LMDB_ROOT / f"{stage}_qqq_v3_0dte.lmdb"
        logger.info("=== step_lmdb %s -> %s", feat.name, out.name)
        subprocess.run(
            [
                PYTHON,
                str(_REPO / "qqq_btc/tools/build_lmdb.py"),
                "--feature-root",
                str(feat),
                "--config",
                str(cfg),
                "--symbol-map",
                str(SYMBOL_MAP),
                "--output",
                str(out),
                "--symbols",
                sym,
                "--window-step",
                "1",
            ],
            check=True,
        )


def step_train(epochs: int = 20) -> None:
    cfg = _feature_config()
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    log_path = CKPT_DIR / "train.log"
    cmd = [
        PYTHON,
        "-m",
        "qqq_btc.model.train",
        "--mode",
        "pretrain",
        "--config",
        str(cfg),
        "--data-root",
        str(LMDB_ROOT),
        "--train-lmdb",
        "train_qqq_v3_0dte.lmdb",
        "--val-lmdbs",
        "val_qqq_v3_0dte.lmdb",
        "--checkpoint-dir",
        str(CKPT_DIR),
        "--epochs",
        str(epochs),
        "--device",
        "auto",
    ]
    logger.info("=== step_train: %s", " ".join(cmd))
    with open(log_path, "w", encoding="utf-8") as logf:
        subprocess.run(cmd, check=True, stdout=logf, stderr=subprocess.STDOUT)
    logger.info("train log -> %s", log_path)


def step_ic_report() -> None:
    import torch

    cfg_path = _feature_config()
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    from qqq_btc.model.dataset import LMDBAlphaDataset

    report = {
        "model": str(CHECKPOINT),
        "data_source_1m": str(V3_1M),
        "feature_config": str(cfg_path),
        "lmdb": {},
        "v4_reference_ic": 0.271,
        "legacy_9dte_ic": 0.2475,
    }
    for stage, name in [("train", "train_qqq_v3_0dte.lmdb"), ("val", "val_qqq_v3_0dte.lmdb")]:
        p = LMDB_ROOT / name
        if p.exists():
            ds = LMDBAlphaDataset(str(p), cfg)
            report["lmdb"][stage] = {"keys": len(ds.keys), **ds.sanity_check(sample=3000)}

    if CHECKPOINT.exists():
        ck = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
        report["best_val_ic"] = float(ck.get("best_ic", 0.0))
        report["best_epoch"] = int(ck.get("epoch", -1)) + 1
        report["verdict"] = "IC_NORMAL" if report["best_val_ic"] >= 0.15 else "IC_LOW"

    IC_REPORT.parent.mkdir(parents=True, exist_ok=True)
    IC_REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("ic report -> %s", IC_REPORT)
    logger.info("best_val_ic=%.4f", report.get("best_val_ic", float("nan")))


def step_eval() -> None:
    cfg = _feature_config()
    if not CHECKPOINT.exists():
        logger.warning("skip eval: missing checkpoint %s", CHECKPOINT)
        return

    logger.info("=== step_eval -> %s", EVAL_OUT)
    subprocess.run(
        [
            PYTHON,
            str(_REPO / "qqq_btc/tools/eval_test_set.py"),
            "--checkpoint",
            str(CHECKPOINT),
            "--config",
            str(cfg),
            "--feature-root",
            str(TEST_DIR),
            "--option-1m-root",
            str(V3_1M),
            "--output-dir",
            str(EVAL_OUT),
            "--device",
            "auto",
        ],
        check=True,
    )


def print_eval_summary() -> None:
    summary_path = EVAL_OUT / "replay_summary.json"
    if not summary_path.exists():
        logger.warning("no replay summary at %s", summary_path)
        return
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    logger.info("eval summary: %s", summary)

    trades_path = EVAL_OUT / "replay_trades.parquet"
    if trades_path.exists():
        tr = pd.read_parquet(trades_path)
        tr["month"] = pd.to_datetime(tr["entry_ts"]).dt.month
        f = summary.get("position_frac", 0.25)
        for m, g in tr.groupby("month"):
            eq = 1.0
            for nr in g["net_return"]:
                eq *= 1.0 + f * nr
            logger.info("month %s: trades=%d return=%.2f%%", m, len(g), (eq - 1) * 100)


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild QQQ features from databento_v3")
    parser.add_argument(
        "--step",
        default="all",
        choices=[
            "all",
            "day_iv",
            "monthly_iv",
            "bucketed_iv",
            "merge",
            "split",
            "norm",
            "labels",
            "lmdb",
            "train",
            "ic_report",
            "eval",
            "post",
            "fit",
        ],
        help="post = split+norm+labels; fit = lmdb+train+ic_report (+eval if ckpt exists)",
    )
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    ensure_anchor_config()

    steps = {
        "day_iv": step_day_iv,
        "monthly_iv": step_monthly_iv,
        "bucketed_iv": step_bucketed_iv,
        "merge": step_feature_merge,
        "split": step_split,
        "norm": step_rolling_norm,
        "labels": step_labels,
        "lmdb": step_lmdb,
        "train": lambda: step_train(epochs=args.epochs),
        "ic_report": step_ic_report,
        "eval": step_eval,
    }
    order = [
        "day_iv", "monthly_iv", "bucketed_iv", "merge",
        "split", "norm", "labels", "lmdb", "train", "ic_report", "eval",
    ]
    if args.step == "all":
        run = order
    elif args.step == "post":
        run = ["split", "norm", "labels"]
    elif args.step == "fit":
        run = ["lmdb", "train", "ic_report", "eval"]
    else:
        run = [args.step]

    for name in run:
        steps[name]()

    if "eval" in run:
        print_eval_summary()


if __name__ == "__main__":
    try:
        multiprocessing.set_start_method("fork")
    except RuntimeError:
        pass
    main()

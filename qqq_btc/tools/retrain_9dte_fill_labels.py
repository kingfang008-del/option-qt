#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Step 4: 9DTE legacy 期权 fill 标签重训 + replay 验证。

train 2023-03..2025-12 | val 2026-01 | test 2026-02
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import torch

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

logger = logging.getLogger("retrain_9dte_fill")
PYTHON = sys.executable
ANCHOR = _REPO / "qqq_btc/CONFIG/anchor_qqq_9dte_legacy.json"
FEATURE_CFG = _REPO / "qqq_btc/CONFIG/slow_feature_qqq_v2.json"
SYM_MAP = _REPO / "qqq_btc/CONFIG/symbol_map.json"
LMDB = Path.home() / "train_data/lmdb"
CKPT = _REPO / "checkpoints_qqq_9dte_fill_janval"
TMP = Path.home() / "train_data/_9dte_fill_janval_febtest"
REPORT = _REPO / "qqq_btc/results/9dte_fill_retrain_summary.json"


def run(cmd: list[str]) -> None:
    logger.info("run: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def step_labels() -> None:
    for stage in ("train", "val"):  # test 2026-04+ 无 day_iv,跳过
        inp = Path.home() / f"train_data/quote_features_{stage}_9dte_legacy/QQQ/regular/09:30-16:00/1min"
        run([
            PYTHON, str(_REPO / "qqq_btc/tools/label_pipeline.py"),
            "--input", str(inp),
            "--output", str(inp),
            "--symbol", "QQQ",
            "--anchor-config", str(ANCHOR),
            "--report", f"/tmp/label_report_9dte_fill_{stage}.json",
        ])


def _copy_month(stage: str, month: str) -> None:
    src = Path.home() / f"train_data/quote_features_{stage}_9dte_legacy/QQQ/regular/09:30-16:00"
    dst = TMP / f"quote_features_{stage}/QQQ/regular/09:30-16:00"
    for res in ("1min", "5min"):
        (dst / res).mkdir(parents=True, exist_ok=True)
        shutil.copy2(src / res / f"{month}.parquet", dst / res / f"{month}.parquet")


def step_lmdb() -> None:
    run([
        PYTHON, str(_REPO / "qqq_btc/tools/build_lmdb.py"),
        "--feature-root", str(Path.home() / "train_data/quote_features_train_9dte_legacy"),
        "--config", str(FEATURE_CFG),
        "--symbol-map", str(SYM_MAP),
        "--output", str(LMDB / "train_qqq_9dte_fill.lmdb"),
        "--symbols", "QQQ", "--window-step", "1",
    ])
    if TMP.exists():
        shutil.rmtree(TMP)
    _copy_month("val", "2026-01")
    run([
        PYTHON, str(_REPO / "qqq_btc/tools/build_lmdb.py"),
        "--feature-root", str(TMP / "quote_features_val"),
        "--config", str(FEATURE_CFG),
        "--symbol-map", str(SYM_MAP),
        "--output", str(LMDB / "val_qqq_9dte_fill_jan2026.lmdb"),
        "--symbols", "QQQ", "--window-step", "1",
    ])


def step_train(epochs: int) -> None:
    CKPT.mkdir(parents=True, exist_ok=True)
    with open(CKPT / "train.log", "w", encoding="utf-8") as lf:
        subprocess.run([
            PYTHON, "-m", "qqq_btc.model.train",
            "--mode", "pretrain",
            "--config", str(FEATURE_CFG),
            "--data-root", str(LMDB),
            "--train-lmdb", "train_qqq_9dte_fill.lmdb",
            "--val-lmdbs", "val_qqq_9dte_fill_jan2026.lmdb",
            "--checkpoint-dir", str(CKPT),
            "--epochs", str(epochs),
            "--device", "auto",
        ], check=True, stdout=lf, stderr=subprocess.STDOUT)


def step_eval_replay() -> dict:
    from qqq_btc.tools.eval_9dte_legacy_replay import run_month
    from qqq_btc.tools.eval_test_set import merge_1m_5m, _feat_names_by_res, drop_embedded_exec_columns, label_metrics
    from qqq_btc.tools.run_inference import load_model, run_inference_df
    from qqq_btc.qqq import config_9dte_legacy as cfg9
    from qqq_btc.common.replay_harness import run_strict_replay
    from dataclasses import replace

    cfg = json.loads(FEATURE_CFG.read_text())
    sym = json.loads(SYM_MAP.read_text())["QQQ"]
    _, feats_5m = _feat_names_by_res(cfg)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(CKPT / "best.pth", cfg, dev)

    def infer_ic(month: str) -> dict:
        root = Path.home() / "train_data/quote_features_val_9dte_legacy/QQQ/regular/09:30-16:00"
        df = merge_1m_5m(root / "1min" / f"{month}.parquet", root / "5min" / f"{month}.parquet", feats_5m)
        pred = run_inference_df(df, model, cfg, stock_id=int(sym["stock_id"]), sector_id=int(sym["sector_id"]), device=dev, use_carryover=True)
        return label_metrics(drop_embedded_exec_columns(pred))

    # fill 标签模型:阈值用 0DTE 量级网格
    thresholds = [0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05]
    quantiles = [None, 0.80, 0.85, 0.90]
    val_root = Path.home() / "train_data/quote_features_val_9dte_legacy/QQQ/regular/09:30-16:00"
    df_val = merge_1m_5m(val_root / "1min/2026-01.parquet", val_root / "5min/2026-01.parquet", feats_5m)
    pred_val = drop_embedded_exec_columns(run_inference_df(
        df_val, model, cfg, stock_id=int(sym["stock_id"]), sector_id=int(sym["sector_id"]), device=dev, use_carryover=True,
    ))
    best = None
    grid = []
    for th in thresholds:
        for q in quantiles:
            rep = replace(cfg9.REPLAY, entry_threshold=th, entry_quantile=q,
                            entry_threshold_schedule=((15, th), (270, th * 1.2), (330, th * 1.5)))
            r = run_strict_replay(pred_val, cfg9.FILL_MODEL, rep, cfg9.EXIT_RAILS,
                                  edge_col="net_edge", edge_q10_col=cfg9.EDGE_Q10_COL,
                                  call_edge_col=cfg9.CALL_EDGE_COL, put_edge_col=cfg9.PUT_EDGE_COL,
                                  put_gate_col=cfg9.PUT_GATE_COL)
            f = rep.position_frac
            eq = 1.0
            for t in r.trades:
                eq *= 1.0 + f * t.net_return
            ret = (eq - 1) * 100
            row = {"th": th, "q": q, "val_return_pct": ret, "trades": len(r.trades)}
            grid.append(row)
            if best is None or ret > best["val_return_pct"]:
                best = row

    rep_cfg = replace(cfg9.REPLAY, entry_threshold=best["th"], entry_quantile=best["q"],
                      entry_threshold_schedule=((15, best["th"]), (270, best["th"] * 1.2), (330, best["th"] * 1.5)))

    out_dir = Path("/tmp/qqq_btc_eval_9dte_fill")
    # temporarily patch replay for eval script - run replay manually on test
    df_test = merge_1m_5m(val_root / "1min/2026-02.parquet", val_root / "5min/2026-02.parquet", feats_5m)
    pred_test = drop_embedded_exec_columns(run_inference_df(
        df_test, model, cfg, stock_id=int(sym["stock_id"]), sector_id=int(sym["sector_id"]), device=dev, use_carryover=True,
    ))
    from qqq_btc.tools.eval_test_set import attach_exec_quotes
    pred_test = attach_exec_quotes(pred_test, cfg9.RAW_1M_ROOT, "QQQ",
                                   call_bucket=cfg9.TRADE_BUCKET_ID, put_bucket=cfg9.PUT_BUCKET_ID)
    r_test = run_strict_replay(pred_test, cfg9.FILL_MODEL, rep_cfg, cfg9.EXIT_RAILS,
                               edge_col="net_edge", edge_q10_col=cfg9.EDGE_Q10_COL,
                               call_edge_col=cfg9.CALL_EDGE_COL, put_edge_col=cfg9.PUT_EDGE_COL,
                               put_gate_col=cfg9.PUT_GATE_COL)
    f = rep_cfg.position_frac
    eq = 1.0
    for t in r_test.trades:
        eq *= 1.0 + f * t.net_return
    test_ret = (eq - 1) * 100

    ck = torch.load(CKPT / "best.pth", map_location="cpu", weights_only=False)
    report = {
        "checkpoint": str(CKPT / "best.pth"),
        "best_val_ic_train": float(ck.get("best_ic", 0)),
        "infer_ic_val_jan": infer_ic("2026-01"),
        "infer_ic_test_feb": infer_ic("2026-02"),
        "threshold_grid_val": grid,
        "best_threshold": best,
        "test_replay_return_pct": test_ret,
        "test_replay_summary": r_test.summary(),
    }
    REPORT.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", default="all", choices=["all", "labels", "lmdb", "train", "eval"])
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    steps = {
        "labels": step_labels,
        "lmdb": step_lmdb,
        "train": lambda: step_train(args.epochs),
        "eval": step_eval_replay,
    }
    order = ["labels", "lmdb", "train", "eval"] if args.step == "all" else [args.step]
    result = None
    for s in order:
        result = steps[s]() if s == "eval" else steps[s]()

    if result:
        print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()

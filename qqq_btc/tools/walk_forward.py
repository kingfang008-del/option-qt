#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
双周滚动 walk-forward:切数据 → 建 LMDB → 重训 → 交易窗推理+回放 → 汇总。

动机:0DTE 信号月内衰减明显(六月实验:训练到 2025-12 的模型 -27%,
训练到 2026-04 的模型 -3%,且月初强月末弱)。每个交易窗都用离它最近的
数据重训,是把这个衰减压到最小的机制化方案。

每折:
  trade  = [t0, t0 + step_days)
  val    = [t0 - val_days, t0)          # 早停/选权重
  train  = [源数据起点, t0 - val_days)   # 全历史

用法:
  python qqq_btc/tools/walk_forward.py \
      --trade-start 2026-04-01 --trade-end 2026-06-30 \
      --step-days 14 --val-days 14 \
      --workdir /tmp/qqq_wf_biweekly

源特征目录要求:已跑完 label_pipeline(期权 fill 标签)与 rolling_norm 的
月度 parquet(默认取 quote_features_{train,val,test} 三个旧 stage 的并集)。
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parent.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from qqq_btc.common.replay_harness import run_strict_replay
from qqq_btc.qqq import config as qcfg

logger = logging.getLogger("qqq_btc.walk_forward")

PYTHON = sys.executable
SESSION_REL = Path("QQQ/regular/09:30-16:00")


@dataclass(frozen=True)
class Fold:
    idx: int
    train_end: pd.Timestamp   # 不含
    val_start: pd.Timestamp
    val_end: pd.Timestamp     # 不含
    trade_start: pd.Timestamp
    trade_end: pd.Timestamp   # 不含

    @property
    def name(self) -> str:
        return f"fold{self.idx:02d}_{self.trade_start:%Y%m%d}"


def build_folds(
    trade_start: pd.Timestamp,
    trade_end: pd.Timestamp,
    step_days: int,
    val_days: int,
) -> list[Fold]:
    folds = []
    t0 = trade_start
    i = 0
    while t0 <= trade_end:
        t1 = min(t0 + pd.Timedelta(days=step_days), trade_end + pd.Timedelta(days=1))
        v0 = t0 - pd.Timedelta(days=val_days)
        folds.append(
            Fold(
                idx=i,
                train_end=v0,
                val_start=v0,
                val_end=t0,
                trade_start=t0,
                trade_end=t1,
            )
        )
        t0 = t1
        i += 1
    return folds


def collect_source_files(source_roots: list[Path], res: str) -> list[Path]:
    """各 stage 目录同名月份文件只取第一个命中(stage 间月份本应互斥)。"""
    seen: dict[str, Path] = {}
    for root in source_roots:
        d = root / SESSION_REL / res
        if not d.is_dir():
            continue
        for f in sorted(d.glob("*.parquet")):
            seen.setdefault(f.stem, f)
    return [seen[k] for k in sorted(seen)]


def slice_stage(
    files_by_res: dict[str, list[Path]],
    out_root: Path,
    start: pd.Timestamp | None,
    end: pd.Timestamp,  # 不含
) -> int:
    """把月度文件按 [start, end) 的纽约日期切片写入 out_root。返回总行数。"""
    total = 0
    tz = "America/New_York"
    lo = start.tz_localize(tz) if start is not None else None
    hi = end.tz_localize(tz)
    for res, files in files_by_res.items():
        out_dir = out_root / SESSION_REL / res
        out_dir.mkdir(parents=True, exist_ok=True)
        for f in files:
            month_start = pd.Timestamp(f.stem + "-01", tz=tz)
            month_end = month_start + pd.offsets.MonthBegin(1)
            if month_end <= (lo or month_start) or month_start >= hi:
                continue
            df = pd.read_parquet(f)
            ts = df["timestamp"]
            mask = ts < hi
            if lo is not None:
                mask &= ts >= lo
            sub = df[mask]
            if sub.empty:
                continue
            sub.to_parquet(out_dir / f.name, index=False)
            total += len(sub) if res == "1min" else 0
    return total


def run_cmd(cmd: list[str], log_path: Path) -> None:
    with open(log_path, "a", encoding="utf-8") as f:
        f.write("\n$ " + " ".join(cmd) + "\n")
        f.flush()
        subprocess.run(cmd, check=True, stdout=f, stderr=subprocess.STDOUT, cwd=str(_REPO))


def infer_trade_window(
    fold_dir: Path,
    checkpoint: Path,
    config_path: Path,
    symbol_map_path: Path,
    option_1m_root: Path,
    device: str,
) -> pd.DataFrame:
    """对交易窗做推理,补 exec 盘口,返回带预测列的分钟帧。"""
    import torch
    from qqq_btc.tools.eval_test_set import (
        _feat_names_by_res,
        attach_exec_quotes,
        merge_1m_5m,
    )
    from qqq_btc.tools.run_inference import load_model, run_inference_df

    dev = torch.device(
        device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    with open(symbol_map_path, "r", encoding="utf-8") as f:
        sym = json.load(f)["QQQ"]

    _, feats_5m = _feat_names_by_res(config)
    model = load_model(checkpoint, config, dev)

    root = fold_dir / "trade" / SESSION_REL
    parts = []
    for f1 in sorted((root / "1min").glob("*.parquet")):
        df = merge_1m_5m(f1, root / "5min" / f1.name, feats_5m)
        pred = run_inference_df(
            df,
            model,
            config,
            stock_id=int(sym["stock_id"]),
            sector_id=int(sym["sector_id"]),
            device=dev,
            use_carryover=True,
        )
        if not {"exec_call_bid", "exec_call_ask"}.issubset(pred.columns):
            pred = attach_exec_quotes(
                pred, option_1m_root, "QQQ",
                call_bucket=qcfg.TRADE_BUCKET_ID, put_bucket=0,
            )
        parts.append(pred)
    out = pd.concat(parts, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    ts = pd.to_datetime(out["timestamp"])
    out["timestamp"] = (
        ts.dt.tz_localize("America/New_York") if ts.dt.tz is None
        else ts.dt.tz_convert("America/New_York")
    )
    return out


def summarize_trades(trades: pd.DataFrame, position_frac: float) -> dict:
    if trades.empty:
        return {"trades": 0}
    rets = trades["net_return"].to_numpy()
    eq = np.cumprod(1.0 + position_frac * rets)
    peak = np.maximum.accumulate(eq)
    wins, losses = rets[rets > 0], rets[rets < 0]
    return {
        "trades": int(len(rets)),
        "position_frac": position_frac,
        "total_net_return": float(eq[-1] - 1.0),
        "avg_net_return": float(rets.mean()),
        "sum_net_return": float(rets.sum()),
        "hit_rate": float((rets > 0).mean()),
        "profit_factor": float(wins.sum() / -losses.sum()) if losses.size else float("inf"),
        "max_drawdown": float(((eq - peak) / peak).min()),
        "worst_trade": float(rets.min()),
        "exit_reasons": trades["exit_reason"].value_counts().to_dict(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="qqq_btc 双周滚动 walk-forward")
    parser.add_argument("--trade-start", required=True)
    parser.add_argument("--trade-end", required=True)
    parser.add_argument("--step-days", type=int, default=14)
    parser.add_argument("--val-days", type=int, default=14)
    parser.add_argument(
        "--source-roots",
        default="~/train_data/quote_features_train,"
        "~/train_data/quote_features_val,~/train_data/quote_features_test",
        help="逗号分隔;已带标签+归一化的月度特征根目录",
    )
    parser.add_argument("--workdir", default="/tmp/qqq_wf_biweekly")
    parser.add_argument("--config", default="qqq_btc/CONFIG/slow_feature_qqq_v2.json")
    parser.add_argument("--symbol-map", default="qqq_btc/CONFIG/symbol_map.json")
    parser.add_argument("--option-1m-root", default="/mnt/s990/data/raw_1m/options_databento")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--keep-fold-data", action="store_true", help="保留每折切片数据")
    parser.add_argument(
        "--init-checkpoint", default=None,
        help="第一折的 warm-start 权重;后续折自动接上一折 best.pth",
    )
    parser.add_argument(
        "--cold-start", action="store_true",
        help="每折从零训练(默认 warm-start:折间续训,降低单折退化风险)",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    source_roots = [Path(p).expanduser() for p in args.source_roots.split(",")]
    files_by_res = {
        "1min": collect_source_files(source_roots, "1min"),
        "5min": collect_source_files(source_roots, "5min"),
    }
    if not files_by_res["1min"]:
        raise SystemExit("源目录没有 1min 特征文件")
    logger.info(
        "源月份: %s ~ %s (%d 个月)",
        files_by_res["1min"][0].stem, files_by_res["1min"][-1].stem,
        len(files_by_res["1min"]),
    )

    folds = build_folds(
        pd.Timestamp(args.trade_start), pd.Timestamp(args.trade_end),
        args.step_days, args.val_days,
    )
    workdir = Path(args.workdir).expanduser()
    workdir.mkdir(parents=True, exist_ok=True)
    config_path = _REPO / args.config
    symbol_map_path = _REPO / args.symbol_map

    all_trades: list[pd.DataFrame] = []
    fold_reports: list[dict] = []
    prev_ckpt: Path | None = (
        Path(args.init_checkpoint).expanduser() if args.init_checkpoint else None
    )

    for fold in folds:
        fold_dir = workdir / fold.name
        log_path = fold_dir / "fold.log"
        logger.info(
            "=== %s | train<%s val=[%s,%s) trade=[%s,%s) ===",
            fold.name, fold.train_end.date(),
            fold.val_start.date(), fold.val_end.date(),
            fold.trade_start.date(), fold.trade_end.date(),
        )
        infer_path = fold_dir / "trade_infer.parquet"
        if not infer_path.exists():
            if fold_dir.exists():
                shutil.rmtree(fold_dir)
            fold_dir.mkdir(parents=True)

            n_train = slice_stage(files_by_res, fold_dir / "train", None, fold.train_end)
            n_val = slice_stage(files_by_res, fold_dir / "val", fold.val_start, fold.val_end)
            n_trade = slice_stage(files_by_res, fold_dir / "trade", fold.trade_start, fold.trade_end)
            logger.info("rows train=%d val=%d trade=%d", n_train, n_val, n_trade)
            if n_val == 0 or n_trade == 0:
                logger.warning("%s: val/trade 无数据,跳过", fold.name)
                shutil.rmtree(fold_dir)
                continue

            for stage in ("train", "val"):
                run_cmd(
                    [
                        PYTHON, str(_REPO / "qqq_btc/tools/build_lmdb.py"),
                        "--feature-root", str(fold_dir / stage),
                        "--config", str(config_path),
                        "--symbol-map", str(symbol_map_path),
                        "--output", str(fold_dir / f"{stage}.lmdb"),
                        "--symbols", "QQQ",
                    ],
                    log_path,
                )
            train_cmd = [
                PYTHON, "-m", "qqq_btc.model.train",
                "--mode", "pretrain",
                "--config", str(config_path),
                "--data-root", str(fold_dir),
                "--train-lmdb", "train.lmdb",
                "--val-lmdbs", "val.lmdb",
                "--checkpoint-dir", str(fold_dir / "ckpt"),
                "--epochs", str(args.epochs),
            ]
            if not args.cold_start and prev_ckpt is not None and prev_ckpt.exists():
                train_cmd += ["--init-checkpoint", str(prev_ckpt)]
            run_cmd(train_cmd, log_path)

            infer = infer_trade_window(
                fold_dir, fold_dir / "ckpt" / "best.pth",
                config_path, symbol_map_path,
                Path(args.option_1m_root), args.device,
            )
            infer.to_parquet(infer_path, index=False)
            if not args.keep_fold_data:
                for stage in ("train", "val", "trade"):
                    shutil.rmtree(fold_dir / stage, ignore_errors=True)
                for stage in ("train", "val"):
                    shutil.rmtree(fold_dir / f"{stage}.lmdb", ignore_errors=True)
        else:
            logger.info("%s: 复用已有推理结果", fold.name)
            infer = pd.read_parquet(infer_path)
            infer["timestamp"] = pd.to_datetime(infer["timestamp"]).dt.tz_convert(
                "America/New_York"
            )
        if (fold_dir / "ckpt" / "best.pth").exists():
            prev_ckpt = fold_dir / "ckpt" / "best.pth"

        result = run_strict_replay(
            infer, qcfg.FILL_MODEL, qcfg.REPLAY, qcfg.EXIT_RAILS,
            edge_col="net_edge", edge_q10_col=qcfg.EDGE_Q10_COL,
            call_edge_col=qcfg.CALL_EDGE_COL,
            put_edge_col=qcfg.PUT_EDGE_COL,
            put_gate_col=qcfg.PUT_GATE_COL,
        )
        trades = result.trades_frame()
        if not trades.empty:
            trades["fold"] = fold.name
            all_trades.append(trades)
        fold_summary = summarize_trades(
            trades if not trades.empty else pd.DataFrame(),
            qcfg.REPLAY.position_frac,
        )
        fold_summary["fold"] = fold.name
        fold_summary["trade_window"] = f"{fold.trade_start.date()}~{fold.trade_end.date()}"
        fold_reports.append(fold_summary)
        logger.info("%s: %s", fold.name, fold_summary)

    combined = (
        pd.concat(all_trades, ignore_index=True).sort_values("entry_ts")
        if all_trades else pd.DataFrame()
    )
    overall = summarize_trades(combined, qcfg.REPLAY.position_frac)
    report = {
        "step_days": args.step_days,
        "val_days": args.val_days,
        "position_frac": qcfg.REPLAY.position_frac,
        "overall": overall,
        "folds": fold_reports,
    }
    with open(workdir / "walk_forward_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    if not combined.empty:
        combined.to_parquet(workdir / "walk_forward_trades.parquet", index=False)
    logger.info("overall: %s", overall)
    logger.info("report -> %s", workdir / "walk_forward_report.json")


if __name__ == "__main__":
    main()

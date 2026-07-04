#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
qqq_btc 标签管线 —— feature_merge 产出物 → 双腿 fill 价标签。

在 rolling_norm 之前运行(time/trend 为 calc=raw,由本脚本补算)。

用法:
  python qqq_btc/tools/label_pipeline.py \\
      --input ~/train_data/quote_features_merged/QQQ/regular/2022-03-01_2025-06-30/1min \\
      --symbol QQQ \\
      --anchor-config qqq_btc/CONFIG/anchor_qqq_0dte.json \\
      --output ~/train_data/quote_features_qqq_v2/QQQ/regular/2022-03-01_2025-06-30/1min

对目录下所有 *.parquet 逐文件处理;也可 --input 指向单个文件。
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

_REPO = Path(__file__).resolve().parent.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from qqq_btc.common.labels import build_dual_leg_net_labels, label_quality_report
from qqq_btc.common.time_features import add_time_features
from qqq_btc.common.trend_features import add_trend_features
from qqq_btc.qqq import anchor
from qqq_btc.qqq import config as qcfg

logger = logging.getLogger("qqq_btc.label_pipeline")

LABEL_COLS = [
    "label_return_fwd_net",
    "label_return_fwd_gross",
    "label_execution_cost",
    "label_direction_net",
    "label_net_valid",
    "label_call_return_fwd_net",
    "label_put_return_fwd_net",
    "label_put_net_valid",
    "label_straddle_return_fwd_net",
    "label_straddle_valid",
]


def process_dataframe(df: pd.DataFrame, symbol: str, anchor_cfg: dict) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    if "timestamp" not in df.columns:
        raise ValueError("输入缺少 timestamp 列")
    df = df.sort_values("timestamp").reset_index(drop=True)
    # 允许原地重跑:先清掉旧 exec_* / 旧标签列
    drop_cols = [
        c for c in df.columns
        if c.startswith("exec_") or c.startswith("label_")
    ]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    df = anchor.merge_dual_leg_exec_quotes(df, symbol, anchor_cfg)
    # 缺腿时补 NaN 列,由 label_net_valid 标记无效行(不整文件失败)
    for leg in ("call", "put"):
        for side in ("bid", "ask", "mid"):
            col = f"exec_{leg}_{side}"
            if col not in df.columns:
                df[col] = float("nan")
        mid_col = f"exec_{leg}_mid"
        bid_col, ask_col = f"exec_{leg}_bid", f"exec_{leg}_ask"
        need_mid = df[mid_col].isna() | (df[mid_col] <= 0)
        if need_mid.any():
            mid = (pd.to_numeric(df[bid_col], errors="coerce") + pd.to_numeric(df[ask_col], errors="coerce")) / 2.0
            df.loc[need_mid, mid_col] = mid[need_mid]

    n_call = int((pd.to_numeric(df["exec_call_bid"], errors="coerce") > 0).sum())
    n_put = int((pd.to_numeric(df["exec_put_bid"], errors="coerce") > 0).sum())
    if n_call == 0 and n_put == 0:
        raise ValueError(
            "双腿报价均为空。检查 quote_options_day_iv / locked map 与 anchor paths。"
        )
    logger.info("exec quotes: call_rows=%d put_rows=%d / %d", n_call, n_put, len(df))

    # time/trend 已在 feature_merge 写入时可跳过重算;缺列时补算
    if "time_session_sin" not in df.columns:
        df = add_time_features(df)
    price_col = "close" if "close" in df.columns else None
    if price_col is None:
        for c in ("price", "vwap"):
            if c in df.columns:
                price_col = c
                break
    if price_col is None:
        raise ValueError("缺 close/price 列,无法计算 trend 特征")
    if "trend_fit_ret_30m" not in df.columns:
        df = add_trend_features(df, price_col=price_col)
    df = build_dual_leg_net_labels(df, qcfg.FILL_MODEL, qcfg.LABEL_HORIZON)
    return df


def process_file(src: Path, dst: Path, symbol: str, anchor_cfg: dict) -> dict:
    df = pd.read_parquet(src)
    out = process_dataframe(df, symbol, anchor_cfg)
    dst.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(dst, index=False)
    rep = label_quality_report(out)
    rep["file"] = src.name
    return rep


def main() -> None:
    parser = argparse.ArgumentParser(description="qqq_btc 双腿 fill 价标签管线")
    parser.add_argument("--input", required=True, help="parquet 文件或目录")
    parser.add_argument("--output", required=True, help="输出文件或目录")
    parser.add_argument("--symbol", default="QQQ")
    parser.add_argument("--anchor-config", default=None)
    parser.add_argument("--report", default=None, help="汇总 JSON 报告路径")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg_path = Path(args.anchor_config) if args.anchor_config else anchor.ANCHOR_CONFIG_PATH
    anchor_cfg = anchor.load_anchor_config(cfg_path)

    src = Path(args.input).expanduser()
    dst = Path(args.output).expanduser()
    reports = []

    if src.is_file():
        reports.append(process_file(src, dst, args.symbol, anchor_cfg))
    else:
        files = sorted(src.glob("**/*.parquet"))
        if not files:
            raise FileNotFoundError(f"未找到 parquet: {src}")
        for fp in files:
            rel = fp.relative_to(src)
            out_fp = dst / rel
            logger.info("processing %s", fp)
            reports.append(process_file(fp, out_fp, args.symbol, anchor_cfg))

    summary = {
        "files": len(reports),
        "avg_net_std": float(sum(r["net_std"] for r in reports) / max(1, len(reports))),
        "avg_valid_rows": float(sum(r["valid_rows"] for r in reports) / max(1, len(reports))),
        "details": reports,
    }
    logger.info("done: %d files, avg net_std=%.6f", summary["files"], summary["avg_net_std"])
    if args.report:
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()

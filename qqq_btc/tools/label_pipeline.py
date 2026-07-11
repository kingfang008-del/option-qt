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

from qqq_btc.common.labels import (
    LabelHorizon,
    build_dynamic_ladder_net_labels_subminute,
    build_dual_leg_net_labels,
    build_dual_leg_net_labels_subminute,
    label_quality_report,
)
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
    "label_best_bucket_id",
    "label_best_side",
]


def process_dataframe(
    df: pd.DataFrame,
    symbol: str,
    anchor_cfg: dict,
    *,
    horizon: LabelHorizon | None = None,
) -> pd.DataFrame:
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
    hz = horizon or qcfg.LABEL_HORIZON
    if hz.entry_delay_seconds is not None and bool(anchor_cfg.get("dynamic_ladder_label", False)):
        raw_1s_value = str(anchor_cfg.get("paths", {}).get("sniper_option_dir", "")).strip()
        raw_1s_dir = Path(raw_1s_value).expanduser() if raw_1s_value else None
        put_buckets = tuple(int(x) for x in anchor_cfg.get("dynamic_put_buckets", [0, 1, 2, 3]))
        call_buckets = tuple(int(x) for x in anchor_cfg.get("dynamic_call_buckets", [4, 5, 6, 7]))
        quotes_by_bucket = {}
        for bucket_id in put_buckets + call_buckets:
            q = anchor.load_bucket_second_quotes(
                symbol,
                df["timestamp"],
                int(bucket_id),
                anchor_cfg,
                prefix=f"exec_b{int(bucket_id)}",
                raw_1s_dir=raw_1s_dir,
            )
            if q.empty:
                raise ValueError(f"1s 报价为空,无法构建 dynamic ladder 标签: bucket={bucket_id}")
            quotes_by_bucket[int(bucket_id)] = q
        df = build_dynamic_ladder_net_labels_subminute(
            df,
            qcfg.FILL_MODEL,
            hz,
            quotes_by_bucket=quotes_by_bucket,
            put_buckets=put_buckets,
            call_buckets=call_buckets,
            selection_mode=str(anchor_cfg.get("dynamic_ladder_selection", "oracle")),
        )
    elif hz.entry_delay_seconds is not None:
        legs = anchor_cfg.get("dual_leg_buckets") or anchor.DUAL_LEG_BUCKETS
        raw_1s_value = str(anchor_cfg.get("paths", {}).get("sniper_option_dir", "")).strip()
        raw_1s_dir = Path(raw_1s_value).expanduser() if raw_1s_value else None
        call_q = anchor.load_bucket_second_quotes(
            symbol, df["timestamp"], int(legs["exec_call"]), anchor_cfg, prefix="exec_call", raw_1s_dir=raw_1s_dir
        )
        put_q = anchor.load_bucket_second_quotes(
            symbol, df["timestamp"], int(legs["exec_put"]), anchor_cfg, prefix="exec_put", raw_1s_dir=raw_1s_dir
        )
        if call_q.empty or put_q.empty:
            raise ValueError("1s 报价为空,无法构建子分钟标签")
        df = build_dual_leg_net_labels_subminute(
            df, qcfg.FILL_MODEL, hz, call_quotes_1s=call_q, put_quotes_1s=put_q
        )
    else:
        df = build_dual_leg_net_labels(df, qcfg.FILL_MODEL, hz)
    return df


def process_file(
    src: Path,
    dst: Path,
    symbol: str,
    anchor_cfg: dict,
    *,
    horizon: LabelHorizon | None = None,
) -> dict:
    df = pd.read_parquet(src)
    out = process_dataframe(df, symbol, anchor_cfg, horizon=horizon)
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
    parser.add_argument(
        "--entry-delay-seconds",
        type=int,
        default=None,
        help="子分钟入场延迟(如 30);设置后用 databento 1s 报价构建标签",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg_path = Path(args.anchor_config) if args.anchor_config else anchor.ANCHOR_CONFIG_PATH
    anchor_cfg = anchor.load_anchor_config(cfg_path)
    horizon = qcfg.LABEL_HORIZON
    if args.entry_delay_seconds is not None:
        horizon = LabelHorizon(
            entry_delay_bars=0,
            entry_delay_seconds=int(args.entry_delay_seconds),
            hold_bars=qcfg.LABEL_HORIZON.hold_bars,
            flat_margin=qcfg.LABEL_HORIZON.flat_margin,
            min_entry_premium=qcfg.LABEL_HORIZON.min_entry_premium,
            net_clip=qcfg.LABEL_HORIZON.net_clip,
            signal_offset_seconds=qcfg.LABEL_HORIZON.signal_offset_seconds,
        )
        logger.info("subminute labels: entry_delay_seconds=%d", args.entry_delay_seconds)

    src = Path(args.input).expanduser()
    dst = Path(args.output).expanduser()
    reports = []

    if src.is_file():
        reports.append(process_file(src, dst, args.symbol, anchor_cfg, horizon=horizon))
    else:
        files = sorted(src.glob("**/*.parquet"))
        if not files:
            raise FileNotFoundError(f"未找到 parquet: {src}")
        for fp in files:
            rel = fp.relative_to(src)
            out_fp = dst / rel
            logger.info("processing %s", fp)
            reports.append(process_file(fp, out_fp, args.symbol, anchor_cfg, horizon=horizon))

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

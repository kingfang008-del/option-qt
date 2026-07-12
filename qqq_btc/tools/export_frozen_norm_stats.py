#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
从 quote_features_{stage} 原始特征导出 FCS 冻结归一化统计量(mean/std)。

与 preprocess/ask_bid/apply_rolling_norm_standalone.py 同口径:
  - window=2000, min_periods=100
  - fat-tail signed log1p 预处理
  - 各 stage 独立滚动(不跨 train→val 泄漏)

用法:
  python qqq_btc/tools/export_frozen_norm_stats.py \\
    --symbol QQQ \\
    --stage test \\
    --upto-month 2026-05 \\
    --output qqq_btc/CONFIG/frozen_norm_qqq_test_upto202605.npz

对拍 2026-06 日时,用 upto-month=2026-05 导出 end-of-May 统计量,
FCS 设置 FCS_FROZEN_NORM_PATH 指向该文件。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[2]
_DEFAULT_SLOW_CONFIG = _REPO / "qqq_btc" / "CONFIG" / "slow_feature_qqq_v2.json"
_DEFAULT_FAST_CONFIG = Path.home() / "quant_project/config/fast_feature.json"

ROLLING_WINDOW = 2000
MIN_PERIODS = 100
FAT_TAIL_FEATURES = frozenset(
    {"options_iv_momentum", "options_gamma_accel", "options_iv_divergence"}
)
BOUNDED_FEATURES = {
    "session",
    "day_of_week",
    "hour",
    "is_holiday",
    "rsi_divergence",
    "rsi",
    "k",
    "d",
    "adx",
    "rvi",
    "vw_delta",
    "vp_corr_15",
    "minute",
    "is_expiry",
    "is_fed_meeting",
    "stock_id",
    "timestamp",
    "date",
    "symbol",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "fast_vol",
    "spy_roc_5min",
    "qqq_roc_5min",
}


def load_fcs_feature_names(fast_config: Path, slow_config: Path) -> list[str]:
    """与 FCS FeatureComputeServiceV8.all_feat_names 同序。"""
    with open(fast_config, encoding="utf-8") as f:
        fast_infos = json.load(f).get("features", [])
    with open(slow_config, encoding="utf-8") as f:
        slow_infos = json.load(f).get("features", [])
    fast_names = list(dict.fromkeys(x["name"] for x in fast_infos))
    slow_names = list(dict.fromkeys(x["name"] for x in slow_infos))
    return sorted(set(fast_names + slow_names))


def load_fcs_feature_config(fast_config: Path, slow_config: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for path in (fast_config, slow_config):
        with open(path, encoding="utf-8") as f:
            for feat in json.load(f).get("features", []):
                out[feat["name"]] = feat
    return out


def load_target_features(config_path: Path) -> list[str]:
    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)
    target_features: set[str] = set()
    for feat in config.get("features", []):
        name = feat["name"]
        is_real = feat.get("type") == "real"
        is_not_raw = feat.get("calc") != "raw"
        if is_real and is_not_raw and name not in BOUNDED_FEATURES and not name.startswith("label_"):
            target_features.add(name)
    return sorted(target_features)


def load_feature_config_dict(config_path: Path) -> dict[str, dict]:
    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)
    return {f["name"]: f for f in config.get("features", [])}


def build_categorical_mask(feature_names: list[str], feat_config: dict[str, dict]) -> np.ndarray:
    mask = np.zeros(len(feature_names), dtype=bool)
    for i, name in enumerate(feature_names):
        cfg = feat_config.get(name, {})
        should_normalize = True
        if name in BOUNDED_FEATURES:
            should_normalize = False
        if cfg.get("type") == "categorical":
            should_normalize = False
        if cfg.get("calc") == "raw":
            should_normalize = False
        if name.startswith("label_"):
            should_normalize = False
        if not should_normalize:
            mask[i] = True
    return mask


def _feature_dirs(stage: str, symbol: str, *, pre_norm: bool = True) -> tuple[Path, Path | None]:
    leaf = Path("regular") / "09:30-16:00" / "1min"
    data_root = Path.home() / "train_data" / (
        "quote_features_raw" if pre_norm else f"quote_features_{stage}"
    )
    data_dir = data_root / symbol / leaf
    stage_filter_dir = None
    if pre_norm and stage != "raw":
        stage_filter_dir = Path.home() / "train_data" / f"quote_features_{stage}" / symbol / leaf
    return data_dir, stage_filter_dir


def _filter_rows_upto_date(df: pd.DataFrame, upto_date: str | None) -> pd.DataFrame:
    """保留 timestamp 日期 <= upto_date 的行(含当日)。"""
    if not upto_date or "timestamp" not in df.columns:
        return df
    ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    cutoff = pd.Timestamp(upto_date).date()
    return df.loc[ts.dt.tz_convert("America/New_York").dt.date <= cutoff]


def compute_frozen_stats(
    *,
    feature_names: list[str],
    files: list[Path],
    categorical_mask: np.ndarray,
    upto_date: str | None = None,
) -> dict:
    fat_tail_idx = [i for i, n in enumerate(feature_names) if n in FAT_TAIL_FEATURES]
    buffer_df: pd.DataFrame | None = None
    frame_count = 0
    roll_mean = pd.Series(0.0, index=feature_names, dtype=np.float32)
    roll_std = pd.Series(1.0, index=feature_names, dtype=np.float32)

    for file_path in files:
        df = pd.read_parquet(file_path)
        df = _filter_rows_upto_date(df, upto_date)
        if df.empty:
            continue
        cols = [c for c in feature_names if c in df.columns]
        if not cols:
            continue
        block = df[cols].astype(np.float32).copy()
        for name in FAT_TAIL_FEATURES:
            if name in block.columns:
                vals = block[name].values
                block[name] = np.sign(vals) * np.log1p(np.abs(vals))

        if buffer_df is not None:
            combined = pd.concat(
                [buffer_df.reindex(columns=cols), block],
                axis=0,
                ignore_index=True,
            )
        else:
            combined = block

        if len(combined) >= MIN_PERIODS:
            tail = combined.iloc[-ROLLING_WINDOW:]
            roll_mean = tail.mean(axis=0)
            roll_std = tail.std(axis=0)
            roll_std = roll_std.mask(roll_std < 1e-6, 1.0)

        full_mean = np.zeros(len(feature_names), dtype=np.float32)
        full_std = np.ones(len(feature_names), dtype=np.float32)
        for j, name in enumerate(feature_names):
            if name in cols:
                if len(combined) >= MIN_PERIODS:
                    full_mean[j] = float(roll_mean[name])
                    full_std[j] = float(roll_std[name])
        full_mean[categorical_mask] = 0.0
        full_std[categorical_mask] = 1.0

        frame_count += len(df)
        if len(combined) > ROLLING_WINDOW:
            buffer_df = combined.iloc[-ROLLING_WINDOW:].copy()
        else:
            buffer_df = combined.copy()

    # 序列种子: 末 window 根 bar 的 raw(已 fat-tail 变换),按 feature_names 对齐,
    # 供 FCS raw_buffer 开盘预热(与 RollingWindowNormalizer.process_frame 口径一致)。
    if buffer_df is not None and not buffer_df.empty:
        seed = buffer_df.reindex(columns=feature_names).fillna(0.0)
        buffer_mat = seed.to_numpy(dtype=np.float32)
    else:
        buffer_mat = np.zeros((0, len(feature_names)), dtype=np.float32)

    return {
        "mean": full_mean,
        "std": full_std,
        "count": int(frame_count),
        "fat_tail_idx": np.array(fat_tail_idx, dtype=np.int32),
        "buffer": buffer_mat,
    }


def export_frozen_norm(
    *,
    symbol: str,
    stage: str,
    fast_config: Path,
    slow_config: Path,
    output: Path,
    upto_month: str | None = None,
    upto_date: str | None = None,
    pre_norm: bool = True,
) -> Path:
    feat_names = load_fcs_feature_names(fast_config, slow_config)
    feat_config = load_fcs_feature_config(fast_config, slow_config)
    categorical_mask = build_categorical_mask(feat_names, feat_config)

    if upto_date and not upto_month:
        upto_month = str(pd.Timestamp(upto_date).strftime("%Y-%m"))

    feat_dir, stage_filter_dir = _feature_dirs(stage, symbol, pre_norm=pre_norm)
    if not feat_dir.exists():
        raise FileNotFoundError(f"feature dir not found: {feat_dir}")

    files = sorted(feat_dir.glob("*.parquet"))
    if stage_filter_dir is not None and stage_filter_dir.exists():
        allowed = {p.name for p in stage_filter_dir.glob("*.parquet")}
        files = [p for p in files if p.name in allowed]
    if upto_month:
        files = [p for p in files if p.stem <= upto_month]
    if not files:
        raise FileNotFoundError(f"no parquet files under {feat_dir} (upto={upto_month})")

    stats = compute_frozen_stats(
        feature_names=feat_names,
        files=files,
        categorical_mask=categorical_mask,
        upto_date=upto_date,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        feature_names=np.array(feat_names, dtype=object),
        mean=stats["mean"],
        std=stats["std"],
        count=np.int32(stats["count"]),
        categorical_mask=categorical_mask,
        fat_tail_idx=stats["fat_tail_idx"],
        buffer=stats["buffer"],
        window=np.int32(ROLLING_WINDOW),
        use_tanh=np.int8(1),
        symbol=np.array(symbol),
        stage=np.array(stage),
        upto_month=np.array(upto_month or ""),
        upto_date=np.array(upto_date or ""),
        source_dir=np.array(str(feat_dir)),
        fast_config_path=np.array(str(fast_config)),
        slow_config_path=np.array(str(slow_config)),
    )
    print(
        f"[export] {output} | symbol={symbol} stage={stage} dims={len(feat_names)} "
        f"files={len(files)} frames={stats['count']} buffer={stats['buffer'].shape[0]} "
        f"upto={upto_date or upto_month or 'ALL'}"
    )
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="Export frozen rolling norm stats for FCS parity")
    parser.add_argument("--symbol", default="QQQ")
    parser.add_argument("--stage", default="test", choices=("train", "val", "test"))
    parser.add_argument("--slow-config", default=str(_DEFAULT_SLOW_CONFIG))
    parser.add_argument("--fast-config", default=str(_DEFAULT_FAST_CONFIG))
    parser.add_argument(
        "--config",
        default=None,
        help="(deprecated) 同 --slow-config",
    )
    parser.add_argument(
        "--upto-month",
        default=None,
        help="仅处理 YYYY-MM 及之前的月份 parquet(对拍当月日前一月)",
    )
    parser.add_argument(
        "--upto-date",
        default=None,
        help="按日截断: 统计量只用 YYYY-MM-DD 及之前的 bar(每日收盘后刷新用)",
    )
    parser.add_argument(
        "--post-norm",
        action="store_true",
        help="从 quote_features_{stage} 已归一化 parquet 导出(默认用 quote_features_raw 预归一化值)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="输出 .npz 路径(默认 qqq_btc/CONFIG/frozen_norm_{symbol}_{stage}.npz)",
    )
    args = parser.parse_args()

    slow_config = Path(args.slow_config if not args.config else args.config).expanduser()
    fast_config = Path(args.fast_config).expanduser()
    if args.upto_date:
        suffix = f"_upto{args.upto_date.replace('-', '')}"
    elif args.upto_month:
        suffix = f"_upto{args.upto_month.replace('-', '')}"
    else:
        suffix = ""
    output = (
        Path(args.output).expanduser()
        if args.output
        else _REPO / "qqq_btc" / "CONFIG" / f"frozen_norm_{args.symbol.lower()}_{args.stage}{suffix}.npz"
    )
    export_frozen_norm(
        symbol=args.symbol,
        stage=args.stage,
        fast_config=fast_config,
        slow_config=slow_config,
        output=output,
        upto_month=args.upto_month,
        upto_date=args.upto_date,
        pre_norm=not args.post_norm,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

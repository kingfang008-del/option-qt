#!/usr/bin/env python3
"""为 V4 模型准备因果日频 VX 特征及消融配置。

输入应为尚未包含 VX 的冻结 quote_features。脚本只复制并改写 1min
parquet；其他文件使用硬链接，避免污染基线和重复占用磁盘。
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


VX_FEATURES = (
    "vx_curve_slope",
    "vx_cm30_level_z63",
    "vx_curve_slope_z63",
)


def _feature_spec(name: str) -> dict[str, str]:
    return {
        "name": name,
        "type": "real",
        "calc": "raw",
        "resolution": "1min",
        "description": "前一已完成 Databento VX UTC 日桶；按 NY 交易日广播",
    }


def write_configs(base_path: Path, output_dir: Path) -> tuple[Path, Path]:
    base = json.loads(base_path.read_text(encoding="utf-8"))
    base_features = [
        feature for feature in base["features"] if feature["name"] not in VX_FEATURES
    ]

    hybrid = dict(base)
    hybrid["comment_vx_model"] = (
        "Hybrid: 保留 5min VIXY vix_level，并加入前一完成日桶的 VX 期限结构。"
    )
    hybrid["features"] = base_features + [_feature_spec(name) for name in VX_FEATURES]

    vx_only = dict(base)
    vx_only["comment_vx_model"] = (
        "VX-only 消融: 移除 VIXY vix_level，仅加入前一完成日桶的 VX 期限结构。"
    )
    vx_only["features"] = [
        feature for feature in base_features if feature["name"] != "vix_level"
    ] + [_feature_spec(name) for name in VX_FEATURES]

    output_dir.mkdir(parents=True, exist_ok=True)
    hybrid_path = output_dir / "slow_feature_qqq_v4_hybrid_vx.json"
    vx_only_path = output_dir / "slow_feature_qqq_v4_vx_only.json"
    hybrid_path.write_text(
        json.dumps(hybrid, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    vx_only_path.write_text(
        json.dumps(vx_only, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return hybrid_path, vx_only_path


def load_vx_term(path: Path) -> pd.DataFrame:
    vx = pd.read_parquet(path, columns=["date", *VX_FEATURES]).copy()
    vx["source_date"] = pd.to_datetime(vx.pop("date"), utc=True).dt.date
    vx = vx.sort_values("source_date").drop_duplicates("source_date", keep="last")
    if vx.empty:
        raise ValueError(f"VX term structure 为空: {path}")
    return vx.reset_index(drop=True)


def add_causal_vx_columns(frame: pd.DataFrame, vx: pd.DataFrame) -> pd.DataFrame:
    if "timestamp" not in frame:
        raise ValueError("1min parquet 缺少 timestamp")

    ts = pd.to_datetime(frame["timestamp"])
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize("UTC")
    trading_days = ts.dt.tz_convert("America/New_York").dt.date

    source_dates = np.asarray(vx["source_date"], dtype="datetime64[D]")
    requested = np.asarray(trading_days, dtype="datetime64[D]")
    prior_idx = np.searchsorted(source_dates, requested, side="left") - 1
    if np.any(prior_idx < 0):
        first_missing = trading_days.iloc[int(np.flatnonzero(prior_idx < 0)[0])]
        raise ValueError(
            f"VX 历史覆盖不足：交易日 {first_missing} 之前没有已完成日桶；"
            "请先补下载更早的 VX 数据"
        )

    out = frame.copy()
    for name in VX_FEATURES:
        values = pd.to_numeric(vx[name], errors="coerce").to_numpy(dtype=np.float32)
        mapped = values[prior_idx]
        # z63 在历史开头 min_periods 不足时以中性值 0 表示；原始 slope 不应缺失。
        if name == "vx_curve_slope" and not np.isfinite(mapped).all():
            raise ValueError("vx_curve_slope 存在非有限值，拒绝静默填充")
        out[name] = np.nan_to_num(mapped, nan=0.0, posinf=0.0, neginf=0.0)
    return out


def _link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def prepare_stage(
    source: Path,
    output: Path,
    vx: pd.DataFrame,
    overwrite: bool,
    *,
    start_month: str | None = None,
) -> int:
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"输出已存在，传 --overwrite 才可重建: {output}")
        shutil.rmtree(output)

    parquet_count = 0
    for src in sorted(source.rglob("*")):
        relative = src.relative_to(source)
        dst = output / relative
        if src.is_dir():
            dst.mkdir(parents=True, exist_ok=True)
            continue
        if (
            start_month is not None
            and src.suffix == ".parquet"
            and len(src.stem) >= 7
            and src.stem[:7] < start_month
        ):
            continue
        if src.suffix == ".parquet" and src.parent.name == "1min":
            frame = pd.read_parquet(src)
            patched = add_causal_vx_columns(frame, vx)
            dst.parent.mkdir(parents=True, exist_ok=True)
            patched.to_parquet(dst, index=False)
            parquet_count += 1
        else:
            _link_or_copy(src, dst)
    if parquet_count == 0:
        raise ValueError(f"未在 {source} 找到 1min parquet")
    return parquet_count


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[2]
    default_bak = Path.home() / "train_data" / "_bak_pre4c"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-config",
        type=Path,
        default=repo / "qqq_btc/CONFIG/slow_feature_qqq_v4.json",
    )
    parser.add_argument(
        "--vx-term",
        type=Path,
        default=Path(
            "/mnt/s990/data/raw_1m/vix_futures_databento/vx_term_structure_1d.parquet"
        ),
    )
    parser.add_argument(
        "--source-train",
        type=Path,
        default=default_bak / "quote_features_train_QQQ",
    )
    parser.add_argument(
        "--source-val",
        type=Path,
        default=default_bak / "quote_features_val_QQQ",
    )
    parser.add_argument(
        "--source-test",
        type=Path,
        default=default_bak / "quote_features_test_QQQ",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path.home() / "train_data/builds/v4_vx_model_ablation",
    )
    parser.add_argument(
        "--config-output-dir",
        type=Path,
        default=repo / "qqq_btc/CONFIG",
    )
    parser.add_argument(
        "--train-start-month",
        default=None,
        help="可选 YYYY-MM；跳过更早 train 月份（例如 VX 仅覆盖 2024 起）",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    vx = load_vx_term(args.vx_term.expanduser())
    config_paths = write_configs(
        args.base_config.expanduser(), args.config_output_dir.expanduser()
    )
    total = 0
    for stage in ("train", "val", "test"):
        source = getattr(args, f"source_{stage}").expanduser()
        # build_lmdb 以 <feature-root>/QQQ/... 定位 symbol；冻结快照本身
        # 名为 quote_features_*_QQQ，内部从 regular/ 开始，复制时补回该层。
        stage_root = args.output_root.expanduser() / f"quote_features_{stage}"
        if args.overwrite and stage_root.exists():
            shutil.rmtree(stage_root)
        output = stage_root / "QQQ"
        count = prepare_stage(
            source,
            output,
            vx,
            args.overwrite,
            start_month=args.train_start_month if stage == "train" else None,
        )
        total += count
        print(f"{stage}: wrote {count} monthly 1min parquet -> {output}")
    print(f"configs: {config_paths[0]}, {config_paths[1]}")
    print(f"done: patched_1min_files={total}")


if __name__ == "__main__":
    main()

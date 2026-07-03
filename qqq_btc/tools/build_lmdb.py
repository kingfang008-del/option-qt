#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
从 qqq_btc 标签化后的 parquet 构建 LMDB(兼容 LMDBAlphaDataset)。

无 Postgres 依赖;stock_id/sector_id 来自 CONFIG/symbol_map.json。

前置:label_pipeline.py 已写入双腿标签列;rolling_norm 已跑完(特征列 z-score,
time/trend calc=raw 保持原值)。

用法:
  python qqq_btc/tools/build_lmdb.py \\
      --feature-root ~/train_data/quote_features_qqq_v2_norm \\
      --config qqq_btc/CONFIG/slow_feature_qqq_v2.json \\
      --symbol-map qqq_btc/CONFIG/symbol_map.json \\
      --output ~/train_data/lmdb/train_qqq.lmdb \\
      --symbols QQQ
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import msgpack
import msgpack_numpy
import numpy as np
import pandas as pd
import lmdb
import zstandard as zstd
from tqdm import tqdm

_REPO = Path(__file__).resolve().parent.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

msgpack_numpy.patch()
logger = logging.getLogger("qqq_btc.build_lmdb")

WINDOW_1M = 30
WINDOW_5M = 6
WINDOW_STEP = 5

REQUIRED_LABELS = [
    "label_return_fwd_net",
    "label_return_fwd_gross",
    "label_execution_cost",
    "label_direction_net",
]

OPTIONAL_LABELS = [
    "label_call_return_fwd_net",
    "label_put_return_fwd_net",
    "label_straddle_return_fwd_net",
    "label_net_valid",
    "label_put_net_valid",
    "label_straddle_valid",
]


def load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def feature_columns(config: dict) -> tuple[list[str], list[str]]:
    feats_1m, feats_5m = [], []
    static = {"stock_id", "sector_id", "day_of_week"}
    for f in config.get("features", []):
        name = f["name"]
        if name in static:
            continue
        res = f.get("resolution", "1min")
        if res == "1min":
            feats_1m.append(name)
        elif res == "5min":
            feats_5m.append(name)
        elif res == "both":
            feats_1m.append(name)
            feats_5m.append(name)
    return feats_1m, feats_5m


def session_info(path: Path) -> tuple[str | None, str | None, str | None]:
    parts = path.parts
    if len(parts) < 4:
        return None, None, None
    return parts[-4], parts[-3], parts[-2]


def align_timestamp_ny(df: pd.DataFrame, shift_minutes: int) -> pd.DataFrame:
    df = df.copy()
    ts = pd.to_datetime(df["timestamp"])
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize("UTC")
    ts = ts.dt.tz_convert("America/New_York") + pd.Timedelta(minutes=shift_minutes)
    df["timestamp"] = ts.astype(np.int64)
    return df.sort_values("timestamp")


def to_np(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    for c in cols:
        if c not in df.columns:
            df[c] = 0.0
    return df[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).values.astype(np.float32)


def build_samples_for_pair(
    f_1m: Path,
    f_5m: Path,
    symbol: str,
    stock_id: int,
    sector_id: int,
    feats_1m: list[str],
    feats_5m: list[str],
    label_cols: list[str],
) -> list[tuple[bytes, bytes]]:
    df_1m = pd.read_parquet(f_1m)
    df_5m = pd.read_parquet(f_5m)
    for col in REQUIRED_LABELS:
        if col not in df_1m.columns:
            raise ValueError(f"{f_1m}: 缺必需标签 {col},请先跑 label_pipeline.py")

    df_1m = align_timestamp_ny(df_1m, shift_minutes=1)
    df_5m = align_timestamp_ny(df_5m, shift_minutes=5)
    ts_5m_map = {int(t): i for i, t in enumerate(df_5m["timestamp"].values)}
    ts_1m = df_1m["timestamp"].values.astype(np.int64)

    arr_1m = to_np(df_1m, feats_1m)
    arr_5m = to_np(df_5m, feats_5m)
    active_labels = REQUIRED_LABELS + [c for c in OPTIONAL_LABELS if c in df_1m.columns]
    arr_lbl = to_np(df_1m, active_labels)

    cctx = zstd.ZstdCompressor(level=3)
    out: list[tuple[bytes, bytes]] = []
    start_idx = WINDOW_1M - 1
    for temp_i in range(start_idx, min(start_idx + WINDOW_STEP, len(df_1m))):
        if int(ts_1m[temp_i]) in ts_5m_map:
            start_idx = temp_i
            break
    else:
        return out

    for i in range(start_idx, len(df_1m), WINDOW_STEP):
        t = int(ts_1m[i])
        if t not in ts_5m_map:
            continue
        idx_5 = ts_5m_map[t]
        if idx_5 < WINDOW_5M - 1:
            continue
        idx_start_1m = i - (WINDOW_1M - 1)
        idx_start_5m = idx_5 - (WINDOW_5M - 1)
        if idx_start_1m < 0:
            continue

        sample = {
            "1min": {n: arr_1m[idx_start_1m : i + 1, j].copy() for j, n in enumerate(feats_1m)},
            "5min": {n: arr_5m[idx_start_5m : idx_5 + 1, j].copy() for j, n in enumerate(feats_5m)},
            "labels": {n: float(arr_lbl[i, j]) for j, n in enumerate(active_labels)},
            "metadata": {
                "symbol": symbol,
                "timestamp": t,
                "stock_id": int(stock_id),
                "sector_id": int(sector_id),
            },
        }
        key = f"{symbol}_{t}".encode("ascii")
        out.append((key, cctx.compress(msgpack.packb(sample, use_bin_type=True))))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="qqq_btc LMDB 建库")
    parser.add_argument("--feature-root", required=True)
    parser.add_argument("--config", default="qqq_btc/CONFIG/slow_feature_qqq_v2.json")
    parser.add_argument("--symbol-map", default="qqq_btc/CONFIG/symbol_map.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--symbols", default="QQQ", help="逗号分隔")
    parser.add_argument("--map-size-gb", type=int, default=50)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    root = Path(args.feature_root).expanduser()
    out_path = Path(args.output).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    config = load_config(Path(args.config))
    with open(Path(args.symbol_map), "r", encoding="utf-8") as f:
        sym_map = json.load(f)

    feats_1m, feats_5m = feature_columns(config)
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    all_samples: list[tuple[bytes, bytes]] = []

    for symbol in symbols:
        meta = sym_map.get(symbol)
        if not meta:
            raise KeyError(f"symbol_map 缺 {symbol}")
        stock_id = int(meta["stock_id"])
        sector_id = int(meta["sector_id"])
        files = sorted(root.glob(f"{symbol}/**/1min/*.parquet"))
        if not files:
            files = sorted(root.glob(f"**/{symbol}/**/1min/*.parquet"))
        by_group: dict[tuple, list[Path]] = defaultdict(list)
        for fp in files:
            sess, tr, res = session_info(fp)
            if sess == "regular" and res == "1min":
                by_group[tr].append(fp)

        for _tr, flist in by_group.items():
            for f_1m in tqdm(flist, desc=f"LMDB {symbol}"):
                f_5m = Path(str(f_1m).replace("/1min/", "/5min/"))
                if not f_5m.exists():
                    logger.warning("skip (no 5min): %s", f_1m)
                    continue
                all_samples.extend(
                    build_samples_for_pair(
                        f_1m, f_5m, symbol, stock_id, sector_id, feats_1m, feats_5m,
                        REQUIRED_LABELS + OPTIONAL_LABELS,
                    )
                )

    if not all_samples:
        raise RuntimeError("未生成任何样本,检查 feature-root 路径与标签列")

    env = lmdb.open(
        str(out_path),
        map_size=args.map_size_gb * (1024**3),
        writemap=False,
    )
    keys_meta = []
    with env.begin(write=True) as txn:
        for key, val in tqdm(all_samples, desc="write LMDB"):
            txn.put(key, val)
            keys_meta.append(key)
        cctx = zstd.ZstdCompressor(level=3)
        txn.put(b"__keys__", cctx.compress(msgpack.packb(keys_meta, use_bin_type=True)))
    env.close()
    logger.info("written %s samples -> %s", len(all_samples), out_path)


if __name__ == "__main__":
    main()

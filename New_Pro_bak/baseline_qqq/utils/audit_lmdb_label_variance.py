#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Audit TFT LMDB label variance by timestamp.

The training warning
    "发现 N 个时间点的真实收益率(Return)方差为0"
can be benign when a whole cross-section is FLAT, but it is dangerous if labels
were accidentally filled with zero. This script reads LMDB labels directly and
reports whether zero-variance timestamps are real sparse executable labels or a
data generation problem.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import lmdb
import msgpack
import numpy as np
import pandas as pd
import zstandard as zstd


try:
    import msgpack_numpy

    msgpack_numpy.patch()
except Exception:
    msgpack_numpy = None


LABEL_COLS = [
    "label_return_fwd",
    "label_direction",
    "label_option_alpha_300s",
    "label_option_direction_300s",
    "label_option_best_net_ret_300s",
    "label_option_tradable_300s",
    "label_option_training_enabled",
    "label_option_source_has_alpha",
    "label_option_source_has_direction",
    "label_option_source_has_primary_exec",
    "label_option_source_has_matrix_primary",
    "label_option_source_missing_count",
    "label_call_delayed_exec_ret_300s",
    "label_put_delayed_exec_ret_300s",
    "label_call_delayed_exec_ret_d60s_h300s",
    "label_put_delayed_exec_ret_d60s_h300s",
]


def _key_to_symbol_ts(key: bytes | str) -> tuple[str, int]:
    text = key.decode("ascii") if isinstance(key, bytes) else str(key)
    symbol, ts = text.rsplit("_", 1)
    return symbol, int(ts)


def _lmdb_ts_to_epoch_seconds(ts: int) -> int:
    if abs(ts) > 10**17:
        return int(round(ts / 1_000_000_000))
    if abs(ts) > 10**14:
        return int(round(ts / 1_000_000))
    if abs(ts) > 10**11:
        return int(round(ts / 1_000))
    return int(ts)


def _date_from_epoch(ts: int) -> str:
    return pd.to_datetime(ts, unit="s", utc=True).tz_convert("America/New_York").strftime("%Y%m%d")


def _ny_time_from_epoch(ts: int) -> str:
    return pd.to_datetime(ts, unit="s", utc=True).tz_convert("America/New_York").strftime("%Y-%m-%d %H:%M:%S")


def _as_float(value: Any) -> float:
    try:
        out = float(value)
        return out if np.isfinite(out) else np.nan
    except Exception:
        return np.nan


def read_lmdb_labels(lmdb_path: Path, sample_limit: int = 0, dates: set[str] | None = None) -> pd.DataFrame:
    env = lmdb.open(str(lmdb_path), readonly=True, lock=False, readahead=False, meminit=False)
    dctx = zstd.ZstdDecompressor()
    rows: list[dict[str, Any]] = []
    with env.begin() as txn:
        keys_blob = txn.get(b"__keys__")
        if not keys_blob:
            raise RuntimeError(f"{lmdb_path} missing __keys__")
        keys = msgpack.unpackb(dctx.decompress(keys_blob), raw=False)
        if sample_limit > 0:
            keys = keys[:sample_limit]

        for key in keys:
            symbol, raw_ts = _key_to_symbol_ts(key)
            ts = _lmdb_ts_to_epoch_seconds(raw_ts)
            date = _date_from_epoch(ts)
            if dates and date not in dates:
                continue

            blob = txn.get(key)
            if not blob:
                continue
            sample = msgpack.unpackb(dctx.decompress(blob), raw=False)
            labels = sample.get("labels", {})
            meta = sample.get("metadata", {})
            row: dict[str, Any] = {
                "symbol": str(meta.get("symbol", symbol)),
                "raw_ts": raw_ts,
                "ts": ts,
                "date": date,
                "ny_time": _ny_time_from_epoch(ts),
            }
            for col in LABEL_COLS:
                row[col] = _as_float(labels.get(col, np.nan))
            rows.append(row)
    env.close()
    return pd.DataFrame(rows)


def _summarize_label(df: pd.DataFrame, col: str) -> dict[str, float]:
    s = pd.to_numeric(df[col], errors="coerce")
    return {
        "non_null%": float(s.notna().mean() * 100.0),
        "nonzero%": float((s.abs() > 1e-12).mean() * 100.0),
        "mean": float(s.mean()),
        "std": float(s.std()),
        "p10": float(s.quantile(0.10)),
        "p50": float(s.quantile(0.50)),
        "p90": float(s.quantile(0.90)),
    }


def run_audit(df: pd.DataFrame, zero_eps: float, min_cross_section: int, out_csv: Path | None) -> int:
    if df.empty:
        print("❌ 没有读到任何 LMDB label 样本。")
        return 2

    ret = pd.to_numeric(df["label_return_fwd"], errors="coerce").fillna(0.0)
    df = df.copy()
    df["ret_abs_nonzero"] = ret.abs() > zero_eps
    df["ret_pos"] = ret > zero_eps
    df["ret_neg"] = ret < -zero_eps

    print("=" * 92)
    print("LMDB label variance audit")
    print("=" * 92)
    print(
        f"samples={len(df)} dates={sorted(df['date'].unique())} "
        f"symbols={len(df['symbol'].unique())} timestamps={df['raw_ts'].nunique()}"
    )

    print("\n[核心 label 分布]")
    dist_rows = []
    for col in [
        "label_return_fwd",
        "label_option_alpha_300s",
        "label_option_best_net_ret_300s",
        "label_option_tradable_300s",
        "label_call_delayed_exec_ret_300s",
        "label_put_delayed_exec_ret_300s",
    ]:
        if col in df.columns:
            row = {"label": col}
            row.update(_summarize_label(df, col))
            dist_rows.append(row)
    print(pd.DataFrame(dist_rows).round(6).to_string(index=False))

    print("\n[Source flags]")
    flag_cols = [
        "label_option_training_enabled",
        "label_option_source_has_alpha",
        "label_option_source_has_direction",
        "label_option_source_has_primary_exec",
        "label_option_source_has_matrix_primary",
        "label_option_source_missing_count",
    ]
    flag_rows = []
    for col in flag_cols:
        s = pd.to_numeric(df[col], errors="coerce")
        flag_rows.append(
            {
                "field": col,
                "non_null%": s.notna().mean() * 100.0,
                "mean": s.mean(),
                "min": s.min(),
                "max": s.max(),
            }
        )
    print(pd.DataFrame(flag_rows).round(4).to_string(index=False))

    grp = (
        df.groupby(["date", "raw_ts", "ny_time"], dropna=False)
        .agg(
            n=("symbol", "size"),
            symbols=("symbol", "nunique"),
            ret_std=("label_return_fwd", "std"),
            ret_mean=("label_return_fwd", "mean"),
            nonzero=("ret_abs_nonzero", "sum"),
            pos=("ret_pos", "sum"),
            neg=("ret_neg", "sum"),
            tradable_mean=("label_option_tradable_300s", "mean"),
            best_net_mean=("label_option_best_net_ret_300s", "mean"),
        )
        .reset_index()
    )
    grp["zero_var"] = (grp["ret_std"].fillna(0.0).abs() <= zero_eps) & (grp["n"] >= min_cross_section)
    grp["all_zero"] = grp["zero_var"] & (grp["nonzero"] == 0)
    grp["same_nonzero"] = grp["zero_var"] & (grp["nonzero"] > 0)

    print("\n[按时间截面 label_return_fwd 方差]")
    total_ts = len(grp)
    zero_ts = int(grp["zero_var"].sum())
    all_zero_ts = int(grp["all_zero"].sum())
    same_nonzero_ts = int(grp["same_nonzero"].sum())
    print(
        f"timestamps={total_ts} zero_var={zero_ts} ({zero_ts/max(total_ts,1)*100:.2f}%) "
        f"all_zero={all_zero_ts} same_nonzero={same_nonzero_ts}"
    )
    print(
        "解释: all_zero 多通常表示 executable label 很稀疏或被补 0；"
        "same_nonzero 多则更像映射/复制错误。"
    )

    by_date = (
        grp.groupby("date")
        .agg(
            timestamps=("raw_ts", "size"),
            zero_var=("zero_var", "sum"),
            all_zero=("all_zero", "sum"),
            same_nonzero=("same_nonzero", "sum"),
            avg_symbols=("symbols", "mean"),
            avg_nonzero=("nonzero", "mean"),
        )
        .reset_index()
    )
    by_date["zero_var%"] = by_date["zero_var"] / by_date["timestamps"].clip(lower=1) * 100.0
    by_date["all_zero%"] = by_date["all_zero"] / by_date["timestamps"].clip(lower=1) * 100.0
    print("\n[日期汇总]")
    print(by_date.round(3).to_string(index=False))

    print("\n[zero-var 时间点样本 Top30]")
    show_cols = [
        "date",
        "ny_time",
        "n",
        "symbols",
        "ret_mean",
        "ret_std",
        "nonzero",
        "pos",
        "neg",
        "tradable_mean",
        "best_net_mean",
        "all_zero",
        "same_nonzero",
    ]
    print(grp[grp["zero_var"]].sort_values(["date", "ny_time"]).head(30)[show_cols].round(6).to_string(index=False))

    sym = (
        df.groupby("symbol")
        .agg(
            n=("symbol", "size"),
            nonzero=("ret_abs_nonzero", "mean"),
            pos=("ret_pos", "mean"),
            neg=("ret_neg", "mean"),
            ret_mean=("label_return_fwd", "mean"),
            ret_std=("label_return_fwd", "std"),
            tradable=("label_option_tradable_300s", "mean"),
        )
        .reset_index()
    )
    for col in ["nonzero", "pos", "neg", "tradable"]:
        sym[col] *= 100.0
    print("\n[Symbol label 覆盖 Top/Bottom]")
    print(sym.sort_values("nonzero").head(20).round(4).to_string(index=False))
    print("\n--- nonzero 最高 ---")
    print(sym.sort_values("nonzero", ascending=False).head(20).round(4).to_string(index=False))

    if out_csv:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        grp.to_csv(out_csv, index=False)
        print(f"\n已写出时间截面明细: {out_csv}")

    print("\n[判断]")
    nonzero_ratio = float(df["ret_abs_nonzero"].mean())
    all_zero_ratio = all_zero_ts / max(total_ts, 1)
    same_nonzero_ratio = same_nonzero_ts / max(total_ts, 1)
    if nonzero_ratio < 0.005:
        print("  ❌ label_return_fwd 非零样本低于 0.5%，高度疑似 label 未生成或被补 0。")
        return 1
    if same_nonzero_ratio > 0.05:
        print("  ❌ 大量时间点所有标的同一个非零 label，疑似 label 复制/映射错误。")
        return 1
    if all_zero_ratio > 0.50:
        print("  ⚠️ 超过一半时间截面全 0；若使用 executable label 可能正常，但需要确认 tradable 正样本足够。")
    else:
        print("  ✅ label 方差 warning 更像可交易标签稀疏导致，不像整体数据损坏。")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lmdb", type=Path, required=True)
    parser.add_argument("--dates", type=str, default="", help="Comma-separated NY dates, e.g. 20260302,20260303")
    parser.add_argument("--sample-limit", type=int, default=0, help="0 means all keys")
    parser.add_argument("--zero-eps", type=float, default=1e-12)
    parser.add_argument("--min-cross-section", type=int, default=3)
    parser.add_argument("--out-csv", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dates = {x.strip() for x in args.dates.split(",") if x.strip()} or None
    df = read_lmdb_labels(args.lmdb, sample_limit=args.sample_limit, dates=dates)
    return run_audit(df, args.zero_eps, args.min_cross_section, args.out_csv)


if __name__ == "__main__":
    raise SystemExit(main())

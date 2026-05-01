#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Audit feature parquet rows that correspond to generated alpha timestamps.

This is the next-layer debug tool after audit_alpha_executable_edge.py:

1. Check whether alpha timestamps can be found in quote_features parquet.
2. Check missing / NaN / zero / outlier feature columns from slow_feature.json.
3. Check whether parquet delayed executable labels match sqlite option bid/ask.

It does not need model weights. It is meant to isolate feature_merge / timestamp
alignment / label generation issues before blaming strategy logic.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype, is_numeric_dtype

try:
    from audit_alpha_executable_edge import _date_to_db, _parse_locked_atm_quote
except ImportError:
    from production.baseline.utils.audit_alpha_executable_edge import _date_to_db, _parse_locked_atm_quote


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_HIST_DIR = Path.home() / "quant_project/data/history_sqlite_1s"
DEFAULT_FEATURE_ROOTS = [
    Path.home() / "train_data/quote_features_raw",
    Path.home() / "train_data/quote_features_train",
    Path.home() / "train_data/quote_features_val",
    Path.home() / "train_data/quote_features_test",
]
DEFAULT_SLOW_CONFIG = Path.home() / "notebook/train/slow_feature.json"


@dataclass(frozen=True)
class FeatureAuditConfig:
    delay_seconds: int = 60
    holding_seconds: int = 300
    sample_per_symbol_day: int = 80
    max_abs_feature: float = 20.0
    zero_ratio_warn: float = 0.95
    label_tolerance: float = 1e-6
    option_table: str = "option_snapshots_1m"


def _parse_dates(args) -> list[str]:
    if args.dates:
        return [x.strip() for x in args.dates.split(",") if x.strip()]
    if args.start_date and args.end_date:
        return [
            d.strftime("%Y%m%d")
            for d in pd.date_range(args.start_date, args.end_date, freq="D")
        ]
    raise SystemExit("请提供 --dates 或 --start-date/--end-date")


def _load_slow_feature_names(config_path: Path) -> list[str]:
    if not config_path.exists():
        raise FileNotFoundError(f"slow feature config 不存在: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    names = []
    for item in cfg.get("features", []):
        name = item.get("name")
        if name and not str(name).startswith("label_"):
            names.append(str(name))
    return names


def _feature_file(root: Path, symbol: str, date: str, res: str = "1min") -> Path:
    month = f"{date[:4]}-{date[4:6]}.parquet"
    return root / symbol / "regular" / "09:30-16:00" / res / month


def _timestamp_candidates(ts_col: pd.Series) -> dict[str, pd.Series]:
    if is_numeric_dtype(ts_col):
        raw = pd.to_numeric(ts_col, errors="coerce")
        finite = raw.dropna()
        scale = 1.0
        if not finite.empty:
            med = float(finite.abs().median())
            if med > 1e17:
                scale = 1e9
            elif med > 1e14:
                scale = 1e6
            elif med > 1e11:
                scale = 1e3
        sec = (raw / scale).round().astype("Int64")
        return {"numeric": sec}

    parsed = ts_col if is_datetime64_any_dtype(ts_col) else pd.to_datetime(ts_col, errors="coerce")
    out = {}
    try:
        if parsed.dt.tz is None:
            utc = parsed.dt.tz_localize("UTC")
            ny = parsed.dt.tz_localize("America/New_York").dt.tz_convert("UTC")
        else:
            utc = parsed.dt.tz_convert("UTC")
            ny = parsed.dt.tz_convert("America/New_York").dt.tz_convert("UTC")
        out["datetime_as_utc"] = (utc.astype("int64") // 1_000_000_000).astype("Int64")
        out["datetime_as_ny"] = (ny.astype("int64") // 1_000_000_000).astype("Int64")
    except Exception:
        pass
    return out


def _choose_timestamp_mapping(df: pd.DataFrame, target_ts: set[int]) -> tuple[str, pd.Series, int, int]:
    if "timestamp" not in df.columns:
        return "missing_timestamp", pd.Series(pd.NA, index=df.index, dtype="Int64"), 0, 0

    best = ("unmatched", pd.Series(pd.NA, index=df.index, dtype="Int64"), -1, 0)
    for name, sec in _timestamp_candidates(df["timestamp"]).items():
        for offset in (0, 60, -60):
            aligned = sec + offset
            hits = int(aligned.isin(target_ts).sum())
            if hits > best[2]:
                best = (f"{name}{offset:+d}s", aligned, hits, offset)
    return best


def _load_alpha_rows(hist_dir: Path, dates: Iterable[str]) -> pd.DataFrame:
    frames = []
    for date in dates:
        db_path = _date_to_db(hist_dir, date)
        if not db_path.exists():
            print(f"⚠️ 缺少数据库: {db_path}")
            continue
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        alpha = pd.read_sql(
            "SELECT ts, datetime_ny, symbol, alpha, price, vol_z FROM alpha_logs",
            conn,
        )
        conn.close()
        if alpha.empty:
            continue
        alpha["date"] = date
        alpha["ts"] = alpha["ts"].astype(int)
        frames.append(alpha)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _load_executable_returns_from_sqlite(
    db_path: Path,
    cfg: FeatureAuditConfig,
) -> pd.DataFrame:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    alpha = pd.read_sql(
        "SELECT ts, datetime_ny, symbol, alpha, price, vol_z FROM alpha_logs",
        conn,
    )
    if alpha.empty:
        conn.close()
        return pd.DataFrame()

    alpha["ts"] = alpha["ts"].astype(int)
    target_ts = sorted(
        set((alpha["ts"] + cfg.delay_seconds).astype(int))
        | set((alpha["ts"] + cfg.delay_seconds + cfg.holding_seconds).astype(int))
    )
    ts_csv = ",".join(str(int(x)) for x in target_ts)
    option = pd.read_sql(
        f"SELECT ts, symbol, buckets_json FROM {cfg.option_table} WHERE ts IN ({ts_csv})",
        conn,
    )
    conn.close()
    if option.empty:
        return pd.DataFrame()

    parsed = option["buckets_json"].map(_parse_locked_atm_quote)
    quote = pd.DataFrame(
        parsed.tolist(),
        columns=[
            "put_bid",
            "put_ask",
            "put_contract",
            "call_bid",
            "call_ask",
            "call_contract",
        ],
    )
    option = pd.concat([option[["ts", "symbol"]], quote], axis=1)
    option["ts"] = option["ts"].astype(int)

    entry = option.rename(
        columns={
            "ts": "entry_ts",
            "put_ask": "entry_put_ask",
            "call_ask": "entry_call_ask",
            "put_contract": "entry_put_contract",
            "call_contract": "entry_call_contract",
        }
    )
    entry["ts"] = entry["entry_ts"] - cfg.delay_seconds
    entry = entry[
        [
            "ts",
            "symbol",
            "entry_put_ask",
            "entry_call_ask",
            "entry_put_contract",
            "entry_call_contract",
        ]
    ]

    exit_ = option.rename(
        columns={
            "ts": "exit_ts",
            "put_bid": "exit_put_bid",
            "call_bid": "exit_call_bid",
            "put_contract": "exit_put_contract",
            "call_contract": "exit_call_contract",
        }
    )
    exit_["ts"] = exit_["exit_ts"] - cfg.delay_seconds - cfg.holding_seconds
    exit_ = exit_[
        [
            "ts",
            "symbol",
            "exit_put_bid",
            "exit_call_bid",
            "exit_put_contract",
            "exit_call_contract",
        ]
    ]

    df = alpha.merge(entry, on=["ts", "symbol"], how="left").merge(
        exit_, on=["ts", "symbol"], how="left"
    )
    df["date"] = db_path.stem.replace("market_", "")
    df["call_same_contract"] = df["entry_call_contract"].eq(df["exit_call_contract"])
    df["put_same_contract"] = df["entry_put_contract"].eq(df["exit_put_contract"])
    df["call_ret"] = df["exit_call_bid"] / df["entry_call_ask"] - 1.0
    df["put_ret"] = df["exit_put_bid"] / df["entry_put_ask"] - 1.0
    df["exec_ret"] = np.where(df["alpha"] >= 0, df["call_ret"], df["put_ret"])
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df[df["call_same_contract"] & df["put_same_contract"]].copy()


def _sample_alpha(alpha: pd.DataFrame, sample_per_symbol_day: int) -> pd.DataFrame:
    if sample_per_symbol_day <= 0:
        return alpha.copy()
    parts = []
    for _, group in alpha.groupby(["date", "symbol"], sort=False):
        if len(group) <= sample_per_symbol_day:
            parts.append(group)
        else:
            top = group.reindex(group["alpha"].abs().sort_values(ascending=False).index)
            parts.append(top.head(sample_per_symbol_day))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _collect_feature_rows(
    alpha: pd.DataFrame,
    feature_root: Path,
    feature_names: list[str],
    cfg: FeatureAuditConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    file_stats = []
    sampled = _sample_alpha(alpha, cfg.sample_per_symbol_day)
    need_cols = set(feature_names) | {
        "timestamp",
        "label_option_alpha_300s",
        "label_option_direction_300s",
        "label_call_delayed_exec_ret_300s",
        "label_put_delayed_exec_ret_300s",
        f"label_call_delayed_exec_ret_d{cfg.delay_seconds}s_h{cfg.holding_seconds}s",
        f"label_put_delayed_exec_ret_d{cfg.delay_seconds}s_h{cfg.holding_seconds}s",
        "label_option_tradable_300s",
    }

    for (date, symbol), group in sampled.groupby(["date", "symbol"], sort=False):
        path = _feature_file(feature_root, symbol, date)
        stat = {
            "root": str(feature_root),
            "date": date,
            "symbol": symbol,
            "alpha_samples": len(group),
            "file": str(path),
            "exists": path.exists(),
            "matched": 0,
            "timestamp_mapping": "",
            "missing_feature_cols": 0,
        }
        if not path.exists():
            file_stats.append(stat)
            continue

        try:
            df = pd.read_parquet(path)
        except Exception as exc:
            stat["error"] = str(exc)
            file_stats.append(stat)
            continue

        target_ts = set(group["ts"].astype(int).tolist())
        mapping_name, aligned_ts, hits, _ = _choose_timestamp_mapping(df, target_ts)
        stat["timestamp_mapping"] = mapping_name
        stat["matched"] = hits
        stat["missing_feature_cols"] = len([c for c in feature_names if c not in df.columns])

        keep_cols = [c for c in need_cols if c in df.columns]
        tmp = df.loc[aligned_ts.isin(target_ts), keep_cols].copy()
        if tmp.empty:
            file_stats.append(stat)
            continue
        tmp["ts"] = aligned_ts[aligned_ts.isin(target_ts)].astype(int).to_numpy()
        tmp["date"] = date
        tmp["symbol"] = symbol
        tmp = tmp.merge(
            group[["date", "symbol", "ts", "alpha", "datetime_ny"]],
            on=["date", "symbol", "ts"],
            how="left",
        )
        rows.append(tmp)
        file_stats.append(stat)

    feature_rows = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    stats = pd.DataFrame(file_stats)
    return feature_rows, stats


def _feature_quality_table(rows: pd.DataFrame, feature_names: list[str], cfg: FeatureAuditConfig) -> pd.DataFrame:
    available = [c for c in feature_names if c in rows.columns]
    if not available or rows.empty:
        return pd.DataFrame()
    data = rows[available].apply(pd.to_numeric, errors="coerce")
    out = []
    for col in available:
        s = data[col]
        bounded_like = col in {
            "rsi",
            "k",
            "d",
            "adx",
            "hour",
            "minute",
            "day_of_week",
            "session",
            "stock_id",
        }
        outlier = (s.abs() > cfg.max_abs_feature).mean() * 100.0
        if bounded_like:
            outlier = 0.0
        out.append(
            {
                "feature": col,
                "n": int(s.notna().sum()),
                "nan%": s.isna().mean() * 100.0,
                "zero%": (s.fillna(0.0).abs() < 1e-12).mean() * 100.0,
                "mean": s.mean(),
                "std": s.std(),
                "p01": s.quantile(0.01),
                "p99": s.quantile(0.99),
                "max_abs": s.abs().max(),
                "outlier%": outlier,
            }
        )
    q = pd.DataFrame(out)
    q["bad_score"] = q["nan%"] + q["outlier%"] + np.maximum(0.0, q["zero%"] - cfg.zero_ratio_warn * 100.0)
    return q.sort_values(["bad_score", "max_abs"], ascending=False)


def _label_parity(edge_df: pd.DataFrame, rows: pd.DataFrame, cfg: FeatureAuditConfig) -> pd.DataFrame:
    if rows.empty or edge_df.empty:
        return pd.DataFrame()
    required_edge_cols = {"date", "symbol", "ts", "call_ret", "put_ret"}
    if not required_edge_cols.issubset(edge_df.columns):
        return pd.DataFrame()
    label_cols = {
        "call_parquet": f"label_call_delayed_exec_ret_d{cfg.delay_seconds}s_h{cfg.holding_seconds}s",
        "put_parquet": f"label_put_delayed_exec_ret_d{cfg.delay_seconds}s_h{cfg.holding_seconds}s",
    }
    if label_cols["call_parquet"] not in rows.columns or label_cols["put_parquet"] not in rows.columns:
        label_cols = {
            "call_parquet": "label_call_delayed_exec_ret_300s",
            "put_parquet": "label_put_delayed_exec_ret_300s",
        }
    missing = [c for c in label_cols.values() if c not in rows.columns]
    if missing:
        return pd.DataFrame()

    merged = rows[
        ["date", "symbol", "ts", "alpha", label_cols["call_parquet"], label_cols["put_parquet"]]
    ].merge(
        edge_df[["date", "symbol", "ts", "call_ret", "put_ret"]],
        on=["date", "symbol", "ts"],
        how="inner",
        suffixes=("", "_sqlite"),
    )
    if merged.empty:
        return pd.DataFrame()

    call_diff = pd.to_numeric(merged[label_cols["call_parquet"]], errors="coerce") - merged["call_ret"]
    put_diff = pd.to_numeric(merged[label_cols["put_parquet"]], errors="coerce") - merged["put_ret"]
    return pd.DataFrame(
        [
            {
                "side": "CALL",
                "n": int(call_diff.notna().sum()),
                "mean_abs_diff": call_diff.abs().mean(),
                "p99_abs_diff": call_diff.abs().quantile(0.99),
                "mismatch%": (call_diff.abs() > cfg.label_tolerance).mean() * 100.0,
            },
            {
                "side": "PUT",
                "n": int(put_diff.notna().sum()),
                "mean_abs_diff": put_diff.abs().mean(),
                "p99_abs_diff": put_diff.abs().quantile(0.99),
                "mismatch%": (put_diff.abs() > cfg.label_tolerance).mean() * 100.0,
            },
        ]
    )


def run_feature_audit(
    hist_dir: Path,
    dates: list[str],
    feature_roots: list[Path],
    slow_config: Path,
    cfg: FeatureAuditConfig,
) -> int:
    feature_names = _load_slow_feature_names(slow_config)
    alpha = _load_alpha_rows(hist_dir, dates)
    if alpha.empty:
        print("❌ 没有 alpha_logs 样本")
        return 2

    print("\n" + "=" * 92)
    print("Alpha feature parity audit")
    print("=" * 92)
    print(f"hist_dir={hist_dir}")
    print(f"slow_config={slow_config}")
    print(
        f"dates={dates} alpha_samples={len(alpha)} slow_features={len(feature_names)} "
        f"option_table_for_label_check={cfg.option_table}"
    )

    edge_frames = []
    for date in dates:
        db = _date_to_db(hist_dir, date)
        if db.exists():
            day = _load_executable_returns_from_sqlite(db, cfg)
            if not day.empty:
                edge_frames.append(day)
    edge_df = pd.concat(edge_frames, ignore_index=True) if edge_frames else pd.DataFrame()

    failures = []
    warnings_out = []
    for root in feature_roots:
        print("\n" + "-" * 92)
        print(f"[Feature root] {root}")
        rows, stats = _collect_feature_rows(alpha, root, feature_names, cfg)
        if stats.empty:
            print("❌ 没有找到任何 parquet 文件")
            failures.append(f"{root}: no parquet stats")
            continue

        stats_view = stats.copy()
        stats_view["match%"] = np.where(
            stats_view["alpha_samples"] > 0,
            stats_view["matched"] / stats_view["alpha_samples"] * 100.0,
            0.0,
        )
        print("\n[文件/时间覆盖]")
        print(
            stats_view[
                [
                    "date",
                    "symbol",
                    "exists",
                    "alpha_samples",
                    "matched",
                    "match%",
                    "timestamp_mapping",
                    "missing_feature_cols",
                ]
            ]
            .sort_values(["exists", "match%", "missing_feature_cols"], ascending=[True, True, False])
            .head(30)
            .round(3)
            .to_string(index=False)
        )
        total_alpha = int(stats_view["alpha_samples"].sum())
        total_matched = int(stats_view["matched"].sum())
        coverage = total_matched / total_alpha if total_alpha else 0.0
        print(f"coverage={coverage * 100.0:.2f}% matched={total_matched}/{total_alpha}")
        if coverage < 0.95:
            failures.append(f"{root}: timestamp coverage {coverage:.2%} < 95%")

        if rows.empty:
            print("❌ 没有匹配到 feature rows")
            failures.append(f"{root}: no matched rows")
            continue

        missing_features = [c for c in feature_names if c not in rows.columns]
        print(f"\n[特征列覆盖] missing={len(missing_features)}/{len(feature_names)}")
        if missing_features:
            print("missing sample:", missing_features[:30])
            warnings_out.append(f"{root}: missing slow feature columns {missing_features[:30]}")

        quality = _feature_quality_table(rows, feature_names, cfg)
        if not quality.empty:
            print("\n[最可疑特征 Top30]")
            print(
                quality.head(30)
                .round(6)
                .to_string(index=False)
            )
            bad = quality[(quality["nan%"] > 0.0) | (quality["outlier%"] > 0.0) | (quality["zero%"] > cfg.zero_ratio_warn * 100.0)]
            if not bad.empty:
                failures.append(f"{root}: suspicious feature columns {len(bad)}")

        label_check = _label_parity(edge_df, rows, cfg)
        print("\n[parquet label vs sqlite ask->bid 复算]")
        if label_check.empty:
            print("⚠️ 缺少 delayed executable label 列，无法比对")
            failures.append(f"{root}: missing executable label columns")
        else:
            print(label_check.round(8).to_string(index=False))
            if (label_check["mismatch%"] > 1.0).any():
                failures.append(f"{root}: executable label mismatch > 1%")

    print("\n" + "=" * 92)
    print("[定位结论]")
    if warnings_out:
        for item in warnings_out:
            print(f"  ⚠️ {item}")
    if failures:
        for item in failures:
            print(f"  ❌ {item}")
        print("  优先沿着上述 root 的时间覆盖/特征异常/label mismatch 去查 feature_merge 或 rolling norm。")
        return 1

    print("  ✅ parquet 时间覆盖、数值质量、label 复算均未发现明显异常。")
    if warnings_out:
        print("  ⚠️ 仍建议统一 slow_feature.json 与实际 parquet 列，避免 S0 静默补 0。")
    print("  若 alpha 仍无排序能力，问题更偏模型训练目标/模型容量/样本分布，而不是 feature_merge 基础计算。")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hist-dir", type=Path, default=DEFAULT_HIST_DIR)
    parser.add_argument("--dates", type=str, default="")
    parser.add_argument("--start-date", type=str, default="")
    parser.add_argument("--end-date", type=str, default="")
    parser.add_argument("--feature-root", action="append", type=Path, default=[])
    parser.add_argument("--slow-config", type=Path, default=DEFAULT_SLOW_CONFIG)
    parser.add_argument("--delay", type=int, default=60)
    parser.add_argument("--holding", type=int, default=300)
    parser.add_argument("--sample-per-symbol-day", type=int, default=80)
    parser.add_argument("--max-abs-feature", type=float, default=20.0)
    parser.add_argument(
        "--option-table",
        type=str,
        default="option_snapshots_1m",
        choices=["option_snapshots_1m", "option_snapshots_1s"],
        help="Label parity 复算使用的 sqlite option 表。训练 parquet 通常应使用 1m；实盘偏差审计可用 1s。",
    )
    args = parser.parse_args()

    cfg = FeatureAuditConfig(
        delay_seconds=args.delay,
        holding_seconds=args.holding,
        sample_per_symbol_day=args.sample_per_symbol_day,
        max_abs_feature=args.max_abs_feature,
        option_table=args.option_table,
    )
    feature_roots = args.feature_root or DEFAULT_FEATURE_ROOTS
    dates = _parse_dates(args)
    return run_feature_audit(
        hist_dir=args.hist_dir,
        dates=dates,
        feature_roots=feature_roots,
        slow_config=args.slow_config,
        cfg=cfg,
    )


if __name__ == "__main__":
    raise SystemExit(main())

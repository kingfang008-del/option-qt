#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Audit generated alpha_logs against delayed executable option returns.

This script is intentionally downstream of model inference. It reads the
already generated history_sqlite_1s databases, joins alpha_logs to locked ATM
option quotes, then checks whether stronger alphas rank better on:

    entry ask at T + delay  ->  exit bid at T + delay + holding

Example:
    python3 production/baseline/utils/audit_alpha_executable_edge.py \
        --dates 20260302,20260303,20260304
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


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_HIST_DIR = Path.home() / "quant_project/data/history_sqlite_1s"


@dataclass(frozen=True)
class AuditConfig:
    delay_seconds: int = 60
    holding_seconds: int = 300
    min_edge: float = 0.02
    top_quantile: float = 0.80
    min_top_mean_ret: float = 0.0
    min_top_tradable_ratio: float = 0.15
    min_decile_spread: float = 0.005
    diagnostics: bool = True


def _safe_float(value, default=np.nan) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _parse_locked_atm_quote(buckets_json: str):
    """Return PUT_ATM and CALL_ATM bid/ask/contract from locked bucket JSON."""
    if not buckets_json or buckets_json in {"{}", "None"}:
        return (np.nan, np.nan, None, np.nan, np.nan, None)
    try:
        payload = json.loads(buckets_json)
        buckets = payload.get("buckets") or []
        contracts = payload.get("contracts") or []

        def side(idx: int):
            if len(buckets) <= idx or len(buckets[idx]) <= 9:
                return np.nan, np.nan, None
            bid = _safe_float(buckets[idx][8])
            ask = _safe_float(buckets[idx][9])
            contract = contracts[idx] if len(contracts) > idx else None
            return bid, ask, contract

        put_bid, put_ask, put_contract = side(0)
        call_bid, call_ask, call_contract = side(2)
        return put_bid, put_ask, put_contract, call_bid, call_ask, call_contract
    except Exception:
        return (np.nan, np.nan, None, np.nan, np.nan, None)


def _date_to_db(hist_dir: Path, date: str) -> Path:
    return hist_dir / f"market_{date}.db"


def _load_one_day(db_path: Path, cfg: AuditConfig) -> pd.DataFrame:
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
        f"SELECT ts, symbol, buckets_json FROM option_snapshots_1s WHERE ts IN ({ts_csv})",
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
    df["selected_side"] = np.where(df["alpha"] >= 0, "CALL", "PUT")
    df["same_contract"] = np.where(
        df["alpha"] >= 0, df["call_same_contract"], df["put_same_contract"]
    )
    df["exec_ret"] = np.where(df["alpha"] >= 0, df["call_ret"], df["put_ret"])
    df["abs_alpha"] = df["alpha"].abs()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df[df["same_contract"] & df["exec_ret"].notna()].copy()
    return df


def load_audit_frame(hist_dir: Path, dates: Iterable[str], cfg: AuditConfig) -> pd.DataFrame:
    frames = []
    for date in dates:
        db_path = _date_to_db(hist_dir, date)
        if not db_path.exists():
            print(f"⚠️ 缺少数据库: {db_path}")
            continue
        day_df = _load_one_day(db_path, cfg)
        if day_df.empty:
            print(f"⚠️ 无可审计样本: {db_path.name}")
            continue
        frames.append(day_df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _summarize_return(series: pd.Series, min_edge: float) -> dict:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return {
            "n": 0,
            "mean%": np.nan,
            "median%": np.nan,
            "win%": np.nan,
            "tradable%": np.nan,
            "p10%": np.nan,
            "p90%": np.nan,
        }
    return {
        "n": int(len(s)),
        "mean%": s.mean() * 100.0,
        "median%": s.median() * 100.0,
        "win%": (s > 0.0).mean() * 100.0,
        "tradable%": (s > min_edge).mean() * 100.0,
        "p10%": s.quantile(0.10) * 100.0,
        "p90%": s.quantile(0.90) * 100.0,
    }


def _print_zscore_health(df: pd.DataFrame) -> dict:
    grouped = df.groupby(["date", "ts"])["alpha"]
    snap = grouped.agg(["count", "mean", "std", "min", "max"]).reset_index()
    snap["std"] = snap["std"].fillna(0.0)
    health = {
        "snapshots": len(snap),
        "avg_count": snap["count"].mean(),
        "mean_abs_cs_mean": snap["mean"].abs().mean(),
        "median_cs_std": snap["std"].median(),
        "p10_cs_std": snap["std"].quantile(0.10),
        "p90_cs_std": snap["std"].quantile(0.90),
        "thin_snapshot_ratio": (snap["count"] < 8).mean(),
        "low_std_ratio": (snap["std"] < 0.50).mean(),
        "high_std_ratio": (snap["std"] > 1.50).mean(),
    }
    print("\n[全局 zscore 健康度]")
    print(
        pd.DataFrame(
            [
                {
                    "snapshots": health["snapshots"],
                    "avg_count": health["avg_count"],
                    "mean_abs_cs_mean": health["mean_abs_cs_mean"],
                    "median_cs_std": health["median_cs_std"],
                    "p10_cs_std": health["p10_cs_std"],
                    "p90_cs_std": health["p90_cs_std"],
                    "thin_snapshot%": health["thin_snapshot_ratio"] * 100.0,
                    "low_std%": health["low_std_ratio"] * 100.0,
                    "high_std%": health["high_std_ratio"] * 100.0,
                }
            ]
        )
        .round(4)
        .to_string(index=False)
    )
    return health


def _direction_mode_summary(df: pd.DataFrame, cfg: AuditConfig) -> pd.DataFrame:
    both = df[df["call_same_contract"] & df["put_same_contract"]].copy()
    if both.empty:
        return pd.DataFrame()

    modes = {
        "normal_alpha_sign": np.where(both["alpha"] >= 0, both["call_ret"], both["put_ret"]),
        "inverted_alpha_sign": np.where(both["alpha"] >= 0, both["put_ret"], both["call_ret"]),
        "call_only": both["call_ret"],
        "put_only": both["put_ret"],
        "oracle_best_side": np.maximum(both["call_ret"], both["put_ret"]),
    }
    rows = []
    for name, values in modes.items():
        s = pd.Series(values).replace([np.inf, -np.inf], np.nan).dropna()
        if s.empty:
            continue
        rows.append(
            {
                "mode": name,
                "n": len(s),
                "mean%": s.mean() * 100.0,
                "median%": s.median() * 100.0,
                "win%": (s > 0.0).mean() * 100.0,
                "tradable%": (s > cfg.min_edge).mean() * 100.0,
                "p90%": s.quantile(0.90) * 100.0,
            }
        )
    return pd.DataFrame(rows)


def _print_direction_and_oracle_diagnostics(df: pd.DataFrame, cfg: AuditConfig) -> dict:
    mode_df = _direction_mode_summary(df, cfg)
    print("\n[方向/可交易机会诊断]")
    if mode_df.empty:
        print("  ⚠️ CALL/PUT 双边合约覆盖不足，无法诊断方向。")
        return {"has_oracle_edge": False, "inverted_better": False}

    print(mode_df.round(3).to_string(index=False))
    normal = mode_df.loc[mode_df["mode"].eq("normal_alpha_sign"), "mean%"]
    inverted = mode_df.loc[mode_df["mode"].eq("inverted_alpha_sign"), "mean%"]
    oracle = mode_df.loc[mode_df["mode"].eq("oracle_best_side"), "mean%"]
    normal_mean = float(normal.iloc[0]) if not normal.empty else np.nan
    inverted_mean = float(inverted.iloc[0]) if not inverted.empty else np.nan
    oracle_mean = float(oracle.iloc[0]) if not oracle.empty else np.nan

    both = df[df["call_same_contract"] & df["put_same_contract"]].copy()
    both["call_minus_put"] = both["call_ret"] - both["put_ret"]
    corr_rows = []
    for col in ["call_ret", "put_ret", "call_minus_put"]:
        x = both[["alpha", col]].replace([np.inf, -np.inf], np.nan).dropna()
        corr_rows.append(
            {
                "target": col,
                "pearson": x["alpha"].corr(x[col]),
                "spearman": x["alpha"].corr(x[col], method="spearman"),
            }
        )
    print("\n[alpha 与 CALL/PUT executable return 相关性]")
    print(pd.DataFrame(corr_rows).round(4).to_string(index=False))

    return {
        "has_oracle_edge": bool(oracle_mean > 0.0),
        "inverted_better": bool(inverted_mean > normal_mean + 0.25),
        "normal_mean": normal_mean,
        "inverted_mean": inverted_mean,
        "oracle_mean": oracle_mean,
    }


def _print_root_cause_hint(
    z_health: dict,
    direction_diag: dict,
    top_summary: dict,
    decile_spread: float,
    cfg: AuditConfig,
) -> None:
    print("\n[定位判断]")
    zscore_looks_ok = (
        z_health["avg_count"] >= 8
        and z_health["mean_abs_cs_mean"] < 0.20
        and 0.70 <= z_health["median_cs_std"] <= 1.30
        and z_health["low_std_ratio"] < 0.10
    )
    if not zscore_looks_ok:
        print("  ⚠️ 全局 zscore 健康度异常，优先检查截面样本数、指数是否混入、以及重复归一化。")
    else:
        print("  ✅ 全局 zscore 分布基本正常，暂时不像是 zscore 算法本身把 alpha 打坏。")

    if direction_diag.get("inverted_better"):
        print("  ⚠️ 反向 CALL/PUT 映射明显更好，优先检查 label_direction / alpha sign 映射。")
    else:
        print("  ✅ 反向交易没有明显改善，暂时不像是简单方向取反问题。")

    has_alpha_rank_edge = (
        top_summary["mean%"] > cfg.min_top_mean_ret * 100.0
        and decile_spread > cfg.min_decile_spread
    )
    if direction_diag.get("has_oracle_edge") and not has_alpha_rank_edge:
        print("  ❌ 数据里存在可交易机会，但 alpha 没有把机会排到高分桶。")
        print("     更可能的问题：训练/推理特征分布不一致、feature_merge 特征计算错位、或模型目标与实盘入场规则不一致。")
    elif not direction_diag.get("has_oracle_edge"):
        print("  ⚠️ oracle_best_side 也没有正收益，优先检查 option bid/ask 数据质量或当前日期本身不可交易。")
    else:
        print("  ✅ alpha 对可交易收益有排序迹象，可以继续用策略阈值/风控做二次筛选。")


def run_audit(df: pd.DataFrame, cfg: AuditConfig) -> int:
    print("\n" + "=" * 88)
    print(
        f"Alpha executable edge audit | delay={cfg.delay_seconds}s "
        f"holding={cfg.holding_seconds}s min_edge={cfg.min_edge:.2%}"
    )
    print("=" * 88)
    print(f"samples={len(df)} dates={sorted(df['date'].unique())} symbols={sorted(df['symbol'].unique())}")

    overall = _summarize_return(df["exec_ret"], cfg.min_edge)
    print("\n[全样本 selected CALL/PUT ask->bid]")
    print(pd.DataFrame([overall]).round(3).to_string(index=False))
    z_health = _print_zscore_health(df) if cfg.diagnostics else {}
    direction_diag = (
        _print_direction_and_oracle_diagnostics(df, cfg) if cfg.diagnostics else {}
    )

    df = df.copy()
    df["abs_decile"] = pd.qcut(df["abs_alpha"].rank(method="first"), 10, labels=False) + 1
    decile = (
        df.groupby("abs_decile")
        .agg(
            n=("exec_ret", "size"),
            mean_abs_alpha=("abs_alpha", "mean"),
            mean_ret=("exec_ret", "mean"),
            median_ret=("exec_ret", "median"),
            win=("exec_ret", lambda x: (x > 0.0).mean()),
            tradable=("exec_ret", lambda x: (x > cfg.min_edge).mean()),
            p90=("exec_ret", lambda x: x.quantile(0.90)),
        )
        .reset_index()
    )
    for col in ["mean_ret", "median_ret", "win", "tradable", "p90"]:
        decile[col] *= 100.0
    print("\n[|alpha| 十分桶]")
    print(decile.round(3).to_string(index=False))

    top_threshold = df["abs_alpha"].quantile(cfg.top_quantile)
    top = df[df["abs_alpha"] >= top_threshold].copy()
    bottom = df[df["abs_decile"] == 1].copy()
    top_summary = _summarize_return(top["exec_ret"], cfg.min_edge)
    bottom_summary = _summarize_return(bottom["exec_ret"], cfg.min_edge)
    decile_spread = top["exec_ret"].mean() - bottom["exec_ret"].mean()

    print(f"\n[强信号 |alpha| >= p{int(cfg.top_quantile * 100)} threshold={top_threshold:.4f}]")
    print(pd.DataFrame([top_summary]).round(3).to_string(index=False))
    print(f"top_vs_bottom_mean_spread={decile_spread * 100.0:.3f}%")

    by_date = (
        top.groupby("date")
        .agg(
            n=("exec_ret", "size"),
            mean=("exec_ret", "mean"),
            median=("exec_ret", "median"),
            win=("exec_ret", lambda x: (x > 0.0).mean()),
            tradable=("exec_ret", lambda x: (x > cfg.min_edge).mean()),
        )
        .reset_index()
    )
    for col in ["mean", "median", "win", "tradable"]:
        by_date[col] *= 100.0
    print("\n[强信号按日期]")
    print(by_date.round(3).to_string(index=False))

    by_symbol = (
        top.groupby("symbol")
        .agg(
            n=("exec_ret", "size"),
            mean=("exec_ret", "mean"),
            win=("exec_ret", lambda x: (x > 0.0).mean()),
            tradable=("exec_ret", lambda x: (x > cfg.min_edge).mean()),
            avg_abs_alpha=("abs_alpha", "mean"),
        )
        .sort_values("mean", ascending=False)
        .reset_index()
    )
    for col in ["mean", "win", "tradable"]:
        by_symbol[col] *= 100.0
    print("\n[强信号按标的]")
    print(by_symbol.round(3).to_string(index=False))

    side = (
        top.groupby("selected_side")
        .agg(
            n=("exec_ret", "size"),
            mean=("exec_ret", "mean"),
            win=("exec_ret", lambda x: (x > 0.0).mean()),
            tradable=("exec_ret", lambda x: (x > cfg.min_edge).mean()),
            avg_abs_alpha=("abs_alpha", "mean"),
        )
        .reset_index()
    )
    for col in ["mean", "win", "tradable"]:
        side[col] *= 100.0
    print("\n[强信号按方向]")
    print(side.round(3).to_string(index=False))

    failures = []
    if top_summary["mean%"] < cfg.min_top_mean_ret * 100.0:
        failures.append(
            f"强信号平均收益 {top_summary['mean%']:.3f}% < "
            f"{cfg.min_top_mean_ret * 100.0:.3f}%"
        )
    if top_summary["tradable%"] < cfg.min_top_tradable_ratio * 100.0:
        failures.append(
            f"强信号 tradable {top_summary['tradable%']:.3f}% < "
            f"{cfg.min_top_tradable_ratio * 100.0:.3f}%"
        )
    if decile_spread < cfg.min_decile_spread:
        failures.append(
            f"强弱分桶收益差 {decile_spread * 100.0:.3f}% < "
            f"{cfg.min_decile_spread * 100.0:.3f}%"
        )

    print("\n[达标判断]")
    if failures:
        for item in failures:
            print(f"  ❌ {item}")
        if cfg.diagnostics:
            _print_root_cause_hint(
                z_health=z_health,
                direction_diag=direction_diag,
                top_summary=top_summary,
                decile_spread=decile_spread,
                cfg=cfg,
            )
        return 1

    print("  ✅ alpha 对 delayed executable return 具备基本排序能力")
    if cfg.diagnostics:
        _print_root_cause_hint(
            z_health=z_health,
            direction_diag=direction_diag,
            top_summary=top_summary,
            decile_spread=decile_spread,
            cfg=cfg,
        )
    return 0


def _parse_dates(args) -> list[str]:
    if args.dates:
        return [x.strip() for x in args.dates.split(",") if x.strip()]
    if args.start_date and args.end_date:
        return [
            d.strftime("%Y%m%d")
            for d in pd.date_range(args.start_date, args.end_date, freq="D")
        ]
    raise SystemExit("请提供 --dates 或 --start-date/--end-date")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hist-dir", type=Path, default=DEFAULT_HIST_DIR)
    parser.add_argument("--dates", type=str, default="")
    parser.add_argument("--start-date", type=str, default="")
    parser.add_argument("--end-date", type=str, default="")
    parser.add_argument("--delay", type=int, default=60)
    parser.add_argument("--holding", type=int, default=300)
    parser.add_argument("--min-edge", type=float, default=0.02)
    parser.add_argument("--top-quantile", type=float, default=0.80)
    parser.add_argument("--min-top-mean-ret", type=float, default=0.0)
    parser.add_argument("--min-top-tradable-ratio", type=float, default=0.15)
    parser.add_argument("--min-decile-spread", type=float, default=0.005)
    parser.add_argument("--no-diagnostics", action="store_true")
    args = parser.parse_args()

    cfg = AuditConfig(
        delay_seconds=args.delay,
        holding_seconds=args.holding,
        min_edge=args.min_edge,
        top_quantile=args.top_quantile,
        min_top_mean_ret=args.min_top_mean_ret,
        min_top_tradable_ratio=args.min_top_tradable_ratio,
        min_decile_spread=args.min_decile_spread,
        diagnostics=not args.no_diagnostics,
    )
    dates = _parse_dates(args)
    df = load_audit_frame(args.hist_dir, dates, cfg)
    if df.empty:
        print("❌ 没有可审计样本")
        return 2
    return run_audit(df, cfg)


if __name__ == "__main__":
    raise SystemExit(main())

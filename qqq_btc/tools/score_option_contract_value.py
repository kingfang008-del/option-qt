#!/usr/bin/env python3
"""Score daily option contracts by tradability and stock-move efficiency.

This is a diagnostic/selection tool for full-day option IV files produced by
preprocess/raw_data_deal/option_cac_day_vectorized.py.

It answers two related questions:
1. Ex-ante: at each timestamp, which contracts are most efficient for an
   expected up/down move in the underlying?
2. Ex-post: over a forward horizon, which contracts actually paid the best?
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


NY = "America/New_York"
DEFAULT_ROOT = Path.home() / "train_data/nq_options_day_iv"


def _to_ny_timestamp(series: pd.Series) -> pd.Series:
    ts = pd.to_datetime(series, errors="coerce")
    if ts.dt.tz is None:
        return ts.dt.tz_localize(NY, ambiguous="infer")
    return ts.dt.tz_convert(NY)


def _safe_div(a, b, default=0.0):
    out = np.divide(a, b, out=np.full_like(a, default, dtype=float), where=np.abs(b) > 1e-12)
    return np.nan_to_num(out, nan=default, posinf=default, neginf=default)


def _parse_horizons(raw: str) -> list[int]:
    vals = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if not vals:
        raise ValueError("--horizons must contain at least one integer")
    return sorted(set(vals))


def load_day_file(root: Path, symbol: str, date: str) -> pd.DataFrame:
    path = root / symbol / f"{symbol}_{date}.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_parquet(path)
    if df.empty:
        raise ValueError(f"empty day file: {path}")
    if "timestamp" not in df.columns:
        raise ValueError(f"missing timestamp column: {path}")
    df["timestamp"] = _to_ny_timestamp(df["timestamp"])
    return df


def add_value_scores(
    df: pd.DataFrame,
    *,
    expected_move_pct: float,
    hold_minutes: int,
    min_volume: float,
    min_premium: float,
    max_premium_pct: float,
    min_abs_delta: float,
    max_abs_delta: float,
) -> pd.DataFrame:
    required = {
        "ticker",
        "timestamp",
        "expiration_date",
        "contract_type",
        "strike_price",
        "close",
        "volume",
        "delta",
        "gamma",
        "theta",
        "vega",
        "stock_close",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")

    out = df.copy()
    out["expiration_date"] = pd.to_datetime(out["expiration_date"], errors="coerce")
    if out["expiration_date"].dt.tz is None:
        out["expiration_date"] = out["expiration_date"].dt.tz_localize(NY, ambiguous="infer")
    else:
        out["expiration_date"] = out["expiration_date"].dt.tz_convert(NY)

    price = pd.to_numeric(out["close"], errors="coerce").to_numpy(dtype=float)
    stock = pd.to_numeric(out["stock_close"], errors="coerce").to_numpy(dtype=float)
    strike = pd.to_numeric(out["strike_price"], errors="coerce").to_numpy(dtype=float)
    volume = pd.to_numeric(out["volume"], errors="coerce").fillna(0).to_numpy(dtype=float)
    delta = pd.to_numeric(out["delta"], errors="coerce").fillna(0).to_numpy(dtype=float)
    gamma = pd.to_numeric(out["gamma"], errors="coerce").fillna(0).to_numpy(dtype=float)
    theta = pd.to_numeric(out["theta"], errors="coerce").fillna(0).to_numpy(dtype=float)
    vega = pd.to_numeric(out["vega"], errors="coerce").fillna(0).to_numpy(dtype=float)

    premium_pct = _safe_div(price, stock)
    abs_delta = np.abs(delta)
    moneyness = np.log(_safe_div(strike, stock, default=np.nan))
    dte_days = (
        out["expiration_date"].to_numpy(dtype="datetime64[ns]")
        - out["timestamp"].dt.tz_localize(None).to_numpy(dtype="datetime64[ns]")
    ).astype("timedelta64[s]").astype(float) / 86400.0
    dte_days = np.maximum(dte_days, 0.0)

    if {"high", "low"}.issubset(out.columns):
        high = pd.to_numeric(out["high"], errors="coerce").to_numpy(dtype=float)
        low = pd.to_numeric(out["low"], errors="coerce").to_numpy(dtype=float)
        hl_range_pct = _safe_div(np.maximum(high - low, 0.0), price)
    else:
        hl_range_pct = np.full(len(out), 0.02)

    # No bid/ask in this data source. Use intrabar high-low and volume as a conservative tradability proxy.
    cost_proxy = np.clip(0.25 * hl_range_pct + 0.002 / np.sqrt(np.maximum(volume, 1.0)), 0.001, 0.30)
    liquidity_score = np.log1p(volume) / np.log1p(max(float(np.nanmax(volume)), 1.0))

    hold_days = hold_minutes / 390.0
    move_abs = stock * float(expected_move_pct)
    up_pnl = delta * move_abs + 0.5 * gamma * move_abs * move_abs + theta * hold_days
    down_pnl = -delta * move_abs + 0.5 * gamma * move_abs * move_abs + theta * hold_days
    convexity_roi = _safe_div(0.5 * gamma * move_abs * move_abs + theta * hold_days, price)

    score_up_raw = _safe_div(up_pnl, price) - cost_proxy
    score_down_raw = _safe_div(down_pnl, price) - cost_proxy

    # Penalize contracts that will become "wrong object" quickly if spot moves.
    gamma_drift = _safe_div(np.abs(gamma) * move_abs, np.maximum(abs_delta, 0.05))
    stability = 1.0 / (1.0 + gamma_drift)
    dte_penalty = 1.0 / (1.0 + np.maximum(0.0, 1.0 - dte_days) * 2.0)

    tradability = np.sqrt(np.clip(liquidity_score, 0.0, 1.0)) * stability * dte_penalty

    out["dte_days"] = dte_days
    out["moneyness"] = moneyness
    out["premium_pct"] = premium_pct
    out["abs_delta"] = abs_delta
    out["hl_range_pct"] = hl_range_pct
    out["cost_proxy"] = cost_proxy
    out["liquidity_score"] = liquidity_score
    out["stability_score"] = stability
    out["leverage_per_premium"] = _safe_div(abs_delta * stock, price)
    out["gamma_roi_1move"] = _safe_div(0.5 * gamma * move_abs * move_abs, price)
    out["theta_drag_hold"] = _safe_div(np.abs(theta) * hold_days, price)
    out["convexity_roi"] = convexity_roi
    out["score_up_raw"] = score_up_raw
    out["score_down_raw"] = score_down_raw
    out["score_up"] = score_up_raw * tradability
    out["score_down"] = score_down_raw * tradability

    ok = (
        np.isfinite(price)
        & np.isfinite(stock)
        & (price >= min_premium)
        & (premium_pct <= max_premium_pct)
        & (volume >= min_volume)
        & (abs_delta >= min_abs_delta)
        & (abs_delta <= max_abs_delta)
        & np.isfinite(moneyness)
    )
    return out.loc[ok].copy()


def add_realized_returns(df: pd.DataFrame, horizons: Iterable[int]) -> pd.DataFrame:
    out = df.sort_values(["ticker", "timestamp"]).copy()
    for h in horizons:
        future = out.groupby("ticker", sort=False)["close"].shift(-int(h))
        out[f"realized_return_h{h}"] = future / out["close"] - 1.0
    return out


def top_by_timestamp(df: pd.DataFrame, score_col: str, top_n: int) -> pd.DataFrame:
    if df.empty:
        return df
    return (
        df.sort_values(["timestamp", score_col], ascending=[True, False])
        .groupby("timestamp", group_keys=False)
        .head(top_n)
        .reset_index(drop=True)
    )


def summarize_selection(df: pd.DataFrame, score_col: str, horizons: Iterable[int]) -> dict:
    if df.empty:
        return {"rows": 0}
    summary = {
        "rows": int(len(df)),
        "unique_tickers": int(df["ticker"].nunique()),
        "mean_score": float(pd.to_numeric(df[score_col], errors="coerce").mean()),
        "median_abs_delta": float(pd.to_numeric(df["abs_delta"], errors="coerce").median()),
        "median_dte_days": float(pd.to_numeric(df["dte_days"], errors="coerce").median()),
        "median_premium_pct": float(pd.to_numeric(df["premium_pct"], errors="coerce").median()),
        "median_cost_proxy": float(pd.to_numeric(df["cost_proxy"], errors="coerce").median()),
    }
    for h in horizons:
        col = f"realized_return_h{h}"
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce").dropna()
            if not vals.empty:
                summary[col] = {
                    "mean": float(vals.mean()),
                    "median": float(vals.median()),
                    "hit_pos": float((vals > 0).mean()),
                    "n": int(len(vals)),
                }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Score full-chain option contracts by value efficiency")
    parser.add_argument("--root", default=str(DEFAULT_ROOT), help="day-IV root, e.g. ~/train_data/nq_options_day_iv")
    parser.add_argument("--symbol", default="QQQ")
    parser.add_argument("--date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--output-dir", default="qqq_btc/results/contract_value_score")
    parser.add_argument("--expected-move-pct", type=float, default=0.005, help="absolute spot move scenario, e.g. 0.005=0.5%")
    parser.add_argument("--hold-minutes", type=int, default=30)
    parser.add_argument("--horizons", default="5,15,30", help="forward realized return horizons in minutes")
    parser.add_argument("--top-n", type=int, default=3)
    parser.add_argument("--min-volume", type=float, default=10.0)
    parser.add_argument("--min-premium", type=float, default=0.05)
    parser.add_argument("--max-premium-pct", type=float, default=0.08)
    parser.add_argument("--min-abs-delta", type=float, default=0.05)
    parser.add_argument("--max-abs-delta", type=float, default=0.95)
    args = parser.parse_args()

    horizons = _parse_horizons(args.horizons)
    root = Path(args.root).expanduser()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_day_file(root, args.symbol, args.date)
    scored = add_value_scores(
        df,
        expected_move_pct=args.expected_move_pct,
        hold_minutes=args.hold_minutes,
        min_volume=args.min_volume,
        min_premium=args.min_premium,
        max_premium_pct=args.max_premium_pct,
        min_abs_delta=args.min_abs_delta,
        max_abs_delta=args.max_abs_delta,
    )
    scored = add_realized_returns(scored, horizons)

    keep_cols = [
        "timestamp",
        "ticker",
        "contract_type",
        "expiration_date",
        "strike_price",
        "stock_close",
        "close",
        "volume",
        "dte_days",
        "moneyness",
        "premium_pct",
        "abs_delta",
        "delta",
        "gamma",
        "theta",
        "vega",
        "cost_proxy",
        "liquidity_score",
        "stability_score",
        "leverage_per_premium",
        "gamma_roi_1move",
        "theta_drag_hold",
        "score_up",
        "score_down",
        "score_up_raw",
        "score_down_raw",
    ] + [f"realized_return_h{h}" for h in horizons]
    keep_cols = [c for c in keep_cols if c in scored.columns]

    top_up = top_by_timestamp(scored[scored["contract_type"] == "c"], "score_up", args.top_n)[keep_cols]
    top_down = top_by_timestamp(scored[scored["contract_type"] == "p"], "score_down", args.top_n)[keep_cols]
    top_oracle = {}
    for h in horizons:
        col = f"realized_return_h{h}"
        top_oracle[h] = top_by_timestamp(scored.dropna(subset=[col]), col, args.top_n)[keep_cols]

    prefix = f"{args.symbol}_{args.date}"
    scored[keep_cols].to_parquet(out_dir / f"{prefix}_all_scored.parquet", index=False)
    top_up.to_parquet(out_dir / f"{prefix}_top_up.parquet", index=False)
    top_down.to_parquet(out_dir / f"{prefix}_top_down.parquet", index=False)
    for h, frame in top_oracle.items():
        frame.to_parquet(out_dir / f"{prefix}_oracle_h{h}.parquet", index=False)

    report = {
        "symbol": args.symbol,
        "date": args.date,
        "root": str(root),
        "rows_raw": int(len(df)),
        "rows_scored": int(len(scored)),
        "params": vars(args),
        "top_up": summarize_selection(top_up, "score_up", horizons),
        "top_down": summarize_selection(top_down, "score_down", horizons),
        "oracle": {
            str(h): summarize_selection(frame, f"realized_return_h{h}", horizons)
            for h, frame in top_oracle.items()
        },
        "outputs": {
            "all_scored": str(out_dir / f"{prefix}_all_scored.parquet"),
            "top_up": str(out_dir / f"{prefix}_top_up.parquet"),
            "top_down": str(out_dir / f"{prefix}_top_down.parquet"),
            "oracle": {str(h): str(out_dir / f"{prefix}_oracle_h{h}.parquet") for h in horizons},
        },
    }
    report_path = out_dir / f"{prefix}_summary.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Report MAG7 short-DTE coverage by trade_weekday × expiry_weekday × dte.

Reads the existing locked map (and optionally micro contract_1s) and writes:
  - enriched weekday locked map parquet
  - coverage matrices (CSV + summary.json)
  - optional micro liquidity snapshot by weekday×dte

Does not re-download data.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from stock_options.common.short_dte_config import (
    DEFAULT_LOCKED_MAP,
    DEFAULT_LOCKED_MAP_WEEKDAY,
    DEFAULT_MICRO_ROOT,
    RESEARCH_START,
    enrich_locked_map_weekdays,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--locked-map", default=str(DEFAULT_LOCKED_MAP))
    p.add_argument("--output-map", default=str(DEFAULT_LOCKED_MAP_WEEKDAY))
    p.add_argument(
        "--output-dir",
        default="stock_options/results/mag7_short_dte_weekday_coverage",
    )
    p.add_argument("--symbols", default="NVDA,TSLA")
    p.add_argument("--start-date", default=RESEARCH_START)
    p.add_argument("--end-date", default="")
    p.add_argument("--micro-root", default=str(DEFAULT_MICRO_ROOT))
    p.add_argument(
        "--probe-micro",
        action="store_true",
        help="Sample one ATM-ish contract per day×dte for spread/quote stats.",
    )
    p.add_argument("--micro-sample-days", type=int, default=40)
    return p.parse_args()


def _day_bucket(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (symbol, date_str, selected_dte) with weekday labels."""
    keys = ["symbol", "date_str", "selected_dte", "expiration"]
    g = (
        df.groupby(keys, as_index=False)
        .agg(
            n_contracts=("contract_symbol", "nunique"),
            trade_weekday=("trade_weekday", "first"),
            trade_weekday_name=("trade_weekday_name", "first"),
            expiry_weekday=("expiry_weekday", "first"),
            expiry_weekday_name=("expiry_weekday_name", "first"),
            is_mon_wed_fri_expiry=("is_mon_wed_fri_expiry", "first"),
        )
        .sort_values(["symbol", "date_str", "selected_dte"])
    )
    return g


def coverage_matrices(day: pd.DataFrame) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for sym, sub in day.groupby("symbol"):
        # trade_weekday × dte day counts
        tw = (
            sub.groupby(["trade_weekday_name", "selected_dte"])
            .size()
            .unstack(fill_value=0)
            .reindex(columns=[0, 1, 2], fill_value=0)
        )
        tw["symbol"] = sym
        out[f"trade_weekday_x_dte_{sym}"] = tw.reset_index()

        ew = (
            sub.groupby(["expiry_weekday_name", "selected_dte"])
            .size()
            .unstack(fill_value=0)
            .reindex(columns=[0, 1, 2], fill_value=0)
        )
        ew["symbol"] = sym
        out[f"expiry_weekday_x_dte_{sym}"] = ew.reset_index()

        # joint skeleton: which (trade_wd, expiry_wd, dte) exist
        joint = (
            sub.groupby(
                [
                    "trade_weekday_name",
                    "expiry_weekday_name",
                    "selected_dte",
                ]
            )
            .size()
            .reset_index(name="n_days")
        )
        joint["symbol"] = sym
        out[f"joint_trade_expiry_dte_{sym}"] = joint
    return out


def summarize(day: pd.DataFrame, start: str, end: str | None) -> dict:
    symbols = sorted(day["symbol"].unique().tolist())
    blob: dict = {
        "start_date": start,
        "end_date": end or str(day["date_str"].max()),
        "n_day_buckets": int(len(day)),
        "symbols": {},
        "design_note": (
            "Shared model + weekday/DTE features; do not train one model per weekday. "
            "Calibrate thresholds on (symbol, selected_dte, trade_weekday, expiry_weekday)."
        ),
    }
    for sym in symbols:
        sub = day[day["symbol"] == sym]
        n_days = int(sub["date_str"].nunique())
        by_dte = sub.groupby("selected_dte")["date_str"].nunique().to_dict()
        by_exp = (
            sub.groupby("expiry_weekday_name")["date_str"].nunique().sort_values(ascending=False).to_dict()
        )
        by_trade = (
            sub.groupby("trade_weekday_name")["date_str"].nunique().sort_values(ascending=False).to_dict()
        )
        # Share of day-buckets on primary Mon/Wed/Fri expiries
        primary_share = float(sub["is_mon_wed_fri_expiry"].mean()) if len(sub) else 0.0
        blob["symbols"][sym] = {
            "n_trade_days": n_days,
            "n_day_buckets": int(len(sub)),
            "days_by_dte": {str(k): int(v) for k, v in by_dte.items()},
            "trade_days_by_trade_weekday": {str(k): int(v) for k, v in by_trade.items()},
            "trade_days_by_expiry_weekday": {str(k): int(v) for k, v in by_exp.items()},
            "pct_mon_wed_fri_expiry_buckets": primary_share,
            "date_min": str(sub["date_str"].min()),
            "date_max": str(sub["date_str"].max()),
        }
    return blob


def probe_micro_liquidity(
    day: pd.DataFrame,
    micro_root: Path,
    sample_days: int,
) -> pd.DataFrame:
    """Light probe: median spread_pct / quote count for sampled days."""
    rows = []
    for (sym, dte), sub in day.groupby(["symbol", "selected_dte"]):
        dates = sorted(sub["date_str"].unique().tolist())
        # Stratify a bit: take first/mid/last chunks.
        if len(dates) > sample_days:
            step = max(1, len(dates) // sample_days)
            dates = dates[::step][:sample_days]
        root = micro_root / "contract_1s" / sym
        for d in dates:
            fp = root / f"{sym}_{d}.parquet"
            if not fp.exists():
                rows.append(
                    {
                        "symbol": sym,
                        "date_str": d,
                        "selected_dte": int(dte),
                        "micro_exists": 0,
                    }
                )
                continue
            try:
                raw = pd.read_parquet(fp)
            except Exception:
                rows.append(
                    {
                        "symbol": sym,
                        "date_str": d,
                        "selected_dte": int(dte),
                        "micro_exists": 0,
                        "read_error": 1,
                    }
                )
                continue
            dte_col = "selected_dte" if "selected_dte" in raw.columns else "target_dte"
            if dte_col not in raw.columns:
                rows.append(
                    {
                        "symbol": sym,
                        "date_str": d,
                        "selected_dte": int(dte),
                        "micro_exists": 1,
                        "n_rows": 0,
                        "missing_dte_col": 1,
                    }
                )
                continue
            part = raw[pd.to_numeric(raw[dte_col], errors="coerce") == int(dte)]
            meta = sub[sub["date_str"] == d].iloc[0]
            if part.empty:
                rows.append(
                    {
                        "symbol": sym,
                        "date_str": d,
                        "selected_dte": int(dte),
                        "micro_exists": 1,
                        "n_rows": 0,
                        "trade_weekday_name": meta["trade_weekday_name"],
                        "expiry_weekday_name": meta["expiry_weekday_name"],
                    }
                )
                continue
            tradable = (
                (part.get("ask", pd.Series(dtype=float)) >= 0.15)
                & (part.get("bid", pd.Series(dtype=float)) > 0)
                & (part.get("spread_pct", pd.Series(dtype=float)) <= 0.12)
            )
            rows.append(
                {
                    "symbol": sym,
                    "date_str": d,
                    "selected_dte": int(dte),
                    "micro_exists": 1,
                    "n_rows": int(len(part)),
                    "n_tradable": int(tradable.sum()) if len(tradable) else 0,
                    "median_spread_pct": float(part["spread_pct"].median())
                    if "spread_pct" in part.columns
                    else None,
                    "p90_spread_pct": float(part["spread_pct"].quantile(0.9))
                    if "spread_pct" in part.columns
                    else None,
                    "trade_weekday_name": meta["trade_weekday_name"],
                    "expiry_weekday_name": meta["expiry_weekday_name"],
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    locked = Path(args.locked_map)
    if not locked.exists():
        raise SystemExit(f"locked map missing: {locked}")

    df = pd.read_parquet(locked)
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    df = df[df["symbol"].isin(symbols)].copy()
    if args.start_date:
        df = df[df["date_str"] >= args.start_date]
    if args.end_date:
        df = df[df["date_str"] <= args.end_date]
    if df.empty:
        raise SystemExit("no locked-map rows after filters")

    df = enrich_locked_map_weekdays(df)
    out_map = Path(args.output_map)
    out_map.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_map, index=False)

    day = _day_bucket(df)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    day.to_csv(out_dir / "day_buckets.csv", index=False)

    matrices = coverage_matrices(day)
    for name, mat in matrices.items():
        mat.to_csv(out_dir / f"{name}.csv", index=False)

    # Combined convenience tables
    tw_all = pd.concat(
        [v for k, v in matrices.items() if k.startswith("trade_weekday_x_dte_")],
        ignore_index=True,
    )
    ew_all = pd.concat(
        [v for k, v in matrices.items() if k.startswith("expiry_weekday_x_dte_")],
        ignore_index=True,
    )
    tw_all.to_csv(out_dir / "trade_weekday_x_dte.csv", index=False)
    ew_all.to_csv(out_dir / "expiry_weekday_x_dte.csv", index=False)

    summary = summarize(day, args.start_date, args.end_date or None)
    summary["files"] = {
        "weekday_locked_map": str(out_map),
        "day_buckets": str(out_dir / "day_buckets.csv"),
        "trade_weekday_x_dte": str(out_dir / "trade_weekday_x_dte.csv"),
        "expiry_weekday_x_dte": str(out_dir / "expiry_weekday_x_dte.csv"),
        "summary": str(out_dir / "summary.json"),
    }

    if args.probe_micro:
        micro = probe_micro_liquidity(day, Path(args.micro_root), args.micro_sample_days)
        micro_path = out_dir / "micro_liquidity_probe.csv"
        micro.to_csv(micro_path, index=False)
        summary["files"]["micro_liquidity_probe"] = str(micro_path)
        if not micro.empty and "median_spread_pct" in micro.columns:
            agg = (
                micro.dropna(subset=["median_spread_pct"])
                .groupby(["symbol", "selected_dte", "expiry_weekday_name"], dropna=False)
                .agg(
                    n_days=("date_str", "nunique"),
                    median_of_median_spread=("median_spread_pct", "median"),
                    median_tradable_rows=("n_tradable", "median"),
                    pct_micro_exists=("micro_exists", "mean"),
                )
                .reset_index()
            )
            agg_path = out_dir / "micro_liquidity_by_expiry_weekday.csv"
            agg.to_csv(agg_path, index=False)
            summary["files"]["micro_liquidity_by_expiry_weekday"] = str(agg_path)
            summary["micro_probe_rows"] = int(len(micro))

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str))
    print(f"results -> {out_dir}")
    print(f"weekday map -> {out_map}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Oracle diagnostics for 0DTE short-horizon option-edge labels.

This does not train a model. It answers a prerequisite question:
if we could perfectly rank minute/contract opportunities, does the
0DTE universe contain enough executable edge at 1/3/5/10 minute horizons?
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_labels(label_dir: Path, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    files = sorted((label_dir / symbol).glob(f"{symbol}_*.parquet"))
    files = [
        p for p in files
        if start_date <= p.stem.replace(f"{symbol}_", "") <= end_date
    ]
    if not files:
        raise SystemExit(f"no label files under {label_dir / symbol} for {start_date}..{end_date}")
    frames = [pd.read_parquet(p) for p in files]
    df = pd.concat(frames, ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("America/New_York")
    df["month"] = df["timestamp"].dt.strftime("%Y-%m")
    return df


def summarize_returns(values: pd.Series) -> dict:
    v = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if v.empty:
        return {"n": 0}
    return {
        "n": int(len(v)),
        "mean": float(v.mean()),
        "median": float(v.median()),
        "hit_rate": float((v > 0).mean()),
        "p90": float(v.quantile(0.90)),
        "p95": float(v.quantile(0.95)),
        "p99": float(v.quantile(0.99)),
        "worst": float(v.min()),
        "best": float(v.max()),
    }


def top_fraction_summary(df: pd.DataFrame, value_col: str, fractions: list[float]) -> dict:
    work = df.dropna(subset=[value_col]).copy()
    if work.empty:
        return {}
    out: dict[str, dict] = {}
    for frac in fractions:
        n = max(1, int(len(work) * frac))
        top = work.nlargest(n, value_col)
        tag = f"top{int(frac * 100)}"
        out[tag] = {
            **summarize_returns(top[value_col]),
            "trades_per_day": float(n / max(1, top["date_str"].nunique())),
            "side_dist": top["side"].value_counts().to_dict() if "side" in top.columns else {},
            "bucket_dist": {str(k): int(v) for k, v in top["bucket_id"].value_counts().to_dict().items()}
            if "bucket_id" in top.columns else {},
        }
    return out


def best_by_timestamp(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    col = f"ret_{horizon}m"
    work = df.dropna(subset=[col]).copy()
    if work.empty:
        return work
    idx = work.groupby("timestamp")[col].idxmax()
    return work.loc[idx].sort_values("timestamp").reset_index(drop=True)


def side_best_by_timestamp(df: pd.DataFrame, horizon: int, side: str) -> pd.DataFrame:
    col = f"ret_{horizon}m"
    work = df[(df["side"] == side)].dropna(subset=[col]).copy()
    if work.empty:
        return work
    idx = work.groupby("timestamp")[col].idxmax()
    return work.loc[idx].sort_values("timestamp").reset_index(drop=True)


def threshold_summary(best: pd.DataFrame, horizon: int, thresholds: list[float]) -> dict:
    col = f"ret_{horizon}m"
    out = {}
    for th in thresholds:
        sub = best[pd.to_numeric(best[col], errors="coerce") >= th]
        out[str(th)] = {
            **summarize_returns(sub[col]),
            "trades_per_day": float(len(sub) / max(1, sub["date_str"].nunique())) if not sub.empty else 0.0,
            "side_dist": sub["side"].value_counts().to_dict() if not sub.empty else {},
            "bucket_dist": {str(k): int(v) for k, v in sub["bucket_id"].value_counts().to_dict().items()}
            if not sub.empty else {},
        }
    return out


def evaluate_month(df: pd.DataFrame, horizons: list[int], fractions: list[float], thresholds: list[float]) -> dict:
    result: dict[str, object] = {
        "rows": int(len(df)),
        "days": int(df["date_str"].nunique()),
        "minutes": int(df["timestamp"].nunique()),
    }
    for h in horizons:
        any_best = best_by_timestamp(df, h)
        call_best = side_best_by_timestamp(df, h, "CALL")
        put_best = side_best_by_timestamp(df, h, "PUT")
        col = f"ret_{h}m"
        result[f"h{h}m"] = {
            "all_contracts": summarize_returns(df[col]) if col in df.columns else {"n": 0},
            "best_any_per_minute": summarize_returns(any_best[col]) if not any_best.empty else {"n": 0},
            "best_call_per_minute": summarize_returns(call_best[col]) if not call_best.empty else {"n": 0},
            "best_put_per_minute": summarize_returns(put_best[col]) if not put_best.empty else {"n": 0},
            "top_fraction_oracle": top_fraction_summary(any_best, col, fractions),
            "threshold_oracle": threshold_summary(any_best, h, thresholds),
            "best_side_dist": any_best["side"].value_counts().to_dict() if not any_best.empty else {},
            "best_bucket_dist": {str(k): int(v) for k, v in any_best["bucket_id"].value_counts().to_dict().items()}
            if not any_best.empty else {},
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate oracle edge in 0DTE option labels")
    parser.add_argument("--label-dir", default=str(Path.home() / "train_data/option_edge_labels_0dte"))
    parser.add_argument("--symbol", default="QQQ")
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--horizons", default="1,3,5,10")
    parser.add_argument("--top-fracs", default="0.20,0.10,0.05,0.01")
    parser.add_argument("--thresholds", default="0.02,0.05,0.10,0.20")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    horizons = [int(x) for x in args.horizons.split(",") if x.strip()]
    fractions = [float(x) for x in args.top_fracs.split(",") if x.strip()]
    thresholds = [float(x) for x in args.thresholds.split(",") if x.strip()]
    labels = load_labels(Path(args.label_dir).expanduser(), args.symbol, args.start_date, args.end_date)

    report = {
        "symbol": args.symbol,
        "label_dir": str(Path(args.label_dir).expanduser()),
        "start_date": args.start_date,
        "end_date": args.end_date,
        "horizons": horizons,
        "overall": evaluate_month(labels, horizons, fractions, thresholds),
        "months": {},
    }
    for month, g in labels.groupby("month"):
        report["months"][month] = evaluate_month(g, horizons, fractions, thresholds)

    out_path = Path(args.output).expanduser() if args.output else (
        Path(args.label_dir).expanduser() / f"{args.symbol}_{args.start_date}_{args.end_date}_oracle.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    compact = {
        "symbol": report["symbol"],
        "start_date": report["start_date"],
        "end_date": report["end_date"],
        "months": {
            m: {
                f"h{h}m_best_any_mean": report["months"][m][f"h{h}m"]["best_any_per_minute"].get("mean")
                for h in horizons
            }
            for m in report["months"]
        },
        "output": str(out_path),
    }
    print(json.dumps(compact, indent=2, default=str))


if __name__ == "__main__":
    main()

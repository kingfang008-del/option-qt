#!/usr/bin/env python3
"""Convert sniper raw_1s option dumps into State Gate micro contract_1s layout.

Input layout (examples):
  /mnt/s990/data/raw_1s/dte1_options/QQQ/QQQ_YYYY-MM-DD.parquet
  /mnt/s990/data/raw_1s/options/NVDA/NVDA_YYYY-MM-DD.parquet

Output layout (State Gate loader):
  {output}/contract_1s/{SYMBOL}/{SYMBOL}_YYYY-MM-DD.parquet

This is a research bridge: raw sniper files often lack full quote-event / trade-flow
microstructure. Missing event columns are filled with 0 so the existing factor
pipeline can run; flow-based scores will be weaker than Polygon micro downloads.
Prefer `download_short_dte_microstructure.py` when API download is available.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


NY = "America/New_York"
EVENT_ZERO_COLS = [
    "quote_events",
    "bid_up_events",
    "bid_down_events",
    "ask_up_events",
    "ask_down_events",
    "mid_up_events",
    "mid_down_events",
    "spread_tighten_events",
    "spread_widen_events",
    "trade_count",
    "trade_volume",
    "trade_notional",
    "buy_volume",
    "sell_volume",
    "unknown_volume",
    "net_buy_volume",
    "buy_ratio",
    "last_trade_price",
    "mid_std",
]


def infer_side(ticker: str) -> str:
    m = re.search(r"[0-9]([CP])[0-9]{8}$", str(ticker))
    if not m:
        return ""
    return "CALL" if m.group(1) == "C" else "PUT"


def infer_strike(ticker: str) -> float:
    m = re.search(r"[CP](\d{8})$", str(ticker))
    return float(m.group(1)) / 1000.0 if m else np.nan


def convert_day(raw: pd.DataFrame, *, symbol: str, selected_dte: float | None) -> pd.DataFrame:
    df = raw.copy()
    if "timestamp" not in df.columns and "ts" in df.columns:
        df["timestamp"] = pd.to_datetime(df["ts"], unit="s", utc=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(NY)
    df = df[
        (df["timestamp"].dt.time >= pd.Timestamp("09:30").time())
        & (df["timestamp"].dt.time < pd.Timestamp("16:00").time())
    ].copy()
    if df.empty:
        return df

    for col in ["bid", "ask", "bid_size", "ask_size", "bucket_id", "strike"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "ticker" not in df.columns and "contract_symbol" in df.columns:
        df["ticker"] = df["contract_symbol"].astype(str).str.replace("^O:", "", regex=True)
    df["ticker"] = df["ticker"].astype(str).str.replace("^O:", "", regex=True)
    df["contract_symbol"] = df.get("contract_symbol", df["ticker"])
    df["contract_symbol"] = df["contract_symbol"].astype(str).map(
        lambda x: x if x.startswith("O:") else f"O:{x}"
    )

    if "side" not in df.columns or df["side"].astype(str).str.upper().isin(["CALL", "PUT"]).mean() < 0.5:
        df["side"] = df["ticker"].map(infer_side)
    else:
        df["side"] = df["side"].astype(str).str.upper()

    if "strike" not in df.columns or df["strike"].isna().all():
        df["strike"] = df["ticker"].map(infer_strike)

    if "mid" not in df.columns:
        mid_src = df["mid_price"] if "mid_price" in df.columns else (df["bid"] + df["ask"]) / 2.0
        df["mid"] = pd.to_numeric(mid_src, errors="coerce")
    df["spread_pct"] = (df["ask"] - df["bid"]) / df["mid"].replace(0, np.nan)
    df["quote_imbalance"] = (df["bid_size"] - df["ask_size"]) / (df["bid_size"] + df["ask_size"]).replace(0, np.nan)

    for col in EVENT_ZERO_COLS:
        if col not in df.columns:
            df[col] = 0.0
    if "tag" not in df.columns:
        df["tag"] = ""
    if "target_abs_delta" not in df.columns:
        df["target_abs_delta"] = np.nan
    if "abs_delta_at_lock" not in df.columns:
        df["abs_delta_at_lock"] = np.nan
    if selected_dte is not None:
        df["selected_dte"] = float(selected_dte)
    elif "selected_dte" not in df.columns and "front_dte" in df.columns:
        df["selected_dte"] = pd.to_numeric(df["front_dte"], errors="coerce")
    elif "selected_dte" not in df.columns:
        df["selected_dte"] = np.nan

    df["underlying"] = symbol.upper()
    keep = [
        "timestamp",
        "bid",
        "ask",
        "bid_size",
        "ask_size",
        "mid",
        "spread_pct",
        "quote_imbalance",
        *EVENT_ZERO_COLS,
        "ticker",
        "contract_symbol",
        "bucket_id",
        "tag",
        "side",
        "strike",
        "target_abs_delta",
        "abs_delta_at_lock",
        "selected_dte",
        "underlying",
    ]
    out = df[[c for c in keep if c in df.columns]].sort_values(["ticker", "timestamp"])
    return out.drop_duplicates(["ticker", "timestamp"], keep="last")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--raw-root", required=True, help="directory containing {SYMBOL}_YYYY-MM-DD.parquet")
    p.add_argument("--symbol", required=True)
    p.add_argument("--start-date", default="2026-01-01")
    p.add_argument("--end-date", default="2026-03-31")
    p.add_argument("--selected-dte", type=float, default=None, help="stamp selected_dte if raw lacks it")
    p.add_argument("--output-dir", required=True, help="micro root, e.g. /mnt/s990/data/microstructure/qqq_1dte")
    p.add_argument("--force", action="store_true")
    p.add_argument("--report", default="")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    sym = args.symbol.upper()
    raw_root = Path(args.raw_root)
    out_dir = Path(args.output_dir) / "contract_1s" / sym
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(raw_root.glob(f"{sym}_*.parquet"))
    prefix = f"{sym}_"
    files = [p for p in files if args.start_date <= p.stem.replace(prefix, "") <= args.end_date]
    rows = []
    for fp in files:
        date_str = fp.stem.replace(prefix, "")
        dest = out_dir / f"{sym}_{date_str}.parquet"
        if dest.exists() and not args.force:
            rows.append({"date_str": date_str, "status": "skip_exists", "rows": None})
            continue
        raw = pd.read_parquet(fp)
        day = convert_day(raw, symbol=sym, selected_dte=args.selected_dte)
        if day.empty:
            rows.append({"date_str": date_str, "status": "empty", "rows": 0})
            continue
        day.to_parquet(dest, index=False)
        rows.append({"date_str": date_str, "status": "wrote", "rows": int(len(day)), "path": str(dest)})
        print(f"[raw2micro] {date_str} rows={len(day)} -> {dest}", flush=True)

    summary = {
        "symbol": sym,
        "raw_root": str(raw_root),
        "output_dir": str(Path(args.output_dir)),
        "start_date": args.start_date,
        "end_date": args.end_date,
        "selected_dte": args.selected_dte,
        "n_input_files": len(files),
        "n_wrote": int(sum(r["status"] == "wrote" for r in rows)),
        "n_skip": int(sum(r["status"] == "skip_exists" for r in rows)),
        "days": rows,
        "note": "Event/flow columns may be zero-filled from sniper raw; prefer Polygon micro when possible.",
    }
    report = Path(args.report) if args.report else Path(args.output_dir) / "raw2micro_report.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in summary if k != "days"}, indent=2))
    print(f"report -> {report}")


if __name__ == "__main__":
    main()

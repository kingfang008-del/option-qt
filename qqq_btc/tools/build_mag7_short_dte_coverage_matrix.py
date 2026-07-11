#!/usr/bin/env python3
"""Phase 0 coverage matrix: symbol × trading-DTE × data layer.

Does not download data. Scans known roots / locked maps and writes a matrix for
QQQ / NVDA / TSLA × dte∈{0,1,2}.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


NY_DATES = re.compile(r"(\d{4}-\d{2}-\d{2})")


def date_span(files: list[Path], prefix: str) -> dict:
    if not files:
        return {"n_files": 0, "first": None, "last": None, "dates": []}
    dates = []
    for fp in files:
        stem = fp.stem
        if stem.startswith(prefix):
            dates.append(stem[len(prefix) :])
        else:
            m = NY_DATES.search(stem)
            if m:
                dates.append(m.group(1))
    dates = sorted(d for d in dates if NY_DATES.fullmatch(d))
    return {
        "n_files": len(dates),
        "first": dates[0] if dates else None,
        "last": dates[-1] if dates else None,
        "dates": dates,
    }


def scan_glob(root: Path, pattern: str, prefix: str) -> dict:
    if not root.exists():
        return {"exists": False, "n_files": 0, "first": None, "last": None, "dates": []}
    files = sorted(root.glob(pattern))
    span = date_span(files, prefix)
    return {"exists": True, **span}


def locked_info(path: Path, symbol: str, dte: int | None) -> dict:
    if not path.exists():
        return {"exists": False, "n_rows": 0, "n_days": 0, "first": None, "last": None, "dte_values": []}
    df = pd.read_parquet(path)
    if "symbol" in df.columns:
        df = df[df["symbol"].astype(str).str.upper().eq(symbol.upper())]
    dte_col = None
    for c in ("front_dte", "selected_dte", "target_dte", "dte"):
        if c in df.columns:
            dte_col = c
            break
    dte_vals = []
    if dte_col is not None:
        dte_vals = sorted(pd.to_numeric(df[dte_col], errors="coerce").dropna().unique().tolist())
        if dte is not None:
            df = df[pd.to_numeric(df[dte_col], errors="coerce") == dte]
    if "date_str" not in df.columns or df.empty:
        return {
            "exists": True,
            "n_rows": int(len(df)),
            "n_days": 0,
            "first": None,
            "last": None,
            "dte_values": dte_vals,
            "dte_col": dte_col,
        }
    days = sorted(df["date_str"].astype(str).unique())
    return {
        "exists": True,
        "n_rows": int(len(df)),
        "n_days": int(len(days)),
        "first": days[0],
        "last": days[-1],
        "dte_values": dte_vals,
        "dte_col": dte_col,
    }


def infer_expiry_dte_from_ticker(ticker: str, trade_date: str) -> int | None:
    """OCC-ish: ROOT + YYMMDD + C/P + strike. Returns calendar-day approx via business days later."""
    m = re.search(r"[A-Z](\d{6})[CP]\d{8}$", str(ticker).replace("O:", ""))
    if not m:
        return None
    yymmdd = m.group(1)
    expiry = pd.Timestamp(f"20{yymmdd[:2]}-{yymmdd[2:4]}-{yymmdd[4:6]}")
    td = pd.Timestamp(trade_date)
    # trading DTE: count business days from trade_date to expiry (inclusive of expiry session)
    if expiry < td:
        return None
    bdays = pd.bdate_range(td, expiry, freq="C")
    # same day => 0; next business day => 1
    return int(len(bdays) - 1)


def sample_stock_raw_dte_mix(raw_root: Path, symbol: str, sample_dates: list[str]) -> dict:
    """Estimate available trading DTEs from option tickers on sample days."""
    if not raw_root.exists():
        return {"sampled_days": 0, "dte_day_counts": {}}
    counts: dict[str, int] = {}
    used = 0
    for d in sample_dates:
        fp = raw_root / f"{symbol}_{d}.parquet"
        if not fp.exists():
            continue
        try:
            df = pd.read_parquet(fp, columns=["ticker"])
        except Exception:
            df = pd.read_parquet(fp)
            if "ticker" not in df.columns:
                continue
        dtes = set()
        for t in df["ticker"].astype(str).drop_duplicates().head(2000):
            dte = infer_expiry_dte_from_ticker(t, d)
            if dte is not None and dte <= 5:
                dtes.add(dte)
        for dte in dtes:
            key = str(int(dte))
            counts[key] = counts.get(key, 0) + 1
        used += 1
    return {"sampled_days": used, "dte_day_counts": dict(sorted(counts.items(), key=lambda kv: int(kv[0])))}


def build_matrix() -> tuple[pd.DataFrame, dict]:
    rows = []
    specs = [
        {
            "symbol": "QQQ",
            "dte": 0,
            "locked": Path("/home/kingfang007/train_data/locked_targets_map_0dte.parquet"),
            "raw": Path("/mnt/s990/data/raw_1s/options/QQQ"),  # may not exist; 0dte often via micro
            "raw_alt": None,
            "stock": Path("/mnt/s990/data/raw_1s/stocks/QQQ"),
            "micro": Path("/mnt/s990/data/microstructure/qqq_0dte_api_ladder/contract_1s/QQQ"),
        },
        {
            "symbol": "QQQ",
            "dte": 1,
            "locked": Path("/home/kingfang007/train_data/locked_targets_map_1dte.parquet"),
            "raw": Path("/mnt/s990/data/raw_1s/dte1_options/QQQ"),
            "stock": Path("/mnt/s990/data/raw_1s/stocks/QQQ"),
            "micro": Path("/mnt/s990/data/microstructure/qqq_1dte/contract_1s/QQQ"),
        },
        {
            "symbol": "QQQ",
            "dte": 2,
            "locked": Path("/home/kingfang007/train_data/locked_targets_map_2dte_ladder.parquet"),
            "raw": Path("/mnt/s990/data/raw_1s/dte2_options/QQQ"),
            "stock": Path("/mnt/s990/data/raw_1s/stocks/QQQ"),
            "micro": Path("/mnt/s990/data/microstructure/qqq_short_dte/contract_1s/QQQ"),
        },
    ]
    for sym in ("NVDA", "TSLA"):
        for dte in (0, 1, 2):
            specs.append(
                {
                    "symbol": sym,
                    "dte": dte,
                    "locked": Path("/home/kingfang007/train_data/locked_targets_map_stock_0dte.parquet"),
                    "raw": Path(f"/mnt/s990/data/raw_1s/options/{sym}"),
                    "stock": Path(f"/mnt/s990/data/raw_1s/stocks/{sym}"),
                    "micro": Path(f"/mnt/s990/data/microstructure/{sym.lower()}_{dte}dte/contract_1s/{sym}"),
                }
            )

    extras = {}
    for spec in specs:
        sym = spec["symbol"]
        dte = int(spec["dte"])
        prefix = f"{sym}_"
        locked = locked_info(spec["locked"], sym, dte if sym == "QQQ" else None)
        # For stocks, locked map missing or not dte-split: mark accordingly
        if sym != "QQQ":
            locked["note"] = "stock locked map missing or not split by dte"
        raw = scan_glob(spec["raw"], f"{prefix}*.parquet", prefix)
        stock = scan_glob(spec["stock"], f"{prefix}*.parquet", prefix)
        micro = scan_glob(spec["micro"], f"{prefix}*.parquet", prefix)

        # readiness
        ready = bool(locked.get("exists") and locked.get("n_days", 0) > 0 and micro.get("n_files", 0) > 0 and stock.get("n_files", 0) > 0)
        if sym != "QQQ":
            ready = False  # no dte-classified locked map yet

        rows.append(
            {
                "symbol": sym,
                "dte": dte,
                "locked_exists": locked.get("exists"),
                "locked_days": locked.get("n_days"),
                "locked_first": locked.get("first"),
                "locked_last": locked.get("last"),
                "locked_dte_values": locked.get("dte_values"),
                "raw_exists": raw.get("exists"),
                "raw_days": raw.get("n_files"),
                "raw_first": raw.get("first"),
                "raw_last": raw.get("last"),
                "stock_days": stock.get("n_files"),
                "stock_first": stock.get("first"),
                "stock_last": stock.get("last"),
                "micro_days": micro.get("n_files"),
                "micro_first": micro.get("first"),
                "micro_last": micro.get("last"),
                "gate_ready": ready,
                "blocker": (
                    None
                    if ready
                    else (
                        "need dte-split stock locked map"
                        if sym != "QQQ"
                        else (
                            "missing micro"
                            if micro.get("n_files", 0) == 0
                            else ("missing/empty locked" if not locked.get("n_days") else "check stock/raw")
                        )
                    )
                ),
            }
        )

    # sample NVDA/TSLA raw for dynamic dte availability (Fridays + nearby in Jan 2026)
    sample = [
        "2026-01-02",
        "2026-01-05",
        "2026-01-06",
        "2026-01-07",
        "2026-01-08",
        "2026-01-09",
        "2026-01-12",
        "2026-01-13",
        "2026-01-14",
        "2026-01-15",
        "2026-01-16",
        "2026-01-20",
        "2026-01-21",
        "2026-01-22",
        "2026-01-23",
        "2026-01-26",
        "2026-01-27",
        "2026-01-28",
        "2026-01-29",
        "2026-01-30",
    ]
    for sym in ("NVDA", "TSLA"):
        extras[f"{sym}_raw_dte_mix_jan2026"] = sample_stock_raw_dte_mix(
            Path(f"/mnt/s990/data/raw_1s/options/{sym}"),
            sym,
            sample,
        )

    return pd.DataFrame(rows), extras


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default="qqq_btc/results/mag7_short_dte_coverage")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    matrix, extras = build_matrix()
    matrix.to_csv(out / "coverage_matrix.csv", index=False)
    summary = {
        "position_frac_default": 0.25,
        "dte_definition": "trading_dte",
        "symbols": ["QQQ", "NVDA", "TSLA"],
        "dtes": [0, 1, 2],
        "gate_ready_rows": matrix[matrix["gate_ready"]].to_dict("records"),
        "blocked_rows": matrix[~matrix["gate_ready"]][["symbol", "dte", "blocker", "micro_days", "locked_days"]].to_dict(
            "records"
        ),
        "extras": extras,
        "next": "P1: QQQ 0DTE expand-month validation on 2026-01..03 vs Apr-Jun curated",
    }
    (out / "coverage_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(matrix.to_string(index=False))
    print(json.dumps({"extras": extras, "blocked": summary["blocked_rows"]}, indent=2, default=str))
    print(f"results -> {out}")


if __name__ == "__main__":
    main()

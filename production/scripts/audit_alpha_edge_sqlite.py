#!/usr/bin/env python3
"""Audit alpha edge against stock and option 1s data in local SQLite history."""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
from collections import defaultdict
from pathlib import Path

import pandas as pd


DEFAULT_DB_DIR = Path("production/preprocess/backtest/history_sqlite_1s")
HORIZONS = (60, 180, 300)


def _safe_float(value, default=math.nan):
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _parse_contract_type(contract_id: str) -> str | None:
    if not contract_id:
        return None
    marker = contract_id[-9:-8]
    if marker in ("C", "P"):
        return marker
    for ch in reversed(contract_id):
        if ch in ("C", "P"):
            return ch
    return None


def _parse_strike(contract_id: str) -> float | None:
    if not contract_id:
        return None
    raw = contract_id[-8:]
    if raw.isdigit():
        return int(raw) / 1000.0
    return None


def _contracts_from_snapshot(raw_json: str | None) -> list[dict]:
    if not raw_json:
        return []
    try:
        payload = json.loads(raw_json)
    except Exception:
        return []
    buckets = payload.get("buckets") or []
    contracts = payload.get("contracts") or []
    out = []
    for idx, row in enumerate(buckets):
        if idx >= len(contracts) or not isinstance(row, list) or len(row) < 10:
            continue
        contract_id = contracts[idx]
        mid = _safe_float(row[0])
        bid = _safe_float(row[8])
        ask = _safe_float(row[9])
        strike = _safe_float(row[5])
        parsed_strike = _parse_strike(contract_id)
        if math.isfinite(parsed_strike or math.nan):
            strike = parsed_strike
        opt_type = _parse_contract_type(contract_id)
        if opt_type not in ("C", "P") or not math.isfinite(mid) or mid <= 0:
            continue
        out.append(
            {
                "contract_id": contract_id,
                "type": opt_type,
                "strike": strike,
                "mid": mid,
                "bid": bid,
                "ask": ask,
            }
        )
    return out


def _choose_atm_contract(raw_json: str | None, opt_type: str, stock_price: float) -> dict | None:
    contracts = [c for c in _contracts_from_snapshot(raw_json) if c["type"] == opt_type]
    contracts = [c for c in contracts if math.isfinite(c["strike"]) and c["mid"] > 0]
    if not contracts:
        return None
    return min(contracts, key=lambda c: abs(c["strike"] - stock_price))


def _find_contract(raw_json: str | None, contract_id: str) -> dict | None:
    for contract in _contracts_from_snapshot(raw_json):
        if contract["contract_id"] == contract_id:
            return contract
    return None


def _load_day(db_path: Path, min_abs_alpha: float, max_abs_alpha: float | None) -> list[dict]:
    conn = sqlite3.connect(db_path)
    alpha_sql = "SELECT ts, symbol, alpha, vol_z, price FROM alpha_logs WHERE ABS(alpha) >= ?"
    params: list[float] = [min_abs_alpha]
    if max_abs_alpha is not None:
        alpha_sql += " AND ABS(alpha) <= ?"
        params.append(max_abs_alpha)
    df_alpha = pd.read_sql_query(alpha_sql, conn, params=params)
    df_bar = pd.read_sql_query(
        "SELECT symbol, ts, close FROM market_bars_1s ORDER BY symbol, ts",
        conn,
    )
    df_opt = pd.read_sql_query(
        "SELECT symbol, ts, buckets_json FROM option_snapshots_1s ORDER BY symbol, ts",
        conn,
    )
    conn.close()

    bar_by_sym = {
        sym: dict(zip(g["ts"].astype(int), g["close"].astype(float)))
        for sym, g in df_bar.groupby("symbol", sort=False)
    }
    opt_by_sym = {
        sym: dict(zip(g["ts"].astype(int), g["buckets_json"]))
        for sym, g in df_opt.groupby("symbol", sort=False)
    }

    rows = []
    for rec in df_alpha.itertuples(index=False):
        label_ts = int(rec.ts)
        sym = rec.symbol
        alpha = float(rec.alpha)
        direction = 1 if alpha > 0 else -1
        opt_type = "C" if direction > 0 else "P"
        bars = bar_by_sym.get(sym, {})
        opts = opt_by_sym.get(sym, {})
        rows.append(
            _audit_event(
                date=db_path.stem.replace("market_", ""),
                symbol=sym,
                label_ts=label_ts,
                alpha=alpha,
                vol_z=_safe_float(rec.vol_z),
                direction=direction,
                opt_type=opt_type,
                bars=bars,
                opts=opts,
                entry_offset=0,
            )
        )
        rows.append(
            _audit_event(
                date=db_path.stem.replace("market_", ""),
                symbol=sym,
                label_ts=label_ts,
                alpha=alpha,
                vol_z=_safe_float(rec.vol_z),
                direction=direction,
                opt_type=opt_type,
                bars=bars,
                opts=opts,
                entry_offset=60,
            )
        )
    return rows


def _audit_event(
    date: str,
    symbol: str,
    label_ts: int,
    alpha: float,
    vol_z: float,
    direction: int,
    opt_type: str,
    bars: dict[int, float],
    opts: dict[int, str],
    entry_offset: int,
) -> dict:
    entry_ts = label_ts + entry_offset
    stock_entry = bars.get(entry_ts)
    row = {
        "date": date,
        "symbol": symbol,
        "label_ts": label_ts,
        "entry_offset": entry_offset,
        "alpha": alpha,
        "abs_alpha": abs(alpha),
        "vol_z": vol_z,
        "dir": direction,
        "opt_type": opt_type,
        "has_stock_entry": stock_entry is not None,
        "has_option_entry": False,
    }
    if stock_entry is not None and stock_entry > 0:
        for h in HORIZONS:
            future = bars.get(entry_ts + h)
            if future is not None and future > 0:
                row[f"stock_edge_{h}s"] = direction * (future / stock_entry - 1.0)

    if stock_entry is None or stock_entry <= 0:
        return row
    entry_contract = _choose_atm_contract(opts.get(entry_ts), opt_type, stock_entry)
    if not entry_contract:
        return row
    row["has_option_entry"] = True
    row["contract_id"] = entry_contract["contract_id"]
    row["entry_mid"] = entry_contract["mid"]
    row["entry_bid"] = entry_contract["bid"]
    row["entry_ask"] = entry_contract["ask"]
    spread = entry_contract["ask"] - entry_contract["bid"]
    row["entry_spread_pct_mid"] = spread / entry_contract["mid"] if entry_contract["mid"] > 0 else math.nan
    for h in HORIZONS:
        future_contract = _find_contract(opts.get(entry_ts + h), entry_contract["contract_id"])
        if not future_contract:
            continue
        row[f"opt_mid_edge_{h}s"] = future_contract["mid"] / entry_contract["mid"] - 1.0
        if entry_contract["ask"] > 0:
            row[f"opt_ask_bid_edge_{h}s"] = future_contract["bid"] / entry_contract["ask"] - 1.0
    return row


def _summarize(df: pd.DataFrame, label: str, cols: list[str]) -> None:
    print(f"\n[{label}] n={len(df)}")
    for col in cols:
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if s.empty:
            print(f"  {col}: no data")
            continue
        print(
            f"  {col}: mean={s.mean()*100:7.3f}% median={s.median()*100:7.3f}% "
            f"win={(s > 0).mean()*100:5.1f}% p10={s.quantile(.1)*100:7.3f}% p90={s.quantile(.9)*100:7.3f}%"
        )


def _bucket_summary(df: pd.DataFrame, metric: str) -> None:
    if metric not in df.columns:
        return
    data = df.dropna(subset=[metric]).copy()
    if data.empty:
        return
    data["alpha_bucket"] = pd.qcut(data["abs_alpha"], 5, duplicates="drop")
    grouped = data.groupby("alpha_bucket", observed=True)[metric].agg(["count", "mean", "median"])
    print(f"\n[{metric} by abs_alpha quintile]")
    print((grouped * pd.Series({"count": 1, "mean": 100, "median": 100})).to_string())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-dir", type=Path, default=DEFAULT_DB_DIR)
    parser.add_argument("--dates", nargs="*", default=["20260302", "20260303", "20260304", "20260305", "20260306"])
    parser.add_argument("--min-abs-alpha", type=float, default=0.5)
    parser.add_argument("--max-abs-alpha", type=float, default=None)
    args = parser.parse_args()

    all_rows = []
    for date in args.dates:
        db_path = args.db_dir / f"market_{date}.db"
        if not db_path.exists():
            print(f"missing {db_path}")
            continue
        rows = _load_day(db_path, args.min_abs_alpha, args.max_abs_alpha)
        print(f"loaded {date}: {len(rows)} audited rows")
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    if df.empty:
        print("no rows")
        return 1

    print(f"\nTotal rows: {len(df)}  events={len(df)//2}  min_abs_alpha={args.min_abs_alpha}")
    print("Coverage:")
    print(df.groupby("entry_offset")[["has_stock_entry", "has_option_entry"]].mean().mul(100).round(2).to_string())

    metrics = []
    for h in HORIZONS:
        metrics.extend([f"stock_edge_{h}s", f"opt_mid_edge_{h}s", f"opt_ask_bid_edge_{h}s"])
    for offset, name in [(0, "label_ts"), (60, "available_ts_plus_60s")]:
        sub = df[df["entry_offset"] == offset]
        _summarize(sub, name, metrics)
        _bucket_summary(sub, "stock_edge_300s")
        _bucket_summary(sub, "opt_mid_edge_300s")
        _bucket_summary(sub, "opt_ask_bid_edge_300s")

    opt = df[df["has_option_entry"]].copy()
    if not opt.empty:
        print("\n[Entry spread pct mid]")
        print(
            opt.groupby("entry_offset")["entry_spread_pct_mid"]
            .agg(["count", "mean", "median", lambda s: s.quantile(.9)])
            .rename(columns={"<lambda_0>": "p90"})
            .mul(pd.Series({"count": 1, "mean": 100, "median": 100, "p90": 100}))
            .round(3)
            .to_string()
        )
        print("\n[Worst symbols: available + ask_to_bid 300s]")
        metric = "opt_ask_bid_edge_300s"
        sub = opt[(opt["entry_offset"] == 60) & opt[metric].notna()]
        by_sym = sub.groupby("symbol")[metric].agg(["count", "mean", "median"])
        by_sym = by_sym[by_sym["count"] >= 50].sort_values("mean").head(12)
        print((by_sym * pd.Series({"count": 1, "mean": 100, "median": 100})).round(3).to_string())
        print("\n[Best symbols: available + ask_to_bid 300s]")
        print((sub.groupby("symbol")[metric].agg(["count", "mean", "median"]).query("count >= 50").sort_values("mean", ascending=False).head(12) * pd.Series({"count": 1, "mean": 100, "median": 100})).round(3).to_string())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

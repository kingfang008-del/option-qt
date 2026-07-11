#!/usr/bin/env python3
"""Download selected short-DTE option quote/trade ticks and build 1s microstructure features.

This intentionally downloads only contracts already selected in a locked map.  Full OPRA
quote flat files are too large for iterative research (100GB+ compressed per day), while
the REST quotes/trades endpoints are practical for a small 0DTE/2DTE ladder universe.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import json
import logging
import os
import re
import ast
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from polygon import RESTClient
from tqdm import tqdm

try:
    from preprocess.download.step2_polygon_second_sniper_v1 import API_KEY as LEGACY_API_KEY
except Exception:
    LEGACY_API_KEY = ""


def load_legacy_api_key() -> str:
    """Read the existing downloader's API key fallback without duplicating it here."""
    if LEGACY_API_KEY:
        return LEGACY_API_KEY
    legacy_path = Path(__file__).with_name("step2_polygon_second_sniper_v1.py")
    try:
        tree = ast.parse(legacy_path.read_text(encoding="utf-8"))
        for node in tree.body:
            if not isinstance(node, ast.Assign):
                continue
            if not any(isinstance(t, ast.Name) and t.id == "API_KEY" for t in node.targets):
                continue
            # Handles: API_KEY = os.environ.get("POLYGON_API_KEY", "...")
            val = node.value
            if isinstance(val, ast.Call) and len(val.args) >= 2 and isinstance(val.args[1], ast.Constant):
                return str(val.args[1].value)
            if isinstance(val, ast.Constant):
                return str(val.value)
    except Exception:
        return ""
    return ""


API_KEY = os.environ.get("POLYGON_API_KEY") or load_legacy_api_key()
EASTERN = "America/New_York"
RTH_START = dt.time(9, 30)
RTH_END = dt.time(16, 0)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logging.getLogger("urllib3").setLevel(logging.ERROR)
logger = logging.getLogger("short_dte_microstructure")


def normalize_contract(x: Any) -> str:
    return str(x).replace("O:", "")


def polygon_contract(x: Any) -> str:
    s = str(x)
    return s if s.startswith("O:") else f"O:{s}"


def parse_strike(ticker: str) -> float:
    m = re.search(r"[CP](\d{8})$", normalize_contract(ticker))
    return float(m.group(1)) / 1000.0 if m else np.nan


def to_eastern_ns(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, unit="ns", utc=True).dt.tz_convert(EASTERN)


def in_rth(ts: pd.Series) -> pd.Series:
    t = ts.dt.time
    return (t >= RTH_START) & (t < RTH_END)


def safe_get(obj: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        value = getattr(obj, name, None)
        if value is not None:
            return value
    return default


def fetch_quotes(client: RESTClient, ticker: str, date_str: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for q in client.list_quotes(
        ticker=polygon_contract(ticker),
        timestamp_gte=date_str,
        timestamp_lte=date_str,
        limit=50000,
    ):
        rows.append(
            {
                "sip_timestamp": safe_get(q, "sip_timestamp", "participant_timestamp", default=0),
                "bid": safe_get(q, "bid_price", default=np.nan),
                "ask": safe_get(q, "ask_price", default=np.nan),
                "bid_size": safe_get(q, "bid_size", default=np.nan),
                "ask_size": safe_get(q, "ask_size", default=np.nan),
                "bid_exchange": safe_get(q, "bid_exchange", default=None),
                "ask_exchange": safe_get(q, "ask_exchange", default=None),
            }
        )
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df = df[(df["sip_timestamp"] > 0) & (df["bid"] > 0) & (df["ask"] >= df["bid"])].copy()
    if df.empty:
        return df
    df["timestamp"] = to_eastern_ns(df["sip_timestamp"])
    df = df[in_rth(df["timestamp"])].copy()
    if df.empty:
        return df
    df["ticker"] = normalize_contract(ticker)
    df = df.sort_values("timestamp")
    df["mid"] = (df["bid"] + df["ask"]) / 2.0
    df["spread"] = df["ask"] - df["bid"]
    df["spread_pct"] = df["spread"] / df["mid"].replace(0, np.nan)
    size_sum = df["bid_size"].astype(float) + df["ask_size"].astype(float)
    df["quote_imbalance"] = (df["bid_size"].astype(float) - df["ask_size"].astype(float)) / size_sum.replace(0, np.nan)
    return df


def fetch_trades(client: RESTClient, ticker: str, date_str: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for tr in client.list_trades(
        ticker=polygon_contract(ticker),
        timestamp_gte=date_str,
        timestamp_lte=date_str,
        limit=50000,
    ):
        rows.append(
            {
                "sip_timestamp": safe_get(tr, "sip_timestamp", "participant_timestamp", default=0),
                "price": safe_get(tr, "price", default=np.nan),
                "size": safe_get(tr, "size", default=np.nan),
                "exchange": safe_get(tr, "exchange", default=None),
                "conditions": safe_get(tr, "conditions", default=None),
            }
        )
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df = df[(df["sip_timestamp"] > 0) & (df["price"] > 0) & (df["size"] > 0)].copy()
    if df.empty:
        return df
    df["timestamp"] = to_eastern_ns(df["sip_timestamp"])
    df = df[in_rth(df["timestamp"])].copy()
    if df.empty:
        return df
    df["ticker"] = normalize_contract(ticker)
    return df.sort_values("timestamp")


def quote_1s_features(quotes: pd.DataFrame) -> pd.DataFrame:
    if quotes.empty:
        return pd.DataFrame()
    work = quotes.sort_values("timestamp").copy()
    work["ts_1s"] = work["timestamp"].dt.floor("1s")
    for col in ["bid", "ask", "mid", "spread_pct", "bid_size", "ask_size"]:
        work[f"d_{col}"] = work[col].diff()
    work["bid_up_event"] = (work["d_bid"] > 0).astype(int)
    work["bid_down_event"] = (work["d_bid"] < 0).astype(int)
    work["ask_up_event"] = (work["d_ask"] > 0).astype(int)
    work["ask_down_event"] = (work["d_ask"] < 0).astype(int)
    work["mid_up_event"] = (work["d_mid"] > 0).astype(int)
    work["mid_down_event"] = (work["d_mid"] < 0).astype(int)
    work["spread_tighten_event"] = (work["d_spread_pct"] < 0).astype(int)
    work["spread_widen_event"] = (work["d_spread_pct"] > 0).astype(int)

    last_cols = ["bid", "ask", "bid_size", "ask_size", "mid", "spread_pct", "quote_imbalance"]
    last = work.drop_duplicates("ts_1s", keep="last").set_index("ts_1s")[last_cols]
    agg = work.groupby("ts_1s").agg(
        quote_events=("timestamp", "size"),
        bid_up_events=("bid_up_event", "sum"),
        bid_down_events=("bid_down_event", "sum"),
        ask_up_events=("ask_up_event", "sum"),
        ask_down_events=("ask_down_event", "sum"),
        mid_up_events=("mid_up_event", "sum"),
        mid_down_events=("mid_down_event", "sum"),
        spread_tighten_events=("spread_tighten_event", "sum"),
        spread_widen_events=("spread_widen_event", "sum"),
        mid_std=("mid", "std"),
    )
    out = last.join(agg, how="outer").reset_index().rename(columns={"ts_1s": "timestamp"})
    out["mid_std"] = out["mid_std"].fillna(0.0)
    return out


def trade_1s_features(trades: pd.DataFrame, quotes: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()

    work = trades.sort_values("timestamp").copy()
    if not quotes.empty:
        q = quotes[["timestamp", "bid", "ask", "mid"]].sort_values("timestamp")
        work = pd.merge_asof(work, q, on="timestamp", direction="backward", tolerance=pd.Timedelta("3s"))
        eps = 1e-9
        work["aggressor"] = np.where(
            work["price"] >= work["ask"].fillna(np.inf) - eps,
            1,
            np.where(work["price"] <= work["bid"].fillna(-np.inf) + eps, -1, 0),
        )
    else:
        work["aggressor"] = np.sign(work["price"].diff()).fillna(0.0).astype(int)

    work["ts_1s"] = work["timestamp"].dt.floor("1s")
    work["buy_size"] = np.where(work["aggressor"] > 0, work["size"], 0.0)
    work["sell_size"] = np.where(work["aggressor"] < 0, work["size"], 0.0)
    work["unknown_size"] = np.where(work["aggressor"] == 0, work["size"], 0.0)
    work["notional"] = work["price"] * work["size"]

    out = work.groupby("ts_1s").agg(
        trade_count=("timestamp", "size"),
        trade_volume=("size", "sum"),
        trade_notional=("notional", "sum"),
        buy_volume=("buy_size", "sum"),
        sell_volume=("sell_size", "sum"),
        unknown_volume=("unknown_size", "sum"),
        last_trade_price=("price", "last"),
    )
    out["net_buy_volume"] = out["buy_volume"] - out["sell_volume"]
    out["buy_ratio"] = out["buy_volume"] / out["trade_volume"].replace(0, np.nan)
    return out.reset_index().rename(columns={"ts_1s": "timestamp"})


def build_contract_1s(row: pd.Series, client: RESTClient) -> tuple[pd.DataFrame, dict[str, Any]]:
    ticker = normalize_contract(row["contract_symbol"])
    date_str = str(row["date_str"])
    summary: dict[str, Any] = {
        "date_str": date_str,
        "ticker": ticker,
        "bucket_id": int(row["bucket_id"]),
        "side": str(row.get("side", "")),
        "quote_events": 0,
        "trade_events": 0,
        "rows_1s": 0,
        "error": None,
    }
    try:
        quotes = fetch_quotes(client, ticker, date_str)
        trades = fetch_trades(client, ticker, date_str)
        summary["quote_events"] = int(len(quotes))
        summary["trade_events"] = int(len(trades))

        q1 = quote_1s_features(quotes)
        t1 = trade_1s_features(trades, quotes)
        if q1.empty and t1.empty:
            return pd.DataFrame(), summary

        if q1.empty:
            out = t1
        elif t1.empty:
            out = q1
        else:
            out = pd.merge(q1, t1, on="timestamp", how="outer")

        out = out.sort_values("timestamp").reset_index(drop=True)
        out["ticker"] = ticker
        out["contract_symbol"] = row["contract_symbol"]
        out["bucket_id"] = int(row["bucket_id"])
        out["tag"] = row.get("tag", "")
        out["side"] = row.get("side", "")
        out["strike"] = float(row.get("strike", parse_strike(ticker)))
        out["target_abs_delta"] = float(row.get("target_abs_delta", np.nan))
        out["abs_delta_at_lock"] = float(row.get("abs_delta_at_lock", np.nan))
        out["selected_dte"] = float(row.get("selected_dte", row.get("target_dte", np.nan)))

        fill_zero = [
            "quote_events", "bid_up_events", "bid_down_events", "ask_up_events", "ask_down_events",
            "mid_up_events", "mid_down_events", "spread_tighten_events", "spread_widen_events",
            "trade_count", "trade_volume", "trade_notional", "buy_volume", "sell_volume",
            "unknown_volume", "net_buy_volume",
        ]
        for col in fill_zero:
            if col in out.columns:
                out[col] = out[col].fillna(0.0)
        summary["rows_1s"] = int(len(out))
        return out, summary
    except Exception as exc:
        summary["error"] = str(exc)
        return pd.DataFrame(), summary


def aggregate_cross_contract(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    work = df.copy()
    work["side_norm"] = work["side"].astype(str).str.upper()
    work["delta_weight"] = pd.to_numeric(work["abs_delta_at_lock"], errors="coerce").fillna(
        pd.to_numeric(work["target_abs_delta"], errors="coerce")
    ).fillna(1.0)
    for col in ["net_buy_volume", "trade_volume", "buy_volume", "sell_volume", "quote_imbalance", "quote_events", "spread_pct"]:
        if col not in work.columns:
            work[col] = 0.0
        work[col] = pd.to_numeric(work[col], errors="coerce").fillna(0.0)
    work["weighted_net_buy"] = work["net_buy_volume"] * work["delta_weight"]
    work["weighted_quote_imb"] = work["quote_imbalance"] * work["delta_weight"]
    work["weighted_spread"] = work["spread_pct"] * work["quote_events"].clip(lower=1)

    pieces: list[pd.DataFrame] = []
    for side, prefix in [("CALL", "call"), ("PUT", "put")]:
        sub = work[work["side_norm"] == side]
        if sub.empty:
            continue
        g = sub.groupby("timestamp").agg(
            **{
                f"{prefix}_trade_volume": ("trade_volume", "sum"),
                f"{prefix}_buy_volume": ("buy_volume", "sum"),
                f"{prefix}_sell_volume": ("sell_volume", "sum"),
                f"{prefix}_net_buy_volume": ("net_buy_volume", "sum"),
                f"{prefix}_weighted_net_buy": ("weighted_net_buy", "sum"),
                f"{prefix}_quote_events": ("quote_events", "sum"),
                f"{prefix}_weighted_quote_imb": ("weighted_quote_imb", "sum"),
                f"{prefix}_spread_num": ("weighted_spread", "sum"),
                f"{prefix}_spread_den": ("quote_events", "sum"),
            }
        )
        g[f"{prefix}_buy_ratio"] = g[f"{prefix}_buy_volume"] / g[f"{prefix}_trade_volume"].replace(0, np.nan)
        g[f"{prefix}_quote_imbalance"] = g[f"{prefix}_weighted_quote_imb"] / g[f"{prefix}_quote_events"].replace(0, np.nan)
        g[f"{prefix}_spread_pct"] = g[f"{prefix}_spread_num"] / g[f"{prefix}_spread_den"].replace(0, np.nan)
        g = g.drop(columns=[f"{prefix}_spread_num", f"{prefix}_spread_den"])
        pieces.append(g)

    if not pieces:
        return pd.DataFrame()
    out = pd.concat(pieces, axis=1).sort_index().reset_index()
    for col in out.columns:
        if col != "timestamp":
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
    out["cp_net_buy_diff"] = out.get("call_net_buy_volume", 0.0) - out.get("put_net_buy_volume", 0.0)
    out["cp_weighted_net_buy_diff"] = out.get("call_weighted_net_buy", 0.0) - out.get("put_weighted_net_buy", 0.0)
    out["cp_buy_ratio_diff"] = out.get("call_buy_ratio", 0.0) - out.get("put_buy_ratio", 0.0)
    out["cp_quote_imbalance_diff"] = out.get("call_quote_imbalance", 0.0) - out.get("put_quote_imbalance", 0.0)
    out["cp_quote_event_diff"] = out.get("call_quote_events", 0.0) - out.get("put_quote_events", 0.0)
    out["cp_spread_diff"] = out.get("call_spread_pct", 0.0) - out.get("put_spread_pct", 0.0)
    return out


def process_day(day_df: pd.DataFrame, args: argparse.Namespace) -> dict[str, Any]:
    date_str = str(day_df["date_str"].iloc[0])
    symbol = str(day_df["symbol"].iloc[0])
    contract_dir = Path(args.output_dir) / "contract_1s" / symbol
    feature_dir = Path(args.output_dir) / "features_1s" / symbol
    contract_dir.mkdir(parents=True, exist_ok=True)
    feature_dir.mkdir(parents=True, exist_ok=True)
    contract_path = contract_dir / f"{symbol}_{date_str}.parquet"
    feature_path = feature_dir / f"{symbol}_{date_str}.parquet"
    report: dict[str, Any] = {
        "date_str": date_str,
        "symbol": symbol,
        "contracts": int(day_df["contract_symbol"].nunique()),
        "contract_path": str(contract_path),
        "feature_path": str(feature_path),
        "downloaded_contracts": 0,
        "contract_rows_1s": 0,
        "feature_rows_1s": 0,
        "quote_events": 0,
        "trade_events": 0,
        "errors": [],
        "skipped": False,
    }
    if contract_path.exists() and feature_path.exists() and not args.force:
        report["skipped"] = True
        return report

    client = RESTClient(args.api_key)
    frames: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []
    for _, row in day_df.iterrows():
        frame, summary = build_contract_1s(row, client)
        summaries.append(summary)
        if summary.get("error"):
            report["errors"].append(summary)
        if not frame.empty:
            frames.append(frame)

    if frames:
        contract_df = pd.concat(frames, ignore_index=True).sort_values(["timestamp", "bucket_id", "ticker"])
        contract_df.to_parquet(contract_path, index=False, compression="zstd")
        feature_df = aggregate_cross_contract(contract_df)
        feature_df.to_parquet(feature_path, index=False, compression="zstd")
        report["downloaded_contracts"] = int(contract_df["ticker"].nunique())
        report["contract_rows_1s"] = int(len(contract_df))
        report["feature_rows_1s"] = int(len(feature_df))

    report["quote_events"] = int(sum(x.get("quote_events", 0) for x in summaries))
    report["trade_events"] = int(sum(x.get("trade_events", 0) for x in summaries))
    return report


def load_target_map(args: argparse.Namespace) -> pd.DataFrame:
    df = pd.read_parquet(args.locked_map)
    required = {"date_str", "contract_symbol", "bucket_id", "symbol"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"locked map missing columns: {sorted(missing)}")
    df = df[df["symbol"].astype(str) == args.symbol].copy()
    df = df[(df["date_str"].astype(str) >= args.start_date) & (df["date_str"].astype(str) <= args.end_date)].copy()
    if args.max_dte is not None:
        dte_col = "selected_dte" if "selected_dte" in df.columns else "target_dte"
        df = df[pd.to_numeric(df[dte_col], errors="coerce") <= args.max_dte].copy()
    if args.bucket_ids:
        keep = {int(x) for x in args.bucket_ids.split(",") if str(x).strip()}
        df = df[df["bucket_id"].astype(int).isin(keep)].copy()
    if df.empty:
        raise ValueError("target map has no rows after filters")
    df["contract_symbol"] = df["contract_symbol"].map(polygon_contract)
    df = df.drop_duplicates(["date_str", "contract_symbol", "bucket_id"]).sort_values(["date_str", "bucket_id"])
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download short-DTE selected option microstructure data")
    parser.add_argument("--locked-map", default=str(Path.home() / "train_data/locked_targets_map_2dte_ladder_202204_202606.parquet"))
    parser.add_argument("--symbol", default="QQQ")
    parser.add_argument("--start-date", default="2026-04-01")
    parser.add_argument("--end-date", default="2026-06-30")
    parser.add_argument("--output-dir", default="/mnt/s990/data/microstructure/qqq_short_dte")
    parser.add_argument("--report", default="qqq_btc/results/short_dte_microstructure_download_report.json")
    parser.add_argument("--max-dte", type=int, default=2)
    parser.add_argument("--bucket-ids", default="", help="optional comma list, e.g. 0,1,2,3,4,5,6,7")
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--api-key", default=API_KEY)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.api_key:
        raise SystemExit("POLYGON_API_KEY is not set and legacy key could not be imported.")
    target = load_target_map(args)
    days = [(d, g.copy()) for d, g in target.groupby("date_str")]
    logger.info(
        "microstructure download | symbol=%s days=%d contract_rows=%d output=%s",
        args.symbol,
        len(days),
        len(target),
        args.output_dir,
    )

    reports: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        futs = {ex.submit(process_day, g, args): d for d, g in days}
        for fut in tqdm(concurrent.futures.as_completed(futs), total=len(futs), desc="microstructure days"):
            reports.append(fut.result())

    reports = sorted(reports, key=lambda x: x["date_str"])
    summary = {
        "symbol": args.symbol,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "locked_map": args.locked_map,
        "output_dir": args.output_dir,
        "days": len(reports),
        "skipped_days": int(sum(r["skipped"] for r in reports)),
        "downloaded_contracts_sum": int(sum(r["downloaded_contracts"] for r in reports)),
        "contract_rows_1s": int(sum(r["contract_rows_1s"] for r in reports)),
        "feature_rows_1s": int(sum(r["feature_rows_1s"] for r in reports)),
        "quote_events": int(sum(r["quote_events"] for r in reports)),
        "trade_events": int(sum(r["trade_events"] for r in reports)),
        "error_contracts": int(sum(len(r["errors"]) for r in reports)),
    }
    payload = {"summary": summary, "days_detail": reports}
    out = Path(args.report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"report -> {out}")


if __name__ == "__main__":
    main()

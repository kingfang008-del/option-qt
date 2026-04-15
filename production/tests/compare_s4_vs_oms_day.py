#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASELINE_DIR = PROJECT_ROOT / "baseline"
if str(BASELINE_DIR) not in sys.path:
    sys.path.insert(0, str(BASELINE_DIR))

from config import COMMISSION_PER_CONTRACT, NY_TZ, PG_DB_URL  # type: ignore

try:
    import psycopg2
except Exception:  # pragma: no cover
    psycopg2 = None


@dataclass
class CompareConfig:
    date_str: str
    time_tolerance_sec: float


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="对比 S4 PG 1s MockIBKR 账本 与 OMS/backtest 账本的一天交易结果。"
    )
    p.add_argument("--date", required=True, help="YYYYMMDD")
    p.add_argument(
        "--s4-csv",
        default=str(Path.home() / "quant_project" / "logs" / "replay_trades_s2_pg_1s.csv"),
        help="s4_run_historical_replay_pg_1s.py 输出的 MockIBKR CSV",
    )
    p.add_argument(
        "--oms-source",
        choices=["pg", "csv"],
        default="pg",
        help="OMS 对比源: PostgreSQL trade_logs_backtest 或本地 CSV",
    )
    p.add_argument(
        "--oms-csv",
        default=str(Path.home() / "quant_project" / "logs" / "replay_trades_v8.csv"),
        help="若 --oms-source=csv，则读取该 CSV",
    )
    p.add_argument(
        "--out-csv",
        default="",
        help="逐笔对比结果输出路径，默认 logs/compare_s4_vs_oms_YYYYMMDD.csv",
    )
    p.add_argument(
        "--time-tolerance-sec",
        type=float,
        default=1.0,
        help="逐笔配对允许的 entry/exit 时间误差秒数",
    )
    return p.parse_args()


def _safe_json_loads(v) -> dict:
    if isinstance(v, dict):
        return v
    if not isinstance(v, str) or not v:
        return {}
    try:
        return json.loads(v)
    except Exception:
        return {}


def _safe_float(v, default=0.0) -> float:
    try:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return float(default)
        return float(v)
    except Exception:
        return float(default)


def _safe_int(v, default=0) -> int:
    try:
        return int(float(v))
    except Exception:
        return int(default)


def _date_window(date_str: str) -> tuple[float, float]:
    dt_ny = NY_TZ.localize(pd.Timestamp(f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]} 00:00:00").to_pydatetime())
    start_ts = float(dt_ny.timestamp())
    return start_ts, start_ts + 86400.0


def _normalize_trade_df(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=[
                "source", "symbol", "entry_ts", "exit_ts", "entry_price", "exit_price",
                "qty", "pnl", "roi", "reason", "opt_dir", "entry_stock", "exit_stock",
            ]
        )

    out = df.copy()
    out["source"] = source_name
    for col in ["entry_ts", "exit_ts", "entry_price", "exit_price", "qty", "pnl", "roi", "entry_stock", "exit_stock"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
        else:
            out[col] = 0.0
    for col in ["symbol", "reason", "opt_dir"]:
        if col not in out.columns:
            out[col] = ""
        out[col] = out[col].fillna("").astype(str)
    out = out.sort_values(["symbol", "entry_ts", "exit_ts"]).reset_index(drop=True)
    return out


def load_s4_csv(path: str, date_str: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"S4 CSV not found: {p}")
    df = pd.read_csv(p)
    if "date" in df.columns:
        target_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
        df = df[df["date"].astype(str) == target_date].copy()
    elif "exit_ts" in df.columns:
        start_ts, end_ts = _date_window(date_str)
        df = df[(pd.to_numeric(df["exit_ts"], errors="coerce") >= start_ts) & (pd.to_numeric(df["exit_ts"], errors="coerce") < end_ts)].copy()
    return _normalize_trade_df(df, "S4")


def _extract_open_close_events_from_pg(date_str: str) -> pd.DataFrame:
    if psycopg2 is None:
        raise RuntimeError("psycopg2 is unavailable, cannot read PostgreSQL source.")

    start_ts, end_ts = _date_window(date_str)
    conn = psycopg2.connect(PG_DB_URL)
    try:
        sql = """
            SELECT ts, datetime_ny, symbol, action, qty, price, details_json
            FROM trade_logs_backtest
            WHERE ts >= %s AND ts < %s
              AND action IN ('OPEN', 'CLOSE')
            ORDER BY ts, symbol
        """
        df = pd.read_sql(sql, conn, params=(start_ts, end_ts))
    finally:
        conn.close()
    return df


def _rebuild_round_trips_from_events(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()

    open_books: dict[str, deque] = defaultdict(deque)
    rows: list[dict] = []

    for rec in events.itertuples(index=False):
        details = _safe_json_loads(getattr(rec, "details_json", ""))
        strategy_note = _safe_json_loads(details.get("strategy_note", ""))
        symbol = str(getattr(rec, "symbol", ""))
        action = str(getattr(rec, "action", "")).upper()
        qty = _safe_float(getattr(rec, "qty", 0.0))
        px = _safe_float(getattr(rec, "price", 0.0))
        ts = _safe_float(getattr(rec, "ts", 0.0))
        stock_px = _safe_float(details.get("stock_price", 0.0))
        tag = str(strategy_note.get("tag", "") or details.get("tag", "") or "")
        opt_dir = "CALL" if "CALL" in tag.upper() else ("PUT" if "PUT" in tag.upper() else "")

        if action == "OPEN":
            open_books[symbol].append(
                {
                    "qty": qty,
                    "entry_ts": ts,
                    "entry_price": px,
                    "entry_stock": stock_px,
                    "tag": tag,
                    "opt_dir": opt_dir,
                    "reason": str(strategy_note.get("reason", "")),
                }
            )
            continue

        if action != "CLOSE":
            continue

        remaining = qty
        close_reason = str(strategy_note.get("reason", ""))
        while remaining > 1e-9 and open_books[symbol]:
            op = open_books[symbol][0]
            matched_qty = min(_safe_float(op["qty"]), remaining)
            entry_price = _safe_float(op["entry_price"])
            exit_price = px
            cost = entry_price * matched_qty * 100.0 + matched_qty * COMMISSION_PER_CONTRACT
            proceeds = exit_price * matched_qty * 100.0 - matched_qty * COMMISSION_PER_CONTRACT
            pnl = proceeds - cost
            roi = ((exit_price - entry_price) * matched_qty * 100.0 / cost) if cost > 0 else 0.0
            rows.append(
                {
                    "symbol": symbol,
                    "entry_ts": _safe_float(op["entry_ts"]),
                    "exit_ts": ts,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "qty": matched_qty,
                    "pnl": pnl,
                    "roi": roi,
                    "reason": close_reason,
                    "opt_dir": op.get("opt_dir", ""),
                    "entry_stock": _safe_float(op.get("entry_stock", 0.0)),
                    "exit_stock": stock_px,
                }
            )
            remaining -= matched_qty
            op["qty"] = _safe_float(op["qty"]) - matched_qty
            if _safe_float(op["qty"]) <= 1e-9:
                open_books[symbol].popleft()

    return pd.DataFrame(rows)


def load_oms_from_pg(date_str: str) -> pd.DataFrame:
    events = _extract_open_close_events_from_pg(date_str)
    df = _rebuild_round_trips_from_events(events)
    return _normalize_trade_df(df, "OMS_PG")


def load_oms_from_csv(path: str, date_str: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"OMS CSV not found: {p}")
    df = pd.read_csv(p)
    if "date" in df.columns:
        target_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
        df = df[df["date"].astype(str) == target_date].copy()
    elif "exit_ts" in df.columns:
        start_ts, end_ts = _date_window(date_str)
        df = df[(pd.to_numeric(df["exit_ts"], errors="coerce") >= start_ts) & (pd.to_numeric(df["exit_ts"], errors="coerce") < end_ts)].copy()
    return _normalize_trade_df(df, "OMS_CSV")


def _summary(df: pd.DataFrame) -> dict:
    trade_count = int(len(df))
    total_pnl = _safe_float(df["pnl"].sum()) if trade_count else 0.0
    total_qty = _safe_float(df["qty"].sum()) if trade_count else 0.0
    win_rate = float((df["pnl"] > 0).mean()) if trade_count else 0.0
    avg_pnl = total_pnl / trade_count if trade_count else 0.0
    return {
        "trades": trade_count,
        "qty": total_qty,
        "total_pnl": total_pnl,
        "win_rate": win_rate,
        "avg_pnl": avg_pnl,
    }


def _per_symbol_summary(df: pd.DataFrame, label: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["symbol", f"{label}_trades", f"{label}_pnl", f"{label}_win_rate"])
    grp = df.groupby("symbol", as_index=False).agg(
        trades=("symbol", "count"),
        pnl=("pnl", "sum"),
        win_rate=("pnl", lambda s: float((s > 0).mean()) if len(s) else 0.0),
    )
    return grp.rename(
        columns={
            "trades": f"{label}_trades",
            "pnl": f"{label}_pnl",
            "win_rate": f"{label}_win_rate",
        }
    )


def pair_trades(s4_df: pd.DataFrame, oms_df: pd.DataFrame, cfg: CompareConfig) -> pd.DataFrame:
    oms_unused = set(oms_df.index.tolist())
    rows: list[dict] = []

    for s4_idx, s4_row in s4_df.iterrows():
        candidates = []
        for oms_idx in list(oms_unused):
            oms_row = oms_df.loc[oms_idx]
            if s4_row["symbol"] != oms_row["symbol"]:
                continue
            if abs(_safe_float(s4_row["qty"]) - _safe_float(oms_row["qty"])) > 1e-9:
                continue
            entry_diff = abs(_safe_float(s4_row["entry_ts"]) - _safe_float(oms_row["entry_ts"]))
            exit_diff = abs(_safe_float(s4_row["exit_ts"]) - _safe_float(oms_row["exit_ts"]))
            if entry_diff <= cfg.time_tolerance_sec and exit_diff <= cfg.time_tolerance_sec:
                candidates.append((entry_diff + exit_diff, entry_diff, exit_diff, oms_idx))

        if candidates:
            candidates.sort()
            _, entry_diff, exit_diff, oms_idx = candidates[0]
            oms_unused.remove(oms_idx)
            oms_row = oms_df.loc[oms_idx]
            rows.append(
                {
                    "match_status": "matched",
                    "symbol": s4_row["symbol"],
                    "qty": _safe_float(s4_row["qty"]),
                    "entry_ts_s4": _safe_float(s4_row["entry_ts"]),
                    "exit_ts_s4": _safe_float(s4_row["exit_ts"]),
                    "entry_ts_oms": _safe_float(oms_row["entry_ts"]),
                    "exit_ts_oms": _safe_float(oms_row["exit_ts"]),
                    "entry_time_diff_sec": entry_diff,
                    "exit_time_diff_sec": exit_diff,
                    "entry_price_s4": _safe_float(s4_row["entry_price"]),
                    "exit_price_s4": _safe_float(s4_row["exit_price"]),
                    "entry_price_oms": _safe_float(oms_row["entry_price"]),
                    "exit_price_oms": _safe_float(oms_row["exit_price"]),
                    "pnl_s4": _safe_float(s4_row["pnl"]),
                    "pnl_oms": _safe_float(oms_row["pnl"]),
                    "pnl_diff": _safe_float(oms_row["pnl"]) - _safe_float(s4_row["pnl"]),
                    "reason_s4": str(s4_row["reason"]),
                    "reason_oms": str(oms_row["reason"]),
                    "opt_dir_s4": str(s4_row["opt_dir"]),
                    "opt_dir_oms": str(oms_row["opt_dir"]),
                }
            )
        else:
            rows.append(
                {
                    "match_status": "missing_in_oms",
                    "symbol": s4_row["symbol"],
                    "qty": _safe_float(s4_row["qty"]),
                    "entry_ts_s4": _safe_float(s4_row["entry_ts"]),
                    "exit_ts_s4": _safe_float(s4_row["exit_ts"]),
                    "entry_ts_oms": pd.NA,
                    "exit_ts_oms": pd.NA,
                    "entry_time_diff_sec": pd.NA,
                    "exit_time_diff_sec": pd.NA,
                    "entry_price_s4": _safe_float(s4_row["entry_price"]),
                    "exit_price_s4": _safe_float(s4_row["exit_price"]),
                    "entry_price_oms": pd.NA,
                    "exit_price_oms": pd.NA,
                    "pnl_s4": _safe_float(s4_row["pnl"]),
                    "pnl_oms": pd.NA,
                    "pnl_diff": pd.NA,
                    "reason_s4": str(s4_row["reason"]),
                    "reason_oms": "",
                    "opt_dir_s4": str(s4_row["opt_dir"]),
                    "opt_dir_oms": "",
                }
            )

    for oms_idx in sorted(oms_unused):
        oms_row = oms_df.loc[oms_idx]
        rows.append(
            {
                "match_status": "extra_in_oms",
                "symbol": oms_row["symbol"],
                "qty": _safe_float(oms_row["qty"]),
                "entry_ts_s4": pd.NA,
                "exit_ts_s4": pd.NA,
                "entry_ts_oms": _safe_float(oms_row["entry_ts"]),
                "exit_ts_oms": _safe_float(oms_row["exit_ts"]),
                "entry_time_diff_sec": pd.NA,
                "exit_time_diff_sec": pd.NA,
                "entry_price_s4": pd.NA,
                "exit_price_s4": pd.NA,
                "entry_price_oms": _safe_float(oms_row["entry_price"]),
                "exit_price_oms": _safe_float(oms_row["exit_price"]),
                "pnl_s4": pd.NA,
                "pnl_oms": _safe_float(oms_row["pnl"]),
                "pnl_diff": pd.NA,
                "reason_s4": "",
                "reason_oms": str(oms_row["reason"]),
                "opt_dir_s4": "",
                "opt_dir_oms": str(oms_row["opt_dir"]),
            }
        )

    out = pd.DataFrame(rows)
    if not out.empty:
        sort_ts = out["exit_ts_s4"].fillna(out["exit_ts_oms"]).fillna(0)
        out = out.assign(_sort_ts=sort_ts).sort_values(["match_status", "symbol", "_sort_ts"]).drop(columns="_sort_ts")
    return out


def _print_summary(title: str, stats: dict) -> None:
    print(title)
    print(f"  Trades   : {stats['trades']}")
    print(f"  Qty      : {stats['qty']:.2f}")
    print(f"  TotalPnL : ${stats['total_pnl']:+,.2f}")
    print(f"  AvgPnL   : ${stats['avg_pnl']:+,.2f}")
    print(f"  WinRate  : {stats['win_rate']:.2%}")


def main() -> int:
    args = _parse_args()
    cfg = CompareConfig(date_str=args.date, time_tolerance_sec=float(args.time_tolerance_sec))

    s4_df = load_s4_csv(args.s4_csv, args.date)
    oms_df = load_oms_from_pg(args.date) if args.oms_source == "pg" else load_oms_from_csv(args.oms_csv, args.date)

    s4_stats = _summary(s4_df)
    oms_stats = _summary(oms_df)
    paired = pair_trades(s4_df, oms_df, cfg)

    out_path = Path(args.out_csv) if args.out_csv else (Path("logs") / f"compare_s4_vs_oms_{args.date}.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    paired.to_csv(out_path, index=False)

    sym_s4 = _per_symbol_summary(s4_df, "s4")
    sym_oms = _per_symbol_summary(oms_df, "oms")
    sym_cmp = sym_s4.merge(sym_oms, on="symbol", how="outer").fillna(0.0)
    sym_cmp["trade_diff"] = sym_cmp["oms_trades"] - sym_cmp["s4_trades"]
    sym_cmp["pnl_diff"] = sym_cmp["oms_pnl"] - sym_cmp["s4_pnl"]

    matched = paired[paired["match_status"] == "matched"].copy()
    missing_count = int((paired["match_status"] == "missing_in_oms").sum())
    extra_count = int((paired["match_status"] == "extra_in_oms").sum())
    pnl_diff_matched = _safe_float(matched["pnl_diff"].sum()) if not matched.empty else 0.0

    print("\n" + "=" * 80)
    print(f"Trade Compare | {args.date}")
    print("=" * 80)
    _print_summary("S4 MockIBKR", s4_stats)
    _print_summary(f"OMS ({args.oms_source.upper()})", oms_stats)
    print("-" * 80)
    print(f"Matched trades      : {len(matched)}")
    print(f"Missing in OMS      : {missing_count}")
    print(f"Extra in OMS        : {extra_count}")
    print(f"Matched pnl diff    : ${pnl_diff_matched:+,.2f}")
    print(f"Total pnl diff      : ${(oms_stats['total_pnl'] - s4_stats['total_pnl']):+,.2f}")
    print(f"Detail CSV          : {out_path}")

    if not sym_cmp.empty:
        sym_cmp = sym_cmp.sort_values(["pnl_diff", "trade_diff", "symbol"], ascending=[False, False, True])
        print("\nPer-symbol summary diff:")
        print(sym_cmp.to_string(index=False, justify="left"))

    mismatches = paired[paired["match_status"] != "matched"]
    if not mismatches.empty:
        print("\nNon-matched trades preview:")
        preview_cols = [
            "match_status", "symbol", "qty", "entry_ts_s4", "exit_ts_s4",
            "entry_ts_oms", "exit_ts_oms", "pnl_s4", "pnl_oms",
            "reason_s4", "reason_oms",
        ]
        print(mismatches[preview_cols].head(30).to_string(index=False, justify="left"))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

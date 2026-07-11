#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a dynamic option-universe target map.

Unlike step1_build_target_map.py, this script does not lock a single ATM/OTM
execution contract. It locks a small same-DTE contract ladder so later stages can
switch to the contract with the best value as spot moves intraday.

Default universe:
  - target trading DTE = 2
  - PUT abs-delta ladder 0.25 / 0.40 / 0.55 / 0.70 => bucket 0..3
  - CALL delta ladder    0.25 / 0.40 / 0.55 / 0.70 => bucket 4..7

Output schema is compatible with step2_polygon_second_sniper_v1.py:
  date_str, contract_symbol, bucket_id, symbol

Extra metadata columns are preserved for diagnostics and dynamic selection.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import logging
import multiprocessing
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from dte_utils import compute_dte_series


NY = "America/New_York"
DEFAULT_RAW_DIR = Path.home() / "train_data/nq_options_day_iv"
DEFAULT_OUTPUT = Path.home() / "train_data/locked_targets_map_2dte_ladder.parquet"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("step1_build_new_map")


def _parse_float_list(raw: str) -> list[float]:
    vals = [float(x.strip()) for x in raw.split(",") if x.strip()]
    if not vals:
        raise ValueError("expected at least one float")
    return vals


def _parse_int_list(raw: str) -> list[int]:
    vals = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if not vals:
        raise ValueError("expected at least one int")
    return vals


def _to_ny_quote_time(series: pd.Series) -> pd.Series:
    """Quote timestamp: naive raw day-IV timestamps are treated as UTC."""
    ts = pd.to_datetime(series, errors="coerce")
    if ts.dt.tz is None:
        return ts.dt.tz_localize("UTC").dt.tz_convert(NY)
    return ts.dt.tz_convert(NY)


def _to_ny_expiration(series: pd.Series) -> pd.Series:
    """Expiration date is a calendar date, not UTC midnight."""
    exp = pd.to_datetime(series, errors="coerce")
    if exp.dt.tz is None:
        return exp.dt.tz_localize(NY, ambiguous="infer")
    return exp.dt.tz_convert(NY)


def _normalize_contract_type(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.upper()
    return np.where(s.str.startswith("C"), "C", np.where(s.str.startswith("P"), "P", ""))


def _contract_exists(df: pd.DataFrame, contract_symbol: str) -> bool:
    return str(contract_symbol).replace("O:", "") in {
        str(x).replace("O:", "") for x in df["contract_symbol"].dropna().unique()
    }


def _prepare_day_df(file_path: Path, use_trading_dte: bool) -> pd.DataFrame:
    df = pd.read_parquet(file_path)
    if df.empty:
        return df

    rename_map = {
        "expiration_date": "expiration",
        "strike_price": "strike",
        "ticker": "contract_symbol",
        "c": "close",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    required = {"timestamp", "contract_symbol", "expiration", "contract_type", "strike", "close", "delta"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{file_path.name} missing columns: {sorted(missing)}")

    df["timestamp"] = _to_ny_quote_time(df["timestamp"])
    df["expiration"] = _to_ny_expiration(df["expiration"])
    df["contract_type_norm"] = _normalize_contract_type(df["contract_type"])
    df["date_str"] = df["timestamp"].dt.date.astype(str)
    df["dte"] = compute_dte_series(
        df["timestamp"],
        df["expiration"],
        use_trading_dte=use_trading_dte,
    )
    df["abs_delta"] = pd.to_numeric(df["delta"], errors="coerce").abs()
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["strike"] = pd.to_numeric(df["strike"], errors="coerce")
    if "stock_close" in df.columns:
        df["stock_close"] = pd.to_numeric(df["stock_close"], errors="coerce")
    else:
        df["stock_close"] = np.nan
    if "volume" in df.columns:
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0)
    else:
        df["volume"] = 0.0
    return df.dropna(subset=["timestamp", "contract_symbol", "expiration", "strike", "close", "abs_delta"])


def _lock_snapshot(day_df: pd.DataFrame, lock_window_minutes: int) -> pd.DataFrame:
    day_start = day_df["timestamp"].min()
    open_cut = day_start + pd.Timedelta(minutes=lock_window_minutes)
    open_df = day_df[day_df["timestamp"] <= open_cut].copy()
    if open_df.empty:
        open_df = day_df.copy()
    # One row per contract, using the latest open-window snapshot.
    return open_df.sort_values("timestamp").groupby("contract_symbol", as_index=False).last()


def _select_one(
    snapshot: pd.DataFrame,
    *,
    target_dte: int,
    side: str,
    target_delta: float,
    min_volume: float,
    min_premium: float,
    max_premium_pct: float,
    exclude_contracts: set[str] | None = None,
) -> pd.Series | None:
    sub = snapshot[
        (snapshot["dte"].astype(int) == int(target_dte))
        & (snapshot["contract_type_norm"] == side)
        & (snapshot["volume"] >= float(min_volume))
        & (snapshot["close"] >= float(min_premium))
    ].copy()
    if sub.empty:
        return None
    if exclude_contracts:
        normalized_exclude = {str(x).replace("O:", "") for x in exclude_contracts}
        sub = sub[
            ~sub["contract_symbol"].astype(str).str.replace("O:", "", regex=False).isin(normalized_exclude)
        ].copy()
        if sub.empty:
            return None

    premium_pct = sub["close"] / sub["stock_close"].replace(0, np.nan)
    sub = sub[premium_pct.fillna(np.inf) <= float(max_premium_pct)].copy()
    if sub.empty:
        return None

    sub["delta_dist"] = (sub["abs_delta"] - float(target_delta)).abs()
    sub["moneyness_abs"] = np.log(sub["strike"] / sub["stock_close"].replace(0, np.nan)).abs()
    sub["premium_pct"] = premium_pct.loc[sub.index]
    sub = sub.sort_values(["delta_dist", "moneyness_abs", "premium_pct"])
    return sub.iloc[0]


def build_ladder_for_file(
    file_path: Path,
    symbol: str,
    *,
    target_dtes: list[int],
    deltas: list[float],
    lock_window_minutes: int,
    min_volume: float,
    min_premium: float,
    max_premium_pct: float,
    require_complete: bool,
    use_trading_dte: bool,
) -> pd.DataFrame | None:
    df = _prepare_day_df(file_path, use_trading_dte=use_trading_dte)
    if df.empty:
        return None

    out_rows: list[dict] = []
    expected_bucket_count = len(deltas) * 2
    for date_str, day_df in df.groupby("date_str"):
        snapshot = _lock_snapshot(day_df, lock_window_minutes=lock_window_minutes)
        day_rows: list[dict] = []

        # Prefer the requested target DTE order. Most runs use a single value, e.g. 2DTE.
        selected_dte = None
        for dte in target_dtes:
            temp_rows: list[dict] = []
            used_contracts: set[str] = set()
            for idx, target_delta in enumerate(deltas):
                put = _select_one(
                    snapshot,
                    target_dte=dte,
                    side="P",
                    target_delta=target_delta,
                    min_volume=min_volume,
                    min_premium=min_premium,
                    max_premium_pct=max_premium_pct,
                    exclude_contracts=used_contracts,
                )
                if put is not None:
                    temp_rows.append(_row_from_selection(put, symbol, date_str, idx, "PUT", target_delta, dte))
                    used_contracts.add(str(put["contract_symbol"]))

            call_offset = len(deltas)
            for idx, target_delta in enumerate(deltas):
                call = _select_one(
                    snapshot,
                    target_dte=dte,
                    side="C",
                    target_delta=target_delta,
                    min_volume=min_volume,
                    min_premium=min_premium,
                    max_premium_pct=max_premium_pct,
                    exclude_contracts=used_contracts,
                )
                if call is not None:
                    temp_rows.append(
                        _row_from_selection(call, symbol, date_str, call_offset + idx, "CALL", target_delta, dte)
                    )
                    used_contracts.add(str(call["contract_symbol"]))

            if not require_complete or len(temp_rows) == expected_bucket_count:
                selected_dte = dte
                day_rows = temp_rows
                break

        if not day_rows:
            continue
        if require_complete and len(day_rows) != expected_bucket_count:
            continue
        # Last guard: never output contracts that are not in the source day file.
        day_rows = [r for r in day_rows if _contract_exists(day_df, r["contract_symbol"])]
        if require_complete and len(day_rows) != expected_bucket_count:
            continue
        for r in day_rows:
            r["selected_dte"] = int(selected_dte) if selected_dte is not None else int(r["target_dte"])
        out_rows.extend(day_rows)

    if not out_rows:
        return None
    return pd.DataFrame(out_rows)


def _row_from_selection(
    row: pd.Series,
    symbol: str,
    date_str: str,
    bucket_id: int,
    side: str,
    target_delta: float,
    target_dte: int,
) -> dict:
    stock_close = float(row.get("stock_close", np.nan))
    strike = float(row.get("strike", np.nan))
    close = float(row.get("close", np.nan))
    premium_pct = close / stock_close if stock_close > 0 else np.nan
    moneyness = np.log(strike / stock_close) if stock_close > 0 and strike > 0 else np.nan
    tag = f"{side}_D{int(round(target_delta * 100)):02d}"
    return {
        "date_str": date_str,
        "contract_symbol": str(row["contract_symbol"]),
        "bucket_id": int(bucket_id),
        "symbol": symbol,
        "tag": tag,
        "side": side,
        "target_abs_delta": float(target_delta),
        "target_dte": int(target_dte),
        "expiration": pd.Timestamp(row["expiration"]).strftime("%Y-%m-%d"),
        "strike": strike,
        "stock_close_at_lock": stock_close,
        "premium_at_lock": close,
        "premium_pct_at_lock": premium_pct,
        "delta_at_lock": float(row.get("delta", np.nan)),
        "abs_delta_at_lock": float(row.get("abs_delta", np.nan)),
        "moneyness_at_lock": moneyness,
        "volume_at_lock": float(row.get("volume", 0.0)),
        "lock_timestamp": pd.Timestamp(row["timestamp"]).isoformat(),
    }


def process_one(args) -> tuple[pd.DataFrame | None, str | None]:
    file_path, symbol, kwargs = args
    try:
        return build_ladder_for_file(file_path, symbol, **kwargs), None
    except Exception as exc:
        return None, f"{file_path.name}: {exc}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build dynamic DTE ladder option-universe map")
    parser.add_argument("--raw-dir", default=str(DEFAULT_RAW_DIR), help="day-IV root")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="output parquet")
    parser.add_argument("--symbols", default="QQQ", help="comma-separated symbols")
    parser.add_argument("--start-date", default=None, help="YYYY-MM-DD")
    parser.add_argument("--end-date", default=None, help="YYYY-MM-DD")
    parser.add_argument("--target-dtes", default="2", help="preferred DTE list, e.g. 2 or 2,3")
    parser.add_argument("--deltas", default="0.25,0.40,0.55,0.70", help="abs-delta ladder")
    parser.add_argument("--lock-window-minutes", type=int, default=10)
    parser.add_argument("--min-volume", type=float, default=10.0)
    parser.add_argument("--min-premium", type=float, default=0.05)
    parser.add_argument("--max-premium-pct", type=float, default=0.08)
    parser.add_argument("--calendar-dte", action="store_true", help="use calendar DTE instead of trading DTE")
    parser.add_argument("--allow-partial", action="store_true", help="allow days with missing ladder buckets")
    parser.add_argument("--workers", type=int, default=max(1, multiprocessing.cpu_count() - 2))
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir).expanduser()
    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    target_dtes = _parse_int_list(args.target_dtes)
    deltas = _parse_float_list(args.deltas)

    kwargs = {
        "target_dtes": target_dtes,
        "deltas": deltas,
        "lock_window_minutes": args.lock_window_minutes,
        "min_volume": args.min_volume,
        "min_premium": args.min_premium,
        "max_premium_pct": args.max_premium_pct,
        "require_complete": not args.allow_partial,
        "use_trading_dte": not args.calendar_dte,
    }

    tasks = []
    for symbol in symbols:
        src_dir = raw_dir / symbol
        if not src_dir.exists():
            logger.warning("skip %s: missing %s", symbol, src_dir)
            continue
        for p in sorted(src_dir.glob(f"{symbol}_*.parquet")):
            if "high_features" in p.name:
                continue
            date_str = p.stem.split("_")[-1]
            if args.start_date and date_str < args.start_date:
                continue
            if args.end_date and date_str > args.end_date:
                continue
            tasks.append((p, symbol, kwargs))

    if not tasks:
        raise SystemExit("no input day-IV parquet files found")

    logger.info(
        "build dynamic map | symbols=%s | target_dtes=%s | deltas=%s | complete=%s | tasks=%d",
        symbols,
        target_dtes,
        deltas,
        not args.allow_partial,
        len(tasks),
    )

    parts: list[pd.DataFrame] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=max(1, args.workers)) as ex:
        for df, err in tqdm(ex.map(process_one, tasks), total=len(tasks), desc="dynamic-map"):
            if err:
                tqdm.write(f"WARN {err}")
            if df is not None and not df.empty:
                parts.append(df)

    if not parts:
        raise SystemExit("no contracts selected")

    final = pd.concat(parts, ignore_index=True)
    final = final.sort_values(["symbol", "date_str", "bucket_id"]).reset_index(drop=True)
    final.to_parquet(output, index=False, compression="zstd")

    pattern = final.groupby(["symbol", "date_str"])["bucket_id"].apply(lambda s: tuple(sorted(s.astype(int))))
    logger.info("done | rows=%d days=%d -> %s", len(final), final["date_str"].nunique(), output)
    logger.info("bucket patterns:\n%s", pattern.value_counts().to_string())
    logger.info("target DTE dist:\n%s", final.groupby("date_str")["selected_dte"].first().value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()

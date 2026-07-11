#!/usr/bin/env python3
"""Build QQQ 0DTE strike-near-spot ladder map.

Fixed-delta selection is brittle for 0DTE because delta changes violently around
the open.  This map instead locks the nearest tradable strikes around spot:
PUT buckets first, then CALL buckets.  The output schema stays compatible with
the microstructure downloader and existing locked-map utilities.
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

from step1_build_new_map import _prepare_day_df

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("build_0dte_strike_ladder_map")


def _latest_open_snapshot(day_df: pd.DataFrame, lock_window_minutes: int) -> pd.DataFrame:
    start = day_df["timestamp"].min()
    cutoff = start + pd.Timedelta(minutes=lock_window_minutes)
    open_df = day_df[day_df["timestamp"] <= cutoff].copy()
    if open_df.empty:
        open_df = day_df.copy()
    return open_df.sort_values("timestamp").groupby("contract_symbol", as_index=False).last()


def _side_select(
    snapshot: pd.DataFrame,
    *,
    side: str,
    n_per_side: int,
    min_volume: float,
    min_premium: float,
    max_premium_pct: float,
) -> pd.DataFrame:
    sub = snapshot[
        (snapshot["dte"].astype(int) == 0)
        & (snapshot["contract_type_norm"] == side)
        & (snapshot["volume"] >= min_volume)
        & (snapshot["close"] >= min_premium)
    ].copy()
    if sub.empty:
        return sub
    sub["premium_pct"] = sub["close"] / sub["stock_close"].replace(0, np.nan)
    sub = sub[sub["premium_pct"].fillna(np.inf) <= max_premium_pct].copy()
    if sub.empty:
        return sub
    sub["moneyness"] = np.log(sub["strike"] / sub["stock_close"].replace(0, np.nan))
    sub["abs_moneyness"] = sub["moneyness"].abs()
    # Bias PUT slightly below spot and CALL slightly above spot, but do not drop
    # nearby ITM contracts: 0DTE liquidity often straddles the money.
    if side == "P":
        sub["side_rank"] = np.where(sub["strike"] <= sub["stock_close"], 0, 1)
    else:
        sub["side_rank"] = np.where(sub["strike"] >= sub["stock_close"], 0, 1)
    sub = sub.sort_values(["side_rank", "abs_moneyness", "premium_pct", "volume"], ascending=[True, True, True, False])
    return sub.head(n_per_side)


def _row(row: pd.Series, symbol: str, date_str: str, bucket_id: int, side: str, rank: int) -> dict:
    stock_close = float(row.get("stock_close", np.nan))
    strike = float(row.get("strike", np.nan))
    close = float(row.get("close", np.nan))
    premium_pct = close / stock_close if stock_close > 0 else np.nan
    moneyness = np.log(strike / stock_close) if stock_close > 0 and strike > 0 else np.nan
    return {
        "date_str": date_str,
        "contract_symbol": str(row["contract_symbol"]),
        "bucket_id": int(bucket_id),
        "symbol": symbol,
        "tag": f"{side}_K{rank:02d}",
        "side": side,
        "target_abs_delta": float(row.get("abs_delta", np.nan)),
        "target_dte": 0,
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
        "selected_dte": 0,
    }


def build_one_file(
    file_path: Path,
    *,
    symbol: str,
    n_per_side: int,
    lock_window_minutes: int,
    min_volume: float,
    min_premium: float,
    max_premium_pct: float,
    require_complete: bool,
) -> tuple[pd.DataFrame | None, str | None]:
    try:
        df = _prepare_day_df(file_path, use_trading_dte=True)
        if df.empty:
            return None, None
        rows: list[dict] = []
        for date_str, day_df in df.groupby("date_str"):
            snap = _latest_open_snapshot(day_df, lock_window_minutes)
            puts = _side_select(
                snap,
                side="P",
                n_per_side=n_per_side,
                min_volume=min_volume,
                min_premium=min_premium,
                max_premium_pct=max_premium_pct,
            )
            calls = _side_select(
                snap,
                side="C",
                n_per_side=n_per_side,
                min_volume=min_volume,
                min_premium=min_premium,
                max_premium_pct=max_premium_pct,
            )
            if require_complete and (len(puts) < n_per_side or len(calls) < n_per_side):
                continue
            for i, (_, r) in enumerate(puts.iterrows()):
                rows.append(_row(r, symbol, date_str, i, "PUT", i))
            for i, (_, r) in enumerate(calls.iterrows()):
                rows.append(_row(r, symbol, date_str, n_per_side + i, "CALL", i))
        return (pd.DataFrame(rows) if rows else None), None
    except Exception as exc:
        return None, f"{file_path.name}: {exc}"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build 0DTE strike-near-spot ladder map")
    p.add_argument("--raw-dir", default=str(Path.home() / "train_data/nq_options_day_iv"))
    p.add_argument("--output", default=str(Path.home() / "train_data/locked_targets_map_0dte_strike_ladder.parquet"))
    p.add_argument("--symbol", default="QQQ")
    p.add_argument("--start-date", default="2026-01-01")
    p.add_argument("--end-date", default="2026-06-30")
    p.add_argument("--n-per-side", type=int, default=6)
    p.add_argument("--lock-window-minutes", type=int, default=10)
    p.add_argument("--min-volume", type=float, default=1.0)
    p.add_argument("--min-premium", type=float, default=0.01)
    p.add_argument("--max-premium-pct", type=float, default=0.20)
    p.add_argument("--allow-partial", action="store_true")
    p.add_argument("--workers", type=int, default=max(1, multiprocessing.cpu_count() - 2))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    src = Path(args.raw_dir).expanduser() / args.symbol
    files = []
    for f in sorted(src.glob(f"{args.symbol}_*.parquet")):
        if "high_features" in f.name:
            continue
        date_str = f.stem.split("_")[-1]
        if args.start_date <= date_str <= args.end_date:
            files.append(f)
    if not files:
        raise SystemExit(f"no files under {src}")
    logger.info("build 0DTE strike ladder | files=%d n_per_side=%d complete=%s", len(files), args.n_per_side, not args.allow_partial)
    parts = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futs = [
            ex.submit(
                build_one_file,
                f,
                symbol=args.symbol,
                n_per_side=args.n_per_side,
                lock_window_minutes=args.lock_window_minutes,
                min_volume=args.min_volume,
                min_premium=args.min_premium,
                max_premium_pct=args.max_premium_pct,
                require_complete=not args.allow_partial,
            )
            for f in files
        ]
        for fut in tqdm(concurrent.futures.as_completed(futs), total=len(futs), desc="0dte-strike-map"):
            df, err = fut.result()
            if err:
                tqdm.write(f"WARN {err}")
            if df is not None and not df.empty:
                parts.append(df)
    if not parts:
        raise SystemExit("no contracts selected")
    out = pd.concat(parts, ignore_index=True).sort_values(["symbol", "date_str", "bucket_id"]).reset_index(drop=True)
    out_path = Path(args.output).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False, compression="zstd")
    pattern = out.groupby("date_str")["bucket_id"].apply(lambda s: tuple(sorted(s.astype(int))))
    logger.info("done rows=%d days=%d -> %s", len(out), out["date_str"].nunique(), out_path)
    logger.info("bucket patterns:\n%s", pattern.value_counts().head(20).to_string())


if __name__ == "__main__":
    main()

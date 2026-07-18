#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Polygon / Massive 正股秒级 (1s) OHLCV 下载器。

从 step2_polygon_second_sniper_v1 提炼：只负责 underlying 1s bars，不碰期权 quote。
输出布局与 sniper 一致：
  {stock_output_dir}/{SYMBOL}/{SYMBOL}_YYYY-MM-DD.parquet
列：ts, open, high, low, close, volume, timestamp(ET)

用法:
  export MASSIVE_API_KEY=...   # 或 POLYGON_API_KEY

  # 按标的 + 日期区间
  python preprocess/download/download_stock_1s.py \\
      --symbols NVDA,TSLA,GOOGL --start-date 2026-05-01 --end-date 2026-07-13

  # 按锁约表里的 (symbol, date_str) 预检补齐（step2 同款）
  python preprocess/download/download_stock_1s.py \\
      --target-map ~/train_data/locked_targets_map_....parquet \\
      --start-date 2026-05-01 --end-date 2026-07-13

  # 强制重下
  python preprocess/download/download_stock_1s.py --symbols QQQ --start-date 2026-07-01 --end-date 2026-07-01 --force
"""
from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import logging
import os
from typing import Iterable

import pandas as pd
from polygon import RESTClient
from pytz import timezone
from tqdm import tqdm

DEFAULT_STOCK_OUTPUT_DIR = "/mnt/s990/data/raw_1s/stocks"
MIN_STOCK_1S_ROWS = 1000

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logging.getLogger("urllib3").setLevel(logging.ERROR)
logger = logging.getLogger("download_stock_1s")
eastern = timezone("America/New_York")


def resolve_api_key() -> str:
    for k in ("MASSIVE_API_KEY", "POLYGON_API_KEY", "POLYGON_KEY"):
        v = os.environ.get(k, "").strip()
        if v:
            return v
    raise SystemExit("缺少 API key：请设置环境变量 MASSIVE_API_KEY 或 POLYGON_API_KEY")


def stock_1s_path(symbol: str, date_str: str, *, stock_output_dir: str = DEFAULT_STOCK_OUTPUT_DIR) -> str:
    return os.path.join(stock_output_dir, symbol.upper(), f"{symbol.upper()}_{date_str}.parquet")


def stock_1s_file_ok(path: str, *, min_rows: int = MIN_STOCK_1S_ROWS) -> bool:
    """True only for a real 1s parquet with enough rows (reject empty / thin stubs)."""
    if not path or not os.path.isfile(path):
        return False
    try:
        n = len(pd.read_parquet(path, columns=["ts"]))
    except Exception:
        try:
            n = len(pd.read_parquet(path))
        except Exception:
            return False
    return n >= int(min_rows)


def download_stock_1s_day(
    client: RESTClient,
    symbol: str,
    date_str: str,
    *,
    stock_output_dir: str = DEFAULT_STOCK_OUTPUT_DIR,
    force: bool = False,
    min_rows: int = MIN_STOCK_1S_ROWS,
) -> bool:
    """Download/cache Polygon second aggs for one symbol-day. Returns True if file is OK."""
    sym = str(symbol).upper()
    stock_path = stock_1s_path(sym, date_str, stock_output_dir=stock_output_dir)
    if (not force) and stock_1s_file_ok(stock_path, min_rows=min_rows):
        return True
    os.makedirs(os.path.dirname(stock_path), exist_ok=True)
    try:
        aggs = list(
            client.list_aggs(
                ticker=sym,
                multiplier=1,
                timespan="second",
                from_=date_str,
                to=date_str,
                limit=50000,
            )
        )
        if not aggs:
            logger.warning("Polygon 1s stock empty for %s %s", sym, date_str)
            return False
        stk_df = pd.DataFrame(
            [
                {
                    "ts": a.timestamp / 1000.0,
                    "open": a.open,
                    "high": a.high,
                    "low": a.low,
                    "close": a.close,
                    "volume": a.volume,
                }
                for a in aggs
            ]
        )
        stk_df["timestamp"] = pd.to_datetime(stk_df["ts"], unit="s", utc=True).dt.tz_convert(eastern)
        stk_df.to_parquet(stock_path, index=False)
        ok = stock_1s_file_ok(stock_path, min_rows=min_rows)
        if not ok:
            logger.warning(
                "Polygon 1s stock thin for %s %s (rows=%d < %d)",
                sym,
                date_str,
                len(stk_df),
                min_rows,
            )
        return ok
    except Exception as exc:
        logger.warning("Polygon 1s stock download failed for %s %s: %s", sym, date_str, exc)
        return False


def ensure_stock_1s_pairs(
    pairs: Iterable[tuple[str, str]],
    *,
    stock_output_dir: str = DEFAULT_STOCK_OUTPUT_DIR,
    max_workers: int = 12,
    api_key: str | None = None,
    force: bool = False,
    min_rows: int = MIN_STOCK_1S_ROWS,
) -> dict[str, int]:
    """Download every missing/thin (symbol, date) pair."""
    uniq: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for sym, d in pairs:
        key = (str(sym).upper(), str(d))
        if key in seen:
            continue
        seen.add(key)
        uniq.append(key)

    missing: list[tuple[str, str]] = []
    for sym, d in uniq:
        path = stock_1s_path(sym, d, stock_output_dir=stock_output_dir)
        if force or not stock_1s_file_ok(path, min_rows=min_rows):
            missing.append((sym, d))

    stats = {"pairs": int(len(uniq)), "missing": int(len(missing)), "ok": 0, "fail": 0}
    if not missing:
        logger.info("Stock 1s: all %d symbol-days present under %s", stats["pairs"], stock_output_dir)
        return stats

    key = api_key or resolve_api_key()
    logger.info(
        "Stock 1s: %d/%d to download (workers=%d force=%s) out=%s",
        len(missing),
        len(uniq),
        max_workers,
        force,
        stock_output_dir,
    )

    def _one(item: tuple[str, str]) -> bool:
        sym, d = item
        return download_stock_1s_day(
            RESTClient(key),
            sym,
            d,
            stock_output_dir=stock_output_dir,
            force=force,
            min_rows=min_rows,
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, int(max_workers))) as ex:
        for ok in tqdm(ex.map(_one, missing), total=len(missing), desc="Stock 1s"):
            if ok:
                stats["ok"] += 1
            else:
                stats["fail"] += 1
    logger.info("Stock 1s done: ok=%d fail=%d", stats["ok"], stats["fail"])
    return stats


def ensure_stock_1s_for_map(
    target_map: pd.DataFrame,
    *,
    stock_output_dir: str = DEFAULT_STOCK_OUTPUT_DIR,
    max_workers: int = 12,
    api_key: str | None = None,
    force: bool = False,
    min_rows: int = MIN_STOCK_1S_ROWS,
) -> dict[str, int]:
    """Download every missing/thin 1s underlying referenced by a lock/target map."""
    if "symbol" not in target_map.columns or "date_str" not in target_map.columns:
        raise ValueError("target map needs symbol + date_str columns")
    pairs = (
        target_map[["symbol", "date_str"]]
        .dropna()
        .astype({"symbol": str, "date_str": str})
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    return ensure_stock_1s_pairs(
        pairs,
        stock_output_dir=stock_output_dir,
        max_workers=max_workers,
        api_key=api_key,
        force=force,
        min_rows=min_rows,
    )


def load_stock_price_map(
    client: RESTClient,
    symbol: str,
    date_str: str,
    *,
    stock_output_dir: str = DEFAULT_STOCK_OUTPUT_DIR,
    force: bool = False,
    min_rows: int = MIN_STOCK_1S_ROWS,
) -> dict[int, float]:
    """Load 1s close keyed by unix-second int (download if missing)."""
    if not stock_1s_file_ok(
        stock_1s_path(symbol, date_str, stock_output_dir=stock_output_dir),
        min_rows=min_rows,
    ):
        download_stock_1s_day(
            client,
            symbol,
            date_str,
            stock_output_dir=stock_output_dir,
            force=force,
            min_rows=min_rows,
        )
    path = stock_1s_path(symbol, date_str, stock_output_dir=stock_output_dir)
    if not os.path.isfile(path):
        return {}
    try:
        stk_df = pd.read_parquet(path)
    except Exception:
        return {}
    if stk_df is None or stk_df.empty or "ts" not in stk_df.columns:
        return {}
    stk_df = stk_df.copy()
    stk_df["ts_int"] = stk_df["ts"].round(0).astype("int64")
    return dict(zip(stk_df["ts_int"], stk_df["close"]))


def _daterange(start: str, end: str) -> list[str]:
    s = dt.date.fromisoformat(start)
    e = dt.date.fromisoformat(end)
    out: list[str] = []
    cur = s
    while cur <= e:
        # skip weekends; holidays still hit API (empty → fail, harmless)
        if cur.weekday() < 5:
            out.append(cur.isoformat())
        cur += dt.timedelta(days=1)
    return out


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Download Polygon/Massive 1s stock bars")
    ap.add_argument("--symbols", default=None, help="comma-separated, e.g. NVDA,TSLA,GOOGL")
    ap.add_argument("--start-date", default=None, help="YYYY-MM-DD inclusive")
    ap.add_argument("--end-date", default=None, help="YYYY-MM-DD inclusive")
    ap.add_argument(
        "--target-map",
        default=None,
        help="parquet with symbol + date_str (lock/miss map); alternative to --symbols",
    )
    ap.add_argument("--stock-output-dir", default=DEFAULT_STOCK_OUTPUT_DIR)
    ap.add_argument("--max-workers", type=int, default=12)
    ap.add_argument("--min-rows", type=int, default=MIN_STOCK_1S_ROWS, help="row threshold for 'ok'")
    ap.add_argument("--force", action="store_true", help="re-download even if file looks ok")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    api_key = resolve_api_key()
    out_dir = os.path.expanduser(args.stock_output_dir)

    if args.target_map:
        path = os.path.expanduser(args.target_map)
        if not os.path.isfile(path):
            raise SystemExit(f"target map not found: {path}")
        mp = pd.read_parquet(path)
        if "date_str" not in mp.columns:
            raise SystemExit(f"map missing date_str: {list(mp.columns)}")
        mp["date_str"] = mp["date_str"].astype(str)
        if args.start_date:
            mp = mp[mp["date_str"] >= args.start_date]
        if args.end_date:
            mp = mp[mp["date_str"] <= args.end_date]
        if args.symbols:
            want = {s.strip().upper() for s in args.symbols.split(",") if s.strip()}
            mp = mp[mp["symbol"].astype(str).str.upper().isin(want)]
        if mp.empty:
            raise SystemExit("no rows left after filters")
        stats = ensure_stock_1s_for_map(
            mp,
            stock_output_dir=out_dir,
            max_workers=args.max_workers,
            api_key=api_key,
            force=bool(args.force),
            min_rows=int(args.min_rows),
        )
    else:
        if not args.symbols or not args.start_date or not args.end_date:
            raise SystemExit("need --symbols + --start-date + --end-date, or --target-map")
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
        days = _daterange(args.start_date, args.end_date)
        pairs = [(sym, d) for sym in symbols for d in days]
        stats = ensure_stock_1s_pairs(
            pairs,
            stock_output_dir=out_dir,
            max_workers=args.max_workers,
            api_key=api_key,
            force=bool(args.force),
            min_rows=int(args.min_rows),
        )

    print(stats, flush=True)


if __name__ == "__main__":
    main()

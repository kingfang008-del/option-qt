#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把本地 1m 股票 parquet 灌进 PG market_bars_1m/5m，供 FCS Deep Warmup 使用。

实盘：盘后/维护窗写入历史；开盘前 FCS 从 PG 自动预热。
对拍：开跑前先 seed 交易日起之前的 bars。

用法:
  python qqq_btc/tools/seed_pg_warmup_bars.py \\
      --symbols QQQ,VIXY \\
      --start 2026-06-01 --end 2026-06-30 \\
      --root ~/train_data/spnq_train
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extras

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("seed_pg_warmup")

DEFAULT_PG = "dbname=quant_trade user=postgres password=postgres host=localhost port=5432"
NY = "America/New_York"


def _ensure_tables(conn) -> None:
    c = conn.cursor()
    # 生产库通常已是按日 RANGE(ts) 分区表;此处仅兜底建非分区表(空库)
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS market_bars_1m (
            symbol TEXT NOT NULL,
            ts DOUBLE PRECISION NOT NULL,
            open DOUBLE PRECISION,
            high DOUBLE PRECISION,
            low DOUBLE PRECISION,
            close DOUBLE PRECISION,
            volume DOUBLE PRECISION,
            vwap DOUBLE PRECISION,
            PRIMARY KEY (symbol, ts)
        );
        """
    )
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS market_bars_5m (
            symbol TEXT NOT NULL,
            ts DOUBLE PRECISION NOT NULL,
            open DOUBLE PRECISION,
            high DOUBLE PRECISION,
            low DOUBLE PRECISION,
            close DOUBLE PRECISION,
            volume DOUBLE PRECISION,
            vwap DOUBLE PRECISION,
            PRIMARY KEY (symbol, ts)
        );
        """
    )
    conn.commit()


def _day_partition_bounds(day: pd.Timestamp) -> tuple[float, float, str]:
    """NY 日历日 [00:00, next 00:00) → unix ts；分区名 YYYYMMDD。"""
    d0 = pd.Timestamp(day.date(), tz=NY)
    d1 = d0 + pd.Timedelta(days=1)
    ymd = d0.strftime("%Y%m%d")
    return float(d0.timestamp()), float(d1.timestamp()), ymd


def _ensure_day_partitions(conn, days: list[pd.Timestamp]) -> None:
    """为 market_bars_1m/5m 创建按日分区(若父表已分区)。"""
    c = conn.cursor()
    for table in ("market_bars_1m", "market_bars_5m"):
        c.execute(
            """
            SELECT c.relkind
            FROM pg_class c
            JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE n.nspname='public' AND c.relname=%s
            """,
            (table,),
        )
        row = c.fetchone()
        if not row or row[0] != "p":
            continue
        for day in days:
            lo, hi, ymd = _day_partition_bounds(day)
            part = f"{table}_{ymd}"
            c.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {part}
                PARTITION OF {table}
                FOR VALUES FROM (%s) TO (%s)
                """,
                (lo, hi),
            )
    conn.commit()


def _load_month(root: Path, symbol: str, month: str) -> pd.DataFrame:
    fp = root / symbol / f"{month}.parquet"
    if not fp.exists():
        # alternate layouts
        alt = root / symbol / f"{symbol}_{month}.parquet"
        fp = alt if alt.exists() else fp
    if not fp.exists():
        return pd.DataFrame()
    df = pd.read_parquet(fp)
    if "timestamp" not in df.columns:
        raise ValueError(f"{fp} missing timestamp")
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(NY)
    for col in ("open", "high", "low", "close", "volume"):
        if col not in df.columns:
            raise ValueError(f"{fp} missing {col}")
    if "vwap" not in df.columns:
        df["vwap"] = df["close"]
    # RTH only
    t = df["timestamp"].dt.time
    import datetime as dt

    m = (t >= dt.time(9, 30)) & (t < dt.time(16, 0))
    return df.loc[m].sort_values("timestamp")


def _to_rows(df: pd.DataFrame, symbol: str) -> list[tuple]:
    rows = []
    for r in df.itertuples(index=False):
        ts = float(pd.Timestamp(r.timestamp).timestamp())
        rows.append(
            (
                symbol,
                ts,
                float(r.open),
                float(r.high),
                float(r.low),
                float(r.close),
                float(r.volume),
                float(getattr(r, "vwap", r.close)),
            )
        )
    return rows


def _aggregate_5m(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    x = df.set_index("timestamp").sort_index()
    ohlc = x.resample("5min", label="left", closed="left").agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
            "vwap": "mean",
        }
    )
    ohlc = ohlc.dropna(subset=["open", "close"]).reset_index()
    return ohlc


def _upsert(conn, table: str, rows: list[tuple]) -> int:
    if not rows:
        return 0
    sql = f"""
        INSERT INTO {table} (symbol, ts, open, high, low, close, volume, vwap)
        VALUES %s
        ON CONFLICT (symbol, ts) DO UPDATE SET
            open=EXCLUDED.open, high=EXCLUDED.high, low=EXCLUDED.low,
            close=EXCLUDED.close, volume=EXCLUDED.volume, vwap=EXCLUDED.vwap
    """
    with conn.cursor() as c:
        psycopg2.extras.execute_values(c, sql, rows, page_size=1000)
    conn.commit()
    return len(rows)


def seed_range(
    *,
    root: Path,
    symbols: list[str],
    start: str,
    end: str,
    pg_url: str,
) -> dict:
    conn = psycopg2.connect(pg_url)
    try:
        _ensure_tables(conn)
        days = list(pd.date_range(start, end, freq="D"))
        _ensure_day_partitions(conn, days)
        months = sorted({f"{d.year:04d}-{d.month:02d}" for d in days})
        start_ts = pd.Timestamp(start, tz=NY)
        end_ts = pd.Timestamp(end, tz=NY) + pd.Timedelta(days=1)
        summary = {}
        for sym in symbols:
            n1 = n5 = 0
            for month in months:
                df = _load_month(root, sym, month)
                if df.empty:
                    logger.warning("missing %s %s under %s", sym, month, root)
                    continue
                df = df[(df["timestamp"] >= start_ts) & (df["timestamp"] < end_ts)]
                rows1 = _to_rows(df, sym)
                n1 += _upsert(conn, "market_bars_1m", rows1)
                rows5 = _to_rows(_aggregate_5m(df), sym)
                n5 += _upsert(conn, "market_bars_5m", rows5)
            summary[sym] = {"bars_1m": n1, "bars_5m": n5}
            logger.info("seeded %s 1m=%d 5m=%d", sym, n1, n5)
        return summary
    finally:
        conn.close()


def main() -> None:
    p = argparse.ArgumentParser(description="Seed PG market_bars for FCS deep warmup")
    p.add_argument("--root", default=str(Path.home() / "train_data/spnq_train"))
    p.add_argument("--symbols", default="QQQ,VIXY")
    p.add_argument("--start", required=True, help="YYYY-MM-DD inclusive")
    p.add_argument("--end", required=True, help="YYYY-MM-DD inclusive")
    p.add_argument("--pg-url", default=DEFAULT_PG)
    args = p.parse_args()
    summary = seed_range(
        root=Path(args.root).expanduser(),
        symbols=[s.strip() for s in args.symbols.split(",") if s.strip()],
        start=args.start,
        end=args.end,
        pg_url=args.pg_url,
    )
    print(summary)


if __name__ == "__main__":
    main()

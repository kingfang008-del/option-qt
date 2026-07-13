#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""确保 FCS Deep Warmup 所需 PG market_bars 完整。

流程:
  1) 检测 QQQ/VIXY 在 lookback 交易日窗口内的 1m/5m 覆盖
  2) 缺口优先从本地 spnq_train parquet seed
  3) 仍缺则用 Massive/Polygon 分钟聚合下载并 UPSERT
  4) 再检测；过关 exit 0，否则 exit 2

用法:
  python qqq_btc/tools/ensure_fcs_warmup_ready.py
  python qqq_btc/tools/ensure_fcs_warmup_ready.py --asof 2026-07-13 --lookback-tdays 12
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, time as dtime
from pathlib import Path

import pandas as pd
import psycopg2

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from qqq_btc.tools.seed_pg_warmup_bars import (  # noqa: E402
    DEFAULT_PG,
    NY,
    _aggregate_5m,
    _ensure_day_partitions,
    _ensure_tables,
    _to_rows,
    _upsert,
    seed_range,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("ensure_warmup")

# Massive = Polygon 新品牌；默认仍走 polygon.io（同 key）
DEFAULT_API_BASE = os.environ.get("MASSIVE_API_BASE", "https://api.polygon.io").rstrip("/")


def _api_key() -> str:
    for k in ("MASSIVE_API_KEY", "POLYGON_API_KEY", "POLYGON_KEY"):
        v = os.environ.get(k, "").strip()
        if v:
            return v
    return "JXuIcG_dpoRiCE6jP7c73nVWweEVSpUp"


def _trading_days(end_exclusive: pd.Timestamp, n_tdays: int) -> list[pd.Timestamp]:
    """取 end_exclusive 之前的 n 个工作日（不含 end 当天）。"""
    days: list[pd.Timestamp] = []
    cur = pd.Timestamp(end_exclusive.date(), tz=NY) - pd.Timedelta(days=1)
    while len(days) < n_tdays:
        if cur.weekday() < 5:
            days.append(cur)
        cur -= pd.Timedelta(days=1)
    return list(reversed(days))


def _day_bounds(day: pd.Timestamp) -> tuple[float, float]:
    d0 = pd.Timestamp(day.date(), tz=NY)
    d1 = d0 + pd.Timedelta(days=1)
    return float(d0.timestamp()), float(d1.timestamp())


def _count_bars(conn, table: str, symbol: str, lo: float, hi: float) -> int:
    with conn.cursor() as c:
        c.execute(
            f"SELECT COUNT(*) FROM {table} WHERE symbol=%s AND ts>=%s AND ts<%s",
            (symbol, lo, hi),
        )
        return int(c.fetchone()[0])


def _inspect(
    conn,
    symbols: list[str],
    days: list[pd.Timestamp],
    *,
    min_day_bars: int,
    min_total_1m: int,
) -> dict:
    lo0, _ = _day_bounds(days[0])
    _, hi1 = _day_bounds(days[-1])
    by_sym = {}
    missing: dict[str, list[str]] = {s: [] for s in symbols}
    for sym in symbols:
        day_counts = {}
        for d in days:
            lo, hi = _day_bounds(d)
            n = _count_bars(conn, "market_bars_1m", sym, lo, hi)
            day_counts[d.strftime("%Y-%m-%d")] = n
            if n < min_day_bars:
                missing[sym].append(d.strftime("%Y-%m-%d"))
        total = _count_bars(conn, "market_bars_1m", sym, lo0, hi1)
        total_5m = _count_bars(conn, "market_bars_5m", sym, lo0, hi1)
        by_sym[sym] = {
            "total_1m": total,
            "total_5m": total_5m,
            "day_counts": day_counts,
            "missing_days": missing[sym],
            "ok_total": total >= min_total_1m,
            "ok_days": len(missing[sym]) == 0,
        }
    ready = all(v["ok_total"] and v["ok_days"] for v in by_sym.values())
    return {
        "ready": ready,
        "window": [days[0].strftime("%Y-%m-%d"), days[-1].strftime("%Y-%m-%d")],
        "symbols": by_sym,
        "missing_union": sorted({d for ds in missing.values() for d in ds}),
    }


def _fetch_json(url: str, *, retries: int = 5) -> dict:
    for attempt in range(retries):
        try:
            time.sleep(0.25 * (attempt + 1))
            with urllib.request.urlopen(url, timeout=90) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            if exc.code in (429, 403, 500, 502, 503) and attempt + 1 < retries:
                wait = 2.0 * (attempt + 1)
                logger.warning("HTTP %s → sleep %.1fs", exc.code, wait)
                time.sleep(wait)
                continue
            raise
        except Exception:
            if attempt + 1 < retries:
                time.sleep(1.5 * (attempt + 1))
                continue
            raise
    return {}


def download_day_aggs(
    symbol: str,
    day: str,
    *,
    api_key: str,
    api_base: str,
) -> pd.DataFrame:
    """Massive/Polygon 1m aggs → RTH DataFrame(timestamp, ohlcv, vwap)."""
    enc = urllib.parse.quote(symbol, safe="")
    url = (
        f"{api_base}/v2/aggs/ticker/{enc}/range/1/minute/{day}/{day}"
        f"?adjusted=true&sort=asc&limit=50000&apiKey={api_key}"
    )
    data = _fetch_json(url)
    status = str(data.get("status", ""))
    results = list(data.get("results") or [])
    if status not in ("OK", "DELAYED") and not results:
        logger.warning("%s %s api status=%s results=0", symbol, day, status)
        return pd.DataFrame()
    rows = []
    for r in results:
        ts_ms = int(r["t"])
        dt = pd.Timestamp(ts_ms, unit="ms", tz="UTC").tz_convert(NY)
        t = dt.time()
        if not (dtime(9, 30) <= t < dtime(16, 0)):
            continue
        rows.append(
            {
                "timestamp": dt,
                "open": float(r["o"]),
                "high": float(r["h"]),
                "low": float(r["l"]),
                "close": float(r["c"]),
                "volume": float(r.get("v") or 0.0),
                "vwap": float(r.get("vw") or r["c"]),
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("timestamp")


def _upsert_day_frames(conn, symbol: str, df: pd.DataFrame) -> tuple[int, int]:
    if df is None or df.empty:
        return 0, 0
    days = [pd.Timestamp(ts).tz_convert(NY) for ts in df["timestamp"]]
    day_keys = sorted({pd.Timestamp(ts.date(), tz=NY) for ts in days})
    _ensure_day_partitions(conn, day_keys)
    n1 = _upsert(conn, "market_bars_1m", _to_rows(df, symbol))
    n5 = _upsert(conn, "market_bars_5m", _to_rows(_aggregate_5m(df), symbol))
    return n1, n5


def main() -> int:
    ap = argparse.ArgumentParser(description="Ensure PG bars ready for FCS Deep Warmup")
    ap.add_argument("--symbols", default="QQQ,VIXY")
    ap.add_argument("--asof", default=None, help="YYYY-MM-DD（默认今天 NY）；窗口不含当天")
    ap.add_argument("--lookback-tdays", type=int, default=12, help="需要的历史交易日数")
    ap.add_argument("--min-day-bars", type=int, default=300, help="完整交易日最少 1m bars")
    ap.add_argument("--min-total-1m", type=int, default=1800, help="窗口内最少总 1m bars/标的")
    ap.add_argument("--stock-root", default=str(Path.home() / "train_data/spnq_train"))
    ap.add_argument("--pg-url", default=os.environ.get("PG_DB_URL", DEFAULT_PG))
    ap.add_argument("--api-base", default=DEFAULT_API_BASE)
    ap.add_argument("--no-download", action="store_true", help="只检测不下载")
    ap.add_argument("--out", default=None, help="写 JSON 报告路径")
    args = ap.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    now_ny = pd.Timestamp.now(tz=NY)
    asof = pd.Timestamp(args.asof, tz=NY) if args.asof else now_ny
    days = _trading_days(asof, args.lookback_tdays)
    api_key = _api_key()
    stock_root = Path(args.stock_root).expanduser()

    conn = psycopg2.connect(args.pg_url)
    try:
        _ensure_tables(conn)
        _ensure_day_partitions(conn, days)

        before = _inspect(
            conn,
            symbols,
            days,
            min_day_bars=args.min_day_bars,
            min_total_1m=args.min_total_1m,
        )
        logger.info(
            "inspect before: ready=%s missing=%s totals=%s",
            before["ready"],
            before["missing_union"],
            {s: before["symbols"][s]["total_1m"] for s in symbols},
        )

        fill_report: dict = {}
        if not before["ready"] and not args.no_download:
            missing = before["missing_union"]
            if missing:
                logger.info("local seed %s → %s under %s", missing[0], missing[-1], stock_root)
                try:
                    fill_report["local_seed"] = seed_range(
                        root=stock_root,
                        symbols=symbols,
                        start=missing[0],
                        end=missing[-1],
                        pg_url=args.pg_url,
                    )
                except Exception as e:
                    logger.warning("local seed error: %s", e)
                    fill_report["local_seed"] = {"error": str(e)}

            mid = _inspect(
                conn,
                symbols,
                days,
                min_day_bars=args.min_day_bars,
                min_total_1m=args.min_total_1m,
            )
            still = mid["missing_union"]
            fill_report["downloaded"] = {}
            fill_report["still_empty"] = []
            for d in still:
                for sym in symbols:
                    lo, hi = _day_bounds(pd.Timestamp(d, tz=NY))
                    if _count_bars(conn, "market_bars_1m", sym, lo, hi) >= args.min_day_bars:
                        continue
                    logger.info("Massive/Polygon download %s %s", sym, d)
                    try:
                        df = download_day_aggs(
                            sym, d, api_key=api_key, api_base=args.api_base
                        )
                        n1, n5 = _upsert_day_frames(conn, sym, df)
                        fill_report["downloaded"].setdefault(d, {})[sym] = {
                            "bars_1m": n1,
                            "bars_5m": n5,
                        }
                        logger.info("  upserted %s %s → 1m=%d 5m=%d", sym, d, n1, n5)
                    except Exception as e:
                        logger.error("download %s %s failed: %s", sym, d, e)
                        fill_report["downloaded"].setdefault(d, {})[sym] = {
                            "error": str(e)
                        }
                    time.sleep(0.35)

            holiday_like = []
            for d in still:
                lo, hi = _day_bounds(pd.Timestamp(d, tz=NY))
                counts = {
                    s: _count_bars(conn, "market_bars_1m", s, lo, hi) for s in symbols
                }
                if all(v == 0 for v in counts.values()):
                    holiday_like.append(d)
                elif any(v < args.min_day_bars for v in counts.values()):
                    fill_report["still_empty"].append({"date": d, "counts": counts})
            fill_report["holiday_like"] = holiday_like

        after = _inspect(
            conn,
            symbols,
            days,
            min_day_bars=args.min_day_bars,
            min_total_1m=args.min_total_1m,
        )
        holidays = set((fill_report or {}).get("holiday_like") or [])
        if holidays:
            for _sym, info in after["symbols"].items():
                info["missing_days"] = [d for d in info["missing_days"] if d not in holidays]
                info["ok_days"] = len(info["missing_days"]) == 0
            after["missing_union"] = sorted(
                {d for info in after["symbols"].values() for d in info["missing_days"]}
            )
            after["ready"] = all(
                v["ok_total"] and v["ok_days"] for v in after["symbols"].values()
            )

        summary = {
            "asof": asof.strftime("%Y-%m-%d"),
            "lookback_tdays": args.lookback_tdays,
            "min_day_bars": args.min_day_bars,
            "min_total_1m": args.min_total_1m,
            "api_base": args.api_base,
            "before": before,
            "fill": fill_report,
            "after": after,
            "ready": bool(after["ready"]),
        }
        text = json.dumps(summary, indent=2, ensure_ascii=False, default=str)
        print(text)
        if args.out:
            out = Path(args.out).expanduser()
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(text, encoding="utf-8")

        if after["ready"]:
            logger.info("WARMUP DATA READY")
            return 0
        logger.error("WARMUP DATA NOT READY: missing=%s", after["missing_union"])
        return 2
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())

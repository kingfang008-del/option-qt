#!/usr/bin/env python3
"""Mag7 stock-only market-data publisher (separate IB client from options/OMS).

Keeps the stock MD client alive across option-side restarts. Writes:
  - fused stock stream ``…:stock``
  - tape / rth_opens
  - ``maga7:feed_health_stock:{session_id}``
"""
from __future__ import annotations

import argparse
import asyncio
import signal
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from maga7.common.config import load_profile, resolve_live_sessions_dir
from maga7.live.ibkr_connector import Mag7IbkrConfig, Mag7IbkrConnector

NY = ZoneInfo("America/New_York")


def _now_ny() -> datetime:
    return datetime.now(tz=NY)


async def _main(args: argparse.Namespace) -> int:
    profile = load_profile(args.profile)
    paths = profile.get("paths") or {}
    live_root = resolve_live_sessions_dir(paths)
    trade_date = args.trade_date or _now_ny().strftime("%Y-%m-%d")
    session_id = str(args.session_id or "").strip()
    if not session_id:
        raise SystemExit("--session-id is required (share with options engine)")
    session_dir = live_root / trade_date / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / "stock_md.role").write_text("stock\n", encoding="utf-8")

    symbols = [str(s).upper() for s in (profile.get("symbols") or [])]
    refs = ["QQQ"]
    config = Mag7IbkrConfig(
        host=args.ib_host,
        port=int(args.ib_port),
        client_id=int(args.client_id),
        redis_host=args.redis_host,
        redis_port=int(args.redis_port),
        redis_db=int(args.redis_db),
        market_data_type=int(args.market_data_type),
        md_role="stock",
    )
    connector = Mag7IbkrConnector(
        session_id=session_id,
        symbols=symbols,
        reference_symbols=refs,
        trade_date=trade_date,
        session_dir=session_dir,
        config=config,
        resume=bool(args.resume),
    )
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop.set)
        except NotImplementedError:
            pass

    await connector.connect(retries=args.connect_retries)
    await connector.subscribe_stocks()
    # Late-start / restart: backfill any missing official 09:30 opens from IB.
    try:
        from maga7.live.rth_open_store import (
            missing_rth_open_symbols,
            resolve_rth_opens,
        )

        rth_symbols = list(symbols) + list(refs)
        have = resolve_rth_opens(
            live_root,
            trade_date,
            redis_client=connector.redis,
            symbols=rth_symbols,
            recover_tapes=True,
        )
        missing = missing_rth_open_symbols(have, rth_symbols)
        now = _now_ny()
        past_open = (now.hour, now.minute) >= (9, 30) or str(now.date()) > str(
            trade_date
        )
        if missing and past_open:
            hist = await connector.fetch_rth_opens_historical(missing)
            if hist:
                print(
                    f"STOCK_MD RTH_OPEN_HIST n={len(hist)} symbols={sorted(hist)}",
                    flush=True,
                )
            still = missing_rth_open_symbols(
                resolve_rth_opens(
                    live_root,
                    trade_date,
                    redis_client=connector.redis,
                    symbols=rth_symbols,
                    recover_tapes=False,
                ),
                rth_symbols,
            )
            if still:
                print(f"STOCK_MD WARN RTH_OPEN_MISSING symbols={still}", flush=True)
    except Exception as exc:
        print(f"STOCK_MD WARN RTH_OPEN_BACKFILL_FAILED err={exc}", flush=True)
    tasks = [
        asyncio.create_task(connector.publish_loop(), name="stock-publisher"),
        asyncio.create_task(connector.heartbeat_loop(), name="stock-heartbeat"),
    ]
    connector.publish_status("STOCK_MD_RUNNING")
    print(
        f"STOCK_MD_RUNNING session={session_id} client_id={args.client_id} "
        f"symbols={len(symbols)+len(refs)} stream={connector.keys.get('stream_stock')}",
        flush=True,
    )
    end_at = None
    if args.end_time and args.end_time != "off":
        hh, mm = args.end_time.split(":")
        end_at = pd.Timestamp(
            year=int(trade_date[:4]),
            month=int(trade_date[5:7]),
            day=int(trade_date[8:10]),
            hour=int(hh),
            minute=int(mm),
            tz=NY,
        )
    try:
        while not stop.is_set():
            if end_at is not None and pd.Timestamp.now(tz=NY) >= end_at:
                break
            await asyncio.sleep(1.0)
    finally:
        connector.publish_status("STOCK_MD_STOPPING")
        connector.stop()
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        try:
            connector.ib.disconnect()
        except Exception:
            pass
        connector.publish_status("STOCK_MD_STOPPED")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        default="maga7/CONFIG/strategy_profiles/single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1.json",
    )
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--trade-date", default="")
    parser.add_argument("--ib-host", default="127.0.0.1")
    parser.add_argument("--ib-port", type=int, default=4001)
    parser.add_argument("--client-id", type=int, default=212)
    parser.add_argument("--redis-host", default="127.0.0.1")
    parser.add_argument("--redis-port", type=int, default=6379)
    parser.add_argument("--redis-db", type=int, default=0)
    parser.add_argument("--market-data-type", type=int, default=1)
    parser.add_argument("--connect-retries", type=int, default=0)
    parser.add_argument("--end-time", default="auto")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.end_time == "auto":
        args.end_time = "16:05"
    raise SystemExit(asyncio.run(_main(args)))


if __name__ == "__main__":
    main()

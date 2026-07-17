#!/usr/bin/env python3
"""Sync Mag7 event blackout calendar from APIs → JSON (+ optional Redis).

Phase A (predictable):
  - FOMC from Fed HTML (fallback built-in schedule)
  - Earnings from Finnhub (FINNHUB_API_KEY) or Polygon Benzinga
  - Manual overlay CONFIG/event_calendar_manual.json

Writes a file compatible with ``MAG7_EVENT_CALENDAR_PATH`` /
``resolve_live_event_blackout`` (``dates`` + rich ``events``).

Examples:
  python maga7/tools/sync_event_calendar.py --start 2026-05-01 --end 2026-07-31
  FINNHUB_API_KEY=... python maga7/tools/sync_event_calendar.py --redis
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from maga7.common.event_providers import (
    build_sync_payload,
    earnings_to_blackout_dates,
    fetch_earnings_finnhub,
    fetch_earnings_polygon,
    fetch_fomc_events_fed,
    load_manual_events,
    merge_event_rows,
)

DEFAULT_OUT = ROOT / "maga7" / "CONFIG" / "event_calendar_live.json"
DEFAULT_MANUAL = ROOT / "maga7" / "CONFIG" / "event_calendar_manual.json"
DEFAULT_SYMBOLS = ["NVDA", "TSLA", "AAPL", "AMZN", "META", "MSFT", "AMD", "GOOGL"]


def _write_redis(dates: list[str], *, key: str) -> str:
    try:
        import redis  # type: ignore
    except ImportError as exc:
        raise SystemExit(f"redis package required for --redis: {exc}") from exc
    url = os.environ.get("REDIS_URL", "redis://127.0.0.1:6379/0")
    client = redis.Redis.from_url(url, decode_responses=True)
    payload = json.dumps(dates)
    client.set(key, payload)
    return f"{url} key={key} n={len(dates)}"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--start", default="2026-05-01")
    p.add_argument("--end", default="2026-12-31")
    p.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--manual", default=str(DEFAULT_MANUAL))
    p.add_argument("--no-fomc", action="store_true")
    p.add_argument("--no-earnings", action="store_true")
    p.add_argument("--no-manual", action="store_true")
    p.add_argument("--fomc-meeting-day", action="store_true", default=True)
    p.add_argument("--no-fomc-meeting-day", action="store_true")
    p.add_argument(
        "--earnings-ah-plus1-cal",
        action="store_true",
        help="Also blackout calendar day after AH earnings (approx; prefer session expand)",
    )
    p.add_argument("--earnings-provider", choices=("auto", "finnhub", "polygon", "none"), default="auto")
    p.add_argument("--redis", action="store_true", help="Publish dates JSON list to Redis")
    p.add_argument(
        "--redis-key",
        default=os.environ.get("MAG7_EVENT_BLACKOUT_REDIS_KEY", "maga7:event_blackout"),
    )
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    include_meeting = bool(args.fomc_meeting_day) and not args.no_fomc_meeting_day
    rows: list[dict[str, Any]] = []
    sources: list[str] = []

    if not args.no_fomc:
        fomc, src = fetch_fomc_events_fed(
            start=args.start,
            end=args.end,
            include_meeting_day=include_meeting,
            include_decision_day=True,
        )
        rows.extend(fomc)
        sources.append(src)
        print(f"FOMC: {len(fomc)} events via {src}", flush=True)

    if not args.no_earnings and args.earnings_provider != "none":
        earn: list[dict[str, Any]] = []
        esrc = "none"
        provider = args.earnings_provider
        if provider == "auto":
            earn, esrc = fetch_earnings_finnhub(symbols, start=args.start, end=args.end)
            if not earn and "missing_key" in esrc:
                earn, esrc = fetch_earnings_polygon(symbols, start=args.start, end=args.end)
        elif provider == "finnhub":
            earn, esrc = fetch_earnings_finnhub(symbols, start=args.start, end=args.end)
        elif provider == "polygon":
            earn, esrc = fetch_earnings_polygon(symbols, start=args.start, end=args.end)
        mapped = earnings_to_blackout_dates(
            earn,
            ah_include_session=True,
            ah_include_next_cal_day=bool(args.earnings_ah_plus1_cal),
            bmo_include_session=True,
        )
        rows.extend(mapped)
        sources.append(esrc)
        print(f"Earnings: {len(mapped)} events via {esrc}", flush=True)
        if "missing_key" in esrc:
            print(
                "  hint: set FINNHUB_API_KEY or POLYGON_API_KEY/MASSIVE_API_KEY",
                flush=True,
            )

    if not args.no_manual:
        manual = load_manual_events(args.manual)
        # filter range
        manual = [e for e in manual if args.start <= str(e["date"]) <= args.end]
        rows.extend(manual)
        if manual:
            sources.append("manual")
        print(f"Manual: {len(manual)} events from {args.manual}", flush=True)

    merged = merge_event_rows(rows)
    payload = build_sync_payload(
        merged, start=args.start, end=args.end, sources=sources
    )
    print(
        f"Merged: {len(payload['dates'])} unique dates → {payload['dates']}",
        flush=True,
    )

    if args.dry_run:
        print(json.dumps(payload, indent=2)[:2000], flush=True)
        return

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}", flush=True)

    if args.redis:
        info = _write_redis(payload["dates"], key=args.redis_key)
        print(f"redis: {info}", flush=True)


if __name__ == "__main__":
    main()

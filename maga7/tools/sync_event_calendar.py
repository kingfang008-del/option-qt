#!/usr/bin/env python3
"""Sync Mag7 event blackout calendar from APIs → JSON (+ optional Redis).

Phase A (predictable):
  - FOMC from Fed HTML (fallback built-in schedule)
  - Earnings from Finnhub (FINNHUB_API_KEY) or Polygon Benzinga
  - Manual overlay CONFIG/event_calendar_manual.json

Phase C (company news — not a direction oracle):
  - Finnhub company-news + Investing RSS
  - Default ``--news-mode hard_risk``: score all; auto symbol-block **CEO only**
  - ``audit``: score only (no news_* rows); deals/capex never auto-ban
  - LLM stance is dash-only — never written into calendar here
  - Full scored headlines → event_news_audit.json

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
from datetime import date, timedelta
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from maga7.common.company_news import (
    DEFAULT_SUPPRESS_PATH,
    collect_company_news_events,
    filter_suppressed_events,
    write_news_audit,
)
from maga7.common.event_news_policy import (
    DEFAULT_NEWS_MODE,
    POLICY_SUMMARY_ZH,
    normalize_news_mode,
)
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
DEFAULT_NEWS_AUDIT = ROOT / "maga7" / "CONFIG" / "event_news_audit.json"
DEFAULT_NEWS_SUPPRESS = DEFAULT_SUPPRESS_PATH
DEFAULT_SYMBOLS = ["NVDA", "TSLA", "AAPL", "AMZN", "META", "MSFT", "AMD", "GOOGL"]


def _write_redis(payload: dict[str, Any], *, key: str) -> str:
    try:
        import redis  # type: ignore
    except ImportError as exc:
        raise SystemExit(f"redis package required for --redis: {exc}") from exc
    url = os.environ.get("REDIS_URL", "redis://127.0.0.1:6379/0")
    client = redis.Redis.from_url(url, decode_responses=True)
    # Structured: full-day dates + symbol_blackout (not a flat date list)
    body = {
        "dates": payload.get("dates") or [],
        "symbol_blackout": payload.get("symbol_blackout") or {},
        "events": payload.get("events") or [],
    }
    client.set(key, json.dumps(body))
    n_full = len(body["dates"])
    n_sym = sum(len(v) for v in body["symbol_blackout"].values())
    return f"{url} key={key} full_days={n_full} symbol_blocks={n_sym}"


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
    p.add_argument(
        "--no-news",
        action="store_true",
        help="Disable Finnhub company-news + Investing RSS ingest",
    )
    p.add_argument(
        "--news-mode",
        choices=("hard_risk", "audit", "blackout"),
        default=normalize_news_mode(os.environ.get("MAG7_NEWS_MODE", DEFAULT_NEWS_MODE)),
        help="hard_risk/blackout=CEO succession symbol-block only; "
        "audit=score only. News never sets trade direction.",
    )
    p.add_argument("--no-finnhub-news", action="store_true")
    p.add_argument("--no-rss-news", action="store_true")
    p.add_argument(
        "--rss-url",
        default=os.environ.get("MAG7_RSS_NEWS_URL", ""),
        help="Override Investing RSS URL (default cn.investing.com news_356)",
    )
    p.add_argument(
        "--news-audit-out",
        default=str(DEFAULT_NEWS_AUDIT),
        help="Scored headlines JSON (written whenever news is enabled)",
    )
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

    news_audit: list[dict[str, Any]] = []
    if not args.no_news:
        news_mode = normalize_news_mode(args.news_mode)
        print(f"NewsPolicy: {POLICY_SUMMARY_ZH} (mode={news_mode})", flush=True)
        # Prefer a short lookback for news (free Finnhub + RSS noise); clamp to args range.
        news_end = min(args.end, date.today().isoformat())
        news_start = max(args.start, (date.today() - timedelta(days=5)).isoformat())
        news_events, news_audit, news_sources = collect_company_news_events(
            symbols,
            start=news_start,
            end=news_end,
            rss_url=args.rss_url or None,
            news_mode=news_mode,
            enable_finnhub=not args.no_finnhub_news,
            enable_rss=not args.no_rss_news,
        )
        before = len(news_events)
        news_events = filter_suppressed_events(
            news_events, suppress_path=DEFAULT_NEWS_SUPPRESS
        )
        rows.extend(news_events)
        sources.extend(news_sources)
        print(
            f"CompanyNews: {len(news_events)} hard-risk rows "
            f"(mode={news_mode}, scored={len(news_audit)}, "
            f"suppressed={before - len(news_events)}) via {news_sources}",
            flush=True,
        )
        for e in news_events[:12]:
            print(
                f"  news {e.get('date')} {e.get('symbol')} {e.get('tag')}: "
                f"{(e.get('note') or '')[:80]}",
                flush=True,
            )

    merged = merge_event_rows(rows)
    payload = build_sync_payload(
        merged, start=args.start, end=args.end, sources=sources
    )
    print(
        f"Merged: full_days={payload['dates']} "
        f"symbol_blackout={payload.get('symbol_blackout')}",
        flush=True,
    )

    if args.dry_run:
        print(json.dumps(payload, indent=2)[:2000], flush=True)
        if news_audit:
            print(
                f"[dry-run] would write news audit n={len(news_audit)} → {args.news_audit_out}",
                flush=True,
            )
        return

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}", flush=True)

    if not args.no_news:
        write_news_audit(args.news_audit_out, news_audit)
        print(f"wrote news audit {args.news_audit_out}", flush=True)

    if args.redis:
        info = _write_redis(payload, key=args.redis_key)
        print(f"redis: {info}", flush=True)


if __name__ == "__main__":
    main()

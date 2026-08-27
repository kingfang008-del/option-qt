"""External calendar providers for Mag7 event blackout (Phase A).

Sources (best-effort, offline-safe fallbacks):
  - FOMC: Fed HTML scrape + built-in 2025–2026 schedule
  - Earnings: Finnhub calendar API (``FINNHUB_API_KEY``) or Polygon
    (``POLYGON_API_KEY`` / ``MASSIVE_API_KEY``)
  - Manual overlay JSON (IPO / rate-spike dates APIs will miss)

All returns are ``{date, tag, source, symbol?}`` dicts.
"""
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

NY = "America/New_York"

# Fed-published meeting windows (first day = meeting, second = decision when 2-day).
# Kept as offline fallback when scrape fails. Update yearly.
FOMC_MEETINGS_BUILTIN: list[tuple[str, str | None]] = [
    # (meeting_or_decision_day, decision_day_or_None if single-day)
    ("2025-01-28", "2025-01-29"),
    ("2025-03-18", "2025-03-19"),
    ("2025-05-06", "2025-05-07"),
    ("2025-06-17", "2025-06-18"),
    ("2025-07-29", "2025-07-30"),
    ("2025-09-16", "2025-09-17"),
    ("2025-10-28", "2025-10-29"),
    ("2025-12-09", "2025-12-10"),
    ("2026-01-27", "2026-01-28"),
    ("2026-03-17", "2026-03-18"),
    ("2026-05-05", "2026-05-06"),
    ("2026-06-16", "2026-06-17"),
    ("2026-07-28", "2026-07-29"),
    ("2026-09-15", "2026-09-16"),
    ("2026-10-27", "2026-10-28"),
    ("2026-12-08", "2026-12-09"),
]

FED_FOMC_URL = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
FINNHUB_EARNINGS_URL = "https://finnhub.io/api/v1/calendar/earnings"
POLYGON_EARNINGS_URL = "https://api.polygon.io/vX/reference/financials"  # not ideal; use benzinga if present


def _http_get(url: str, *, timeout: float = 20.0) -> str:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "maga7-event-calendar/1.0"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def _in_range(d: str, start: str, end: str) -> bool:
    return start <= d <= end


def fomc_events_from_builtin(
    *,
    start: str,
    end: str,
    include_meeting_day: bool = True,
    include_decision_day: bool = True,
    include_plus1: bool = False,
) -> list[dict[str, Any]]:
    """Expand built-in FOMC windows into blackout event rows."""
    out: list[dict[str, Any]] = []
    for meet, decision in FOMC_MEETINGS_BUILTIN:
        dec = decision or meet
        if include_meeting_day and meet != dec and _in_range(meet, start, end):
            out.append(
                {
                    "date": meet,
                    "tag": "fomc_meeting",
                    "source": "fomc_builtin",
                    "symbol": None,
                }
            )
        if include_decision_day and _in_range(dec, start, end):
            out.append(
                {
                    "date": dec,
                    "tag": "fomc_decision",
                    "source": "fomc_builtin",
                    "symbol": None,
                }
            )
        if include_plus1:
            plus = (date.fromisoformat(dec) + timedelta(days=1)).isoformat()
            # caller may expand via trading sessions; here calendar +1 only
            if _in_range(plus, start, end):
                out.append(
                    {
                        "date": plus,
                        "tag": "fomc_plus1_cal",
                        "source": "fomc_builtin",
                        "symbol": None,
                    }
                )
    return out


def fetch_fomc_events_fed(
    *,
    start: str,
    end: str,
    include_meeting_day: bool = True,
    include_decision_day: bool = True,
) -> tuple[list[dict[str, Any]], str]:
    """Scrape Fed FOMC page for month/day pairs; fall back to builtin.

    Returns ``(events, source_label)``.
    """
    try:
        html = _http_get(FED_FOMC_URL)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        ev = fomc_events_from_builtin(
            start=start,
            end=end,
            include_meeting_day=include_meeting_day,
            include_decision_day=include_decision_day,
            include_plus1=False,
        )
        return ev, f"fomc_builtin(fallback:{exc.__class__.__name__})"

    # Match patterns like "June 16-17" / "January 27-28, 2026" on the page.
    year_blocks = re.findall(
        r"(20\d{2})\s+FOMC\s+Meetings(.*?)(?:20\d{2}\s+FOMC\s+Meetings|</div>\s*</div>\s*</div>)",
        html,
        flags=re.I | re.S,
    )
    month_map = {
        "january": 1,
        "february": 2,
        "march": 3,
        "april": 4,
        "may": 5,
        "june": 6,
        "july": 7,
        "august": 8,
        "september": 9,
        "october": 10,
        "november": 11,
        "december": 12,
    }
    scraped: list[tuple[str, str]] = []
    if not year_blocks:
        # looser: "June 16-17" near a year
        for m in re.finditer(
            r"(January|February|March|April|May|June|July|August|September|October|November|December)"
            r"\s+(\d{1,2})\s*[-–]\s*(\d{1,2})",
            html,
            flags=re.I,
        ):
            # year: look back for 20xx
            head = html[max(0, m.start() - 200) : m.start()]
            ys = re.findall(r"20\d{2}", head)
            if not ys:
                continue
            y = int(ys[-1])
            mo = month_map[m.group(1).lower()]
            d1, d2 = int(m.group(2)), int(m.group(3))
            scraped.append(
                (
                    date(y, mo, d1).isoformat(),
                    date(y, mo, d2).isoformat(),
                )
            )
    else:
        for y_str, block in year_blocks:
            y = int(y_str)
            for m in re.finditer(
                r"(January|February|March|April|May|June|July|August|September|October|November|December)"
                r"\s+(\d{1,2})\s*[-–]\s*(\d{1,2})",
                block,
                flags=re.I,
            ):
                mo = month_map[m.group(1).lower()]
                d1, d2 = int(m.group(2)), int(m.group(3))
                scraped.append(
                    (
                        date(y, mo, d1).isoformat(),
                        date(y, mo, d2).isoformat(),
                    )
                )

    if not scraped:
        ev = fomc_events_from_builtin(
            start=start,
            end=end,
            include_meeting_day=include_meeting_day,
            include_decision_day=include_decision_day,
        )
        return ev, "fomc_builtin(scrape_empty)"

    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for meet, dec in scraped:
        if include_meeting_day and meet != dec and _in_range(meet, start, end):
            key = (meet, "fomc_meeting")
            if key not in seen:
                seen.add(key)
                out.append(
                    {
                        "date": meet,
                        "tag": "fomc_meeting",
                        "source": "fomc_fed_html",
                        "symbol": None,
                    }
                )
        if include_decision_day and _in_range(dec, start, end):
            key = (dec, "fomc_decision")
            if key not in seen:
                seen.add(key)
                out.append(
                    {
                        "date": dec,
                        "tag": "fomc_decision",
                        "source": "fomc_fed_html",
                        "symbol": None,
                    }
                )
    if not out:
        # Page parsed but nothing in requested window → builtin for that window.
        ev = fomc_events_from_builtin(
            start=start,
            end=end,
            include_meeting_day=include_meeting_day,
            include_decision_day=include_decision_day,
        )
        return ev, "fomc_builtin(scrape_out_of_range)"
    return out, "fomc_fed_html"


def _finnhub_key() -> str | None:
    """Resolve Finnhub token: env first, then local key files (no export needed).

    Search order:
      1. ``FINNHUB_API_KEY`` / ``FINNHUB_KEY``
      2. ``MAG7_FINNHUB_KEY_FILE`` path
      3. ``~/finnhub.txt``
      4. ``~/.config/maga7/finnhub.txt``
    """
    for k in ("FINNHUB_API_KEY", "FINNHUB_KEY"):
        v = os.environ.get(k, "").strip()
        if v:
            return v
    candidates: list[Path] = []
    env_path = os.environ.get("MAG7_FINNHUB_KEY_FILE", "").strip()
    if env_path:
        candidates.append(Path(env_path).expanduser())
    home = Path.home()
    candidates.extend(
        [
            home / "finnhub.txt",
            home / ".config" / "maga7" / "finnhub.txt",
        ]
    )
    for p in candidates:
        try:
            if p.is_file():
                text = p.read_text(encoding="utf-8").strip()
                # allow "KEY=xxx" or first non-empty/non-comment line
                for line in text.splitlines():
                    s = line.split("#", 1)[0].strip()
                    if not s:
                        continue
                    if "=" in s and s.upper().startswith("FINNHUB"):
                        s = s.split("=", 1)[1].strip().strip('"').strip("'")
                    if s:
                        return s
        except OSError:
            continue
    return None


def _polygon_key() -> str | None:
    for k in ("POLYGON_API_KEY", "MASSIVE_API_KEY", "POLYGON_KEY"):
        v = os.environ.get(k, "").strip()
        if v:
            return v
    return None


def fetch_earnings_finnhub(
    symbols: Iterable[str],
    *,
    start: str,
    end: str,
) -> tuple[list[dict[str, Any]], str]:
    """Finnhub earnings calendar; AH → tag ``earnings_ah``, BMO → ``earnings_bmo``."""
    key = _finnhub_key()
    if not key:
        return [], "finnhub(missing_key)"
    sym_set = {str(s).upper() for s in symbols}
    q = urllib.parse.urlencode({"from": start, "to": end, "token": key})
    url = f"{FINNHUB_EARNINGS_URL}?{q}"
    try:
        raw = json.loads(_http_get(url))
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        return [], f"finnhub(error:{exc.__class__.__name__})"
    rows = raw.get("earningsCalendar") or raw.get("data") or []
    out: list[dict[str, Any]] = []
    for r in rows:
        sym = str(r.get("symbol") or "").upper()
        if sym_set and sym not in sym_set:
            continue
        d = str(r.get("date") or "").strip()
        if not d or not _in_range(d, start, end):
            continue
        hour = str(r.get("hour") or r.get("time") or "").strip().lower()
        if hour in {"amc", "ah", "aftermarket", "after-market", "after hours"}:
            tag = "earnings_ah"
        elif hour in {"bmo", "am", "premarket", "before-market", "before open"}:
            tag = "earnings_bmo"
        else:
            tag = "earnings"
        out.append(
            {
                "date": d,
                "tag": tag,
                "source": "finnhub",
                "symbol": sym,
            }
        )
    return out, "finnhub"


def fetch_earnings_polygon(
    symbols: Iterable[str],
    *,
    start: str,
    end: str,
) -> tuple[list[dict[str, Any]], str]:
    """Polygon Benzinga earnings (if plan supports ``/benzinga/v1/earnings``)."""
    key = _polygon_key()
    if not key:
        return [], "polygon(missing_key)"
    sym_set = {str(s).upper() for s in symbols}
    out: list[dict[str, Any]] = []
    # One request with date filter; filter symbols client-side.
    q = urllib.parse.urlencode(
        {
            "apiKey": key,
            "date.gte": start,
            "date.lte": end,
            "limit": 1000,
        }
    )
    url = f"https://api.polygon.io/benzinga/v1/earnings?{q}"
    try:
        raw = json.loads(_http_get(url))
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        return [], f"polygon(error:{exc.__class__.__name__})"
    rows = raw.get("results") or []
    for r in rows:
        sym = str(r.get("ticker") or r.get("symbol") or "").upper()
        if sym_set and sym not in sym_set:
            continue
        d = str(r.get("date") or r.get("earnings_date") or "").strip()[:10]
        if not d or not _in_range(d, start, end):
            continue
        timing = str(r.get("time") or r.get("hour") or "").lower()
        if "after" in timing or timing in {"amc", "ah"}:
            tag = "earnings_ah"
        elif "before" in timing or timing in {"bmo"}:
            tag = "earnings_bmo"
        else:
            tag = "earnings"
        out.append({"date": d, "tag": tag, "source": "polygon", "symbol": sym})
    return out, "polygon"


def load_manual_events(path: str | Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    p = Path(path).expanduser()
    if not p.is_file():
        return []
    raw = json.loads(p.read_text(encoding="utf-8"))
    rows = raw.get("events") or raw.get("dates") or raw
    out: list[dict[str, Any]] = []
    for x in rows if isinstance(rows, list) else []:
        if isinstance(x, str):
            out.append({"date": x, "tag": "manual", "source": "manual", "symbol": None})
        elif isinstance(x, dict) and x.get("date"):
            out.append(
                {
                    "date": str(x["date"]),
                    "tag": str(x.get("tag") or "manual"),
                    "source": str(x.get("source") or "manual"),
                    "symbol": x.get("symbol"),
                }
            )
    return out


def earnings_to_blackout_dates(
    earnings: Iterable[dict[str, Any]],
    *,
    ah_include_session: bool = True,
    ah_include_next_cal_day: bool = False,
    bmo_include_session: bool = True,
) -> list[dict[str, Any]]:
    """Map earnings rows to session blackout events.

    AH: block the earnings date (RTH already happened or noisy AH tape);
    research full_day used NVDA AH day + next session explicitly via curated list.
    """
    out: list[dict[str, Any]] = []
    for e in earnings:
        d = str(e.get("date") or "")
        tag = str(e.get("tag") or "earnings")
        if tag == "earnings_ah":
            if ah_include_session:
                out.append({**e, "tag": "earnings_ah"})
            if ah_include_next_cal_day:
                nxt = (date.fromisoformat(d) + timedelta(days=1)).isoformat()
                out.append(
                    {
                        **e,
                        "date": nxt,
                        "tag": "earnings_ah_plus1_cal",
                    }
                )
        elif tag == "earnings_bmo":
            if bmo_include_session:
                out.append({**e, "tag": "earnings_bmo"})
        else:
            out.append(e)
    return out


def merge_event_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Dedupe by (date, tag, symbol); keep first source."""
    seen: set[tuple[str, str, str]] = set()
    out: list[dict[str, Any]] = []
    for r in sorted(rows, key=lambda x: (str(x.get("date")), str(x.get("tag")), str(x.get("symbol") or ""))):
        d = str(r.get("date") or "").strip()
        if not d:
            continue
        key = (d, str(r.get("tag") or ""), str(r.get("symbol") or "").upper())
        if key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "date": d,
                "tag": str(r.get("tag") or "event"),
                "source": str(r.get("source") or "unknown"),
                "symbol": r.get("symbol"),
            }
        )
    return out


def build_sync_payload(
    events: list[dict[str, Any]],
    *,
    start: str,
    end: str,
    sources: list[str],
) -> dict[str, Any]:
    from maga7.common.event_calendar import plan_from_events

    plan = plan_from_events(events)
    return {
        "description": (
            "Auto-synced Mag7 event blackout. "
            "dates=full-day macro only; symbol_blackout=earnings/news per ticker."
        ),
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "range": {"start": start, "end": end},
        "sources": sources,
        "dates": sorted(plan.full_days),
        "symbol_blackout": {
            d: sorted(syms) for d, syms in sorted(plan.symbol_days.items())
        },
        "events": events,
    }

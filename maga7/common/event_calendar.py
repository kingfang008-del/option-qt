"""Event-day blackout helpers (FOMC / mega earnings / giant IPO).

Research filter: skip Mag7 entries on known high-noise calendar days.
Dates are session dates (YYYY-MM-DD). ``blackout_sessions`` extends the
block to the next N trading sessions after each event (0 = event day only).

Live ingest (pre-open):
  - profile ``regime.event_calendar_block`` + preset/dates
  - file ``MAG7_EVENT_CALENDAR_PATH`` (JSON or one date per line)
  - Redis key ``MAG7_EVENT_BLACKOUT_REDIS_KEY`` (default ``maga7:event_blackout``)
  - force today: env ``MAG7_EVENT_BLACKOUT_TODAY=1``
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable

# Curated May–Jul 2026 research calendar (US equity sessions).
# Tags are documentation only; blocking uses the date set.
DEFAULT_EVENTS_MAY_JUL_2026: list[dict[str, str]] = [
    {"date": "2026-05-19", "tag": "ust_30y_spike"},
    {"date": "2026-05-20", "tag": "nvda_earnings_ah"},
    {"date": "2026-05-21", "tag": "post_nvda_earnings"},
    {"date": "2026-06-12", "tag": "spacex_ipo"},
    {"date": "2026-06-16", "tag": "fomc_day1"},
    {"date": "2026-06-17", "tag": "fomc_decision"},
    {"date": "2026-06-18", "tag": "fomc_plus1"},
]

# Tighter set: primary catalysts only (earnings AH / IPO / FOMC decision).
CORE_EVENTS_MAY_JUL_2026: list[dict[str, str]] = [
    {"date": "2026-05-20", "tag": "nvda_earnings_ah"},
    {"date": "2026-06-12", "tag": "spacex_ipo"},
    {"date": "2026-06-17", "tag": "fomc_decision"},
]


def _as_date_list(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        return [raw.strip()] if raw.strip() else []
    out: list[str] = []
    for x in raw:
        if isinstance(x, dict):
            d = str(x.get("date") or "").strip()
        else:
            d = str(x).strip()
        if d:
            out.append(d)
    return out


def event_dates_from_cfg(cfg: dict[str, Any] | None) -> list[str]:
    """Resolve event dates from regime/trade cfg.

    ``event_calendar``:
      - ``"default"`` / ``"may_jul_2026"`` → full curated list
      - ``"core"`` → earnings / IPO / FOMC decision only
      - list of dates or ``[{date, tag}, ...]``
    ``event_dates``: explicit override list (wins over ``event_calendar`` preset).
    """
    cfg = cfg or {}
    explicit = cfg.get("event_dates")
    if explicit is not None:
        return sorted(set(_as_date_list(explicit)))
    preset = str(cfg.get("event_calendar") or "").strip().lower()
    if preset in {"", "off", "none", "false", "0"}:
        return []
    if preset in {"default", "may_jul_2026", "full"}:
        return sorted({e["date"] for e in DEFAULT_EVENTS_MAY_JUL_2026})
    if preset in {"core", "tight"}:
        return sorted({e["date"] for e in CORE_EVENTS_MAY_JUL_2026})
    # treat as single date string fallback
    return _as_date_list(preset)


def expand_blackout_dates(
    event_dates: Iterable[str],
    *,
    session_dates: Iterable[str],
    blackout_sessions: int = 0,
) -> set[str]:
    """Include each event date plus the next ``blackout_sessions`` sessions."""
    sessions = sorted({str(d) for d in session_dates})
    if not sessions:
        return set(str(d) for d in event_dates)
    idx = {d: i for i, d in enumerate(sessions)}
    n_extra = max(0, int(blackout_sessions or 0))
    out: set[str] = set()
    for d in event_dates:
        ds = str(d)
        if ds not in idx:
            out.add(ds)
            continue
        i0 = idx[ds]
        for j in range(0, n_extra + 1):
            k = i0 + j
            if k < len(sessions):
                out.add(sessions[k])
    return out


def resolve_event_blackout(
    cfg: dict[str, Any] | None,
    *,
    session_dates: Iterable[str],
) -> set[str]:
    """Full blackout set from cfg, or empty if disabled.

    Requires ``event_calendar_block=true``. When enabled with no dates/preset,
    uses the full May–Jul 2026 curated list.
    """
    cfg = cfg or {}
    enabled = cfg.get("event_calendar_block", False)
    if isinstance(enabled, str):
        enabled = enabled.strip().lower() in {"1", "true", "yes", "on"}
    if not enabled:
        return set()
    dates = event_dates_from_cfg(cfg)
    if not dates:
        dates = sorted({e["date"] for e in DEFAULT_EVENTS_MAY_JUL_2026})
    blackout_sessions = int(cfg.get("event_blackout_sessions", 0) or 0)
    return expand_blackout_dates(
        dates, session_dates=session_dates, blackout_sessions=blackout_sessions
    )


def is_event_blackout_day(date: str, blackout: set[str] | None) -> bool:
    if not blackout:
        return False
    return str(date) in blackout


def event_cfg_from_profile(profile: dict[str, Any] | None) -> dict[str, Any]:
    """Merge regime.* then trade.* event-calendar keys."""
    profile = profile or {}
    trade = profile.get("trade") or {}
    reg = profile.get("regime") or {}
    keys = (
        "event_calendar_block",
        "event_calendar",
        "event_dates",
        "event_blackout_sessions",
        "event_calendar_path",
    )
    out: dict[str, Any] = {}
    for k in keys:
        if k in reg:
            out[k] = reg[k]
    for k in keys:
        if k in trade:
            out[k] = trade[k]
    return out


def load_event_dates_file(path: str | Path | None) -> list[str]:
    """Load dates from JSON ``{dates|events: [...]}`` or plain text lines."""
    if path is None:
        return []
    p = Path(path).expanduser()
    if not p.is_file():
        return []
    text = p.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if p.suffix.lower() == ".json" or text[:1] in "[{":
        raw = json.loads(text)
        if isinstance(raw, dict):
            if "dates" in raw:
                return _as_date_list(raw["dates"])
            if "events" in raw:
                return _as_date_list(raw["events"])
            if "event_dates" in raw:
                return _as_date_list(raw["event_dates"])
        return _as_date_list(raw)
    dates: list[str] = []
    for line in text.splitlines():
        s = line.split("#", 1)[0].strip()
        if s:
            dates.append(s)
    return sorted(set(dates))


def load_event_dates_redis(redis_client: Any, *, key: str | None = None) -> list[str]:
    """Read JSON list / CSV / set members from Redis."""
    if redis_client is None:
        return []
    key = key or os.environ.get("MAG7_EVENT_BLACKOUT_REDIS_KEY", "maga7:event_blackout")
    try:
        raw = redis_client.get(key)
    except Exception:
        return []
    if raw is None:
        # optional Redis SET
        try:
            members = redis_client.smembers(key)
            if members:
                return sorted(
                    {
                        (m.decode() if isinstance(m, (bytes, bytearray)) else str(m)).strip()
                        for m in members
                        if m
                    }
                )
        except Exception:
            return []
        return []
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8", errors="replace")
    text = str(raw).strip()
    if not text:
        return []
    if text[:1] in "[{":
        try:
            return _as_date_list(json.loads(text))
        except Exception:
            pass
    return sorted({x.strip() for x in text.replace(";", ",").split(",") if x.strip()})


def resolve_live_event_blackout(
    profile: dict[str, Any] | None,
    *,
    trade_date: str,
    session_dates: Iterable[str] | None = None,
    redis_client: Any = None,
) -> tuple[set[str], dict[str, Any]]:
    """Resolve blackout for a live session and report sources.

    Returns ``(blackout_dates, meta)``. Meta includes ``active_today`` and
    which sources contributed dates.
    """
    cfg = event_cfg_from_profile(profile)
    sources: list[str] = []
    dates: list[str] = []

    # 1) profile preset / explicit dates (only if block enabled)
    enabled = cfg.get("event_calendar_block", False)
    if isinstance(enabled, str):
        enabled = enabled.strip().lower() in {"1", "true", "yes", "on"}
    if enabled:
        dates.extend(event_dates_from_cfg(cfg))
        if not event_dates_from_cfg(cfg):
            dates.extend(e["date"] for e in DEFAULT_EVENTS_MAY_JUL_2026)
        sources.append("profile")

    # 2) file path (env wins over profile path)
    path = os.environ.get("MAG7_EVENT_CALENDAR_PATH") or cfg.get("event_calendar_path")
    file_dates = load_event_dates_file(path)
    if file_dates:
        dates.extend(file_dates)
        sources.append(f"file:{path}")

    # 3) Redis
    redis_dates = load_event_dates_redis(redis_client)
    if redis_dates:
        dates.extend(redis_dates)
        sources.append("redis")

    # 4) force today
    force_today = os.environ.get("MAG7_EVENT_BLACKOUT_TODAY", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if force_today:
        dates.append(str(trade_date))
        sources.append("env:MAG7_EVENT_BLACKOUT_TODAY")

    dates = sorted(set(dates))
    # If only external sources (file/redis/env) provided dates, treat as enabled.
    if not enabled and not dates:
        meta = {
            "enabled": False,
            "active_today": False,
            "sources": [],
            "event_dates": [],
            "blackout_dates": [],
            "trade_date": trade_date,
        }
        return set(), meta

    sessions = list(session_dates) if session_dates is not None else []
    if not sessions:
        # minimal session axis so +N expand still works for nearby dates
        sessions = sorted(set(dates) | {str(trade_date)})
    blackout_sessions = int(cfg.get("event_blackout_sessions", 0) or 0)
    # External-only ingest: do not expand unless profile asked for it.
    blackout = expand_blackout_dates(
        dates, session_dates=sessions, blackout_sessions=blackout_sessions
    )
    active = str(trade_date) in blackout
    meta = {
        "enabled": True,
        "active_today": active,
        "sources": sources,
        "event_dates": dates,
        "blackout_dates": sorted(blackout),
        "event_blackout_sessions": blackout_sessions,
        "trade_date": trade_date,
    }
    return blackout, meta

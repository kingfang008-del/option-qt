"""Event-day blackout helpers (FOMC / mega earnings / giant IPO / company news).

Two scopes (important for live):
  - **full_day**: macro / Mag7-wide — no new entries that session
  - **symbol**: earnings / company-news — only that ticker is blocked

Research presets (``event_calendar=feb_jul`` …) remain full-day date lists.
Live sync file carries rich ``events`` + ``dates`` (full only) + ``symbol_blackout``.

Live ingest (pre-open):
  - profile ``regime.event_calendar_block`` + preset/dates
  - file ``MAG7_EVENT_CALENDAR_PATH`` (JSON or one date per line)
  - Redis key ``MAG7_EVENT_BLACKOUT_REDIS_KEY`` (default ``maga7:event_blackout``)
  - force today: env ``MAG7_EVENT_BLACKOUT_TODAY=1`` (full-day)
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

# Macro / Mag7-wide — always full-day even if a symbol is attached.
FULL_DAY_TAGS: frozenset[str] = frozenset(
    {
        "fomc_meeting",
        "fomc_decision",
        "fomc_plus1",
        "fomc_day1",
        "fomc_plus1_cal",
        "nfp",
        "cpi",
        "geopol_risk",
        "ust_30y_spike",
        "spacex_ipo",
        "mag7_capex_shock",
    }
)

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

# High-confidence misses from ≤−5% day scan (Feb–Apr) — not in May–Jul default.
# See results/.../loss_day_event_scan.md. Opt-in via event_calendar=feb_jul / extended.
LOSS_SCAN_EVENTS_FEB_APR_2026: list[dict[str, str]] = [
    {"date": "2026-02-05", "tag": "mag7_capex_shock"},
    {"date": "2026-02-11", "tag": "nfp"},
    {"date": "2026-02-13", "tag": "cpi"},
    {"date": "2026-03-03", "tag": "geopol_risk"},
    {"date": "2026-04-29", "tag": "fomc_decision"},
]

EXTENDED_EVENTS_FEB_JUL_2026: list[dict[str, str]] = (
    LOSS_SCAN_EVENTS_FEB_APR_2026 + DEFAULT_EVENTS_MAY_JUL_2026
)

# Soft research overlay: AAPL CEO succession (Cook→Ternus) ~AH 04-21 / session 04-22.
# See results/.../remaining9_stock_news.md. Opt-in via event_calendar=feb_jul_aapl_ceo.
AAPL_CEO_EVENTS_2026: list[dict[str, str]] = [
    {"date": "2026-04-21", "tag": "aapl_ceo_succession"},
    {"date": "2026-04-22", "tag": "aapl_ceo_succession"},
]

EXTENDED_EVENTS_FEB_JUL_AAPL_CEO_2026: list[dict[str, str]] = (
    EXTENDED_EVENTS_FEB_JUL_2026 + AAPL_CEO_EVENTS_2026
)


@dataclass
class EventBlackoutPlan:
    """Full-day dates + per-session symbol blocks."""

    full_days: set[str] = field(default_factory=set)
    symbol_days: dict[str, set[str]] = field(default_factory=dict)

    def blocks_day(self, date: str) -> bool:
        return str(date) in self.full_days

    def blocks_symbol(self, date: str, symbol: str | None) -> bool:
        if self.blocks_day(date):
            return True
        if not symbol:
            return False
        return str(symbol).upper() in (self.symbol_days.get(str(date)) or set())

    def symbols_blocked_on(self, date: str) -> set[str]:
        return set(self.symbol_days.get(str(date)) or set())

    def merge(self, other: "EventBlackoutPlan") -> "EventBlackoutPlan":
        out = EventBlackoutPlan(full_days=set(self.full_days), symbol_days={})
        out.full_days |= set(other.full_days)
        for src in (self.symbol_days, other.symbol_days):
            for d, syms in src.items():
                out.symbol_days.setdefault(str(d), set()).update(
                    str(s).upper() for s in syms if s
                )
        return out

    def as_meta(self, *, trade_date: str | None = None) -> dict[str, Any]:
        td = str(trade_date) if trade_date else None
        return {
            "full_days": sorted(self.full_days),
            "symbol_blackout": {
                d: sorted(syms) for d, syms in sorted(self.symbol_days.items())
            },
            "active_today_full": bool(td and self.blocks_day(td)),
            "active_today_symbols": sorted(self.symbols_blocked_on(td)) if td else [],
        }


def event_scope(event: dict[str, Any]) -> str:
    """Return ``full`` or ``symbol`` for one event row."""
    tag = str(event.get("tag") or "").strip()
    sym = str(event.get("symbol") or "").strip().upper()
    if tag in FULL_DAY_TAGS:
        return "full"
    if tag.startswith("news_") or tag.startswith("earnings"):
        return "symbol" if sym else "full"
    # Curated research tags like nvda_earnings_ah without symbol → Mag7-wide
    if sym:
        return "symbol"
    return "full"


def plan_from_events(events: Iterable[dict[str, Any]]) -> EventBlackoutPlan:
    plan = EventBlackoutPlan()
    for e in events:
        d = str(e.get("date") or "").strip()
        if not d:
            continue
        if event_scope(e) == "symbol":
            sym = str(e.get("symbol") or "").strip().upper()
            if sym:
                plan.symbol_days.setdefault(d, set()).add(sym)
            else:
                plan.full_days.add(d)
        else:
            plan.full_days.add(d)
    return plan


def plan_from_symbol_blackout(raw: Any) -> EventBlackoutPlan:
    """Parse ``{date: [sym, ...]}`` or ``{date: sym}`` maps."""
    plan = EventBlackoutPlan()
    if not isinstance(raw, dict):
        return plan
    for d, syms in raw.items():
        ds = str(d).strip()
        if not ds:
            continue
        if isinstance(syms, str):
            items = [syms]
        else:
            items = list(syms or [])
        for s in items:
            su = str(s).strip().upper()
            if su:
                plan.symbol_days.setdefault(ds, set()).add(su)
    return plan


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
      - ``"feb_jul"`` → May–Jul + loss-scan Feb–Apr
      - ``"feb_jul_aapl_ceo"`` → feb_jul + AAPL CEO succession 04-21/22
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
    if preset in {"feb_jul", "feb_jul_2026", "extended", "loss_scan"}:
        return sorted({e["date"] for e in EXTENDED_EVENTS_FEB_JUL_2026})
    if preset in {
        "feb_jul_aapl_ceo",
        "feb_jul_aapl_ceo_2026",
        "extended_aapl_ceo",
        "aapl_ceo",
    }:
        return sorted({e["date"] for e in EXTENDED_EVENTS_FEB_JUL_AAPL_CEO_2026})
    if preset in {"feb_apr", "feb_apr_2026", "loss_scan_feb_apr"}:
        return sorted({e["date"] for e in LOSS_SCAN_EVENTS_FEB_APR_2026})
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


def resolve_event_blackout_plan(
    cfg: dict[str, Any] | None,
    *,
    session_dates: Iterable[str],
) -> EventBlackoutPlan:
    """Resolve full-day + symbol plan from regime/trade cfg.

    Preset / ``event_dates`` → **full-day** (research curated Mag7-wide).
    Optional ``event_calendar_path`` / ``event_symbol_blackout`` add symbol scope.
    """
    cfg = cfg or {}
    enabled = cfg.get("event_calendar_block", False)
    if isinstance(enabled, str):
        enabled = enabled.strip().lower() in {"1", "true", "yes", "on"}
    if not enabled:
        return EventBlackoutPlan()

    dates = event_dates_from_cfg(cfg)
    if not dates:
        dates = sorted({e["date"] for e in DEFAULT_EVENTS_MAY_JUL_2026})
    blackout_sessions = int(cfg.get("event_blackout_sessions", 0) or 0)
    full = expand_blackout_dates(
        dates, session_dates=session_dates, blackout_sessions=blackout_sessions
    )
    plan = EventBlackoutPlan(full_days=set(full))
    # Offline research stays on preset dates (full-day). Live file/Redis is
    # ingested only via ``resolve_live_event_blackout`` so replay is not polluted.
    plan = plan.merge(plan_from_symbol_blackout(cfg.get("event_symbol_blackout")))
    return plan


def resolve_event_blackout(
    cfg: dict[str, Any] | None,
    *,
    session_dates: Iterable[str],
) -> set[str]:
    """Full-day blackout dates only (backward compatible).

    Company-news / single-name earnings live in
    ``resolve_event_blackout_plan(...).symbol_days``.
    """
    return set(resolve_event_blackout_plan(cfg, session_dates=session_dates).full_days)


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
        "event_symbol_blackout",
    )
    out: dict[str, Any] = {}
    for k in keys:
        if k in reg:
            out[k] = reg[k]
    for k in keys:
        if k in trade:
            out[k] = trade[k]
    return out


def load_event_plan_file(path: str | Path | None) -> EventBlackoutPlan:
    """Load full-day + symbol plan from sync JSON (or legacy date list)."""
    if path is None:
        return EventBlackoutPlan()
    p = Path(path).expanduser()
    if not p.is_file():
        return EventBlackoutPlan()
    text = p.read_text(encoding="utf-8").strip()
    if not text:
        return EventBlackoutPlan()
    if p.suffix.lower() == ".json" or text[:1] in "[{":
        raw = json.loads(text)
        if isinstance(raw, dict):
            events = raw.get("events") if isinstance(raw.get("events"), list) else []
            # New sync format: dates=full-only + symbol_blackout map
            if "symbol_blackout" in raw:
                plan = EventBlackoutPlan(full_days=set(_as_date_list(raw.get("dates"))))
                plan = plan.merge(plan_from_symbol_blackout(raw.get("symbol_blackout")))
                # Reclassify events in case map lagged behind
                plan = plan.merge(plan_from_events(events))
                return plan
            if events:
                plan = plan_from_events(events)
                dated = {str(e.get("date")) for e in events if e.get("date")}
                for d in _as_date_list(raw.get("dates") or raw.get("event_dates")):
                    if d not in dated:
                        plan.full_days.add(d)
                return plan
            return EventBlackoutPlan(
                full_days=set(_as_date_list(raw.get("dates") or raw.get("event_dates")))
            )
        # bare list → full-day legacy
        return EventBlackoutPlan(full_days=set(_as_date_list(raw)))
    dates: list[str] = []
    for line in text.splitlines():
        s = line.split("#", 1)[0].strip()
        if s:
            dates.append(s)
    return EventBlackoutPlan(full_days=set(dates))


def load_event_dates_file(path: str | Path | None) -> list[str]:
    """Load **full-day** dates from JSON / text (legacy helper)."""
    return sorted(load_event_plan_file(path).full_days)


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

    Returns ``(full_day_dates, meta)``.
    ``meta["active_today"]`` is True only for **full-day** macro blocks
    (OMS ``day_halted``). Symbol blocks are in ``meta["symbol_blackout"]`` /
    ``meta["active_today_symbols"]`` — Scanner skips those tickers only.
    """
    cfg = event_cfg_from_profile(profile)
    sources: list[str] = []
    plan = EventBlackoutPlan()

    enabled = cfg.get("event_calendar_block", False)
    if isinstance(enabled, str):
        enabled = enabled.strip().lower() in {"1", "true", "yes", "on"}
    if enabled:
        dates = event_dates_from_cfg(cfg)
        if not dates:
            dates = [e["date"] for e in DEFAULT_EVENTS_MAY_JUL_2026]
        plan.full_days |= set(dates)
        sources.append("profile")

    path = os.environ.get("MAG7_EVENT_CALENDAR_PATH") or cfg.get("event_calendar_path")
    file_plan = load_event_plan_file(path)
    if file_plan.full_days or file_plan.symbol_days:
        plan = plan.merge(file_plan)
        sources.append(f"file:{path}")

    redis_plan = load_event_plan_redis(redis_client)
    if redis_plan.full_days or redis_plan.symbol_days:
        plan = plan.merge(redis_plan)
        sources.append("redis")

    force_today = os.environ.get("MAG7_EVENT_BLACKOUT_TODAY", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if force_today:
        plan.full_days.add(str(trade_date))
        sources.append("env:MAG7_EVENT_BLACKOUT_TODAY")

    if not enabled and not plan.full_days and not plan.symbol_days:
        meta = {
            "enabled": False,
            "active_today": False,
            "active_today_full": False,
            "active_today_symbols": [],
            "sources": [],
            "event_dates": [],
            "blackout_dates": [],
            "symbol_blackout": {},
            "trade_date": trade_date,
        }
        return set(), meta

    sessions = list(session_dates) if session_dates is not None else []
    if not sessions:
        sessions = sorted(plan.full_days | {str(trade_date)} | set(plan.symbol_days))
    blackout_sessions = int(cfg.get("event_blackout_sessions", 0) or 0)
    if plan.full_days:
        plan.full_days = expand_blackout_dates(
            plan.full_days, session_dates=sessions, blackout_sessions=blackout_sessions
        )

    active_full = plan.blocks_day(trade_date)
    today_syms = sorted(plan.symbols_blocked_on(trade_date))
    meta = {
        "enabled": True,
        "active_today": active_full,  # OMS day_halt — full-day only
        "active_today_full": active_full,
        "active_today_symbols": today_syms,
        "sources": sources,
        "event_dates": sorted(plan.full_days),
        "blackout_dates": sorted(plan.full_days),
        "symbol_blackout": {
            d: sorted(s) for d, s in sorted(plan.symbol_days.items())
        },
        "event_blackout_sessions": blackout_sessions,
        "trade_date": trade_date,
        "event_plan": plan,
    }
    return set(plan.full_days), meta


def load_event_plan_redis(redis_client: Any, *, key: str | None = None) -> EventBlackoutPlan:
    """Redis: JSON list → full-day; JSON object may include symbol_blackout."""
    if redis_client is None:
        return EventBlackoutPlan()
    key = key or os.environ.get("MAG7_EVENT_BLACKOUT_REDIS_KEY", "maga7:event_blackout")
    try:
        raw = redis_client.get(key)
    except Exception:
        return EventBlackoutPlan()
    if raw is None:
        dates = load_event_dates_redis(redis_client, key=key)
        return EventBlackoutPlan(full_days=set(dates))
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8", errors="replace")
    text = str(raw).strip()
    if not text:
        return EventBlackoutPlan()
    if text[:1] == "{":
        try:
            obj = json.loads(text)
        except Exception:
            return EventBlackoutPlan()
        if isinstance(obj, dict):
            if "symbol_blackout" in obj or "events" in obj:
                # reuse file loader logic via temp-like dict
                plan = EventBlackoutPlan(full_days=set(_as_date_list(obj.get("dates"))))
                plan = plan.merge(plan_from_symbol_blackout(obj.get("symbol_blackout")))
                if isinstance(obj.get("events"), list):
                    plan = plan.merge(plan_from_events(obj["events"]))
                return plan
            return EventBlackoutPlan(full_days=set(_as_date_list(obj.get("dates"))))
    if text[:1] == "[":
        try:
            return EventBlackoutPlan(full_days=set(_as_date_list(json.loads(text))))
        except Exception:
            return EventBlackoutPlan()
    return EventBlackoutPlan(
        full_days={x.strip() for x in text.replace(";", ",").split(",") if x.strip()}
    )

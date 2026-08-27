"""Company-news ingest for Mag7 live event blackout (no LLM direction).

Sources:
  - Finnhub ``/company-news`` (free-tier friendly)
  - Investing.com Mag7-ish RSS (same feed as ``~/notebook/rss_feed/rss_feed_stable.py``)

Policy: see ``event_news_policy.py``. Ordinary headlines are scored for dash/LLM
audit only; default ``hard_risk`` auto-blocks CEO succession (symbol), never
uses news as a trade-direction oracle.
"""
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo

NY = ZoneInfo("America/New_York")
FINNHUB_NEWS_URL = "https://finnhub.io/api/v1/company-news"
DEFAULT_RSS_URL = "https://cn.investing.com/rss/news_356.rss"

# Mag7 (+GOOGL) aliases for EN/CN headlines.
SYMBOL_ALIASES: dict[str, tuple[str, ...]] = {
    "NVDA": ("nvda", "nvidia", "英伟达"),
    "TSLA": ("tsla", "tesla", "特斯拉"),
    "AAPL": ("aapl", "apple", "苹果公司", "苹果"),
    "AMZN": ("amzn", "amazon", "亚马逊"),
    "META": ("meta", "facebook", "脸书", "元宇宙平台"),
    "MSFT": ("msft", "microsoft", "微软"),
    "AMD": ("amd", "超威", "超微半导体"),
    "GOOGL": ("googl", "google", "alphabet", "谷歌", "Alphabet"),
}

# Risk-ish only — auto-block under news_mode=hard_risk / blackout (Mag7-matched).
HARD_RULES: list[tuple[str, list[str]]] = [
    (
        "ceo_succession",
        [
            r"\bceo\b.{0,48}(resign|retire|step(?:s|ping)?\s+down|succession)",
            r"(named|appoint(?:s|ed)?|names)\s+.{0,24}\bceo\b",
            r"(succession|successor).{0,24}\bceo\b",
            r"(卸任|继任|接任|任命|辞职|退休).{0,12}(CEO|首席执行官)",
            r"(CEO|首席执行官).{0,16}(卸任|辞职|退休|交接|继任|接任)",
        ],
    ),
]

# Bullish / mixed catalysts — audit for dash, never auto-blackout.
SOFT_RULES: list[tuple[str, list[str]]] = [
    (
        "mega_deal",
        [
            r"multiyear.{0,40}(deal|agreement|contract|order)",
            r"(multi[- ]year).{0,40}(deal|agreement|contract)",
            r"(签署|达成|敲定).{0,20}(多年|多年期).{0,24}(协议|合同|订单|采购)",
            r"\b(acquires?|acquisition|merger)\b.{0,50}\b("
            r"nvidia|apple|microsoft|meta|amazon|google|alphabet|amd|tesla|"
            r"nvda|aapl|msft|amzn|googl|tsla"
            r")\b",
            r"(收购|并购).{0,20}(英伟达|苹果|微软|亚马逊|谷歌|特斯拉|超威)",
        ],
    ),
    (
        "capex_shock",
        [
            r"(raises?|hikes?|boosts?|doubles?|surges?|jumps?).{0,24}\bcapex\b",
            r"\bcapex\b.{0,24}(surge|jump|soar|shock|blowout|double)",
            r"(raises?|hikes?|boosts?).{0,24}(capital expenditure)",
            r"(资本开支|资本支出).{0,20}(上调|大增|超预期|冲击|翻倍)",
        ],
    ),
    (
        "partnership",
        [
            r"\b(partnership|partners with|strategic (deal|alliance))\b",
            r"(合作|伙伴|联手|供应协议)",
        ],
    ),
    (
        "chip_fab",
        [
            r"\b(terafab|foundry|fab(ricat)?)\b",
            r"(晶圆厂|芯片厂|自研芯片)",
        ],
    ),
]

# Soft tags match title first (Finnhub summaries are noisy).
_SOFT_TITLE_ONLY = frozenset({"mega_deal", "capex_shock"})
# Only these hard tags become symbol blocks under hard_risk mode.
AUTO_BLACKOUT_TAGS = frozenset({"ceo_succession"})


def _http_get(url: str, *, timeout: float = 20.0, headers: dict[str, str] | None = None) -> bytes:
    req = urllib.request.Request(
        url,
        headers=headers
        or {
            "User-Agent": "maga7-company-news/1.0",
            "Accept": "application/json,application/rss+xml,application/xml,text/xml,*/*",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _finnhub_key() -> str | None:
    from maga7.common.event_providers import _finnhub_key as _key

    return _key()


def match_symbols(text: str, symbols: Iterable[str] | None = None) -> list[str]:
    """Return Mag7 tickers mentioned in title/summary (alias aware)."""
    blob = (text or "").lower()
    if not blob.strip():
        return []
    want = {str(s).upper() for s in (symbols or SYMBOL_ALIASES)}
    hits: list[str] = []
    for sym, aliases in SYMBOL_ALIASES.items():
        if sym not in want:
            continue
        # word-ish ticker OR any alias
        pats = [rf"\b{re.escape(sym.lower())}\b"] + [re.escape(a.lower()) for a in aliases]
        if any(re.search(p, blob) for p in pats):
            hits.append(sym)
    return hits


def classify_headline(
    text: str,
    *,
    title: str | None = None,
) -> tuple[str | None, str | None]:
    """Return ``(tier, tag)`` where tier is ``hard`` / ``soft`` / None.

    CEO rules may use title+summary; deal/capex match **title only**.
    """
    full = text or ""
    title_blob = title if title is not None else full
    for tag, pats in HARD_RULES:
        if any(re.search(p, full, flags=re.IGNORECASE) for p in pats):
            return "hard", tag
    for tag, pats in SOFT_RULES:
        blob = title_blob if tag in _SOFT_TITLE_ONLY else full
        if any(re.search(p, blob, flags=re.IGNORECASE) for p in pats):
            return "soft", tag
    return None, None


def published_to_session_date(dt: datetime) -> str:
    """Map publish time to the US equity session it contaminates."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    ny = dt.astimezone(NY)
    d = ny.date()
    # After regular close → next weekday session.
    if (ny.hour, ny.minute) >= (16, 0):
        d = d + timedelta(days=1)
    while d.weekday() >= 5:
        d += timedelta(days=1)
    return d.isoformat()


def _parse_dt(raw: Any) -> datetime | None:
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        # Finnhub unix seconds
        return datetime.fromtimestamp(float(raw), tz=timezone.utc)
    s = str(raw).strip()
    if not s:
        return None
    if s.isdigit():
        return datetime.fromtimestamp(float(s), tz=timezone.utc)
    try:
        dt = parsedate_to_datetime(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (TypeError, ValueError, IndexError):
        pass
    s2 = s.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(s2)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except ValueError:
        pass
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(s[: len(fmt) + 2], fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def fetch_company_news_finnhub(
    symbols: Iterable[str],
    *,
    start: str,
    end: str,
) -> tuple[list[dict[str, Any]], str]:
    key = _finnhub_key()
    if not key:
        return [], "finnhub_news(missing_key)"
    out: list[dict[str, Any]] = []
    errors = 0
    for sym in sorted({str(s).upper() for s in symbols}):
        q = urllib.parse.urlencode(
            {"symbol": sym, "from": start, "to": end, "token": key}
        )
        url = f"{FINNHUB_NEWS_URL}?{q}"
        try:
            raw = json.loads(_http_get(url).decode("utf-8", errors="replace"))
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError):
            errors += 1
            continue
        if not isinstance(raw, list):
            continue
        for r in raw:
            title = str(r.get("headline") or r.get("title") or "").strip()
            summary = str(r.get("summary") or "").strip()
            dt = _parse_dt(r.get("datetime") or r.get("date"))
            if not title or dt is None:
                continue
            out.append(
                {
                    "source": "finnhub_news",
                    "symbol": sym,
                    "title": title,
                    "summary": summary,
                    "url": str(r.get("url") or ""),
                    "published_at": dt.astimezone(timezone.utc).isoformat(),
                    "session_date": published_to_session_date(dt),
                }
            )
    label = "finnhub_news"
    if errors:
        label = f"finnhub_news(errors={errors})"
    return out, label


def fetch_company_news_rss(
    *,
    rss_url: str | None = None,
    symbols: Iterable[str] | None = None,
    lookback_days: int = 3,
) -> tuple[list[dict[str, Any]], str]:
    """One-shot fetch of Investing RSS (stdlib XML; no feedparser required)."""
    url = (
        rss_url
        or os.environ.get("MAG7_RSS_NEWS_URL")
        or DEFAULT_RSS_URL
    ).strip()
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
        ),
        "Accept": "application/rss+xml,application/xml,text/xml,*/*",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        "Referer": "https://cn.investing.com/",
    }
    try:
        body = _http_get(url, headers=headers)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return [], f"rss(error:{exc.__class__.__name__})"

    try:
        root = ET.fromstring(body)
    except ET.ParseError as exc:
        return [], f"rss(parse:{exc})"

    # RSS 2.0: channel/item ; Atom: entry
    items = root.findall(".//item")
    if not items:
        items = root.findall(".//{http://www.w3.org/2005/Atom}entry")

    cutoff = datetime.now(timezone.utc) - timedelta(days=max(1, int(lookback_days)))
    out: list[dict[str, Any]] = []
    for item in items:
        title_el = item.find("title")
        if title_el is None:
            title_el = item.find("{http://www.w3.org/2005/Atom}title")
        link_el = item.find("link")
        if link_el is None:
            link_el = item.find("{http://www.w3.org/2005/Atom}link")
        desc_el = item.find("description")
        if desc_el is None:
            desc_el = item.find("{http://www.w3.org/2005/Atom}summary")
        pub_el = item.find("pubDate")
        if pub_el is None:
            pub_el = item.find("{http://www.w3.org/2005/Atom}updated")
        if pub_el is None:
            pub_el = item.find("{http://www.w3.org/2005/Atom}published")

        title = (title_el.text or "").strip() if title_el is not None else ""
        if link_el is not None and link_el.get("href"):
            link = str(link_el.get("href"))
        else:
            link = (link_el.text or "").strip() if link_el is not None else ""
        summary = (desc_el.text or "").strip() if desc_el is not None else ""
        # strip crude HTML
        summary = re.sub(r"<[^>]+>", " ", summary)
        pub_raw = (pub_el.text or "").strip() if pub_el is not None else ""
        dt = _parse_dt(pub_raw)
        if not title or dt is None:
            continue
        if dt.astimezone(timezone.utc) < cutoff:
            continue
        text = f"{title} {summary}"
        hits = match_symbols(text, symbols)
        if symbols is not None and not hits:
            # keep unmatched in audit path via symbol=None? skip for RSS noise
            continue
        out.append(
            {
                "source": "investing_rss",
                "symbol": hits[0] if hits else None,
                "symbols": hits,
                "title": title,
                "summary": summary[:500],
                "url": link,
                "published_at": dt.astimezone(timezone.utc).isoformat(),
                "session_date": published_to_session_date(dt),
            }
        )
    return out, "investing_rss"


def score_news_items(
    items: Iterable[dict[str, Any]],
    *,
    symbols: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    """Attach tier/tag/symbols; Finnhub rows already have symbol."""
    scored: list[dict[str, Any]] = []
    for it in items:
        title = str(it.get("title") or "")
        summary = str(it.get("summary") or "")
        text = f"{title}\n{summary}"
        tier, tag = classify_headline(text, title=title)
        syms = list(it.get("symbols") or [])
        if it.get("symbol") and str(it["symbol"]).upper() not in syms:
            syms.insert(0, str(it["symbol"]).upper())
        if not syms:
            syms = match_symbols(text, symbols)
        row = {
            **it,
            "symbols": syms,
            "symbol": syms[0] if syms else it.get("symbol"),
            "tier": tier,
            "tag": tag,
        }
        scored.append(row)
    return scored


def news_to_blackout_events(
    scored: Iterable[dict[str, Any]],
    *,
    mode: str = "hard_risk",
    start: str | None = None,
    end: str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split optional auto-blackout events vs audit rows.

    ``mode`` (see ``event_news_policy.normalize_news_mode``):
      - ``audit``: score only — no news_* rows
      - ``hard_risk`` / ``blackout``: only ``AUTO_BLACKOUT_TAGS`` (CEO) → symbol events

    Never derives UP/DN trade direction from headlines.
    """
    from maga7.common.event_news_policy import news_mode_auto_blocks, normalize_news_mode

    mode_n = normalize_news_mode(mode)
    events: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    for row in scored:
        audit.append(row)
        if not news_mode_auto_blocks(mode_n):
            continue
        if row.get("tier") != "hard":
            continue
        if str(row.get("tag") or "") not in AUTO_BLACKOUT_TAGS:
            continue
        if not row.get("symbols"):
            continue
        d = str(row.get("session_date") or "")[:10]
        if not d:
            continue
        if start and d < start:
            continue
        if end and d > end:
            continue
        events.append(
            {
                "date": d,
                "tag": f"news_{row.get('tag') or 'hard'}",
                "source": str(row.get("source") or "company_news"),
                "symbol": row.get("symbol"),
                "note": str(row.get("title") or "")[:180],
                "policy": "hard_risk_symbol_only",
            }
        )
    return events, audit


def collect_company_news_events(
    symbols: Iterable[str],
    *,
    start: str,
    end: str,
    rss_url: str | None = None,
    news_mode: str = "hard_risk",
    enable_finnhub: bool = True,
    enable_rss: bool = True,
    rss_lookback_days: int = 3,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    """Fetch + score + map. Returns ``(events, audit, source_labels)``.

    Does not call LLM; LLM is dash-only and must not feed this path.
    """
    from maga7.common.event_news_policy import assert_llm_not_in_blackout_pipeline

    assert_llm_not_in_blackout_pipeline()
    items: list[dict[str, Any]] = []
    sources: list[str] = []
    syms = [str(s).upper() for s in symbols]

    if enable_finnhub:
        rows, label = fetch_company_news_finnhub(syms, start=start, end=end)
        items.extend(rows)
        sources.append(label)

    if enable_rss:
        rows, label = fetch_company_news_rss(
            rss_url=rss_url, symbols=syms, lookback_days=rss_lookback_days
        )
        items.extend(rows)
        sources.append(label)

    scored = score_news_items(items, symbols=syms)
    events, audit = news_to_blackout_events(
        scored, mode=news_mode, start=start, end=end
    )
    return events, audit, sources


def write_news_audit(path: str | os.PathLike[str], audit: list[dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    hard = [a for a in audit if a.get("tier") == "hard"]
    soft = [a for a in audit if a.get("tier") == "soft"]
    payload = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "n_total": len(audit),
        "n_hard": len(hard),
        "n_soft": len(soft),
        "items": audit,
    }
    p.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


DEFAULT_SUPPRESS_PATH = Path(__file__).resolve().parents[1] / "CONFIG" / "event_news_suppress.json"


def normalize_news_tag(tag: str) -> str:
    t = str(tag or "").strip()
    if not t or t.startswith("news_"):
        return t
    if t in {"ceo_succession", "mega_deal", "capex_shock", "partnership", "chip_fab"}:
        return f"news_{t}"
    return t


def suppress_key(row: dict[str, Any]) -> str:
    """Stable id for dash reject / sync filter."""
    url = str(row.get("url") or "").strip()
    title = str(row.get("title") or row.get("note") or "").strip()[:160]
    d = str(row.get("date") or row.get("session_date") or "")[:10]
    tag = normalize_news_tag(str(row.get("tag") or ""))
    sym = str(row.get("symbol") or "").upper()
    if url:
        return f"{d}|{sym}|{tag}|{url}"
    return f"{d}|{sym}|{tag}|{title}"


def load_news_suppress(path: str | os.PathLike[str] | None = None) -> list[dict[str, Any]]:
    p = Path(path) if path else DEFAULT_SUPPRESS_PATH
    if not p.is_file():
        return []
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    items = raw.get("items") if isinstance(raw, dict) else raw
    if not isinstance(items, list):
        return []
    return [x for x in items if isinstance(x, dict)]


def save_news_suppress(
    items: list[dict[str, Any]],
    path: str | os.PathLike[str] | None = None,
) -> Path:
    p = Path(path) if path else DEFAULT_SUPPRESS_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    # dedupe by key
    by_key: dict[str, dict[str, Any]] = {}
    for it in items:
        k = str(it.get("key") or suppress_key(it))
        row = {**it, "key": k}
        by_key[k] = row
    payload = {
        "description": "Dash-reviewed false-positive company-news blackouts (skipped on sync)",
        "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "items": sorted(by_key.values(), key=lambda x: str(x.get("key") or "")),
    }
    p.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return p


def filter_suppressed_events(
    events: list[dict[str, Any]],
    suppress: list[dict[str, Any]] | None = None,
    *,
    suppress_path: str | os.PathLike[str] | None = None,
) -> list[dict[str, Any]]:
    """Drop news_* event rows matching suppress keys / (date,symbol,tag)."""
    items = suppress if suppress is not None else load_news_suppress(suppress_path)
    if not items:
        return list(events)
    keys = {str(x.get("key") or suppress_key(x)) for x in items}
    urls = {str(x.get("url") or "").strip() for x in items if x.get("url")}
    triples = {
        (
            str(x.get("date") or x.get("session_date") or "")[:10],
            str(x.get("symbol") or "").upper(),
            normalize_news_tag(str(x.get("tag") or "")),
        )
        for x in items
    }
    out: list[dict[str, Any]] = []
    for e in events:
        tag = str(e.get("tag") or "")
        if not tag.startswith("news_"):
            out.append(e)
            continue
        k = suppress_key({**e, "title": e.get("note") or e.get("title")})
        trip = (
            str(e.get("date") or "")[:10],
            str(e.get("symbol") or "").upper(),
            normalize_news_tag(tag),
        )
        eurl = str(e.get("url") or "").strip()
        if k in keys or trip in triples or (eurl and eurl in urls):
            continue
        out.append(e)
    return out


def apply_suppress_to_calendar_payload(
    payload: dict[str, Any],
    suppress: list[dict[str, Any]] | None = None,
    *,
    suppress_path: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Rewrite live calendar JSON after dash rejects news hits."""
    from maga7.common.event_calendar import plan_from_events

    events = list(payload.get("events") or [])
    kept = filter_suppressed_events(events, suppress, suppress_path=suppress_path)
    plan = plan_from_events(kept)
    out = dict(payload)
    out["events"] = kept
    out["dates"] = sorted(plan.full_days)
    out["symbol_blackout"] = {
        d: sorted(syms) for d, syms in sorted(plan.symbol_days.items())
    }
    out["suppress_applied_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return out

"""LLM stance/impact analysis for Mag7 company-news audit (dash).

OpenAI-compatible Chat Completions (OpenAI / DeepSeek / Azure-style base URL).
Results are cached under ``CONFIG/event_news_llm.json`` for human review —
analysis does **not** auto-blackout (news ≠ ban).
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

DEFAULT_LLM_CACHE = Path(__file__).resolve().parents[1] / "CONFIG" / "event_news_llm.json"

SYSTEM_PROMPT = """你是美股 Mag7（NVDA/TSLA/AAPL/AMZN/META/MSFT/AMD/GOOGL）期权盘前风控分析助手。
根据标题与摘要，判断对该标的 **当日股价路径** 的含义，而不是公关口径。

务必考虑二阶效应，例如：
- 「涨价 / 提价」表面像利好定价权，也可能被解读为需求走弱或伤害销量 → 股价暴跌；
- 「多年大单 / 合作」多为叙事利好，但期权路径仍可能 whipsaw；
- 「CEO 交接 / 裁员 / 监管」偏风险事件。

只输出一个 JSON 对象，不要 Markdown。字段：
{
  "stance": "bullish|bearish|mixed|unclear",
  "impact": "high|medium|low",
  "surprise_risk": "high|medium|low",
  "trade_hint": "caution_long|caution_short|avoid|neutral",
  "rationale_zh": "中文 1-3 句，点明表面叙事 vs 可能的真实定价"
}
"""


def _read_key_file(path: Path) -> str | None:
    try:
        if not path.is_file():
            return None
        text = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    for line in text.splitlines():
        s = line.split("#", 1)[0].strip()
        if not s:
            continue
        if "=" in s and any(
            s.upper().startswith(p)
            for p in ("OPENAI", "DEEPSEEK", "API", "KEY", "LLM")
        ):
            s = s.split("=", 1)[1].strip().strip('"').strip("'")
        if s:
            return s
    return None


def resolve_llm_api_key() -> str | None:
    for k in (
        "MAG7_LLM_API_KEY",
        "OPENAI_API_KEY",
        "DEEPSEEK_API_KEY",
        "LLM_API_KEY",
    ):
        v = os.environ.get(k, "").strip()
        if v:
            return v
    env_path = os.environ.get("MAG7_LLM_KEY_FILE", "").strip()
    candidates: list[Path] = []
    if env_path:
        candidates.append(Path(env_path).expanduser())
    home = Path.home()
    candidates.extend(
        [
            home / "openai.txt",
            home / "deepseek.txt",
            home / ".config" / "maga7" / "openai.txt",
            home / ".config" / "maga7" / "llm.txt",
        ]
    )
    for p in candidates:
        key = _read_key_file(p)
        if key:
            return key
    return None


def resolve_llm_endpoint() -> tuple[str, str]:
    """Return ``(base_url_without_trailing_slash, model)``."""
    has_deepseek = bool(os.environ.get("DEEPSEEK_API_KEY", "").strip()) or bool(
        _read_key_file(Path.home() / "deepseek.txt")
    )
    default_base = (
        "https://api.deepseek.com/v1" if has_deepseek else "https://api.openai.com/v1"
    )
    default_model = "deepseek-chat" if has_deepseek else "gpt-4o-mini"
    base = (
        os.environ.get("MAG7_LLM_BASE_URL")
        or os.environ.get("OPENAI_BASE_URL")
        or os.environ.get("DEEPSEEK_BASE_URL")
        or default_base
    ).rstrip("/")
    model = (
        os.environ.get("MAG7_LLM_MODEL")
        or os.environ.get("OPENAI_MODEL")
        or os.environ.get("DEEPSEEK_MODEL")
        or default_model
    )
    if "deepseek" in base.lower() and model == "gpt-4o-mini":
        model = "deepseek-chat"
    return base, model


def load_llm_cache(path: str | Path | None = None) -> dict[str, Any]:
    p = Path(path) if path else DEFAULT_LLM_CACHE
    if not p.is_file():
        return {"items": {}, "updated_at": None}
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"items": {}, "updated_at": None}
    if not isinstance(raw, dict):
        return {"items": {}, "updated_at": None}
    items = raw.get("items")
    if not isinstance(items, dict):
        items = {}
    return {"items": items, "updated_at": raw.get("updated_at")}


def save_llm_cache(cache: dict[str, Any], path: str | Path | None = None) -> Path:
    p = Path(path) if path else DEFAULT_LLM_CACHE
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "description": "LLM stance/impact for Mag7 news audit (dash). Not auto-blackout.",
        "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "items": cache.get("items") or {},
    }
    p.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return p


def _chat_json(
    *,
    system: str,
    user: str,
    api_key: str,
    base_url: str,
    model: str,
    timeout: float = 60.0,
) -> dict[str, Any]:
    url = f"{base_url}/chat/completions"
    body = {
        "model": model,
        "temperature": 0.2,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "maga7-news-llm/1.0",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = json.loads(resp.read().decode("utf-8", errors="replace"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:400]
        raise RuntimeError(f"LLM HTTP {exc.code}: {detail}") from exc
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"LLM request failed: {exc}") from exc

    content = (
        ((raw.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
    ).strip()
    if content.startswith("```"):
        content = content.strip("`")
        if content.lower().startswith("json"):
            content = content[4:].strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"LLM returned non-JSON: {content[:300]}") from exc


def analyze_headline(
    item: dict[str, Any],
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    key = api_key or resolve_llm_api_key()
    if not key:
        raise RuntimeError(
            "缺少 LLM key：设置 OPENAI_API_KEY / DEEPSEEK_API_KEY / MAG7_LLM_API_KEY，"
            "或写入 ~/openai.txt / ~/deepseek.txt"
        )
    b, m = resolve_llm_endpoint()
    if base_url:
        b = base_url.rstrip("/")
    if model:
        m = model
    sym = str(item.get("symbol") or "").upper() or "?"
    title = str(item.get("title") or "").strip()
    summary = str(item.get("summary") or "").strip()[:800]
    user = (
        f"标的: {sym}\n"
        f"会话日: {item.get('session_date') or item.get('date') or ''}\n"
        f"关键词标签: {item.get('tag') or ''} / tier={item.get('tier') or ''}\n"
        f"标题: {title}\n"
        f"摘要: {summary or '(无)'}\n"
        f"链接: {item.get('url') or ''}\n"
    )
    parsed = _chat_json(system=SYSTEM_PROMPT, user=user, api_key=key, base_url=b, model=m)
    stance = str(parsed.get("stance") or "unclear").lower()
    if stance not in {"bullish", "bearish", "mixed", "unclear"}:
        stance = "unclear"
    impact = str(parsed.get("impact") or "medium").lower()
    if impact not in {"high", "medium", "low"}:
        impact = "medium"
    surprise = str(parsed.get("surprise_risk") or "medium").lower()
    if surprise not in {"high", "medium", "low"}:
        surprise = "medium"
    hint = str(parsed.get("trade_hint") or "neutral").lower()
    if hint not in {"caution_long", "caution_short", "avoid", "neutral"}:
        hint = "neutral"
    return {
        "stance": stance,
        "impact": impact,
        "surprise_risk": surprise,
        "trade_hint": hint,
        "rationale_zh": str(parsed.get("rationale_zh") or "")[:500],
        "model": m,
        "analyzed_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "symbol": sym,
        "title": title[:200],
        "session_date": str(item.get("session_date") or item.get("date") or "")[:10],
        "url": str(item.get("url") or ""),
    }


def analyze_batch(
    items: Iterable[dict[str, Any]],
    *,
    cache_path: str | Path | None = None,
    max_n: int = 20,
    skip_cached: bool = True,
    key_fn: Any = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Analyze up to ``max_n`` items; merge into cache. Returns (new_rows, cache)."""
    from maga7.common.company_news import suppress_key

    kn = key_fn or (
        lambda it: suppress_key(
            {
                **it,
                "date": it.get("session_date") or it.get("date"),
                "tag": it.get("tag"),
            }
        )
    )
    cache = load_llm_cache(cache_path)
    items_map: dict[str, Any] = dict(cache.get("items") or {})
    out: list[dict[str, Any]] = []
    n = 0
    for it in items:
        if n >= max_n:
            break
        k = kn(it)
        if skip_cached and k in items_map:
            continue
        try:
            result = analyze_headline(it)
            result["key"] = k
            items_map[k] = result
            out.append(result)
            n += 1
        except Exception as exc:  # keep batch going
            err = {
                "key": k,
                "error": str(exc)[:300],
                "title": str(it.get("title") or "")[:160],
                "symbol": it.get("symbol"),
                "analyzed_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
            items_map[k] = {**items_map.get(k, {}), **err}
            out.append(err)
            n += 1
    cache = {"items": items_map, "updated_at": None}
    save_llm_cache(cache, cache_path)
    return out, load_llm_cache(cache_path)

"""Mag7 company-news / calendar policy (research → live).

Agreed rules from loss-day news scan (Feb–Jul remaining ≤−5% days):

1. **Do not use ordinary company news as a direction oracle.**
   Meta×NVDA / Terafab / Corning style headlines are often bullish narratives
   that still tox/fade; AMD hike / AAPL CEO show *wrong-side* risk, not alpha.

2. **Hard calendar only** (auto block):
   - full-day: FOMC / NFP / CPI / Mag7-wide macro (preset + sync FOMC)
   - symbol: earnings_* from calendar API
   - symbol: narrow hard-risk headlines (CEO succession) when news mode allows

3. **Ordinary headlines** (deals / partnership / capex / fab):
   - score + dash + optional LLM stance/impact for humans
   - **never** auto-blackout, **never** set trade direction

4. Path losses after that stay with tox / size / time stops — not news direction.
"""
from __future__ import annotations

from typing import Any

# Modes for sync / live company-news ingest.
# hard_risk (== blackout): score all; auto symbol-block only AUTO_BLACKOUT_TAGS
# audit: score all; no news_* auto rows
NEWS_MODE_AUDIT = "audit"
NEWS_MODE_HARD_RISK = "hard_risk"
NEWS_MODE_ALIASES = {
    "audit": NEWS_MODE_AUDIT,
    "score": NEWS_MODE_AUDIT,
    "off_auto": NEWS_MODE_AUDIT,
    "hard_risk": NEWS_MODE_HARD_RISK,
    "hard": NEWS_MODE_HARD_RISK,
    "blackout": NEWS_MODE_HARD_RISK,  # legacy name
    "ceo_only": NEWS_MODE_HARD_RISK,
}

DEFAULT_NEWS_MODE = NEWS_MODE_HARD_RISK

POLICY_SUMMARY_ZH = (
    "公司新闻不定方向："
    "宏观/财报/CEO 交接等硬风险才自动禁入；"
    "大单/合作/capex 只审计+LLM 提示；"
    "路径亏损靠 tox/仓位，不靠新闻猜涨跌。"
)


def normalize_news_mode(raw: Any) -> str:
    s = str(raw or DEFAULT_NEWS_MODE).strip().lower()
    if s in {"", "none", "default"}:
        return DEFAULT_NEWS_MODE
    return NEWS_MODE_ALIASES.get(s, NEWS_MODE_AUDIT if s not in NEWS_MODE_ALIASES else s)


def news_mode_auto_blocks(mode: str) -> bool:
    """True if mode may emit news_* symbol blackout rows (CEO only)."""
    return normalize_news_mode(mode) == NEWS_MODE_HARD_RISK


def policy_from_profile(profile: dict[str, Any] | None) -> dict[str, Any]:
    """Read regime/trade news policy knobs (direction always false)."""
    profile = profile or {}
    reg = profile.get("regime") or {}
    trade = profile.get("trade") or {}
    mode = normalize_news_mode(
        reg.get("company_news_mode")
        or trade.get("company_news_mode")
        or DEFAULT_NEWS_MODE
    )
    return {
        "company_news_mode": mode,
        "company_news_auto_block": news_mode_auto_blocks(mode),
        # Direction from headlines is unsupported — always off.
        "company_news_direction_from_news": False,
        # LLM stance/impact is dash-only; never drives calendar rows.
        "company_news_use_llm_for_blackout": False,
        "summary_zh": POLICY_SUMMARY_ZH,
    }


def assert_llm_not_in_blackout_pipeline() -> None:
    """Documentation hook: LLM cache must not feed sync blackout builders."""
    # Intentionally no import of news_llm here — sync must not depend on it.
    return None

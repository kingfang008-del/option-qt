from maga7.common.company_news import news_to_blackout_events, score_news_items
from maga7.common.event_news_policy import (
    DEFAULT_NEWS_MODE,
    normalize_news_mode,
    news_mode_auto_blocks,
    policy_from_profile,
)


def test_normalize_aliases():
    assert normalize_news_mode("blackout") == "hard_risk"
    assert normalize_news_mode("hard_risk") == "hard_risk"
    assert normalize_news_mode("audit") == "audit"
    assert DEFAULT_NEWS_MODE == "hard_risk"
    assert news_mode_auto_blocks("hard_risk")
    assert not news_mode_auto_blocks("audit")


def test_policy_never_direction_or_llm_blackout():
    p = policy_from_profile(
        {
            "regime": {
                "company_news_mode": "hard_risk",
                "company_news_direction_from_news": True,
                "company_news_use_llm_for_blackout": True,
            }
        }
    )
    assert p["company_news_direction_from_news"] is False
    assert p["company_news_use_llm_for_blackout"] is False
    assert p["company_news_mode"] == "hard_risk"


def test_hard_risk_blocks_ceo_not_deal():
    scored = score_news_items(
        [
            {
                "symbol": "AAPL",
                "title": "Apple CEO Tim Cook to resign; successor named",
                "summary": "",
                "session_date": "2026-04-22",
            },
            {
                "symbol": "NVDA",
                "title": "Meta and Nvidia multiyear deal for Blackwell GPUs",
                "summary": "",
                "session_date": "2026-02-18",
            },
        ]
    )
    ev, _ = news_to_blackout_events(scored, mode="hard_risk")
    assert len(ev) == 1
    assert ev[0]["symbol"] == "AAPL"
    assert "ceo" in ev[0]["tag"]
    assert news_to_blackout_events(scored, mode="audit")[0] == []

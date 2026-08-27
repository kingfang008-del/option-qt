from datetime import datetime, timezone

from maga7.common.company_news import (
    classify_headline,
    match_symbols,
    news_to_blackout_events,
    published_to_session_date,
    score_news_items,
)


def test_match_symbols_en_cn():
    assert "NVDA" in match_symbols("Nvidia signs multiyear GPU deal with Meta")
    assert "AAPL" in match_symbols("苹果公司宣布CEO交接")
    assert match_symbols("原油大涨地缘紧张") == []


def test_hard_ceo_deal_is_soft():
    tier, tag = classify_headline("Apple CEO Tim Cook to resign; John Ternus named CEO")
    assert tier == "hard" and tag == "ceo_succession"
    # Bullish multiyear deals must NOT auto-ban
    tier, tag = classify_headline("Meta and Nvidia multiyear deal for Blackwell GPUs")
    assert tier == "soft" and tag == "mega_deal"


def test_soft_not_hard():
    tier, tag = classify_headline("Nvidia partnership with Corning on optical cable")
    assert tier == "soft" and tag == "partnership"


def test_session_date_after_close():
    # 2026-04-21 20:00 ET → next session 04-22
    dt = datetime(2026, 4, 22, 0, 0, tzinfo=timezone.utc)  # 20:00 ET prior day-ish
    # Explicit NY evening: 2026-04-21 17:00 ET = 21:00 UTC
    dt = datetime(2026, 4, 21, 21, 0, tzinfo=timezone.utc)
    assert published_to_session_date(dt) == "2026-04-22"


def test_suppress_filters_news_events(tmp_path):
    from maga7.common.company_news import (
        filter_suppressed_events,
        save_news_suppress,
        suppress_key,
    )

    events = [
        {
            "date": "2026-04-22",
            "tag": "news_ceo_succession",
            "source": "finnhub_news",
            "symbol": "AAPL",
            "note": "Apple CEO resigns",
            "url": "https://example.com/aapl-ceo",
        },
        {
            "date": "2026-07-22",
            "tag": "earnings_ah",
            "source": "finnhub",
            "symbol": "GOOGL",
        },
    ]
    save_news_suppress(
        [
            {
                "date": "2026-04-22",
                "symbol": "AAPL",
                "tag": "news_ceo_succession",
                "url": "https://example.com/aapl-ceo",
                "title": "Apple CEO resigns",
            }
        ],
        tmp_path / "suppress.json",
    )
    kept = filter_suppressed_events(events, suppress_path=tmp_path / "suppress.json")
    assert len(kept) == 1 and kept[0]["tag"] == "earnings_ah"
    assert "ceo" in suppress_key(events[0])


def test_news_to_blackout_requires_hard_and_symbol():
    scored = score_news_items(
        [
            {
                "source": "t",
                "symbol": "AAPL",
                "title": "Apple CEO Tim Cook to resign; successor named",
                "summary": "",
                "session_date": "2026-04-22",
            },
            {
                "source": "t",
                "symbol": "NVDA",
                "title": "Nvidia partnership expands optical supply",
                "summary": "",
                "session_date": "2026-05-06",
            },
        ]
    )
    events, audit = news_to_blackout_events(scored, mode="blackout")
    assert len(audit) == 2
    assert len(events) == 1
    assert events[0]["date"] == "2026-04-22"
    assert events[0]["symbol"] == "AAPL"

    events_audit, _ = news_to_blackout_events(scored, mode="audit")
    assert events_audit == []

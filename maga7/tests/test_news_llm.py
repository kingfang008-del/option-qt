from maga7.common.news_llm import analyze_batch, load_llm_cache, save_llm_cache


def test_llm_cache_roundtrip(tmp_path):
    p = tmp_path / "llm.json"
    save_llm_cache(
        {
            "items": {
                "k1": {
                    "stance": "bearish",
                    "impact": "high",
                    "surprise_risk": "high",
                    "trade_hint": "caution_long",
                    "rationale_zh": "提价可能被解读为需求走弱",
                }
            }
        },
        p,
    )
    c = load_llm_cache(p)
    assert c["items"]["k1"]["stance"] == "bearish"


def test_analyze_batch_uses_mock(monkeypatch, tmp_path):
    from maga7.common import news_llm as mod

    def fake_analyze(item, **kwargs):
        return {
            "stance": "bearish",
            "impact": "high",
            "surprise_risk": "high",
            "trade_hint": "caution_long",
            "rationale_zh": "表面涨价，二阶偏空",
            "model": "mock",
            "analyzed_at": "2026-07-20T00:00:00Z",
            "symbol": item.get("symbol"),
            "title": item.get("title"),
            "session_date": item.get("session_date"),
            "url": "",
        }

    monkeypatch.setattr(mod, "analyze_headline", fake_analyze)
    items = [
        {
            "symbol": "AAPL",
            "title": "Apple raises iPhone prices ahead of launch",
            "summary": "",
            "session_date": "2026-04-06",
            "tag": "mega_deal",
            "tier": "soft",
        }
    ]
    out, cache = analyze_batch(
        items, cache_path=tmp_path / "llm.json", max_n=5, skip_cached=False
    )
    assert len(out) == 1
    assert out[0]["stance"] == "bearish"
    assert len(cache["items"]) == 1

from maga7.common.event_providers import (
    build_sync_payload,
    earnings_to_blackout_dates,
    fomc_events_from_builtin,
    merge_event_rows,
)


def test_fomc_builtin_covers_jun2026():
    rows = fomc_events_from_builtin(
        start="2026-05-01",
        end="2026-07-31",
        include_meeting_day=True,
        include_decision_day=True,
    )
    dates = {r["date"] for r in rows}
    assert "2026-06-16" in dates
    assert "2026-06-17" in dates
    tags = {(r["date"], r["tag"]) for r in rows}
    assert ("2026-06-17", "fomc_decision") in tags


def test_earnings_ah_mapping():
    mapped = earnings_to_blackout_dates(
        [{"date": "2026-05-20", "tag": "earnings_ah", "source": "finnhub", "symbol": "NVDA"}],
        ah_include_session=True,
        ah_include_next_cal_day=True,
    )
    assert {m["date"] for m in mapped} == {"2026-05-20", "2026-05-21"}


def test_merge_and_payload():
    rows = merge_event_rows(
        [
            {"date": "2026-06-17", "tag": "fomc_decision", "source": "a", "symbol": None},
            {"date": "2026-06-17", "tag": "fomc_decision", "source": "b", "symbol": None},
            {"date": "2026-05-20", "tag": "earnings_ah", "source": "finnhub", "symbol": "NVDA"},
        ]
    )
    assert len(rows) == 2
    payload = build_sync_payload(rows, start="2026-05-01", end="2026-07-31", sources=["t"])
    # Macro (FOMC) stays in full-day dates; symbol earnings go to symbol_blackout.
    assert payload["dates"] == ["2026-06-17"]
    assert payload["symbol_blackout"] == {"2026-05-20": ["NVDA"]}

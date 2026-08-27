import json
from pathlib import Path

from maga7.common.event_calendar import (
    AAPL_CEO_EVENTS_2026,
    CORE_EVENTS_MAY_JUL_2026,
    EXTENDED_EVENTS_FEB_JUL_2026,
    event_dates_from_cfg,
    expand_blackout_dates,
    load_event_dates_file,
    load_event_plan_file,
    plan_from_events,
    resolve_event_blackout,
    resolve_live_event_blackout,
)


def test_expand_plus_one_session():
    sessions = ["2026-06-12", "2026-06-13", "2026-06-16", "2026-06-17"]
    out = expand_blackout_dates(["2026-06-12"], session_dates=sessions, blackout_sessions=1)
    assert out == {"2026-06-12", "2026-06-13"}


def test_core_preset():
    cfg = {"event_calendar_block": True, "event_calendar": "core"}
    out = resolve_event_blackout(
        cfg, session_dates=["2026-05-20", "2026-06-12", "2026-06-17"]
    )
    assert out == {e["date"] for e in CORE_EVENTS_MAY_JUL_2026}


def test_feb_jul_aapl_ceo_preset():
    dates = event_dates_from_cfg({"event_calendar": "feb_jul_aapl_ceo"})
    base = {e["date"] for e in EXTENDED_EVENTS_FEB_JUL_2026}
    ceo = {e["date"] for e in AAPL_CEO_EVENTS_2026}
    assert set(dates) == base | ceo
    assert "2026-04-21" in dates and "2026-04-22" in dates


def test_enabled_default_list():
    cfg = {"event_calendar_block": True}
    out = resolve_event_blackout(cfg, session_dates=["2026-05-20", "2026-06-17"])
    assert "2026-05-20" in out and "2026-06-17" in out


def test_disabled_empty():
    assert resolve_event_blackout(
        {"event_calendar": "core"}, session_dates=["2026-05-20"]
    ) == set()


def test_load_event_dates_file(tmp_path: Path):
    p = tmp_path / "cal.json"
    p.write_text(json.dumps({"dates": ["2026-06-12", "2026-06-17"]}), encoding="utf-8")
    assert load_event_dates_file(p) == ["2026-06-12", "2026-06-17"]


def test_live_file_activates_today(tmp_path: Path, monkeypatch):
    p = tmp_path / "cal.json"
    p.write_text(json.dumps({"dates": ["2026-07-17"]}), encoding="utf-8")
    monkeypatch.setenv("MAG7_EVENT_CALENDAR_PATH", str(p))
    blackout, meta = resolve_live_event_blackout({}, trade_date="2026-07-17")
    assert "2026-07-17" in blackout
    assert meta["active_today"] is True
    assert any(s.startswith("file:") for s in meta["sources"])


def test_plan_splits_news_vs_fomc():
    plan = plan_from_events(
        [
            {"date": "2026-06-17", "tag": "fomc_decision", "symbol": None},
            {
                "date": "2026-04-22",
                "tag": "news_ceo_succession",
                "symbol": "AAPL",
            },
            {"date": "2026-07-22", "tag": "earnings_ah", "symbol": "GOOGL"},
        ]
    )
    assert plan.full_days == {"2026-06-17"}
    assert plan.symbol_days["2026-04-22"] == {"AAPL"}
    assert plan.symbol_days["2026-07-22"] == {"GOOGL"}
    assert plan.blocks_symbol("2026-04-22", "AAPL")
    assert not plan.blocks_symbol("2026-04-22", "NVDA")
    assert plan.blocks_symbol("2026-06-17", "NVDA")  # full day


def test_live_file_jul22_tsla_googl_earnings_ah():
    """Patched Mag7 AH earnings must be symbol-scoped (not full-day halt)."""
    root = Path(__file__).resolve().parents[1]
    p = root / "CONFIG" / "event_calendar_live.json"
    plan = load_event_plan_file(p)
    assert plan.symbol_days.get("2026-07-22") == {"GOOGL", "TSLA"}
    assert not plan.blocks_day("2026-07-22")
    assert plan.blocks_symbol("2026-07-22", "TSLA")
    assert plan.blocks_symbol("2026-07-22", "GOOGL")
    assert not plan.blocks_symbol("2026-07-22", "NVDA")
    # Next session after AH gap is NOT auto-blacked (straddle / +1 still research).
    assert "TSLA" not in (plan.symbol_days.get("2026-07-23") or set())


def test_peer3_offline_symbol_blackout_jul22():
    root = Path(__file__).resolve().parents[1]
    prof = json.loads(
        (root / "CONFIG" / "strategy_profiles" /
         "single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1.json").read_text(
            encoding="utf-8"
        )
    )
    from maga7.common.event_calendar import event_cfg_from_profile, resolve_event_blackout_plan

    cfg = event_cfg_from_profile(prof)
    plan = resolve_event_blackout_plan(
        cfg, session_dates=["2026-07-22", "2026-07-23", "2026-07-24"]
    )
    assert plan.blocks_symbol("2026-07-22", "TSLA")
    assert plan.blocks_symbol("2026-07-22", "GOOGL")
    assert not plan.blocks_day("2026-07-22")


def test_live_symbol_does_not_day_halt(tmp_path: Path, monkeypatch):
    p = tmp_path / "cal.json"
    p.write_text(
        json.dumps(
            {
                "dates": [],
                "symbol_blackout": {"2026-07-20": ["AAPL"]},
                "events": [
                    {
                        "date": "2026-07-20",
                        "tag": "news_ceo_succession",
                        "symbol": "AAPL",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("MAG7_EVENT_CALENDAR_PATH", str(p))
    blackout, meta = resolve_live_event_blackout({}, trade_date="2026-07-20")
    assert blackout == set()
    assert meta["active_today"] is False
    assert meta["active_today_symbols"] == ["AAPL"]
    plan = load_event_plan_file(p)
    assert plan.blocks_symbol("2026-07-20", "AAPL")
    assert not plan.blocks_day("2026-07-20")

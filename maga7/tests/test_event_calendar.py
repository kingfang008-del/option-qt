import json
from pathlib import Path

from maga7.common.event_calendar import (
    CORE_EVENTS_MAY_JUL_2026,
    expand_blackout_dates,
    load_event_dates_file,
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

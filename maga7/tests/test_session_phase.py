from __future__ import annotations

import pandas as pd

from maga7.live.session_phase import (
    session_phase,
    tape_phase_dir,
    tape_symbol_path,
)


def test_session_phase_boundaries():
    day = "2026-07-16"
    assert (
        session_phase(
            pd.Timestamp("2026-07-16 09:29:59", tz="America/New_York"),
            trade_date=day,
        )
        == "PRE"
    )
    assert (
        session_phase(
            pd.Timestamp("2026-07-16 09:30:00", tz="America/New_York"),
            trade_date=day,
        )
        == "RTH"
    )
    assert (
        session_phase(
            pd.Timestamp("2026-07-16 15:59:59", tz="America/New_York"),
            trade_date=day,
        )
        == "RTH"
    )
    assert (
        session_phase(
            pd.Timestamp("2026-07-16 16:00:00", tz="America/New_York"),
            trade_date=day,
        )
        == "POST"
    )


def test_tape_paths(tmp_path):
    assert tape_phase_dir(tmp_path, "PRE") == tmp_path / "tape" / "pre"
    assert tape_symbol_path(
        tmp_path, phase="RTH", symbol="aapl", trade_date="2026-07-16"
    ) == tmp_path / "tape" / "rth" / "AAPL_2026-07-16.jsonl"


def test_session_phase_accepts_unix_seconds():
    ts = pd.Timestamp("2026-07-16 09:15:00", tz="America/New_York").timestamp()
    assert session_phase(ts, trade_date="2026-07-16") == "PRE"
    ts = pd.Timestamp("2026-07-16 10:30:00", tz="America/New_York").timestamp()
    assert session_phase(int(ts), trade_date="2026-07-16") == "RTH"

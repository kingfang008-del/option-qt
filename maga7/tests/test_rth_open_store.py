from pathlib import Path

from maga7.common.signals import StreamSignalState
from maga7.live.rth_open_store import (
    load_rth_opens,
    resolve_rth_opens,
    save_rth_opens,
    seed_scanner_day_opens,
    upsert_rth_open,
)


def test_rth_opens_persist_across_sessions(tmp_path: Path):
    trade_date = "2026-07-28"
    save_rth_opens(tmp_path, trade_date, {"AMD": 451.0}, source="test")
    upsert_rth_open(tmp_path, trade_date, "TSLA", 303.0, source="test")
    # Do not overwrite an already recorded open.
    upsert_rth_open(tmp_path, trade_date, "AMD", 999.0, source="late")
    opens = load_rth_opens(tmp_path, trade_date)
    assert opens == {"AMD": 451.0, "TSLA": 303.0}


def test_stream_state_does_not_latch_mid_session_first_bar():
    state = StreamSignalState("AMD", {"mf_window": 10, "vol_ma_window": 20}, emit_all=True)
    bar = {
        "timestamp": "2026-07-28T09:59:00-04:00",
        "open": 451.91,
        "high": 452.0,
        "low": 451.0,
        "close": 451.5,
        "volume": 100.0,
    }
    assert state.on_bar(bar) is None
    assert state.day_open is None


def test_stream_state_latches_true_rth_open():
    state = StreamSignalState("AMD", {"mf_window": 10, "vol_ma_window": 20}, emit_all=True)
    bar = {
        "timestamp": "2026-07-28T09:30:00-04:00",
        "open": 450.0,
        "high": 451.0,
        "low": 449.0,
        "close": 450.5,
        "volume": 100.0,
    }
    assert state.on_bar(bar) is None
    assert state.day_open == 450.0


def test_seed_scanner_day_opens_fills_missing(tmp_path: Path):
    class _State:
        day_open = None

    class _Scout:
        def __init__(self):
            self.seen = {}

        def seed_day_open(self, symbol, px, *, force=False):
            self.seen[symbol] = px

    class _Scanner:
        def __init__(self):
            self.states = {"AMD": _State()}
            self._am_pulse_scout = _Scout()
            self._pending_rth_opens = {}

    scanner = _Scanner()
    seeded = seed_scanner_day_opens(scanner, {"AMD": 450.25})
    assert seeded == ["AMD"]
    assert scanner.states["AMD"].day_open == 450.25
    assert scanner._am_pulse_scout.seen["AMD"] == 450.25
    assert scanner._pending_rth_opens["AMD"] == 450.25


def test_seed_scanner_day_opens_force_overwrites_pseudo(tmp_path: Path):
    class _State:
        day_open = 451.91  # late pseudo open

    class _Scout:
        def __init__(self):
            self.seen = {}
            self.force_flags = {}

        def seed_day_open(self, symbol, px, *, force=False):
            self.seen[symbol] = px
            self.force_flags[symbol] = force

    class _Scanner:
        def __init__(self):
            self.states = {"AMD": _State()}
            self._am_pulse_scout = _Scout()
            self._pending_rth_opens = {}

    scanner = _Scanner()
    seeded = seed_scanner_day_opens(scanner, {"AMD": 456.55}, force=True)
    assert seeded == ["AMD"]
    assert scanner.states["AMD"].day_open == 456.55
    assert scanner._am_pulse_scout.seen["AMD"] == 456.55
    assert scanner._am_pulse_scout.force_flags["AMD"] is True


def test_missing_rth_open_symbols():
    from maga7.live.rth_open_store import missing_rth_open_symbols

    assert missing_rth_open_symbols({"AMD": 450.0}, ["AMD", "TSLA", "QQQ"]) == [
        "TSLA",
        "QQQ",
    ]


def test_resolve_merges_tape_recovery(tmp_path: Path):
    trade_date = "2026-07-28"
    session = tmp_path / trade_date / "live_20260728_demo"
    tape = session / "tape" / "rth"
    tape.mkdir(parents=True)
    # 2026-07-28 09:30:00 America/New_York
    ts = 1785245400.0
    (tape / f"AMD_{trade_date}.jsonl").write_text(
        '{"ts": %s, "symbol": "AMD", "open": 450.5, "close": 450.6}\n' % ts,
        encoding="utf-8",
    )
    opens = resolve_rth_opens(tmp_path, trade_date, symbols=["AMD"], recover_tapes=True)
    assert opens["AMD"] == 450.5
    assert load_rth_opens(tmp_path, trade_date)["AMD"] == 450.5


def test_resolve_tape_iso_utc_converts_to_ny(tmp_path: Path):
    trade_date = "2026-07-28"
    session = tmp_path / trade_date / "live_20260728_iso"
    tape = session / "tape" / "rth"
    tape.mkdir(parents=True)
    # 09:30 ET = 13:30 UTC
    (tape / f"TSLA_{trade_date}.jsonl").write_text(
        '{"timestamp": "2026-07-28T13:30:00+00:00", "symbol": "TSLA", '
        '"open": 303.5, "close": 303.6}\n',
        encoding="utf-8",
    )
    opens = resolve_rth_opens(tmp_path, trade_date, symbols=["TSLA"], recover_tapes=True)
    assert opens["TSLA"] == 303.5


def test_resolve_disk_beats_tape(tmp_path: Path):
    trade_date = "2026-07-28"
    save_rth_opens(tmp_path, trade_date, {"AMD": 450.0}, source="official")
    session = tmp_path / trade_date / "live_20260728_bad"
    tape = session / "tape" / "rth"
    tape.mkdir(parents=True)
    ts = 1785245400.0
    (tape / f"AMD_{trade_date}.jsonl").write_text(
        '{"ts": %s, "symbol": "AMD", "open": 999.0, "close": 999.0}\n' % ts,
        encoding="utf-8",
    )
    opens = resolve_rth_opens(tmp_path, trade_date, symbols=["AMD"], recover_tapes=True)
    assert opens["AMD"] == 450.0


def test_seed_after_restore_overwrites_snapshot_pseudo():
    """Mirrors resume order: restore day_open then force seed durable opens."""

    class _State:
        def __init__(self):
            self.day_open = 451.91

    class _Scout:
        def __init__(self):
            self._day_open = {"AMD": 451.91}

        def seed_day_open(self, symbol, px, *, force=False):
            if force or symbol not in self._day_open:
                self._day_open[symbol] = px

        def snapshot_state(self):
            return {"date": "2026-07-28", "day_open": dict(self._day_open)}

        def restore_state(self, payload):
            self._day_open = {
                str(k).upper(): float(v)
                for k, v in (payload.get("day_open") or {}).items()
            }

    class _Scanner:
        def __init__(self, day_open: float):
            self.states = {"AMD": _State()}
            self.states["AMD"].day_open = day_open
            self._am_pulse_scout = _Scout()
            self._pending_rth_opens = {}

    # Snapshot had a late pseudo open.
    snapshot_scanner = _Scanner(451.91)
    restored = _Scanner(0.0)
    restored.states["AMD"].day_open = snapshot_scanner.states["AMD"].day_open
    restored._am_pulse_scout.restore_state(
        snapshot_scanner._am_pulse_scout.snapshot_state()
    )
    assert restored.states["AMD"].day_open == 451.91
    seeded = seed_scanner_day_opens(restored, {"AMD": 456.55}, force=True)
    assert seeded == ["AMD"]
    assert restored.states["AMD"].day_open == 456.55
    assert restored._am_pulse_scout._day_open["AMD"] == 456.55

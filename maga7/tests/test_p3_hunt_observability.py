from __future__ import annotations

from types import SimpleNamespace

from maga7.live.broker_oms import Mag7BrokerOms
from maga7.live.scanner import ScannerSignal
from maga7.live.scanner_state import scanner_snapshot


def test_scanner_snapshot_includes_watchdog_hunt_block():
    sc = SimpleNamespace(
        current_date="2026-07-02",
        states={},
        day_fires=[],
        signals=[],
        n_done={},
        last_exit={},
        last_win={},
        regime_gate=None,
        minute_agg=None,
        watchdog=SimpleNamespace(
            hunt_armed=True,
            hunt_candidates=[],
            hunt_budget_remaining=lambda: 1,
        ),
        _watchdog_state="normal",
        _watchdog_reason="ok",
        _watchdog_route="baseline",
        _watchdog_closed=True,
        _day_halt=False,
        n_hunt_signals=1,
        n_hunt_emitted=0,
        n_hunt_budget_skip=0,
        n_hunt_mutex_skip=0,
        pending_hunts=[{"symbol": "AMD", "dir": "UP", "entry_ts": "x", "detector": "washout_reclaim"}],
        day_hunt_symbols={"AMD"},
        stock_by={},
    )
    payload = scanner_snapshot(sc)
    assert "watchdog" in payload
    wd = payload["watchdog"]
    assert wd["hunt_armed"] is True
    assert wd["n_hunt_signals"] == 1
    assert wd["pending_hunts"] == 1
    assert wd["day_hunt_symbols"] == ["AMD"]


def test_signal_source_fields_and_position_frac_override():
    sig = ScannerSignal(
        date="2026-07-02",
        symbol="AMD",
        direction="UP",
        sig_ts=__import__("pandas").Timestamp("2026-07-02 10:00", tz="America/New_York"),
        spot=100.0,
        rank=0,
        bucket_id=0,
        contract="AMD...",
        moneyness="ATM",
        meta={
            "event_source": "hunt",
            "watchdog_state": "hunt",
            "route": "hunt",
            "hunt_detector": "washout_reclaim",
            "position_frac": 0.10,
        },
    )
    fields = Mag7BrokerOms._signal_source_fields(sig)
    assert fields["event_source"] == "hunt"
    assert fields["hunt_detector"] == "washout_reclaim"
    assert Mag7BrokerOms._position_frac_override(sig) == 0.10
    assert Mag7BrokerOms._position_frac_override(None) is None

"""Live OMS size_gate: wall-clock occupancy vs lagged sig_ts after close/resume."""
from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from maga7.live.broker_oms import LivePosition, Mag7BrokerOms


def _oms(tmp_path: Path) -> Mag7BrokerOms:
    class FakeRedis:
        def hget(self, *args):
            return b"0"

        def xadd(self, *args, **kwargs):
            return b"1-0"

        def pipeline(self, transaction=True):
            return self

        def set(self, *args, **kwargs):
            return self

        def delete(self, *args, **kwargs):
            return self

        def hset(self, *args, **kwargs):
            return self

        def execute(self):
            return []

    class Scanner:
        def record_fill(self, *args, **kwargs):
            return None

        states = {}
        stock_by = {}

    profile = {
        "trade": {
            "hold_minutes": 30,
            "tp_mult": 1.6,
            "sl_mult": 0.45,
            "position_frac": 0.25,
            "position_sizing": "concurrent",
            "max_concurrent_positions": 2,
            "trade_toxic": {"enabled": False},
            "risk": {
                "max_stock_staleness_sec": 30.0,
                "max_option_staleness_sec": 30.0,
                "max_exit_mid_jump_pct": 0.9,
                "max_gap_hold_ticks": 3,
            },
        },
        "fill": {"entry_frac": 0.8, "exit_frac": 0.8},
        "signal": {"top_k": 2},
    }
    connector = SimpleNamespace(
        ib=SimpleNamespace(isConnected=lambda: True),
        redis=FakeRedis(),
        config=SimpleNamespace(
            port=4002,
            account="DU1",
            max_stock_staleness_sec=30.0,
            max_option_staleness_sec=30.0,
        ),
        lock_status="LOCKED",
        data_mode="LIVE",
        option_quotes={},
        last_stock_tick={},
        locks={},
        ensure_option_subscription=lambda *_: True,
        release_on_demand_subscription=lambda *_: None,
    )
    return Mag7BrokerOms(
        profile=profile,
        scanner=Scanner(),
        connector=connector,
        session_id="sizegatesession01",
        trade_date="2026-07-27",
        session_dir=tmp_path,
        mode="shadow",
        equity=100_000.0,
    )


def test_size_uses_wall_clock_after_close_not_lagged_sig_ts(tmp_path):
    """Reproduce 07-27 AM: close stamps open_until=wall; deferred sig_ts lags → false size_gate."""
    oms = _oms(tmp_path)
    wall = pd.Timestamp.now(tz="America/New_York")
    # Just-closed AMZN/META seats (wall clock), as after GAP_FLATTEN.
    oms.open_until = {
        "AMZN": wall,
        "META": wall,
    }
    lagged = wall - pd.Timedelta(minutes=1)
    qty, frac = oms._size("NVDA", lagged, entry_price=2.50)
    assert qty > 0
    assert frac > 0


def test_size_still_blocks_when_two_positions_open(tmp_path):
    oms = _oms(tmp_path)
    now = time.time()
    for sym in ("AMZN", "META"):
        oms.positions[sym] = LivePosition(
            symbol=sym,
            contract=f"{sym}260727P00100000",
            con_id=1,
            direction="DN",
            qty=1,
            entry_price=2.0,
            entry_ts=now,
            signal_ts=now,
            rank=0,
            qty_frac=0.1,
            entry_bid=1.9,
            entry_ask=2.1,
            last_good_mid=2.0,
        )
    qty, _ = oms._size(
        "NVDA",
        pd.Timestamp.now(tz="America/New_York"),
        entry_price=2.50,
    )
    assert qty == 0
    assert oms._last_size_reject is not None
    assert oms._last_size_reject.get("size_reason") == "max_concurrent"


def test_restore_force_closes_shadow_exit_pending(tmp_path):
    oms = _oms(tmp_path)
    now = time.time()
    oms.positions["AMZN"] = LivePosition(
        symbol="AMZN",
        contract="AMZN260727P00232500",
        con_id=11,
        direction="DN",
        qty=1,
        entry_price=2.5,
        entry_ts=now - 60,
        signal_ts=now - 90,
        rank=0,
        qty_frac=0.1,
        entry_bid=2.4,
        entry_ask=2.6,
        last_good_mid=2.5,
        status="EXIT_PENDING",
    )
    # Persist then restore via a fresh OMS on the same state file.
    oms.publish_state()
    oms2 = _oms(tmp_path)
    assert "AMZN" not in oms2.positions
    # Stale open_until at/before wall should be pruned.
    past = pd.Timestamp.now(tz="America/New_York") - pd.Timedelta(seconds=5)
    oms2.open_until["ZZZ"] = past
    out = oms2._reconcile_restored_occupancy()
    assert "ZZZ" in out["open_until_pruned"]
    assert "ZZZ" not in oms2.open_until

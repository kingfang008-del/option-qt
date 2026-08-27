"""Unit tests for Mag7 entry iceberg sizing / OMS wiring."""
from __future__ import annotations

import time
from types import SimpleNamespace

from maga7.live.broker_oms import Mag7BrokerOms
from maga7.live.iceberg import (
    decode_chunk_queue,
    encode_chunk_queue,
    iceberg_config_from_trade,
    plan_entry_chunks,
)
from maga7.live.scanner import ScannerSignal
from maga7.common.replay import to_ny


def test_plan_splits_by_ask_size_and_notional():
    cfg = iceberg_config_from_trade(
        {
            "iceberg": {
                "enabled": True,
                "ask_size_frac": 0.5,
                "fallback_notional": 8_000,
                "max_chunks": 5,
            }
        }
    )
    # mid=2 → $200/contract; fallback clip = 8000/200 = 40
    # ask_size=20 → half = 10 → clip = min(10, 40) = 10
    chunks = plan_entry_chunks(25, mid=2.0, ask_size=20.0, cfg=cfg)
    assert chunks == [10, 10, 5]
    assert sum(chunks) == 25


def test_plan_single_when_small_or_disabled():
    cfg = iceberg_config_from_trade({"iceberg": {"enabled": True, "fallback_notional": 50_000}})
    assert plan_entry_chunks(3, mid=2.0, ask_size=100.0, cfg=cfg) == [3]
    off = iceberg_config_from_trade({"iceberg": {"enabled": False}})
    assert plan_entry_chunks(50, mid=2.0, ask_size=5.0, cfg=off) == [50]


def test_queue_codec():
    assert encode_chunk_queue([3, 2]) == "3,2"
    assert decode_chunk_queue("3,2,") == [3, 2]


def _oms(tmp_path, *, max_qty: int = 20) -> Mag7BrokerOms:
    class Scanner:
        def record_fill(self, *args, **kwargs):
            return None

        states = {}

    profile = {
        "trade": {
            "position_frac": 0.2,
            "position_sizing": "fixed",
            "hold_minutes": 30,
            "iceberg": {
                "enabled": True,
                "ask_size_frac": 0.5,
                "fallback_notional": 2_000,
                "max_chunks": 5,
                "max_total_sec": 15,
            },
            "risk": {
                "max_stock_staleness_sec": 30.0,
                "max_option_staleness_sec": 30.0,
                "max_spread_pct": 0.5,
                "max_entry_mid_jump_pct": 1.0,
                "require_entry_quote_stable_ticks": 1,
            },
        },
        "fill": {"entry_frac": 0.8, "exit_frac": 0.8},
        "signal": {"top_k": 2},
    }

    class FakeRedis:
        def hget(self, *a, **k):
            return b"0"

        def xadd(self, *a, **k):
            return b"1-0"

        def pipeline(self, transaction=True):
            return self

        def set(self, *a, **k):
            return True

        def hset(self, *a, **k):
            return True

        def delete(self, *a, **k):
            return True

        def execute(self):
            return []

    lock = SimpleNamespace(
        local_symbol="NVDA  260701C00100000",
        con_id=99,
        symbol="NVDA",
        strike=100.0,
        bucket_id=2,
    )
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
        locks={"NVDA": [lock]},
        option_contracts={99: object()},
        option_quotes={
            ("NVDA", "NVDA  260701C00100000"): {
                "ts": time.time(),
                "bid": 1.0,
                "ask": 1.2,
                "mid": 1.1,
                "ask_size": 4.0,
                "bid_size": 4.0,
            }
        },
        last_stock_tick={"NVDA": time.time()},
        ensure_option_subscription=lambda con_id: True,
        release_on_demand_subscription=lambda con_id: None,
    )
    oms = Mag7BrokerOms(
        profile=profile,
        scanner=Scanner(),
        connector=connector,
        session_id="iceberg_test",
        trade_date="2026-07-01",
        session_dir=tmp_path,
        mode="shadow",
        max_qty=max_qty,
        equity=100_000.0,
    )
    return oms


def test_shadow_entry_splits_into_iceberg_clips(tmp_path):
    oms = _oms(tmp_path, max_qty=10)
    # mid~1.1 → $110/ct; fallback 2000 → clip 18; ask_size 4 * 0.5 = 2 → clip=2
    # qty from equity: 100k*0.2 / 110 ≈ 181 → capped max_qty=10 → chunks [2,2,2,2,2]
    sig = ScannerSignal(
        date="2026-07-01",
        symbol="NVDA",
        direction="UP",
        sig_ts=to_ny("2026-07-01 11:00:00"),
        spot=100.0,
        rank=1,
        bucket_id=2,
        contract="NVDA  260701C00100000",
        moneyness="ATM",
        meta={},
    )
    assert oms.process_signal(sig) is True
    pos = oms.positions["NVDA"]
    assert pos.qty == 10
    plans = [e for e in _read_events(tmp_path) if e.get("kind") == "ICEBERG_PLAN"]
    assert plans and plans[0]["chunks"] == [2, 2, 2, 2, 2]
    opens = [e for e in _read_events(tmp_path) if e.get("kind") == "POSITION_OPEN"]
    assert len(opens) == 5


def _read_events(session_dir):
    import json
    from pathlib import Path

    path = Path(session_dir) / "order_events.jsonl"
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        rows.append(json.loads(line))
    return rows

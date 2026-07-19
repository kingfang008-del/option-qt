"""Unit tests for Mag7 live risk guards."""
from __future__ import annotations

import time
from types import SimpleNamespace

from maga7.live.broker_oms import Mag7BrokerOms
from maga7.live.risk_guards import (
    entry_quote_ok,
    fill_adverse,
    is_fresh,
    observe_exit_mid,
    quote_spread_fields,
    risk_config_from_trade,
)


def test_stock_and_option_freshness():
    now = 1_000.0
    assert is_fresh(now - 1.5, now=now, max_age_sec=2.0)
    assert not is_fresh(now - 2.5, now=now, max_age_sec=2.0)
    assert not is_fresh(None, now=now, max_age_sec=2.0)


def test_entry_rejects_mid_jump_and_wide_spread():
    cfg = risk_config_from_trade(
        {
            "risk": {
                "max_spread_pct": 0.25,
                "max_entry_mid_jump_pct": 0.15,
            }
        }
    )
    ok, reason, mid = entry_quote_ok(bid=1.0, ask=1.1, prev_mid=1.05, cfg=cfg)
    assert ok and reason == "ok" and mid is not None

    ok, reason, _ = entry_quote_ok(bid=1.0, ask=1.5, prev_mid=1.05, cfg=cfg)
    assert not ok and reason == "spread_too_wide"

    ok, reason, _ = entry_quote_ok(bid=1.4, ask=1.5, prev_mid=1.05, cfg=cfg)
    assert not ok and reason == "entry_mid_jump"


def test_exit_gap_hold_then_force():
    cfg = risk_config_from_trade({"risk": {"max_exit_mid_jump_pct": 0.20, "max_gap_hold_ticks": 3}})
    good, hold, status = observe_exit_mid(
        last_good_mid=1.0, mid=1.5, gap_hold_count=0, cfg=cfg
    )
    assert status == "gap" and good == 1.0 and hold == 1
    good, hold, status = observe_exit_mid(
        last_good_mid=good, mid=1.5, gap_hold_count=hold, cfg=cfg
    )
    assert status == "gap"
    good, hold, status = observe_exit_mid(
        last_good_mid=good, mid=1.5, gap_hold_count=hold, cfg=cfg
    )
    assert status == "gap_force" and hold == 3


def test_quote_spread_fields_open_close():
    open_f = quote_spread_fields(1.0, 1.2, fill_px=1.16, side="BUY")
    assert abs(float(open_f["spread"]) - 0.2) < 1e-9
    assert abs(float(open_f["spread_pct"]) - (0.2 / 1.1)) < 1e-9
    assert abs(float(open_f["fill_spread_frac"]) - 0.8) < 1e-9
    close_f = quote_spread_fields(1.0, 1.2, fill_px=1.04, side="SELL")
    assert abs(float(close_f["fill_spread_frac"]) - 0.8) < 1e-9


def test_adverse_fill_buy_through_ask():
    cfg = risk_config_from_trade({"risk": {"max_fill_spread_frac": 0.95}})
    bad, reason, _ = fill_adverse(
        bid=1.0, ask=1.2, fill_px=1.25, side="BUY", cfg=cfg
    )
    assert bad and reason == "fill_above_ask"
    adverse, reason, _ = fill_adverse(
        bid=1.0, ask=1.2, fill_px=1.18, side="BUY", cfg=cfg
    )
    assert not adverse and reason == "ok"


def test_oms_stock_fresh_and_day_circuit_flatten(tmp_path):
    events = []

    class FakeRedis:
        def hget(self, *args):
            return b"0"

        def xadd(self, *args, **kwargs):
            return b"1-0"

        def pipeline(self, transaction=True):
            return self

        def delete(self, *args):
            return self

        def hset(self, *args, **kwargs):
            return self

        def execute(self):
            return []

    class Scanner:
        def record_fill(self, *args, **kwargs):
            return None

        states = {}

    profile = {
        "trade": {
            "day_circuit": -0.05,
            "position_frac": 0.2,
            "position_sizing": "fixed",
            "risk": {
                "max_stock_staleness_sec": 2.0,
                "day_circuit_force_flatten": True,
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
            max_stock_staleness_sec=2.0,
            max_option_staleness_sec=5.0,
        ),
        lock_status="LOCKED",
        data_mode="LIVE",
        option_quotes={},
        last_stock_tick={"AAPL": time.time()},
        locks={},
        ensure_option_subscription=lambda *_: True,
        release_on_demand_subscription=lambda *_: None,
    )
    oms = Mag7BrokerOms(
        profile=profile,
        scanner=Scanner(),
        connector=connector,
        session_id="testsession01",
        trade_date="2026-07-16",
        session_dir=tmp_path,
        mode="shadow",
        equity=100_000.0,
    )
    oms._event = lambda kind, payload: events.append((kind, payload))  # type: ignore

    assert oms._stock_fresh("AAPL")[0] is True
    connector.last_stock_tick["AAPL"] = time.time() - 10
    assert oms._stock_fresh("AAPL")[1] == "stock_stale"

    from maga7.live.broker_oms import LivePosition

    oms.positions["NVDA"] = LivePosition(
        symbol="NVDA",
        contract="NVDA260717C00100000",
        con_id=1,
        direction="UP",
        qty=1,
        entry_price=1.0,
        entry_ts=time.time(),
        signal_ts=time.time(),
        rank=1,
        qty_frac=0.2,
        entry_bid=0.9,
        entry_ask=1.1,
        last_bid=0.9,
        last_ask=1.1,
        last_good_mid=1.0,
    )
    connector.option_quotes[("NVDA", "NVDA260717C00100000")] = {
        "bid": 0.9,
        "ask": 1.1,
        "ts": time.time(),
    }
    oms.equity = 90_000.0  # -10%
    oms._trip_day_circuit(day_ret=-0.10)
    assert oms.day_halted is True
    assert any(kind == "DAY_CIRCUIT" for kind, _ in events)
    assert "NVDA" not in oms.positions  # shadow force flatten closes

"""Unit tests for Mag7 dynamic LMT requote."""
from __future__ import annotations

import time
from types import SimpleNamespace

from maga7.live.broker_oms import LivePosition, Mag7BrokerOms, PendingIntent
from maga7.live.requote import (
    entry_requote_limit,
    exit_requote_limit,
    is_urgent_exit_reason,
    requote_config_from_trade,
)


def test_entry_requote_escalates_but_respects_cap_and_ask():
    cfg = requote_config_from_trade(
        {
            "risk": {
                "requote": {
                    "entry_frac_step": 0.05,
                    "entry_max_slippage_pct": 0.08,
                    "entry_step_cap_pct": 0.50,
                }
            }
        }
    )
    px1 = entry_requote_limit(
        bid=1.0,
        ask=1.20,
        attempt_no=1,
        base_frac=0.80,
        max_frac=0.95,
        ref_price=1.10,
        prev_limit=1.16,
        cfg=cfg,
    )
    assert px1 is not None
    assert px1 <= 1.19  # never cross ask-tick
    assert px1 >= 1.16

    # Cap vs ref mid 1.10 * 1.08 = 1.188
    px_cap = entry_requote_limit(
        bid=1.0,
        ask=2.0,
        attempt_no=3,
        base_frac=0.80,
        max_frac=0.95,
        ref_price=1.10,
        prev_limit=1.16,
        cfg=cfg,
    )
    assert px_cap is None or px_cap <= 1.10 * 1.08 + 1e-9


def test_exit_requote_normal_stays_at_or_above_bid():
    cfg = requote_config_from_trade({})
    px = exit_requote_limit(
        bid=1.0,
        ask=1.2,
        attempt_no=2,
        prev_limit=1.05,
        urgent=False,
        cfg=cfg,
    )
    assert px == 1.04 or px == 1.0 or px >= 1.0
    assert px is not None and px >= 1.0


def test_exit_urgent_may_go_below_bid():
    cfg = requote_config_from_trade(
        {"risk": {"requote": {"exit_step": 0.01, "exit_urgent_max_abs_discount": 0.15}}}
    )
    px = exit_requote_limit(
        bid=1.0,
        ask=1.2,
        attempt_no=2,
        prev_limit=1.0,
        urgent=True,
        cfg=cfg,
    )
    assert px is not None and px < 1.0
    assert is_urgent_exit_reason("SL")
    assert is_urgent_exit_reason("DAY_CIRCUIT")
    assert is_urgent_exit_reason("PROFIT_PROTECT")
    assert not is_urgent_exit_reason("TP")


def test_oms_try_requote_exit_places_child(tmp_path):
    placed = []

    class FakeRedis:
        def hget(self, *args):
            return b"0"

        def xadd(self, *args, **kwargs):
            return b"1-0"

        def pipeline(self, transaction=True):
            return self

        def delete(self, *args):
            return self

        def set(self, *args, **kwargs):
            return True

        def hset(self, *args, **kwargs):
            return self

        def execute(self):
            return []

    class Scanner:
        states = {}

        def record_fill(self, *args, **kwargs):
            return None

    connector = SimpleNamespace(
        ib=SimpleNamespace(
            isConnected=lambda: True,
            cancelOrder=lambda *_: None,
            positions=lambda: [
                SimpleNamespace(
                    account="DU1",
                    contract=SimpleNamespace(
                        localSymbol="AAPL260717C00100000"
                    ),
                    position=1,
                )
            ],
        ),
        redis=FakeRedis(),
        config=SimpleNamespace(
            port=4002,
            account="DU1",
            max_stock_staleness_sec=2.0,
            max_option_staleness_sec=5.0,
        ),
        lock_status="LOCKED",
        data_mode="LIVE",
        option_quotes={
            ("AAPL", "AAPL260717C00100000"): {
                "bid": 1.0,
                "ask": 1.2,
                "ts": time.time(),
            }
        },
        last_stock_tick={"AAPL": time.time()},
        locks={},
        option_contracts={},
        ensure_option_subscription=lambda *_: True,
        release_on_demand_subscription=lambda *_: None,
    )
    profile = {
        "trade": {
            "risk": {"requote": {"enabled": True, "max_exit_requotes": 3}},
        },
        "fill": {"entry_frac": 0.8, "exit_frac": 0.8},
        "signal": {"top_k": 2},
    }
    oms = Mag7BrokerOms(
        profile=profile,
        scanner=Scanner(),
        connector=connector,
        session_id="testsession02",
        trade_date="2026-07-16",
        session_dir=tmp_path,
        mode="paper",
        equity=100_000.0,
    )
    oms.reconcile_ok = True
    oms.account_ready = True

    def fake_place(intent):
        placed.append(intent.intent_id)
        intent.status = "SUBMITTED"
        oms.trades[intent.intent_id] = SimpleNamespace(
            order=SimpleNamespace(orderId=1),
            orderStatus=SimpleNamespace(status="Submitted", filled=0, avgFillPrice=0, permId=1),
        )

    oms._place_broker_order = fake_place  # type: ignore
    oms.positions["AAPL"] = LivePosition(
        symbol="AAPL",
        contract="AAPL260717C00100000",
        con_id=42,
        direction="UP",
        qty=1,
        entry_price=1.0,
        entry_ts=time.time(),
        signal_ts=time.time(),
        rank=1,
        qty_frac=0.2,
        entry_bid=0.9,
        entry_ask=1.1,
        status="EXIT_PENDING",
        last_bid=1.0,
        last_ask=1.2,
        last_good_mid=1.1,
    )
    parent = PendingIntent(
        intent_id="M7-parent-S-1",
        action="SELL",
        symbol="AAPL",
        contract="AAPL260717C00100000",
        con_id=42,
        qty=1,
        limit_price=1.05,
        reason="TP",
        created_at=time.time(),
        status="CANCELLED",
        requote_attempt=0,
        ref_price=1.1,
    )
    oms.intents[parent.intent_id] = parent
    assert oms._try_requote(parent, remaining_qty=1) is True
    assert parent.replaced_by
    child = oms.intents[parent.replaced_by]
    assert child.requote_attempt == 1
    assert child.limit_price <= 1.05
    assert child.limit_price >= 1.0
    assert placed == [child.intent_id]
    assert oms.positions["AAPL"].status == "EXIT_PENDING"
    oms.force_flatten("EOD")
    assert placed == [child.intent_id]
    child.status = "FILLED"
    oms.positions["AAPL"].status = "OPEN"
    connector.ib.isConnected = lambda: False
    oms.force_flatten("EOD")
    assert oms.positions["AAPL"].status == "EXIT_PENDING"
    assert oms._pending_force_exits == {"AAPL": "EOD"}
    assert placed == [child.intent_id]
    connector.ib.isConnected = lambda: True
    oms.on_feed_reconnected()
    assert oms._pending_force_exits == {}
    assert len(placed) == 2

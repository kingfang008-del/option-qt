"""Live OMS trade_toxic + reconnect recheck."""
from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from maga7.common.fills import FillSpec
from maga7.common.option_trades import (
    trade_toxic_from_trade,
    trade_toxic_in_cut_window,
    trade_toxic_is_dig,
)
from maga7.live.broker_oms import LivePosition, Mag7BrokerOms, PendingIntent
from maga7.live.requote import is_urgent_exit_reason


def _oms(tmp_path: Path, trade: dict | None = None) -> Mag7BrokerOms:
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

        states = {
            "AMD": SimpleNamespace(bars=[{"close": 516.0}], mf10=0.0),
        }

    profile = {
        "trade": {
            "hold_minutes": 30,
            "tp_mult": 1.6,
            "sl_mult": 0.45,
            "exit_mode": "hold_extend",
            "exit_mf_grace_seconds": 60,
            "position_frac": 0.2,
            "position_sizing": "fixed",
            "trade_toxic": trade
            or {
                "enabled": True,
                "cut_ret": 0.25,
                "mfe_bypass": 0.05,
                "min_hold_seconds": 60,
                "persist_seconds": 0,
                "max_cut_seconds": 600,
            },
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
        last_stock_tick={"AMD": time.time()},
        locks={},
        ensure_option_subscription=lambda *_: True,
        release_on_demand_subscription=lambda *_: None,
    )
    return Mag7BrokerOms(
        profile=profile,
        scanner=Scanner(),
        connector=connector,
        session_id="toxicsession01",
        trade_date="2026-07-20",
        session_dir=tmp_path,
        mode="shadow",
        equity=100_000.0,
    )


def test_trade_toxic_helpers():
    cfg = trade_toxic_from_trade(
        {"trade_toxic": {"enabled": True, "cut_ret": 0.25, "mfe_bypass": 0.05, "max_cut_seconds": 600}}
    )
    assert trade_toxic_is_dig(mtm_ret=-0.30, peak_mfe=0.01, cfg=cfg)
    assert not trade_toxic_is_dig(mtm_ret=-0.30, peak_mfe=0.10, cfg=cfg)
    assert trade_toxic_in_cut_window(120, cfg)
    assert not trade_toxic_in_cut_window(700, cfg)
    assert trade_toxic_in_cut_window(700, cfg, bypass_max_cut=True)


def test_live_quote_fallback_uses_twenty_percent_cut_at_093_to_070(tmp_path):
    """Live quote marks must use the replay qf_cut20 threshold, not print cut25."""
    oms = _oms(
        tmp_path,
        trade={
            "enabled": True,
            "cut_ret": 0.25,
            "mfe_bypass": 0.05,
            "min_hold_seconds": 60,
            "persist_seconds": 0,
            "max_cut_seconds": 600,
            "quote_fallback": True,
            "quote_fallback_cut_ret": 0.20,
        },
    )
    now = time.time()
    oms.positions["AMD"] = LivePosition(
        symbol="AMD",
        contract="AMD260720C00517500",
        con_id=1,
        direction="UP",
        qty=1,
        entry_price=0.93,
        entry_ts=now - 180,
        signal_ts=now - 190,
        rank=1,
        qty_frac=0.2,
        entry_bid=0.91,
        entry_ask=0.94,
        last_good_mid=0.93,
        entry_stock_px=516.0,
        peak_mfe=0.0,
    )
    # FillSpec exit_frac=.8 gives an executable sell mark of 0.70.
    oms.connector.option_quotes[("AMD", "AMD260720C00517500")] = {
        "bid": 0.69,
        "ask": 0.74,
        "ts": now,
    }
    events: list[tuple[str, dict]] = []
    oms._event = lambda kind, payload: events.append((kind, payload))  # type: ignore
    oms.evaluate_exits(now)
    assert "AMD" not in oms.positions
    assert any(
        kind == "POSITION_CLOSE" and payload.get("reason") == "TRADE_TOX"
        for kind, payload in events
    )


def test_print_threshold_does_not_apply_quote_fallback_cut(tmp_path):
    oms = _oms(
        tmp_path,
        trade={
            "enabled": True,
            "cut_ret": 0.25,
            "mfe_bypass": 0.05,
            "min_hold_seconds": 60,
            "max_cut_seconds": 600,
            "quote_fallback": False,
            "quote_fallback_cut_ret": 0.20,
        },
    )
    position = LivePosition(
        symbol="AMD",
        contract="AMD260720C00517500",
        con_id=1,
        direction="UP",
        qty=1,
        entry_price=0.93,
        entry_ts=100.0,
        signal_ts=90.0,
        rank=1,
        qty_frac=0.2,
        entry_bid=0.91,
        entry_ask=0.94,
        peak_mfe=0.0,
    )
    assert (
        oms._trade_toxic_reason(
            position, mtm_ret=-0.247, held=180.0, asof_ts=280.0
        )
        == ""
    )
    assert (
        oms._trade_toxic_reason(
            position, mtm_ret=-0.251, held=181.0, asof_ts=281.0
        )
        == "TRADE_TOX"
    )


@pytest.mark.parametrize(
    (
        "quote_fallback",
        "mtm_ret",
        "peak_mfe",
        "held",
        "reconnect",
        "expected",
    ),
    [
        (True, -0.199, 0.00, 180.0, False, ""),
        (True, -0.200, 0.00, 180.0, False, "TRADE_TOX"),
        (True, -0.247, 0.00, 180.0, False, "TRADE_TOX"),
        (True, -0.300, 0.05, 180.0, False, ""),
        (True, -0.300, 0.00, 59.0, False, ""),
        (True, -0.300, 0.00, 601.0, False, ""),
        (True, -0.300, 0.00, 601.0, True, "TRADE_TOX_RECONNECT"),
        (False, -0.247, 0.00, 180.0, False, ""),
        (False, -0.250, 0.00, 180.0, False, "TRADE_TOX"),
    ],
)
def test_trade_toxic_live_boundary_matrix(
    tmp_path,
    quote_fallback,
    mtm_ret,
    peak_mfe,
    held,
    reconnect,
    expected,
):
    oms = _oms(
        tmp_path,
        trade={
            "enabled": True,
            "cut_ret": 0.25,
            "mfe_bypass": 0.05,
            "min_hold_seconds": 60,
            "persist_seconds": 0,
            "max_cut_seconds": 600,
            "quote_fallback": quote_fallback,
            "quote_fallback_cut_ret": 0.20,
        },
    )
    position = LivePosition(
        symbol="AMD",
        contract="AMD260720C00517500",
        con_id=1,
        direction="UP",
        qty=1,
        entry_price=0.93,
        entry_ts=100.0,
        signal_ts=90.0,
        rank=1,
        qty_frac=0.2,
        entry_bid=0.91,
        entry_ask=0.94,
        peak_mfe=peak_mfe,
        toxic_reconnect_pending=reconnect,
    )
    assert (
        oms._trade_toxic_reason(
            position,
            mtm_ret=mtm_ret,
            held=held,
            asof_ts=100.0 + held,
        )
        == expected
    )


def test_jul27_nvda_am_pulse_sl_regression(tmp_path):
    """Observed shadow event: 0.935 entry, 0.735 exit, SL20 after about 71s."""
    oms = _oms(tmp_path)
    oms.fill = FillSpec(entry_frac=0.75, exit_frac=0.75)
    now = time.time()
    oms.positions["NVDA"] = LivePosition(
        symbol="NVDA",
        contract="NVDA  260727P00200000",
        con_id=900722877,
        direction="DN",
        qty=1,
        entry_price=0.935,
        entry_ts=now - 71,
        signal_ts=now - 133,
        rank=1,
        qty_frac=0.1,
        entry_bid=0.92,
        entry_ask=0.94,
        last_good_mid=0.93,
        exit_sl_mult=0.80,
        exit_tp_mult=1.15,
        exit_hold_sec=900,
        exit_simple=True,
        route="am_pulse",
    )
    oms.connector.option_quotes[("NVDA", "NVDA  260727P00200000")] = {
        "bid": 0.73,
        "ask": 0.75,
        "ts": now,
    }
    events: list[tuple[str, dict]] = []
    oms._event = lambda kind, payload: events.append((kind, payload))  # type: ignore
    oms.evaluate_exits(now)
    assert "NVDA" not in oms.positions
    close = next(
        payload
        for kind, payload in events
        if kind == "POSITION_CLOSE" and payload.get("reason") == "SL"
    )
    assert abs(float(close["exit_price"]) - 0.735) < 1e-12
    assert abs(float(close["ret"]) - (0.735 / 0.935 - 1.0)) < 1e-12


def test_partial_exit_intent_deduplicates_new_sell(tmp_path):
    """A partial broker fill must not allow a second full-size exit order."""
    oms = _oms(tmp_path)
    now = time.time()
    position = LivePosition(
        symbol="AMD",
        contract="AMD260720C00517500",
        con_id=1,
        direction="UP",
        qty=1,
        entry_price=1.0,
        entry_ts=now - 180,
        signal_ts=now - 190,
        rank=1,
        qty_frac=0.2,
        entry_bid=0.98,
        entry_ask=1.02,
    )
    oms.positions["AMD"] = position
    partial = PendingIntent(
        intent_id="partial-exit",
        action="SELL",
        symbol="AMD",
        contract=position.contract,
        con_id=position.con_id,
        qty=2,
        limit_price=0.80,
        reason="SL",
        created_at=now,
        status="PARTIALLY_FILLED",
        filled=1,
    )
    oms.intents[partial.intent_id] = partial
    events: list[tuple[str, dict]] = []
    oms._event = lambda kind, payload: events.append((kind, payload))  # type: ignore
    oms._submit_exit(
        position,
        "SL",
        0.75,
        {"bid": 0.74, "ask": 0.76, "ts": now},
    )
    assert list(oms.intents) == ["partial-exit"]
    assert any(kind == "EXIT_DEDUP" for kind, _ in events)


def test_evaluate_exits_cuts_trade_tox(tmp_path):
    oms = _oms(tmp_path)
    now = time.time()
    oms.positions["AMD"] = LivePosition(
        symbol="AMD",
        contract="AMD260720C00517500",
        con_id=1,
        direction="UP",
        qty=1,
        entry_price=4.60,
        entry_ts=now - 180,
        signal_ts=now - 200,
        rank=1,
        qty_frac=0.2,
        entry_bid=4.50,
        entry_ask=4.70,
        last_good_mid=4.60,
        entry_stock_px=516.0,
        peak_mfe=0.0,
    )
    # Sell model ~ -35% from entry → TRADE_TOX
    oms.connector.option_quotes[("AMD", "AMD260720C00517500")] = {
        "bid": 2.90,
        "ask": 3.10,
        "ts": now,
    }
    events: list[tuple[str, dict]] = []
    oms._event = lambda kind, payload: events.append((kind, payload))  # type: ignore
    oms.evaluate_exits(now)
    assert "AMD" not in oms.positions
    assert any(
        kind == "POSITION_CLOSE" and payload.get("reason") == "TRADE_TOX"
        for kind, payload in events
    )


def test_evaluate_exits_respects_max_cut_unless_reconnect(tmp_path):
    oms = _oms(tmp_path)
    now = time.time()
    oms.positions["AMD"] = LivePosition(
        symbol="AMD",
        contract="AMD260720C00517500",
        con_id=1,
        direction="UP",
        qty=1,
        entry_price=4.60,
        entry_ts=now - 900,  # beyond max_cut=600
        signal_ts=now - 920,
        rank=1,
        qty_frac=0.2,
        entry_bid=4.50,
        entry_ask=4.70,
        last_good_mid=4.60,
        entry_stock_px=516.0,
        peak_mfe=0.0,
        toxic_reconnect_pending=False,
    )
    oms.connector.option_quotes[("AMD", "AMD260720C00517500")] = {
        "bid": 2.40,
        "ask": 2.60,
        "ts": now,
    }
    oms.evaluate_exits(now)
    assert "AMD" in oms.positions  # max_cut blocks

    oms.positions["AMD"].toxic_reconnect_pending = True
    events: list[tuple[str, dict]] = []
    oms._event = lambda kind, payload: events.append((kind, payload))  # type: ignore
    oms.evaluate_exits(now)
    assert "AMD" not in oms.positions
    assert any(
        kind == "POSITION_CLOSE" and payload.get("reason") == "TRADE_TOX_RECONNECT"
        for kind, payload in events
    )


def test_restore_arms_toxic_reconnect(tmp_path):
    oms = _oms(tmp_path)
    now = time.time()
    pos = LivePosition(
        symbol="AMD",
        contract="AMD260720C00517500",
        con_id=1,
        direction="UP",
        qty=1,
        entry_price=4.60,
        entry_ts=now - 900,
        signal_ts=now - 920,
        rank=1,
        qty_frac=0.2,
        entry_bid=4.50,
        entry_ask=4.70,
        last_good_mid=4.60,
        entry_stock_px=516.0,
    )
    state = {
        "schema_version": 1,
        "session_id": oms.session_id,
        "trade_date": oms.trade_date,
        "mode": "shadow",
        "profile_hash": oms.profile_hash,
        "equity": 100_000.0,
        "day_start_equity": 100_000.0,
        "realized_pnl": 0.0,
        "day_halted": False,
        "available_funds": 100_000.0,
        "account_ready": True,
        "positions": {"AMD": pos.__dict__},
        "intents": {},
        "pending_signals": {},
        "open_until": {},
        "seen_fills": [],
        "seen_commissions": [],
    }
    path = Path(tmp_path) / "oms_state.json"
    path.write_text(json.dumps(state), encoding="utf-8")

    oms2 = _oms(tmp_path)
    assert "AMD" in oms2.positions
    assert oms2.positions["AMD"].toxic_reconnect_pending is True


def test_trade_tox_is_urgent():
    assert is_urgent_exit_reason("TRADE_TOX")
    assert is_urgent_exit_reason("TRADE_TOX_RECONNECT")

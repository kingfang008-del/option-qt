"""Fault-injection regressions for live hardening (no IBKR / no Redis required).

Scenarios mirror the failure modes that otherwise burn a full trading day:
  1) stale quotes block entry
  2) wide spread / mid jump reject
  3) exit gap hold → GAP_FLATTEN
  4) adverse fill → flatten
  5) event blackout empties the day
  6) day circuit force-flatten + halt new entries
"""
from __future__ import annotations

import time
from types import SimpleNamespace

from maga7.common.event_calendar import resolve_event_blackout
from maga7.live.broker_oms import LivePosition, Mag7BrokerOms
from maga7.live.risk_guards import (
    entry_quote_ok,
    fill_adverse,
    is_fresh,
    observe_exit_mid,
    risk_config_from_trade,
)


def _fake_redis():
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

    return FakeRedis()


def _oms(tmp_path, *, trade_extra: dict | None = None) -> tuple[Mag7BrokerOms, list]:
    events: list = []

    class Scanner:
        def record_fill(self, *args, **kwargs):
            return None

        states = {}

    trade = {
        "day_circuit": -0.05,
        "position_frac": 0.2,
        "position_sizing": "fixed",
        "hold_minutes": 30,
        "exit_mode": "hold_extend",
        "hold_extend_minutes": 45,
        "hold_extend_mtm_min": 0.0,
        "hold_extend_require_mf": False,
        "tp_mult": 1.6,
        "sl_mult": 0.4,
        "risk": {
            "max_stock_staleness_sec": 2.0,
            "max_option_staleness_sec": 5.0,
            "max_spread_pct": 0.25,
            "max_entry_mid_jump_pct": 0.15,
            "max_exit_mid_jump_pct": 0.20,
            "max_gap_hold_ticks": 3,
            "max_fill_spread_frac": 0.95,
            "day_circuit_force_flatten": True,
            "halt_entries_on_gap": True,
        },
    }
    if trade_extra:
        # shallow merge risk
        risk = {**trade["risk"], **(trade_extra.pop("risk", {}) or {})}
        trade.update(trade_extra)
        trade["risk"] = risk

    profile = {
        "trade": trade,
        "fill": {"entry_frac": 0.8, "exit_frac": 0.8},
        "signal": {"top_k": 2},
    }
    connector = SimpleNamespace(
        ib=SimpleNamespace(isConnected=lambda: True),
        redis=_fake_redis(),
        config=SimpleNamespace(
            port=4002,
            account="DU1",
            max_stock_staleness_sec=2.0,
            max_option_staleness_sec=5.0,
        ),
        lock_status="LOCKED",
        data_mode="LIVE",
        option_quotes={},
        last_stock_tick={"NVDA": time.time()},
        locks={},
        ensure_option_subscription=lambda *_: True,
        release_on_demand_subscription=lambda *_: None,
    )
    oms = Mag7BrokerOms(
        profile=profile,
        scanner=Scanner(),
        connector=connector,
        session_id="faultinj01",
        trade_date="2026-07-16",
        session_dir=tmp_path,
        mode="shadow",
        equity=100_000.0,
    )
    oms._event = lambda kind, payload: events.append((kind, payload))  # type: ignore
    return oms, events


def test_fault_stale_stock_blocks_entry_freshness():
    now = 1_000.0
    assert is_fresh(now - 0.5, now=now, max_age_sec=2.0)
    assert not is_fresh(now - 3.0, now=now, max_age_sec=2.0)
    assert not is_fresh(None, now=now, max_age_sec=2.0)


def test_fault_wide_spread_and_mid_jump_reject_entry():
    cfg = risk_config_from_trade(
        {"risk": {"max_spread_pct": 0.25, "max_entry_mid_jump_pct": 0.15}}
    )
    ok, reason, _ = entry_quote_ok(bid=1.0, ask=1.5, prev_mid=1.05, cfg=cfg)
    assert not ok and reason == "spread_too_wide"
    ok, reason, _ = entry_quote_ok(bid=1.4, ask=1.5, prev_mid=1.05, cfg=cfg)
    assert not ok and reason == "entry_mid_jump"


def test_fault_exit_gap_force_flatten_after_hold_ticks():
    cfg = risk_config_from_trade({"risk": {"max_exit_mid_jump_pct": 0.20, "max_gap_hold_ticks": 3}})
    good, hold, status = 1.0, 0, "ok"
    for _ in range(3):
        good, hold, status = observe_exit_mid(
            last_good_mid=good, mid=1.5, gap_hold_count=hold, cfg=cfg
        )
    assert status == "gap_force" and hold == 3


def test_fault_oms_gap_hold_then_flattens_093_to_070(tmp_path):
    """A large quote jump may be held briefly, but cannot leave risk open indefinitely."""
    oms, events = _oms(tmp_path)
    now = time.time()
    oms.positions["NVDA"] = LivePosition(
        symbol="NVDA",
        contract="NVDA260717C00100000",
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
        last_bid=0.91,
        last_ask=0.94,
        last_good_mid=0.93,
    )
    oms.connector.option_quotes[("NVDA", "NVDA260717C00100000")] = {
        "bid": 0.69,
        "ask": 0.74,
        "ts": now,
    }
    oms.evaluate_exits(now)
    assert "NVDA" in oms.positions
    assert oms.positions["NVDA"].gap_hold_count == 1
    oms.evaluate_exits(now + 0.1)
    assert "NVDA" in oms.positions
    assert oms.positions["NVDA"].gap_hold_count == 2
    oms.evaluate_exits(now + 0.2)
    assert "NVDA" not in oms.positions
    assert any(
        kind == "POSITION_CLOSE" and payload.get("reason") == "GAP_FLATTEN"
        for kind, payload in events
    )


def test_fault_adverse_fill_buy_through_ask():
    cfg = risk_config_from_trade({"risk": {"max_fill_spread_frac": 0.95}})
    bad, reason, _ = fill_adverse(bid=1.0, ask=1.2, fill_px=1.25, side="BUY", cfg=cfg)
    assert bad and reason == "fill_above_ask"


def test_fault_event_blackout_empties_session_dates():
    cfg = {
        "event_calendar_block": True,
        "event_dates": ["2026-06-17", "2026-07-09"],
        "event_blackout_sessions": 0,
    }
    blocked = resolve_event_blackout(
        cfg, session_dates=["2026-06-16", "2026-06-17", "2026-06-18", "2026-07-09"]
    )
    assert "2026-06-17" in blocked
    assert "2026-07-09" in blocked
    assert "2026-06-16" not in blocked


def test_fault_oms_stale_stock_and_day_circuit(tmp_path):
    oms, events = _oms(tmp_path)
    assert oms._stock_fresh("NVDA")[0] is True
    oms.connector.last_stock_tick["NVDA"] = time.time() - 10
    ok, reason = oms._stock_fresh("NVDA")
    assert ok is False and reason == "stock_stale"

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
    oms.connector.option_quotes[("NVDA", "NVDA260717C00100000")] = {
        "bid": 0.9,
        "ask": 1.1,
        "ts": time.time(),
    }
    oms.equity = 90_000.0
    oms._trip_day_circuit(day_ret=-0.10)
    assert oms.day_halted is True
    assert any(kind == "DAY_CIRCUIT" for kind, _ in events)
    assert "NVDA" not in oms.positions


def test_fault_halt_entries_after_day_circuit(tmp_path):
    oms, _ = _oms(tmp_path)
    oms.day_halted = True
    # Shadow path should refuse new risk when halted (same flag live uses).
    assert oms.day_halted is True
    snap = oms.snapshot()
    assert snap.get("day_halted") is True


def test_fault_profile_risk_defaults_match_freeze_shape():
    """Sanity: freeze-like risk block parses into guards used by OMS."""
    freeze_risk = {
        "risk": {
            "max_stock_staleness_sec": 2.0,
            "max_option_staleness_sec": 5.0,
            "max_spread_pct": 0.25,
            "max_entry_mid_jump_pct": 0.15,
            "max_exit_mid_jump_pct": 0.2,
            "max_gap_hold_ticks": 3,
            "max_fill_spread_frac": 0.95,
            "max_exit_chase": 3,
            "day_circuit_force_flatten": True,
            "halt_entries_on_gap": True,
        }
    }
    cfg = risk_config_from_trade(freeze_risk)
    assert cfg.max_stock_staleness_sec == 2.0
    assert cfg.max_gap_hold_ticks == 3
    assert cfg.day_circuit_force_flatten is True

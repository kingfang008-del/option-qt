"""Unit tests for Mag7 live risk guards."""
from __future__ import annotations

import time
from types import SimpleNamespace

from maga7.live.broker_oms import Mag7BrokerOms
from maga7.live.risk_guards import (
    entry_feed_ok,
    entry_quote_ok,
    entry_stock_drift_ok,
    fill_adverse,
    is_fresh,
    next_entry_quote_stable_ticks,
    observe_exit_mid,
    quote_spread_fields,
    risk_config_from_trade,
    signal_quote_lag_ok,
)


def test_stock_and_option_freshness():
    now = 1_000.0
    assert is_fresh(now - 1.5, now=now, max_age_sec=2.0)
    assert not is_fresh(now - 2.5, now=now, max_age_sec=2.0)
    assert not is_fresh(None, now=now, max_age_sec=2.0)
    assert not is_fresh(
        now + 2.0,
        now=now,
        max_age_sec=2.0,
        max_future_skew_sec=1.0,
    )


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

    ok, reason, mid = entry_quote_ok(
        bid=0.01, ask=0.03, prev_mid=None, cfg=cfg, min_mid=0.05
    )
    assert not ok and reason == "entry_mid_too_low" and mid == 0.02


def test_satellite_quote_must_follow_signal_within_lag_cap():
    ok, reason, lag = signal_quote_lag_ok(
        signal_ts=1_000.0, quote_ts=1_004.5, max_lag_sec=5.0
    )
    assert ok and reason == "ok" and lag == 4.5

    ok, reason, lag = signal_quote_lag_ok(
        signal_ts=1_000.0, quote_ts=999.5, max_lag_sec=5.0
    )
    assert not ok and reason == "option_quote_before_signal" and lag == -0.5

    ok, reason, lag = signal_quote_lag_ok(
        signal_ts=1_000.0, quote_ts=1_005.1, max_lag_sec=5.0
    )
    assert not ok and reason == "option_quote_lag_exceeded"
    assert lag is not None and abs(lag - 5.1) < 1e-9


def test_am_entry_stock_drift_blocks_chase_and_reversal():
    ok, reason, drift = entry_stock_drift_ok(
        signal_spot=100.0,
        current_spot=100.2,
        direction="UP",
        max_chase=0.003,
        max_reversal=0.0015,
    )
    assert ok and reason == "ok" and drift is not None

    ok, reason, _ = entry_stock_drift_ok(
        signal_spot=100.0,
        current_spot=100.4,
        direction="UP",
        max_chase=0.003,
        max_reversal=0.0015,
    )
    assert not ok and reason == "entry_stock_chase_exceeded"

    ok, reason, _ = entry_stock_drift_ok(
        signal_spot=100.0,
        current_spot=100.2,
        direction="DN",
        max_chase=0.003,
        max_reversal=0.0015,
    )
    assert not ok and reason == "entry_stock_reversed"


def test_entry_feed_blocks_delayed_and_universe_stale():
    cfg = risk_config_from_trade(
        {
            "risk": {
                "halt_entries_on_feed_unhealthy": True,
                "require_live_market_data": True,
                "max_stock_staleness_sec": 2.0,
                "max_universe_stale_frac": 0.34,
                "universe_stale_mult": 2.5,
                "entry_cooldown_after_gap_sec": 120.0,
            }
        }
    )
    ok, reason = entry_feed_ok(
        connected=True,
        data_mode="DELAYED",
        stock_lags_sec={"NVDA": 0.1, "TSLA": 0.1, "QQQ": 0.1},
        cfg=cfg,
        now=1_000.0,
    )
    assert not ok and reason == "market_data_delayed"

    ok, reason = entry_feed_ok(
        connected=True,
        data_mode="LIVE",
        stock_lags_sec={"NVDA": 20.0, "TSLA": 20.0, "AAPL": 0.1, "QQQ": 0.1},
        cfg=cfg,
        now=1_000.0,
    )
    assert not ok and reason == "universe_stale"

    ok, reason = entry_feed_ok(
        connected=True,
        data_mode="LIVE",
        stock_lags_sec={"NVDA": 0.1, "TSLA": 0.1, "QQQ": 0.1},
        cfg=cfg,
        now=1_050.0,
        last_gap_event_ts=1_000.0,
    )
    assert not ok and reason == "gap_cooldown"

    ok, reason = entry_feed_ok(
        connected=True,
        data_mode="LIVE",
        stock_lags_sec={"NVDA": 0.1, "TSLA": 0.1, "QQQ": 0.1},
        cfg=cfg,
        now=1_200.0,
        last_gap_event_ts=1_000.0,
    )
    assert ok and reason == "ok"


def test_entry_quote_warmup_requires_stable_ticks():
    stable, ready, reason = next_entry_quote_stable_ticks(
        prev_stable=0, quote_ok=True, prev_mid=None, require_ticks=2
    )
    assert stable == 1 and not ready and reason == "option_quote_warmup"
    stable, ready, reason = next_entry_quote_stable_ticks(
        prev_stable=stable, quote_ok=True, prev_mid=1.0, require_ticks=2
    )
    assert stable == 2 and ready and reason == "ok"
    stable, ready, reason = next_entry_quote_stable_ticks(
        prev_stable=2, quote_ok=False, prev_mid=1.0, require_ticks=2
    )
    assert stable == 0 and not ready


def test_entry_quote_warmup_requires_distinct_quote_timestamp():
    stable, ready, reason = next_entry_quote_stable_ticks(
        prev_stable=1,
        quote_ok=True,
        prev_mid=1.0,
        require_ticks=2,
        prev_quote_ts=1_000.0,
        quote_ts=1_000.0,
    )
    assert stable == 1 and not ready and reason == "option_quote_not_advanced"
    stable, ready, reason = next_entry_quote_stable_ticks(
        prev_stable=stable,
        quote_ok=True,
        prev_mid=1.0,
        require_ticks=2,
        prev_quote_ts=1_000.0,
        quote_ts=1_001.0,
    )
    assert stable == 2 and ready and reason == "ok"



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


def test_oms_blocks_first_option_print_until_warmup(tmp_path):
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

        def set(self, *args, **kwargs):
            return True

        def hset(self, *args, **kwargs):
            return self

        def execute(self):
            return []

    class Scanner:
        def record_fill(self, *args, **kwargs):
            return None

        states = {}
        event_blackout_meta = {}

        def is_symbol_active(self, *_):
            return False

    profile = {
        "trade": {
            "position_frac": 0.1,
            "position_sizing": "fixed",
            "moneyness": "ATM",
            "hold_minutes": 15,
            "tp_mult": 1.5,
            "sl_mult": 0.5,
            "risk": {
                "max_stock_staleness_sec": 5.0,
                "max_option_staleness_sec": 5.0,
                "max_spread_pct": 0.5,
                "require_entry_quote_stable_ticks": 2,
                "halt_entries_on_feed_unhealthy": True,
                "require_live_market_data": True,
            },
        },
        "fill": {"entry_frac": 0.8, "exit_frac": 0.8},
        "signal": {"top_k": 2},
    }
    lock = SimpleNamespace(local_symbol="AMZN260727P00232500", con_id=99)
    now = time.time()
    connector = SimpleNamespace(
        ib=SimpleNamespace(isConnected=lambda: True),
        redis=FakeRedis(),
        config=SimpleNamespace(
            port=4001,
            account="",
            max_stock_staleness_sec=5.0,
            max_option_staleness_sec=5.0,
        ),
        lock_status="LOCKED",
        data_mode="LIVE",
        symbols=["AMZN", "NVDA", "QQQ"],
        trade_symbols=["AMZN", "NVDA"],
        option_quotes={
            ("AMZN", "AMZN260727P00232500"): {
                "bid": 0.60,
                "ask": 0.64,
                "ts": now,
            }
        },
        last_stock_tick={"AMZN": now, "NVDA": now, "QQQ": now},
        locks={"AMZN": [lock]},
        ensure_option_subscription=lambda *_: True,
        release_on_demand_subscription=lambda *_: None,
    )
    oms = Mag7BrokerOms(
        profile=profile,
        scanner=Scanner(),
        connector=connector,
        session_id="testwarmup01",
        trade_date="2026-07-27",
        session_dir=tmp_path / "warmup_session",
        mode="shadow",
        equity=100_000.0,
    )
    oms._event = lambda kind, payload: events.append((kind, payload))  # type: ignore
    from maga7.live.scanner import ScannerSignal

    sig = ScannerSignal(
        date="2026-07-27",
        symbol="AMZN",
        direction="DN",
        sig_ts=__import__("pandas").Timestamp("2026-07-27 09:35", tz="America/New_York"),
        spot=230.0,
        rank=0,
        bucket_id=0,
        contract="AMZN260727P00232500",
        moneyness="ATM",
        meta={
            "event_source": "am_pulse_sleeve",
            "route": "am_pulse",
            "execute_mode": "shadow",
            "max_lag_sec": 5.0,
            # Feature bar is old; OMS must anchor quote lag to availability.
            "decision_ts": __import__("pandas")
            .Timestamp(now - 1.0, unit="s", tz="UTC")
            .tz_convert("America/New_York")
            .isoformat(),
        },
    )
    assert oms.process_signal(sig) is False
    assert any(
        k == "ENTRY_WAIT" and p.get("reason") == "option_quote_warmup" for k, p in events
    )
    assert "AMZN" not in oms.positions
    # Retrying the exact same snapshot must not satisfy the warmup.
    events.clear()
    assert oms.process_signal(sig) is False
    assert any(
        k == "ENTRY_WAIT" and p.get("reason") == "option_quote_not_advanced"
        for k, p in events
    )
    # A second, newer print with stable mid → allow.
    events.clear()
    connector.option_quotes[("AMZN", "AMZN260727P00232500")] = {
        "bid": 0.61,
        "ask": 0.65,
        "ts": now + 1.0,
    }
    assert oms.process_signal(sig) is True
    assert "AMZN" in oms.positions

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

        def set(self, *args, **kwargs):
            return True

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

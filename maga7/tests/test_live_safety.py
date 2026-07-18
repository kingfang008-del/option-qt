from types import SimpleNamespace

import pandas as pd

from maga7.common.fills import FillSpec
from maga7.common.bar_agg import MultiSymbolMinuteAgg
from maga7.common.replay import simulate_trade
from maga7.common.signals import StreamSignalState
from maga7.live.broker_oms import Mag7BrokerOms
from maga7.live.ibkr_connector import Mag7IbkrConfig, Mag7IbkrConnector
from maga7.live.live_engine import LiveEngineMetrics, Mag7LiveFrameEngine
from maga7.live.live_regime import LiveRegimeGate
from maga7.live.redis_fused import pack_batch, run_keys, unpack_batch
from maga7.live.scanner_state import restore_scanner, scanner_snapshot


class FakeIb:
    def isConnected(self):
        return True


class FakeRedis:
    def __init__(self, armed: bool):
        self.armed = armed

    def hget(self, key, field):
        assert key == "meta:runtime_trading_controls:maga7"
        assert field == "trading_enabled"
        return b"1" if self.armed else b"0"


def bare_oms(*, mode: str, port: int, armed: bool, reconcile_ok: bool = True):
    oms = object.__new__(Mag7BrokerOms)
    oms.mode = mode
    oms.ib = FakeIb()
    oms.connector = SimpleNamespace(
        lock_status="LOCKED",
        data_mode="LIVE",
        config=SimpleNamespace(port=port, account="DU123"),
    )
    oms.redis = FakeRedis(armed)
    oms.trade_date = "2026-07-16"
    oms.profile_hash = "a" * 64
    oms.reconcile_ok = reconcile_ok
    oms.account_ready = True
    return oms


def test_live_gate_requires_all_independent_arms(monkeypatch):
    oms = bare_oms(mode="live", port=4001, armed=True)
    monkeypatch.setenv("MAG7_LIVE_TRADING", "1")
    monkeypatch.setenv("MAG7_LIVE_CONFIRM", "2026-07-16:" + "a" * 12)
    assert oms.live_gate() == (True, "armed")

    monkeypatch.delenv("MAG7_LIVE_TRADING")
    assert oms.live_gate()[1] == "MAG7_LIVE_TRADING_not_enabled"


def test_paper_gate_needs_paper_port_and_reconciliation():
    wrong_port = bare_oms(mode="paper", port=4001, armed=False)
    assert wrong_port.live_gate()[1] == "paper_requires_port_4002"
    stale = bare_oms(mode="paper", port=4002, armed=False, reconcile_ok=False)
    assert stale.live_gate()[1] == "broker_reconcile_failed"


def test_live_qqq_alignment_uses_completed_causal_minute():
    gate = LiveRegimeGate(
        {
            "qqq_align": True,
            "qqq_from_prev_eps": 0.0,
            "block_on_missing": True,
        }
    )
    gate.on_stock_second(
        "QQQ",
        {
            "timestamp": pd.Timestamp(
                "2026-07-16 09:30:00", tz="America/New_York"
            ).timestamp(),
            "open": 99.0,
            "high": 99.0,
            "low": 99.0,
            "close": 99.0,
            "volume": 1.0,
            "previous_close": 100.0,
        },
    )
    gate.on_stock_second(
        "QQQ",
        {
            "timestamp": pd.Timestamp(
                "2026-07-16 09:31:00", tz="America/New_York"
            ).timestamp(),
            "open": 99.0,
            "high": 99.0,
            "low": 99.0,
            "close": 99.0,
            "volume": 1.0,
            "previous_close": 100.0,
        },
    )
    assert gate.check("UP", None).reason == "qqq_align_up"
    assert gate.check("DN", None).allow is True


def test_mf_exit_only_sees_completed_left_labeled_stock_bar():
    times = pd.to_datetime(
        ["2026-07-16 10:31:00", "2026-07-16 10:32:00"]
    ).tz_localize("America/New_York")
    path = pd.DataFrame(
        {
            "timestamp": times,
            "bid": [1.0, 1.0],
            "ask": [1.0, 1.0],
        }
    )
    stock = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2026-07-16 10:30:00", "2026-07-16 10:31:00"]
            ).tz_localize("America/New_York"),
            "mf10": [1.0, -1.0],
            "streak_up": [1, 0],
            "streak_dn": [0, 1],
        }
    )
    result = simulate_trade(
        path,
        times[0],
        fill=FillSpec(),
        tp_mult=2.0,
        sl_mult=0.1,
        direction="UP",
        stock_day=stock,
        exit_mode="mf_flip",
        exit_mf_grace_seconds=0,
        stock_bar_delay_seconds=60,
    )
    assert result is not None
    assert result.reason == "MF_FLIP"
    assert result.exit_ts == times[1]


def test_live_frame_phases_use_current_quote_and_unix_seconds():
    calls = []
    connector = SimpleNamespace(option_quotes={})

    class Scanner:
        states = {"AAPL": object()}

        def on_stock_second(self, symbol, tick):
            assert ("AAPL", "AAPL260717C00100000") in connector.option_quotes
            assert tick["timestamp"].year == 2026
            calls.append("scan")
            return "signal"

    class Oms:
        def evaluate_exits(self, ts):
            calls.append("exit")

        def process_signal(self, signal):
            calls.append("entry")
            return True

    class Pipe:
        def xack(self, *args):
            return self

        def set(self, *args):
            return self

        def execute(self):
            return []

    class Redis:
        def pipeline(self, transaction=True):
            return Pipe()

        def hset(self, *args, **kwargs):
            return 1

        def xack(self, *args):
            return 1

    engine = object.__new__(Mag7LiveFrameEngine)
    engine.redis = Redis()
    engine.session_id = "test"
    engine.keys = run_keys("test")
    engine.scanner = Scanner()
    engine.oms = Oms()
    engine.connector = connector
    engine.consumer_name = "consumer"
    engine.metrics = LiveEngineMetrics()
    engine.seen = set()
    ts = pd.Timestamp(
        "2026-07-16 10:31:00", tz="America/New_York"
    ).timestamp()
    payload = {
        "run_id": "test",
        "frame_id": "test:1",
        "symbol": "AAPL",
        "ts": ts,
        "stock": {
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "volume": 10.0,
        },
        "option_contracts": [
            {
                "localSymbol": "AAPL260717C00100000",
                "bid": 1.0,
                "ask": 1.1,
            }
        ],
    }
    engine._process_message(b"1-0", {b"batch": pack_batch([payload])})
    assert calls == ["scan", "exit", "entry"]
    assert engine.metrics.frames == 1


def test_connector_publishes_completed_second_without_future_option_quote():
    captured = []

    class Redis:
        def xadd(self, stream, fields, **kwargs):
            captured.append(unpack_batch(fields["batch"]))
            return b"1-0"

    connector = object.__new__(Mag7IbkrConnector)
    connector.session_id = "test"
    connector.symbols = ["AAPL"]
    connector.config = Mag7IbkrConfig()
    connector.redis = Redis()
    connector.keys = {"stream": "test"}
    connector.last_published_second = 0
    connector.partial_frame_drops = 0
    connector._partial_targets = set()
    connector.stock_bar_history = {
        "AAPL": {
            100: {
                "ts": 100.0,
                "open": 10.0,
                "high": 11.0,
                "low": 9.0,
                "close": 10.5,
                "volume": 5.0,
            }
        }
    }
    connector.stock_previous_close = {"AAPL": 9.5}
    connector.option_quote_history = {
        ("AAPL", "OLD"): {
            99: {"ts": 99.5, "bid": 1.0, "ask": 1.1},
        },
        ("AAPL", "FUTURE"): {
            101: {"ts": 101.1, "bid": 2.0, "ask": 2.1},
        },
    }
    assert connector.publish_frame(100) == 1
    assert captured[0][0]["stock"]["volume"] == 5.0
    assert [row["localSymbol"] for row in captured[0][0]["option_contracts"]] == [
        "OLD"
    ]


def test_scanner_and_regime_state_round_trip():
    cfg = {
        "mf_window": 10,
        "vol_ma_window": 20,
        "window_start": "10:30",
        "window_end": "14:00",
    }

    def build():
        state = StreamSignalState("AAPL", cfg, emit_all=True)
        state.date = "2026-07-16"
        state.prev_close = 100.0
        state.cum = 123.0
        state.mf10 = -4.0
        state.streak_dn = 3
        state.bars = [
            {
                "timestamp": pd.Timestamp(
                    "2026-07-16 10:30", tz="America/New_York"
                ),
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.5,
                "volume": 10.0,
                "net$": -4.0,
            }
        ]
        gate = LiveRegimeGate({"qqq_align": True})
        gate.qqq_previous_close = 500.0
        gate.qqq_close = 501.0
        return SimpleNamespace(
            states={"AAPL": state},
            minute_agg=MultiSymbolMinuteAgg(["AAPL"]),
            regime_gate=gate,
            current_date="2026-07-16",
            day_fires=[],
            signals=[],
            n_done={"AAPL": 1},
            last_exit={"AAPL": None},
            last_win={"AAPL": True},
        )

    original = build()
    payload = scanner_snapshot(original)
    restored = build()
    restored.states["AAPL"].cum = 0.0
    restored.regime_gate.qqq_close = 0.0
    restore_scanner(restored, payload)
    assert restored.states["AAPL"].cum == 123.0
    assert restored.states["AAPL"].streak_dn == 3
    assert restored.regime_gate.qqq_close == 501.0
    assert restored.n_done == {"AAPL": 1}

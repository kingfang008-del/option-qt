import json
from types import SimpleNamespace
import pickle

import pandas as pd
import pytest

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
from maga7.tools.run_live_session import (
    _consumes_baseline_topk,
    _sanitize_resume_topk,
)


class FakeIb:
    def isConnected(self):
        return True


def test_live_redis_wire_rejects_pickle_payload():
    with pytest.raises(Exception):
        unpack_batch(pickle.dumps([{"run_id": "unsafe"}]))


def test_resume_satellite_signals_do_not_consume_baseline_topk():
    am = SimpleNamespace(
        symbol="NVDA",
        meta={"event_source": "am_pulse_sleeve", "route": "am_pulse"},
    )
    am_ext = SimpleNamespace(
        symbol="AMD",
        meta={
            "event_source": "am_pulse_extension_sleeve",
            "route": "am_pulse_extension",
        },
    )
    baseline = SimpleNamespace(
        symbol="META",
        meta={"event_source": "baseline", "route": "baseline"},
    )
    scanner = SimpleNamespace(
        day_fires=[am, am_ext, baseline],
        day_topk_syms={"NVDA", "AMD", "META"},
        day_hunt_symbols=set(),
        n_done={"NVDA": 2, "AMD": 1, "META": 1},
        states={
            "NVDA": SimpleNamespace(
                fired_today=True,
                first_fire={"symbol": "NVDA", "dir": "DN"},
            ),
            "AMD": SimpleNamespace(
                fired_today=True,
                first_fire={"symbol": "AMD", "dir": "DN"},
            ),
            "META": SimpleNamespace(
                fired_today=True,
                first_fire={"symbol": "META", "dir": "DN"},
            ),
        },
    )

    assert not _consumes_baseline_topk(am)
    assert not _consumes_baseline_topk(am_ext)
    assert _consumes_baseline_topk(baseline)
    assert _sanitize_resume_topk(scanner) == 2
    assert scanner.day_fires == [baseline]
    assert scanner.day_topk_syms == {"META"}
    assert scanner.n_done == {"NVDA": 0, "AMD": 0, "META": 1}
    assert scanner.states["NVDA"].fired_today is False
    assert scanner.states["NVDA"].first_fire is None
    assert scanner.states["AMD"].fired_today is False
    assert scanner.states["AMD"].first_fire is None


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


def test_connector_publishes_completed_second_without_future_option_quote(tmp_path):
    captured = []

    class Redis:
        def xadd(self, stream, fields, **kwargs):
            captured.append((stream, unpack_batch(fields["batch"])))
            return b"1-0"

    frame_ts = int(
        pd.Timestamp("2026-07-16 10:30:00", tz="America/New_York").timestamp()
    )
    connector = object.__new__(Mag7IbkrConnector)
    connector.session_id = "test"
    connector.trade_date = "2026-07-16"
    connector.session_dir = tmp_path
    connector.symbols = ["AAPL"]
    connector.config = Mag7IbkrConfig()
    connector.redis = Redis()
    connector.keys = {
        "stream": "test:rth",
        "stream_pre": "test:pre",
        "stream_post": "test:post",
    }
    connector.last_published_second = 0
    connector.last_validation_second = 0
    connector.validation_publishes = 0
    connector.tape_writes = 0
    connector.partial_frame_drops = 0
    connector._partial_targets = set()
    connector.stock_bar_history = {
        "AAPL": {
            frame_ts: {
                "ts": float(frame_ts),
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
            frame_ts - 1: {"ts": float(frame_ts) - 0.5, "bid": 1.0, "ask": 1.1},
        },
        ("AAPL", "FUTURE"): {
            frame_ts + 1: {"ts": float(frame_ts) + 1.1, "bid": 2.0, "ask": 2.1},
        },
    }
    assert connector.publish_frame(frame_ts) == 1
    assert captured[0][0] == "test:rth"
    assert captured[0][1][0]["stock"]["volume"] == 5.0
    assert [row["localSymbol"] for row in captured[0][1][0]["option_contracts"]] == [
        "OLD"
    ]
    tape = tmp_path / "tape" / "rth" / "AAPL_2026-07-16.jsonl"
    assert tape.is_file()
    option_tape = (
        tmp_path / "tape" / "rth" / "options" / "AAPL_2026-07-16.jsonl"
    )
    assert option_tape.is_file()
    option_row = json.loads(option_tape.read_text(encoding="utf-8").splitlines()[0])
    assert [quote["localSymbol"] for quote in option_row["quotes"]] == ["OLD"]
    assert connector.option_tape_frames == 1
    assert connector.option_tape_quotes == 1


def test_connector_partial_second_is_not_compressed_into_recovery(tmp_path):
    captured = []

    class Redis:
        def xadd(self, stream, fields, **kwargs):
            captured.append(unpack_batch(fields["batch"]))
            return b"1-0"

    first = int(
        pd.Timestamp("2026-07-16 10:30:00", tz="America/New_York").timestamp()
    )
    connector = object.__new__(Mag7IbkrConnector)
    connector.session_id = "test_partial"
    connector.trade_date = "2026-07-16"
    connector.session_dir = tmp_path
    connector.symbols = ["AAPL", "MSFT"]
    connector.config = Mag7IbkrConfig()
    connector.redis = Redis()
    connector.keys = {"stream": "rth", "stream_pre": "pre", "stream_post": "post"}
    connector.last_published_second = 0
    connector.last_validation_second = 0
    connector.validation_publishes = 0
    connector.tape_writes = 0
    connector.partial_frame_drops = 0
    connector._partial_targets = set()
    connector.stock_previous_close = {}
    connector.option_quote_history = {}
    connector.stock_bar_history = {
        "AAPL": {
            first: {
                "open": 10.0,
                "high": 11.0,
                "low": 9.0,
                "close": 10.5,
                "volume": 100.0,
            },
            first + 1: {
                "open": 10.5,
                "high": 10.7,
                "low": 10.4,
                "close": 10.6,
                "volume": 2.0,
            },
        },
        "MSFT": {
            first + 1: {
                "open": 20.0,
                "high": 20.2,
                "low": 19.9,
                "close": 20.1,
                "volume": 3.0,
            }
        },
    }

    assert connector.publish_frame(first) == 0
    assert connector.last_published_second == first
    assert connector.publish_frame(first + 1) == 2
    aapl = next(row for row in captured[0] if row["symbol"] == "AAPL")
    assert aapl["stock"]["open"] == 10.5
    assert aapl["stock"]["volume"] == 2.0


def test_connector_does_not_publish_authority_frame_while_disconnected(tmp_path):
    frame_ts = int(
        pd.Timestamp("2026-07-16 10:30:00", tz="America/New_York").timestamp()
    )
    connector = object.__new__(Mag7IbkrConnector)
    connector.ib = SimpleNamespace(isConnected=lambda: False)
    connector.data_mode = "LIVE"
    connector.session_id = "test_disconnected"
    connector.trade_date = "2026-07-16"
    connector.session_dir = tmp_path
    connector.symbols = ["AAPL"]
    connector.config = Mag7IbkrConfig()
    connector.last_published_second = 0
    connector.last_validation_second = 0
    connector.stock_bar_history = {"AAPL": {}}
    connector.option_quote_history = {}

    assert connector.publish_frame(frame_ts) == 0
    assert connector.last_published_second == frame_ts


def test_connector_premarket_partial_goes_to_pre_stream(tmp_path):
    captured = []

    class Redis:
        def xadd(self, stream, fields, **kwargs):
            captured.append((stream, unpack_batch(fields["batch"])))
            return b"1-0"

    frame_ts = int(
        pd.Timestamp("2026-07-16 09:15:00", tz="America/New_York").timestamp()
    )
    connector = object.__new__(Mag7IbkrConnector)
    connector.session_id = "test_pre"
    connector.trade_date = "2026-07-16"
    connector.session_dir = tmp_path
    connector.symbols = ["AAPL", "MSFT"]
    connector.config = Mag7IbkrConfig()
    connector.redis = Redis()
    connector.keys = {
        "stream": "test:rth",
        "stream_pre": "test:pre",
        "stream_post": "test:post",
    }
    connector.last_published_second = 0
    connector.last_validation_second = 0
    connector.validation_publishes = 0
    connector.tape_writes = 0
    connector.partial_frame_drops = 0
    connector._partial_targets = set()
    connector.stock_bar_history = {
        "AAPL": {
            frame_ts: {
                "ts": float(frame_ts),
                "open": 10.0,
                "high": 11.0,
                "low": 9.0,
                "close": 10.5,
                "volume": 1.0,
            }
        }
    }
    connector.stock_previous_close = {"AAPL": 9.5}
    connector.option_quote_history = {}
    assert connector.publish_frame(frame_ts) == 1
    assert captured[0][0] == "test:pre"
    assert connector.validation_publishes == 1
    assert connector.partial_frame_drops == 0
    assert (tmp_path / "tape" / "pre" / "AAPL_2026-07-16.jsonl").is_file()
    assert not (tmp_path / "tape" / "rth").exists()


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
        stock_by = {
            "AAPL": pd.DataFrame(
                [
                    {
                        "timestamp": pd.Timestamp(
                            "2026-07-16 09:45", tz="America/New_York"
                        ),
                        "date": "2026-07-16",
                        "open": 100.0,
                        "high": 101.0,
                        "low": 99.0,
                        "close": 100.5,
                        "volume": 10.0,
                        "mf10": -4.0,
                        "streak_dn": 2,
                    }
                ]
            ),
            "QQQ": pd.DataFrame(
                [
                    {
                        "timestamp": pd.Timestamp(
                            "2026-07-16 09:45", tz="America/New_York"
                        ),
                        "date": "2026-07-16",
                        "open": 500.0,
                        "high": 501.0,
                        "low": 499.0,
                        "close": 500.5,
                        "volume": 100.0,
                        "mf10": 1.0,
                    }
                ]
            ),
        }
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
            stock_by=stock_by,
            stock_by_frozen=False,
        )

    original = build()
    payload = scanner_snapshot(original)
    assert "AAPL" in payload["stock_by"]
    assert "QQQ" in payload["stock_by"]
    assert payload["stock_by"]["AAPL"][0]["mf10"] == -4.0
    restored = build()
    restored.states["AAPL"].cum = 0.0
    restored.regime_gate.qqq_close = 0.0
    restored.stock_by = {}
    restore_scanner(restored, payload)
    assert restored.states["AAPL"].cum == 123.0
    assert restored.states["AAPL"].streak_dn == 3
    assert restored.regime_gate.qqq_close == 501.0
    assert restored.n_done == {"AAPL": 1}
    assert len(restored.stock_by["AAPL"]) == 1
    assert float(restored.stock_by["AAPL"].iloc[0]["mf10"]) == -4.0
    assert "QQQ" in restored.stock_by

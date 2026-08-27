"""Step4 shadow wire: am_v2 launch drain emits Mag7 satellite ScannerSignal."""
from __future__ import annotations

import pandas as pd

from maga7.common.am_v2_sleeve import AmV2LaunchTracker, am_v2_enabled, load_am_v2_cfg
from maga7.common.config import load_profile
from maga7.common.entry_contract import ContractBooks
from maga7.common.signals import StreamSignalState
from maga7.live.broker_oms import Mag7BrokerOms
from maga7.live.scanner import Mag7Scanner, ScannerSignal
from maga7.live.scanner_state import scanner_snapshot


def test_peer3_has_am_v2_shadow_block():
    profile = load_profile(
        "maga7/CONFIG/strategy_profiles/"
        "single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1.json"
    )
    assert am_v2_enabled(profile)
    cfg = load_am_v2_cfg(profile)
    assert cfg["execute_mode"] == "shadow"
    assert cfg["abs_ret_min"] == 0.0025
    assert cfg["cooldown_sec"] == 300
    assert cfg["tp"] == 0.15
    assert cfg["sl"] == 0.25
    assert cfg["window_start"] == "10:00"


def test_launch_tracker_rising_edge_up():
    day = "2026-07-24"
    tr = AmV2LaunchTracker(slope_sec=3, abs_ret_min=0.002, cooldown_sec=1)
    closes = [100.0] * 5 + [100.0, 100.05, 100.15, 100.35]
    fired = None
    for i, c in enumerate(closes):
        ts = pd.Timestamp(f"{day} 10:10:00", tz="America/New_York") + pd.Timedelta(seconds=i)
        a = tr.on_close(ts, c)
        if a is not None:
            fired = a
    assert fired is not None
    assert fired.dir == "UP"
    assert fired.ret_k >= 0.002


def _profile(day: str = "2026-07-24") -> dict:
    return {
        "symbols": ["TSLA", "NVDA"],
        "signal": {"top_k": 2, "mf_window": 10, "vol_ma_window": 20},
        "trade": {
            "moneyness": "ATM",
            "hold_minutes": 30,
            "tp_mult": 1.6,
            "sl_mult": 0.4,
            "contract_mode": "day_lock",
        },
        "fill": {"entry_frac": 0.75, "exit_frac": 0.75},
        "paths": {},
        "date_range": {"start": day, "end": day},
        "am_v2": {
            "enabled": True,
            "execute_mode": "shadow",
            "wired": True,
            "window_start": "10:00",
            "window_end": "11:30",
            "flatten_before": "11:45",
            "slope_sec": 3,
            "abs_ret_min": 0.002,
            "cooldown_sec": 300,
            "peak_lookback_sec": 60,
            "dirs": ["UP", "DN"],
            "tp": 0.15,
            "sl": 0.25,
            "max_hold_sec": 900,
            "max_lag_sec": 5.0,
            "max_spread_pct": 0.15,
            "min_mid": 0.05,
            "entry_frac": 0.75,
            "exit_frac": 0.75,
            "position_frac": 0.10,
            "moneyness": "ATM",
            "prefer_dte": 0,
            "allowed_dte": [0, 1, 2],
        },
        "lock": {
            "dte_mode": "trading",
            "prefer_dte": 0,
            "allowed_dte": [0, 1, 2],
            "otm_rungs": 3,
        },
    }


def test_drain_am_v2_emits_launch_up_once():
    day = "2026-07-24"
    profile = _profile(day)
    ticker = "TSLA260724C00320000"
    books = ContractBooks(
        mode="day_lock",
        # UP/ATM → bucket 2
        flat_idx={("TSLA", day): {2: ticker}},
        prefer_dte=0,
        allowed_dte=[0, 1, 2],
    )
    sc = Mag7Scanner(
        profile=profile,
        states={
            "TSLA": StreamSignalState("TSLA", profile["signal"]),
            "NVDA": StreamSignalState("NVDA", profile["signal"]),
        },
        books=books,
        stock_by={},
    )
    base = 320.0
    start = pd.Timestamp(f"{day} 10:05:00", tz="America/New_York")
    # Warm-up flat seconds
    for i in range(8):
        sc.on_stock_second(
            "TSLA",
            {
                "timestamp": start + pd.Timedelta(seconds=i),
                "close": base,
                "volume": 100,
            },
        )
    # Impulse: +0.35% over 3s
    impulse = [base, base * 1.001, base * 1.002, base * 1.0035]
    for i, px in enumerate(impulse):
        sc.on_stock_second(
            "TSLA",
            {
                "timestamp": start + pd.Timedelta(seconds=8 + i),
                "close": px,
                "volume": 200,
            },
        )

    sigs = [s for s in sc.signals if (s.meta or {}).get("event_source") == "am_v2_sleeve"]
    assert len(sigs) >= 1
    sig = sigs[0]
    assert sig.symbol == "TSLA" and sig.direction == "UP"
    assert sig.contract == ticker
    assert sig.meta.get("route") == "am_v2"
    assert sig.meta.get("execute_mode") == "shadow"
    assert sig.meta.get("exit_simple") is True
    assert sig.meta.get("exit_flatten_before") == "11:45"
    assert abs(float(sig.meta.get("exit_tp_mult")) - 1.15) < 1e-9
    assert abs(float(sig.meta.get("exit_sl_mult")) - 0.75) < 1e-9
    assert Mag7BrokerOms._signal_source_fields(sig)["event_source"] == "am_v2_sleeve"
    assert sc.day_fires == []
    snap = scanner_snapshot(sc)
    assert snap["am_v2"]["enabled"] is True
    assert snap["am_v2"]["n_emitted"] >= 1
    assert snap["am_v2"]["execute_mode"] == "shadow"


def test_oms_rejects_am_v2_shadow_on_paper():
    sig = ScannerSignal(
        date="2026-07-24",
        symbol="TSLA",
        direction="UP",
        sig_ts=pd.Timestamp("2026-07-24 10:10", tz="America/New_York"),
        spot=320.0,
        rank=0,
        bucket_id=2,
        contract="TSLA260724C00320000",
        moneyness="ATM",
        meta={
            "event_source": "am_v2_sleeve",
            "route": "am_v2",
            "execute_mode": "shadow",
            "exit_simple": True,
            "position_frac": 0.1,
        },
    )
    oms = Mag7BrokerOms.__new__(Mag7BrokerOms)
    oms.mode = "paper"
    oms.positions = {}
    oms.day_halted = False
    oms.events = []
    oms.scanner = None
    oms.risk_cfg = type("R", (), {"halt_entries_on_gap": False})()
    oms._has_active_buy = lambda _s: False  # type: ignore
    oms.has_position = lambda _s: False  # type: ignore
    oms._event = lambda kind, payload: oms.events.append((kind, payload))  # type: ignore
    assert oms.process_signal(sig) is False
    assert any(
        k == "ENTRY_REJECT" and p.get("reason") == "am_v2_shadow_only"
        for k, p in oms.events
    )


def test_am_v2_disabled_emits_nothing():
    day = "2026-07-24"
    profile = _profile(day)
    profile["am_v2"]["enabled"] = False
    books = ContractBooks(
        mode="day_lock",
        flat_idx={("TSLA", day): {0: "TSLA260724C00320000"}},
        prefer_dte=0,
        allowed_dte=[0, 1, 2],
    )
    sc = Mag7Scanner(
        profile=profile,
        states={"TSLA": StreamSignalState("TSLA", profile["signal"])},
        books=books,
        stock_by={},
    )
    start = pd.Timestamp(f"{day} 10:05:00", tz="America/New_York")
    for i in range(20):
        px = 100.0 * (1.0 + 0.01 * max(0, i - 10))
        sc.on_stock_second(
            "TSLA",
            {"timestamp": start + pd.Timedelta(seconds=i), "close": px, "volume": 1},
        )
    assert not [
        s for s in sc.signals if (s.meta or {}).get("event_source") == "am_v2_sleeve"
    ]

"""Shadow wire: am_pulse drain emits Mag7 satellite ScannerSignal."""
from __future__ import annotations

import pandas as pd

from maga7.common.am_pulse_scout import am_pulse_enabled, load_am_pulse_cfg
from maga7.common.entry_contract import ContractBooks
from maga7.common.signals import StreamSignalState
from maga7.live.broker_oms import Mag7BrokerOms, LivePosition
from maga7.live.scanner import Mag7Scanner, ScannerSignal
from maga7.live.scanner_state import scanner_snapshot


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
        "am_pulse": {
            "enabled": True,
            "execute_mode": "shadow",
            "arm": "FO",
            "dirs": ["DN"],
            "window_start": "09:30",
            "window_end": "10:25",
            "flatten_before": "10:30",
            "min_fav_from_open": 0.01,
            "lookback_bars": 2,
            "min_lookback_ret": 0.99,
            "tp": 0.15,
            "sl": 0.20,
            "max_hold_sec": 900,
            "position_frac": 0.10,
        },
        "lock": {"dte_mode": "trading", "prefer_dte": 0, "allowed_dte": [0, 1, 2], "otm_rungs": 3},
    }


def test_load_am_pulse_cfg_defaults():
    cfg = load_am_pulse_cfg({"am_pulse": {"enabled": True}})
    assert am_pulse_enabled({"am_pulse": {"enabled": True}})
    assert cfg["execute_mode"] == "shadow"
    assert cfg["window_end"] == "10:25"
    assert cfg["flatten_before"] == "10:30"


def test_drain_am_pulse_emits_fo_dn_once():
    day = "2026-07-24"
    profile = _profile(day)
    assert am_pulse_enabled(profile)
    # Flat lock map: TSLA DN ATM put for the day
    ticker = "TSLA260724P00320000"
    books = ContractBooks(
        mode="day_lock",
        flat_idx={( "TSLA", day): {0: ticker}},
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
    # Grind DN from 320 → cross -1% fo around bar ~10
    open_px = 320.0
    for i in range(15):
        ts = pd.Timestamp(f"{day} 09:30:00", tz="America/New_York") + pd.Timedelta(minutes=i)
        px = open_px - i * 0.40
        bar = {
            "timestamp": ts,
            "open": open_px if i == 0 else px + 0.2,
            "high": max(open_px, px) + 0.05,
            "low": px - 0.1,
            "close": px,
            "volume": 1000,
            "symbol": "TSLA",
        }
        sc.on_stock_bar("TSLA", bar)

    pulses = [s for s in sc.signals if (s.meta or {}).get("event_source") == "am_pulse_sleeve"]
    assert len(pulses) == 1
    sig = pulses[0]
    assert sig.symbol == "TSLA" and sig.direction == "DN"
    assert sig.contract == ticker
    assert sig.meta.get("route") == "am_pulse"
    assert sig.meta.get("execute_mode") == "shadow"
    assert sig.meta.get("exit_simple") is True
    assert sig.meta.get("exit_flatten_before") == "10:30"
    assert abs(float(sig.meta.get("position_frac")) - 0.10) < 1e-12
    # hold capped before CORE
    assert float(sig.meta.get("exit_hold_sec")) <= (
        pd.Timestamp(f"{day} 10:30", tz="America/New_York") - sig.sig_ts
    ).total_seconds() + 1
    assert Mag7BrokerOms._signal_source_fields(sig)["event_source"] == "am_pulse_sleeve"
    # once/symbol/day
    late = pd.Timestamp(f"{day} 10:00:00", tz="America/New_York")
    sc.on_stock_bar(
        "TSLA",
        {
            "timestamp": late,
            "open": 300.0,
            "high": 300.0,
            "low": 295.0,
            "close": 296.0,
            "volume": 1,
            "symbol": "TSLA",
        },
    )
    pulses2 = [s for s in sc.signals if (s.meta or {}).get("event_source") == "am_pulse_sleeve"]
    assert len(pulses2) == 1
    snap = scanner_snapshot(sc)
    assert snap["am_pulse"]["enabled"] is True
    assert snap["am_pulse"]["n_emitted"] == 1
    assert snap["am_pulse"]["execute_mode"] == "shadow"


def test_oms_rejects_am_pulse_shadow_on_paper():
    sig = ScannerSignal(
        date="2026-07-24",
        symbol="TSLA",
        direction="DN",
        sig_ts=pd.Timestamp("2026-07-24 09:44", tz="America/New_York"),
        spot=317.0,
        rank=0,
        bucket_id=0,
        contract="TSLA260724P00320000",
        moneyness="ATM",
        meta={
            "event_source": "am_pulse_sleeve",
            "route": "am_pulse",
            "execute_mode": "shadow",
            "exit_simple": True,
            "position_frac": 0.1,
        },
    )
    # Minimal OMS: mode paper must reject shadow-only sleeve
    class _Dummy:
        pass

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
    assert any(k == "ENTRY_REJECT" for k, _ in oms.events)


def test_flatten_before_core_reason():
    pos = LivePosition(
        symbol="TSLA",
        contract="X",
        con_id=1,
        direction="DN",
        qty=1,
        entry_price=1.0,
        entry_ts=1.0,
        signal_ts=1.0,
        rank=0,
        qty_frac=0.1,
        entry_bid=1.0,
        entry_ask=1.05,
        exit_simple=True,
        exit_flatten_before="10:30",
        exit_tp_mult=1.15,
        exit_sl_mult=0.80,
        exit_hold_sec=900,
    )
    # 10:30 NY
    asof = pd.Timestamp("2026-07-24 10:30:00", tz="America/New_York").timestamp()
    ny = pd.Timestamp(float(asof), unit="s", tz="UTC").tz_convert("America/New_York")
    parts = str(pos.exit_flatten_before).split(":")
    flat_m = int(parts[0]) * 60 + int(parts[1])
    assert (ny.hour * 60 + ny.minute) >= flat_m

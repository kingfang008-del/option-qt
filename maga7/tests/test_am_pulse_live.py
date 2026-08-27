"""Shadow wire: am_pulse drain emits Mag7 satellite ScannerSignal."""
from __future__ import annotations

import pandas as pd

from maga7.common.am_pulse_scout import am_pulse_enabled, load_am_pulse_cfg
from maga7.common.config import load_profile
from maga7.common.entry_contract import ContractBooks
from maga7.common.signals import StreamSignalState
from maga7.live.broker_oms import Mag7BrokerOms, LivePosition
from maga7.live.scanner import Mag7Scanner, ScannerSignal
from maga7.live.scanner_state import restore_scanner, scanner_snapshot


def test_active_spine_has_shadow_optimization_parameters():
    profile = load_profile(
        "maga7/CONFIG/strategy_profiles/"
        "single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1.json"
    )
    assert profile["am_pulse"]["profit_protect"] == {
        "enabled": True,
        "arm_ret": 0.08,
        "floor_ret": 0.03,
        "note": (
            "Shadow candidate ladder_08_03; "
            "research_am_A_lock_profit_protect_20260728"
        ),
    }
    assert profile["am_pulse_extension"]["confirm_abort"]["abort_thr"] == 0.10
    assert profile["am_pulse"]["event_calendar_block"] is False
    assert profile["am_pulse_extension"]["event_calendar_block"] is False
    assert profile["am_pulse"]["entry_stock_drift_gate"] == {
        "enabled": True,
        "max_chase": 0.003,
        "max_reversal": 0.0015,
    }


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
            "dirs": ["DN", "UP"],
            "window_start": "09:30",
            "window_end": "10:30",
            "flatten_before": "10:45",
            "min_fav_from_open": 0.01,
            "lookback_bars": 2,
            "min_lookback_ret": 0.99,
            "tp": 0.15,
            "sl": 0.20,
            "max_hold_sec": 900,
            "max_lag_sec": 5.0,
            "max_spread_pct": 0.15,
            "min_mid": 0.05,
            "entry_stock_drift_gate": {
                "enabled": True,
                "max_chase": 0.003,
                "max_reversal": 0.0015,
            },
            "position_frac": 0.10,
            "profit_protect": {
                "enabled": True,
                "arm_ret": 0.08,
                "floor_ret": 0.03,
            },
        },
        "am_pulse_extension": {
            "enabled": True,
            "execute_mode": "shadow",
            "arm": "FO",
            "dirs": ["DN", "UP"],
            "window_start": "10:30",
            "window_end": "11:30",
            "flatten_before": "11:45",
            "min_fav_from_open": 0.008,
            "lookback_bars": 2,
            "min_lookback_ret": 0.99,
            "tp": 0.15,
            "sl": 0.20,
            "max_hold_sec": 900,
            "max_lag_sec": 5.0,
            "max_spread_pct": 0.15,
            "min_mid": 0.05,
            "entry_stock_drift_gate": {
                "enabled": True,
                "max_chase": 0.003,
                "max_reversal": 0.0015,
            },
            "position_frac": 0.10,
            "prefer_dte": 0,
            "allowed_dte": [0],
            "confirm_abort": {
                "enabled": True,
                "confirm_sec": 60,
                "confirm_thr": 0.02,
                "abort_thr": 0.10,
                "on_timeout": "abort",
                "only_entry_before": None,
                "only_dirs": ["UP"],
            },
        },
        "lock": {"dte_mode": "trading", "prefer_dte": 0, "allowed_dte": [0, 1, 2], "otm_rungs": 3},
    }


def test_load_am_pulse_cfg_defaults():
    cfg = load_am_pulse_cfg({"am_pulse": {"enabled": True}})
    assert am_pulse_enabled({"am_pulse": {"enabled": True}})
    assert cfg["execute_mode"] == "shadow"
    assert cfg["window_end"] == "10:30"
    assert cfg["flatten_before"] == "10:45"


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
    # Macro event dates are a CORE baseline gate, not an A/B sleeve halt.
    sc.set_event_blackout({day}, {"active_today": True})
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
    assert sig.meta.get("exit_flatten_before") == "10:45"
    assert sig.meta.get("max_lag_sec") == 5.0
    assert sig.meta.get("min_mid") == 0.05
    assert sig.meta.get("entry_stock_drift_gate") == {
        "enabled": True,
        "max_chase": 0.003,
        "max_reversal": 0.0015,
    }
    assert (
        pd.Timestamp(sig.meta["decision_ts"])
        - pd.Timestamp(sig.meta["feature_ts"])
    ).total_seconds() == 60
    assert sig.meta.get("profit_protect") == {
        "enabled": True,
        "arm_ret": 0.08,
        "floor_ret": 0.03,
    }
    assert abs(float(sig.meta.get("position_frac")) - 0.10) < 1e-12
    # hold capped before CORE
    assert float(sig.meta.get("exit_hold_sec")) <= (
        pd.Timestamp(f"{day} 10:45", tz="America/New_York") - sig.sig_ts
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


def test_original_and_extension_have_independent_symbol_budget_and_topk():
    day = "2026-07-24"
    profile = _profile(day)
    profile["signal"].update({"window_start": "14:00", "window_end": "14:01"})
    ticker = "TSLA260724P00320000"
    books = ContractBooks(
        mode="day_lock",
        flat_idx={("TSLA", day): {0: ticker}},
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
    open_px = 320.0
    start = pd.Timestamp(f"{day} 09:30", tz="America/New_York")
    for i in range(121):
        ts = start + pd.Timedelta(minutes=i)
        px = open_px * (1.0 - min(0.02, i * 0.0002))
        sc.on_stock_bar(
            "TSLA",
            {
                "timestamp": ts,
                "open": open_px if i == 0 else px + 0.02,
                "high": max(open_px if i == 0 else px + 0.02, px),
                "low": px - 0.02,
                "close": px,
                "volume": 1000,
                "symbol": "TSLA",
            },
        )
    am = [
        s
        for s in sc.signals
        if (s.meta or {}).get("event_source") == "am_pulse_sleeve"
    ]
    ext = [
        s
        for s in sc.signals
        if (s.meta or {}).get("event_source") == "am_pulse_extension_sleeve"
    ]
    assert len(am) == 1
    assert len(ext) == 1
    assert ext[0].meta["route"] == "am_pulse_extension"
    assert ext[0].meta["watchdog_reason"] == "am_pulse_extension"
    assert ext[0].meta["exit_flatten_before"] == "11:45"
    assert ext[0].meta["confirm_abort"]["enabled"] is True
    assert ext[0].meta["confirm_abort"]["confirm_sec"] == 60
    assert ext[0].meta["confirm_abort"]["abort_thr"] == 0.10
    assert "confirm_abort" not in (am[0].meta or {})
    assert "profit_protect" not in (ext[0].meta or {})
    assert sc.n_am_pulse_signals == sc.n_am_pulse_extension_signals == 1
    assert sc.day_fires == []
    assert sc.day_topk_syms == set()


def test_extension_pending_counters_and_detector_restore():
    day = "2026-07-24"
    profile = _profile(day)

    def build() -> Mag7Scanner:
        return Mag7Scanner(
            profile=profile,
            states={
                "TSLA": StreamSignalState("TSLA", profile["signal"]),
                "NVDA": StreamSignalState("NVDA", profile["signal"]),
            },
            books=ContractBooks(
                mode="day_lock",
                flat_idx={("TSLA", day): {0: "TSLA260724P00320000"}},
                prefer_dte=0,
                allowed_dte=[0, 1, 2],
            ),
            stock_by={},
        )

    sc = build()
    sc._roll_day(day)
    for hhmm, px in (("09:30", 320.0), ("10:30", 316.0)):
        sc._feed_am_pulse_lane_bar(
            "am_pulse_extension",
            "TSLA",
            {
                "timestamp": pd.Timestamp(f"{day} {hhmm}", tz="America/New_York"),
                "open": 320.0,
                "high": 320.0,
                "low": px,
                "close": px,
            },
        )
    assert sc.n_am_pulse_extension_signals == 1
    assert len(sc.pending_am_pulse_extension) == 1
    payload = scanner_snapshot(sc)
    restored = build()
    restore_scanner(restored, payload)
    assert restored.n_am_pulse_extension_signals == 1
    assert len(restored.pending_am_pulse_extension) == 1
    assert restored._am_pulse_extension_scout_date == day
    assert restored._am_pulse_extension_scout._alerts_n[("TSLA", "FO")] == 1
    out = restored.drain_am_pulse_extension(
        pd.Timestamp(f"{day} 10:31", tz="America/New_York")
    )
    assert len(out) == 1
    assert out[0].meta["event_source"] == "am_pulse_extension_sleeve"


def test_oms_rejects_am_pulse_extension_shadow_on_live():
    sig = ScannerSignal(
        date="2026-07-24",
        symbol="TSLA",
        direction="DN",
        sig_ts=pd.Timestamp("2026-07-24 10:30", tz="America/New_York"),
        spot=316.0,
        rank=0,
        bucket_id=0,
        contract="TSLA260724P00320000",
        moneyness="ATM",
        meta={
            "event_source": "am_pulse_extension_sleeve",
            "route": "am_pulse_extension",
            "execute_mode": "shadow",
        },
    )
    oms = Mag7BrokerOms.__new__(Mag7BrokerOms)
    oms.mode = "live"
    oms.positions = {}
    oms.day_halted = False
    oms.events = []
    oms.scanner = None
    oms.risk_cfg = type("R", (), {"halt_entries_on_gap": False})()
    oms._has_active_buy = lambda _s: False  # type: ignore
    oms.has_position = lambda _s: False  # type: ignore
    oms._event = lambda kind, payload: oms.events.append((kind, payload))  # type: ignore
    assert oms.process_signal(sig) is False
    assert oms.events[-1][1]["event_source"] == "am_pulse_extension_sleeve"


def test_am_pulse_no_atm_waits_then_emits_when_books_ready():
    day = "2026-07-24"
    profile = _profile(day)
    books_empty = ContractBooks(
        mode="day_lock",
        flat_idx={},
        prefer_dte=0,
        allowed_dte=[0, 1, 2],
    )
    sc = Mag7Scanner(
        profile=profile,
        states={
            "TSLA": StreamSignalState("TSLA", profile["signal"]),
            "NVDA": StreamSignalState("NVDA", profile["signal"]),
        },
        books=books_empty,
        stock_by={},
    )
    open_px = 320.0
    for i in range(12):
        ts = pd.Timestamp(f"{day} 09:30:00", tz="America/New_York") + pd.Timedelta(minutes=i)
        px = open_px - i * 0.40
        sc.on_stock_bar(
            "TSLA",
            {
                "timestamp": ts,
                "open": open_px if i == 0 else px + 0.2,
                "high": max(open_px, px) + 0.05,
                "low": px - 0.1,
                "close": px,
                "volume": 1000,
                "symbol": "TSLA",
            },
        )
    assert sc.n_am_pulse_signals >= 1
    assert sc.n_am_pulse_emitted == 0
    assert sc.n_am_pulse_wait >= 1
    assert sc.pending_am_pulse
    assert any(e.get("reason") == "no_atm_contract" for e in sc.am_pulse_skip_events)

    # Locks arrive — next bar should emit instead of dropping the alert.
    sc.books = ContractBooks(
        mode="day_lock",
        flat_idx={("TSLA", day): {0: "TSLA260724P00320000"}},
        prefer_dte=0,
        allowed_dte=[0, 1, 2],
    )
    late = pd.Timestamp(f"{day} 09:45:00", tz="America/New_York")
    sc.on_stock_bar(
        "TSLA",
        {
            "timestamp": late,
            "open": 315.0,
            "high": 315.0,
            "low": 314.0,
            "close": 314.5,
            "volume": 1,
            "symbol": "TSLA",
        },
    )
    pulses = [s for s in sc.signals if (s.meta or {}).get("event_source") == "am_pulse_sleeve"]
    assert len(pulses) == 1
    assert sc.n_am_pulse_emitted == 1
    assert sc.pending_am_pulse == []


def test_am_pulse_past_flatten_permanent_skip():
    day = "2026-07-24"
    profile = _profile(day)
    sc = Mag7Scanner(
        profile=profile,
        states={
            "TSLA": StreamSignalState("TSLA", profile["signal"]),
            "NVDA": StreamSignalState("NVDA", profile["signal"]),
        },
        books=ContractBooks(mode="day_lock", flat_idx={}, prefer_dte=0, allowed_dte=[0, 1, 2]),
        stock_by={},
    )
    open_px = 320.0
    for i in range(12):
        ts = pd.Timestamp(f"{day} 09:30:00", tz="America/New_York") + pd.Timedelta(minutes=i)
        px = open_px - i * 0.40
        sc.on_stock_bar(
            "TSLA",
            {
                "timestamp": ts,
                "open": open_px if i == 0 else px + 0.2,
                "high": max(open_px, px) + 0.05,
                "low": px - 0.1,
                "close": px,
                "volume": 1000,
                "symbol": "TSLA",
            },
        )
    assert sc.pending_am_pulse
    # Past flatten_before — permanent skip, pending cleared.
    sc.on_stock_bar(
        "TSLA",
        {
            "timestamp": pd.Timestamp(f"{day} 10:45:00", tz="America/New_York"),
            "open": 300.0,
            "high": 300.0,
            "low": 299.0,
            "close": 299.5,
            "volume": 1,
            "symbol": "TSLA",
        },
    )
    assert sc.pending_am_pulse == []
    assert sc.n_am_pulse_skip >= 1
    assert any(e.get("reason") == "past_flatten" for e in sc.am_pulse_skip_events)


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

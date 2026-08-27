"""Shadow wire: qqq_open_cont drain emits satellite ScannerSignal."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from maga7.common.entry_contract import ContractBooks
from maga7.common.qqq_open_cont import open_cont_enabled, signal_from_open_spot
from maga7.common.signals import StreamSignalState
from maga7.live.broker_oms import Mag7BrokerOms
from maga7.live.scanner import Mag7Scanner, ScannerSignal
from maga7.live.scanner_state import scanner_snapshot


def test_signal_from_open_spot_threshold():
    ts = pd.Timestamp("2026-07-13 09:45", tz="America/New_York")
    assert signal_from_open_spot(
        date="2026-07-13", open_px=100.0, spot=100.1, entry_ts=ts, from_open_min=0.002
    ) is None
    sig = signal_from_open_spot(
        date="2026-07-13", open_px=100.0, spot=100.25, entry_ts=ts, from_open_min=0.002
    )
    assert sig is not None and sig.direction == "UP" and abs(sig.from_open - 0.0025) < 1e-9


def test_drain_open_cont_emits_once_with_memory_tape(tmp_path: Path):
    # Minimal day file so resolve_atm can find a ticker if we point quote root here.
    qroot = tmp_path / "QQQ"
    qroot.mkdir()
    day = "2026-06-12"
    # Synthetic ATM call bucket (UP→2)
    rows = []
    t0 = pd.Timestamp(f"{day} 09:30", tz="America/New_York")
    for i in range(0, 1200, 5):
        ts = t0 + pd.Timedelta(seconds=i)
        rows.append(
            {
                "timestamp": ts,
                "bid": 1.0,
                "ask": 1.05,
                "ticker": f"QQQ{day.replace('-', '')[2:]}C00580000",
                "strike": 580.0,
                "bucket_id": 2,
            }
        )
    pd.DataFrame(rows).to_parquet(qroot / f"QQQ_{day}.parquet", index=False)

    profile = {
        "symbols": ["NVDA"],
        "signal": {"top_k": 2, "mf_window": 10, "vol_ma_window": 20},
        "trade": {"moneyness": "ATM", "hold_minutes": 30, "tp_mult": 1.6, "sl_mult": 0.4},
        "fill": {"entry_frac": 0.75, "exit_frac": 0.75},
        "paths": {"stock_1s_root": "/mnt/s990/data/raw_1s/stocks"},
        "date_range": {"start": day, "end": day},
        "qqq_open_cont": {
            "enabled": True,
            "clock": "09:45",
            "from_open_min": 0.002,
            "tp": 0.10,
            "sl": 0.25,
            "max_hold_sec": 900,
            "position_frac": 0.10,
            "quote_1s_root": str(qroot),
        },
        "lock": {"dte_mode": "trading", "prefer_dte": 0, "allowed_dte": [0, 1, 2], "otm_rungs": 3},
    }
    assert open_cont_enabled(profile)
    sc = Mag7Scanner(
        profile=profile,
        states={"NVDA": StreamSignalState("NVDA", profile["signal"])},
        books=ContractBooks(mode="day_lock", flat_idx={}),
        stock_by={},
    )
    # Feed RTH open + clock spot via reference seconds
    open_ts = pd.Timestamp(f"{day} 09:30:00", tz="America/New_York")
    clock_ts = pd.Timestamp(f"{day} 09:45:00", tz="America/New_York")
    sc.on_reference_second(
        "QQQ", {"timestamp": open_ts, "open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0, "volume": 1}
    )
    sc.on_reference_second(
        "QQQ",
        {
            "timestamp": clock_ts,
            "open": 100.3,
            "high": 100.3,
            "low": 100.3,
            "close": 100.3,
            "volume": 1,
        },
    )
    before = sc.drain_open_cont(clock_ts - pd.Timedelta(seconds=1))
    assert before == []
    out = sc.drain_open_cont(clock_ts + pd.Timedelta(seconds=1))
    assert len(out) == 1
    sig = out[0]
    assert sig.symbol == "QQQ"
    assert sig.meta.get("event_source") == "qqq_open_cont"
    assert sig.meta.get("route") == "qqq_open_cont"
    assert abs(float(sig.meta.get("position_frac")) - 0.10) < 1e-12
    assert Mag7BrokerOms._position_frac_override(sig) == 0.10
    assert Mag7BrokerOms._signal_source_fields(sig)["event_source"] == "qqq_open_cont"
    # once/day
    assert sc.drain_open_cont(clock_ts + pd.Timedelta(minutes=5)) == []
    snap = scanner_snapshot(sc)
    assert snap["qqq_open_cont"]["n_emitted"] == 1
    assert snap["qqq_open_cont"]["enabled"] is True

    restored = Mag7Scanner(
        profile=profile,
        states={"NVDA": StreamSignalState("NVDA", profile["signal"])},
        books=ContractBooks(mode="day_lock", flat_idx={}, allowed_dte=[0, 1, 2]),
        stock_by={},
    )
    from maga7.live.scanner_state import restore_scanner

    restore_scanner(restored, snap)
    assert restored.n_open_cont_emitted == 1
    assert restored._open_cont_done_date == day
    assert restored._qqq_rth_open == 100.0
    assert restored._qqq_last_px == 100.3


def test_exit_simple_meta_on_signal():
    sig = ScannerSignal(
        date="2026-06-12",
        symbol="QQQ",
        direction="UP",
        sig_ts=pd.Timestamp("2026-06-12 09:45", tz="America/New_York"),
        spot=100.0,
        rank=0,
        bucket_id=2,
        contract="QQQ...",
        moneyness="ATM",
        meta={
            "event_source": "qqq_open_cont",
            "exit_simple": True,
            "exit_tp_mult": 1.10,
            "exit_sl_mult": 0.75,
            "exit_hold_sec": 900,
            "position_frac": 0.1,
        },
    )
    assert sig.meta["exit_simple"] is True
    assert Mag7BrokerOms._signal_source_fields(sig)["event_source"] == "qqq_open_cont"

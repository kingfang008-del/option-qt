"""Tests for AM_EXT post-fill confirm-or-abort."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from maga7.common.confirm_abort import (
    ConfirmAbortState,
    confirm_abort_applies,
    confirm_abort_from_raw,
    confirm_abort_on_tick,
)
from maga7.live.broker_oms import LivePosition, Mag7BrokerOms


def test_confirm_abort_from_raw_defaults() -> None:
    cfg = confirm_abort_from_raw(
        {
            "enabled": True,
            "confirm_sec": 60,
            "confirm_thr": 0.02,
            "abort_thr": 0.08,
            "only_entry_before": "10:26",
        }
    )
    assert cfg.enabled is True
    assert cfg.confirm_sec == 60
    assert cfg.confirm_thr == 0.02
    assert cfg.abort_thr == 0.08
    assert cfg.only_entry_before == "10:26"
    assert cfg.only_dirs is None


def test_only_dirs_ca_up_only() -> None:
    cfg = confirm_abort_from_raw(
        {
            "enabled": True,
            "only_entry_before": None,
            "only_dirs": ["UP"],
        }
    )
    assert cfg.only_dirs == ("UP",)
    assert cfg.only_entry_before is None
    t = pd.Timestamp("2026-07-24 10:45:00", tz="America/New_York").timestamp()
    assert confirm_abort_applies(cfg, t, direction="UP") is True
    assert confirm_abort_applies(cfg, t, direction="DN") is False
    assert confirm_abort_applies(cfg, t, direction=None) is False


def test_only_entry_before_gates_open_window() -> None:
    cfg = confirm_abort_from_raw({"enabled": True, "only_entry_before": "10:26"})
    t_1025 = pd.Timestamp("2026-07-24 10:25:05", tz="America/New_York").timestamp()
    t_1030 = pd.Timestamp("2026-07-24 10:30:00", tz="America/New_York").timestamp()
    assert confirm_abort_applies(cfg, t_1025) is True
    assert confirm_abort_applies(cfg, t_1030) is False


def test_confirm_abort_timeout() -> None:
    cfg = confirm_abort_from_raw(
        {"enabled": True, "confirm_sec": 60, "confirm_thr": 0.02, "abort_thr": None}
    )
    st = ConfirmAbortState()
    abort, reason, st = confirm_abort_on_tick(st, cfg=cfg, held_seconds=30, opt_mtm=0.01)
    assert abort is False
    abort, reason, st = confirm_abort_on_tick(st, cfg=cfg, held_seconds=61, opt_mtm=0.01)
    assert abort is True
    assert reason == "confirm_abort"


def test_early_abort_and_confirm() -> None:
    cfg = confirm_abort_from_raw(
        {"enabled": True, "confirm_sec": 60, "confirm_thr": 0.02, "abort_thr": 0.08}
    )
    st = ConfirmAbortState()
    abort, reason, st = confirm_abort_on_tick(st, cfg=cfg, held_seconds=20, opt_mtm=-0.09)
    assert abort is True and reason == "early_abort"

    st2 = ConfirmAbortState()
    abort, reason, st2 = confirm_abort_on_tick(st2, cfg=cfg, held_seconds=15, opt_mtm=0.03)
    assert abort is False and st2.confirmed is True
    abort, reason, st2 = confirm_abort_on_tick(st2, cfg=cfg, held_seconds=70, opt_mtm=-0.05)
    assert abort is False  # confirmed; OMS TP/SL own the rest


def _shadow_oms(tmp_path: Path) -> Mag7BrokerOms:
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

        states: dict = {}

    profile = {
        "trade": {
            "hold_minutes": 30,
            "tp_mult": 1.6,
            "sl_mult": 0.4,
            "exit_mode": "none",
            "position_frac": 0.1,
            "risk": {
                "max_stock_staleness_sec": 30.0,
                "max_option_staleness_sec": 30.0,
                "max_exit_mid_jump_pct": 0.9,
                "max_gap_hold_ticks": 3,
            },
        },
        "fill": {"entry_frac": 0.75, "exit_frac": 0.75},
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
        last_stock_tick={},
        locks={},
        ensure_option_subscription=lambda *_: True,
        release_on_demand_subscription=lambda *_: None,
    )
    return Mag7BrokerOms(
        profile=profile,
        scanner=Scanner(),
        connector=connector,
        session_id="confirmabort01",
        trade_date="2026-07-24",
        session_dir=tmp_path,
        mode="shadow",
        equity=100_000.0,
    )


def test_oms_confirm_abort_skips_dn_when_only_dirs_up(tmp_path) -> None:
    import time

    oms = _shadow_oms(tmp_path)
    day = pd.Timestamp.now(tz="America/New_York").strftime("%Y-%m-%d")
    sig_ts = pd.Timestamp(f"{day} 10:45:05", tz="America/New_York").timestamp()
    now = sig_ts + 65.0
    ca = {
        "enabled": True,
        "confirm_sec": 60,
        "confirm_thr": 0.02,
        "abort_thr": 0.08,
        "only_entry_before": None,
        "only_dirs": ["UP"],
    }
    oms.positions["NVDA"] = LivePosition(
        symbol="NVDA",
        contract="NVDA260724P00200000",
        con_id=1,
        direction="DN",
        qty=1,
        entry_price=1.0,
        entry_ts=sig_ts,
        signal_ts=sig_ts,
        rank=0,
        qty_frac=0.1,
        entry_bid=0.98,
        entry_ask=1.02,
        last_good_mid=1.0,
        exit_simple=True,
        exit_tp_mult=1.15,
        exit_sl_mult=0.80,
        exit_hold_sec=900,
        confirm_abort=ca,
        route="am_pulse_extension",
    )
    oms.connector.option_quotes[("NVDA", "NVDA260724P00200000")] = {
        "bid": 0.99,
        "ask": 1.01,
        "ts": time.time(),
    }
    events: list[tuple[str, dict]] = []
    oms._event = lambda kind, payload: events.append((kind, payload))  # type: ignore
    oms.evaluate_exits(now)
    assert not any(
        payload.get("reason") in {"CONFIRM_ABORT", "EARLY_ABORT"}
        for _, payload in events
    )


def test_oms_confirm_abort_on_timeout(tmp_path) -> None:
    import time

    oms = _shadow_oms(tmp_path)
    # Use today's 10:25 so only_entry_before applies; quote.ts must be wall-fresh.
    day = pd.Timestamp.now(tz="America/New_York").strftime("%Y-%m-%d")
    sig_ts = pd.Timestamp(f"{day} 10:25:05", tz="America/New_York").timestamp()
    now = sig_ts + 65.0
    oms.positions["NVDA"] = LivePosition(
        symbol="NVDA",
        contract="NVDA260724P00200000",
        con_id=1,
        direction="DN",
        qty=1,
        entry_price=1.0,
        entry_ts=sig_ts,
        signal_ts=sig_ts,
        rank=0,
        qty_frac=0.1,
        entry_bid=0.98,
        entry_ask=1.02,
        last_good_mid=1.0,
        exit_simple=True,
        exit_tp_mult=1.15,
        exit_sl_mult=0.80,
        exit_hold_sec=900,
        confirm_abort={
            "enabled": True,
            "confirm_sec": 60,
            "confirm_thr": 0.02,
            "abort_thr": 0.08,
            "only_entry_before": "10:26",
        },
        route="am_pulse_extension",
    )
    oms.connector.option_quotes[("NVDA", "NVDA260724P00200000")] = {
        "bid": 0.99,
        "ask": 1.01,
        "ts": time.time(),
    }
    events: list[tuple[str, dict]] = []
    oms._event = lambda kind, payload: events.append((kind, payload))  # type: ignore
    oms.evaluate_exits(now)
    assert any(
        kind == "EXIT_INTENT" and payload.get("reason") == "CONFIRM_ABORT"
        for kind, payload in events
    ) or any(
        kind == "POSITION_CLOSE" and payload.get("reason") == "CONFIRM_ABORT"
        for kind, payload in events
    )

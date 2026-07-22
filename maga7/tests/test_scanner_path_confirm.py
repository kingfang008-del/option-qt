"""Live scanner S1 path-confirm pending semantics."""
from __future__ import annotations

import pandas as pd

from maga7.live.scanner import Mag7Scanner


def _minimal_profile() -> dict:
    return {
        "symbols": ["NVDA"],
        "signal": {
            "window_start": "10:30",
            "window_end": "14:00",
            "mf_window": 10,
            "streak_min": 8,
            "from_prev_abs": 0.0,
            "vol_z_min": 0.0,
            "top_k": 2,
            "peer_align_min": 0,
        },
        "trade": {
            "moneyness": "ATM",
            "bar_availability_delay_seconds": 0,
            "max_entries_per_symbol": 1,
            "cooldown_minutes": 5,
            "stock_path_confirm": {
                "enabled": True,
                "thr_pos": 0.0015,
                "thr_neg": -0.003,
                "max_wait_seconds": 300,
                "on_timeout": "allow",
                "delay_on_pos": False,
                "tod_start": "10:30",
                "tod_end": "14:00",
            },
        },
        "fill": {"entry_frac": 0.8},
        "contract_mode": "fixed",
    }


def test_scanner_path_confirm_pending_then_neg_block(monkeypatch):
    sc = Mag7Scanner(profile=_minimal_profile(), books=None)
    # Avoid contract resolution; we only care about path gate + queue.
    monkeypatch.setattr(
        Mag7Scanner,
        "_emit_topk_resolved",
        lambda self, ctx: (_ for _ in ()).throw(AssertionError("should not emit")),
    )

    entry = pd.Timestamp("2026-02-18 10:31:00", tz="America/New_York")
    # Anchor only — no later bars yet → pending under asof.
    sc.stock_by = {
        "NVDA": pd.DataFrame(
            {
                "timestamp": [entry],
                "date": ["2026-02-18"],
                "close": [100.0],
                "open": [100.0],
                "high": [100.0],
                "low": [100.0],
                "volume": [1.0],
            }
        )
    }
    status, _, reason = sc._gate_path_confirm(
        route="baseline",
        symbol="NVDA",
        date="2026-02-18",
        direction="UP",
        feature_ts=entry,
        entry_ts=entry,
        asof_ts=entry,
        stash={"spot": 100.0, "already": False, "use_reentry": False},
    )
    assert status == "pending" and reason == "pending"
    assert len(sc.pending_path) == 1

    # Adverse first touch then drain.
    ts = pd.date_range(
        "2026-02-18 10:31:00", periods=3, freq="1min", tz="America/New_York"
    )
    sc.stock_by = {
        "NVDA": pd.DataFrame(
            {
                "timestamp": ts,
                "date": ["2026-02-18"] * 3,
                "close": [100.0, 99.60, 99.50],
                "open": [100.0, 99.60, 99.50],
                "high": [100.0, 99.60, 99.50],
                "low": [100.0, 99.60, 99.50],
                "volume": [1.0, 1.0, 1.0],
            }
        )
    }
    out = sc.drain_path_confirms(pd.Timestamp("2026-02-18 10:33:00", tz="America/New_York"))
    assert out == []
    assert sc.n_stock_path_confirm_block == 1
    assert sc.pending_path == []


def test_scanner_path_confirm_timeout_allow_emits(monkeypatch):
    sc = Mag7Scanner(profile=_minimal_profile(), books=None)
    emitted: list = []

    def _fake_emit(self, ctx):
        emitted.append(ctx)
        return None

    monkeypatch.setattr(Mag7Scanner, "_emit_topk_resolved", _fake_emit)

    entry = pd.Timestamp("2026-02-18 10:31:00", tz="America/New_York")
    ts = pd.date_range(
        "2026-02-18 10:31:00", periods=2, freq="1min", tz="America/New_York"
    )
    sc.stock_by = {
        "NVDA": pd.DataFrame(
            {
                "timestamp": ts,
                "date": ["2026-02-18"] * 2,
                "close": [100.0, 100.05],
                "open": [100.0, 100.05],
                "high": [100.0, 100.05],
                "low": [100.0, 100.05],
                "volume": [1.0, 1.0],
            }
        )
    }
    status, _, _ = sc._gate_path_confirm(
        route="baseline",
        symbol="NVDA",
        date="2026-02-18",
        direction="UP",
        feature_ts=entry,
        entry_ts=entry,
        asof_ts=entry,
        stash={"spot": 100.0, "already": False, "use_reentry": False},
    )
    assert status == "pending"
    sc.drain_path_confirms(entry + pd.Timedelta(seconds=300))
    assert sc.n_stock_path_confirm_ok == 1
    assert len(emitted) == 1
    assert emitted[0].get("path_confirm_reason") == "timeout_allow"

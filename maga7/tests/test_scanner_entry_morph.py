"""Live scanner offline-parity entry morph gates (fo_lod / range_stall)."""
from __future__ import annotations

import pandas as pd

from maga7.live.scanner import Mag7Scanner


def _day_dn_on_lod() -> pd.DataFrame:
    prev = pd.date_range("2026-07-23 15:50", periods=5, freq="1min", tz="America/New_York")
    day = pd.date_range("2026-07-24 09:30", periods=70, freq="1min", tz="America/New_York")
    rows = []
    for t in prev:
        rows.append(
            {
                "date": "2026-07-23",
                "timestamp": t,
                "open": 319.0,
                "high": 319.5,
                "low": 318.5,
                "close": 319.0,
            }
        )
    for i, t in enumerate(day):
        px = 320.0 - min(i, 60) * (11.0 / 60.0)
        rows.append(
            {
                "date": "2026-07-24",
                "timestamp": t,
                "open": 320.0 if i == 0 else px + 0.2,
                "high": px + 0.3,
                "low": px - 0.05,
                "close": px,
            }
        )
    return pd.DataFrame(rows)


def _profile_fo_lod() -> dict:
    return {
        "symbols": ["TSLA"],
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
            "fo_lod_chase_gate": {
                "enabled": True,
                "mode": "block",
                "min_fav_from_open": 0.03,
                "min_chase": 0.9,
                "max_dist_ext": 0.003,
                "dirs": ["DN"],
            },
        },
        "fill": {"entry_frac": 0.8},
        "contract_mode": "fixed",
    }


def test_scanner_fo_lod_blocks_dn_chase_before_seat():
    sc = Mag7Scanner(profile=_profile_fo_lod(), books=None)
    df = _day_dn_on_lod()
    sc.stock_by = {"TSLA": df}
    feature_ts = pd.Timestamp("2026-07-24 10:38:00", tz="America/New_York")

    allow, scale, meta = sc._entry_morph_feature_gates(
        symbol="TSLA",
        date="2026-07-24",
        direction="DN",
        feature_ts=feature_ts,
    )
    assert not allow
    assert sc.n_fo_lod_chase_block == 1
    assert meta.get("fo_lod_chase_reason")


def test_scanner_fo_lod_allows_up():
    sc = Mag7Scanner(profile=_profile_fo_lod(), books=None)
    sc.stock_by = {"TSLA": _day_dn_on_lod()}
    feature_ts = pd.Timestamp("2026-07-24 10:38:00", tz="America/New_York")
    allow, scale, _ = sc._entry_morph_feature_gates(
        symbol="TSLA",
        date="2026-07-24",
        direction="UP",
        feature_ts=feature_ts,
    )
    assert allow and abs(scale - 1.0) < 1e-12
    assert sc.n_fo_lod_chase_block == 0


def test_emit_topk_folds_morph_scale_into_regime_size(monkeypatch):
    profile = _profile_fo_lod()
    profile["trade"]["fo_lod_chase_gate"] = {"enabled": False}

    class _Books:
        mode = "fixed"

    sc = Mag7Scanner(profile=profile, books=_Books())

    class _Pick:
        ticker = "O:TSLA260724P00300000"
        bucket_id = "ATM"
        source = "test"
        dte = 0
        strike = 300.0

    monkeypatch.setattr(
        "maga7.live.scanner.resolve_entry_contract",
        lambda *a, **k: _Pick(),
    )
    sig = sc._emit_topk_resolved(
        {
            "symbol": "TSLA",
            "direction": "DN",
            "date": "2026-07-24",
            "feature_ts": pd.Timestamp("2026-07-24 10:38:00", tz="America/New_York"),
            "entry_ts": pd.Timestamp("2026-07-24 10:39:00", tz="America/New_York"),
            "spot": 309.0,
            "already": False,
            "use_reentry": False,
            "morph_size_scale": 0.5,
            "morph_meta": {"fo_lod_chase_reason": "scale_test"},
        }
    )
    assert sig is not None
    assert abs(float(sig.meta["regime_size_scale"]) - 0.5) < 1e-12
    assert abs(float(sig.meta["entry_morph_feature_scale"]) - 0.5) < 1e-12

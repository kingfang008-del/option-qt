#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


def _bootstrap_imports() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    production_dir = repo_root / "production"
    baseline_dir = production_dir / "baseline"
    dao_dir = baseline_dir / "DAO"
    sys.path.insert(0, str(production_dir))
    sys.path.insert(0, str(baseline_dir))
    sys.path.insert(0, str(dao_dir))


def _build_stats_file() -> str:
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump({}, tmp)
    tmp.flush()
    tmp.close()
    return tmp.name


def _build_history(end_ts: float, periods: int = 40) -> pd.DataFrame:
    from config import NY_TZ  # noqa: E402

    idx = pd.date_range(
        end=pd.Timestamp(end_ts, unit="s", tz=NY_TZ).floor("1min"),
        periods=periods,
        freq="1min",
    )
    vals = np.linspace(100.0, 139.0, len(idx), dtype=np.float32)
    return pd.DataFrame(
        {
            "open": vals,
            "high": vals + 1.0,
            "low": vals - 1.0,
            "close": vals,
            "volume": np.full(len(idx), 1000.0, dtype=np.float32),
            "vwap": vals,
        },
        index=idx,
    )


def _make_snapshot(atm_put_iv: float, atm_call_iv: float) -> np.ndarray:
    snap = np.zeros((6, 12), dtype=np.float32)
    snap[0, 7] = atm_put_iv
    snap[2, 7] = atm_call_iv
    snap[0, 5] = 100.0
    snap[2, 5] = 100.0
    snap[:, 8] = 1.0
    snap[:, 9] = 1.1
    return snap


def test_compute_all_inputs_uses_option_snapshot_5m_for_5min_option_features() -> None:
    _bootstrap_imports()
    from realtime_feature_engine import RealTimeFeatureEngine  # noqa: E402

    stats_path = _build_stats_file()
    engine = RealTimeFeatureEngine(stats_path=stats_path, device="cpu")
    end_ts = 1772725200.0
    history = {"NVDA": _build_history(end_ts, periods=220)}

    one_min_snap = {"NVDA": _make_snapshot(0.10, 0.30)}
    five_min_snap = {"NVDA": _make_snapshot(0.50, 0.70)}

    res = engine.compute_all_inputs(
        history_1min=history,
        fast_feats=[],
        slow_feats=["options_struc_atm_iv"],
        option_snapshots=one_min_snap,
        option_snapshot_5m=five_min_snap,
        feat_resolutions={"options_struc_atm_iv": "5min"},
        current_ts=end_ts,
        is_new_minute=True,
    )

    latest = float(res["NVDA"]["slow_1m"][0, 0, -1].item())
    assert abs(latest - 0.60) < 1e-6


def test_compute_batch_features_preserves_existing_option_history_sequence() -> None:
    _bootstrap_imports()
    from realtime_feature_engine import RealTimeFeatureEngine  # noqa: E402
    import torch  # noqa: E402

    stats_path = _build_stats_file()
    engine = RealTimeFeatureEngine(stats_path=stats_path, device="cpu")

    rows = []
    expected = []
    for i in range(30):
        close = 100.0 + float(i)
        opt_iv = 0.01 * float(i + 1)
        expected.append(opt_iv)
        rows.append([close, close + 1.0, close - 1.0, close, 1000.0, close, opt_iv])
    prices = torch.tensor([rows], dtype=torch.float32)
    feat_idx_map = {"options_vw_iv": 6}
    current_snap = torch.tensor(np.stack([_make_snapshot(0.90, 1.10)]), dtype=torch.float32)

    res = engine.compute_batch_features(
        prices_bh=prices,
        feat_idx_map=feat_idx_map,
        symbol_list=["NVDA"],
        fast_feats=[],
        slow_feats=["options_vw_iv"],
        option_snapshot=current_snap,
        skip_scaling=True,
        global_ctx={},
    )

    seq = res["slow_1m"][0, 0].cpu().numpy()
    np.testing.assert_allclose(seq, np.array(expected, dtype=np.float32), atol=1e-6)


def test_option_feature_formulas_match_offline_front_four_bucket_semantics() -> None:
    _bootstrap_imports()
    from realtime_feature_engine import RealTimeFeatureEngine  # noqa: E402
    import torch  # noqa: E402

    stats_path = _build_stats_file()
    engine = RealTimeFeatureEngine(stats_path=stats_path, device="cpu")

    snap = np.zeros((1, 6, 12), dtype=np.float32)
    snap[0, :, 7] = np.array([0.10, 0.20, 0.30, 0.40, 0.50, 0.70], dtype=np.float32)
    snap[0, 0:4, 6] = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)

    out = engine._calc_opt_feats_batch(
        torch.tensor(snap, dtype=torch.float32),
        torch.tensor([100.0], dtype=torch.float32),
    )

    assert abs(float(out["options_vw_iv"][0]) - 0.30) < 1e-6
    assert abs(float(out["options_pcr_volume"][0]) - (3.0 / 7.0)) < 1e-6
    assert abs(float(out["options_flow_skew"][0]) - (0.15 / 0.35)) < 1e-6
    assert abs(float(out["options_struc_skew"][0]) - 0.50) < 1e-6
    assert abs(float(out["options_struc_atm_iv"][0]) - 0.20) < 1e-6
    assert abs(float(out["options_struc_term"][0]) - 0.40) < 1e-6


def test_vwap_diff_uses_intraday_cumulative_vwap() -> None:
    _bootstrap_imports()
    from realtime_feature_engine import RealTimeFeatureEngine  # noqa: E402

    stats_path = _build_stats_file()
    engine = RealTimeFeatureEngine(stats_path=stats_path, device="cpu")
    idx = pd.date_range("2026-04-24 09:30:00", periods=2, freq="1min")
    df = pd.DataFrame(
        {
            "open": [10.0, 14.0],
            "high": [10.0, 14.0],
            "low": [10.0, 14.0],
            "close": [10.0, 14.0],
            "volume": [1.0, 3.0],
            "vwap": [999.0, 999.0],
        },
        index=idx,
    )

    out = engine._pandas_compute_features(df, ["vwap_diff"])

    expected_second = (14.0 - 13.0) / 13.0
    assert abs(float(out["vwap_diff"].iloc[-1]) - expected_second) < 1e-6

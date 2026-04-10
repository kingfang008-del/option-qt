import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import unittest.mock as mock


CURR_FILE = Path(__file__).resolve()
PROD_ROOT = CURR_FILE.parent.parent
sys.path.append(str(PROD_ROOT))
sys.path.append(str(PROD_ROOT / "baseline"))

sys.modules.setdefault("torch", mock.MagicMock())
sys.modules.setdefault("torch.nn", mock.MagicMock())
sys.modules.setdefault("torch.nn.functional", mock.MagicMock())
sys.modules.setdefault("ta", mock.MagicMock())

import realtime_feature_engine as rfe_mod


def _make_engine():
    engine = rfe_mod.RealTimeFeatureEngine(stats_path="/tmp/does_not_matter.json", device="cpu")
    engine.rfr_cache["20260102"] = 0.05
    return engine


def _make_contracts():
    return [
        "NVDA260109P00192500",
        "NVDA260109P00182500",
        "NVDA260109C00192500",
        "NVDA260109C00195000",
        "NVDA260206P00190000",
        "NVDA260206C00195000",
    ]


def _make_buckets():
    buckets = np.zeros((6, 12), dtype=np.float64)
    buckets[:, rfe_mod.IDX_PRICE] = [5.15, 1.30, 2.42, 1.54, 8.60, 6.65]
    buckets[:, rfe_mod.IDX_STRIKE] = [192.5, 182.5, 192.5, 195.0, 190.0, 195.0]
    return buckets


def test_locates_worst_diff_cell_as_next_call_theta():
    assert rfe_mod.ROW_NEXT_C == 5
    assert rfe_mod.IDX_THETA == 4

    engine = _make_engine()
    buckets = _make_buckets()
    contracts = _make_contracts()

    target_theta = -0.66565883

    def fake_calc(bucket_arr, stock_price, t_years, r=None, contracts=None, current_ts=None):
        bucket_arr[rfe_mod.ROW_NEXT_C, rfe_mod.IDX_THETA] = target_theta
        return bucket_arr

    with patch.object(rfe_mod, "calculate_bucket_greeks", side_effect=fake_calc):
        out = engine.supplement_greeks(
            symbol="NVDA",
            buckets=buckets.copy(),
            contracts=contracts,
            stock_price=191.09,
            timestamp=1767366780.0,
        )

    assert np.isclose(out[5, 4], target_theta)
    assert np.isclose(out[rfe_mod.ROW_NEXT_C, rfe_mod.IDX_THETA], target_theta)


def test_same_minute_timestamps_share_same_greek_time_anchor():
    engine = _make_engine()
    contracts = _make_contracts()

    captured_t_years = []

    def fake_calc(bucket_arr, stock_price, t_years, r=None, contracts=None, current_ts=None):
        captured_t_years.append(float(t_years))
        return bucket_arr

    with patch.object(rfe_mod, "calculate_bucket_greeks", side_effect=fake_calc):
        engine.supplement_greeks(
            symbol="NVDA",
            buckets=_make_buckets(),
            contracts=contracts,
            stock_price=191.09,
            timestamp=1767366780.0,  # 2026-01-02 10:13:00 ET-aligned minute sample in replay
        )
        engine.supplement_greeks(
            symbol="NVDA",
            buckets=_make_buckets(),
            contracts=contracts,
            stock_price=191.09,
            timestamp=1767366839.0,  # same minute, different second
        )

    assert len(captured_t_years) == 2
    assert np.isclose(captured_t_years[0], captured_t_years[1]), (
        f"Expected identical minute anchor, got {captured_t_years}"
    )


if __name__ == "__main__":
    test_locates_worst_diff_cell_as_next_call_theta()
    test_same_minute_timestamps_share_same_greek_time_anchor()
    print("OK: Greek parity locator tests passed")

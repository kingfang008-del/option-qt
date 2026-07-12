#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
离线 parquet vs 在线重算 —— 分层单元测试(秒级,替代 dry_sim 逐特征排查)。

运行:
  python -m unittest qqq_btc.tests.test_offline_live_feature_parity -v
  python qqq_btc/tools/feature_parity_day.py --day 2026-06-26
"""
from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[2]

from qqq_btc.common.feature_parity import (
    DEFAULT_OFFLINE_PARQUET,
    DETERMINISTIC_FEATURES,
    audit_offline_parquet_day,
    compare_feature_tiers,
    recompute_fcs_vix_level,
)

PARQUET = DEFAULT_OFFLINE_PARQUET
TEST_DAY = "2026-06-26"


@unittest.skipUnless(PARQUET.exists(), f"offline parquet 不存在: {PARQUET}")
class TestOfflineLiveFeatureParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        df = pd.read_parquet(PARQUET)
        ts_col = "timestamp" if "timestamp" in df.columns else "ts"
        df[ts_col] = pd.to_datetime(df[ts_col], utc=True)
        cls.day_frame = df[df[ts_col].dt.date == pd.Timestamp(TEST_DAY).date()].copy()
        cls.day_frame = cls.day_frame.sort_values(ts_col).reset_index(drop=True)
        cls.report = compare_feature_tiers(cls.day_frame)

    def test_deterministic_majority_pass(self):
        det = [c for c in self.report.columns if c.tier == "deterministic"]
        self.assertTrue(det)
        rate = sum(1 for c in det if c.pass_) / len(det)
        self.assertGreaterEqual(rate, 0.8, {c.feature: c.med_abs_err for c in det if not c.pass_})

    def test_time_features_exact(self):
        for feat in DETERMINISTIC_FEATURES[:4]:
            col = next((c for c in self.report.columns if c.feature == feat), None)
            self.assertIsNotNone(col, feat)
            self.assertLess(col.med_abs_err, 1e-5, feat)

    def test_vix_known_divergence(self):
        off = pd.to_numeric(self.day_frame["vix_level"], errors="coerce").to_numpy()
        live = recompute_fcs_vix_level(self.day_frame).to_numpy()
        med = float(np.median(np.abs(off - live)))
        self.assertGreater(med, 0.005, "vix 两条路径应 intentionally 不同")
        vix = next(c for c in self.report.columns if c.feature == "vix_level")
        self.assertTrue(vix.pass_)
        self.assertGreater(vix.corr or 0, 0.5)

    def test_audit_entrypoint(self):
        report = audit_offline_parquet_day(PARQUET, TEST_DAY)
        self.assertGreater(report.rows, 300)
        self.assertGreater(report.pass_rate, 0.7)


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from production.preprocess.backtest.PGSQL.oms_replay_pitcher_core import (
    build_inference_payload,
    label_ts_for_frame,
)


class StandaloneOMSReplayPitcherCoreTest(unittest.TestCase):
    def test_label_ts_aligns_to_previous_minute(self):
        # 2026-03-03 09:35:00 America/New_York
        ts_val = 1772548500.0
        self.assertEqual(label_ts_for_frame(ts_val), 1772548440)

    def test_inference_payload_uses_replay_ts_and_separate_label_ts(self):
        ts_val = 1772548500.0
        alpha_rows = {
            "NVDA": {"alpha": -0.40, "iv": 0.49, "price": 178.48, "vol_z": -0.21},
            "AAPL": {"alpha": 0.25, "iv": 0.33, "price": 245.10, "vol_z": 0.12},
        }
        batch_payloads = [
            {
                "symbol": "NVDA",
                "stock": {"close": 178.485, "volume": 12345.0},
                "option_buckets": [[0.0] * 12 for _ in range(6)],
                "option_contracts": [""] * 6,
                "option_buckets_5m": [[0.0] * 12 for _ in range(6)],
                "option_contracts_5m": [""] * 6,
            },
            {
                "symbol": "AAPL",
                "stock": {"close": 245.10, "volume": 8888.0},
                "option_buckets": [[0.0] * 12 for _ in range(6)],
                "option_contracts": [""] * 6,
                "option_buckets_5m": [[0.0] * 12 for _ in range(6)],
                "option_contracts_5m": [""] * 6,
            },
        ]
        roc_row = {"spy_roc_5min": 0.001, "qqq_roc_5min": 0.002}

        payload = build_inference_payload(
            ts_val,
            batch_payloads,
            alpha_rows,
            roc_row,
            target_symbols=["NVDA", "AAPL"],
        )

        self.assertIsNotNone(payload)
        assert payload is not None
        self.assertEqual(payload["ts"], ts_val)
        self.assertEqual(payload["log_ts"], ts_val)
        self.assertEqual(payload["source_ts"], ts_val)
        self.assertEqual(payload["frame_id"], str(int(ts_val)))
        self.assertTrue(payload["is_new_minute"])
        self.assertEqual(payload["symbols"], ["NVDA", "AAPL"])
        self.assertEqual(payload["alpha_label_ts"][0], 1772548440.0)
        self.assertEqual(payload["alpha_label_ts"][1], 1772548440.0)
        self.assertEqual(payload["alpha_available_ts"][0], ts_val)
        self.assertAlmostEqual(float(payload["precalc_alpha"][0]), -0.40, places=6)
        self.assertAlmostEqual(float(payload["precalc_alpha"][1]), 0.25, places=6)
        self.assertIn("NVDA", payload["live_options"])
        self.assertIn("AAPL", payload["live_options"])


if __name__ == "__main__":
    unittest.main()

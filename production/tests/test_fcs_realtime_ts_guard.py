#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
回归测试：实盘模式下若 payload.ts 超前系统时钟，不应提前触发分钟标签写入。
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from unittest.mock import patch

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


def _make_hist(ts: float) -> pd.DataFrame:
    from config import NY_TZ  # noqa: E402

    idx = pd.date_range(
        end=pd.Timestamp(ts, unit="s", tz=NY_TZ).floor("1min"),
        periods=40,
        freq="1min",
    )
    return pd.DataFrame(
        {
            "open": np.full(len(idx), 100.0),
            "high": np.full(len(idx), 101.0),
            "low": np.full(len(idx), 99.0),
            "close": np.full(len(idx), 100.5),
            "volume": np.full(len(idx), 1000.0),
            "vwap": np.full(len(idx), 100.3),
        },
        index=idx,
    )


class _SvcStub:
    def __init__(self, ts_anchor: float):
        self.symbols = ["NVDA"]
        self.preaggregated_1m_mode = False
        self.last_model_minute_ts = int(ts_anchor // 60) * 60
        self.global_last_minute = pd.Timestamp(ts_anchor, unit="s", tz="America/New_York").replace(second=0, microsecond=0)
        self.history_1min = {"NVDA": _make_hist(ts_anchor)}
        self.history_5min = {"NVDA": pd.DataFrame()}
        self.option_snapshot = {"NVDA": np.zeros((6, 12), dtype=np.float32)}
        self.latest_opt_buckets = {"NVDA": np.zeros((6, 12), dtype=np.float32)}
        self.latest_opt_contracts = {"NVDA": []}
        self.pending_finalization = {}
        self.current_bars_5s = {"NVDA": []}
        self.last_tick_price = {"NVDA": 100.5}
        self.latest_prices = {"NVDA": 100.5}
        self.last_stock_update_ts = {"NVDA": ts_anchor}
        self.last_option_update_ts = {"NVDA": ts_anchor}
        self.warmup_needed = {"NVDA": False}
        self.HISTORY_LEN = 500

    def _set_feature_sync_ack(self, ts_val, frame_id=None):
        return None

    def _step_engine_compute(self, data_ts, is_new_minute, alpha_label_ts):
        return [np.zeros(2, dtype=np.float32)], [True], {"NVDA": {"ok": 1}}

    def _apply_normalization_sequence(self, batch_raw, valid_mask, data_ts, is_new_minute):
        return np.zeros((1, 30, 1), dtype=np.float32)

    def _assemble_compute_payload(
        self,
        norm_seq_30,
        batch_raw,
        valid_mask,
        results_map,
        alpha_label_ts,
        data_ts,
        is_new_minute,
        ready_symbols,
    ):
        return {
            "ts": float(alpha_label_ts),
            "log_ts": float(data_ts),
            "source_ts": float(data_ts),
            "is_new_minute": bool(is_new_minute),
            "is_warmed_up": True,
            "symbols": list(ready_symbols),
            "frame_id": str(int(data_ts)),
        }

    def _atomic_commit_minute_payload(self, payload: dict) -> bool:
        return True


def test_fcs_realtime_ts_guard() -> None:
    _bootstrap_imports()
    from fcs_realtime_pipeline import FCSRealtimePipeline  # noqa: E402
    from config import NY_TZ  # noqa: E402

    wall_ts = float(pd.Timestamp("2026-04-15 10:10:50", tz=NY_TZ).timestamp())
    payload_ts_ahead = float(pd.Timestamp("2026-04-15 10:11:00", tz=NY_TZ).timestamp())

    svc = _SvcStub(ts_anchor=wall_ts)
    pipeline = FCSRealtimePipeline(svc)

    old_mode = os.environ.get("RUN_MODE")
    old_lead = os.environ.get("FCS_MAX_TS_LEAD_SEC")
    os.environ["RUN_MODE"] = "REALTIME_DRY"
    os.environ["FCS_MAX_TS_LEAD_SEC"] = "1.0"
    try:
        with patch("fcs_realtime_pipeline.time.time", return_value=wall_ts):
            payload = asyncio.run(pipeline.run_compute_cycle(ts_from_payload=payload_ts_ahead, return_payload=True))
    finally:
        if old_mode is None:
            os.environ.pop("RUN_MODE", None)
        else:
            os.environ["RUN_MODE"] = old_mode
        if old_lead is None:
            os.environ.pop("FCS_MAX_TS_LEAD_SEC", None)
        else:
            os.environ["FCS_MAX_TS_LEAD_SEC"] = old_lead

    assert payload is not None
    assert bool(payload["is_new_minute"]) is False, "超前 ts 不应触发分钟结算"
    # 非分钟帧应保持 ts=log_ts（不会回退成分钟标签）
    assert int(float(payload["ts"])) == int(float(payload["log_ts"]))
    print("[OK] realtime ts guard prevents premature minute label")


def main() -> None:
    test_fcs_realtime_ts_guard()


if __name__ == "__main__":
    main()

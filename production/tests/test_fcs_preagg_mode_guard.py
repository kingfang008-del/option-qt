#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
回归测试：防止 preaggregated_1m 模式误粘连导致标签提前与 alpha 断档。
"""

from __future__ import annotations

import asyncio
import sys
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


class _StubSvc:
    def __init__(self, ts_anchor: float):
        self.msg_count = 0
        self.symbols = ["NVDA"]
        self.preaggregated_1m_mode = True
        self._preagg_mode_logged = True
        self.last_model_minute_ts = 0
        self.global_last_minute = pd.Timestamp(ts_anchor, unit="s", tz="America/New_York").replace(second=0, microsecond=0)
        self.history_1min = {"NVDA": _make_hist(ts_anchor)}
        self.history_5min = {"NVDA": pd.DataFrame()}
        self.option_snapshot = {"NVDA": np.zeros((6, 12), dtype=np.float32)}
        self.latest_opt_buckets = {"NVDA": np.zeros((6, 12), dtype=np.float32)}
        self.latest_opt_contracts = {"NVDA": []}
        self.pending_finalization = {}
        self.cached_payloads = []
        self.ack_calls = []
        self.current_bars_5s = {"NVDA": []}
        self.last_tick_price = {"NVDA": 100.5}
        self.latest_prices = {"NVDA": 100.5}
        self.last_stock_update_ts = {"NVDA": ts_anchor}
        self.last_option_update_ts = {"NVDA": ts_anchor}
        self.warmup_needed = {"NVDA": False}
        self.HISTORY_LEN = 500

    def _update_option_frame_state(self, sym, payload, ts):
        return None

    def _finalize_1min_bar(self, sym, curr_minute, cleanup=False):
        return True

    def _set_feature_sync_ack(self, ts_val, frame_id=None):
        self.ack_calls.append((float(ts_val), str(frame_id) if frame_id is not None else None))

    def _step_engine_compute(self, data_ts, is_new_minute, alpha_label_ts):
        # 返回最小可运行结果，便于 run_compute_cycle 继续组包。
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
        payload = {
            "ts": float(alpha_label_ts),
            "log_ts": float(data_ts),
            "source_ts": float(data_ts),
            "is_new_minute": bool(is_new_minute),
            "is_warmed_up": True,
            "symbols": list(ready_symbols),
            "frame_id": str(int(data_ts)),
        }
        self.cached_payloads.append(payload)
        return payload

    def _atomic_commit_minute_payload(self, payload: dict) -> bool:
        self.cached_payloads.append({"committed_ts": float(payload.get("ts", 0.0))})
        return True


def test_fcs_preagg_mode_guard() -> None:
    _bootstrap_imports()
    from fcs_realtime_pipeline import FCSRealtimePipeline  # noqa: E402
    from config import NY_TZ  # noqa: E402

    ts_101050 = float(pd.Timestamp("2026-04-15 10:10:50", tz=NY_TZ).timestamp())
    ts_101055 = float(pd.Timestamp("2026-04-15 10:10:55", tz=NY_TZ).timestamp())
    ts_101150 = float(pd.Timestamp("2026-04-15 10:11:50", tz=NY_TZ).timestamp())

    svc = _StubSvc(ts_anchor=ts_101050)
    pipeline = FCSRealtimePipeline(svc)

    # 1) 先模拟一次“非 preagg 帧”，应自动回退模式，避免永久粘连。
    asyncio.run(
        pipeline.process_market_data(
            [{"ts": ts_101050, "symbol": "NVDA", "stock": {"close": 100.5}}]
        )
    )
    assert svc.preaggregated_1m_mode is False, "preaggregated_1m_mode 应在普通帧自动回退"
    assert svc._preagg_mode_logged is False, "_preagg_mode_logged 应在回退时重置"

    # 2) 首帧仅建锚点，不应提前产出分钟标签 payload。
    p0 = asyncio.run(pipeline.run_compute_cycle(ts_from_payload=ts_101050, return_payload=True))
    assert p0 is None, "首帧应仅建锚点，不应写分钟标签"
    assert int(svc.last_model_minute_ts) == int(ts_101050 // 60) * 60

    # 3) 同一分钟内后续帧：允许非分钟 payload，但必须 is_new_minute=False，且 ts=log_ts（不提前标签）。
    p1 = asyncio.run(pipeline.run_compute_cycle(ts_from_payload=ts_101055, return_payload=True))
    assert p1 is not None, "同分钟后续帧应可生成 payload"
    assert bool(p1["is_new_minute"]) is False, "同一分钟不应触发分钟结算"
    assert int(float(p1["ts"])) == int(float(p1["log_ts"])), "非分钟帧标签应等于数据时间，不应提前写分钟标签"

    # 4) 下一分钟（10:11:50）触发分钟结算，标签应回退到上一分钟 10:10:00。
    p2 = asyncio.run(pipeline.run_compute_cycle(ts_from_payload=ts_101150, return_payload=True))
    assert p2 is not None and bool(p2["is_new_minute"]) is True, "分钟跨越应触发 is_new_minute"
    expected_label_ts = int(ts_101150 // 60) * 60 - 60
    assert int(float(p2["ts"])) == expected_label_ts, "分钟标签应回退到上一分钟结束标签"

    print("[OK] preagg mode guard prevents sticky mode and premature label writes")


def main() -> None:
    test_fcs_preagg_mode_guard()


if __name__ == "__main__":
    main()

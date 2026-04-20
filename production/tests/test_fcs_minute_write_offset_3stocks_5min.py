#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
诊断脚本：用 3 只股票、5 分钟秒级数据驱动当前 FCS 主链路，
检查 minute label 首次产出时是否早于 label+60+grace。
"""

from __future__ import annotations

import ast
import json
import sys
import types
from dataclasses import dataclass
from datetime import datetime
from datetime import timedelta
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


def _load_option_minute_aggregator_cls():
    repo_root = Path(__file__).resolve().parents[2]
    src_path = repo_root / "production" / "baseline" / "DAO" / "feature_compute_service_v8.py"
    src = src_path.read_text(encoding="utf-8")
    mod = ast.parse(src, filename=str(src_path))
    cls_node = None
    for node in mod.body:
        if isinstance(node, ast.ClassDef) and node.name == "OptionMinuteAggregator":
            cls_node = node
            break
    if cls_node is None:
        raise RuntimeError("未在 feature_compute_service_v8.py 中找到 OptionMinuteAggregator")
    mini_mod = ast.Module(body=[cls_node], type_ignores=[])
    ast.fix_missing_locations(mini_mod)
    ns = {"np": np}
    exec(compile(mini_mod, filename=str(src_path), mode="exec"), ns)
    return ns["OptionMinuteAggregator"]


class _FakeRedis:
    def __init__(self):
        self.hashes: dict[str, dict[bytes, bytes]] = {}

    def hset(self, key, field=None, value=None, mapping=None):
        bucket = self.hashes.setdefault(key, {})
        if mapping is not None:
            for k, v in mapping.items():
                kb = k.encode("utf-8") if isinstance(k, str) else k
                if isinstance(v, bytes):
                    vb = v
                else:
                    vb = str(v).encode("utf-8")
                bucket[kb] = vb
            return
        kb = field.encode("utf-8") if isinstance(field, str) else field
        if isinstance(value, bytes):
            vb = value
        else:
            vb = str(value).encode("utf-8")
        bucket[kb] = vb

    def hgetall(self, key):
        return dict(self.hashes.get(key, {}))

    def get(self, key):
        return None


@dataclass
class _DummyProfile:
    def accept_realtime_tick(self, dt_ny):
        return True

    def should_flush_premarket(self, dt_ny, _last_date):
        return False

    def is_rth_minute(self, _dt_ny):
        return True

    def count_effective_history(self, idx, label_floor):
        return int(np.sum(idx <= label_floor))


def _build_snapshot(sec_idx: int, sym_idx: int, base_price: float) -> np.ndarray:
    arr = np.zeros((6, 12), dtype=np.float32)
    for row in range(6):
        strike = base_price + (row - 2) * 5.0
        bid = 1.0 + sym_idx * 0.1 + row * 0.03 + sec_idx * 0.001
        ask = bid + 0.05
        arr[row, 0] = (bid + ask) * 0.5
        arr[row, 5] = strike
        arr[row, 6] = 50.0 + row
        arr[row, 8] = bid
        arr[row, 9] = ask
        arr[row, 10] = 10.0 + row
        arr[row, 11] = 20.0 + row
    return arr


def _build_contracts(sym: str) -> list[str]:
    return [
        f"O:{sym}260619P00095000",
        f"O:{sym}260619P00100000",
        f"O:{sym}260619C00100000",
        f"O:{sym}260619C00105000",
        f"O:{sym}260619P00110000",
        f"O:{sym}260619C00115000",
    ]


def _bind_fake_service_methods(svc, NY_TZ):
    from fcs_realtime_pipeline import FCSRealtimePipeline  # noqa: E402
    from fcs_persistence_handler import FCSPersistenceHandler  # noqa: E402
    from fcs_support_handler import FCSSupportHandler  # noqa: E402

    pipeline = FCSRealtimePipeline(svc)
    persistence = FCSPersistenceHandler(svc)
    support = FCSSupportHandler(svc)

    svc.realtime_pipeline = pipeline
    svc.persistence_handler = persistence
    svc.support_handler = support
    svc.process_market_data = types.MethodType(FCSRealtimePipeline.process_market_data, pipeline)
    svc.run_compute_cycle = types.MethodType(FCSRealtimePipeline.run_compute_cycle, pipeline)
    svc.commit_ready_minutes = types.MethodType(lambda self, target_minute_dt: persistence.commit_ready_minutes(target_minute_dt), svc)
    svc._merge_option_snapshot_with_greeks = types.MethodType(lambda self, raw_buckets, enriched_buckets: support.merge_option_snapshot_with_greeks(raw_buckets, enriched_buckets), svc)
    svc._extract_tagged_atm_iv = types.MethodType(lambda self, buckets: support.extract_tagged_atm_iv(buckets), svc)
    svc._update_option_frame_state = types.MethodType(lambda self, sym, payload, ts: support.update_option_frame_state(sym, payload, ts), svc)
    svc._set_feature_sync_ack = types.MethodType(lambda self, ts_val, frame_id=None: None, svc)
    svc._log_minute_write_audit = types.MethodType(lambda self, **kwargs: self.minute_audits.append(kwargs), svc)
    svc._atomic_commit_minute_payload = types.MethodType(lambda self, payload: self.minute_payloads.append(dict(payload)) or True, svc)

    def _sync_history_from_redis(self, symbol, limit=500):
        raw_data = self.r.hgetall(f"BAR:1M:{symbol}")
        if not raw_data:
            return
        rows = []
        for ts_b, payload_b in raw_data.items():
            ts_i = int(ts_b.decode("utf-8"))
            payload = json.loads(payload_b.decode("utf-8"))
            rows.append({
                "timestamp": datetime.fromtimestamp(ts_i, NY_TZ),
                "open": payload["open"],
                "high": payload["high"],
                "low": payload["low"],
                "close": payload["close"],
                "volume": payload["volume"],
                "vwap": payload.get("vwap", payload["close"]),
            })
        rows.sort(key=lambda x: x["timestamp"])
        rows = rows[-limit:]
        df = pd.DataFrame(rows).set_index("timestamp")
        self.committed_history_1min[symbol] = df
        self.history_1min[symbol] = df.copy()

    def _sync_option_history_from_redis(self, symbol, limit=500):
        raw_data = self.r.hgetall(f"BAR_OPT:1M:{symbol}")
        if not raw_data:
            return
        sorted_keys = sorted(int(k.decode("utf-8")) for k in raw_data.keys())[-limit:]
        last_ts = sorted_keys[-1]
        payload = json.loads(raw_data[str(last_ts).encode("utf-8")].decode("utf-8"))
        buckets = payload.get("buckets", [])
        arr = np.array(buckets, dtype=np.float32) if buckets else np.zeros((6, 12), dtype=np.float32)
        if arr.shape != (6, 12):
            out = np.zeros((6, 12), dtype=np.float32)
            out[: min(6, arr.shape[0]), : min(12, arr.shape[1])] = arr[:6, :12]
            arr = out
        self.latest_opt_buckets[symbol] = arr.copy()
        self.latest_opt_contracts[symbol] = list(payload.get("contracts", []))
        self.committed_latest_opt_buckets[symbol] = arr.copy()
        self.committed_option_snapshot[symbol] = arr.copy()
        self.committed_option_contracts[symbol] = list(payload.get("contracts", []))
        self.option_snapshot[symbol] = arr.copy()

    svc._sync_history_from_redis = types.MethodType(_sync_history_from_redis, svc)
    svc._sync_option_history_from_redis = types.MethodType(_sync_option_history_from_redis, svc)

    def _step_engine_compute(self, data_ts, is_new_minute, alpha_label_ts):
        batch_raw = [np.array([float(idx + 1)], dtype=np.float32) for idx, _ in enumerate(self.symbols)]
        valid_mask = [True] * len(self.symbols)
        results_map = {sym: {} for sym in self.symbols}
        return batch_raw, valid_mask, results_map

    def _apply_normalization_sequence(self, batch_raw, valid_mask, data_ts, is_new_minute):
        return np.zeros((len(self.symbols), 30, 1), dtype=np.float32)

    def _assemble_compute_payload(self, norm_seq_30, batch_raw, valid_mask, results_map, alpha_label_ts, data_ts, is_new_minute, ready_symbols):
        return {
            "ts": float(alpha_label_ts),
            "log_ts": float(data_ts),
            "source_ts": float(data_ts),
            "is_new_minute": bool(is_new_minute),
            "is_warmed_up": True,
            "symbols": list(ready_symbols),
            "features_dict": {},
        }

    svc._step_engine_compute = types.MethodType(_step_engine_compute, svc)
    svc._apply_normalization_sequence = types.MethodType(_apply_normalization_sequence, svc)
    svc._assemble_compute_payload = types.MethodType(_assemble_compute_payload, svc)


def _build_service():
    _bootstrap_imports()
    from config import NY_TZ  # noqa: E402

    OptionMinuteAggregator = _load_option_minute_aggregator_cls()

    svc = type("DiagSvc", (), {})()
    svc.symbols = ["NVDA", "AAPL", "META"]
    svc.r = _FakeRedis()
    svc.msg_count = 0
    svc.minute_commit_grace_sec = 1.0
    svc.global_last_minute = None
    svc.committed_last_minute = None
    svc.preaggregated_1m_mode = False
    svc._preagg_mode_logged = False
    svc.market_profile = _DummyProfile()
    svc.current_bars_5s = {s: [] for s in svc.symbols}
    svc.minute_working_state = svc.current_bars_5s
    svc.history_1min = {s: pd.DataFrame() for s in svc.symbols}
    svc.history_5min = {s: pd.DataFrame() for s in svc.symbols}
    svc.committed_history_1min = {s: pd.DataFrame() for s in svc.symbols}
    svc.committed_history_5min = {s: pd.DataFrame() for s in svc.symbols}
    svc.option_snapshot = {s: np.zeros((6, 12), dtype=np.float32) for s in svc.symbols}
    svc.option_snapshot_5m = {s: np.zeros((6, 12), dtype=np.float32) for s in svc.symbols}
    svc.committed_option_snapshot = {s: np.zeros((6, 12), dtype=np.float32) for s in svc.symbols}
    svc.committed_option_contracts = {s: [] for s in svc.symbols}
    svc.committed_latest_opt_buckets = {s: np.zeros((6, 12), dtype=np.float32) for s in svc.symbols}
    svc.latest_opt_buckets = {s: np.zeros((6, 12), dtype=np.float32) for s in svc.symbols}
    svc.latest_opt_contracts = {s: [] for s in svc.symbols}
    svc.latest_prices = {s: 0.0 for s in svc.symbols}
    svc.last_tick_price = {s: 0.0 for s in svc.symbols}
    svc.last_stock_update_ts = {s: None for s in svc.symbols}
    svc.last_option_update_ts = {s: None for s in svc.symbols}
    svc.last_cum_volume_5m = {s: np.zeros(6, dtype=np.float32) for s in svc.symbols}
    svc.option_minute_agg = OptionMinuteAggregator(rows=6, cols=12)
    svc.warmup_needed = {s: False for s in svc.symbols}
    svc.last_model_minute_ts = 0
    svc.cached_batch_raw = None
    svc.option_frame_state = {s: {"minute_flags": {}, "last_seq": None} for s in svc.symbols}
    svc.minute_payloads = []
    svc.minute_audits = []

    _bind_fake_service_methods(svc, NY_TZ)
    return svc, NY_TZ


def test_fcs_minute_write_offset_3stocks_5min() -> None:
    svc, NY_TZ = _build_service()
    start_dt = pd.Timestamp("2026-04-17 13:00:00", tz=NY_TZ)
    first_seen: dict[int, float] = {}

    for sec_idx in range(5 * 60):
        tick_dt = start_dt + timedelta(seconds=sec_idx)
        ts = float(tick_dt.timestamp())
        batch = []
        for sym_idx, sym in enumerate(svc.symbols):
            base_px = 100.0 + sym_idx * 10.0
            close_px = base_px + sec_idx * 0.01
            stock = {
                "open": close_px - 0.05,
                "high": close_px + 0.1,
                "low": close_px - 0.1,
                "close": close_px,
                "volume": 1000.0 + sec_idx,
                "vwap": close_px,
            }
            batch.append({
                "symbol": sym,
                "ts": ts,
                "stock": stock,
                "buckets": _build_snapshot(sec_idx, sym_idx, base_px).tolist(),
                "contracts": _build_contracts(sym),
            })
        import asyncio

        asyncio.run(svc.process_market_data(batch, current_replay_ts=ts))
        payload = asyncio.run(svc.run_compute_cycle(ts_from_payload=ts, return_payload=True))
        if payload and bool(payload.get("is_new_minute")):
            label_ts = int(float(payload["ts"]))
            first_seen.setdefault(label_ts, float(payload["log_ts"]))

    assert first_seen, "未产出任何 minute payload，测试无效"

    bad_rows = []
    grace = float(svc.minute_commit_grace_sec)
    for label_ts, source_ts in sorted(first_seen.items()):
        earliest_allowed = float(label_ts) + 60.0 + grace
        if source_ts + 1e-9 < earliest_allowed:
            bad_rows.append((label_ts, source_ts, earliest_allowed))

    print("[INFO] first_seen_labels:")
    for label_ts, source_ts in sorted(first_seen.items()):
        label_ny = pd.Timestamp(label_ts, unit="s", tz=NY_TZ)
        source_ny = pd.Timestamp(source_ts, unit="s", tz=NY_TZ)
        print(
            f"  label={label_ny.strftime('%H:%M:%S')} "
            f"source={source_ny.strftime('%H:%M:%S')} "
            f"offset={source_ts - label_ts:.1f}s"
        )

    if bad_rows:
        detail = "; ".join(
            f"label={pd.Timestamp(label_ts, unit='s', tz=NY_TZ).strftime('%H:%M:%S')} "
            f"source={pd.Timestamp(source_ts, unit='s', tz=NY_TZ).strftime('%H:%M:%S')} "
            f"earliest={pd.Timestamp(earliest, unit='s', tz=NY_TZ).strftime('%H:%M:%S')}"
            for label_ts, source_ts, earliest in bad_rows
        )
        raise AssertionError(f"检测到提前 minute payload: {detail}")

    print("[OK] 3股票*5分钟 秒级驱动下，minute payload 首次产出未早于 label+60+grace")


def main() -> None:
    test_fcs_minute_write_offset_3stocks_5min()


if __name__ == "__main__":
    main()

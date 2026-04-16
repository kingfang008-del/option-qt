#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
模拟“数据接入后 -> payload 生成”流程，验证 Greeks 是否被正确包装。

验证重点：
1) live_options 的 buckets 是否为 raw + Greeks 融合结果；
2) raw 市场事实列（如 price）不被 enriched 覆盖；
3) Greeks / IV 列来自 enriched；
4) greeks_ready 标记是否正确。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


def _bootstrap_imports() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    production_dir = repo_root / "production"
    baseline_dir = production_dir / "baseline"
    dao_dir = baseline_dir / "DAO"
    sys.path.insert(0, str(production_dir))
    sys.path.insert(0, str(dao_dir))
    sys.path.insert(0, str(baseline_dir))


def _mock_contracts(sym: str) -> List[str]:
    return [
        f"O:{sym}260620P00095000",
        f"O:{sym}260620P00100000",
        f"O:{sym}260620C00100000",
        f"O:{sym}260620C00105000",
        f"O:{sym}260620P00110000",
        f"O:{sym}260620C00115000",
    ]


def _build_raw_and_enriched():
    # raw: price 用 2.0，Greeks 与 IV 置零，代表分钟事实底座
    raw = np.zeros((6, 12), dtype=np.float32)
    raw[:, 0] = 2.0
    raw[:, 5] = np.array([95, 100, 100, 105, 110, 115], dtype=np.float32)
    raw[:, 8] = 1.98
    raw[:, 9] = 2.02
    raw[:, 10] = 50
    raw[:, 11] = 60

    # enriched: 故意给不同 price=9.9，Greeks/IV 非零，测试“只覆盖 1~4 和 7”
    enriched = raw.copy()
    enriched[:, 0] = 9.9
    enriched[:, 1] = np.array([-0.35, -0.22, 0.18, 0.28, -0.40, 0.45], dtype=np.float32)
    enriched[:, 2] = np.array([0.012, 0.013, 0.014, 0.015, 0.016, 0.017], dtype=np.float32)
    enriched[:, 3] = np.array([0.11, 0.12, 0.13, 0.14, 0.15, 0.16], dtype=np.float32)
    enriched[:, 4] = np.array([-0.020, -0.021, -0.022, -0.023, -0.024, -0.025], dtype=np.float32)
    enriched[:, 7] = np.array([0.24, 0.25, 0.26, 0.27, 0.28, 0.29], dtype=np.float32)
    return raw, enriched


def main():
    _bootstrap_imports()
    from feature_compute_service_v8 import FeatureComputeService  # noqa: E402
    from fcs_support_handler import FCSSupportHandler  # noqa: E402
    from config import NY_TZ  # noqa: E402

    sym = "NVDA"
    ts = 1776254400.0  # 测试分钟

    svc = FeatureComputeService.__new__(FeatureComputeService)
    svc.symbols = [sym]
    svc.slow_feat_names = ["slow_dummy"]
    svc.feat_name_to_idx = {"slow_dummy": 0, "fast_vol": 1}
    svc.option_gate_min_iv = 0.01

    # 支撑处理器（_merge / _is_option_snapshot_complete 依赖）
    svc.support_handler = FCSSupportHandler(svc)

    raw, enriched = _build_raw_and_enriched()
    svc.option_snapshot = {sym: raw}
    svc.latest_opt_buckets = {sym: enriched}
    svc.latest_opt_contracts = {sym: _mock_contracts(sym)}
    svc.option_snapshot_5m = {}

    # history 用于价格/体量提取
    dt_idx = pd.Timestamp(ts, unit="s", tz=NY_TZ).replace(second=0, microsecond=0)
    svc.history_1min = {
        sym: pd.DataFrame(
            [{"open": 100, "high": 101, "low": 99, "close": 100.5, "volume": 12345, "vwap": 100.2}],
            index=[dt_idx],
        )
    }
    svc.normalizers = {sym: type("N", (), {"count": 60})()}

    svc.latest_prices = {sym: 100.5}
    svc.last_stock_update_ts = {sym: ts}
    svc.last_option_update_ts = {sym: ts}

    svc.runtime_payload_audit_enabled = False
    svc.sym_vol_mean = {}
    svc.sym_vol_var = {}
    svc.sym_last_vol_price = {}
    svc.cached_vol_z = {}

    norm_seq_30 = np.zeros((1, 30, 2), dtype=np.float32)
    norm_seq_30[0, :, 0] = np.linspace(-1.0, 1.0, 30, dtype=np.float32)
    norm_seq_30[0, :, 1] = 0.2
    batch_raw = [np.array([0.12, 0.2], dtype=np.float32)]
    valid_mask = [True]
    results_map = {sym: {"dummy": 1}}

    payload = svc._assemble_compute_payload(
        norm_seq_30=norm_seq_30,
        batch_raw=batch_raw,
        valid_mask=valid_mask,
        results_map=results_map,
        alpha_label_ts=ts,
        data_ts=ts + 1,
        is_new_minute=True,
        ready_symbols=[sym],
    )

    if not payload:
        raise RuntimeError("payload 生成失败")

    packaged = payload["live_options"][sym]["buckets"]
    arr = np.array(packaged, dtype=np.float32)
    assert arr.shape == (6, 12), f"buckets shape 异常: {arr.shape}"

    # 断言融合语义：price 应保留 raw(2.0)，Greeks/IV 应来自 enriched(非零)
    assert abs(float(arr[2, 0]) - 2.0) < 1e-6, f"price 未保留 raw 值: got={arr[2,0]}"
    assert abs(float(arr[2, 1]) - float(enriched[2, 1])) < 1e-6, "delta 未从 enriched 覆盖"
    assert abs(float(arr[2, 2]) - float(enriched[2, 2])) < 1e-6, "gamma 未从 enriched 覆盖"
    assert abs(float(arr[2, 3]) - float(enriched[2, 3])) < 1e-6, "vega 未从 enriched 覆盖"
    assert abs(float(arr[2, 4]) - float(enriched[2, 4])) < 1e-6, "theta 未从 enriched 覆盖"
    assert abs(float(arr[2, 7]) - float(enriched[2, 7])) < 1e-6, "iv 未从 enriched 覆盖"
    assert bool(payload["live_options"][sym]["greeks_ready"]) is True, "greeks_ready 应为 True"

    greeks_sum = float(np.sum(np.abs(arr[:, 1:5])))
    print("[OK] case-1 enriched merge passed")
    print(f"[INFO] case-1 symbol={sym} ts={int(ts)} greeks_sum={greeks_sum:.6f}")
    print(f"[INFO] case-1 price(raw-kept)={arr[2,0]:.4f}, iv(enriched)={arr[2,7]:.4f}")

    # ------------------------------------------------------------
    # case-2: enriched 缺失 -> 应回退 raw，且 greeks_ready=False
    # ------------------------------------------------------------
    raw_fallback = raw.copy()
    raw_fallback[:, 1:5] = 0.0
    raw_fallback[:, 7] = 0.0
    svc.option_snapshot[sym] = raw_fallback
    svc.latest_opt_buckets[sym] = None

    payload2 = svc._assemble_compute_payload(
        norm_seq_30=norm_seq_30,
        batch_raw=batch_raw,
        valid_mask=valid_mask,
        results_map=results_map,
        alpha_label_ts=ts + 60,
        data_ts=ts + 61,
        is_new_minute=True,
        ready_symbols=[sym],
    )
    if not payload2:
        raise RuntimeError("case-2 payload 生成失败")

    arr2 = np.array(payload2["live_options"][sym]["buckets"], dtype=np.float32)
    assert abs(float(arr2[2, 0]) - float(raw_fallback[2, 0])) < 1e-6, "case-2 price 回退 raw 失败"
    assert float(np.sum(np.abs(arr2[:, 1:5]))) < 1e-9, "case-2 Greeks 应保持 raw 的零值"
    assert bool(payload2["live_options"][sym]["greeks_ready"]) is False, "case-2 greeks_ready 应为 False"
    print("[OK] case-2 raw fallback passed")
    print(f"[INFO] case-2 greeks_sum={float(np.sum(np.abs(arr2[:,1:5]))):.6f}, greeks_ready={payload2['live_options'][sym]['greeks_ready']}")


if __name__ == "__main__":
    main()


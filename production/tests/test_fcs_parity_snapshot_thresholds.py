#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
直接测试 FCS parity 快照构建与阈值比较逻辑（无需启动 S2 回放进程）。
"""

from __future__ import annotations

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


def _build_history(start_ts: int, rows: int, vwap_shift: float = 0.0) -> pd.DataFrame:
    from config import NY_TZ  # noqa: E402

    idx = pd.date_range(
        start=pd.Timestamp(start_ts, unit="s", tz=NY_TZ) - pd.Timedelta(minutes=rows - 1),
        periods=rows,
        freq="1min",
    )
    base = np.linspace(100.0, 103.9, rows, dtype=np.float64)
    df = pd.DataFrame(
        {
            "open": base,
            "high": base + 0.3,
            "low": base - 0.2,
            "close": base + 0.1,
            "volume": np.linspace(1000.0, 1300.0, rows, dtype=np.float64),
            "vwap": base + 0.05 + float(vwap_shift),
        },
        index=idx,
    )
    return df


def _build_option_snap(scale: float = 1.0) -> np.ndarray:
    arr = np.zeros((6, 12), dtype=np.float32)
    for i in range(6):
        arr[i, 0] = 2.0 + 0.01 * i
        arr[i, 1] = (-0.3 + 0.1 * i) * scale
        arr[i, 2] = (0.01 + 0.002 * i) * scale
        arr[i, 3] = (0.10 + 0.01 * i) * scale
        arr[i, 4] = (-0.02 - 0.001 * i) * scale
        arr[i, 5] = 95.0 + i * 5.0
        arr[i, 7] = (0.20 + 0.01 * i) * scale
        arr[i, 8] = 1.95 + i * 0.02
        arr[i, 9] = 2.05 + i * 0.02
        arr[i, 10] = 50.0 + i
        arr[i, 11] = 60.0 + i
    return arr


def test_fcs_parity_snapshot_thresholds() -> None:
    _bootstrap_imports()
    from fcs_parity_snapshot_utils import (  # noqa: E402
        build_symbol_feature_parity_snapshot,
        save_feature_parity_snapshot,
        should_capture_feature_parity,
    )
    from verify_parity_thresholds import DEFAULT_THRESHOLDS, evaluate_threshold_diffs  # noqa: E402

    symbol = "NVDA"
    ts = 1767366000
    features_base = {
        "cci": np.asarray([np.linspace(-1.2, 1.1, 30, dtype=np.float32)]),
        "options_iv_divergence": np.asarray([np.linspace(0.01, 0.04, 30, dtype=np.float32)]),
    }
    features_1s = {
        "cci": features_base["cci"].copy(),
        "options_iv_divergence": features_base["options_iv_divergence"].copy() + 0.002,
    }

    valid_norm_seq = np.zeros((1, 30, 2), dtype=np.float32)
    valid_norm_seq[0, :, 0] = np.linspace(-0.8, 0.9, 30, dtype=np.float32)
    raw_mat = np.asarray([[0.15, 0.2]], dtype=np.float32)
    normalizer = type(
        "Norm",
        (),
        {"count": 180, "last_mean": np.asarray([0.02]), "last_std": np.asarray([0.8])},
    )()

    minute_snap = _build_option_snap(scale=1.0)
    second_snap = minute_snap.copy()
    second_snap[:, 1:5] += 0.1  # 保持在 option_snapshot_6x12 阈值内

    assert should_capture_feature_parity(
        batch_symbols=[symbol],
        alpha_label_ts=float(ts),
        target_symbol=symbol,
        target_ts=ts,
    )

    minute_snapshot = build_symbol_feature_parity_snapshot(
        symbol=symbol,
        symbol_idx=0,
        features_dict=features_base,
        history_1min=_build_history(ts, rows=45, vwap_shift=0.0),
        valid_norm_seq=valid_norm_seq,
        feat_name_to_idx={"cci": 0},
        raw_mat=raw_mat,
        raw_symbol_idx=0,
        normalizer=normalizer,
        batch_price=102.3,
        batch_fast_vol=0.35,
        cheat_call_iv=0.22,
        cheat_put_iv=0.24,
        source_opt_buckets={symbol: minute_snap},
        source_snap_for_payload={symbol: minute_snap},
        frozen_option_snapshot=minute_snap,
        frozen_latest_opt_buckets=minute_snap,
        valid_mask_value=True,
        real_history_len=120,
        total_history_len=180,
        real_norm_history_len=160,
        has_cross_day_warmup=True,
        alpha_label_ts=float(ts),
    )
    second_snapshot = build_symbol_feature_parity_snapshot(
        symbol=symbol,
        symbol_idx=0,
        features_dict=features_1s,
        history_1min=_build_history(ts, rows=45, vwap_shift=0.01),
        valid_norm_seq=valid_norm_seq,
        feat_name_to_idx={"cci": 0},
        raw_mat=raw_mat,
        raw_symbol_idx=0,
        normalizer=normalizer,
        batch_price=102.31,
        batch_fast_vol=0.36,
        cheat_call_iv=0.221,
        cheat_put_iv=0.239,
        source_opt_buckets={symbol: second_snap},
        source_snap_for_payload={symbol: second_snap},
        frozen_option_snapshot=second_snap,
        frozen_latest_opt_buckets=second_snap,
        valid_mask_value=True,
        real_history_len=120,
        total_history_len=180,
        real_norm_history_len=160,
        has_cross_day_warmup=True,
        alpha_label_ts=float(ts),
    )

    with tempfile.TemporaryDirectory(prefix="fcs-parity-") as tmpdir:
        left_path = save_feature_parity_snapshot(Path(tmpdir) / "minute.npz", minute_snapshot)
        right_path = save_feature_parity_snapshot(Path(tmpdir) / "second.npz", second_snapshot)
        assert left_path.exists() and right_path.exists(), "parity 快照文件保存失败"

    result = evaluate_threshold_diffs(
        left_map=minute_snapshot,
        right_map=second_snapshot,
        thresholds=DEFAULT_THRESHOLDS,
        strict=False,
    )
    if result["failures"]:
        raise AssertionError(f"阈值校验失败: {result['failures']}")

    covered = {row[0] for row in result["listed_results"]}
    required = {"hist_vwap_30", "option_snapshot_6x12", "frozen_option_snapshot_6x12"}
    missing = sorted(required - covered)
    assert not missing, f"阈值覆盖不完整，缺少: {missing}"

    changed_keys = [key for key, diff, _, _ in result["listed_results"] if diff > 1e-12]
    assert changed_keys, "期望存在 1s vs 1m 差异，但未检测到"

    print("[OK] parity 快照阈值回归通过")
    print(f"[INFO] covered threshold keys: {sorted(covered)}")
    print(f"[INFO] changed threshold keys: {sorted(changed_keys)}")


def main() -> None:
    test_fcs_parity_snapshot_thresholds()


if __name__ == "__main__":
    main()

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import sys
from pathlib import Path

import numpy as np


DEFAULT_THRESHOLDS = {
    "hist_vwap_30": 0.06,
    "vwap_diff": 0.02,
    "options_iv_divergence": 0.005,
    "option_snapshot_6x12": 0.60,
    "frozen_option_snapshot_6x12": 0.60,
    "frozen_latest_opt_buckets_6x12": 0.60,
}


def load_npz(path_str: str):
    path = Path(path_str).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"NPZ not found: {path}")
    return path, np.load(path, allow_pickle=False)


def max_abs_diff(a, b) -> float:
    a_arr = np.asarray(a)
    b_arr = np.asarray(b)
    if a_arr.shape != b_arr.shape:
        return float("inf")
    if a_arr.size == 0:
        return 0.0
    if np.issubdtype(a_arr.dtype, np.number) and np.issubdtype(b_arr.dtype, np.number):
        delta = np.abs(a_arr - b_arr)
        both_nan = np.isnan(a_arr) & np.isnan(b_arr)
        if np.all(both_nan):
            return 0.0
        delta = np.where(both_nan, 0.0, delta)
        if np.isnan(delta).all():
            return float("inf")
        return float(np.nanmax(delta))
    return 0.0 if np.array_equal(a_arr, b_arr, equal_nan=True) else float("inf")


def parse_threshold_overrides(items):
    overrides = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid --threshold format: {item!r}. Expected KEY=VALUE.")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Threshold key cannot be empty: {item!r}")
        overrides[key] = float(value)
    return overrides


def load_threshold_file(path_str: str):
    path = Path(path_str).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Threshold file not found: {path}")
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"Threshold file must contain a JSON object: {path}")
    return {str(k): float(v) for k, v in data.items()}


def build_thresholds(args):
    thresholds = dict(DEFAULT_THRESHOLDS)
    if args.threshold_file:
        thresholds.update(load_threshold_file(args.threshold_file))
    thresholds.update(parse_threshold_overrides(args.threshold))
    return thresholds


def _safe_scalar_repr(value):
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.8f}"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return repr(value)


def diff_detail(a, b):
    a_arr = np.asarray(a)
    b_arr = np.asarray(b)
    detail = {
        "left_shape": tuple(a_arr.shape),
        "right_shape": tuple(b_arr.shape),
    }

    if a_arr.shape != b_arr.shape:
        detail["kind"] = "shape_mismatch"
        return detail

    if a_arr.size == 0:
        detail["kind"] = "empty"
        return detail

    if a_arr.ndim == 0 and b_arr.ndim == 0:
        detail["kind"] = "scalar"
        detail["left"] = _safe_scalar_repr(a_arr.item())
        detail["right"] = _safe_scalar_repr(b_arr.item())
        return detail

    if np.issubdtype(a_arr.dtype, np.number) and np.issubdtype(b_arr.dtype, np.number):
        a_num = a_arr.astype(np.float64, copy=False)
        b_num = b_arr.astype(np.float64, copy=False)
        both_nan = np.isnan(a_num) & np.isnan(b_num)
        delta = np.abs(a_num - b_num)
        delta = np.where(both_nan, -1.0, delta)
        flat_idx = int(np.argmax(delta))
        max_delta = float(delta.reshape(-1)[flat_idx])
        if max_delta < 0:
            detail["kind"] = "all_nan"
            return detail
        idx = np.unravel_index(flat_idx, delta.shape)
        detail["kind"] = "numeric"
        detail["max_diff_index"] = tuple(int(i) for i in idx)
        detail["left"] = _safe_scalar_repr(a_num[idx])
        detail["right"] = _safe_scalar_repr(b_num[idx])
        detail["max_diff"] = max_delta
        return detail

    mask = ~(a_arr == b_arr)
    if not np.any(mask):
        detail["kind"] = "equal"
        return detail
    flat_idx = int(np.argmax(mask))
    idx = np.unravel_index(flat_idx, a_arr.shape)
    detail["kind"] = "object"
    detail["first_diff_index"] = tuple(int(i) for i in idx)
    detail["left"] = _safe_scalar_repr(a_arr[idx])
    detail["right"] = _safe_scalar_repr(b_arr[idx])
    return detail


def main():
    parser = argparse.ArgumentParser(description="Parity regression threshold checker")
    parser.add_argument("--left", required=True, help="Left NPZ snapshot")
    parser.add_argument("--right", required=True, help="Right NPZ snapshot")
    parser.add_argument(
        "--threshold-file",
        help="Optional JSON file with per-key thresholds, e.g. {\"hist_vwap_30\": 0.05}"
    )
    parser.add_argument(
        "--threshold",
        action="append",
        default=[],
        help="Override a threshold inline, e.g. --threshold hist_vwap_30=0.05"
    )
    parser.add_argument(
        "--show-passing",
        action="store_true",
        help="Print passing threshold keys too; otherwise only failing/covered summary is shown"
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on any non-listed differing key as well"
    )
    args = parser.parse_args()
    thresholds = build_thresholds(args)

    left_path, left = load_npz(args.left)
    right_path, right = load_npz(args.right)

    left_keys = set(left.files)
    right_keys = set(right.files)
    common = sorted(left_keys & right_keys)
    left_only = sorted(left_keys - right_keys)
    right_only = sorted(right_keys - left_keys)

    print(f"🧪 Threshold Audit")
    print(f"LEFT : {left_path}")
    print(f"RIGHT: {right_path}")
    print("-" * 100)

    failures = []

    if left_only:
        failures.append(f"Left-only keys: {', '.join(left_only)}")
    if right_only:
        failures.append(f"Right-only keys: {', '.join(right_only)}")

    listed_results = []
    other_diffs = []
    differing_details = []
    exact_match_count = 0

    for key in common:
        diff = max_abs_diff(left[key], right[key])
        if diff <= 1e-12:
            exact_match_count += 1
        else:
            differing_details.append((key, diff, diff_detail(left[key], right[key])))
        if key in thresholds:
            threshold = thresholds[key]
            ok = diff <= threshold
            listed_results.append((key, diff, threshold, ok))
            if not ok:
                failures.append(f"{key}: max_diff={diff:.8f} > threshold={threshold:.8f}")
        elif diff > 1e-12:
            other_diffs.append((key, diff))

    print(f"Common keys      : {len(common)}")
    print(f"Exact matches    : {exact_match_count}")
    print(f"Thresholded keys : {len(listed_results)}")
    print(f"Other diffs      : {len(other_diffs)}")
    print("-" * 100)
    print(f"{'Key':<36} | {'MaxDiff':<14} | {'Threshold':<14} | Status")
    print("-" * 100)
    for key, diff, threshold, ok in listed_results:
        if ok and not args.show_passing:
            continue
        status = "PASS" if ok else "FAIL"
        print(f"{key:<36} | {diff:<14.8f} | {threshold:<14.8f} | {status}")

    if other_diffs:
        print("-" * 100)
        print("Other differing keys (informational):")
        for key, diff in sorted(other_diffs, key=lambda x: x[1], reverse=True)[:20]:
            print(f"  {key}: {diff:.8f}")
        if args.strict:
            failures.append("Non-threshold keys also differ under --strict")

    if differing_details:
        print("-" * 100)
        print("Differing values (all non-exact keys):")
        for key, diff, detail in sorted(differing_details, key=lambda x: x[1], reverse=True):
            kind = detail.get("kind")
            if kind == "shape_mismatch":
                print(
                    f"  {key}: shape mismatch | "
                    f"left_shape={detail['left_shape']} | right_shape={detail['right_shape']}"
                )
            elif kind == "scalar":
                print(f"  {key}: left={detail['left']} | right={detail['right']}")
            elif kind == "numeric":
                print(
                    f"  {key}: idx={detail['max_diff_index']} | "
                    f"left={detail['left']} | right={detail['right']} | max_diff={detail['max_diff']:.8f}"
                )
            elif kind == "object":
                print(
                    f"  {key}: idx={detail['first_diff_index']} | "
                    f"left={detail['left']} | right={detail['right']}"
                )
            elif kind == "all_nan":
                print(f"  {key}: both sides are NaN at all compared positions")
            else:
                print(f"  {key}: left_shape={detail['left_shape']} | right_shape={detail['right_shape']}")

    print("-" * 100)
    print(
        "Why minute Greek recompute helps parity:\n"
        "1. Offline snapshots may carry Greeks computed earlier with a different spot, mid, or timestamp anchor.\n"
        "2. Runtime recompute forces both launchers onto the same formula path and the same inputs.\n"
        "3. The alignment improvement is usually not because offline Greeks are 'wrong', but because\n"
        "   mixing offline precomputed values with live recomputed values creates small path-dependent drift."
    )

    if failures:
        print("-" * 100)
        print("❌ Threshold audit failed:")
        for item in failures:
            print(f"  - {item}")
        sys.exit(1)

    print("✅ Threshold audit passed")


if __name__ == "__main__":
    main()

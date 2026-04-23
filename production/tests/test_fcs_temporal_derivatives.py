#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import ast
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def _load_methods():
    repo_root = Path(__file__).resolve().parents[2]
    src_path = repo_root / "production" / "baseline" / "DAO" / "feature_compute_service_v8.py"
    src = src_path.read_text(encoding="utf-8")
    mod = ast.parse(src, filename=str(src_path))
    targets = {}
    for node in mod.body:
        if isinstance(node, ast.ClassDef) and node.name == "FeatureComputeService":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name in {
                    "_rebuild_deriv_history_from_history",
                    "_inject_temporal_derivatives",
                }:
                    targets[item.name] = item
            break
    if len(targets) != 2:
        raise RuntimeError("未找到目标方法")

    ns = {
        "np": np,
        "pd": pd,
        "deque": deque,
        "Optional": Optional,
    }
    for name, fn_node in targets.items():
        mini = ast.Module(body=[fn_node], type_ignores=[])
        ast.fix_missing_locations(mini)
        exec(compile(mini, filename=str(src_path), mode="exec"), ns)
    return ns


def test_rebuild_deriv_history_from_history_uses_recent_valid_rows() -> None:
    ns = _load_methods()
    svc = type("S", (), {})()
    svc.symbols = ["NVDA"]
    svc.option_gate_min_iv = 0.01
    svc.deriv_history = {"NVDA": deque(maxlen=10)}
    idx = pd.date_range("2026-04-22 09:30:00", periods=8, freq="1min")
    svc.history_1min = {
        "NVDA": pd.DataFrame(
            {
                "options_vw_iv": [0.0, 0.02, 0.03, np.nan, 0.05, 0.06, 0.07, 0.08],
                "options_vw_gamma": [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7],
                "close": [100, 101, 102, 103, 104, 105, 106, 107],
            },
            index=idx,
        )
    }
    svc._rebuild_deriv_history_from_history = ns["_rebuild_deriv_history_from_history"].__get__(svc, svc.__class__)

    svc._rebuild_deriv_history_from_history()

    rebuilt = list(svc.deriv_history["NVDA"])
    assert len(rebuilt) == 6
    assert rebuilt[0]["iv"] == 0.02
    assert rebuilt[-1]["iv"] == 0.08
    assert rebuilt[-1]["gamma"] == 1.7
    assert rebuilt[-1]["price"] == 107.0


def test_inject_temporal_derivatives_skips_invalid_iv() -> None:
    ns = _load_methods()
    svc = type("S", (), {})()
    svc.option_gate_min_iv = 0.01
    svc.deriv_history = {"NVDA": deque([{"iv": 0.2, "gamma": 0.3, "price": 100.0}], maxlen=10)}
    svc.feat_name_to_idx = {
        "options_vw_iv": 0,
        "options_vw_gamma": 1,
        "options_iv_momentum": 2,
        "options_gamma_accel": 3,
        "options_iv_divergence": 4,
    }
    svc.latest_prices = {"NVDA": 101.0}
    svc._inject_temporal_derivatives = ns["_inject_temporal_derivatives"].__get__(svc, svc.__class__)
    raw_vec = np.zeros(5, dtype=np.float32)
    raw_vec[0] = 0.0
    raw_vec[1] = 0.5

    svc._inject_temporal_derivatives("NVDA", raw_vec, is_new_minute=True)

    assert len(svc.deriv_history["NVDA"]) == 1
    np.testing.assert_allclose(raw_vec[2:], np.zeros(3, dtype=np.float32), atol=1e-6)


def test_inject_temporal_derivatives_computes_current_values() -> None:
    ns = _load_methods()
    svc = type("S", (), {})()
    svc.option_gate_min_iv = 0.01
    svc.deriv_history = {
        "NVDA": deque(
            [
                {"iv": 0.10, "gamma": 0.20, "price": 100.0},
                {"iv": 0.11, "gamma": 0.21, "price": 101.0},
                {"iv": 0.12, "gamma": 0.22, "price": 102.0},
                {"iv": 0.13, "gamma": 0.23, "price": 103.0},
                {"iv": 0.14, "gamma": 0.24, "price": 104.0},
            ],
            maxlen=10,
        )
    }
    svc.feat_name_to_idx = {
        "options_vw_iv": 0,
        "options_vw_gamma": 1,
        "options_iv_momentum": 2,
        "options_gamma_accel": 3,
        "options_iv_divergence": 4,
    }
    svc.latest_prices = {"NVDA": 110.0}
    svc._inject_temporal_derivatives = ns["_inject_temporal_derivatives"].__get__(svc, svc.__class__)
    raw_vec = np.zeros(5, dtype=np.float32)
    raw_vec[0] = 0.20
    raw_vec[1] = 0.30

    svc._inject_temporal_derivatives("NVDA", raw_vec, is_new_minute=True)

    expected_mom = (0.20 - 0.10) / 0.10
    expected_acc = (0.30 - 0.20) / 0.20
    expected_price_ret = (110.0 - 100.0) / 100.0
    expected_div = expected_mom - expected_price_ret
    assert len(svc.deriv_history["NVDA"]) == 6
    np.testing.assert_allclose(raw_vec[2], expected_mom, atol=1e-6)
    np.testing.assert_allclose(raw_vec[3], expected_acc, atol=1e-6)
    np.testing.assert_allclose(raw_vec[4], expected_div, atol=1e-6)

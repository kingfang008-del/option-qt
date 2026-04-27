#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import logging
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


def _bootstrap_imports() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    production_dir = repo_root / "production"
    baseline_dir = production_dir / "baseline"
    history_replay_dir = production_dir / "history_replay"
    dao_dir = baseline_dir / "DAO"
    sys.path.insert(0, str(production_dir))
    sys.path.insert(0, str(baseline_dir))
    sys.path.insert(0, str(history_replay_dir))
    sys.path.insert(0, str(dao_dir))


def _load_execution_engine_module():
    sys.modules.setdefault("numpy", types.ModuleType("numpy"))
    pandas_stub = types.ModuleType("pandas")
    pandas_stub.DataFrame = object
    sys.modules.setdefault("pandas", pandas_stub)
    sys.modules.setdefault("torch", types.ModuleType("torch"))
    scipy_mod = types.ModuleType("scipy")
    scipy_stats_mod = types.ModuleType("scipy.stats")
    scipy_stats_mod.norm = object()
    scipy_mod.stats = scipy_stats_mod
    sys.modules.setdefault("scipy", scipy_mod)
    sys.modules.setdefault("scipy.stats", scipy_stats_mod)
    hist_stub = types.ModuleType("mock_ibkr_historical_1s")
    hist_stub.MockIBKRHistorical = object
    sys.modules["mock_ibkr_historical_1s"] = hist_stub
    ibkr_stub = types.ModuleType("ibkr_connector_v8")
    ibkr_stub.IBKRConnectorFinal = object
    sys.modules["ibkr_connector_v8"] = ibkr_stub
    with patch.object(logging, "FileHandler", lambda *_args, **_kwargs: logging.NullHandler()):
        import execution_engine_v8 as ee  # noqa: E402
    return ee


def _cfg():
    return SimpleNamespace(
        ENTRY_RANK_ALPHA_POWER=1.35,
        ENTRY_RANK_IV_PENALTY_POWER=0.0,
        ENTRY_RANK_HIGH_ALPHA_FLOOR=1.20,
        ENTRY_RANK_HIGH_ALPHA_BONUS_SCALE=0.35,
        ENTRY_RANK_HIGH_ALPHA_MAX_BONUS=0.50,
        ENTRY_RANK_ROC_ABS_SCALE=100.0,
        ENTRY_RANK_STOCK_ROC_SCALE=120.0,
        ENTRY_RANK_STOCK_ROC_MAX_BONUS=0.35,
        ENTRY_RANK_SNAP_ROC_SCALE=200.0,
        ENTRY_RANK_SNAP_ROC_MAX_BONUS=0.30,
        ENTRY_RANK_MACD_SCALE=8.0,
        ENTRY_RANK_MACD_MAX_BONUS=0.30,
        ENTRY_RANK_TREND_QUALITY_ENABLED=True,
        ENTRY_RANK_TREND_WINDOW_MINS=30,
        ENTRY_RANK_TREND_MIN_OBS=16,
        ENTRY_RANK_TREND_NET_TARGET=0.012,
        ENTRY_RANK_TREND_QUALITY_FLOOR=0.25,
        ENTRY_RANK_TREND_QUALITY_BOOST=0.06,
        ENTRY_RANK_TREND_QUALITY_PENALTY=0.04,
        ENTRY_RANK_TREND_MIN_MULT=0.96,
        ENTRY_RANK_TREND_MAX_MULT=1.06,
        ENTRY_PRIORITY_RESERVED_SLOTS=1,
        ENTRY_PRIORITY_ALPHA_FLOOR=0.9,
        ENTRY_PRIORITY_BOOST=0.80,
        ENTRY_PRIORITY_STOCK_ROC_FLOOR=0.0002,
        ENTRY_PRIORITY_STOCK_BONUS=0.25,
        ENTRY_PRIORITY_SNAP_ROC_FLOOR=0.0,
        ENTRY_PRIORITY_SNAP_BONUS=0.15,
        ENTRY_PRIORITY_MACD_FLOOR=0.01,
        ENTRY_PRIORITY_MACD_BONUS=0.20,
        ENTRY_PRIORITY_MIN_CONFIRMATIONS=2,
        ENTRY_DIRECTION_SPLIT_POOL_ENABLED=True,
    )


def test_strong_alpha_is_not_over_penalized_by_iv() -> None:
    _bootstrap_imports()
    ee = _load_execution_engine_module()
    cfg = _cfg()

    high_alpha = ee.compute_entry_priority_score(
        alpha=2.00,
        iv=0.50,
        roc_5m=0.0008,
        snap_roc=0.0003,
        macd_hist=0.02,
        entry_dir=1,
        cfg=cfg,
    )
    low_iv_moderate_alpha = ee.compute_entry_priority_score(
        alpha=0.90,
        iv=0.22,
        roc_5m=0.0008,
        snap_roc=0.0003,
        macd_hist=0.02,
        entry_dir=1,
        cfg=cfg,
    )

    assert high_alpha["score"] > low_iv_moderate_alpha["score"]


def test_choppy_trend_receives_ranking_penalty() -> None:
    _bootstrap_imports()
    ee = _load_execution_engine_module()
    cfg = _cfg()

    smooth = ee.compute_entry_priority_score(
        alpha=1.30,
        iv=0.40,
        roc_5m=0.0008,
        snap_roc=0.0003,
        macd_hist=0.02,
        entry_dir=1,
        cfg=cfg,
        trend_net=0.012,
        trend_efficiency=0.45,
        trend_r2=0.80,
        trend_observations=31,
    )
    choppy = ee.compute_entry_priority_score(
        alpha=1.30,
        iv=0.40,
        roc_5m=0.0008,
        snap_roc=0.0003,
        macd_hist=0.02,
        entry_dir=1,
        cfg=cfg,
        trend_net=0.0002,
        trend_efficiency=0.02,
        trend_r2=0.05,
        trend_observations=31,
    )

    assert smooth["trend_mult"] > choppy["trend_mult"]
    assert smooth["score"] > choppy["score"]


def test_strong_directional_candidate_gets_priority_boost() -> None:
    _bootstrap_imports()
    ee = _load_execution_engine_module()
    cfg = _cfg()

    strong = ee.compute_entry_priority_score(
        alpha=1.25,
        iv=0.55,
        roc_5m=0.0012,
        snap_roc=0.0008,
        macd_hist=0.03,
        entry_dir=1,
        cfg=cfg,
    )
    weak = ee.compute_entry_priority_score(
        alpha=1.55,
        iv=0.48,
        roc_5m=0.0001,
        snap_roc=-0.0001,
        macd_hist=0.002,
        entry_dir=1,
        cfg=cfg,
    )

    assert strong["priority_mult"] > 1.0
    assert strong["score"] > weak["score"], "强确认候选应能压过普通候选，尽量锁定 entry slots"


def test_strong_short_candidate_can_also_receive_priority_boost() -> None:
    _bootstrap_imports()
    ee = _load_execution_engine_module()
    cfg = _cfg()

    short_sig = ee.compute_entry_priority_score(
        alpha=-1.80,
        iv=0.30,
        roc_5m=-0.0020,
        snap_roc=-0.0009,
        macd_hist=-0.03,
        entry_dir=-1,
        cfg=cfg,
    )

    assert short_sig["priority_mult"] > 1.0


def test_reserved_priority_slot_promotes_generic_strong_candidate() -> None:
    _bootstrap_imports()
    ee = _load_execution_engine_module()
    cfg = _cfg()

    candidates = [
        {"sym": "QQQ", "sig": {"dir": -1}, "alpha_strength": 10.0, "batch_idx": 0, "is_priority_candidate": False},
        {"sym": "INTC", "sig": {"dir": 1}, "alpha_strength": 9.0, "batch_idx": 1, "is_priority_candidate": False},
        {"sym": "META", "sig": {"dir": -1}, "alpha_strength": 6.0, "batch_idx": 2, "is_priority_candidate": True},
    ]

    selected = ee.reserve_priority_entry_slots(candidates, 2, cfg)

    assert [cand["sym"] for cand in selected] == ["QQQ", "META"]


def test_reserved_priority_slot_keeps_existing_priority_pick() -> None:
    _bootstrap_imports()
    ee = _load_execution_engine_module()
    cfg = _cfg()

    candidates = [
        {"sym": "AMD", "sig": {"dir": 1}, "alpha_strength": 12.0, "batch_idx": 0, "is_priority_candidate": True},
        {"sym": "QQQ", "sig": {"dir": -1}, "alpha_strength": 11.0, "batch_idx": 1, "is_priority_candidate": False},
        {"sym": "INTC", "sig": {"dir": 1}, "alpha_strength": 10.0, "batch_idx": 2, "is_priority_candidate": False},
    ]

    selected = ee.reserve_priority_entry_slots(candidates, 2, cfg)

    assert [cand["sym"] for cand in selected] == ["AMD", "QQQ"]


def test_direction_split_keeps_call_slot_when_puts_dominate() -> None:
    _bootstrap_imports()
    ee = _load_execution_engine_module()
    cfg = _cfg()

    candidates = [
        {"sym": "XOM", "sig": {"dir": -1}, "alpha_strength": 20.0, "batch_idx": 0, "is_priority_candidate": False},
        {"sym": "UNH", "sig": {"dir": -1}, "alpha_strength": 18.0, "batch_idx": 1, "is_priority_candidate": False},
        {"sym": "MSFT", "sig": {"dir": 1}, "alpha_strength": 9.0, "batch_idx": 2, "is_priority_candidate": False},
    ]

    selected = ee.select_direction_split_entry_slots(candidates, 2, cfg)

    assert [cand["sym"] for cand in selected] == ["XOM", "MSFT"]


def test_direction_split_falls_back_to_priority_when_one_sided() -> None:
    _bootstrap_imports()
    ee = _load_execution_engine_module()
    cfg = _cfg()

    candidates = [
        {"sym": "XOM", "sig": {"dir": -1}, "alpha_strength": 20.0, "batch_idx": 0, "is_priority_candidate": False},
        {"sym": "UNH", "sig": {"dir": -1}, "alpha_strength": 18.0, "batch_idx": 1, "is_priority_candidate": False},
        {"sym": "META", "sig": {"dir": -1}, "alpha_strength": 6.0, "batch_idx": 2, "is_priority_candidate": True},
    ]

    selected = ee.select_direction_split_entry_slots(candidates, 2, cfg)

    assert [cand["sym"] for cand in selected] == ["XOM", "META"]


def test_direction_split_preserves_priority_within_same_direction() -> None:
    _bootstrap_imports()
    ee = _load_execution_engine_module()
    cfg = _cfg()

    candidates = [
        {"sym": "XOM", "sig": {"dir": -1}, "alpha_strength": 20.0, "batch_idx": 0, "is_priority_candidate": False},
        {"sym": "MSFT", "sig": {"dir": 1}, "alpha_strength": 9.0, "batch_idx": 1, "is_priority_candidate": False},
        {"sym": "AAPL", "sig": {"dir": 1}, "alpha_strength": 7.0, "batch_idx": 2, "is_priority_candidate": True},
    ]

    selected = ee.select_direction_split_entry_slots(candidates, 2, cfg)

    assert [cand["sym"] for cand in selected] == ["XOM", "AAPL"]

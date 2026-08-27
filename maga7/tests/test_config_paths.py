from __future__ import annotations

import os
from pathlib import Path

from maga7.common.config import (
    DEFAULT_LIVE_SESSIONS_DIR,
    DEFAULT_RESULTS_DIR,
    load_profile,
    resolve_live_sessions_dir,
    resolve_results_dir,
)


def test_freeze_profile_uses_s990_roots():
    profile = load_profile(
        "maga7/CONFIG/strategy_profiles/"
        "single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1.json"
    )
    assert profile["_paths"]["results_dir"] == DEFAULT_RESULTS_DIR.resolve()
    assert profile["_paths"]["live_sessions_dir"] == DEFAULT_LIVE_SESSIONS_DIR.resolve()
    assert "文档/GitHub/option-qt/maga7/results" not in str(
        profile["_paths"]["live_sessions_dir"]
    )


def test_env_overrides(monkeypatch, tmp_path):
    results = tmp_path / "results"
    live = tmp_path / "live"
    monkeypatch.setenv("MAG7_RESULTS_DIR", str(results))
    monkeypatch.setenv("MAG7_LIVE_SESSIONS_DIR", str(live))
    assert resolve_results_dir({}) == results.resolve()
    assert resolve_live_sessions_dir({}) == live.resolve()
    monkeypatch.delenv("MAG7_RESULTS_DIR", raising=False)
    monkeypatch.delenv("MAG7_LIVE_SESSIONS_DIR", raising=False)

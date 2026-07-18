from __future__ import annotations

import json

import pytest

from qqq_btc.common.strategy_profile import (
    PROFILE_ENV,
    load_strategy_profile,
    materialize_replay_cfg,
    shell_environment,
)
from qqq_btc.live.session_governor import resolve_replay_cfg


def _write_profile(tmp_path, **overrides):
    path = tmp_path / "strategy.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "profile_id": "test_profile",
                "base_replay": "LIVE_REPLAY",
                "replay_overrides": {
                    "edge_q10_floor": -0.2,
                    "apply_put_entry_quantile": False,
                    **overrides,
                },
                "selector": {"mode": "vx"},
                "execution": {"put_gate_mode": "vixy_z", "tick_exits": "off"},
            }
        )
    )
    return path


def test_materialize_replay_cfg_from_profile(tmp_path):
    profile = load_strategy_profile(
        _write_profile(tmp_path, next_day_put_quarantine_loss=-0.02)
    )
    assert profile is not None
    cfg = materialize_replay_cfg(profile)
    assert cfg.immediate_entry is True
    assert cfg.edge_q10_floor == pytest.approx(-0.2)
    assert cfg.apply_put_entry_quantile is False
    assert cfg.next_day_put_quarantine_loss == pytest.approx(-0.02)


def test_live_resolver_uses_profile_then_env_override(tmp_path, monkeypatch):
    path = _write_profile(tmp_path, put_gate_min=0.25)
    monkeypatch.setenv(PROFILE_ENV, str(path))
    monkeypatch.setenv("QQQ_BTC_PUT_GATE_MIN", "0.4")
    cfg = resolve_replay_cfg()
    assert cfg.immediate_entry is True
    assert cfg.edge_q10_floor == pytest.approx(-0.2)
    assert cfg.put_gate_min == pytest.approx(0.4)


def test_shell_environment_exports_non_replay_settings(tmp_path):
    profile = load_strategy_profile(_write_profile(tmp_path))
    assert profile is not None
    env = shell_environment(profile)
    assert env[PROFILE_ENV] == str(profile.path)
    assert env["QQQ_BTC_RULE_PROFILE_SELECTOR"] == "vx"
    assert env["QQQ_BTC_PUT_GATE_MODE"] == "vixy_z"
    assert env["QQQ_BTC_TICK_EXITS"] == "off"


def test_live_selector_uses_profile_default(tmp_path, monkeypatch):
    path = _write_profile(tmp_path)
    data = json.loads(path.read_text())
    data["selector"]["mode"] = "off"
    path.write_text(json.dumps(data))
    monkeypatch.setenv(PROFILE_ENV, str(path))
    monkeypatch.delenv("QQQ_BTC_RULE_PROFILE_SELECTOR", raising=False)

    from qqq_btc.live.rule_profile_live import rule_profile_selector_enabled

    assert rule_profile_selector_enabled() is False


def test_unknown_replay_override_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="unknown ReplayConfig fields"):
        load_strategy_profile(_write_profile(tmp_path, not_a_real_field=1))

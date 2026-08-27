"""C6 session risk budget — DD resolve + verdict (no replay I/O)."""
from __future__ import annotations

from maga7.common.session_risk_budget import (
    current_drawdown,
    parse_session_risk_budget,
    resolve_session_risk_budget,
)
from maga7.tools.run_core_c6_session_budget import keep_ratio, verdict_c6


def test_parse_default_off():
    assert parse_session_risk_budget(None).enabled is False
    cfg = parse_session_risk_budget({"enabled": True, "dd_trigger": -0.05, "scale": 0.5})
    assert cfg.enabled and cfg.dd_trigger == -0.05 and cfg.scale == 0.5


def test_dd_step_only_below_trigger():
    cfg = parse_session_risk_budget(
        {"enabled": True, "mode": "dd_step", "dd_trigger": -0.05, "scale": 0.5}
    )
    s_ok, r_ok = resolve_session_risk_budget(cfg, current_dd=-0.04)
    assert s_ok == 1.0 and r_ok == "budget_ok"
    s_hit, r_hit = resolve_session_risk_budget(cfg, current_dd=-0.05)
    assert s_hit == 0.5 and "budget_dd" in r_hit
    s_off, r_off = resolve_session_risk_budget(
        parse_session_risk_budget({"enabled": False}), current_dd=-0.20
    )
    assert s_off == 1.0 and r_off == "budget_off"


def test_dd_linear_and_current_drawdown():
    assert abs(current_drawdown(95.0, 100.0) - (-0.05)) < 1e-12
    cfg = parse_session_risk_budget(
        {"enabled": True, "mode": "dd_linear", "dd_span": 0.10, "min_scale": 0.0}
    )
    s, r = resolve_session_risk_budget(cfg, current_dd=-0.05)
    assert abs(s - 0.5) < 1e-12 and "budget_linear" in r
    s0, _ = resolve_session_risk_budget(cfg, current_dd=0.0)
    assert s0 == 1.0


def test_verdict_requires_dd_kind_dual_fire_and_keep():
    ok = verdict_c6(
        strong_keep=0.998,
        weak_keep=0.975,
        weak_maxdd_delta=0.017,
        n_scaled_weak=8,
        n_scaled_strong=2,
        kind="dd_step",
    )
    assert ok["pass"] and ok["reason"] == "pass"

    ctrl = verdict_c6(
        strong_keep=0.99,
        weak_keep=0.99,
        weak_maxdd_delta=0.02,
        n_scaled_weak=8,
        n_scaled_strong=2,
        kind="after_day_loss",
    )
    assert not ctrl["pass"] and ctrl["reason"] == "not_dd_budget"

    fat = verdict_c6(
        strong_keep=0.58,
        weak_keep=0.86,
        weak_maxdd_delta=0.017,
        n_scaled_weak=15,
        n_scaled_strong=14,
        kind="dd_step",
    )
    assert not fat["pass"] and fat["reason"] == "keep_below_bar"


def test_keep_ratio():
    assert abs(keep_ratio(0.90, 1.0) - 0.95) < 1e-12

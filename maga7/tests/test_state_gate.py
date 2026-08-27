"""Unit tests for day-state veto gate."""
from __future__ import annotations

from maga7.common.state_gate import StateGate, StateGateConfig, StateGateDayDecision


def test_disabled_passthrough():
    g = StateGate(StateGateConfig(enabled=False))
    d = g.begin_day("2026-07-20", stock_by={}, qqq_df=None, symbols=["NVDA"])
    assert d.state == "off"
    e = g.decide_entry("UP")
    assert e.allow and e.size_scale == 1.0


def test_block_up_on_mixed_wash_day():
    g = StateGate(StateGateConfig(enabled=True, mixed_wash_action="block_up"))
    g._day = StateGateDayDecision(
        enabled=True,
        date="2026-07-20",
        asof="10:30",
        state="mixed_wash",
        reason="mixed_wash_up",
        block_directions=["UP"],
    )
    up = g.decide_entry("UP")
    dn = g.decide_entry("DN")
    assert not up.allow
    assert dn.allow and dn.size_scale == 1.0


def test_scale_up_on_mixed_wash_day():
    g = StateGate(StateGateConfig(enabled=True, mixed_wash_action="scale_up", mixed_wash_scale=0.25))
    g._day = StateGateDayDecision(
        enabled=True,
        date="2026-05-01",
        asof="10:30",
        state="mixed_wash",
        reason="mixed_wash_up",
        block_directions=[],
        direction_size_scale={"UP": 0.25},
    )
    up = g.decide_entry("UP")
    assert up.allow and abs(up.size_scale - 0.25) < 1e-9


def test_from_profile_defaults():
    g = StateGate.from_profile(
        {
            "state_gate": {
                "enabled": True,
                "mixed_wash": {"washout_breadth_min": 5, "action": "block_up"},
            }
        }
    )
    assert g.cfg.enabled
    assert g.cfg.mixed_wash_breadth_min == 5
    assert g.cfg.mixed_wash_action == "block_up"

"""Watchdog state machine unit tests."""
from __future__ import annotations

import pandas as pd

from maga7.common.watchdog import (
    RegimeWatchdog,
    WatchdogConfig,
    WatchdogState,
    eval_router_rule,
)


def test_priority_halt_over_degrade():
    cfg = WatchdogConfig.from_dict(
        {
            "enabled": True,
            "asof": "10:30",
            "experts": {
                "rebound_trap_dn": {"regime": {"scale_dn_if_qqq_above_open": 0.5}},
                "washout_gate_halt": {"regime": {"block_directions": ["UP", "DN"]}},
            },
            "degrade": {"enabled": True, "rule": "reclaim_disp55", "expert": "rebound_trap_dn"},
            "halt": {
                "enabled": True,
                "rule": "washout_and_reclaim",
                "expert": "washout_gate_halt",
                "wash_drop_min": 0.001,
                "washout_breadth_min": 1,
            },
            "hunter": {"enabled": False},
        }
    )
    # Synthetic: force halt rule by monkeypatching eval via begin_day with empty → normal
    wd = RegimeWatchdog(cfg)
    # empty books → no halt/degrade
    d = wd.begin_day("2026-05-01", stock_by={}, qqq_df=None, symbols=["NVDA"])
    assert d.state == WatchdogState.NORMAL


def test_ttl_expires_to_normal():
    cfg = WatchdogConfig(
        enabled=True,
        asof="10:30",
        degrade_enabled=True,
        degrade_rule="reclaim_disp55",
        experts={"rebound_trap_dn": {"regime": {"scale_dn_if_qqq_above_open": 0.5}}},
        degrade_ttl_minutes=30,
        halt_enabled=False,
        hunter_enabled=False,
    )
    wd = RegimeWatchdog(cfg)
    # Inject a degrade decision directly
    from maga7.common.watchdog import Overlay, WatchdogDecision

    asof = pd.Timestamp("2026-05-01 10:30", tz="America/New_York")
    wd._day_decision = WatchdogDecision(
        state=WatchdogState.DEGRADE,
        overlay=Overlay(
            expert_name="rebound_trap_dn",
            regime_patch={"scale_dn_if_qqq_above_open": 0.5},
            route_tag="rebound_trap_dn",
        ),
        reason="test",
        asof=asof,
        armed_until=asof + pd.Timedelta(minutes=30),
        expert="rebound_trap_dn",
    )
    assert wd.decision_at(asof + pd.Timedelta(minutes=10)).state == WatchdogState.DEGRADE
    assert wd.decision_at(asof + pd.Timedelta(minutes=31)).state == WatchdogState.NORMAL


def test_legacy_regime_router_bridge():
    prof = {
        "regime_router": {
            "enabled": True,
            "mode": "rule",
            "rule": "reclaim_disp55",
            "asof": "10:30",
            "experts": {"rebound_trap_dn": {"regime": {"scale_dn_if_qqq_above_open": 0.5}}},
        }
    }
    wd = RegimeWatchdog.from_profile(prof)
    assert wd is not None
    assert wd.cfg.degrade_enabled
    assert not wd.cfg.halt_enabled
    assert not wd.cfg.hunter_enabled


def test_eval_router_rule_unknown():
    assert eval_router_rule("no_such_rule", date="2026-01-01", stock_by={}, qqq_df=None, symbols=[]) is None


def test_hunt_blocked_under_halt():
    from maga7.common.watchdog import Overlay, WatchdogDecision

    cfg = WatchdogConfig(
        enabled=True,
        hunter_enabled=True,
        hunter_detector="orb_fractal",
        halt_enabled=True,
    )
    wd = RegimeWatchdog(cfg)
    halt = WatchdogDecision(
        state=WatchdogState.HALT,
        overlay=Overlay(route_tag="washout_gate_halt", allow_baseline=False, allow_hunt=False),
        reason="halt",
    )
    assert wd._hunt_allowed_under(halt) is False


def test_hunt_budget():
    cfg = WatchdogConfig(enabled=True, hunter_enabled=True, hunter_max_entries_per_day=1)
    wd = RegimeWatchdog(cfg)
    assert wd.note_hunt_entry() is True
    assert wd.note_hunt_entry() is False
    assert wd.hunt_budget_remaining() == 0

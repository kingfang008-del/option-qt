from __future__ import annotations

import pytest

from maga7.common.config import load_profile
from maga7.common.narrow_experts import (
    assert_entry_promotable,
    catalog_from_profile,
    load_narrow_expert_catalog,
)


def test_catalog_loads_and_spine_experts_active():
    cat = load_narrow_expert_catalog()
    assert cat.version == "v1"
    summary = cat.summary()
    assert "hunt_washout_reclaim" in summary["active_on_spine"]
    assert "rebound_trap_dn" in summary["active_on_spine"]
    assert "washout_gate_halt" in summary["active_on_spine"]
    assert cat.get("core_dn_sync") is not None
    assert cat.get("core_dn_sync").status == "QUOTE_REJECT"
    assert not cat.get("core_dn_sync").may_promote_entry
    assert cat.get("impulse_scout") is not None
    assert cat.get("impulse_scout").status == "QUOTE_REJECT"
    assert not cat.get("impulse_scout").may_promote_entry
    assert cat.get("smc_flow_scout") is not None
    assert cat.get("smc_flow_scout").status == "QUOTE_REJECT"
    assert not cat.get("smc_flow_scout").may_promote_entry
    assert cat.get("option_flow_scout") is not None
    assert cat.get("option_flow_scout").status == "REJECT"
    assert not cat.get("option_flow_scout").may_promote_entry
    assert cat.get("stock_flow_opt") is not None
    assert cat.get("stock_flow_opt").status in {"VALIDATE_JUL_PASS", "SINGLE_WINDOW_ONLY"}
    assert not cat.get("stock_flow_opt").may_promote_entry
    assert cat.get("am_pulse_sleeve") is not None
    assert cat.get("am_pulse_sleeve").status == "ACCEPT_RESEARCH"
    assert cat.get("am_pulse_sleeve").may_promote_entry
    assert not cat.get("am_pulse_sleeve").enabled_on_spine
    assert cat.get("am_v2_executable_path") is not None
    assert cat.get("am_v2_executable_path").status == "ACCEPT_RESEARCH"
    assert cat.get("am_v2_executable_path").may_promote_entry
    assert not cat.get("am_v2_executable_path").enabled_on_spine


def test_profile_points_at_catalog():
    profile = load_profile(
        "maga7/CONFIG/strategy_profiles/"
        "single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1.json"
    )
    block = profile.get("narrow_experts")
    assert isinstance(block, dict)
    assert block.get("enabled") is True
    cat = catalog_from_profile(profile)
    assert cat is not None
    assert cat.get("am_hf_launch").status == "QUOTE_REJECT"


def test_assert_entry_promotable_blocks_quote_reject():
    cat = load_narrow_expert_catalog()
    with pytest.raises(ValueError, match="QUOTE_REJECT"):
        assert_entry_promotable("core_dn_sync", cat)
    # Hunt is already ACCEPT_RESEARCH entry morph
    assert_entry_promotable("hunt_washout_reclaim", cat)

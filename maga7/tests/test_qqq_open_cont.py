from __future__ import annotations

from maga7.common.qqq_open_cont import DEFAULT_CHAMPION, load_champion
from maga7.common.narrow_experts import load_narrow_expert_catalog


def test_champion_defaults():
    cfg = load_champion(None)
    assert cfg["clock"] == "09:45"
    assert cfg["from_open_min"] == 0.002
    assert cfg["tp"] == 0.10
    assert cfg["sl"] == 0.25


def test_catalog_marks_qqq_open_cont_accept():
    cat = load_narrow_expert_catalog()
    ex = cat.get("qqq_open_cont")
    assert ex is not None
    assert ex.status == "ACCEPT_RESEARCH"
    assert ex.kind == "entry"
    assert not ex.enabled_on_spine  # satellite, not Mag7 Rule-A
    assert ex.may_promote_entry

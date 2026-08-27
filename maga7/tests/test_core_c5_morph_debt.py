"""C5 morph-debt — strip verdict (no replay I/O)."""
from __future__ import annotations

from maga7.tools.run_core_c5_morph_debt import MORPH_GATES, strip_overlay, verdict_strip


def test_strip_overlay_disables_named_gates():
    ov = strip_overlay(("peer_gap_gate", "range_stall_gate"))
    assert ov["peer_gap_gate"]["enabled"] is False
    assert ov["range_stall_gate"]["enabled"] is False
    assert set(ov) <= set(MORPH_GATES)


def test_verdict_dead_vs_keep_vs_deprecate():
    dead = verdict_strip(strong_keep=1.0, weak_keep=1.0, n_block_total=0)
    assert dead["status"] == "DEAD" and dead["pass_deprecate"]

    dep = verdict_strip(strong_keep=0.97, weak_keep=1.01, n_block_total=4)
    assert dep["status"] == "DEPRECATED" and dep["pass_deprecate"]

    keep = verdict_strip(strong_keep=0.88, weak_keep=1.05, n_block_total=9)
    assert keep["status"] == "KEEP" and not keep["pass_deprecate"]

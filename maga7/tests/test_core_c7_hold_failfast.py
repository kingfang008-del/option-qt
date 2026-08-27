"""C7 hold fail-fast — weak-primary verdict (no replay I/O)."""
from __future__ import annotations

from maga7.tools.run_core_c7_hold_failfast import verdict_c7


def test_verdict_weak_primary():
    ok = verdict_c7(
        weak_keep=0.92,
        strong_keep=0.81,
        weak_maxdd_delta=0.012,
        weak_clock_loss_delta=0.08,
        n_early_weak=4,
        n_tp_strong=28,
        n_tp_strong_base=34,
    )
    assert ok["pass"] and ok["reason"] == "pass"


def test_verdict_rejects_no_fire_and_gutted_strong():
    no_fire = verdict_c7(
        weak_keep=1.0,
        strong_keep=1.0,
        weak_maxdd_delta=0.02,
        weak_clock_loss_delta=0.1,
        n_early_weak=0,
        n_tp_strong=34,
        n_tp_strong_base=34,
    )
    assert not no_fire["pass"] and no_fire["reason"] == "no_weak_early_cut"

    gutted = verdict_c7(
        weak_keep=0.95,
        strong_keep=0.40,
        weak_maxdd_delta=0.02,
        weak_clock_loss_delta=0.2,
        n_early_weak=6,
        n_tp_strong=10,
        n_tp_strong_base=34,
    )
    assert not gutted["pass"] and gutted["reason"] == "strong_cost_too_high"

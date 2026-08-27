"""C4 entry validator — verdict + skip masks (no replay I/O)."""
from __future__ import annotations

import pandas as pd

from maga7.tools.run_core_c4_entry_validator import (
    keep_ratio,
    skip_fo_lt,
    skip_hunt,
    skip_s1_none,
    verdict_c4,
)


def _book() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "abs_fo": [0.004, 0.02, 0.008, 0.03],
            "s1": ["", "pos", "none", "pos"],
            "route": ["baseline", "hunt", "baseline", "baseline"],
            "tod": ["10:31", "09:55", "11:40", "10:40"],
        }
    )


def test_keep_ratio():
    assert abs(keep_ratio(0.90, 1.0) - 0.95) < 1e-12


def test_skip_masks():
    t = _book()
    assert int(skip_fo_lt(0.01)(t).sum()) == 2
    assert int(skip_s1_none(t).sum()) == 2
    assert int(skip_hunt(t).sum()) == 1


def test_verdict_requires_weak_fire_prec_and_strong_keep():
    ok = verdict_c4(
        strong_keep=0.97,
        weak_keep=1.04,
        weak_maxdd_delta=0.01,
        n_skip_weak=4,
        reject_prec_weak=0.75,
        true_loss_strong=0.05,
    )
    assert ok["pass"] and ok["reason"] == "pass"

    no_fire = verdict_c4(
        strong_keep=1.0,
        weak_keep=1.02,
        weak_maxdd_delta=0.0,
        n_skip_weak=1,
        reject_prec_weak=1.0,
        true_loss_strong=0.0,
    )
    assert not no_fire["pass"] and no_fire["reason"] == "no_weak_rejects"

    fat_cut = verdict_c4(
        strong_keep=0.40,
        weak_keep=0.99,
        weak_maxdd_delta=0.01,
        n_skip_weak=6,
        reject_prec_weak=0.83,
        true_loss_strong=0.24,
    )
    assert not fat_cut["pass"]
    assert fat_cut["reason"] in {"true_loss_too_high", "keep_below_bar"}

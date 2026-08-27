"""C3 STOCK_REV overlay — verdict + config parse (no replay I/O)."""
from __future__ import annotations

from maga7.common.delta_time_stop import stock_rev_exit_from_trade
from maga7.tools.run_core_c3_stock_rev import (
    CHAMPION,
    VARIANTS,
    keep_ratio,
    reason_stats,
    verdict_c3,
)
import pandas as pd


def test_champion_overlay_parses_l3_wash_gate():
    cfg = stock_rev_exit_from_trade(VARIANTS[CHAMPION])
    assert cfg.enabled
    assert cfg.when == "mixed_wash_up"
    assert cfg.min_hold_minutes == 10.0
    assert cfg.stock_max == -0.003
    assert cfg.opt_mtm_max == 0.0
    assert cfg.washout_breadth_min == 3


def test_keep_ratio_compound():
    # baseline +45x-ish vs overlay that retains 95% of (1+ret)
    assert abs(keep_ratio(0.90, 1.0) - 0.95) < 1e-12
    assert abs(keep_ratio(0.0, 0.0) - 1.0) < 1e-12


def test_verdict_requires_fire_keep_and_tail():
    ok = verdict_c3(
        strong_keep=0.96,
        weak_keep=0.97,
        strong_deep=-0.4,
        weak_deep=-0.5,
        base_strong_deep=-0.6,
        base_weak_deep=-0.6,
        n_stock_rev=3,
        n_delta_max=0,
    )
    assert ok["pass"] and ok["reason"] == "pass"

    drift = verdict_c3(
        strong_keep=1.3,
        weak_keep=1.6,
        strong_deep=-0.1,
        weak_deep=-0.1,
        base_strong_deep=-0.6,
        base_weak_deep=-0.6,
        n_stock_rev=4,
        n_delta_max=16,
    )
    assert not drift["pass"] and drift["reason"] == "entry_set_drift"

    no_fire = verdict_c3(
        strong_keep=1.0,
        weak_keep=1.0,
        strong_deep=-0.1,
        weak_deep=-0.1,
        base_strong_deep=-0.6,
        base_weak_deep=-0.6,
        n_stock_rev=0,
    )
    assert not no_fire["pass"] and no_fire["reason"] == "no_stock_rev_fires"

    fat_cut = verdict_c3(
        strong_keep=0.91,
        weak_keep=1.01,
        strong_deep=-0.2,
        weak_deep=-0.2,
        base_strong_deep=-0.6,
        base_weak_deep=-0.6,
        n_stock_rev=4,
    )
    assert not fat_cut["pass"] and fat_cut["reason"] == "keep_below_bar"

    tail_worse = verdict_c3(
        strong_keep=0.98,
        weak_keep=0.98,
        strong_deep=-0.8,
        weak_deep=-0.8,
        base_strong_deep=-0.6,
        base_weak_deep=-0.6,
        n_stock_rev=2,
    )
    assert not tail_worse["pass"] and tail_worse["reason"] == "tail_not_improved"


def test_reason_stats_clock_and_rev():
    t = pd.DataFrame(
        {
            "reason": ["T+30", "T+45", "STOCK_REV", "TP", "SL"],
            "ret": [0.05, -0.20, -0.08, 0.60, -0.55],
        }
    )
    rs = reason_stats(t)
    assert rs["n_clock"] == 2
    assert abs(rs["clock_share"] - 0.4) < 1e-12
    assert rs["n_stock_rev"] == 1
    assert rs["n_deep"] == 2
    assert abs(rs["deep_sum_ret"] - (-0.75)) < 1e-12

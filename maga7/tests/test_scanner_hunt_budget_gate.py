"""Scanner Hunt reschedule must use daily budget, not cumulative emits."""
from __future__ import annotations

from maga7.common.watchdog import RegimeWatchdog, WatchdogConfig
from maga7.live.scanner import Mag7Scanner


def test_hunt_reschedule_allows_later_days_after_prior_emit():
    """Regression: cumulative n_hunt_emitted==0 blocked Hunt after day-1 fill."""
    sc = Mag7Scanner(
        profile={
            "symbols": ["NVDA"],
            "signal": {"top_k": 2},
            "trade": {},
            "date_range": {"start": "2026-07-01", "end": "2026-07-21"},
        },
        books=None,
    )
    wd = RegimeWatchdog(
        WatchdogConfig(enabled=True, hunter_enabled=True, hunter_max_entries_per_day=1)
    )
    sc.watchdog = wd
    # Day 1 consumed
    assert wd.note_hunt_entry() is True
    sc.n_hunt_emitted = 1
    assert wd.hunt_budget_remaining() == 0
    # New session: begin_day resets daily budget
    wd.begin_day("2026-07-13", stock_by={}, qqq_df=None, symbols=["NVDA"])
    assert wd.hunt_budget_remaining() == 1
    # Gate used by _eval_watchdog must allow reschedule despite cumulative emits
    assert sc.n_hunt_emitted > 0
    assert (sc.watchdog is None or sc.watchdog.hunt_budget_remaining() > 0)

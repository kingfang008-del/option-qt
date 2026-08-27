from __future__ import annotations

import pandas as pd

from maga7.common.fills import FillSpec
from maga7.common.replay import simulate_trade

NY = "America/New_York"


def _opt_path(start: str, prices: list[float]) -> pd.DataFrame:
    ts = pd.date_range(start, periods=len(prices), freq="1min", tz=NY)
    return pd.DataFrame(
        {"timestamp": ts, "bid": prices, "ask": [p + 0.05 for p in prices]}
    )


def test_mtm_trail_locks_giveback_after_activate():
    # Entry ~1.0; climb to 1.30 (+30%) then give back to 1.15 → trail 20/12 fires.
    prices = [1.0] + [1.0 + 0.02 * i for i in range(1, 16)] + [1.15] * 10
    opt = _opt_path("2026-05-01 10:30", prices)
    sim = simulate_trade(
        opt,
        entry_ts=pd.Timestamp("2026-05-01 10:30", tz=NY),
        fill=FillSpec(0.5, 0.5),
        tp_mult=10.0,
        sl_mult=0.1,
        hold_minutes=45,
        direction="UP",
        exit_mode="mtm_trail",
        trail_activate=0.20,
        trail_dd=0.12,
        stock_bar_delay_seconds=0,
    )
    assert sim is not None
    assert sim.reason == "TRAIL"
    assert sim.ret > 0.0

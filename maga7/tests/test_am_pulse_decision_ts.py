from __future__ import annotations

import pandas as pd

from maga7.common.am_pulse_scout import am_pulse_decision_ts
from maga7.common.option_quote_tpsl import entry_quote_row
from maga7.common.replay import to_ny


def test_am_pulse_decision_ts_adds_60s():
    feature = pd.Timestamp("2026-07-24 09:44:00", tz="America/New_York")
    decision = am_pulse_decision_ts(feature, delay_seconds=60)
    assert decision == pd.Timestamp("2026-07-24 09:45:00", tz="America/New_York")


def test_entry_quote_uses_decision_not_feature_close():
    """Quotes between feature_ts and decision_ts must not be tradable."""
    feature = pd.Timestamp("2026-07-24 09:44:00", tz="America/New_York")
    decision = am_pulse_decision_ts(feature, delay_seconds=60)
    path = pd.DataFrame(
        [
            {
                "timestamp": feature + pd.Timedelta(seconds=5),
                "bid": 1.00,
                "ask": 1.10,
            },
            {
                "timestamp": decision + pd.Timedelta(seconds=1),
                "bid": 1.20,
                "ask": 1.30,
            },
        ]
    )
    early = entry_quote_row(path, feature, max_lag_sec=5.0, max_spread_pct=0.20)
    assert early is not None
    assert to_ny(early["entry_ts"]) < decision

    causal = entry_quote_row(path, decision, max_lag_sec=5.0, max_spread_pct=0.20)
    assert causal is not None
    assert to_ny(causal["entry_ts"]) >= decision
    assert abs(float(causal["mid"]) - 1.25) < 1e-9

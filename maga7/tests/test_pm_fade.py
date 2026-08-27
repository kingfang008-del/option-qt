from __future__ import annotations

import pandas as pd

from maga7.common.pm_fade import PmFadeConfig, iter_pm_fade_signals, prepare_day


def _day_up_then_fade() -> pd.DataFrame:
    # 14:00=100 → 14:40=101.2 (+1.2%), then reverse so 14:45 confirm is down.
    idx = pd.date_range("2026-07-22 14:00", periods=61, freq="1min", tz="America/New_York")
    closes = []
    for i in range(len(idx)):
        if i <= 40:
            closes.append(100.0 + i * (1.2 / 40.0))
        else:
            closes.append(101.2 - (i - 40) * (0.5 / 20.0))
    return pd.DataFrame(
        {
            "date": ["2026-07-22"] * len(idx),
            "timestamp": idx,
            "open": [100.0] + closes[:-1],
            "close": closes,
        }
    )


def test_pm_fade_fires_on_extension_with_confirm():
    day = prepare_day(_day_up_then_fade(), "2026-07-22")
    cfg = PmFadeConfig(enabled=True, ext_min=0.008, require_confirm=True, confirm_minutes=5)
    sigs = iter_pm_fade_signals(day, date="2026-07-22", symbol="NVDA", cfg=cfg)
    assert len(sigs) >= 1
    assert sigs[0]["dir"] == "DN"
    assert sigs[0]["ext_from_anchor"] >= 0.008


def test_no_fade_without_extension():
    idx = pd.date_range("2026-07-22 14:00", periods=46, freq="1min", tz="America/New_York")
    df = pd.DataFrame(
        {
            "date": ["2026-07-22"] * len(idx),
            "timestamp": idx,
            "open": [100.0] * len(idx),
            "close": [100.0 + 0.001 * i for i in range(len(idx))],
        }
    )
    day = prepare_day(df, "2026-07-22")
    cfg = PmFadeConfig(enabled=True, ext_min=0.008, require_confirm=False)
    sigs = iter_pm_fade_signals(day, date="2026-07-22", symbol="NVDA", cfg=cfg)
    assert sigs == []

"""Unit tests for entry-side adverse volume share gate config."""
from __future__ import annotations

from maga7.common.adverse_vol_share import entry_adv_vol_from_trade


def test_entry_adv_vol_default_off():
    assert entry_adv_vol_from_trade({}).enabled is False


def test_entry_adv_vol_parse():
    cfg = entry_adv_vol_from_trade(
        {
            "entry_adv_vol": {
                "enabled": True,
                "action": "block",
                "window_seconds": 120,
                "max_share": 0.6,
                "tod_start": "10:31",
                "tod_end": "11:00",
            }
        }
    )
    assert cfg.enabled
    assert cfg.action == "block"
    assert cfg.window_seconds == 120
    assert cfg.max_share == 0.6
    assert cfg.tod_start == "10:31"


def test_entry_adv_vol_scale_alias():
    cfg = entry_adv_vol_from_trade(
        {"entry_adv_vol_share": {"enabled": True, "action": "half", "scale": 0.5}}
    )
    assert cfg.action == "scale"
    assert cfg.scale == 0.5


def test_entry_adv_vol_dirs_and_lag():
    cfg = entry_adv_vol_from_trade(
        {
            "entry_adv_vol": {
                "enabled": True,
                "dirs": "UP,DN",
                "lag_seconds": 60,
                "max_share": 0.5,
            }
        }
    )
    assert cfg.dirs == ("UP", "DN")
    assert cfg.lag_seconds == 60
    assert cfg.max_share == 0.5

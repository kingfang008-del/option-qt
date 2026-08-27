from __future__ import annotations

from maga7.common.path_greeks_exit import (
    PathGreeksExitConfig,
    PathGreeksState,
    cfg_from_preset,
)


def test_winner_safe_giveback_requires_flat():
    cfg, naive = cfg_from_preset("winner_safe")
    assert not naive
    assert cfg.giveback_ret_max == 0.0
    assert cfg.giveback_peak_frac == 0.0
    # +22% after +40% peak must NOT giveback under winner_safe
    floor = max(cfg.giveback_ret_max, 0.40 * cfg.giveback_peak_frac)
    assert 0.22 > floor



def test_toxic_only_disables_giveback():
    cfg, _ = cfg_from_preset("toxic_only")
    assert cfg.giveback_peak_min >= 9.0


def test_from_trade_parse():
    cfg = PathGreeksExitConfig.from_trade(
        {"path_greeks_exit": {"enabled": True, "iv_shock_ret_max": 0.12}}
    )
    assert cfg.enabled
    assert cfg.iv_shock_ret_max == 0.12

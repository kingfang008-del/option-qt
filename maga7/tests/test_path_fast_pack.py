from __future__ import annotations

from maga7.common.delta_time_stop import StockRevExitConfig
from maga7.common.path_fast_pack import (
    apply_path_fast_pack_overrides,
    path_fast_pack_day_should_arm,
    path_fast_pack_from_trade,
)


def test_path_fast_pack_default_off():
    assert path_fast_pack_from_trade({}).enabled is False


def test_path_fast_pack_parse_and_apply():
    cfg = path_fast_pack_from_trade(
        {
            "path_fast_pack": {
                "enabled": True,
                "when": "mixed_wash_up",
                "hold_minutes": 20,
                "trail_activate": 0.15,
                "trail_dd": 0.08,
                "stock_rev_min_hold_minutes": 5,
                "stock_rev_opt_mtm_max": 0.05,
            }
        }
    )
    assert cfg.enabled
    assert cfg.when == "mixed_wash_up"
    ov = apply_path_fast_pack_overrides(
        hold_minutes=45,
        trail_activate=0.20,
        trail_dd=0.12,
        stock_rev=StockRevExitConfig(enabled=True, min_hold_minutes=10, opt_mtm_max=0.10),
        pack=cfg,
    )
    assert ov["hold_minutes"] == 20
    assert ov["trail_activate"] == 0.15
    assert ov["trail_dd"] == 0.08
    assert ov["stock_rev_exit"].min_hold_minutes == 5.0
    assert ov["stock_rev_exit"].opt_mtm_max == 0.05


def test_path_fast_pack_always_arms():
    cfg = path_fast_pack_from_trade(
        {"path_fast_pack": {"enabled": True, "when": "always"}}
    )
    assert path_fast_pack_day_should_arm(
        cfg,
        date="2026-07-20",
        stock_by={},
        qqq_df=None,
        symbols=["NVDA"],
    )


def test_path_fast_pack_parses_opt_chop_when():
    cfg = path_fast_pack_from_trade(
        {
            "path_fast_pack": {
                "enabled": True,
                "when": "wash_or_opt_chop",
                "opt_imbalance_max": -0.08,
                "opt_chop_pctile_min": 0.75,
                "opt_gate": "imb_or_chop",
            }
        }
    )
    assert cfg.when == "wash_or_opt_chop"
    assert cfg.opt_imbalance_max == -0.08
    assert cfg.opt_chop_pctile_min == 0.75
    assert cfg.opt_gate == "imb_or_chop"


def test_path_fast_pack_default_opt_gate_imb_only():
    cfg = path_fast_pack_from_trade(
        {"path_fast_pack": {"enabled": True, "when": "wash_and_opt_chop"}}
    )
    assert cfg.opt_gate == "imb_only"


def test_path_fast_pack_parses_wash_refine():
    cfg = path_fast_pack_from_trade(
        {
            "path_fast_pack": {
                "enabled": True,
                "when": "mixed_wash_up",
                "wash_refine": True,
                "wash_refine_chop_max": 1.85,
                "wash_refine_med_stock_ret_max": 0.003,
            }
        }
    )
    assert cfg.wash_refine is True
    assert cfg.wash_refine_chop_max == 1.85
    assert cfg.wash_refine_med_stock_ret_max == 0.003

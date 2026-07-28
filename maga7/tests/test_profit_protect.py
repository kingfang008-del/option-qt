from maga7.common.profit_protect import (
    profit_protect_from_raw,
    profit_protect_on_tick,
)


def test_profit_protect_arms_then_holds_three_percent_floor():
    cfg = profit_protect_from_raw(
        {"enabled": True, "arm_ret": 0.08, "floor_ret": 0.03}
    )
    assert cfg.enabled
    assert not profit_protect_on_tick(cfg=cfg, peak_mfe=0.079, opt_mtm=-0.10)
    assert not profit_protect_on_tick(cfg=cfg, peak_mfe=0.10, opt_mtm=0.031)
    assert profit_protect_on_tick(cfg=cfg, peak_mfe=0.10, opt_mtm=0.03)


def test_profit_protect_rejects_invalid_floor():
    cfg = profit_protect_from_raw(
        {"enabled": True, "arm_ret": 0.08, "floor_ret": 0.09}
    )
    assert not cfg.enabled

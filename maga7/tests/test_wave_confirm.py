"""Post-fill revocable wave abort."""
from __future__ import annotations

from maga7.common.wave_confirm import (
    WaveAbortConfig,
    WaveAbortState,
    wave_abort_from_trade,
    wave_abort_on_tick,
)


def test_arm_then_revoke():
    cfg = WaveAbortConfig(enabled=True, thr_pos=0.0015, thr_neg=-0.003, max_wait_seconds=300, revoke_seconds=1800)
    st = WaveAbortState()
    abort, reason, st = wave_abort_on_tick(st, cfg=cfg, held_seconds=60, stock_signed=0.002)
    assert not abort and st.armed and reason == ""
    abort, reason, st = wave_abort_on_tick(st, cfg=cfg, held_seconds=120, stock_signed=-0.0035)
    assert abort and reason == "revoke"


def test_neg_before_arm():
    cfg = WaveAbortConfig(enabled=True)
    st = WaveAbortState()
    abort, reason, st = wave_abort_on_tick(st, cfg=cfg, held_seconds=30, stock_signed=-0.004)
    assert abort and reason == "neg" and not st.armed


def test_timeout_abort():
    cfg = WaveAbortConfig(enabled=True, on_timeout="abort", max_wait_seconds=300)
    st = WaveAbortState()
    abort, reason, st = wave_abort_on_tick(st, cfg=cfg, held_seconds=301, stock_signed=0.0005)
    assert abort and reason == "timeout"


def test_timeout_allow():
    cfg = WaveAbortConfig(enabled=True, on_timeout="allow", max_wait_seconds=300)
    st = WaveAbortState()
    abort, reason, st = wave_abort_on_tick(st, cfg=cfg, held_seconds=301, stock_signed=0.0005)
    assert not abort and st.done


def test_from_trade():
    cfg = wave_abort_from_trade(
        {"wave_abort": {"enabled": True, "thr_neg": -0.002, "revoke_seconds": 900}}
    )
    assert cfg.enabled and cfg.thr_neg == -0.002 and cfg.revoke_seconds == 900.0


def test_only_directions_from_trade():
    cfg = wave_abort_from_trade(
        {"wave_abort": {"enabled": True, "only_directions": ["UP"]}}
    )
    assert cfg.only_directions == ("UP",)
    cfg2 = wave_abort_from_trade(
        {"wave_abort": {"enabled": True, "only_directions": "up,dn"}}
    )
    assert cfg2.only_directions == ("UP", "DN")


def test_asymmetric_revoke_and_opt_gate():
    cfg = WaveAbortConfig(
        enabled=True,
        thr_pos=0.0015,
        thr_neg=-0.003,
        thr_neg_revoke=-0.005,
        revoke_opt_mtm_max=0.0,
        revoke_seconds=1800,
    )
    st = WaveAbortState(armed=True)
    abort, reason, st = wave_abort_on_tick(st, cfg=cfg, held_seconds=120, stock_signed=-0.004, opt_mtm=-0.02)
    assert not abort  # -40bp < revoke thr -50bp
    abort, reason, st = wave_abort_on_tick(st, cfg=cfg, held_seconds=130, stock_signed=-0.006, opt_mtm=0.05)
    assert not abort and reason == ""  # stock deep but opt green
    abort, reason, st = wave_abort_on_tick(st, cfg=cfg, held_seconds=140, stock_signed=-0.006, opt_mtm=-0.02)
    assert abort and reason == "revoke"


def test_no_revoke_after_arm():
    cfg = WaveAbortConfig(enabled=True, allow_revoke=False)
    st = WaveAbortState()
    abort, _, st = wave_abort_on_tick(st, cfg=cfg, held_seconds=60, stock_signed=0.002)
    assert not abort and st.armed and st.done
    abort, _, st = wave_abort_on_tick(st, cfg=cfg, held_seconds=120, stock_signed=-0.01)
    assert not abort

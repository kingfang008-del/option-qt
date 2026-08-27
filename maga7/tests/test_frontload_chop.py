"""Unit tests for FRONTLOAD_CHOP label + weak sub-state overlay."""
from __future__ import annotations

from types import SimpleNamespace

from maga7.common.frontload_chop import (
    FrontloadChopConfig,
    FrontloadChopGate,
    parse_frontload_chop,
    weak_substate_ok,
)


def test_weak_substate_or_vixy_or_chop():
    cfg = FrontloadChopConfig(
        overlay="weak",
        overlay_combine="or",
        overlay_vixy_z_min=0.75,
        overlay_max_abs_qqq_fp=0.008,
    )
    assert weak_substate_ok(SimpleNamespace(vixy_z=1.2, qqq_from_prev=-0.02, qqq_day_flipped=False), cfg)
    assert weak_substate_ok(SimpleNamespace(vixy_z=-1.0, qqq_from_prev=0.003, qqq_day_flipped=False), cfg)
    assert not weak_substate_ok(
        SimpleNamespace(vixy_z=-1.0, qqq_from_prev=0.015, qqq_day_flipped=False), cfg
    )


def test_weak_substate_and():
    cfg = FrontloadChopConfig(
        overlay="weak",
        overlay_combine="and",
        overlay_vixy_z_min=0.5,
        overlay_max_abs_qqq_fp=0.01,
    )
    assert weak_substate_ok(SimpleNamespace(vixy_z=1.2, qqq_from_prev=-0.003, qqq_day_flipped=True), cfg)
    assert not weak_substate_ok(
        SimpleNamespace(vixy_z=1.2, qqq_from_prev=-0.02, qqq_day_flipped=False), cfg
    )


def test_gate_overlay_skips_strong_substate():
    class Inner:
        cfg = {}

        def check(self, direction, ts):
            from maga7.common.regime import RegimeDecision

            return RegimeDecision(
                allow=True,
                reason="ok",
                qqq_from_prev=0.02,
                vixy_z=-1.0,
                size_scale=1.0,
            )

    fl = FrontloadChopConfig(
        enabled=True,
        mode="scale",
        size_scale=0.5,
        overlay="weak",
        overlay_vixy_z_min=0.75,
        overlay_max_abs_qqq_fp=0.008,
    )
    import pandas as pd

    g = FrontloadChopGate(inner=Inner(), day_flags={"2026-07-21": True}, fl_cfg=fl)
    dec = g.check("UP", pd.Timestamp("2026-07-21 11:00:00", tz="America/New_York"))
    assert dec.allow and dec.size_scale == 1.0
    assert g.n_overlay_skip == 1
    assert g.n_scale == 0


def test_gate_overlay_scales_on_weak():
    class Inner:
        cfg = {}

        def check(self, direction, ts):
            from maga7.common.regime import RegimeDecision

            return RegimeDecision(
                allow=True,
                reason="ok",
                qqq_from_prev=-0.003,
                vixy_z=1.25,
                size_scale=1.0,
            )

    fl = FrontloadChopConfig(
        enabled=True,
        mode="scale",
        size_scale=0.5,
        overlay="weak",
        overlay_vixy_z_min=0.75,
    )
    import pandas as pd

    g = FrontloadChopGate(inner=Inner(), day_flags={"2026-07-22": True}, fl_cfg=fl)
    dec = g.check("DN", pd.Timestamp("2026-07-22 11:00:00", tz="America/New_York"))
    assert dec.allow and abs(dec.size_scale - 0.5) < 1e-9
    assert g.n_scale == 1


def test_parse_overlay_fields():
    cfg = parse_frontload_chop(
        {
            "enabled": True,
            "overlay": "weak",
            "overlay_vixy_z_min": 0.75,
            "overlay_max_abs_qqq_fp": 0.008,
            "overlay_combine": "or",
        }
    )
    assert cfg.overlay == "weak"
    assert cfg.overlay_vixy_z_min == 0.75
    assert cfg.overlay_max_abs_qqq_fp == 0.008

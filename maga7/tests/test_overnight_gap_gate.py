from __future__ import annotations

import pandas as pd

from maga7.common.overnight_gap_gate import (
    parse_overnight_gap_gate,
    resolve_overnight_gap_gate,
)


def _book(prev_close: float, day_open: float) -> pd.DataFrame:
    idx_prev = pd.date_range("2026-07-09 09:30", periods=2, freq="1min", tz="America/New_York")
    idx_day = pd.date_range("2026-07-10 09:30", periods=2, freq="1min", tz="America/New_York")
    return pd.DataFrame(
        {
            "date": ["2026-07-09", "2026-07-09", "2026-07-10", "2026-07-10"],
            "timestamp": list(idx_prev) + list(idx_day),
            "open": [prev_close, prev_close, day_open, day_open],
            "close": [prev_close, prev_close, day_open, day_open],
        }
    )


def test_block_fav_gap_up():
    cfg = parse_overnight_gap_gate(
        {"enabled": True, "max_fav_gap": 0.04, "mode": "block"}
    )
    # +5% gap into UP
    d = resolve_overnight_gap_gate(
        cfg, stock_df=_book(100.0, 105.0), date="2026-07-10", direction="UP"
    )
    assert not d.allow and d.fav_gap is not None and d.fav_gap > 0.04


def test_pass_small_gap_and_scale():
    cfg = parse_overnight_gap_gate(
        {"enabled": True, "max_fav_gap": 0.04, "mode": "scale", "scale": 0.5}
    )
    ok = resolve_overnight_gap_gate(
        cfg, stock_df=_book(100.0, 101.0), date="2026-07-10", direction="UP"
    )
    assert ok.allow and abs(ok.size_scale - 1.0) < 1e-12
    hot = resolve_overnight_gap_gate(
        cfg, stock_df=_book(100.0, 105.0), date="2026-07-10", direction="UP"
    )
    assert hot.allow and abs(hot.size_scale - 0.5) < 1e-12


def test_up_only_skips_dn():
    cfg = parse_overnight_gap_gate(
        {"enabled": True, "up_only": True, "mode": "degrade", "scale": 0.5, "max_fav_gap": 0.04}
    )
    assert cfg.dirs == ("UP",)
    # Large gap-up but DN trade → no degrade
    dn = resolve_overnight_gap_gate(
        cfg, stock_df=_book(100.0, 105.0), date="2026-07-10", direction="DN"
    )
    assert dn.allow and abs(dn.size_scale - 1.0) < 1e-12 and dn.reason == "dir_skip"
    up = resolve_overnight_gap_gate(
        cfg, stock_df=_book(100.0, 105.0), date="2026-07-10", direction="UP"
    )
    assert up.allow and abs(up.size_scale - 0.5) < 1e-12


def test_require_adv_share_filters_cool():
    cfg = parse_overnight_gap_gate(
        {
            "enabled": True,
            "up_only": True,
            "mode": "scale",
            "scale": 0.5,
            "max_fav_gap": 0.04,
            "require_adv_share": 0.55,
        }
    )
    cool = resolve_overnight_gap_gate(
        cfg,
        stock_df=_book(100.0, 105.0),
        date="2026-07-10",
        direction="UP",
        adv_share=0.39,
    )
    assert cool.allow and abs(cool.size_scale - 1.0) < 1e-12 and cool.reason == "gap_hot_adv_cool"
    hot = resolve_overnight_gap_gate(
        cfg,
        stock_df=_book(100.0, 105.0),
        date="2026-07-10",
        direction="UP",
        adv_share=0.56,
    )
    assert hot.allow and abs(hot.size_scale - 0.5) < 1e-12

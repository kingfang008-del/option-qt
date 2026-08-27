"""Unit tests for buyer-side VRP-lite soft prior."""
from __future__ import annotations

import numpy as np
import pandas as pd

from maga7.common.vrp_prior import (
    parse_vrp_size_scale,
    resolve_vrp_size_scale,
    trailing_rv_ann,
)


def test_parse_default_off():
    assert parse_vrp_size_scale(None).enabled is False
    cfg = parse_vrp_size_scale(
        {"enabled": True, "mode": "scale", "scale": 0.5, "rich_pctile": 0.7}
    )
    assert cfg.enabled and cfg.mode == "scale" and cfg.scale == 0.5


def test_trailing_rv_causal_excludes_today():
    closes = pd.Series(
        {
            "2026-05-01": 100.0,
            "2026-05-02": 101.0,
            "2026-05-03": 102.0,
            "2026-05-04": 103.0,
            "2026-05-05": 104.0,
            "2026-05-06": 120.0,  # huge jump — must not enter RV for 05-06
        }
    )
    rv = trailing_rv_ann(closes, date="2026-05-06", lookback_days=4)
    assert rv is not None and np.isfinite(rv)
    # With only calm days before 05-06, RV should be modest vs including +15% day
    closes2 = closes.copy()
    closes2["2026-05-05"] = 120.0
    rv2 = trailing_rv_ann(closes2, date="2026-05-06", lookback_days=4)
    assert rv2 is not None and rv2 > rv


def test_resolve_rich_scale_and_skip():
    day = pd.DataFrame(
        [
            {"date": "2026-05-10", "vrp": 0.12, "rich": True},
            {"date": "2026-05-11", "vrp": 0.01, "rich": False},
        ]
    )
    scale_cfg = parse_vrp_size_scale(
        {"enabled": True, "mode": "scale", "scale": 0.5}
    )
    s, reason = resolve_vrp_size_scale(scale_cfg, date="2026-05-10", day_table=day)
    assert s == 0.5 and "vrp_rich_scale" in reason
    s2, r2 = resolve_vrp_size_scale(scale_cfg, date="2026-05-11", day_table=day)
    assert s2 == 1.0 and r2 == "vrp_ok"

    skip_cfg = parse_vrp_size_scale({"enabled": True, "mode": "skip"})
    s3, r3 = resolve_vrp_size_scale(skip_cfg, date="2026-05-10", day_table=day)
    assert s3 == 0.0 and "vrp_rich_skip" in r3


def test_missing_passthrough():
    cfg = parse_vrp_size_scale({"enabled": True, "missing": "passthrough"})
    s, r = resolve_vrp_size_scale(cfg, date="2026-05-10", day_table=None)
    assert s == 1.0 and "passthrough" in r

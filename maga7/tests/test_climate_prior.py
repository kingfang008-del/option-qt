"""C2 climate prior — soft scale only, causal 10:30 features."""
from __future__ import annotations

import pandas as pd

from maga7.common.climate_prior import parse_climate_prior, resolve_climate_prior


def _day() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"date": "2026-02-10", "vixy_z_1030": 1.4, "mag7_frac_above_open": 0.50},
            {"date": "2026-05-10", "vixy_z_1030": -0.8, "mag7_frac_above_open": 0.875},
            {"date": "2026-06-01", "vixy_z_1030": 0.2, "mag7_frac_above_open": 0.50},
        ]
    )


def test_parse_default_off():
    assert parse_climate_prior(None).enabled is False
    cfg = parse_climate_prior({"enabled": True, "scale": 0.7, "combine": "or"})
    assert cfg.enabled and cfg.scale == 0.7 and cfg.combine == "or"


def test_or_scales_vixy_or_breadth_mid():
    cfg = parse_climate_prior({"enabled": True, "scale": 0.5, "combine": "or"})
    s, r = resolve_climate_prior(cfg, date="2026-02-10", day_table=_day())
    assert s == 0.5 and "vixy_high" in r
    s2, r2 = resolve_climate_prior(cfg, date="2026-06-01", day_table=_day())
    assert s2 == 0.5 and "breadth_mid" in r2
    s3, r3 = resolve_climate_prior(cfg, date="2026-05-10", day_table=_day())
    assert s3 == 1.0 and r3 == "climate_ok"


def test_and_requires_both():
    cfg = parse_climate_prior({"enabled": True, "scale": 0.5, "combine": "and"})
    s, _ = resolve_climate_prior(cfg, date="2026-02-10", day_table=_day())
    assert s == 0.5  # vixy high AND breadth mid
    s2, _ = resolve_climate_prior(cfg, date="2026-06-01", day_table=_day())
    assert s2 == 1.0  # breadth mid only


def test_missing_passthrough():
    cfg = parse_climate_prior({"enabled": True, "missing": "passthrough"})
    s, r = resolve_climate_prior(cfg, date="2026-02-10", day_table=None)
    assert s == 1.0 and "passthrough" in r

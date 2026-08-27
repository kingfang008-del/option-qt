from __future__ import annotations

import pandas as pd

from maga7.common.from_open_gate import (
    parse_from_open_gate,
    resolve_from_open_gate,
    session_from_open,
)


def _day() -> pd.DataFrame:
    idx = pd.date_range("2026-07-22 09:30", periods=120, freq="1min", tz="America/New_York")
    # Open 535, ramp to ~556 by 11:06 (~+4%)
    close = [535.0 + min(i, 96) * (21.0 / 96.0) for i in range(len(idx))]
    return pd.DataFrame(
        {
            "date": ["2026-07-22"] * len(idx),
            "timestamp": idx,
            "open": [535.0] + close[:-1],
            "close": close,
        }
    )


def test_parse_disabled_by_default():
    assert parse_from_open_gate(None).enabled is False


def test_session_from_open_at_1106():
    fo = session_from_open(
        _day(),
        date="2026-07-22",
        asof_ts=pd.Timestamp("2026-07-22 11:06", tz="America/New_York"),
    )
    assert fo is not None
    assert fo > 0.035


def test_block_same_sign_up():
    cfg = parse_from_open_gate(
        {"enabled": True, "max_abs": 0.04, "mode": "block", "same_sign_only": True}
    )
    act, mult, fo = resolve_from_open_gate(cfg, from_open=0.041, direction="UP")
    assert act == "block" and mult == 0.0 and fo == 0.041
    act2, mult2, _ = resolve_from_open_gate(cfg, from_open=0.041, direction="DN")
    assert act2 == "allow" and abs(mult2 - 1.0) < 1e-12


def test_scale_mode():
    cfg = parse_from_open_gate(
        {"enabled": True, "max_abs": 0.035, "mode": "scale", "scale": 0.5}
    )
    act, mult, _ = resolve_from_open_gate(cfg, from_open=0.04, direction="UP")
    assert act == "scale" and abs(mult - 0.5) < 1e-12
    act2, mult2, _ = resolve_from_open_gate(cfg, from_open=0.02, direction="UP")
    assert act2 == "allow" and abs(mult2 - 1.0) < 1e-12

import pandas as pd

from maga7.common.regime import Mag7RegimeGate, _signed


def _frame() -> pd.DataFrame:
    """Two sessions: day1 QQQ red, day2 QQQ green (day flip)."""
    rows = []
    for d, fp, mf, vz in [
        ("2026-05-19", -0.01, -1e6, -1.0),
        ("2026-05-20", 0.012, 1e6, -1.2),
    ]:
        ts = pd.date_range(f"{d} 10:30", periods=5, freq="1min", tz="America/New_York")
        for t in ts:
            rows.append(
                {
                    "timestamp": t,
                    "date": d,
                    "qqq_close": 500.0,
                    "qqq_from_prev": fp,
                    "qqq_mf10": mf,
                    "vixy_z": vz,
                    "vix_reversal_count_30m": 2.0,
                }
            )
    df = pd.DataFrame(rows).set_index("timestamp")
    return df


def test_signed_eps():
    assert _signed(0.001, 0.005) == 0
    assert _signed(-0.01, 0.0) == -1


def test_day_flip_block():
    gate = Mag7RegimeGate(
        frame=_frame(),
        cfg={
            "enabled": True,
            "qqq_align": True,
            "qqq_day_flip_mode": "block",
        },
    )
    # Day1: no prior → allow DN
    d1 = gate.check("DN", pd.Timestamp("2026-05-19 10:32", tz="America/New_York"))
    assert d1.allow and d1.reason == "ok"
    # Day2 UP after red→green flip → block
    d2 = gate.check("UP", pd.Timestamp("2026-05-20 10:32", tz="America/New_York"))
    assert not d2.allow and d2.reason == "qqq_day_flip"
    assert d2.qqq_day_flipped


def test_day_flip_scale():
    gate = Mag7RegimeGate(
        frame=_frame(),
        cfg={
            "enabled": True,
            "qqq_align": True,
            "qqq_day_flip_mode": "scale",
            "qqq_day_flip_scale": 0.5,
        },
    )
    d2 = gate.check("UP", pd.Timestamp("2026-05-20 10:32", tz="America/New_York"))
    assert d2.allow and d2.size_scale == 0.5 and d2.reason == "qqq_day_flip_scale"


def test_put_vixy_z_blocks_dn():
    gate = Mag7RegimeGate(
        frame=_frame(),
        cfg={"enabled": True, "qqq_align": True, "put_vixy_z_min": 0.0},
    )
    d1 = gate.check("DN", pd.Timestamp("2026-05-19 10:32", tz="America/New_York"))
    assert not d1.allow and d1.reason == "put_vixy_z"

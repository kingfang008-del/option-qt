import pandas as pd

from maga7.common.signals import _rule_a_mask, attach_mf_features, first_rule_a_day


def _synth_day(*, n: int = 80, up_net: bool = True) -> pd.DataFrame:
    """Build a day where mf10 turns positive inside the 10:30+ window and streaks grow."""
    # Start at 10:20 so after mf10 warmup the streak builds inside Rule-A hours.
    ts = pd.date_range("2026-05-01 10:20", periods=n, freq="1min", tz="America/New_York")
    close = 100.0
    rows = []
    for i, t in enumerate(ts):
        # After bar 12 (~10:32), push strong buy flow so mf10>0 and streaks grow.
        bull = up_net and i >= 12
        high = close + 1.0
        low = close - 1.0
        c = close + (0.8 if bull else -0.2)
        vol = 10_000.0
        rows.append(
            {
                "timestamp": t,
                "open": close,
                "high": high,
                "low": low,
                "close": c,
                "volume": vol,
            }
        )
        close = c
    df = pd.DataFrame(rows)
    df["date"] = df["timestamp"].dt.strftime("%Y-%m-%d")
    df["tod"] = df["timestamp"].dt.strftime("%H:%M")
    return attach_mf_features(df, mf_window=10, mf_fast_window=3)


def test_mf_fast_column_present():
    df = _synth_day()
    assert "mf_fast" in df.columns and "mf_short" in df.columns
    assert df["mf_fast"].equals(df["mf_short"])


def test_early_path_fires_before_full_streak():
    df = _synth_day()
    day = df[df["date"] == "2026-05-01"].copy()
    # Force from_prev / vol_z / cum so only streak+fast gates matter.
    day["from_prev"] = 0.03
    day["vol_z"] = 2.0
    day["cum"] = day["net$"].cumsum().clip(lower=1.0)

    late = first_rule_a_day(
        day,
        streak_min=8,
        from_prev_abs=0.02,
        vol_z_min=1.0,
        early_on_mf_fast=False,
    )
    early = first_rule_a_day(
        day,
        streak_min=8,
        from_prev_abs=0.02,
        vol_z_min=1.0,
        early_on_mf_fast=True,
        streak_min_fast=5,
    )
    assert late is not None and early is not None
    assert early["sig_ts"] < late["sig_ts"]


def test_early_mask_requires_fast_align():
    df = _synth_day()
    day = df[df["date"] == "2026-05-01"].copy()
    day["from_prev"] = 0.03
    day["vol_z"] = 2.0
    day["cum"] = 1.0
    # Corrupt fast window to opposite sign while streak is mid-range.
    day.loc[day["streak_up"].between(5, 7), "mf_fast"] = -1.0
    day.loc[day["streak_up"].between(5, 7), "mf_short"] = -1.0
    m = _rule_a_mask(
        day,
        direction="UP",
        streak_min=8,
        from_prev_abs=0.02,
        vol_z_min=1.0,
        streak_max=None,
        require_mf_short_align=False,
        early_on_mf_fast=True,
        streak_min_fast=5,
    )
    # Mid streaks must not fire early without fast align; streak>=8 still can.
    mid = day["streak_up"].between(5, 7)
    assert not bool(m[mid].any())

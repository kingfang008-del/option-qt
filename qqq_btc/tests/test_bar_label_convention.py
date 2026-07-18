from __future__ import annotations

from pathlib import Path

import pandas as pd

from qqq_btc.common.bar_label_convention import (
    EXPECTED_FIRST_RTH,
    fix_parquet_to_right_label,
    inspect_parquet_label,
)


def _write_month(tmp_path: Path, first_hhmm: str) -> Path:
    # 09:30 or 09:31 NY on 2026-07-01
    tmp_path.mkdir(parents=True, exist_ok=True)
    start = pd.Timestamp(f"2026-07-01 {first_hhmm}:00", tz="America/New_York")
    idx = pd.date_range(start, periods=5, freq="1min")
    df = pd.DataFrame(
        {
            "timestamp": idx,
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "volume": 1000,
        }
    )
    path = tmp_path / "2026-07.parquet"
    df.to_parquet(path, index=False)
    return path


def test_inspect_left_and_right(tmp_path: Path):
    left = _write_month(tmp_path / "left", "09:30")
    right = _write_month(tmp_path / "right", "09:31")
    # nest like resampled layout for symbol parse (optional)
    assert inspect_parquet_label(left).label == "left"
    assert inspect_parquet_label(left).ok_for_w1 is False
    assert inspect_parquet_label(right).label == "right"
    assert inspect_parquet_label(right).ok_for_w1 is True


def test_fix_left_to_right(tmp_path: Path):
    path = _write_month(tmp_path, "09:30")
    rep = fix_parquet_to_right_label(path, dry_run=False, backup=True)
    assert rep["status"] == "fixed"
    after = inspect_parquet_label(path)
    assert after.label == "right"
    assert after.first_hhmm == EXPECTED_FIRST_RTH
    assert list(tmp_path.glob("*.bak_left_label_*"))

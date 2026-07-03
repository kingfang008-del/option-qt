#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Assert replay_live_parity_utils.index_roc matches pandas pct_change(300) (S4 definition)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _bootstrap():
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "production" / "baseline"))


def test_index_roc_matches_pandas_pct_change_300() -> None:
    _bootstrap()
    from replay_live_parity_utils import index_roc_5min_at_index, index_roc_5min_from_series

    np.random.seed(42)
    n = 500
    closes = 100.0 * np.cumprod(1.0 + np.random.randn(n) * 0.0003)
    s = pd.Series(closes)
    pct = s.pct_change(periods=300).fillna(0.0)

    for i in (300, 350, 400, 499):
        exp = float(pct.iloc[i])
        got = index_roc_5min_at_index(closes, i, periods=300)
        assert abs(got - exp) < 1e-9, f"i={i} got={got} exp={exp}"

    last = index_roc_5min_from_series(closes, periods=300)
    assert abs(last - float(pct.iloc[-1])) < 1e-9


def test_index_roc_short_series_is_zero() -> None:
    _bootstrap()
    from replay_live_parity_utils import index_roc_5min_from_series

    assert index_roc_5min_from_series([100.0, 101.0], periods=300) == 0.0
    assert index_roc_5min_from_series(np.array([])) == 0.0


def main() -> None:
    test_index_roc_matches_pandas_pct_change_300()
    test_index_roc_short_series_is_zero()
    print("[OK] index roc s4 parity tests passed")


if __name__ == "__main__":
    main()

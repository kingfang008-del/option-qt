from __future__ import annotations

import pandas as pd

from maga7.common.chop_gate import (
    ChopGateConfig,
    compute_chop_features,
    is_chop_day,
    load_chop_gate,
)


def _bars(
    open_px: float,
    close_px: float,
    *,
    high: float | None = None,
    low: float | None = None,
) -> pd.DataFrame:
    idx = pd.date_range("2026-07-22 09:30", periods=61, freq="1min", tz="America/New_York")
    closes = [open_px + (close_px - open_px) * i / 60 for i in range(61)]
    h = high if high is not None else max(open_px, close_px) * 1.002
    l = low if low is not None else min(open_px, close_px) * 0.998
    return pd.DataFrame(
        {
            "date": ["2026-07-22"] * len(idx),
            "timestamp": idx,
            "open": [open_px] + closes[:-1],
            "high": [h] * len(idx),
            "low": [l] * len(idx),
            "close": closes,
        }
    )


def test_stock_noise_detects_chop():
    # Flat QQQ, Mag7 mixed with large |from_open|
    qqq = _bars(500.0, 501.0)
    names = ["NVDA", "AMD", "MSFT", "META", "TSLA", "AAPL", "AMZN", "GOOGL"]
    stock_by = {"QQQ": qqq}
    # 3 up / 5 down → frac=0.375; large moves
    for i, s in enumerate(names):
        stock_by[s] = _bars(100.0, 102.0 if i < 3 else 97.5)
    feats = compute_chop_features(
        date="2026-07-22",
        stock_by=stock_by,
        qqq_df=qqq,
        symbols=names,
        asof="10:30",
    )
    cfg = ChopGateConfig(enabled=True)  # defaults: stock_noise
    hit, _ = is_chop_day(cfg, feats)
    assert hit is True
    assert feats["med_abs"] is not None and feats["med_abs"] >= 0.01


def test_trend_day_not_chop():
    qqq = _bars(500.0, 506.0)  # |q_am| large
    stock_by = {"QQQ": qqq, "NVDA": _bars(100.0, 102.0), "AMD": _bars(100.0, 97.0)}
    feats = compute_chop_features(
        date="2026-07-22",
        stock_by=stock_by,
        qqq_df=qqq,
        symbols=["NVDA", "AMD"],
        asof="10:30",
    )
    cfg = ChopGateConfig(enabled=True)
    hit, reason = is_chop_day(cfg, feats)
    assert hit is False
    assert reason == "q_am_trend"


def test_scale_and_block_modes():
    qqq = _bars(500.0, 501.0)
    names = ["NVDA", "AMD", "MSFT", "META", "TSLA", "AAPL", "AMZN", "GOOGL"]
    stock_by = {"QQQ": qqq}
    for i, s in enumerate(names):
        stock_by[s] = _bars(100.0, 102.0 if i < 3 else 97.5)

    soft = load_chop_gate({"chop_gate": {"enabled": True, "mode": "scale", "scale": 0.5}})
    soft.begin_day("2026-07-22", stock_by=stock_by, qqq_df=qqq, symbols=names)
    d = soft.decide_entry("UP")
    assert d.allow and abs(d.size_scale - 0.5) < 1e-9 and d.state == "chop"

    hard = load_chop_gate({"chop_gate": {"enabled": True, "mode": "block"}})
    hard.begin_day("2026-07-22", stock_by=stock_by, qqq_df=qqq, symbols=names)
    d2 = hard.decide_entry("DN")
    assert (not d2.allow) and d2.state == "chop"

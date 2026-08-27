from __future__ import annotations

import pandas as pd

from maga7.common.session_flow_gate import (
    SessionFlowGateConfig,
    cumflow_ranks,
    is_index_chop,
    load_session_flow_gate,
)


def _day(
    open_px: float,
    close_at_1030: float,
    *,
    net_per_bar: float = 1e6,
    n: int = 61,
) -> pd.DataFrame:
    idx = pd.date_range("2026-07-22 09:30", periods=n, freq="1min", tz="America/New_York")
    closes = [open_px + (close_at_1030 - open_px) * i / (n - 1) for i in range(n)]
    nets = [net_per_bar] * n
    out = pd.DataFrame(
        {
            "date": ["2026-07-22"] * n,
            "timestamp": idx,
            "open": [open_px] + closes[:-1],
            "high": [max(open_px, close_at_1030) * 1.001] * n,
            "low": [min(open_px, close_at_1030) * 0.999] * n,
            "close": closes,
            "net$": nets,
        }
    )
    out["cum"] = out["net$"].cumsum()
    return out


def test_index_chop_qqq_only_and_vixy_failsoft():
    feats = {"q_am": 0.002, "vixy_am": None}
    cfg = SessionFlowGateConfig(enabled=True, q_am_max=0.005, vixy_am_max=0.015)
    hit, reason = is_index_chop(cfg, feats)
    assert hit is True
    assert "chop" in reason

    feats2 = {"q_am": 0.01, "vixy_am": 0.0}
    hit2, reason2 = is_index_chop(cfg, feats2)
    assert hit2 is False and reason2 == "q_am_trend"


def test_cumflow_ranks_top_k():
    stock_by = {
        "NVDA": _day(100.0, 101.0, net_per_bar=3e6),
        "AMD": _day(100.0, 101.0, net_per_bar=1e6),
        "MSFT": _day(100.0, 99.0, net_per_bar=-2e6),
    }
    asof = pd.Timestamp("2026-07-22 10:30:00", tz="America/New_York")
    ranks = cumflow_ranks(
        date="2026-07-22",
        asof_ts=asof,
        stock_by=stock_by,
        symbols=["NVDA", "AMD", "MSFT"],
    )
    assert ranks["NVDA"][1] == 1
    assert ranks["MSFT"][1] == 2
    assert ranks["AMD"][1] == 3
    assert ranks["NVDA"][0] > 0
    assert ranks["MSFT"][0] < 0


def test_chop_and_leader_block_non_leader():
    qqq = _day(500.0, 501.0, net_per_bar=0.0)  # flat → chop
    vixy = _day(20.0, 20.1, net_per_bar=0.0)
    stock_by = {
        "QQQ": qqq,
        "VIXY": vixy,
        "NVDA": _day(100.0, 101.0, net_per_bar=5e6),
        "AMD": _day(100.0, 100.5, net_per_bar=1e6),
        "MSFT": _day(100.0, 99.5, net_per_bar=-4e6),
    }
    gate = load_session_flow_gate(
        {
            "session_flow_gate": {
                "enabled": True,
                "when": "chop_only",
                "mode": "block",
                "top_k": 1,
                "q_am_max": 0.005,
                "vixy_am_max": 0.015,
            }
        }
    )
    day = gate.begin_day(
        "2026-07-22",
        stock_by=stock_by,
        qqq_df=qqq,
        vixy_df=vixy,
        symbols=["NVDA", "AMD", "MSFT"],
    )
    assert day.state == "chop"
    asof = pd.Timestamp("2026-07-22 10:35:00", tz="America/New_York")
    ok = gate.decide_entry(symbol="NVDA", direction="UP", asof_ts=asof)
    assert ok.allow and ok.rank == 1
    bad = gate.decide_entry(symbol="AMD", direction="UP", asof_ts=asof)
    assert not bad.allow
    sign = gate.decide_entry(symbol="MSFT", direction="UP", asof_ts=asof)
    # MSFT may be top by |cum| if top_k=1 only NVDA; with top_k=2 MSFT is rank2
    gate2 = load_session_flow_gate(
        {
            "session_flow_gate": {
                "enabled": True,
                "when": "chop_only",
                "mode": "block",
                "top_k": 2,
            }
        }
    )
    gate2.begin_day(
        "2026-07-22",
        stock_by=stock_by,
        qqq_df=qqq,
        vixy_df=vixy,
        symbols=["NVDA", "AMD", "MSFT"],
    )
    sign2 = gate2.decide_entry(symbol="MSFT", direction="UP", asof_ts=asof)
    assert not sign2.allow
    assert "sign_mismatch" in sign2.reason
    put = gate2.decide_entry(symbol="MSFT", direction="DN", asof_ts=asof)
    assert put.allow


def test_trend_day_passes_chop_only():
    qqq = _day(500.0, 510.0)  # strong AM move
    stock_by = {
        "QQQ": qqq,
        "NVDA": _day(100.0, 101.0, net_per_bar=1e5),
        "AMD": _day(100.0, 101.0, net_per_bar=9e6),
    }
    gate = load_session_flow_gate(
        {"session_flow_gate": {"enabled": True, "when": "chop_only", "mode": "block", "top_k": 1}}
    )
    day = gate.begin_day(
        "2026-07-22",
        stock_by=stock_by,
        qqq_df=qqq,
        vixy_df=None,
        symbols=["NVDA", "AMD"],
    )
    assert day.state == "trend"
    asof = pd.Timestamp("2026-07-22 10:35:00", tz="America/New_York")
    d = gate.decide_entry(symbol="NVDA", direction="UP", asof_ts=asof)
    assert d.allow and abs(d.size_scale - 1.0) < 1e-12


def test_boost_mode_upsizes_leader_only():
    qqq = _day(500.0, 501.0)
    stock_by = {
        "QQQ": qqq,
        "NVDA": _day(100.0, 101.0, net_per_bar=5e6),
        "AMD": _day(100.0, 100.5, net_per_bar=1e6),
    }
    gate = load_session_flow_gate(
        {
            "session_flow_gate": {
                "enabled": True,
                "when": "chop_only",
                "mode": "boost",
                "boost": 1.5,
                "non_leader_scale": 0.75,
                "top_k": 1,
                "q_am_max": 0.005,
                "vixy_am_max": None,
            }
        }
    )
    day = gate.begin_day(
        "2026-07-22",
        stock_by=stock_by,
        qqq_df=qqq,
        vixy_df=None,
        symbols=["NVDA", "AMD"],
    )
    assert day.state == "chop"
    asof = pd.Timestamp("2026-07-22 10:35:00", tz="America/New_York")
    lead = gate.decide_entry(symbol="NVDA", direction="UP", asof_ts=asof)
    assert lead.allow and abs(lead.size_scale - 1.5) < 1e-12
    other = gate.decide_entry(symbol="AMD", direction="UP", asof_ts=asof)
    assert other.allow and abs(other.size_scale - 0.75) < 1e-12

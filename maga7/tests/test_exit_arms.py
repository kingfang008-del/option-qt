"""Exit-arm snapshot + health suggestions."""
from __future__ import annotations

from maga7.common.exit_arms import build_exit_arms, build_exit_health
from maga7.common.hold_watchdog import qqq_adverse_from_prices


def test_build_exit_arms_flags():
    arms = build_exit_arms(
        {
            "sl_mult": 0.45,
            "tp_mult": 1.6,
            "hold_minutes": 30,
            "day_circuit": None,
            "trade_toxic": {
                "enabled": True,
                "cut_ret": 0.25,
                "max_cut_seconds": 600,
            },
            "hold_watchdog": {"enabled": False, "qqq_adverse_from_entry": 0.008},
        },
        reason_counts={"SL": 1, "TRADE_TOX": 2},
    )
    assert arms["trade_toxic"]["enabled"] is True
    assert arms["trade_toxic"]["n_triggers"] == 2
    assert arms["hold_watchdog"]["enabled"] is False
    assert arms["sl_tp"]["n_sl"] == 1
    assert arms["day_circuit"]["enabled"] is False


def test_exit_health_suggestions_early_heavy():
    health = build_exit_health(
        {"TRADE_TOX": 2, "SL": 1},
        arms={"trade_toxic": {"n_triggers": 2}},
    )
    assert health["n_closes"] == 3
    assert health["n_early_cut"] == 2
    assert health["auto_disable"] is False
    assert any("early_cut" in s for s in health["suggestions"])


def test_qqq_adverse_from_prices():
    fired, signed = qqq_adverse_from_prices(
        entry_px=100.0, now_px=99.0, direction="UP", thresh=0.008
    )
    assert fired
    assert signed is not None and signed < -0.009

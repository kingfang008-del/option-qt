"""Live OMS hold_watchdog (HOLD_SHOCK) + exit_arms publish."""
from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from maga7.live.broker_oms import LivePosition, Mag7BrokerOms


def _oms(tmp_path: Path, *, hold_wd_on: bool = True) -> Mag7BrokerOms:
    class FakeRedis:
        def hget(self, *args):
            return b"0"

        def xadd(self, *args, **kwargs):
            return b"1-0"

        def pipeline(self, transaction=True):
            return self

        def set(self, *args, **kwargs):
            return self

        def delete(self, *args, **kwargs):
            return self

        def hset(self, *args, **kwargs):
            return self

        def execute(self):
            return []

    class Scanner:
        def record_fill(self, *args, **kwargs):
            return None

        states = {"AMD": SimpleNamespace(bars=[{"close": 516.0}], mf10=0.0)}
        stock_by = {
            "QQQ": pd.DataFrame(
                [
                    {
                        "timestamp": pd.Timestamp(
                            "2026-07-20 10:31:00", tz="America/New_York"
                        ),
                        "close": 700.0,
                        "date": "2026-07-20",
                    },
                    {
                        "timestamp": pd.Timestamp(
                            "2026-07-20 10:40:00", tz="America/New_York"
                        ),
                        "close": 693.0,  # −1% from 700
                        "date": "2026-07-20",
                    },
                ]
            )
        }

    profile = {
        "trade": {
            "hold_minutes": 30,
            "tp_mult": 1.6,
            "sl_mult": 0.45,
            "exit_mode": "hold_extend",
            "exit_mf_grace_seconds": 60,
            "position_frac": 0.2,
            "position_sizing": "fixed",
            "trade_toxic": {"enabled": False},
            "hold_watchdog": {
                "enabled": hold_wd_on,
                "qqq_adverse_from_entry": 0.008,
                "min_hold_seconds": 60,
            },
            "risk": {
                "max_stock_staleness_sec": 30.0,
                "max_option_staleness_sec": 30.0,
                "max_exit_mid_jump_pct": 0.9,
                "max_gap_hold_ticks": 3,
            },
        },
        "fill": {"entry_frac": 0.8, "exit_frac": 0.8},
        "signal": {"top_k": 2},
    }
    connector = SimpleNamespace(
        ib=SimpleNamespace(isConnected=lambda: True),
        redis=FakeRedis(),
        config=SimpleNamespace(
            port=4002,
            account="DU1",
            max_stock_staleness_sec=30.0,
            max_option_staleness_sec=30.0,
        ),
        lock_status="LOCKED",
        data_mode="LIVE",
        option_quotes={},
        last_stock_tick={"AMD": time.time(), "QQQ": time.time()},
        locks={},
        ensure_option_subscription=lambda *_: True,
        release_on_demand_subscription=lambda *_: None,
    )
    return Mag7BrokerOms(
        profile=profile,
        scanner=Scanner(),
        connector=connector,
        session_id="holdwdsession01",
        trade_date="2026-07-20",
        session_dir=tmp_path,
        mode="shadow",
        equity=100_000.0,
    )


def test_hold_shock_flattens_when_enabled(tmp_path):
    oms = _oms(tmp_path, hold_wd_on=True)
    now = time.time()
    oms.positions["AMD"] = LivePosition(
        symbol="AMD",
        contract="AMD260720C00517500",
        con_id=1,
        direction="UP",
        qty=1,
        entry_price=4.60,
        entry_ts=now - 180,
        signal_ts=now - 200,
        rank=1,
        qty_frac=0.2,
        entry_bid=4.50,
        entry_ask=4.70,
        last_good_mid=4.60,
        entry_stock_px=516.0,
        entry_qqq_px=700.0,
    )
    # Option still above SL (sl=0.45 → 2.07); mid ~4.0
    oms.connector.option_quotes[("AMD", "AMD260720C00517500")] = {
        "bid": 3.90,
        "ask": 4.10,
        "ts": now,
    }
    events: list[tuple[str, dict]] = []
    oms._event = lambda kind, payload: events.append((kind, payload))  # type: ignore
    oms.evaluate_exits(now)
    assert "AMD" not in oms.positions
    assert any(
        kind == "POSITION_CLOSE" and payload.get("reason") == "HOLD_SHOCK"
        for kind, payload in events
    )
    assert oms.exit_reason_counts.get("HOLD_SHOCK") == 1
    arms = oms.exit_arms_snapshot()
    assert arms["hold_watchdog"]["enabled"] is True
    assert arms["hold_watchdog"]["n_triggers"] == 1
    health_path = Path(tmp_path) / "exit_health.json"
    assert health_path.is_file()
    doc = json.loads(health_path.read_text())
    assert doc["exit_arms"]["hold_watchdog"]["n_triggers"] == 1


def test_hold_shock_off_does_not_cut(tmp_path):
    oms = _oms(tmp_path, hold_wd_on=False)
    now = time.time()
    oms.positions["AMD"] = LivePosition(
        symbol="AMD",
        contract="AMD260720C00517500",
        con_id=1,
        direction="UP",
        qty=1,
        entry_price=4.60,
        entry_ts=now - 180,
        signal_ts=now - 200,
        rank=1,
        qty_frac=0.2,
        entry_bid=4.50,
        entry_ask=4.70,
        last_good_mid=4.60,
        entry_qqq_px=700.0,
    )
    oms.connector.option_quotes[("AMD", "AMD260720C00517500")] = {
        "bid": 3.90,
        "ask": 4.10,
        "ts": now,
    }
    oms.evaluate_exits(now)
    assert "AMD" in oms.positions

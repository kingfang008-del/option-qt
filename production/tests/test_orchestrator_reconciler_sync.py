#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


def _bootstrap_imports() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    production_dir = repo_root / "production"
    baseline_dir = production_dir / "baseline"
    sys.path.insert(0, str(production_dir))
    sys.path.insert(0, str(baseline_dir))


class _State:
    def __init__(self) -> None:
        self.position = 0
        self.qty = 0
        self.entry_price = 0.0
        self.entry_ts = 0.0
        self.last_opt_price = 0.0
        self.last_price = 123.45
        self.is_pending = False
        self.locked_cash = 0.0
        self.entry_slot_reserved = False
        self.open_fill_confirmed = False


class _DummyIB:
    def isConnected(self) -> bool:
        return True

    def positions(self):
        return []


class _DummyIBKR:
    def __init__(self) -> None:
        self.ib = _DummyIB()

    async def get_account_balance(self):
        return 0.0


class _DummyAccounting:
    def __init__(self) -> None:
        self.logs = []
        self.broadcast_calls = 0

    def _emit_trade_log(self, payload):
        self.logs.append(payload)

    def _schedule_live_state_broadcast(self):
        self.broadcast_calls += 1


class _DummyStateManager:
    def __init__(self) -> None:
        self.save_calls = 0

    def save_state(self):
        self.save_calls += 1


def _build_orch():
    st = _State()
    return SimpleNamespace(
        mode="realtime",
        states={"NVDA": st},
        ibkr=_DummyIBKR(),
        accounting=_DummyAccounting(),
        state_manager=_DummyStateManager(),
        r=None,
        trading_paused=False,
    )


def test_auto_heal_divergence_schedules_live_broadcast() -> None:
    _bootstrap_imports()
    import orchestrator_reconciler as reconciler_mod  # noqa: E402

    orch = _build_orch()
    st = orch.states["NVDA"]
    reconciler = reconciler_mod.OrchestratorReconciler(orch)

    asyncio.run(reconciler._auto_heal_divergence("NVDA", st, local_qty=0, broker_qty=2, broker_entry_price=1.75))

    assert st.position == 1
    assert st.qty == 2
    assert orch.state_manager.save_calls == 1
    assert orch.accounting.broadcast_calls == 1


def test_reconciliation_loop_runs_even_when_runtime_trading_disabled() -> None:
    _bootstrap_imports()
    import orchestrator_reconciler as reconciler_mod  # noqa: E402

    orch = _build_orch()
    reconciler = reconciler_mod.OrchestratorReconciler(orch)
    reconciler._runtime_trading_enabled = lambda: False
    called = {"count": 0}
    sleep_calls = {"count": 0}

    async def _fake_perform():
        called["count"] += 1

    async def _fake_sleep(_secs: float):
        sleep_calls["count"] += 1
        if sleep_calls["count"] > 1:
            raise RuntimeError("stop loop")
        return None

    reconciler._perform_reconciliation = _fake_perform

    try:
        with patch.object(reconciler_mod.asyncio, "sleep", _fake_sleep):
            asyncio.run(reconciler.run_reconciliation_loop())
    except RuntimeError as exc:
        assert "stop loop" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("预期用 stop loop 终止无限循环")

    assert called["count"] == 1, "DISARMED 时也应继续执行 broker 对账"


def test_cash_reconciliation_noops_without_warning_or_pause() -> None:
    _bootstrap_imports()
    import orchestrator_reconciler as reconciler_mod  # noqa: E402

    orch = _build_orch()
    orch.mock_cash = 1273.36

    async def _high_balance():
        return 999999.0

    orch.ibkr.get_account_balance = _high_balance
    reconciler = reconciler_mod.OrchestratorReconciler(orch)

    asyncio.run(reconciler._perform_cash_reconciliation())

    assert orch.trading_paused is False
    assert orch.accounting.logs == []


def main() -> None:
    test_auto_heal_divergence_schedules_live_broadcast()
    test_reconciliation_loop_runs_even_when_runtime_trading_disabled()
    test_cash_reconciliation_noops_without_warning_or_pause()
    print("[OK] orchestrator reconciler sync guards passed")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
回归测试：执行链路关键守卫。

覆盖点：
1) _execute_entry 在早期异常路径不应触发 UnboundLocalError；
2) Dry 平仓应优先使用 bid_size 约束成交量（PUT 亦然）；
3) 执行参数可由 self.orch.cfg 覆盖（不被 config 常量锁死）。
"""

from __future__ import annotations

import asyncio
import os
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
        self.symbol = "NVDA"
        self.position = 0
        self.qty = 0
        self.entry_stock = 0.0
        self.entry_price = 0.0
        self.entry_ts = 0.0
        self.max_roi = 0.0
        self.locked_cash = 0.0
        self.last_valid_iv = 0.35
        self.contract_id = "NVDA260313C00180000"
        self.strike_price = 180.0
        self.expiry_date = None
        self.opt_type = "call"
        self.entry_spy_roc = 0.0
        self.entry_index_trend = 0
        self.entry_alpha_z = 0.0
        self.entry_iv = 0.0
        self.is_pending = False
        self.pending_action = ""
        self.last_opt_price = 2.0
        self.cooldown_until = 0.0


class _DummyRedis:
    def get(self, _key):
        return None


class _DummyIBKR:
    def __init__(self) -> None:
        self.locked_contracts = {}
        self.orders = []
        self.ib = SimpleNamespace(cancelOrder=lambda *_args, **_kwargs: None)

    def place_option_order(self, *args, **kwargs):
        self.orders.append((args, kwargs))
        return None


class _DummyAccounting:
    def __init__(self) -> None:
        self.exit_calls = []

    def _process_open_accounting(self, *args, **kwargs):
        return None

    def _process_exit_accounting(self, *args, **kwargs):
        self.exit_calls.append((args, kwargs))


class _DummyStateManager:
    def save_state(self):
        return None


def _build_orch(mode: str = "realtime") -> SimpleNamespace:
    st = _State()
    cfg = SimpleNamespace(
        MAX_POSITIONS=4,
        POSITION_RATIO=0.5,
        MAX_TRADE_CAP=100000.0,
        GLOBAL_EXPOSURE_LIMIT=0.9,
        COMMISSION_PER_CONTRACT=0.65,
        LIMIT_BUFFER_ENTRY=1.03,
        LIMIT_BUFFER_EXIT=0.97,
        ORDER_TIMEOUT_SECONDS=7,
        ORDER_MAX_RETRIES=2,
        EXIT_ORDER_TYPE="LMT",
        SLIPPAGE_PCT=0.001,
        ENTRY_MAX_REQUOTE_SLIPPAGE_PCT=0.02,
        STOP_LOSS=-0.2,
    )
    orch = SimpleNamespace(
        mode=mode,
        states={"NVDA": st},
        cfg=cfg,
        mock_cash=50000.0,
        global_cooldown_until=0.0,
        MIN_OPTION_PRICE=0.5,
        r=_DummyRedis(),
        ibkr=_DummyIBKR(),
        accounting=_DummyAccounting(),
        state_manager=_DummyStateManager(),
        strategy=SimpleNamespace(cfg=SimpleNamespace(STOP_LOSS=-0.2)),
        use_shared_mem=False,
    )
    orch.ib = orch.ibkr.ib
    orch._get_fair_market_price = lambda base, bid, ask, prev=0.0: float(base)
    return orch


def test_execute_entry_scope_guard_no_unboundlocalerror() -> None:
    _bootstrap_imports()
    import orchestrator_execution as oe  # noqa: E402

    orch = _build_orch(mode="realtime")
    ex = oe.OrchestratorExecution(orch)

    sig = {
        "tag": "CALL_ATM",
        "dir": 1,
        "reason": "UNIT_TEST",
        "price": 2.0,
        "meta": {"iv": 0.3, "contract_id": "NVDA260313C00180000", "strike": 180.0},
    }

    with patch.object(oe.LiquidityRiskManager, "evaluate_order", return_value={"final_alloc": 3000.0, "chunks": 1, "reason": "unit"}):
        with patch.object(oe.OrchestratorExecution, "_get_entry_limit_price", side_effect=RuntimeError("limit calc failed")):
            try:
                asyncio.run(ex._execute_entry("NVDA", sig, stock_price=100.0, curr_ts=1_777_777_777.0, batch_idx=0))
            except RuntimeError as e:
                assert "limit calc failed" in str(e)
            except UnboundLocalError as e:  # pragma: no cover - 失败时才会进入
                raise AssertionError(f"不应再触发 trade 作用域错误: {e}") from e
            else:  # pragma: no cover
                raise AssertionError("预期应抛出 RuntimeError(limit calc failed)")


def test_execute_exit_put_uses_bid_size_in_dry_mode() -> None:
    _bootstrap_imports()
    import orchestrator_execution as oe  # noqa: E402

    orch = _build_orch(mode="realtime")
    st = orch.states["NVDA"]
    st.position = -1
    st.qty = 10
    st.entry_price = 2.2
    st.entry_ts = 1_777_777_000.0

    ex = oe.OrchestratorExecution(orch)
    sig = {
        "reason": "CT_TIMEOUT:5m",
        "price": 2.0,
        "market_price": 2.0,
        "bid": 1.95,
        "ask": 2.05,
        "bid_size": 3,
        "ask_size": 99,
        "original_position": -1,
    }

    with patch.dict(os.environ, {"RUN_MODE": "REALTIME_DRY"}, clear=False):
        with patch.object(oe, "TRADING_ENABLED", False):
            asyncio.run(ex._execute_exit("NVDA", sig, stock_price=101.0, curr_ts=1_777_777_120.0, batch_idx=0))

    assert orch.accounting.exit_calls, "应触发 dry 平仓记账"
    first_call_args = orch.accounting.exit_calls[0][0]
    filled_qty = int(first_call_args[2])  # _process_exit_accounting(..., filled_qty, ...)
    assert filled_qty == 3, f"PUT 平仓应按 bid_size 成交，期望 3，实际 {filled_qty}"


def test_execution_cfg_override_limit_buffers() -> None:
    _bootstrap_imports()
    import orchestrator_execution as oe  # noqa: E402

    orch = _build_orch(mode="realtime")
    orch.cfg.LIMIT_BUFFER_ENTRY = 1.20
    orch.cfg.LIMIT_BUFFER_EXIT = 0.85
    ex = oe.OrchestratorExecution(orch)

    entry_px = ex._get_entry_limit_price({"meta": {"bid": 0.0, "ask": 0.0}}, base_price=2.0, attempt_no=0)
    exit_px = ex._get_exit_limit_price(base_price=2.0, bid=0.0, ask=0.0, attempt_no=0)

    assert abs(entry_px - 2.4) < 1e-9, f"应使用 cfg.LIMIT_BUFFER_ENTRY=1.20，实际 {entry_px}"
    assert abs(exit_px - 1.7) < 1e-9, f"应使用 cfg.LIMIT_BUFFER_EXIT=0.85，实际 {exit_px}"


class _FakeOrderStatus:
    def __init__(self) -> None:
        self.status = "Submitted"
        self.filled = 0
        self.avgFillPrice = 0.0


class _FakeTrade:
    def __init__(self) -> None:
        self.orderStatus = _FakeOrderStatus()
        self.order = object()


def test_entry_requote_cap_blocks_over_2pct() -> None:
    _bootstrap_imports()
    import orchestrator_execution as oe  # noqa: E402

    orch = _build_orch(mode="realtime")
    orch.cfg.ORDER_TIMEOUT_SECONDS = 1
    orch.cfg.ORDER_MAX_RETRIES = 2
    orch.cfg.ENTRY_MAX_REQUOTE_SLIPPAGE_PCT = 0.02
    ex = oe.OrchestratorExecution(orch)
    st = orch.states["NVDA"]

    trade = _FakeTrade()
    sig = {
        "reason": "UNIT_REQUOTE",
        "price": 2.0,
        "meta": {"alpha_available_ts": 1_777_777_777.0},
    }

    async def _fast_sleep(_secs: float):
        return None

    with patch("asyncio.sleep", _fast_sleep):
        with patch.object(oe.OrchestratorExecution, "_get_entry_limit_price", return_value=2.2):
            asyncio.run(
                ex._monitor_realtime_order(
                    "NVDA",
                    trade,
                    object(),
                    cost=1000.0,
                    commission=10.0,
                    expected_qty=5,
                    start_time=1_777_777_700.0,
                    limit_price=2.0,
                    stock_price=100.0,
                    sig=sig,
                    st=st,
                )
            )

    assert len(orch.ibkr.orders) == 0, "追价超过 2% 上限时不应继续发送 re-quote 订单"


def main() -> None:
    test_execute_entry_scope_guard_no_unboundlocalerror()
    test_execute_exit_put_uses_bid_size_in_dry_mode()
    test_execution_cfg_override_limit_buffers()
    test_entry_requote_cap_blocks_over_2pct()
    print("[OK] orchestrator execution guards passed")


if __name__ == "__main__":
    main()


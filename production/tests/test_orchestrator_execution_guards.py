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
        self.entry_slot_reserved = False
        self.open_fill_confirmed = False


class _DummyRedis:
    def get(self, _key):
        return None


class _DummyIBKR:
    def __init__(self) -> None:
        self.locked_contracts = {}
        self.orders = []
        self.cancelled_orders = []
        self.ib = SimpleNamespace(
            cancelOrder=lambda order, *_args, **_kwargs: self.cancelled_orders.append(order),
            isConnected=lambda: True,
        )

    def place_option_order(self, *args, **kwargs):
        self.orders.append((args, kwargs))
        return None


class _DummyAccounting:
    def __init__(self) -> None:
        self.open_calls = []
        self.exit_calls = []

    def _process_open_accounting(self, *args, **kwargs):
        self.open_calls.append((args, kwargs))
        return None

    def _process_exit_accounting(self, *args, **kwargs):
        self.exit_calls.append((args, kwargs))

    def _emit_trade_log(self, *_args, **_kwargs):
        return None


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
        EXIT_ORDER_MAX_RETRIES=5,
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


def test_realtime_iceberg_entry_keeps_pending_until_background_task_finishes() -> None:
    _bootstrap_imports()
    import orchestrator_execution as oe  # noqa: E402

    orch = _build_orch(mode="realtime")
    orch.cfg.LIMIT_BUFFER_ENTRY = 1.0
    ex = oe.OrchestratorExecution(orch)
    st = orch.states["NVDA"]

    sig = {
        "tag": "CALL_ATM",
        "dir": 1,
        "reason": "UNIT_ICEBERG",
        "price": 2.0,
        "meta": {"iv": 0.3, "contract_id": "NVDA260313C00180000", "strike": 180.0, "ask_size": 1},
    }
    captured_coros = []

    def _capture_task(coro):
        captured_coros.append(coro)
        coro.close()
        return SimpleNamespace(cancel=lambda: None)

    with patch.object(oe.LiquidityRiskManager, "evaluate_order", return_value={"final_alloc": 3000.0, "chunks": 2, "reason": "unit"}):
        with patch.object(oe, "DISABLE_ICEBERG", False), patch.object(oe, "SYNC_EXECUTION", False):
            with patch("asyncio.create_task", side_effect=_capture_task):
                asyncio.run(ex._execute_entry("NVDA", sig, stock_price=100.0, curr_ts=1_777_777_777.0, batch_idx=0))

    assert captured_coros, "chunks>1 的 realtime 开仓应启动后台冰山任务"
    assert st.is_pending is True
    assert getattr(st, "_async_entry_order_active", False) is True
    assert st.locked_cash > 0
    assert orch.mock_cash == 50000.0, "冰山单开始执行但未成交前不应改变 Remaining Cash"


def test_entry_allows_fourth_slot_but_blocks_fifth() -> None:
    _bootstrap_imports()
    import orchestrator_execution as oe  # noqa: E402

    orch = _build_orch(mode="realtime")
    orch.cfg.LIMIT_BUFFER_ENTRY = 1.0
    for sym in ["AAPL", "MSFT", "AMZN"]:
        st = _State()
        st.symbol = sym
        st.position = 1
        st.qty = 1
        st.entry_price = 2.0
        st.entry_slot_reserved = True
        orch.states[sym] = st

    ex = oe.OrchestratorExecution(orch)
    sig = {
        "tag": "CALL_ATM",
        "dir": 1,
        "reason": "UNIT_FOURTH",
        "price": 2.0,
        "meta": {"iv": 0.3, "contract_id": "NVDA260313C00180000", "strike": 180.0},
    }

    with patch.object(oe.LiquidityRiskManager, "evaluate_order", return_value={"final_alloc": 1000.0, "chunks": 1, "reason": "unit"}):
        with patch.object(oe, "TRADING_ENABLED", False):
            asyncio.run(ex._execute_entry("NVDA", sig, stock_price=100.0, curr_ts=1_777_777_777.0, batch_idx=0))

    assert orch.states["NVDA"].position != 0, "已有 3 个持仓时，第 4 个应允许开仓"
    assert getattr(orch.states["NVDA"], "entry_slot_reserved", False) is True

    fifth = _State()
    fifth.symbol = "TSLA"
    orch.states["TSLA"] = fifth
    sig2 = {
        "tag": "CALL_ATM",
        "dir": 1,
        "reason": "UNIT_FIFTH",
        "price": 2.0,
        "meta": {"iv": 0.3, "contract_id": "TSLA260313C00180000", "strike": 180.0},
    }

    with patch.object(oe.LiquidityRiskManager, "evaluate_order", return_value={"final_alloc": 1000.0, "chunks": 1, "reason": "unit"}):
        with patch.object(oe, "TRADING_ENABLED", False):
            asyncio.run(ex._execute_entry("TSLA", sig2, stock_price=100.0, curr_ts=1_777_777_778.0, batch_idx=1))

    assert fifth.position == 0, "已有 4 个 active slot 时，第 5 个必须被拒绝"
    assert getattr(fifth, "entry_slot_reserved", False) is False


def test_live_capital_limit_caps_realtime_entry_allocation() -> None:
    _bootstrap_imports()
    import orchestrator_execution as oe  # noqa: E402

    orch = _build_orch(mode="realtime")
    ex = oe.OrchestratorExecution(orch)

    sig = {
        "tag": "CALL_ATM",
        "dir": 1,
        "reason": "UNIT_LIVE_CAP",
        "price": 2.0,
        "meta": {"iv": 0.3, "contract_id": "NVDA260313C00180000", "strike": 180.0},
    }
    seen = {}

    def _fake_liq_eval(_sym, alloc, _price, **_kwargs):
        seen["alloc"] = float(alloc)
        return {"final_alloc": float(alloc), "chunks": 1, "reason": "unit"}

    with patch.object(oe.LiquidityRiskManager, "evaluate_order", side_effect=_fake_liq_eval):
        with patch.object(oe, "LIVE_TRADING_CAPITAL_LIMIT", 5000.0), \
             patch.object(oe, "IS_REALTIME_DRY", False), \
             patch.object(oe, "TRADING_ENABLED", False), \
             patch.object(oe, "SYNC_EXECUTION", True):
            asyncio.run(ex._execute_entry("NVDA", sig, stock_price=100.0, curr_ts=1_777_777_777.0, batch_idx=0))

    assert round(seen["alloc"], 2) == 2500.0, "测试夹具 POSITION_RATIO=50%，实盘总资金上限=5000 时单笔应为 2500 美元"


def test_runtime_live_capital_limit_override_beats_config_default() -> None:
    _bootstrap_imports()
    import orchestrator_execution as oe  # noqa: E402

    orch = _build_orch(mode="realtime")
    ex = oe.OrchestratorExecution(orch)

    sig = {
        "tag": "CALL_ATM",
        "dir": 1,
        "reason": "UNIT_RUNTIME_CAP",
        "price": 2.0,
        "meta": {"iv": 0.3, "contract_id": "NVDA260313C00180000", "strike": 180.0},
    }

    seen = {}

    def _capture(_sym, final_alloc, _price, **_kwargs):
        seen["alloc"] = final_alloc
        return {"final_alloc": final_alloc, "chunks": 1, "reason": "unit"}

    with patch.object(oe, "LIVE_TRADING_CAPITAL_LIMIT", 5000.0), \
         patch.object(oe, "IS_REALTIME_DRY", False), \
         patch.object(oe, "TRADING_ENABLED", False), \
         patch.object(oe, "SYNC_EXECUTION", True), \
         patch.object(oe, "get_runtime_live_trading_capital_limit", return_value=4000.0), \
         patch.object(oe.LiquidityRiskManager, "evaluate_order", side_effect=_capture):
        asyncio.run(ex._execute_entry("NVDA", sig, stock_price=100.0, curr_ts=1_777_777_777.0, batch_idx=0))

    assert round(seen["alloc"], 2) == 2000.0, "Redis runtime cap=4000 时，应覆盖 config 默认 5000，单笔分配为 2000 美元"


def test_runtime_trading_disable_switches_realtime_entry_to_dry_path() -> None:
    _bootstrap_imports()
    import orchestrator_execution as oe  # noqa: E402

    orch = _build_orch(mode="realtime")
    ex = oe.OrchestratorExecution(orch)

    sig = {
        "tag": "CALL_ATM",
        "dir": 1,
        "reason": "UNIT_RUNTIME_DISARM",
        "price": 2.0,
        "meta": {"iv": 0.3, "contract_id": "NVDA260313C00180000", "strike": 180.0},
    }

    with patch.object(oe, "LIVE_TRADING_CAPITAL_LIMIT", 5000.0), \
         patch.object(oe, "IS_REALTIME_DRY", False), \
         patch.object(oe, "TRADING_ENABLED", True), \
         patch.object(oe, "SYNC_EXECUTION", False), \
         patch.object(oe, "get_runtime_trading_enabled", return_value=False), \
         patch.object(oe.OrchestratorExecution, "_get_entry_limit_price", return_value=2.0), \
         patch.object(oe.LiquidityRiskManager, "evaluate_order", return_value={"final_alloc": 2000.0, "chunks": 1, "reason": "unit"}):
        asyncio.run(ex._execute_entry("NVDA", sig, stock_price=100.0, curr_ts=1_777_777_777.0, batch_idx=0))

    assert orch.accounting.open_calls, "Runtime disarm 后，realtime 开仓应走 dry accounting 路径而不是真实下单"


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


def test_entry_limit_price_never_crosses_ask_on_initial_quote() -> None:
    _bootstrap_imports()
    import orchestrator_execution as oe  # noqa: E402

    orch = _build_orch(mode="realtime")
    ex = oe.OrchestratorExecution(orch)

    sig = {"meta": {"bid": 1.96, "ask": 2.04}}
    limit_px = ex._get_entry_limit_price(sig, base_price=2.0, attempt_no=0)

    assert limit_px < 2.04, f"初始建仓限价不应直接等于 ask，实际 {limit_px}"
    assert limit_px >= 1.96, f"初始建仓限价不应低于 bid，实际 {limit_px}"


def test_entry_requote_price_stays_below_live_ask() -> None:
    _bootstrap_imports()
    import orchestrator_execution as oe  # noqa: E402

    orch = _build_orch(mode="realtime")
    ex = oe.OrchestratorExecution(orch)

    sig = {"price": 2.0, "meta": {"bid": 1.96, "ask": 2.04}}
    next_limit = ex._next_entry_requote_price(
        sig=sig,
        prev_limit_price=2.00,
        attempt_no=1,
        cap_price=2.04,
        real_contract=None,
    )

    assert next_limit < 2.04, f"追价也不应直接等于 ask，实际 {next_limit}"
    assert next_limit >= 2.00, f"追价至少不应低于上一笔限价，实际 {next_limit}"


class _FakeOrderStatus:
    def __init__(self) -> None:
        self.status = "Submitted"
        self.filled = 0
        self.avgFillPrice = 0.0


class _FakeTrade:
    def __init__(self) -> None:
        self.orderStatus = _FakeOrderStatus()
        self.order = object()


def test_mkt_urgent_exit_records_accounting_when_broker_reports_fill() -> None:
    _bootstrap_imports()
    import orchestrator_execution as oe  # noqa: E402

    orch = _build_orch(mode="realtime")
    orch.cfg.EXIT_ORDER_TYPE = "MKT"
    orch.cfg.ORDER_TIMEOUT_SECONDS = 1
    ex = oe.OrchestratorExecution(orch)
    st = orch.states["NVDA"]
    st.position = 1
    st.qty = 4
    st.entry_price = 2.0
    st.entry_ts = 1_777_777_000.0
    st.entry_slot_reserved = True
    st.open_fill_confirmed = True

    trade = _FakeTrade()
    trade.orderStatus.status = "Filled"
    trade.orderStatus.filled = 4
    trade.orderStatus.avgFillPrice = 1.85

    def _filled_mkt_order(*args, **kwargs):
        orch.ibkr.orders.append((args, kwargs))
        return trade

    async def _fast_sleep(_secs: float):
        return None

    orch.ibkr.place_option_order = _filled_mkt_order
    with patch("asyncio.sleep", _fast_sleep), \
         patch.object(oe.OrchestratorExecution, "_runtime_trading_enabled", return_value=True):
        asyncio.run(
            ex._smart_exit_order(
                "NVDA",
                real_contract=object(),
                total_qty=4,
                base_price=1.80,
                stock_price=100.0,
                curr_ts=1_777_777_120.0,
                is_force=False,
                bid=1.75,
                ask=1.85,
                reason="HARD_STOP:-16%",
            )
        )

    assert orch.accounting.exit_calls, "MKT 紧急平仓成交后必须进入 exit accounting 闭环"
    call_args = orch.accounting.exit_calls[0][0]
    assert int(call_args[2]) == 4
    assert abs(float(call_args[3]) - 1.85) < 1e-9


def test_realtime_entry_monitor_error_without_fill_resets_pre_entry_state() -> None:
    _bootstrap_imports()
    import orchestrator_execution as oe  # noqa: E402

    orch = _build_orch(mode="realtime")
    orch.cfg.ORDER_TIMEOUT_SECONDS = 1
    orch.cfg.ORDER_MAX_RETRIES = 2
    ex = oe.OrchestratorExecution(orch)
    st = orch.states["NVDA"]
    st.position = 1
    st.qty = 5
    st.entry_price = 2.0
    st.locked_cash = 1000.0
    st.entry_slot_reserved = True
    st.is_pending = True
    trade = _FakeTrade()
    sig = {
        "reason": "UNIT_MONITOR_ERROR",
        "price": 2.0,
        "meta": {"alpha_available_ts": 1_777_777_777.0},
    }

    async def _fast_sleep(_secs: float):
        return None

    with patch("asyncio.sleep", _fast_sleep):
        with patch.object(oe.OrchestratorExecution, "_next_entry_requote_price", side_effect=RuntimeError("requote failed")):
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

    assert st.position == 0
    assert st.qty == 0
    assert st.locked_cash == 0.0
    assert st.entry_slot_reserved is False
    assert st.is_pending is False


def test_entry_monitor_uses_entry_retry_config_not_exit_retry_config() -> None:
    _bootstrap_imports()
    import orchestrator_execution as oe  # noqa: E402

    orch = _build_orch(mode="realtime")
    orch.cfg.ORDER_TIMEOUT_SECONDS = 1
    orch.cfg.ORDER_MAX_RETRIES = 2
    orch.cfg.EXIT_ORDER_MAX_RETRIES = 5
    ex = oe.OrchestratorExecution(orch)
    st = orch.states["NVDA"]
    st.position = 1
    st.qty = 5
    st.entry_price = 2.0
    st.locked_cash = 1000.0
    st.entry_slot_reserved = True
    trade = _FakeTrade()
    sig = {
        "reason": "UNIT_ENTRY_RETRY_CFG",
        "price": 2.0,
        "meta": {"alpha_available_ts": 1_777_777_777.0},
    }

    def _return_unfilled_trade(*args, **kwargs):
        orch.ibkr.orders.append((args, kwargs))
        return _FakeTrade()

    async def _fast_sleep(_secs: float):
        return None

    orch.ibkr.place_option_order = _return_unfilled_trade
    with patch("asyncio.sleep", _fast_sleep):
        with patch.object(oe.OrchestratorExecution, "_next_entry_requote_price", return_value=2.01):
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

    assert len(orch.ibkr.orders) == 1, "开仓监控应按 ORDER_MAX_RETRIES=2，只允许 1 次 re-quote"


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


def test_realtime_order_timeout_uses_ibkr_connector_cancel() -> None:
    _bootstrap_imports()
    import orchestrator_execution as oe  # noqa: E402

    orch = _build_orch(mode="realtime")
    orch.cfg.ORDER_TIMEOUT_SECONDS = 1
    orch.cfg.ORDER_MAX_RETRIES = 1
    delattr(orch, "ib")
    ex = oe.OrchestratorExecution(orch)
    st = orch.states["NVDA"]
    trade = _FakeTrade()
    sig = {
        "reason": "UNIT_TIMEOUT_CANCEL",
        "price": 2.0,
        "meta": {"alpha_available_ts": 1_777_777_777.0},
    }

    async def _fast_sleep(_secs: float):
        return None

    with patch("asyncio.sleep", _fast_sleep):
        asyncio.run(
            ex._monitor_realtime_order(
                "NVDA",
                trade,
                object(),
                cost=1000.0,
                commission=10.0,
                expected_qty=1,
                start_time=1_777_777_700.0,
                limit_price=2.0,
                stock_price=100.0,
                sig=sig,
                st=st,
            )
        )

    assert orch.ibkr.cancelled_orders == [trade.order], "超时撤单必须走 orch.ibkr.ib.cancelOrder"


def main() -> None:
    test_execute_entry_scope_guard_no_unboundlocalerror()
    test_realtime_iceberg_entry_keeps_pending_until_background_task_finishes()
    test_entry_allows_fourth_slot_but_blocks_fifth()
    test_execute_exit_put_uses_bid_size_in_dry_mode()
    test_execution_cfg_override_limit_buffers()
    test_entry_limit_price_never_crosses_ask_on_initial_quote()
    test_entry_requote_price_stays_below_live_ask()
    test_realtime_entry_monitor_error_without_fill_resets_pre_entry_state()
    test_entry_monitor_uses_entry_retry_config_not_exit_retry_config()
    test_entry_requote_cap_blocks_over_2pct()
    test_realtime_order_timeout_uses_ibkr_connector_cancel()
    print("[OK] orchestrator execution guards passed")


if __name__ == "__main__":
    main()

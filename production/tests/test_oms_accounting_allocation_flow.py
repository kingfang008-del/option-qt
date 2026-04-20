#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
回归测试：OMS 开平仓记账与仓位分配联动。

覆盖点：
1) DRY 模式开仓扣款 + 平仓回款后，状态与现金应一致；
2) 亏损后下一次开仓分配额度应下降（使用更新后的现金）；
3) 有仓位不应重复开仓、无仓位不应错误平仓。
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
    def __init__(self, symbol: str = "NVDA") -> None:
        self.symbol = symbol
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
        self.pending_ts = 0.0
        self.pending_side = ""
        self.last_opt_price = 2.0
        self.cooldown_until = 0.0


class _DummyRedis:
    def __init__(self) -> None:
        self.streams = []

    def get(self, _key):
        return None

    def xadd(self, stream, payload, maxlen=None):
        self.streams.append((stream, payload, maxlen))
        return "1-0"


class _DummyIBKR:
    def __init__(self) -> None:
        self.locked_contracts = {}
        self.orders = []
        self.ib = SimpleNamespace(cancelOrder=lambda *_args, **_kwargs: None)

    def place_option_order(self, *args, **kwargs):
        self.orders.append((args, kwargs))
        # DRY 模式下返回 None，走 OMS 的模拟记账分支
        return None


class _DummyStateManager:
    def save_state(self):
        return None


def _build_orch(mode: str = "realtime") -> SimpleNamespace:
    st = _State("NVDA")
    cfg = SimpleNamespace(
        MAX_POSITIONS=4,
        POSITION_RATIO=0.5,
        MAX_TRADE_CAP=100000.0,
        GLOBAL_EXPOSURE_LIMIT=0.9,
        COMMISSION_PER_CONTRACT=0.65,
        LIMIT_BUFFER_ENTRY=1.03,
        LIMIT_BUFFER_EXIT=0.97,
        ORDER_TIMEOUT_SECONDS=5,
        ORDER_MAX_RETRIES=2,
        EXIT_ORDER_TYPE="LMT",
        SLIPPAGE_PCT=0.001,
        ENTRY_MAX_REQUOTE_SLIPPAGE_PCT=0.05,
        STOP_LOSS=-0.2,
        AUTO_TRADING_CAPITAL_RATIO=1.0,
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
        state_manager=_DummyStateManager(),
        strategy=SimpleNamespace(cfg=SimpleNamespace(STOP_LOSS=-0.2)),
        use_shared_mem=False,
        # accounting 统计字段
        stats_counter_trend_long_count=0,
        stats_counter_trend_long_pnl=0.0,
        stats_counter_trend_long_win_count=0,
        stats_counter_trend_short_count=0,
        stats_counter_trend_short_pnl=0.0,
        stats_counter_trend_short_win_count=0,
        consecutive_stop_losses=0,
        CIRCUIT_BREAKER_THRESHOLD=3,
        CIRCUIT_BREAKER_MINUTES=10,
        realized_pnl=0.0,
        total_commission=0.0,
        trade_count=0,
        win_count=0,
        loss_count=0,
        daily_trades=[],
    )
    orch.ib = orch.ibkr.ib
    orch._get_fair_market_price = lambda base, _bid, _ask, prev=0.0: float(base if base > 0 else prev)
    return orch


def _entry_sig(price: float = 2.0) -> dict:
    return {
        "tag": "CALL_ATM",
        "dir": 1,
        "reason": "UNIT_ENTRY",
        "price": price,
        "meta": {"iv": 0.3, "contract_id": "NVDA260313C00180000", "strike": 180.0},
    }


def _exit_sig(price: float, reason: str = "FORCE_TEST") -> dict:
    return {
        "reason": reason,
        "price": price,
        "market_price": price,
        "bid": max(price - 0.05, 0.01),
        "ask": price + 0.05,
        "bid_size": 999,
        "ask_size": 999,
        "original_position": 1,
    }


def test_dry_open_close_accounting_updates_cash_and_position() -> None:
    _bootstrap_imports()
    import config as cfg_mod  # noqa: E402
    import orchestrator_execution as oe  # noqa: E402
    import orchestrator_accounting as oa  # noqa: E402

    orch = _build_orch(mode="realtime")
    orch.accounting = oa.OrchestratorAccounting(orch)
    ex = oe.OrchestratorExecution(orch)
    initial_cash = orch.mock_cash

    def _liq_passthrough(sym, final_alloc, price, **kwargs):
        return {"final_alloc": float(final_alloc), "chunks": 1, "reason": f"unit:{sym}:{price}"}

    with patch.dict(os.environ, {"RUN_MODE": "REALTIME_DRY"}, clear=False):
        with patch.object(cfg_mod, "TRADING_ENABLED", False):
            with patch.object(oe, "TRADING_ENABLED", False):
                with patch.object(oe.LiquidityRiskManager, "evaluate_order", side_effect=_liq_passthrough):
                    asyncio.run(ex._execute_entry("NVDA", _entry_sig(2.0), stock_price=100.0, curr_ts=1_777_777_700.0, batch_idx=0))

                    st = orch.states["NVDA"]
                    assert st.position == 1 and st.qty > 0, "开仓后应持有多头仓位"
                    assert orch.mock_cash < initial_cash, "开仓后应先扣减现金"

                    asyncio.run(ex._execute_exit("NVDA", _exit_sig(2.5, reason="FORCE_PROFIT"), stock_price=101.0, curr_ts=1_777_777_900.0, batch_idx=0))

    st = orch.states["NVDA"]
    assert st.position == 0 and st.qty == 0, "平仓后仓位应归零"
    assert orch.mock_cash > initial_cash, "盈利平仓后现金应高于初始值"
    assert orch.realized_pnl > 0, "盈利平仓后 realized_pnl 应为正"
    assert orch.trade_count == 1, "应记录 1 笔平仓交易"
    assert len(orch.r.streams) >= 2, "应至少写入 OPEN/CLOSE 两条交易日志"


def test_loss_reduces_next_allocation_and_position_guards_work() -> None:
    _bootstrap_imports()
    import config as cfg_mod  # noqa: E402
    import orchestrator_execution as oe  # noqa: E402
    import orchestrator_accounting as oa  # noqa: E402

    orch = _build_orch(mode="realtime")
    orch.accounting = oa.OrchestratorAccounting(orch)
    ex = oe.OrchestratorExecution(orch)

    alloc_inputs = []

    def _liq_capture(sym, final_alloc, price, **kwargs):
        alloc_inputs.append(float(final_alloc))
        return {"final_alloc": float(final_alloc), "chunks": 1, "reason": f"capture:{sym}:{price}"}

    with patch.dict(os.environ, {"RUN_MODE": "REALTIME_DRY"}, clear=False):
        with patch.object(cfg_mod, "TRADING_ENABLED", False):
            with patch.object(oe, "TRADING_ENABLED", False):
                with patch.object(oe.LiquidityRiskManager, "evaluate_order", side_effect=_liq_capture):
                    # 第一笔开仓
                    asyncio.run(ex._execute_entry("NVDA", _entry_sig(2.0), stock_price=100.0, curr_ts=1_777_778_000.0, batch_idx=0))
                    st = orch.states["NVDA"]
                    assert st.position == 1 and st.qty > 0
                    entry_cash_after_open = orch.mock_cash

                    # 有仓位时再发 BUY，应被拒绝（避免无效申请仓位）
                    asyncio.run(ex._execute_entry("NVDA", _entry_sig(2.0), stock_price=100.0, curr_ts=1_777_778_010.0, batch_idx=1))
                    assert orch.mock_cash == entry_cash_after_open, "有仓位重复 BUY 不应再次扣款"

                    # 亏损平仓
                    asyncio.run(ex._execute_exit("NVDA", _exit_sig(1.5, reason="FORCE_LOSS"), stock_price=99.0, curr_ts=1_777_778_120.0, batch_idx=0))
                    assert orch.realized_pnl < 0, "亏损平仓后 realized_pnl 应为负"
                    cash_after_loss = orch.mock_cash

                    # 无仓位时发 SELL，应被拒绝（避免无仓位平仓）
                    asyncio.run(ex._execute_exit("NVDA", _exit_sig(1.4, reason="FORCE_NO_POS"), stock_price=98.0, curr_ts=1_777_778_140.0, batch_idx=2))
                    assert orch.mock_cash == cash_after_loss, "无仓位 SELL 不应改变现金"

                    # 第二笔开仓，额度应小于第一笔（因为现金变少）
                    asyncio.run(ex._execute_entry("NVDA", _entry_sig(2.0), stock_price=100.0, curr_ts=1_777_778_200.0, batch_idx=0))

    assert len(alloc_inputs) >= 2, "应至少捕获两次开仓分配额度"
    first_alloc = alloc_inputs[0]
    second_alloc = alloc_inputs[1]
    assert second_alloc < first_alloc, f"亏损后下一次分配额度应下降: first={first_alloc}, second={second_alloc}"


def main() -> None:
    test_dry_open_close_accounting_updates_cash_and_position()
    test_loss_reduces_next_allocation_and_position_guards_work()
    print("[OK] oms accounting allocation flow passed")


if __name__ == "__main__":
    main()


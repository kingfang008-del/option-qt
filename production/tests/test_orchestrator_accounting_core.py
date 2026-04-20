#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
回归测试：orchestrator_accounting 核心记账逻辑（直测）。

覆盖点：
1) _process_open_accounting: 状态写入 + OPEN 日志发出；
2) _process_exit_accounting: 现金回款、盈亏统计、持仓归零；
3) 部分平仓: qty 递减且仓位不应被误清零。
"""

from __future__ import annotations

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
        self.xadds = []

    def xadd(self, stream, payload, maxlen=None):
        self.xadds.append((stream, payload, maxlen))
        return "1-0"


def _build_orch() -> SimpleNamespace:
    return SimpleNamespace(
        mode="realtime",
        mock_cash=50_000.0,
        r=_DummyRedis(),
        stats_counter_trend_long_count=0,
        stats_counter_trend_long_pnl=0.0,
        stats_counter_trend_long_win_count=0,
        stats_counter_trend_short_count=0,
        stats_counter_trend_short_pnl=0.0,
        stats_counter_trend_short_win_count=0,
        consecutive_stop_losses=0,
        CIRCUIT_BREAKER_THRESHOLD=3,
        CIRCUIT_BREAKER_MINUTES=10,
        global_cooldown_until=0.0,
        realized_pnl=0.0,
        total_commission=0.0,
        trade_count=0,
        win_count=0,
        loss_count=0,
        daily_trades=[],
        stats_liquidity_drought_liquidations=0,
    )


def test_open_accounting_writes_state_and_open_log() -> None:
    _bootstrap_imports()
    import orchestrator_accounting as oa  # noqa: E402
    from utils import serialization_utils as ser  # noqa: E402

    orch = _build_orch()
    acc = oa.OrchestratorAccounting(orch)
    st = _State()

    sig = {
        "dir": 1,
        "tag": "CALL_ATM",
        "reason": "UNIT_OPEN",
        "meta": {
            "spy_roc": 0.001,
            "index_trend": 1,
            "alpha_z": 1.2,
            "iv": 0.28,
        },
    }

    with patch.object(oa, "TRADING_ENABLED", False):
        acc._process_open_accounting(
            "NVDA",
            st,
            filled_qty=5,
            fill_price=2.0,
            stock_price=100.0,
            entry_ts=1_777_777_700.0,
            sig=sig,
            duration=0.2,
            ratio=1.0,
            mode_override=None,
        )

    assert st.position == 1 and st.qty == 5
    assert abs(st.entry_price - 2.0) < 1e-12
    assert st.opt_type == "call"
    assert len(orch.r.xadds) == 1, "OPEN 后应发送 1 条交易日志"

    _stream, payload_dict, _maxlen = orch.r.xadds[0]
    payload = ser.unpack(payload_dict["data"])
    assert payload["action"] == "OPEN"
    assert payload["qty"] == 5
    assert payload["mode"] == "LIVEREPLAY", "TRADING_ENABLED=False 时应路由为 LIVEREPLAY"


def test_exit_accounting_full_close_updates_cash_and_pnl() -> None:
    _bootstrap_imports()
    import orchestrator_accounting as oa  # noqa: E402

    orch = _build_orch()
    orch.mock_cash = 1_000.0
    acc = oa.OrchestratorAccounting(orch)
    st = _State()
    st.position = 1
    st.qty = 10
    st.entry_price = 2.0
    st.entry_stock = 100.0
    st.entry_ts = 1_777_777_000.0
    st.opt_type = "call"

    with patch.object(oa, "COMMISSION_PER_CONTRACT", 1.0):
        acc._process_exit_accounting(
            "NVDA",
            st,
            filled_qty=10,
            fill_price=2.5,
            stock_price=101.0,
            curr_ts=1_777_777_900.0,
            reason="UNIT_PROFIT",
            duration=0.5,
            ratio=1.0,
        )

    # proceeds = 2.5*10*100 - 10 = 2490
    assert abs(orch.mock_cash - 3490.0) < 1e-9
    # gross_pnl = (2.5-2.0)*10*100 - 10 - 10 = 480
    assert abs(orch.realized_pnl - 480.0) < 1e-9
    assert orch.trade_count == 1 and orch.win_count == 1 and orch.loss_count == 0
    assert abs(orch.total_commission - 20.0) < 1e-9
    assert st.position == 0 and st.qty == 0 and abs(st.entry_price) < 1e-12
    assert len(orch.daily_trades) == 1


def test_exit_accounting_partial_close_keeps_remaining_position() -> None:
    _bootstrap_imports()
    import orchestrator_accounting as oa  # noqa: E402

    orch = _build_orch()
    orch.mock_cash = 5_000.0
    acc = oa.OrchestratorAccounting(orch)
    st = _State()
    st.position = 1
    st.qty = 10
    st.entry_price = 2.0
    st.entry_stock = 100.0
    st.entry_ts = 1_777_777_000.0
    st.opt_type = "call"

    with patch.object(oa, "COMMISSION_PER_CONTRACT", 1.0):
        acc._process_exit_accounting(
            "NVDA",
            st,
            filled_qty=4,
            fill_price=2.2,
            stock_price=100.5,
            curr_ts=1_777_777_500.0,
            reason="UNIT_PARTIAL",
            duration=0.3,
            ratio=0.4,
        )

    # proceeds = 2.2*4*100 - 4 = 876
    assert abs(orch.mock_cash - 5876.0) < 1e-9
    # gross_pnl = (2.2-2.0)*4*100 - 4 - 4 = 72
    assert abs(orch.realized_pnl - 72.0) < 1e-9
    assert st.position == 1 and st.qty == 6, "部分平仓后应保留剩余仓位"
    assert abs(st.entry_price - 2.0) < 1e-12


def main() -> None:
    test_open_accounting_writes_state_and_open_log()
    test_exit_accounting_full_close_updates_cash_and_pnl()
    test_exit_accounting_partial_close_keeps_remaining_position()
    print("[OK] orchestrator accounting core passed")


if __name__ == "__main__":
    main()


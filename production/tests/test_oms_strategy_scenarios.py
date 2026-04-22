#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
OMS 场景回归测试（策略 + 资金）

目标：
1) 用可控路径验证关键止盈/止损门槛是否按预期触发；
2) 覆盖多标的交错开平仓后的现金与累计 PnL 对账一致性。
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


def _bootstrap_imports() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    production_dir = repo_root / "production"
    baseline_dir = production_dir / "baseline"
    sys.path.insert(0, str(production_dir))
    sys.path.insert(0, str(baseline_dir))


def _mk_ctx(
    *,
    symbol: str,
    now: datetime,
    curr_ts: float,
    entry_ts: float,
    entry_price: float,
    curr_price: float,
    max_roi: float,
    entry_stock: float = 100.0,
    curr_stock: float = 100.0,
    direction: int = 1,
) -> dict:
    return {
        "symbol": symbol,
        "time": now,
        "curr_ts": curr_ts,
        "price": curr_stock,
        "alpha_z": 1.0,
        "vol_z": 0.0,
        "stock_roc": 0.0,
        "event_prob": 0.0,
        "macd_hist": 0.01,
        "macd_hist_slope": 0.0,
        "spy_roc": 0.0,
        "qqq_roc": 0.0,
        "index_trend": 0,
        "position": direction,
        "cooldown_until": 0.0,
        "is_ready": True,
        "is_banned": False,
        "held_mins": (curr_ts - entry_ts) / 60.0,
        "stock_iv": 0.3,
        "holding": {
            "symbol": symbol,
            "entry_price": entry_price,
            "entry_stock": entry_stock,
            "entry_ts": entry_ts,
            "dir": direction,
            "max_roi": max_roi,
            "entry_spy_roc": 0.0,
            "entry_index_trend": 0,
        },
        "curr_price": curr_price,
        "curr_stock": curr_stock,
        "bid": curr_price * 0.995,
        "ask": curr_price * 1.005,
        "spread_divergence": 0.0,
        "snap_roc": 0.0,
        "global_regime_reversal_cnt": 0,
        "regime_reversal_count": 0,
        "is_volatile_regime": False,
        "regime_band": "calm",
        "regime_score": 0.0,
    }


def test_step_take_profit_boundary_and_trigger() -> None:
    _bootstrap_imports()
    from strategy_core_v0 import StrategyCoreV0  # noqa: E402
    from strategy_config0 import StrategyConfig  # noqa: E402

    core = StrategyCoreV0(StrategyConfig())
    t0 = 1_773_740_400.0  # 2026-03-06 10:00:00 ET (example epoch)
    now = datetime(2026, 3, 6, 10, 3, 0)

    # 当前配置下，max_roi=30% 命中的是 T=20%, floor=15% 档。
    # 边界：curr_roi=15% 正好踩在 floor，不应触发 STEP_PROT。
    ctx_floor_hold = _mk_ctx(
        symbol="NVDA",
        now=now,
        curr_ts=t0 + 180,
        entry_ts=t0,
        entry_price=1.0,
        curr_price=1.151,
        max_roi=0.30,
    )
    sig1 = core.check_exit(ctx_floor_hold)
    assert sig1 is None, f"30%->约15.1% 在 floor 上方应保持持仓，实际: {sig1}"

    # 下穿 floor：30% 回撤到 14% 应触发阶梯止盈
    ctx_floor_break = _mk_ctx(
        symbol="NVDA",
        now=now,
        curr_ts=t0 + 180,
        entry_ts=t0,
        entry_price=1.0,
        curr_price=1.149,
        max_roi=0.30,
    )
    sig2 = core.check_exit(ctx_floor_break)
    assert sig2 is not None, "30%->14% 应触发平仓"
    assert "STEP_PROT" in str(sig2.get("reason", "")), f"应为阶梯止盈，实际: {sig2}"


def test_stop_loss_threshold_behavior() -> None:
    _bootstrap_imports()
    from strategy_core_v0 import StrategyCoreV0  # noqa: E402
    from strategy_config0 import StrategyConfig  # noqa: E402

    core = StrategyCoreV0(StrategyConfig())
    t0 = 1_773_740_400.0
    now = datetime(2026, 3, 6, 10, 2, 0)

    # -10% 本身不应直接打出 HARD_STOP（绝对硬止损在 -15%）
    ctx_minus_10 = _mk_ctx(
        symbol="AAPL",
        now=now,
        curr_ts=t0 + 120,
        entry_ts=t0,
        entry_price=1.0,
        curr_price=0.90,
        max_roi=0.01,
        entry_stock=100.0,
        curr_stock=100.0,  # 避免触发 stock_stop/cond_stop，隔离到硬止损层
    )
    sig1 = core.check_exit(ctx_minus_10)
    assert not (sig1 and "HARD_STOP" in str(sig1.get("reason", ""))), f"-10% 不应直接 HARD_STOP，实际: {sig1}"

    # -16% 触发绝对硬止损
    ctx_minus_16 = _mk_ctx(
        symbol="AAPL",
        now=now,
        curr_ts=t0 + 120,
        entry_ts=t0,
        entry_price=1.0,
        curr_price=0.84,
        max_roi=0.01,
        entry_stock=100.0,
        curr_stock=100.0,
    )
    sig2 = core.check_exit(ctx_minus_16)
    assert sig2 is not None, "-16% 应触发平仓"
    assert "HARD_STOP" in str(sig2.get("reason", "")), f"应为 HARD_STOP，实际: {sig2}"


class _State:
    def __init__(self, symbol: str) -> None:
        self.symbol = symbol
        self.position = 0
        self.qty = 0
        self.entry_stock = 0.0
        self.entry_price = 0.0
        self.entry_ts = 0.0
        self.max_roi = 0.0
        self.last_valid_iv = 0.35
        self.contract_id = f"{symbol}260313C00180000"
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

    def hset(self, *_args, **_kwargs):
        return 1

    def expire(self, *_args, **_kwargs):
        return True


def _build_orch_for_accounting() -> SimpleNamespace:
    return SimpleNamespace(
        mode="backtest",
        mock_cash=50_000.0,
        r=_DummyRedis(),
        state_manager=None,
        _broadcast_state_to_redis=None,
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


def _debit_open_cash(orch, qty: int, fill_price: float, commission_per_contract: float) -> None:
    # 统一按实际交易含手续费扣减，模拟 OMS 开仓资金锁定/扣款。
    orch.mock_cash -= fill_price * qty * 100 + qty * commission_per_contract


def test_multi_symbol_cash_reconciliation() -> None:
    _bootstrap_imports()
    import orchestrator_accounting as oa  # noqa: E402

    orch = _build_orch_for_accounting()
    acc = oa.OrchestratorAccounting(orch)

    nvda = _State("NVDA")
    aapl = _State("AAPL")

    with patch.object(oa, "COMMISSION_PER_CONTRACT", 1.0):
        # NVDA 开仓: 10 @ 2.0
        nvda.position = 1
        nvda.qty = 10
        nvda.entry_price = 2.0
        nvda.entry_stock = 100.0
        nvda.entry_ts = 1_773_740_400.0
        _debit_open_cash(orch, qty=10, fill_price=2.0, commission_per_contract=1.0)

        # AAPL 后续开仓: 5 @ 4.0
        aapl.position = 1
        aapl.qty = 5
        aapl.entry_price = 4.0
        aapl.entry_stock = 200.0
        aapl.entry_ts = 1_773_740_460.0
        _debit_open_cash(orch, qty=5, fill_price=4.0, commission_per_contract=1.0)

        # 不同时间平仓
        acc._process_exit_accounting(
            "NVDA",
            nvda,
            filled_qty=10,
            fill_price=2.6,
            stock_price=101.0,
            curr_ts=1_773_740_520.0,
            reason="UNIT_NVDA_CLOSE",
            duration=0.4,
            ratio=1.0,
        )
        acc._process_exit_accounting(
            "AAPL",
            aapl,
            filled_qty=5,
            fill_price=3.2,
            stock_price=198.0,
            curr_ts=1_773_740_700.0,
            reason="UNIT_AAPL_CLOSE",
            duration=0.5,
            ratio=1.0,
        )

    # 期望现金:
    # 初始 50000
    # 开仓扣减: (2.0*10*100+10) + (4.0*5*100+5) = 2010 + 2005 = 4015
    # 平仓回款: (2.6*10*100-10) + (3.2*5*100-5) = 2590 + 1595 = 4185
    # 最终: 50000 - 4015 + 4185 = 50170
    assert abs(orch.mock_cash - 50_170.0) < 1e-9, f"现金对账失败，实际: {orch.mock_cash}"

    # 累计净收益（含双边手续费）:
    # NVDA: (2.6-2.0)*10*100 - 10 - 10 = +580
    # AAPL: (3.2-4.0)*5*100 - 5 - 5 = -410
    # 合计: +170
    assert abs(orch.realized_pnl - 170.0) < 1e-9, f"realized_pnl 异常: {orch.realized_pnl}"
    assert orch.trade_count == 2 and orch.win_count == 1 and orch.loss_count == 1
    assert nvda.position == 0 and nvda.qty == 0
    assert aapl.position == 0 and aapl.qty == 0

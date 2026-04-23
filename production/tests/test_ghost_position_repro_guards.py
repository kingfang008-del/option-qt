#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import asyncio
import json
import os
import random
import sys
import time
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
    def __init__(self, symbol: str = "MSTR") -> None:
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
        self.entry_slot_reserved = False
        self.open_fill_confirmed = False
        self.warmup_complete = True
        self.locked_cash = 0.0
        self.prices = []
        self.alpha_history = []
        self.pct_history = []
        self.ema_fast_val = None
        self.ema_slow_val = None
        self.dea_val = None
        self.correction_mode = "NORMAL"
        self.prev_macd_hist = 0.0
        self.last_spread_pct = 0.0

    def to_dict(self):
        return {
            "symbol": self.symbol,
            "position": self.position,
            "qty": self.qty,
            "entry_price": self.entry_price,
            "entry_stock": self.entry_stock,
            "entry_ts": self.entry_ts,
            "entry_spy_roc": self.entry_spy_roc,
            "entry_index_trend": self.entry_index_trend,
            "entry_alpha_z": self.entry_alpha_z,
            "entry_iv": self.entry_iv,
            "max_roi": self.max_roi,
            "cooldown_until": self.cooldown_until,
            "contract_id": self.contract_id,
            "strike_price": self.strike_price,
            "expiry_date": None,
            "last_valid_iv": self.last_valid_iv,
            "opt_type": self.opt_type,
            "warmup_complete": self.warmup_complete,
            "correction_mode": self.correction_mode,
            "prev_macd_hist": self.prev_macd_hist,
            "last_spread_pct": self.last_spread_pct,
            "entry_slot_reserved": self.entry_slot_reserved,
            "open_fill_confirmed": self.open_fill_confirmed,
            "prices": self.prices,
            "alpha_history": self.alpha_history,
            "pct_history": self.pct_history,
            "ema_fast_val": self.ema_fast_val,
            "ema_slow_val": self.ema_slow_val,
            "dea_val": self.dea_val,
        }


class _StateWithMidSave(_State):
    def __init__(self, symbol: str = "MSTR") -> None:
        super().__init__(symbol)
        self._mid_save_callback = None
        self._mid_save_triggered = False
        self._mid_save_trigger_attr = "entry_price"

    def __setattr__(self, name, value):
        object.__setattr__(self, name, value)
        if name != getattr(self, "_mid_save_trigger_attr", "entry_price"):
            return
        callback = getattr(self, "_mid_save_callback", None)
        if callback is None or getattr(self, "_mid_save_triggered", False):
            return
        if int(getattr(self, "position", 0) or 0) == 0:
            return
        if int(getattr(self, "qty", 0) or 0) <= 0:
            return
        if float(getattr(self, "entry_ts", 0.0) or 0.0) > 0:
            return
        object.__setattr__(self, "_mid_save_triggered", True)
        callback()


class _DummyRedis:
    def __init__(self) -> None:
        self.xadds = []

    def xadd(self, stream, payload, maxlen=None):
        self.xadds.append((stream, payload, maxlen))
        return "1-0"


class _DummyCursor:
    def __init__(self, rows):
        self.rows = rows
        self.last_query = ""

    def execute(self, query, params=None):
        self.last_query = str(query)

    def fetchone(self):
        if "to_regclass" in self.last_query:
            return ("symbol_state",)
        return None

    def fetchall(self):
        if "FROM symbol_state WHERE namespace = %s" in self.last_query:
            return self.rows
        return []


class _DummyConn:
    def __init__(self, rows):
        self._rows = rows

    def cursor(self):
        return _DummyCursor(self._rows)

    def close(self):
        return None


def _build_state_manager_orch():
    return SimpleNamespace(
        mode="realtime",
        mock_cash=50_000.0,
        cfg=SimpleNamespace(INITIAL_ACCOUNT=50_000.0),
        states={"MSTR": _State("MSTR")},
        locked_cash=0.0,
        positions={},
        pending_orders={},
    )


def _build_execution_orch(st):
    class _DummyRedisNoop:
        def get(self, _key):
            return None

        def xadd(self, _stream, _payload, maxlen=None):
            return "1-0"

    class _DummyIBKRNoop:
        def __init__(self) -> None:
            self.locked_contracts = {}
            self.orders = []
            self.ib = SimpleNamespace(cancelOrder=lambda *_args, **_kwargs: None)

        def place_option_order(self, *args, **kwargs):
            self.orders.append((args, kwargs))
            return None

    return SimpleNamespace(
        mode="realtime",
        states={st.symbol: st},
        cfg=SimpleNamespace(
            MAX_POSITIONS=4,
            POSITION_RATIO=0.5,
            MAX_TRADE_CAP=100000.0,
            GLOBAL_EXPOSURE_LIMIT=0.9,
            COMMISSION_PER_CONTRACT=0.65,
            LIMIT_BUFFER_ENTRY=1.0,
            LIMIT_BUFFER_EXIT=0.97,
            ORDER_TIMEOUT_SECONDS=5,
            ORDER_MAX_RETRIES=2,
            EXIT_ORDER_TYPE="LMT",
            SLIPPAGE_PCT=0.001,
            ENTRY_MAX_REQUOTE_SLIPPAGE_PCT=0.02,
            STOP_LOSS=-0.2,
        ),
        mock_cash=50_000.0,
        global_cooldown_until=0.0,
        MIN_OPTION_PRICE=0.5,
        r=_DummyRedisNoop(),
        ibkr=_DummyIBKRNoop(),
        accounting=None,
        state_manager=None,
        strategy=SimpleNamespace(cfg=SimpleNamespace(STOP_LOSS=-0.2)),
        use_shared_mem=False,
        _get_fair_market_price=lambda base, bid, ask, prev=0.0: float(base),
    )


def _build_accounting_orch():
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
        state_manager=SimpleNamespace(save_state=lambda: None),
    )


def _fault_injection_rounds_from_env(default: int = 0) -> int:
    try:
        return max(0, int(os.environ.get("GHOST_SOAK_ROUNDS", str(default))))
    except Exception:
        return default


def _fault_injection_seed_from_env(default: int = 20260423) -> int:
    try:
        return int(os.environ.get("GHOST_SOAK_SEED", str(default)))
    except Exception:
        return default


def _run_fault_injection_campaign(*, rounds: int, seed: int) -> None:
    _bootstrap_imports()
    import orchestrator_accounting as oa  # noqa: E402
    import orchestrator_execution as oe  # noqa: E402
    import orchestrator_state_manager as osm  # noqa: E402

    rng = random.Random(seed)
    trigger_attrs = ["qty", "entry_stock", "entry_price"]
    symbols = ["MSTR", "TSLA", "NVDA", "QQQ"]

    for batch_idx in range(rounds):
        symbol = rng.choice(symbols)
        trigger_attr = rng.choice(trigger_attrs)
        option_price = round(rng.uniform(1.2, 9.8), 2)
        stock_price = round(rng.uniform(90.0, 320.0), 2)
        final_alloc = round(rng.uniform(900.0, 4200.0), 2)
        curr_ts = float(1_777_777_700 + rng.randint(0, 300))

        st = _StateWithMidSave(symbol)
        st._mid_save_trigger_attr = trigger_attr
        orch = _build_execution_orch(st)
        orch.accounting = oa.OrchestratorAccounting(orch)
        sm = osm.OrchestratorStateManager(orch)
        captured_rows = []

        def _capture_save(state_data, snapshot_ts=None):
            row = dict(state_data[symbol])
            row["_snapshot_ts"] = snapshot_ts
            captured_rows.append(row)

        sm._save_state_to_db = _capture_save
        orch.state_manager = sm
        st._mid_save_callback = sm.save_state

        ex = oe.OrchestratorExecution(orch)
        sig = {
            "tag": "CALL_ATM",
            "dir": 1,
            "reason": f"UNIT_FAULT_INJECT_{trigger_attr}",
            "price": option_price,
            "meta": {"iv": 0.3, "contract_id": f"{symbol}260313C00180000", "strike": 180.0},
        }

        with patch.object(oe, "SYNC_EXECUTION", True), patch.object(oe, "TRADING_ENABLED", False):
            with patch.object(
                oe.LiquidityRiskManager,
                "evaluate_order",
                return_value={"final_alloc": final_alloc, "chunks": 1, "reason": "unit"},
            ):
                asyncio.run(ex._execute_entry(symbol, sig, stock_price=stock_price, curr_ts=curr_ts, batch_idx=batch_idx))

        assert len(captured_rows) >= 2, f"{symbol}/{trigger_attr} 应捕获中间快照和最终快照"
        half_open_row = captured_rows[0]
        final_row = captured_rows[-1]

        assert half_open_row["position"] == 0, f"{symbol}/{trigger_attr} 半开仓快照不应持久化 position"
        assert half_open_row["qty"] == 0, f"{symbol}/{trigger_attr} 半开仓快照不应持久化 qty"
        assert half_open_row["entry_price"] == 0.0, f"{symbol}/{trigger_attr} 半开仓快照不应持久化 entry_price"
        assert half_open_row["entry_ts"] == 0.0, f"{symbol}/{trigger_attr} 半开仓快照不应持久化 entry_ts"
        assert half_open_row["open_fill_confirmed"] is False

        assert final_row["position"] == 1, f"{symbol}/{trigger_attr} 最终快照应保留成交仓位"
        assert final_row["qty"] > 0
        assert final_row["entry_price"] > 0
        assert final_row["entry_ts"] > 0
        assert final_row["open_fill_confirmed"] is True


def test_save_state_zeroizes_half_open_position_before_persist() -> None:
    _bootstrap_imports()
    import orchestrator_state_manager as osm  # noqa: E402

    orch = _build_state_manager_orch()
    st = orch.states["MSTR"]
    st.position = 1
    st.qty = 20
    st.entry_price = 7.12
    st.entry_stock = 175.83
    st.entry_ts = 0.0
    st.entry_slot_reserved = True
    st.open_fill_confirmed = False

    sm = osm.OrchestratorStateManager(orch)
    captured = {}

    def _capture_save(state_data, snapshot_ts=None):
        captured["state_data"] = state_data
        captured["snapshot_ts"] = snapshot_ts

    sm._save_state_to_db = _capture_save
    sm.save_state()

    row = captured["state_data"]["MSTR"]
    assert row["position"] == 0
    assert row["qty"] == 0
    assert row["entry_price"] == 0.0
    assert row["entry_ts"] == 0.0
    assert row["open_fill_confirmed"] is False


def test_load_state_zeroizes_restored_half_open_position() -> None:
    _bootstrap_imports()
    import orchestrator_state_manager as osm  # noqa: E402

    now_ts = time.time()
    rows = [
        (
            "MSTR",
            json.dumps(
                {
                    "symbol": "MSTR",
                    "position": 1,
                    "qty": 20,
                    "entry_price": 7.12,
                    "entry_stock": 175.83,
                    "entry_ts": 0.0,
                    "warmup_complete": True,
                    "entry_slot_reserved": True,
                    "open_fill_confirmed": False,
                }
            ),
            now_ts,
        ),
        (
            "_GLOBAL_STATE_",
            json.dumps({"mock_cash": 50000.0, "mode": "REALTIME_DRY"}),
            now_ts,
        ),
    ]
    orch = _build_state_manager_orch()
    sm = osm.OrchestratorStateManager(orch)
    sm._get_pg_conn = lambda: _DummyConn(rows)

    restored = sm._load_state_from_db()
    row = restored["MSTR"]
    assert row["position"] == 0
    assert row["qty"] == 0
    assert row["entry_price"] == 0.0
    assert row["entry_ts"] == 0.0
    assert row["open_fill_confirmed"] is False


def test_exit_accounting_blocks_ghost_exit_without_cash_mutation() -> None:
    _bootstrap_imports()
    import orchestrator_accounting as oa  # noqa: E402
    from utils import serialization_utils as ser  # noqa: E402

    orch = _build_accounting_orch()
    acc = oa.OrchestratorAccounting(orch)
    st = _State("MSTR")
    st.position = 1
    st.qty = 20
    st.entry_price = 7.12
    st.entry_stock = 175.83
    st.entry_ts = 0.0
    st.entry_slot_reserved = True
    st.open_fill_confirmed = False

    before_cash = orch.mock_cash
    acc._process_exit_accounting(
        "MSTR",
        st,
        filled_qty=20,
        fill_price=7.72,
        stock_price=176.58,
        curr_ts=1_777_777_777.0,
        reason="UNIT_GHOST_CLOSE",
        duration=0.0,
        ratio=1.0,
    )

    assert orch.mock_cash == before_cash
    assert orch.trade_count == 0
    assert st.position == 0
    assert st.qty == 0
    assert st.open_fill_confirmed is False
    assert len(orch.r.xadds) == 1
    _stream, payload_dict, _maxlen = orch.r.xadds[0]
    payload = ser.unpack(payload_dict["data"])
    assert payload["action"] == "GHOST_EXIT_BLOCKED"
    assert payload["symbol"] == "MSTR"


def test_execute_entry_mid_save_persists_zeroized_then_confirmed_state() -> None:
    _bootstrap_imports()
    import orchestrator_accounting as oa  # noqa: E402
    import orchestrator_execution as oe  # noqa: E402
    import orchestrator_state_manager as osm  # noqa: E402

    st = _StateWithMidSave("MSTR")
    orch = _build_execution_orch(st)
    orch.accounting = oa.OrchestratorAccounting(orch)
    sm = osm.OrchestratorStateManager(orch)
    captured_rows = []

    def _capture_save(state_data, snapshot_ts=None):
        row = dict(state_data["MSTR"])
        row["_snapshot_ts"] = snapshot_ts
        captured_rows.append(row)

    sm._save_state_to_db = _capture_save
    orch.state_manager = sm
    st._mid_save_callback = sm.save_state

    ex = oe.OrchestratorExecution(orch)
    sig = {
        "tag": "CALL_ATM",
        "dir": 1,
        "reason": "UNIT_HALF_OPEN_TIMING",
        "price": 2.0,
        "meta": {"iv": 0.3, "contract_id": "MSTR260313C00180000", "strike": 180.0},
    }

    with patch.object(oe, "SYNC_EXECUTION", True), patch.object(oe, "TRADING_ENABLED", False):
        with patch.object(
            oe.LiquidityRiskManager,
            "evaluate_order",
            return_value={"final_alloc": 1000.0, "chunks": 1, "reason": "unit"},
        ):
            asyncio.run(ex._execute_entry("MSTR", sig, stock_price=175.83, curr_ts=1_777_777_700.0, batch_idx=0))

    assert len(captured_rows) >= 2, "应至少捕获一次半开仓快照和一次成交确认快照"
    half_open_row = captured_rows[0]
    final_row = captured_rows[-1]

    assert half_open_row["position"] == 0
    assert half_open_row["qty"] == 0
    assert half_open_row["entry_price"] == 0.0
    assert half_open_row["entry_ts"] == 0.0
    assert half_open_row["open_fill_confirmed"] is False

    assert final_row["position"] == 1
    assert final_row["qty"] > 0
    assert final_row["entry_price"] > 0
    assert final_row["entry_ts"] > 0
    assert final_row["open_fill_confirmed"] is True


def test_execute_entry_fault_injection_fuzz_keeps_half_open_snapshots_zeroized() -> None:
    _run_fault_injection_campaign(rounds=12, seed=20260423)


def test_execute_entry_fault_injection_soak_keeps_invariants_when_enabled() -> None:
    rounds = _fault_injection_rounds_from_env(default=0)
    if rounds <= 0:
        return
    seed = _fault_injection_seed_from_env(default=20260423)
    _run_fault_injection_campaign(rounds=rounds, seed=seed)


def main() -> None:
    test_save_state_zeroizes_half_open_position_before_persist()
    test_load_state_zeroizes_restored_half_open_position()
    test_exit_accounting_blocks_ghost_exit_without_cash_mutation()
    test_execute_entry_mid_save_persists_zeroized_then_confirmed_state()
    test_execute_entry_fault_injection_fuzz_keeps_half_open_snapshots_zeroized()
    test_execute_entry_fault_injection_soak_keeps_invariants_when_enabled()
    print("[OK] ghost position repro guards passed")


if __name__ == "__main__":
    main()

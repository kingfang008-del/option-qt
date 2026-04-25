#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import math
import random
import sys
from pathlib import Path
from types import SimpleNamespace


def _bootstrap_imports() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    production_dir = repo_root / "production"
    baseline_dir = production_dir / "baseline"
    sys.path.insert(0, str(production_dir))
    sys.path.insert(0, str(baseline_dir))


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
    def _process_open_accounting(self, *args, **kwargs):
        return None

    def _process_exit_accounting(self, *args, **kwargs):
        return None


class _DummyStateManager:
    def save_state(self):
        return None


def _build_orch() -> SimpleNamespace:
    cfg = SimpleNamespace(
        MAX_POSITIONS=4,
        POSITION_RATIO=0.25,
        MAX_TRADE_CAP=100000.0,
        GLOBAL_EXPOSURE_LIMIT=0.9,
        COMMISSION_PER_CONTRACT=0.65,
        LIMIT_BUFFER_ENTRY=1.03,
        LIMIT_BUFFER_EXIT=0.97,
        ORDER_TIMEOUT_SECONDS=5,
        ORDER_MAX_RETRIES=3,
        EXIT_ORDER_TYPE="LMT",
        SLIPPAGE_PCT=0.001,
        ENTRY_MAX_REQUOTE_SLIPPAGE_PCT=0.02,
        ENTRY_REQUOTE_STEP_CAP_PCT=0.006,
        STOP_LOSS=-0.2,
    )
    orch = SimpleNamespace(
        mode="realtime",
        states={},
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


def _build_execution_engine():
    _bootstrap_imports()
    import orchestrator_execution as oe  # noqa: E402

    return oe, oe.OrchestratorExecution(_build_orch())


def _quote_from_mid_spread(mid: float, spread_pct: float) -> tuple[float, float]:
    bid = mid * (1.0 - spread_pct / 2.0)
    ask = mid * (1.0 + spread_pct / 2.0)
    return bid, ask


def _bid_ask_ticks(bid: float, ask: float) -> tuple[float, float]:
    bid_tick = math.floor(bid * 100.0) / 100.0
    ask_tick = math.ceil(ask * 100.0) / 100.0
    return bid_tick, ask_tick


def _legacy_entry_limit_price(bid: float, ask: float, mid: float, attempt_no: int, limit_buffer_entry: float = 1.03) -> float:
    improvement = min(0.01 * max(attempt_no, 0), max((ask - mid), 0.0))
    return round(min(mid * limit_buffer_entry + improvement, ask), 2)


def _deterministic_policy_table() -> list[dict]:
    oe, ex = _build_execution_engine()
    rows = []
    for spread_pct in [x / 1000.0 for x in range(5, 101, 5)]:
        mid = 2.0
        bid, ask = _quote_from_mid_spread(mid, spread_pct)
        _bid_tick, ask_tick = _bid_ask_ticks(bid, ask)
        sig = {"meta": {"bid": bid, "ask": ask}}
        legacy_limit = _legacy_entry_limit_price(bid, ask, mid, attempt_no=0)
        limit_price = ex._get_entry_limit_price(sig, base_price=mid, attempt_no=0)
        cap_price = ex._entry_requote_cap_price(mid)
        rows.append(
            {
                "spread_pct": spread_pct * 100.0,
                "bid": bid,
                "ask": ask,
                "ask_tick": ask_tick,
                "legacy_initial_limit": legacy_limit,
                "initial_limit": limit_price,
                "legacy_equals_ask": math.isclose(legacy_limit, ask_tick, abs_tol=1e-9),
                "equals_ask": math.isclose(limit_price, ask_tick, abs_tol=1e-9),
                "cap_blocks_initial": limit_price > cap_price,
            }
        )
    return rows


def _simulate_campaign(
    *,
    runs: int,
    seed: int,
    spread_min: float,
    spread_max: float,
    price_fn,
    drift_abs_max: float = 0.003,
    vol_min: float = 0.001,
    vol_max: float = 0.012,
) -> dict:
    oe, ex = _build_execution_engine()
    rng = random.Random(seed)

    stats = {
        "runs": runs,
        "cap_reject": 0,
        "filled_attempt_1": 0,
        "filled_attempt_2": 0,
        "filled_attempt_3": 0,
        "unfilled_after_retry": 0,
        "initial_limit_equals_ask": 0,
    }

    for _ in range(runs):
        mid = rng.uniform(1.5, 8.0)
        spread_pct = rng.uniform(spread_min, spread_max)
        bid, ask = _quote_from_mid_spread(mid, spread_pct)
        sig = {"meta": {"bid": bid, "ask": ask}}

        current_limit = price_fn(ex, sig, mid, 0)
        _bid_tick, ask_tick = _bid_ask_ticks(bid, ask)
        if math.isclose(current_limit, ask_tick, abs_tol=1e-9):
            stats["initial_limit_equals_ask"] += 1

        cap_price = ex._entry_requote_cap_price(mid)
        if current_limit > cap_price:
            stats["cap_reject"] += 1
            continue

        drift_per_sec = rng.uniform(-drift_abs_max, drift_abs_max)
        vol_per_sec = rng.uniform(vol_min, vol_max)
        filled = False
        latest_bid, latest_ask = bid, ask

        for attempt_no in range(3):
            for _sec in range(5):
                shock = rng.gauss(drift_per_sec, vol_per_sec)
                mid = max(0.05, mid * (1.0 + shock))
                spread_pct = min(0.18, max(0.005, spread_pct + rng.gauss(0.0, 0.004)))
                latest_bid, latest_ask = _quote_from_mid_spread(mid, spread_pct)
                if latest_ask <= current_limit:
                    stats[f"filled_attempt_{attempt_no + 1}"] += 1
                    filled = True
                    break
            if filled:
                break
            if attempt_no >= 2:
                continue
            sig = {"meta": {"bid": latest_bid, "ask": latest_ask}}
            current_limit = price_fn(ex, sig, mid, attempt_no + 1)
            current_limit = min(current_limit, cap_price)
            if current_limit <= 0.0:
                break

        if not filled:
            stats["unfilled_after_retry"] += 1

    total_filled = stats["filled_attempt_1"] + stats["filled_attempt_2"] + stats["filled_attempt_3"]
    stats["fill_rate"] = total_filled / runs if runs else 0.0
    stats["ask_hit_rate"] = stats["initial_limit_equals_ask"] / runs if runs else 0.0
    stats["cap_reject_rate"] = stats["cap_reject"] / runs if runs else 0.0
    return stats


def _print_deterministic_table(rows: list[dict]) -> None:
    print("== 初始限价是否直接打到 ask ==")
    print("spread%\tlegacy\tcurrent\task_tick\tlegacy=ask\tcurrent=ask\tcap_block")
    for row in rows:
        print(
            f"{row['spread_pct']:.1f}\t"
            f"{row['legacy_initial_limit']:.2f}\t"
            f"{row['initial_limit']:.2f}\t"
            f"{row['ask_tick']:.2f}\t"
            f"{int(row['legacy_equals_ask'])}\t"
            f"{int(row['equals_ask'])}\t"
            f"{int(row['cap_blocks_initial'])}"
        )


def _print_campaign(label: str, stats: dict) -> None:
    print(f"\n== {label} ==")
    print(
        {
            "runs": stats["runs"],
            "fill_rate": round(stats["fill_rate"], 4),
            "ask_hit_rate": round(stats["ask_hit_rate"], 4),
            "cap_reject_rate": round(stats["cap_reject_rate"], 4),
            "filled_attempt_1": round(stats["filled_attempt_1"] / stats["runs"], 4),
            "filled_attempt_2": round(stats["filled_attempt_2"] / stats["runs"], 4),
            "filled_attempt_3": round(stats["filled_attempt_3"] / stats["runs"], 4),
            "unfilled_after_retry": round(stats["unfilled_after_retry"] / stats["runs"], 4),
        }
    )


def _current_price_fn(ex, sig, base_price, attempt_no):
    return ex._get_entry_limit_price(sig, base_price=base_price, attempt_no=attempt_no)


def _legacy_price_fn(_ex, sig, base_price, attempt_no):
    bid = float(sig["meta"]["bid"])
    ask = float(sig["meta"]["ask"])
    return _legacy_entry_limit_price(bid, ask, base_price, attempt_no)


def main() -> None:
    rows = _deterministic_policy_table()
    _print_deterministic_table(rows)

    for regime_label, spread_min, spread_max in [
        ("tight_0.5%-2%", 0.005, 0.02),
        ("normal_1%-4%", 0.01, 0.04),
        ("borderline_4%-6%", 0.04, 0.06),
        ("wide_6%-10%", 0.06, 0.10),
    ]:
        _print_campaign(
            f"{regime_label} / legacy",
            _simulate_campaign(
                runs=15000,
                seed=20260423,
                spread_min=spread_min,
                spread_max=spread_max,
                price_fn=_legacy_price_fn,
            ),
        )
        _print_campaign(
            f"{regime_label} / current",
            _simulate_campaign(
                runs=15000,
                seed=20260423,
                spread_min=spread_min,
                spread_max=spread_max,
                price_fn=_current_price_fn,
            ),
        )


if __name__ == "__main__":
    main()

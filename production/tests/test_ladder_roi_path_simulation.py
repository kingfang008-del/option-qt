#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模拟 execution_engine_v8：按帧推进 ROI（线性近似 5 分钟），与 OMS 一样更新 holding.max_roi，
再调用 strategy.check_exit(ctx)，观测首次止盈/止损的 reason。

与真实 EE 对齐的要点：
- current_roi = (curr_price - entry_price) / entry_price
- 每帧 max_roi = max(历史 max_roi, current_roi)（参见 execution_engine_v8.py 2190-2193）
"""

from __future__ import annotations

import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import List, NamedTuple, Optional


def _bootstrap_imports() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    production_dir = repo_root / "production"
    baseline_dir = production_dir / "baseline"
    sys.path.insert(0, str(production_dir))
    sys.path.insert(0, str(baseline_dir))


class ExitEvent(NamedTuple):
    step_index: int
    held_mins: float
    roi: float
    max_roi: float
    curr_price: float
    sig: dict


def _mk_ctx_extended(
    *,
    symbol: str,
    curr_ts: float,
    entry_ts: float,
    held_mins: float,
    entry_price: float,
    curr_price: float,
    max_roi: float,
    direction: int,
    benign_index: bool = True,
    snap_ok: float = 0.0008,
    macd_ok: float = 0.02,
    alpha_ok: float = 1.3,
    stock_frac: float = 0.12,
) -> dict:
    """补充 Trend / V0.check_exit 常见依赖；正股微弱随 ROI 为正，避免误触 STOCK_STOP。"""
    entry_stock = 100.0
    roi = (curr_price - entry_price) / entry_price if entry_price > 0 else 0.0
    curr_stock = entry_stock * (1.0 + stock_frac * roi)
    idx_now = 1 if benign_index and direction >= 1 else (-1 if direction < 0 else 0)
    et = 1 if direction >= 1 else (-1 if direction < 0 else 0)
    ny = datetime.fromtimestamp(curr_ts)

    return {
        "symbol": symbol,
        "time": ny,
        "curr_ts": curr_ts,
        "price": curr_stock,
        "alpha_z": alpha_ok,
        "vol_z": 0.0,
        "stock_roc": roi * 0.5,
        "event_prob": 0.0,
        "macd_hist": macd_ok,
        "macd_hist_slope": 0.0001,
        "spy_roc": 0.0002 if benign_index else 0.0,
        "qqq_roc": 0.0002 if benign_index else 0.0,
        "index_trend": idx_now,
        "position": direction,
        "cooldown_until": 0.0,
        "is_ready": True,
        "is_banned": False,
        "held_mins": held_mins,
        "stock_iv": 0.35,
        "holding": {
            "entry_price": entry_price,
            "entry_stock": entry_stock,
            "entry_ts": entry_ts,
            "dir": direction,
            "max_roi": max_roi,
            "entry_spy_roc": 0.0002 if direction >= 0 else -0.0002,
            "entry_index_trend": et,
        },
        "curr_price": curr_price,
        "curr_stock": curr_stock,
        "bid": max(curr_price * 0.994, 0.02),
        "ask": max(curr_price * 1.004, 0.022),
        "spread_divergence": 0.0,
        "snap_roc": snap_ok if direction >= 1 else -snap_ok,
        "global_regime_reversal_cnt": 0,
        "regime_reversal_count": 0,
        "is_volatile_regime": False,
        "regime_band": "calm",
        "regime_score": 0.0,
        "state": None,
    }


def linspace_segments(peak_roi: float, end_roi: float, n_up: int, n_down: int) -> List[float]:
    """两段线性：先到 peak，再回到 end（不包含重复的 peak）。"""
    if n_up < 2:
        n_up = 2
    up = [peak_roi * i / (n_up - 1) for i in range(n_up)]
    down = []
    if n_down >= 2:
        for j in range(1, n_down):
            frac = j / (n_down - 1)
            down.append(peak_roi + frac * (end_roi - peak_roi))
    elif n_down == 1:
        down.append(end_roi)
    return up + down


def simulate_first_exit(
    strategy,
    *,
    entry_price: float,
    roi_path: List[float],
    horizon_mins: float = 5.0,
    entry_ts_unix: float = 1_700_000_000.0,
) -> Optional[ExitEvent]:
    max_roi_seen = max(0.0, roi_path[0] if roi_path else 0.0)
    n = len(roi_path)
    for i, roi_frac in enumerate(roi_path):
        held = horizon_mins * i / max(n - 1, 1)
        curr_ts = entry_ts_unix + held * 60.0
        curr_price = entry_price * (1.0 + roi_frac)
        max_roi_seen = max(max_roi_seen, roi_frac)
        ctx = _mk_ctx_extended(
            symbol="SIM",
            curr_ts=curr_ts,
            entry_ts=entry_ts_unix,
            held_mins=held,
            entry_price=entry_price,
            curr_price=curr_price,
            max_roi=max_roi_seen,
            direction=1,
        )
        ctx["holding"]["max_roi"] = max_roi_seen
        sig = strategy.check_exit(ctx)
        if sig is not None and sig.get("action") == "SELL":
            return ExitEvent(
                step_index=i,
                held_mins=held,
                roi=roi_frac,
                max_roi=max_roi_seen,
                curr_price=curr_price,
                sig=sig,
            )
        max_roi_seen = max(max_roi_seen, roi_frac)
    return None


def _isolated_cfg():
    """减少时间止损/僵尸/无动量对「阶梯止盈」路径测试的抢先触发。"""
    _bootstrap_imports()
    from strategy_config0 import StrategyConfig  # noqa: E402

    return replace(
        StrategyConfig(),
        NO_MOMENTUM_MINS=9999,
        MID_TIME_STOP_MINS=9999,
        TIME_STOP_MINS=9999,
        EXIT_ZOMBIE_STOP_ENABLED=False,
        ZOMBIE_EXIT_MINS=9999,
        EXIT_SMALL_GAIN_ENABLED=False,
        EXIT_MACD_FADE_ENABLED=False,
        EXIT_SIGNAL_FLIP_ENABLED=False,
        EXIT_COND_STOP_ENABLED=False,
    )


def test_v0_path_20pct_peak_then_10pct_triggers_step_ladder_first() -> None:
    """纯 V0：拉升至 20% 再回落至 10%；应在下穿 17% floor（20% 档阶梯）时已 STEP_PROT."""
    _bootstrap_imports()
    from strategy_core_v0 import StrategyCoreV0  # noqa: E402

    cfg = _isolated_cfg()
    core = StrategyCoreV0(cfg)
    path = linspace_segments(0.20, 0.10, n_up=26, n_down=36)
    ev = simulate_first_exit(core, entry_price=5.0, roi_path=path, horizon_mins=5.0)
    assert ev is not None, "应在回落过程中触发平仓"
    assert "STEP_PROT" in ev.sig.get("reason", ""), f"期望阶梯止盈, got {ev.sig}"
    assert ev.roi < 0.17 + 1e-6, f"首触为20%档位 floor=17%，应 roi<17%，实际 ROI={ev.roi:.4f}"


def test_trend_path_20pct_peak_then_10pct_exits_via_trend_pre_super_rules() -> None:
    """TREND：同一路径会先经过 trend trail protect；通常早于 V0 STEP_PROT。"""
    _bootstrap_imports()
    from strategy_core_trend import StrategyCoreTrend  # noqa: E402

    cfg = _isolated_cfg()
    core = StrategyCoreTrend(cfg)
    path = linspace_segments(0.20, 0.10, n_up=26, n_down=36)
    ev = simulate_first_exit(core, entry_price=5.0, roi_path=path, horizon_mins=5.0)
    assert ev is not None
    reason = ev.sig.get("reason", "")
    assert (
        "TREND_TRAIL" in reason or "STEP_PROT" in reason or "TREND_PROTECT" in reason
    ), f"unexpected first exit reason: {reason}"


def test_trend_path_15pct_peak_then_3pct_step_before_flat() -> None:
    """峰值 15% 回落至 3%：super() 里 15% 档 floor 12%；应在 ROI 下穿 12% 时 STEP_PROT（先于兜底 TREND_PROTECT 3%）。"""
    _bootstrap_imports()
    from strategy_core_trend import StrategyCoreTrend  # noqa: E402

    cfg = _isolated_cfg()
    core = StrategyCoreTrend(cfg)
    path = linspace_segments(0.15, 0.03, n_up=21, n_down=41)
    ev = simulate_first_exit(core, entry_price=5.0, roi_path=path, horizon_mins=5.0)
    assert ev is not None, f"path={len(path)} 步内应触发: {path[:3]} ... {path[-3:]}"
    assert "STEP_PROT" in ev.sig.get("reason", ""), (
        "期望先于 3% 由阶梯(15%→floor12%)拦住，实际: "
        + str(ev.sig)
        + f" roi={ev.roi:.4f} max_roi={ev.max_roi:.4f}"
    )
    assert ev.roi + 1e-8 < 0.12 + 1e-3, f"应略低于12%档位: roi={ev.roi}"


def _print_human_summary() -> None:
    """`python test_ladder_roi_path_simulation.py` 时打印路径摘要。"""
    _bootstrap_imports()
    from strategy_core_v0 import StrategyCoreV0  # noqa: E402
    from strategy_core_trend import StrategyCoreTrend  # noqa: E402

    cfg_i = _isolated_cfg()
    rows = [
        ("V0", 0.20, 0.10, StrategyCoreV0(cfg_i)),
        ("TREND", 0.20, 0.10, StrategyCoreTrend(cfg_i)),
        ("TREND", 0.15, 0.03, StrategyCoreTrend(cfg_i)),
    ]
    print("\n[Ladder ROI sim — mirrors EE max_roi & check_exit]")
    for tag, pk, ek, strat in rows:
        path = (
            linspace_segments(0.15, 0.03, n_up=21, n_down=41)
            if abs(pk - 0.15) < 1e-6
            else linspace_segments(pk, ek, n_up=26, n_down=36)
        )
        ev = simulate_first_exit(strat, entry_price=5.0, roi_path=path, horizon_mins=5.0)
        if ev:
            print(
                f"  {tag} peak={pk:.0%}->{ek:.0%} | step={ev.step_index} "
                f"held={ev.held_mins:.2f}m roi={ev.roi:.2%} peak_tracked={ev.max_roi:.2%} "
                f"| {ev.sig.get('reason')}"
            )
        else:
            print(f"  {tag} peak={pk:.0%}->{ek:.0%} | NO EXIT in {len(path)} steps")


if __name__ == "__main__":
    test_v0_path_20pct_peak_then_10pct_triggers_step_ladder_first()
    test_trend_path_20pct_peak_then_10pct_exits_via_trend_pre_super_rules()
    test_trend_path_15pct_peak_then_3pct_step_before_flat()
    _print_human_summary()
    print("All ladder ROI path simulations passed.")

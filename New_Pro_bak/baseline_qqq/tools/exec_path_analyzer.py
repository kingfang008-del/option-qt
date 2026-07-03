#!/usr/bin/env python3
"""
Replay gate-by-gate: first tradeable frame → option path → exits.

Usage:
  cd New_Pro/baseline_qqq
  MIN_OPTION_PRICE=0.25 python tools/exec_path_analyzer.py
  EXEC_PROFILE=multi_band MIN_OPTION_PRICE=0.25 python tools/exec_path_analyzer.py --scenario actionable
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytz

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from entry_risk_rules import evaluate_entry_liquidity
from exec_profile import (
    ExecMode,
    ExecProfile,
    attach_exec_plan_to_signal,
    multi_band_roll_cooldown_seconds,
    parse_exec_profile,
    resolve_exec_plan,
)
from strategy_config0 import StrategyConfig
from strategy_core_v0 import StrategyCoreV0
from bidirectional_regime import resolve_day_type

NY = pytz.timezone("America/New_York")


@dataclass
class MinuteBar:
    """One decision minute on the replay tape."""
    time_str: str          # HH:MM
    opt_mid: float
    opt_bid: float
    opt_ask: float
    stock_price: float
    stock_roc_5m: float
    snap_roc: float
    alpha: float
    vol_z: float = 1.5
    spy_roc: float = 0.0
    qqq_roc: float = 0.0
    spread_feat: float = 0.08
    iv_momentum: float = 0.15
    is_volatile: bool = False
    is_ready: bool = True
    macd_hist: float = 0.012
    qqq_day_roc: float = 0.0


@dataclass
class TradeLeg:
    entry_time: str
    entry_price: float
    exit_time: str = ""
    exit_price: float = 0.0
    exit_reason: str = ""
    max_roi: float = 0.0
    exec_mode: str = "SWING"
    exec_band: str = ""


@dataclass
class AnalyzerState:
    position: int = 0
    entry_price: float = 0.0
    entry_stock: float = 0.0
    entry_ts: float = 0.0
    max_roi: float = -1.0
    exec_mode: str = ExecMode.SWING.value
    exec_profile: str = ""
    exec_band: str = ""
    hold_profile: str = ""
    band_legs_today: int = 0
    cooldown_until: float = 0.0
    legs: List[TradeLeg] = field(default_factory=list)


def _ts_from_hhmm(day: str, hhmm: str) -> float:
    dt = datetime.strptime(f"{day} {hhmm}", "%Y-%m-%d %H:%M")
    return NY.localize(dt).timestamp()


def qqq_dislocation_scenario() -> List[MinuteBar]:
    """
    合成路径：早盘急跌期权至 $0.30，正股日内约 +0.85%，期权回升至 $3。
    用于演示门禁；实盘请替换为 sniper/1m 真实序列。
    """
    # time, mid, spread%, stock, roc5m, snap, alpha, volatile
    raw = [
        ("09:45", 2.40, 0.06, 520.0, -0.0008, -0.0005, 0.008, False),
        ("09:46", 1.80, 0.08, 519.2, -0.0015, -0.0012, 0.006, False),
        ("09:47", 1.10, 0.10, 518.5, -0.0022, -0.0018, 0.005, True),
        ("09:48", 0.65, 0.14, 517.8, -0.0030, -0.0025, 0.004, True),
        ("09:49", 0.42, 0.16, 517.2, -0.0035, -0.0030, 0.003, True),
        ("09:50", 0.30, 0.20, 516.8, -0.0040, -0.0040, 0.002, True),
        ("09:51", 0.35, 0.18, 517.0, -0.0032, 0.0008, 0.016, True),
        ("09:52", 0.48, 0.14, 517.5, -0.0025, 0.0015, 0.022, True),
        ("09:53", 0.62, 0.12, 518.0, -0.0018, 0.0020, 0.028, False),
        ("09:54", 0.78, 0.10, 518.6, -0.0010, 0.0022, 0.032, False),
        ("09:55", 0.95, 0.09, 519.2, -0.0002, 0.0025, 0.035, False),
        ("09:58", 1.25, 0.08, 520.0, 0.0008, 0.0028, 0.038, False),
        ("10:02", 1.55, 0.07, 521.0, 0.0018, 0.0018, 0.034, False),
        ("10:08", 1.85, 0.06, 522.0, 0.0028, 0.0015, 0.030, False),
        ("10:15", 2.20, 0.06, 523.5, 0.0040, 0.0012, 0.028, False),
        ("10:25", 2.55, 0.05, 525.0, 0.0055, 0.0010, 0.026, False),
        ("10:40", 2.85, 0.05, 526.5, 0.0065, 0.0008, 0.024, False),
        ("11:00", 3.00, 0.05, 528.0, 0.0075, 0.0006, 0.022, False),
        ("11:30", 2.70, 0.06, 527.5, 0.0060, -0.0005, 0.018, False),
        ("12:00", 2.40, 0.07, 526.0, 0.0040, -0.0010, 0.015, False),
    ]
    bars: List[MinuteBar] = []
    for row in raw:
        t, mid, sp_pct, stk, roc5, snap, alpha, volatile = row
        half = mid * sp_pct / 2.0
        bid = max(0.01, mid - half)
        ask = mid + half
        idx = roc5 + 0.0005
        bars.append(
            MinuteBar(
                time_str=t,
                opt_mid=mid,
                opt_bid=bid,
                opt_ask=ask,
                stock_price=stk,
                stock_roc_5m=roc5,
                snap_roc=snap,
                alpha=alpha,
                vol_z=2.8 if volatile else 1.8,
                spy_roc=idx * 0.9,
                qqq_roc=idx,
                spread_feat=min(sp_pct, 0.12),
                iv_momentum=0.42 if volatile else 0.10,
                is_volatile=volatile,
                is_ready=True,
            )
        )
    return bars


def qqq_dislocation_tight_spread_scenario() -> List[MinuteBar]:
    """急跌后修复段 spread 收窄至 ≤5%，用于模拟「去掉 $2 后能否进场」。"""
    bars = qqq_dislocation_scenario()
    for b in bars:
        if b.time_str >= "09:53":
            b.spread_feat = 0.04
            half = b.opt_mid * 0.04 / 2.0
            b.opt_bid = max(0.01, b.opt_mid - half)
            b.opt_ask = b.opt_mid + half
            b.is_volatile = False
            b.iv_momentum = 0.08
        if b.time_str >= "09:55":
            b.stock_roc_5m = max(b.stock_roc_5m, 0.0003)
            b.qqq_roc = max(b.qqq_roc, 0.0002)
            b.spy_roc = max(b.spy_roc, 0.00015)
    return bars


def qqq_dislocation_actionable_scenario() -> List[MinuteBar]:
    """
    修复段：spread≤5%、alpha/动量/MACD 同步放行，用于估算「能进场时」的止盈路径。
  """
    bars = qqq_dislocation_tight_spread_scenario()
    for b in bars:
        if b.time_str >= "09:58":
            b.macd_hist = 0.020
            b.stock_roc_5m = max(b.stock_roc_5m, 0.0004)
            b.qqq_roc = max(b.qqq_roc, 0.00025)
            b.spy_roc = max(b.spy_roc, 0.0002)
            b.alpha = max(b.alpha, 0.032)
            b.snap_roc = max(b.snap_roc, 0.0020)
        if b.time_str >= "10:15":
            b.macd_hist = 0.020
            b.stock_roc_5m = max(b.stock_roc_5m, 0.0006)
    return bars


def qqq_dislocation_multi_band_roll_scenario() -> List[MinuteBar]:
    """
    三腿滚仓演示路径：BAND1 错价(09:51) → BAND2 趋势(10:02) → BAND3 epic(10:40)。
    早盘 V 底段 5m ROC 仍为负，靠 snap_roc + dislocation 门进场。
    """
    raw = [
        ("09:45", 2.40, 0.06, 520.0, -0.0008, -0.0005, 0.008, False),
        ("09:46", 1.80, 0.08, 519.2, -0.0015, -0.0012, 0.006, False),
        ("09:47", 1.10, 0.10, 518.5, -0.0022, -0.0018, 0.005, True),
        ("09:48", 0.65, 0.14, 517.8, -0.0030, -0.0025, 0.004, True),
        ("09:49", 0.42, 0.16, 517.2, -0.0035, -0.0030, 0.003, True),
        ("09:50", 0.30, 0.14, 516.8, -0.0040, -0.0040, 0.002, True),
        ("09:51", 0.35, 0.12, 517.0, -0.0032, 0.0012, 0.018, True),
        ("09:52", 0.44, 0.10, 517.5, -0.0025, 0.0015, 0.022, True),
        ("09:53", 0.52, 0.09, 518.0, -0.0018, 0.0018, 0.026, False),
        ("09:54", 0.55, 0.08, 518.4, -0.0015, 0.0010, 0.024, False),
        ("09:55", 0.48, 0.09, 518.2, -0.0012, -0.0005, 0.020, False),
        ("09:58", 0.62, 0.06, 518.8, -0.0008, 0.0005, 0.022, False),
        ("10:03", 1.48, 0.04, 520.5, 0.0012, 0.0015, 0.034, False),
        ("10:08", 1.85, 0.04, 522.0, 0.0025, 0.0015, 0.032, False),
        ("10:15", 2.20, 0.04, 523.5, 0.0040, 0.0012, 0.030, False),
        ("10:25", 2.55, 0.04, 525.0, 0.0055, 0.0010, 0.028, False),
        ("10:30", 2.05, 0.04, 524.0, 0.0045, -0.0002, 0.026, False),
        ("10:38", 2.35, 0.04, 524.5, 0.0048, 0.0005, 0.025, False),
        ("10:40", 2.55, 0.05, 526.0, 0.0060, 0.0008, 0.024, False),
        ("11:00", 2.95, 0.05, 528.0, 0.0075, 0.0006, 0.022, False),
        ("11:05", 3.15, 0.05, 528.5, 0.0078, 0.0004, 0.021, False),
        ("11:20", 2.82, 0.06, 527.5, 0.0068, -0.0006, 0.019, False),
        ("12:00", 2.50, 0.07, 526.0, 0.0040, -0.0010, 0.015, False),
    ]
    bars: List[MinuteBar] = []
    for row in raw:
        t, mid, sp_pct, stk, roc5, snap, alpha, volatile = row
        half = mid * sp_pct / 2.0
        bid = max(0.01, mid - half)
        ask = mid + half
        idx = roc5 + 0.0005
        macd = 0.012
        if t >= "10:03":
            macd = 0.020
        bars.append(
            MinuteBar(
                time_str=t,
                opt_mid=mid,
                opt_bid=bid,
                opt_ask=ask,
                stock_price=stk,
                stock_roc_5m=roc5,
                snap_roc=snap,
                alpha=alpha,
                vol_z=2.8 if volatile else 1.8,
                spy_roc=idx * 0.9,
                qqq_roc=idx,
                spread_feat=min(sp_pct, 0.12),
                iv_momentum=0.42 if volatile else 0.10,
                is_volatile=volatile,
                is_ready=True,
                macd_hist=macd,
            )
        )
    return bars


def build_ctx(
    bar: MinuteBar,
    st: AnalyzerState,
    curr_ts: float,
    cfg: StrategyConfig,
    *,
    exec_profile: str = "",
) -> dict:
    ny = datetime.fromtimestamp(curr_ts, tz=NY)
    idx_trend = 1 if bar.qqq_roc > 0.0001 else (-1 if bar.qqq_roc < -0.0001 else 0)
    holding = None
    if st.position != 0:
        holding = {
            "entry_price": st.entry_price,
            "entry_stock": st.entry_stock,
            "entry_ts": st.entry_ts,
            "dir": st.position,
            "max_roi": st.max_roi,
            "entry_spy_roc": bar.spy_roc,
            "entry_index_trend": idx_trend,
            "exec_mode": st.exec_mode,
            "exec_band": st.exec_band,
            "exec_profile": st.exec_profile,
            "hold_profile": st.hold_profile,
            "init_ctx": {
                "exec_mode": st.exec_mode,
                "exec_band": st.exec_band,
                "exec_profile": st.exec_profile,
                "alpha_z": bar.alpha,
            },
        }
    return {
        "symbol": "QQQ",
        "time": ny,
        "curr_ts": curr_ts,
        "price": bar.stock_price,
        "alpha": bar.alpha,
        "alpha_z": bar.alpha,
        "vol_z": bar.vol_z,
        "stock_roc": bar.stock_roc_5m,
        "spy_roc": bar.spy_roc,
        "qqq_roc": bar.qqq_roc,
        "index_trend": idx_trend,
        "position": st.position,
        "cooldown_until": st.cooldown_until,
        "is_ready": bar.is_ready,
        "is_banned": False,
        "held_mins": (curr_ts - st.entry_ts) / 60.0 if st.position else 0.0,
        "stock_iv": 0.35,
        "holding": holding,
        "curr_price": bar.opt_mid,
        "curr_stock": bar.stock_price,
        "bid": bar.opt_bid,
        "ask": bar.opt_ask,
        "spread_divergence": 0.0,
        "snap_roc": bar.snap_roc,
        "options_vw_spread": bar.spread_feat,
        "options_iv_momentum": bar.iv_momentum,
        "is_volatile_regime": bar.is_volatile,
        "regime_reversal_count": 8 if bar.is_volatile else 2,
        "macd_hist": float(getattr(bar, "macd_hist", 0.012)),
        "macd_hist_slope": 0.001 if bar.snap_roc > 0 else -0.001,
        "event_prob": 0.0,
        "exec_profile": exec_profile,
        "qqq_day_roc": float(getattr(bar, "qqq_day_roc", 0.0)),
        "day_type": resolve_day_type({
            "qqq_day_roc": float(getattr(bar, "qqq_day_roc", 0.0)),
            "stock_roc": bar.stock_roc_5m,
            "snap_roc": bar.snap_roc,
            "alpha": bar.alpha,
        }).value,
    }


def run_replay(
    bars: List[MinuteBar],
    *,
    exec_profile: str = "auto_hybrid",
    multi_band: bool = False,
    cooldown_mins: int = 15,
    day: str = "2025-05-28",
) -> Dict[str, Any]:
    cfg = StrategyConfig()
    strategy = StrategyCoreV0(cfg)
    profile = parse_exec_profile(exec_profile)
    is_multi_band = profile == ExecProfile.MULTI_BAND or multi_band
    st = AnalyzerState()
    timeline: List[Dict[str, Any]] = []
    target_hit = False

    for bar in bars:
        curr_ts = _ts_from_hhmm(day, bar.time_str)
        ctx = build_ctx(bar, st, curr_ts, cfg, exec_profile=exec_profile)

        row: Dict[str, Any] = {
            "time": bar.time_str,
            "opt_mid": bar.opt_mid,
            "stock": bar.stock_price,
            "alpha": bar.alpha,
            "position": st.position,
            "events": [],
        }

        if bar.opt_mid >= 3.0:
            target_hit = True

        # --- exit path ---
        if st.position != 0:
            roi = (bar.opt_mid - st.entry_price) / st.entry_price if st.entry_price > 0 else 0.0
            st.max_roi = max(st.max_roi, roi)
            ctx["holding"]["max_roi"] = st.max_roi
            exit_sig = strategy.check_exit(ctx)
            if exit_sig:
                reason = str(exit_sig.get("reason", "EXIT"))
                row["events"].append({
                    "type": "EXIT",
                    "reason": reason,
                    "roi": roi,
                    "max_roi": st.max_roi,
                    "entry": st.entry_price,
                    "exit_px": bar.opt_mid,
                    "gates": strategy.get_last_gate_trace(),
                })
                leg = st.legs[-1]
                leg.exit_time = bar.time_str
                leg.exit_price = bar.opt_mid
                leg.exit_reason = reason
                leg.max_roi = st.max_roi
                st.position = 0
                st.entry_price = 0.0
                st.max_roi = -1.0
                st.exec_band = ""
                if is_multi_band:
                    st.cooldown_until = curr_ts + multi_band_roll_cooldown_seconds(
                        cfg, profitable=(roi >= 0), reason=reason,
                    )
                elif multi_band:
                    st.cooldown_until = curr_ts + cooldown_mins * 60
            else:
                row["events"].append({
                    "type": "HOLD",
                    "roi": roi,
                    "max_roi": st.max_roi,
                    "held_mins": ctx["held_mins"],
                })
            timeline.append(row)
            continue

        # --- entry path ---
        if curr_ts < st.cooldown_until:
            row["events"].append({
                "type": "ENTRY_SKIP",
                "reason": "cooldown",
                "remain_mins": (st.cooldown_until - curr_ts) / 60.0,
            })
            timeline.append(row)
            continue

        entry_sig = strategy.decide_entry(ctx)
        gate_trace = strategy.get_last_gate_trace()
        blocked = [g for g in gate_trace if g.get("status") == "block"]

        liq = evaluate_entry_liquidity(
            bid=bar.opt_bid,
            ask=bar.opt_ask,
            curr_price=bar.opt_mid,
            alpha_z=bar.alpha,
            spread_divergence=0.0,
            cfg=cfg,
            spread_threshold_override=(
                cfg.MULTI_BAND1_MAX_SPREAD
                if is_multi_band and bar.opt_mid < cfg.MULTI_BAND2_PRICE_FLOOR
                else None
            ),
        )

        if not entry_sig:
            row["events"].append({
                "type": "ENTRY_REJECT",
                "strategy_blocks": [f"{g['gate']}: {g.get('detail','')}" for g in blocked[:5]],
                "liquidity": liq["reason"] if not liq["ok"] else "ok",
                "liquidity_detail": liq.get("detail", ""),
            })
            timeline.append(row)
            continue

        if not liq["ok"]:
            row["events"].append({
                "type": "ENTRY_REJECT",
                "strategy": "pass",
                "liquidity": liq["reason"],
                "liquidity_detail": liq["detail"],
            })
            timeline.append(row)
            continue

        plan = resolve_exec_plan(profile, ctx, cfg, legs_today=st.band_legs_today)
        if not plan.exec_band and plan.hold_profile == "band_blocked":
            row["events"].append({
                "type": "ENTRY_SKIP",
                "reason": f"multi_band:{plan.reason}",
            })
            timeline.append(row)
            continue

        attach_exec_plan_to_signal(entry_sig, plan)
        fill = bar.opt_ask  # 保守：用 ask 入场
        st.position = int(entry_sig["dir"])
        st.entry_price = fill
        st.entry_stock = bar.stock_price
        st.entry_ts = curr_ts
        st.max_roi = 0.0
        st.exec_mode = plan.mode.value
        st.exec_profile = plan.profile
        st.exec_band = plan.exec_band or ""
        st.hold_profile = plan.hold_profile
        if plan.profile == ExecProfile.MULTI_BAND.value and plan.exec_band:
            st.band_legs_today += 1
        st.legs.append(
            TradeLeg(
                entry_time=bar.time_str,
                entry_price=fill,
                exec_mode=plan.mode.value,
                exec_band=plan.exec_band or "",
            )
        )
        row["events"].append({
            "type": "ENTRY",
            "price": fill,
            "exec_mode": plan.mode.value,
            "exec_band": plan.exec_band or "-",
            "exec_dte": plan.target_dte,
            "reason": entry_sig.get("reason"),
            "route": plan.reason,
            "leg_n": st.band_legs_today if is_multi_band else 1,
            "gates_passed": len([g for g in gate_trace if g.get("status") == "pass"]),
        })
        timeline.append(row)

    # mark-to-market open leg
    open_pnl = 0.0
    if st.position != 0 and bars:
        last = bars[-1]
        open_pnl = (last.opt_mid - st.entry_price) / st.entry_price

    closed_pnl = sum(
        (lg.exit_price - lg.entry_price) / lg.entry_price
        for lg in st.legs
        if lg.exit_time
    )
    theoretical_buy_hold = (bars[-1].opt_mid - bars[0].opt_mid) / bars[0].opt_mid if bars else 0.0
    best_entry = min(b.opt_mid for b in bars)
    theoretical_perfect = (bars[-1].opt_mid - best_entry) / best_entry if best_entry > 0 else 0.0

    return {
        "config": {
            "MIN_OPTION_PRICE": cfg.MIN_OPTION_PRICE,
            "EXEC_PROFILE": exec_profile,
            "multi_band": is_multi_band,
        },
        "target_3_dollars_hit": target_hit,
        "timeline": timeline,
        "legs": [
            {
                "entry": lg.entry_time,
                "entry_px": lg.entry_price,
                "exit": lg.exit_time,
                "exit_px": lg.exit_price,
                "max_roi": lg.max_roi,
                "realized_roi": (lg.exit_price - lg.entry_price) / lg.entry_price if lg.exit_time else None,
                "reason": lg.exit_reason,
                "mode": lg.exec_mode,
                "band": lg.exec_band,
            }
            for lg in st.legs
        ],
        "band_legs_today": st.band_legs_today,
        "summary": {
            "closed_legs": sum(1 for lg in st.legs if lg.exit_time),
            "closed_pnl_sum_roi": closed_pnl,
            "open_leg_roi": open_pnl if st.position else 0.0,
            "total_strategy_roi": closed_pnl + (open_pnl if st.position else 0.0),
            "buy_hold_first_bar_roi": theoretical_buy_hold,
            "perfect_bottom_to_top_roi": theoretical_perfect,
            "capture_vs_perfect_pct": (
                100.0 * (closed_pnl + (open_pnl if st.position else 0.0)) / theoretical_perfect
                if theoretical_perfect > 0
                else 0.0
            ),
        },
    }


def print_report(result: Dict[str, Any]) -> None:
    cfg = result["config"]
    print("=" * 72)
    print(f"Exec Path Analyzer | MIN_OPTION_PRICE=${cfg['MIN_OPTION_PRICE']:.2f} | profile={cfg['EXEC_PROFILE']}")
    print(f"multi_band={cfg['multi_band']} | target $3 hit={result['target_3_dollars_hit']}")
    print("=" * 72)
    for row in result["timeline"]:
        evs_parts = []
        for e in row["events"]:
            if e["type"] == "ENTRY_REJECT":
                strat = e.get("strategy_blocks") or []
                strat_s = strat[0] if strat else "strategy_pass"
                evs_parts.append(
                    f"REJECT liq={e.get('liquidity')} strat={strat_s}"
                )
            elif e["type"] == "HOLD":
                evs_parts.append(f"HOLD roi={e.get('roi', 0):.1%}")
            else:
                evs_parts.append(
                    f"{e['type']}({e.get('reason', e.get('liquidity', ''))})"
                )
        evs = " | ".join(evs_parts)
        print(f"{row['time']}  opt=${row['opt_mid']:.2f}  stk={row['stock']:.1f}  a={row['alpha']:.3f}  pos={row['position']}  → {evs}")

    print("-" * 72)
    print("Legs:")
    for lg in result["legs"]:
        r = lg.get("realized_roi")
        r_s = f"{r:.1%}" if r is not None else "OPEN"
        print(
            f"  {lg['entry']} @{lg['entry_px']:.2f} → {lg['exit'] or '...'} @{lg['exit_px'] or 0:.2f} "
            f"| max={lg['max_roi']:.1%} realized={r_s} | {lg['reason']} "
            f"[{lg.get('band') or lg['mode']}]"
        )
    s = result["summary"]
    print("-" * 72)
    print(f"Closed legs: {s['closed_legs']} | band_legs_today: {result.get('band_legs_today', 0)}")
    print(f"Strategy total ROI (closed+open): {s['total_strategy_roi']:.1%}")
    if s['closed_legs'] > 1:
        closed_only = sum(
            lg.get("realized_roi") or 0.0
            for lg in result["legs"]
            if lg.get("exit")
        )
        print(f"Cumulative closed-leg ROI (additive): {closed_only:.1%}")
    print(f"Perfect bottom→top ROI:           {s['perfect_bottom_to_top_roi']:.1%}")
    print(f"Capture vs perfect:               {s['capture_vs_perfect_pct']:.1f}%")
    print(f"Buy&hold from 09:45 bar:          {s['buy_hold_first_bar_roi']:.1%}")


def main():
    parser = argparse.ArgumentParser(description="Replay entry/exit gates on synthetic QQQ dislocation path")
    parser.add_argument("--profile", default=os.environ.get("EXEC_PROFILE", "auto_hybrid"))
    parser.add_argument(
        "--multi-band", action="store_true",
        help="Legacy: re-enter after exit; prefer EXEC_PROFILE=multi_band",
    )
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--scenario", choices=("default", "tight_spread", "actionable", "multi_band_roll"), default="default")
    parser.add_argument("--day", default="2025-05-28")
    args = parser.parse_args()

    if args.scenario == "multi_band_roll":
        bars = qqq_dislocation_multi_band_roll_scenario()
        if args.profile == os.environ.get("EXEC_PROFILE", "auto_hybrid") and args.profile == "auto_hybrid":
            args.profile = "multi_band"
    elif args.scenario == "actionable":
        bars = qqq_dislocation_actionable_scenario()
    elif args.scenario == "tight_spread":
        bars = qqq_dislocation_tight_spread_scenario()
    else:
        bars = qqq_dislocation_scenario()
    result = run_replay(
        bars,
        exec_profile=args.profile,
        multi_band=args.multi_band,
        day=args.day,
    )
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print_report(result)


if __name__ == "__main__":
    main()

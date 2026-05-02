#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Trend Hunter strategy core.

This core keeps the existing FCS -> SE -> OMS execution path intact, but changes
the trading idea:

* Regime/trend state decides whether a trade is allowed.
* TFT alpha is used as a radar/confirmation score, not as the sole trigger.
* Exits are fast and based on executable option ROI plus stock/index trend break.
"""

from __future__ import annotations

import logging
from typing import Dict, List

from strategy_config0 import StrategyConfig
from strategy_core_v0 import GATE_REGISTRY, StrategyCoreV0
from strategy_exit_rails import merged_trend_v0_option_stop_floors
from config import option_bucket_tag, option_legacy_tag


logger = logging.getLogger("StrategyCore")


def _f(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


class StrategyCoreTrend(StrategyCoreV0):
    """A trend-regime-first strategy core compatible with OMS StrategyCore API."""

    def __init__(self, config: StrategyConfig = None):
        super().__init__(config if config else StrategyConfig())

    def _trend_quality(self, ctx: dict, direction: int) -> Dict[str, float]:
        st = ctx.get("state")
        prices = getattr(st, "prices", []) if st is not None else []
        try:
            window = int(getattr(self.cfg, "TREND_CORE_WINDOW_MINS", 30) or 30)
            vals = [float(x) for x in list(prices)[-(window + 1):] if float(x) > 0]
        except Exception:
            vals = []

        if len(vals) < 2:
            return {
                "obs": float(len(vals)),
                "net": 0.0,
                "efficiency": 0.0,
                "r2": 0.0,
                "path": 0.0,
            }

        raw_net = (vals[-1] - vals[0]) / vals[0] if vals[0] > 0 else 0.0
        net = raw_net * (1 if direction >= 0 else -1)
        returns = [
            (vals[i] - vals[i - 1]) / vals[i - 1]
            for i in range(1, len(vals))
            if vals[i - 1] > 0
        ]
        path = sum(abs(x) for x in returns)
        efficiency = max(0.0, net) / path if path > 1e-12 else 0.0

        n = len(vals)
        xs = list(range(n))
        mean_x = sum(xs) / n
        mean_y = sum(vals) / n
        sxx = sum((x - mean_x) ** 2 for x in xs)
        syy = sum((y - mean_y) ** 2 for y in vals)
        if sxx > 0 and syy > 0:
            sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, vals))
            r2 = max(0.0, min(1.0, (sxy * sxy) / (sxx * syy)))
        else:
            r2 = 0.0

        return {
            "obs": float(n),
            "net": float(net),
            "efficiency": float(efficiency),
            "r2": float(r2),
            "path": float(path),
        }

    def _index_supports(self, ctx: dict, direction: int) -> bool:
        spy = _f(ctx.get("spy_roc", 0.0))
        qqq = _f(ctx.get("qqq_roc", 0.0))
        idx_trend = int(ctx.get("index_trend", 0) or 0)
        min_idx = float(getattr(self.cfg, "TREND_CORE_MIN_INDEX_ROC", 0.00015) or 0.00015)

        if direction > 0:
            # Longs can work in neutral index state, but at least one index must
            # confirm and the other cannot be meaningfully adverse.
            return idx_trend >= 0 and max(spy, qqq) >= min_idx and min(spy, qqq) > -min_idx * 2.0
        return idx_trend <= 0 and min(spy, qqq) <= -min_idx and max(spy, qqq) < min_idx * 2.0

    def _alpha_supports(self, ctx: dict, direction: int) -> bool:
        alpha = _f(ctx.get("alpha_z", ctx.get("alpha", 0.0)))
        abs_alpha = abs(alpha)
        min_abs = float(getattr(self.cfg, "TREND_CORE_MIN_ALPHA_ABS", 0.35) or 0.35)
        align_min = float(getattr(self.cfg, "TREND_CORE_ALPHA_ALIGN_MIN_ABS", 0.80) or 0.80)
        if abs_alpha < min_abs:
            return False
        # A strong opposite alpha is a warning; weak alpha is only a radar score.
        if abs_alpha >= align_min and alpha * direction < 0:
            return False
        return True

    def _direction_candidate(self, ctx: dict, direction: int) -> Dict[str, float] | None:
        if direction < 0 and not bool(getattr(self.cfg, "TREND_CORE_ALLOW_SHORT", True)):
            return None

        stock_roc = _f(ctx.get("stock_roc", 0.0))
        snap_roc = _f(ctx.get("snap_roc", 0.0))
        macd_hist = _f(ctx.get("macd_hist", 0.0))

        min_stock = float(getattr(self.cfg, "TREND_CORE_MIN_STOCK_ROC", 0.00045) or 0.00045)
        min_snap = float(getattr(self.cfg, "TREND_CORE_MIN_SNAP_ROC", -0.00010) or -0.00010)
        min_macd = float(getattr(self.cfg, "TREND_CORE_MIN_MACD_HIST", 0.010) or 0.010)

        if stock_roc * direction < min_stock:
            return None
        if snap_roc * direction < min_snap:
            return None
        if macd_hist * direction < min_macd:
            return None
        if not self._index_supports(ctx, direction):
            return None
        if not self._alpha_supports(ctx, direction):
            return None

        tq = self._trend_quality(ctx, direction)
        min_obs = int(getattr(self.cfg, "TREND_CORE_MIN_OBS", 16) or 16)
        min_net = float(getattr(self.cfg, "TREND_CORE_MIN_NET", 0.004) or 0.004)
        min_eff = float(getattr(self.cfg, "TREND_CORE_MIN_EFFICIENCY", 0.22) or 0.22)
        min_r2 = float(getattr(self.cfg, "TREND_CORE_MIN_R2", 0.08) or 0.08)
        if tq["obs"] < min_obs or tq["net"] < min_net:
            return None
        if tq["efficiency"] < min_eff and tq["r2"] < min_r2:
            return None

        alpha_abs = abs(_f(ctx.get("alpha_z", ctx.get("alpha", 0.0))))
        trend_score = min(2.0, tq["net"] / max(float(getattr(self.cfg, "TREND_CORE_STRONG_NET", 0.010)), 1e-6))
        momentum_score = (
            max(0.0, stock_roc * direction) * 300.0
            + max(0.0, snap_roc * direction) * 200.0
            + max(0.0, macd_hist * direction) * 4.0
        )
        score = (
            float(getattr(self.cfg, "TREND_CORE_SCORE_TREND_WEIGHT", 1.0)) * trend_score
            + float(getattr(self.cfg, "TREND_CORE_SCORE_MOMENTUM_WEIGHT", 0.65)) * momentum_score
            + float(getattr(self.cfg, "TREND_CORE_SCORE_ALPHA_WEIGHT", 0.35)) * alpha_abs
        )
        return {
            "score": float(score),
            "trend_net": tq["net"],
            "trend_eff": tq["efficiency"],
            "trend_r2": tq["r2"],
            "stock_roc": stock_roc,
            "snap_roc": snap_roc,
            "macd": macd_hist,
        }

    def decide_entry(self, ctx: dict) -> dict:
        self._last_reject_reason = None
        self._last_gate_trace = []

        if not self._check_entry_pre_conditions(ctx):
            if not self._last_reject_reason:
                self._last_reject_reason = "pre_conditions"
            return None

        regime_band = str(ctx.get("regime_band", "calm") or "calm").lower()
        if bool(getattr(self.cfg, "TREND_CORE_BLOCK_VOLATILE_REGIME", True)) and regime_band == "volatile":
            self._trace("T1.regime", "block", "volatile regime")
            self._last_reject_reason = "trend_regime"
            return None
        if regime_band == "mixed" and not bool(getattr(self.cfg, "TREND_CORE_ALLOW_MIXED_REGIME", True)):
            self._trace("T1.regime", "block", "mixed regime disabled")
            self._last_reject_reason = "trend_regime"
            return None
        self._trace("T1.regime", "pass", f"band={regime_band}")

        candidates: List[tuple[int, Dict[str, float]]] = []
        for direction in (1, -1):
            detail = self._direction_candidate(ctx, direction)
            if detail is not None:
                candidates.append((direction, detail))

        if not candidates:
            self._trace(
                "T2.trend_state",
                "block",
                (
                    f"alpha={_f(ctx.get('alpha_z', 0.0)):.2f} "
                    f"stock_roc={_f(ctx.get('stock_roc', 0.0)):.4f} "
                    f"snap={_f(ctx.get('snap_roc', 0.0)):.4f} "
                    f"macd={_f(ctx.get('macd_hist', 0.0)):.4f} "
                    f"idx={ctx.get('index_trend', 0)}"
                ),
            )
            self._last_reject_reason = "trend_state"
            return None

        direction, detail = max(candidates, key=lambda x: x[1]["score"])
        self._trace(
            "T2.trend_state",
            "pass",
            (
                f"dir={direction} score={detail['score']:.2f} "
                f"net={detail['trend_net']:.2%} eff={detail['trend_eff']:.2f} "
                f"r2={detail['trend_r2']:.2f}"
            ),
        )

        if self._cfg_enabled("ENTRY_LIQUIDITY_GUARD_ENABLED", True):
            if not self._check_entry_liquidity_guard(ctx):
                self._last_reject_reason = "liquidity_guard"
                return None
        else:
            self._trace("E14.liquidity_spread", "skip", "ENTRY_LIQUIDITY_GUARD_ENABLED=False")

        alpha_val = _f(ctx.get("alpha_z", ctx.get("alpha", 0.0)))
        return {
            "action": "BUY",
            "dir": direction,
            "tag": option_bucket_tag(direction),
            "legacy_tag": option_legacy_tag(direction),
            "score": float(detail["score"]),
            "reason": (
                f"TREND_HUNTER|dir:{direction}|A:{alpha_val:.2f}|"
                f"R:{detail['stock_roc']:.4f}|M:{detail['macd']:.3f}|"
                f"N:{detail['trend_net']:.2%}"
            ),
        }

    def check_exit(self, ctx: dict) -> dict:
        """
        Trend-first exits, then delegate to StrategyCoreV0.check_exit so the
        same strategy_config0 rails apply: merged option stops via
        strategy_exit_rails.merged_trend_v0_option_stop_floors; profit ladder /
        flash / epic trailing live in strategy_exit_rails.evaluate_profit_rails
        (called from V0 step 8).
        """
        pos = ctx.get("holding")
        if not pos:
            return None
        self._last_gate_trace = []

        sig = self._check_exit_pre_conditions(ctx, pos)
        if sig:
            return sig

        held_mins = _f(ctx.get("held_mins", 0.0))
        curr_price = _f(ctx.get("curr_price", 0.0))
        entry_price = _f(pos.get("entry_price", 0.0))
        roi = (curr_price - entry_price) / entry_price if entry_price > 0 and curr_price > 0 else 0.0
        max_roi = max(_f(pos.get("max_roi", 0.0)), roi)
        direction = 1 if int(pos.get("dir", 1) or 1) >= 0 else -1
        stock_roi = 0.0
        if _f(pos.get("entry_stock", 0.0)) > 0 and _f(ctx.get("curr_stock", 0.0)) > 0:
            stock_roi = (_f(ctx.get("curr_stock")) - _f(pos.get("entry_stock"))) / _f(pos.get("entry_stock"))

        hard_sl, soft_sl = merged_trend_v0_option_stop_floors(self.cfg)

        if roi <= hard_sl:
            self._trace("TX1.option_stop", "block", f"roi={roi:.1%} absolute merged={hard_sl:.1%}")
            return {"action": "SELL", "reason": f"TREND_ABS_STOP:{roi:.1%}"}
        if roi <= soft_sl:
            self._trace("TX1.option_stop", "block", f"roi={roi:.1%} soft merged={soft_sl:.1%}")
            return {"action": "SELL", "reason": f"TREND_STOP:{roi:.1%}"}
        self._trace("TX1.option_stop", "pass", f"roi={roi:.1%}")

        adverse_stock = stock_roi * direction
        stock_stop = float(getattr(self.cfg, "TREND_EXIT_STOCK_ADVERSE_ROC", 0.0020) or 0.0020)
        if adverse_stock <= -stock_stop:
            self._trace("TX2.stock_break", "block", f"stock_roi={stock_roi:.2%}")
            return {"action": "SELL", "reason": f"TREND_STOCK_BREAK:{stock_roi:.2%}"}
        self._trace("TX2.stock_break", "pass", f"stock_roi={stock_roi:.2%}")

        snap = _f(ctx.get("snap_roc", 0.0))
        macd = _f(ctx.get("macd_hist", 0.0))
        snap_break = float(getattr(self.cfg, "TREND_EXIT_SNAP_BREAK", 0.0007) or 0.0007)
        macd_break = float(getattr(self.cfg, "TREND_EXIT_MACD_BREAK", 0.006) or 0.006)
        if snap * direction <= -snap_break:
            self._trace("TX3.momentum_break", "block", f"snap={snap:.4f}")
            return {"action": "SELL", "reason": f"TREND_SNAP_BREAK:{snap:.4f}"}
        if macd * direction <= -macd_break:
            self._trace("TX3.momentum_break", "block", f"macd={macd:.4f}")
            return {"action": "SELL", "reason": f"TREND_MACD_BREAK:{macd:.4f}"}
        self._trace("TX3.momentum_break", "pass", f"snap={snap:.4f} macd={macd:.4f}")

        idx_trend = int(ctx.get("index_trend", 0) or 0)
        min_idx_mins = float(getattr(self.cfg, "TREND_EXIT_INDEX_BREAK_MIN_MINS", 1.0) or 1.0)
        if held_mins >= min_idx_mins and idx_trend * direction < 0:
            self._trace("TX4.index_break", "block", f"idx={idx_trend} held={held_mins:.1f}m")
            return {"action": "SELL", "reason": f"TREND_INDEX_BREAK:{idx_trend}"}
        self._trace("TX4.index_break", "pass", f"idx={idx_trend}")

        no_prog_mins = float(getattr(self.cfg, "TREND_EXIT_NO_PROGRESS_MINS", 3.0) or 3.0)
        no_prog_roi = float(getattr(self.cfg, "TREND_EXIT_NO_PROGRESS_ROI", 0.0) or 0.0)
        if held_mins >= no_prog_mins and roi <= no_prog_roi and max_roi < 0.04:
            self._trace("TX5.no_progress", "block", f"held={held_mins:.1f}m roi={roi:.1%} max={max_roi:.1%}")
            return {"action": "SELL", "reason": f"TREND_NO_PROGRESS:{roi:.1%}"}
        self._trace("TX5.no_progress", "pass", f"held={held_mins:.1f}m roi={roi:.1%}")

        protect_trigger = float(getattr(self.cfg, "TREND_EXIT_PROTECT_TRIGGER", 0.10) or 0.10)
        protect_floor = float(getattr(self.cfg, "TREND_EXIT_PROTECT_FLOOR", 0.03) or 0.03)
        if max_roi >= protect_trigger and roi <= protect_floor:
            self._trace("TX6.profit_protect", "block", f"max={max_roi:.1%} roi={roi:.1%}")
            return {"action": "SELL", "reason": f"TREND_PROTECT:{max_roi:.1%}->{roi:.1%}"}

        trail_trigger = float(getattr(self.cfg, "TREND_EXIT_TRAIL_TRIGGER", 0.22) or 0.22)
        trail_keep = float(getattr(self.cfg, "TREND_EXIT_TRAIL_KEEP", 0.55) or 0.55)
        if max_roi >= trail_trigger and roi <= max_roi * trail_keep:
            self._trace("TX6.profit_protect", "block", f"trail max={max_roi:.1%} roi={roi:.1%}")
            return {"action": "SELL", "reason": f"TREND_TRAIL:{max_roi:.1%}->{roi:.1%}"}
        self._trace("TX6.profit_protect", "pass", f"max={max_roi:.1%} roi={roi:.1%}")

        time_stop_mins = float(getattr(self.cfg, "TREND_EXIT_TIME_STOP_MINS", 15.0) or 15.0)
        time_stop_roi = float(getattr(self.cfg, "TREND_EXIT_TIME_STOP_ROI", 0.05) or 0.05)
        if held_mins >= time_stop_mins and roi < time_stop_roi:
            self._trace("TX7.time_stop", "block", f"held={held_mins:.1f}m roi={roi:.1%}")
            return {"action": "SELL", "reason": f"TREND_TIME_STOP:{held_mins:.0f}m"}
        max_hold = float(getattr(self.cfg, "TREND_EXIT_MAX_HOLD_MINS", 30.0) or 30.0)
        if held_mins >= max_hold:
            self._trace("TX7.time_stop", "block", f"held={held_mins:.1f}m max")
            return {"action": "SELL", "reason": f"TREND_MAX_HOLD:{held_mins:.0f}m"}
        self._trace("TX7.time_stop", "pass", f"held={held_mins:.1f}m")

        return super().check_exit(ctx)
        #return None



__all__ = ["StrategyCoreTrend", "StrategyConfig", "GATE_REGISTRY"]

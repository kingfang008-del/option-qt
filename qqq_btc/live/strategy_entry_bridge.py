#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
StrategyCore.decide_entry → qqq_btc choose_entry 桥接。

QQQ_BTC_LIVE 模式下保留 V0 前置门控(E1–E6b)与流动性校验,
核心 alpha/spread/q10/会话窗语义与 strict replay 共用 entry_decision。
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from qqq_btc.common.entry_decision import choose_entry
from qqq_btc.common.time_features import session_minute
from qqq_btc.live.session_governor import get_session_governor
from qqq_btc.qqq import config as qcfg


def _session_bar_from_ctx(ctx: dict) -> int:
    t = ctx.get("time")
    if t is None:
        return 0
    try:
        return int(session_minute(pd.Series([pd.Timestamp(t)])).iloc[0])
    except Exception:
        return 0


def _spread_pct_from_ctx(ctx: dict) -> float:
    bid = float(ctx.get("bid", 0.0) or 0.0)
    ask = float(ctx.get("ask", 0.0) or 0.0)
    mid = float(ctx.get("curr_price", 0.0) or 0.0)
    if mid > 0.01 and ask >= bid > 0:
        return (ask - bid) / mid
    return 0.0


def _f_ctx(ctx: dict, key: str) -> Optional[float]:
    raw = ctx.get(key)
    if raw is None:
        return None
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def _edge_q10_from_ctx(ctx: dict) -> Optional[float]:
    raw = ctx.get("net_edge_q10")
    if raw is None:
        return None
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def decide_entry_via_replay(self, ctx: dict) -> Optional[dict]:
    """用 choose_entry 替换 V0 simple/legacy 入场路径；前置/流动性仍走 StrategyCoreV0。"""
    self._last_reject_reason = None
    self._last_gate_trace = []

    if getattr(self.cfg, "BIDIRECTIONAL_ENABLED", True):
        from strategy.regime import enrich_ctx_regime

        enrich_ctx_regime(ctx, self.cfg)

    if not self._check_entry_pre_conditions(ctx):
        if not self._last_reject_reason:
            self._last_reject_reason = "pre_conditions"
        return None

    replay_cfg = qcfg.REPLAY
    session_bar = _session_bar_from_ctx(ctx)
    sym = str(ctx.get("symbol", "QQQ") or "QQQ")
    curr_ts = float(ctx.get("curr_ts", 0.0) or 0.0)
    edge = float(ctx.get("net_edge_raw", ctx.get("alpha_z", 0.0)) or 0.0)
    call_edge = float(ctx.get("call_edge", edge) or edge)
    put_edge = float(ctx.get("put_edge", 0.0) or 0.0)
    spread_pct = _spread_pct_from_ctx(ctx)
    edge_q10 = _edge_q10_from_ctx(ctx)

    dual_mode = not bool(replay_cfg.long_only)
    has_put = dual_mode

    spot_day_ret = _f_ctx(ctx, "spot_day_ret")
    if spot_day_ret is None:
        spot_day_ret = _f_ctx(ctx, "qqq_day_roc")
    spot_ret_5bar = _f_ctx(ctx, "spot_ret_5bar")
    if spot_ret_5bar is None:
        spot_ret_5bar = _f_ctx(ctx, "stock_roc")
    trend_ret_30m = _f_ctx(ctx, "trend_fit_ret_30m")
    trend_r2_30m = _f_ctx(ctx, "trend_fit_r2_30m")
    vix_rev = _f_ctx(ctx, "vix_reversal_count_30m")
    spot_range_30m = _f_ctx(ctx, "spot_range_30m")

    gov = get_session_governor(replay_cfg)
    if curr_ts > 0:
        gov.maybe_reset_day(sym, curr_ts)
    blocked, block_reason = gov.blocked_for_entry(
        sym,
        curr_ts=curr_ts if curr_ts > 0 else 0.0,
        cooldown_until=float(ctx.get("cooldown_until", 0.0) or 0.0),
    )
    if blocked:
        self._trace("E9.qqq_btc_governor", "block", block_reason)
        self._last_reject_reason = block_reason
        return None

    if not replay_cfg.session_allows_entry(session_bar):
        self._trace(
            "E9.qqq_btc_session",
            "block",
            f"session_bar={session_bar} outside [{replay_cfg.session_entry_start_bar},{replay_cfg.session_entry_end_bar}]",
        )
        self._last_reject_reason = "session_window"
        return None
    self._trace("E9.qqq_btc_session", "pass", f"session_bar={session_bar}")

    gov.record_edges(
        sym,
        session_bar=session_bar,
        call_edge=call_edge,
        put_edge=put_edge,
        dual_mode=dual_mode,
        trend_r2_30m=trend_r2_30m,
        spot_day_ret=spot_day_ret,
        vix_reversal_count_30m=vix_rev,
        spot_range_30m=spot_range_30m,
        trend_ret_30m=trend_ret_30m,
    )
    dyn_th, put_dyn_th = gov.dynamic_thresholds(sym)
    straddles_today = gov.straddles_today_for(sym)

    decision = choose_entry(
        replay_cfg,
        session_bar=session_bar,
        edge=edge,
        call_edge=call_edge,
        put_edge=put_edge,
        edge_q10=edge_q10,
        spread_pct=spread_pct,
        dual_mode=dual_mode,
        has_put=has_put,
        straddle_enabled=dual_mode,
        straddles_today=straddles_today,
        default_leg="CALL",
        dynamic_threshold=dyn_th,
        put_dynamic_threshold=put_dyn_th,
        put_gate=_f_ctx(ctx, "vix_level"),
        open30_max_ret=_f_ctx(ctx, "open30_max_ret"),
        open30_peak_dd=_f_ctx(ctx, "open30_peak_dd"),
        spot_ret_5bar=spot_ret_5bar,
        trend_ret_30m=trend_ret_30m,
        trend_r2_30m=trend_r2_30m,
        vix_reversal_count_30m=vix_rev,
        spot_day_ret=spot_day_ret,
        spot_range_30m=spot_range_30m,
    )

    if decision is None:
        th = replay_cfg.threshold_at(session_bar)
        dyn_note = ""
        if dyn_th is not None:
            dyn_note = f", dyn={dyn_th:.4f}"
        if put_dyn_th is not None:
            dyn_note += f", put_dyn={put_dyn_th:.4f}"
        q10_note = ""
        if edge_q10 is not None and edge > 0:
            q10_note = f", q10={edge_q10:.4f}"
        self._trace(
            "E9.qqq_btc_entry",
            "block",
            f"edge={edge:.4f} th={th:.4f}{dyn_note} spread={spread_pct:.4f}{q10_note}",
        )
        self._last_reject_reason = "qqq_btc_entry"
        return None

    self._trace(
        "E9.qqq_btc_entry",
        "pass",
        f"leg={decision.leg} edge={decision.edge:.4f} th={decision.threshold:.4f}",
    )

    direction = 1 if decision.leg == "CALL" else -1
    try:
        from config import option_bucket_tag, option_legacy_tag
    except ImportError:
        option_bucket_tag = lambda d, m=None: "CALL" if d >= 0 else "PUT"  # noqa: E731
        option_legacy_tag = option_bucket_tag

    sig: Dict[str, Any] = {
        "action": "BUY",
        "dir": direction,
        "tag": option_bucket_tag(direction),
        "legacy_tag": option_legacy_tag(direction),
        "score": abs(decision.edge),
        "reason": f"QQQ_BTC_ENTRY|{decision.leg}|E:{decision.edge:.3f}(Th:{decision.threshold:.3f})",
    }

    if getattr(self, "_cfg_enabled", None) and self._cfg_enabled("ENTRY_LIQUIDITY_GUARD_ENABLED", True):
        if not self._check_entry_liquidity_guard(
            ctx,
            spread_threshold_override=replay_cfg.max_spread_pct,
        ):
            self._last_reject_reason = "liquidity_guard"
            return None
    return sig


def apply_strategy_entry_patch(StrategyCore) -> None:
    """Monkey-patch StrategyCore.decide_entry → choose_entry(qqq_btc 口径)。"""

    _orig = StrategyCore.decide_entry

    def _patched_decide_entry(self, ctx: dict):
        sig = decide_entry_via_replay(self, ctx)
        if sig is not None:
            return sig
        return _orig(self, ctx)

    def _replay_only(self, ctx: dict):
        return decide_entry_via_replay(self, ctx)

    import os

    if os.environ.get("QQQ_BTC_ENTRY_FALLBACK", "").strip().lower() in ("1", "true", "yes"):
        StrategyCore.decide_entry = _patched_decide_entry
    else:
        StrategyCore.decide_entry = _replay_only

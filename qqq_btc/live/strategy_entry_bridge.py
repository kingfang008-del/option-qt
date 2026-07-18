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
from qqq_btc.live.live_clock import live_session_bar
from qqq_btc.live.session_governor import get_session_governor
from qqq_btc.live.signal_audit_writer import record_entry_signal_audit
from qqq_btc.qqq import config as qcfg


def _session_bar_from_ctx(ctx: dict) -> int:
    t = ctx.get("time")
    if t is None:
        return 0
    try:
        return live_session_bar(t)
    except Exception:
        return 0


def _spread_pct(bid: float, ask: float, mid: float = 0.0) -> float:
    bid = float(bid or 0.0)
    ask = float(ask or 0.0)
    mid = float(mid or 0.0)
    if mid <= 0.01 and bid > 0 and ask >= bid:
        mid = 0.5 * (bid + ask)
    if mid > 0.01 and ask >= bid > 0:
        return (ask - bid) / mid
    return 0.0


def _spread_pct_from_ctx(ctx: dict) -> float:
    """兼容旧路径:ctx.bid/ask 可能是 alpha 符号选腿后的盘口(空仓时常为 CALL)。"""
    return _spread_pct(
        float(ctx.get("bid", 0.0) or 0.0),
        float(ctx.get("ask", 0.0) or 0.0),
        float(ctx.get("curr_price", 0.0) or 0.0),
    )


def _leg_spreads_from_ctx(ctx: dict) -> tuple[float, Optional[float], Optional[float]]:
    """
    与 replay_session._choose_entry 对齐:
      spread_pct = CALL 盘口; put_spread_pct = PUT 盘口。
    空仓时 OMS 常按 alpha>0 把 ctx.bid/ask 填成 CALL,若只传这一路,
    PUT 会用过紧的 CALL spread 过门(July7 第二笔主因)。
    """
    call_bid = _f_ctx(ctx, "call_bid")
    call_ask = _f_ctx(ctx, "call_ask")
    put_bid = _f_ctx(ctx, "put_bid")
    put_ask = _f_ctx(ctx, "put_ask")

    call_sp = _f_ctx(ctx, "call_spread_pct")
    if call_sp is None and call_bid is not None and call_ask is not None:
        call_sp = _spread_pct(call_bid, call_ask)
    if call_sp is None:
        call_sp = _spread_pct_from_ctx(ctx)

    put_sp = _f_ctx(ctx, "put_spread_pct")
    if put_sp is None and put_bid is not None and put_ask is not None:
        put_sp = _spread_pct(put_bid, put_ask)

    straddle_sp = None
    if put_sp is not None:
        straddle_sp = max(float(call_sp), float(put_sp))
    return float(call_sp), put_sp, straddle_sp


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


def record_session_edges_from_ctx(ctx: dict, replay_cfg: Any = None) -> None:
    """空仓分钟喂入滚动分位缓冲。

    与 offline replay_session CLOSE 一致：冷却 / V0 入场窗 / max_trades
    只挡开仓，不挡 edge 观测。持仓分钟不喂。
    """
    if not ctx.get("is_ready", False):
        return

    from qqq_btc.live.session_governor import resolve_replay_cfg

    cfg = resolve_replay_cfg(replay_cfg)
    session_bar = _session_bar_from_ctx(ctx)
    sym = str(ctx.get("symbol", "QQQ") or "QQQ")
    curr_ts = float(ctx.get("curr_ts", 0.0) or 0.0)
    gov = get_session_governor(cfg)
    if curr_ts > 0:
        gov.maybe_reset_day(sym, curr_ts)
    # 状态识别必须观察持仓分钟；edge 分位缓冲仍只记录空仓分钟。
    gov.note_vixy_open_shock_regime(
        sym,
        session_bar=session_bar,
        open30_ret=_f_ctx(ctx, "open30_ret"),
        open30_peak_dd=_f_ctx(ctx, "open30_peak_dd"),
        trend_r2_30m=_f_ctx(ctx, "trend_fit_r2_30m"),
    )
    if int(ctx.get("position", 0) or 0) != 0:
        return
    if not cfg.session_allows_entry(session_bar):
        return

    edge = float(ctx.get("net_edge_raw", ctx.get("alpha_z", 0.0)) or 0.0)
    call_edge = float(ctx.get("call_edge", edge) or edge)
    put_edge = float(ctx.get("put_edge", 0.0) or 0.0)
    dual_mode = not bool(cfg.long_only)

    spot_day_ret = _f_ctx(ctx, "spot_day_ret")
    if spot_day_ret is None:
        spot_day_ret = _f_ctx(ctx, "qqq_day_roc")
    trend_ret_30m = _f_ctx(ctx, "trend_fit_ret_30m")
    trend_r2_30m = _f_ctx(ctx, "trend_fit_r2_30m")
    vix_rev = _f_ctx(ctx, "vix_reversal_count_30m")
    spot_range_30m = _f_ctx(ctx, "spot_range_30m")
    day_range_pos = _f_ctx(ctx, "day_range_pos")
    bb_width = _f_ctx(ctx, "bb_width")

    spot_px = _f_ctx(ctx, "spot_close")
    if spot_px is None:
        spot_px = _f_ctx(ctx, "stock_price")
    if spot_px is None:
        spot_px = _f_ctx(ctx, "close")
    gov.record_bounce_inputs(
        sym,
        spot_close=spot_px,
        vwap_log_return=_f_ctx(ctx, "vwap_log_return"),
        ts=curr_ts,
    )
    gov.record_cross_asset_inputs(
        sym,
        spot_close=spot_px,
        vix_proxy_close=_f_ctx(ctx, "vix_proxy_close"),
        ts=curr_ts,
    )
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
        day_range_pos=day_range_pos,
        bb_width=bb_width,
        curr_ts=curr_ts,
    )


def decide_entry_via_replay(self, ctx: dict) -> Optional[dict]:
    """用 choose_entry 替换 V0 simple/legacy 入场路径；前置/流动性仍走 StrategyCoreV0。"""
    import os
    from datetime import date as _date

    self._last_reject_reason = None
    self._last_gate_trace = []

    if getattr(self.cfg, "BIDIRECTIONAL_ENABLED", True):
        from strategy.regime import enrich_ctx_regime

        enrich_ctx_regime(ctx, self.cfg)

    # Live 默认 REPLAY(含 entry_delay);QQQ_BTC_USE_LIVE_REPLAY=1 时用 immediate。
    # 阈值 override 与 fill 侧共用 resolve_replay_cfg，避免 governor 双实例。
    from qqq_btc.live.live_clock import live_end_label_ts
    from qqq_btc.live.session_governor import resolve_replay_cfg

    replay_cfg = resolve_replay_cfg()
    session_bar = _session_bar_from_ctx(ctx)
    sym = str(ctx.get("symbol", "QQQ") or "QQQ")
    curr_ts = float(ctx.get("curr_ts", 0.0) or 0.0)

    # 必须在 V0 pre_conditions / cooldown 之前喂缓冲，否则冷却期空仓 bar
    # 会被静默丢掉，周终 edge_buf 会系统性低于 offline（例如 296 vs 471）。
    record_session_edges_from_ctx(ctx, replay_cfg)

    # V0 START_MINUTE/NO_ENTRY 看 ctx["time"] 的时钟分钟；FCS 是 start-label。
    # 若不先换成 end-label，09:44 标签会被 START_MINUTE=45 挡掉，整周缺 sb15，
    # 导致 7/6 贵买、7/8 撞上 put_trend 晚进。
    ctx_pre = dict(ctx)
    if ctx_pre.get("time") is not None:
        try:
            ctx_pre["time"] = live_end_label_ts(ctx_pre["time"]).to_pydatetime()
        except Exception:
            pass

    if not self._check_entry_pre_conditions(ctx_pre):
        if not self._last_reject_reason:
            self._last_reject_reason = "pre_conditions"
        return None

    # 连续流预热日禁止新开仓:QQQ_BTC_TRADE_FROM_DATE=YYYYMMDD|YYYY-MM-DD
    trade_from = os.environ.get("QQQ_BTC_TRADE_FROM_DATE", "").strip()
    if trade_from and curr_ts > 0:
        try:
            if len(trade_from) == 8 and trade_from.isdigit():
                cutoff = _date(int(trade_from[:4]), int(trade_from[4:6]), int(trade_from[6:8]))
            else:
                cutoff = pd.Timestamp(trade_from).date()
            day = pd.Timestamp(curr_ts, unit="s", tz="UTC").tz_convert("America/New_York").date()
            if day < cutoff:
                self._trace("E9.qqq_btc_trade_from", "block", f"day={day} < TRADE_FROM={cutoff}")
                self._last_reject_reason = f"before_trade_from_date:{cutoff}"
                return None
        except Exception:
            pass
    edge = float(ctx.get("net_edge_raw", ctx.get("alpha_z", 0.0)) or 0.0)
    call_edge = float(ctx.get("call_edge", edge) or edge)
    put_edge = float(ctx.get("put_edge", 0.0) or 0.0)
    spread_pct, put_spread_pct, straddle_spread_pct = _leg_spreads_from_ctx(ctx)
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
    day_range_pos = _f_ctx(ctx, "day_range_pos")
    bb_width = _f_ctx(ctx, "bb_width")

    gov = get_session_governor(replay_cfg)
    if curr_ts > 0:
        gov.maybe_reset_day(sym, curr_ts)
    # 日切后生效配置可能已套 OPEN_DEFENSE/CHOP；后续门控必须用 gov.replay_cfg。
    replay_cfg = gov.replay_cfg
    # bounce/cross-asset 输入已在 record_session_edges_from_ctx 中于 V0 门控前记录，
    # 此处只读取，避免被 pre_conditions 拦掉的分钟造成历史断裂。
    spot_px = _f_ctx(ctx, "spot_close")
    if spot_px is None:
        spot_px = _f_ctx(ctx, "stock_price")
    if spot_px is None:
        spot_px = _f_ctx(ctx, "close")
    gov.record_cross_asset_inputs(
        sym,
        spot_close=spot_px,
        vix_proxy_close=_f_ctx(ctx, "vix_proxy_close"),
        ts=float(curr_ts) if curr_ts and curr_ts > 0 else 0.0,
    )
    rolling_spot_15, rolling_vix_15 = gov.cross_asset_returns(sym, bars=15)
    # FCS history 可直接给出严格 15-bar 收益；governor 为重启/接线兼容兜底。
    spot_ret_15bar = _f_ctx(ctx, "spot_ret_15bar")
    if spot_ret_15bar is None:
        spot_ret_15bar = rolling_spot_15
    vix_ret_15bar = _f_ctx(ctx, "vix_ret_15bar")
    if vix_ret_15bar is None:
        vix_ret_15bar = rolling_vix_15

    blocked, block_reason = gov.blocked_for_entry(
        sym,
        curr_ts=curr_ts if curr_ts > 0 else 0.0,
        cooldown_until=float(ctx.get("cooldown_until", 0.0) or 0.0),
    )
    if blocked:
        self._trace("E9.qqq_btc_governor", "block", block_reason)
        self._last_reject_reason = block_reason
        record_entry_signal_audit(
            ctx=ctx,
            block_reason=block_reason,
            session_bar=session_bar,
            mode=getattr(self, "mode", ""),
        )
        return None

    gated, gate_reason = gov.cross_day_gates(
        sym,
        session_bar=session_bar,
        edge_q10=edge_q10,
    )
    if gated:
        self._trace("E9.qqq_btc_cross_day", "block", gate_reason)
        self._last_reject_reason = gate_reason
        record_entry_signal_audit(
            ctx=ctx,
            block_reason=gate_reason,
            session_bar=session_bar,
            mode=getattr(self, "mode", ""),
        )
        return None

    if not replay_cfg.session_allows_entry(session_bar):
        self._trace(
            "E9.qqq_btc_session",
            "block",
            f"session_bar={session_bar} outside [{replay_cfg.session_entry_start_bar},{replay_cfg.session_entry_end_bar}]",
        )
        self._last_reject_reason = "session_window"
        record_entry_signal_audit(
            ctx=ctx,
            block_reason="session_window",
            session_bar=session_bar,
            mode=getattr(self, "mode", ""),
        )
        return None
    self._trace("E9.qqq_btc_session", "pass", f"session_bar={session_bar}")

    dyn_th, put_dyn_th = gov.dynamic_thresholds(sym)
    straddles_today = gov.straddles_today_for(sym)
    gov.note_put_structure_veto(
        sym,
        session_bar=session_bar,
        put_edge=put_edge,
        open30_max_ret=_f_ctx(ctx, "open30_max_ret"),
    )
    gov.note_vixy_open_shock_regime(
        sym,
        session_bar=session_bar,
        open30_ret=_f_ctx(ctx, "open30_ret"),
        open30_peak_dd=_f_ctx(ctx, "open30_peak_dd"),
        trend_r2_30m=trend_r2_30m,
    )
    c_mult, p_mult = gov.entry_threshold_mults(sym)

    decision = choose_entry(
        replay_cfg,
        session_bar=session_bar,
        edge=edge,
        call_edge=call_edge,
        put_edge=put_edge,
        edge_q10=edge_q10,
        spread_pct=spread_pct,
        put_spread_pct=put_spread_pct,
        straddle_spread_pct=straddle_spread_pct,
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
        spot_ret_15bar=spot_ret_15bar,
        vix_ret_15bar=vix_ret_15bar,
        trend_ret_30m=trend_ret_30m,
        trend_r2_30m=trend_r2_30m,
        vix_reversal_count_30m=vix_rev,
        spot_day_ret=spot_day_ret,
        spot_range_30m=spot_range_30m,
        day_range_pos=day_range_pos,
        bb_width=bb_width,
        best_side_put_prob=_f_ctx(ctx, "best_side_put_prob"),
        best_side_none_prob=_f_ctx(ctx, "best_side_none_prob"),
        best_side_call_prob=_f_ctx(ctx, "best_side_call_prob"),
        spot_down_prob=_f_ctx(ctx, "spot_down_prob"),
        spot_flat_prob=_f_ctx(ctx, "spot_flat_prob"),
        spot_up_prob=_f_ctx(ctx, "spot_up_prob"),
        call_threshold_mult=c_mult,
        put_threshold_mult=p_mult,
        put_structure_veto_until_bar=gov.put_structure_veto_until_bar(sym),
        vixy_open_shock_regime_active=gov.vixy_open_shock_regime_active(sym),
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
        put_sp_note = ""
        if put_spread_pct is not None:
            put_sp_note = f", put_sp={put_spread_pct:.4f}"
        self._trace(
            "E9.qqq_btc_entry",
            "block",
            f"edge={edge:.4f} th={th:.4f}{dyn_note} spread={spread_pct:.4f}{put_sp_note}{q10_note}",
        )
        self._last_reject_reason = "qqq_btc_entry"
        record_entry_signal_audit(
            ctx=ctx,
            block_reason="qqq_btc_entry",
            session_bar=session_bar,
            dyn_threshold=dyn_th,
            put_dyn_threshold=put_dyn_th,
            mode=getattr(self, "mode", ""),
        )
        return None

    gated_leg, gate_leg_reason = gov.cross_day_gates(
        sym,
        session_bar=session_bar,
        edge_q10=edge_q10,
        leg=decision.leg,
    )
    if gated_leg:
        self._trace("E9.qqq_btc_cross_day", "block", gate_leg_reason)
        self._last_reject_reason = gate_leg_reason
        record_entry_signal_audit(
            ctx=ctx,
            block_reason=gate_leg_reason,
            session_bar=session_bar,
            dyn_threshold=dyn_th,
            put_dyn_threshold=put_dyn_th,
            mode=getattr(self, "mode", ""),
        )
        return None

    if curr_ts > 0 and gov.leg_blocked_for_entry(
        sym, leg=decision.leg, curr_ts=curr_ts
    ):
        block_reason = f"tick_stop_leg_lock:{decision.leg}"
        self._trace("E9.qqq_btc_governor", "block", block_reason)
        self._last_reject_reason = block_reason
        record_entry_signal_audit(
            ctx=ctx,
            block_reason=block_reason,
            session_bar=session_bar,
            dyn_threshold=dyn_th,
            put_dyn_threshold=put_dyn_th,
            mode=getattr(self, "mode", ""),
        )
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
        "meta": {
            "position_frac": gov.effective_position_frac(sym),
            "position_size_mult": gov.position_size_mult(sym),
            "all_leg_defense": gov.all_leg_defense_active(sym),
            "vx_curve_slope": getattr(gov._state(sym), "vx_curve_slope", None),
            "active_profile": getattr(gov._state(sym), "active_profile", None),
        },
    }

    # 流动性/审计盘口切到选定腿,避免 PUT 仍用 CALL bid/ask
    liq_ctx = ctx
    if decision.leg == "PUT":
        pb, pa = _f_ctx(ctx, "put_bid"), _f_ctx(ctx, "put_ask")
        if pb is not None and pa is not None and pb > 0 and pa >= pb:
            liq_ctx = dict(ctx)
            liq_ctx["bid"] = pb
            liq_ctx["ask"] = pa
            liq_ctx["curr_price"] = 0.5 * (pb + pa)
            if put_spread_pct is not None:
                liq_ctx["put_spread_pct"] = put_spread_pct
    elif decision.leg == "CALL":
        cb, ca = _f_ctx(ctx, "call_bid"), _f_ctx(ctx, "call_ask")
        if cb is not None and ca is not None and cb > 0 and ca >= cb:
            liq_ctx = dict(ctx)
            liq_ctx["bid"] = cb
            liq_ctx["ask"] = ca
            liq_ctx["curr_price"] = 0.5 * (cb + ca)

    if getattr(self, "_cfg_enabled", None) and self._cfg_enabled("ENTRY_LIQUIDITY_GUARD_ENABLED", True):
        if not self._check_entry_liquidity_guard(
            liq_ctx,
            spread_threshold_override=replay_cfg.max_spread_pct,
        ):
            self._last_reject_reason = "liquidity_guard"
            record_entry_signal_audit(
                ctx=liq_ctx,
                decision=decision,
                block_reason="liquidity_guard",
                session_bar=session_bar,
                dyn_threshold=dyn_th,
                put_dyn_threshold=put_dyn_th,
                mode=getattr(self, "mode", ""),
            )
            return None
    record_entry_signal_audit(
        ctx=liq_ctx,
        decision=decision,
        session_bar=session_bar,
        dyn_threshold=dyn_th,
        put_dyn_threshold=put_dyn_th,
        mode=getattr(self, "mode", ""),
    )
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

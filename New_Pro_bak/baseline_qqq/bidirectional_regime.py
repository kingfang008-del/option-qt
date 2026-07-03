"""
QQQ bidirectional trading: day regime, dual-edge inference, Phase 4 profile routing.

Shared by signal_engine (Phase 3), strategy_core (Phase 2), exec_profile (Phase 4),
and tools/bidirectional_phase1_audit.py (Phase 1).
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Mapping, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn.functional as F
except Exception:  # pragma: no cover
    torch = None
    F = None


class DayType(str, Enum):
    TREND_UP = "trend_up"
    TREND_DOWN = "trend_down"
    CHOP = "chop"
    DISLOCATION = "dislocation"
    UNKNOWN = "unknown"


def _f(value: Any, default: float = 0.0) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return float(default)


def resolve_day_type(
    ctx: Mapping[str, Any],
    *,
    day_roc_threshold: float = 0.0035,
    dislocation_snap_min: float = 0.0008,
    dislocation_day_roc_max: float = 0.0020,
) -> DayType:
    """
    Rule-based session regime (ARCHITECTURE_BIDIRECTIONAL §4.1).
    Uses QQQ day ROC + 5m ROC; optional dislocation when snap diverges from day move.
    """
    day_roc = _f(ctx.get("qqq_day_roc", ctx.get("day_roc", 0.0)))
    roc_5m = _f(ctx.get("stock_roc", ctx.get("roc_5m", ctx.get("qqq_roc", 0.0))))
    snap = _f(ctx.get("snap_roc", 0.0))
    alpha = _f(ctx.get("alpha", ctx.get("alpha_z", 0.0)))

    same_dir_up = day_roc >= day_roc_threshold and roc_5m >= day_roc_threshold * 0.5
    same_dir_down = day_roc <= -day_roc_threshold and roc_5m <= -day_roc_threshold * 0.5

    if abs(day_roc) < dislocation_day_roc_max:
        if alpha > 0 and snap >= dislocation_snap_min and roc_5m < 0:
            return DayType.DISLOCATION
        if alpha < 0 and snap <= -dislocation_snap_min and roc_5m > 0:
            return DayType.DISLOCATION

    if same_dir_up:
        return DayType.TREND_UP
    if same_dir_down:
        return DayType.TREND_DOWN
    if abs(day_roc) < day_roc_threshold:
        return DayType.CHOP
    return DayType.CHOP


def resolve_micro_regime(ctx: Mapping[str, Any], cfg: Any = None) -> str:
    """Fast-channel tradable vs blocked (spread / IV)."""
    spread_max = 0.12
    iv_max = 0.50
    if cfg is not None:
        spread_max = _f(getattr(cfg, "FAST_GATE_SPREAD_MAX", spread_max), spread_max)
        iv_max = _f(getattr(cfg, "FAST_GATE_IV_MOMENTUM_ABS_MAX", iv_max), iv_max)

    spread = _f(ctx.get("options_vw_spread", 0.0))
    iv_mom = abs(_f(ctx.get("options_iv_momentum", 0.0)))
    if spread > spread_max > 0 or iv_mom > iv_max > 0:
        return "blocked"
    return "tradable"


def dual_edges_from_model_out(model_out: Mapping[str, Any], *, batch_index: int = 0):
    """
    Phase 3: extract non-negative call_edge / put_edge tensors or scalars.

    Priority: explicit heads → net_edge + logits_dir → signed net_edge clamp.
    """
    if torch is None:
        raise RuntimeError("torch required for dual_edges_from_model_out")

    if "call_net_edge" in model_out and "put_net_edge" in model_out:
        call_t = model_out["call_net_edge"].detach()
        put_t = model_out["put_net_edge"].detach()
        if call_t.dim() > 1:
            call_t = call_t.squeeze(-1)
        if put_t.dim() > 1:
            put_t = put_t.squeeze(-1)
        return F.relu(call_t), F.relu(put_t)

    net = model_out.get("net_edge")
    if net is None:
        net = model_out.get("rank_score")
    if net is None:
        z = torch.zeros(1)
        return z, z

    net_t = net.detach()
    if net_t.dim() > 1:
        net_t = net_t.squeeze(-1)

    logits = model_out.get("logits_dir")
    if logits is not None:
        probs = F.softmax(logits.detach(), dim=-1)
        p_put = probs[:, 0]
        p_call = probs[:, 2] if probs.shape[-1] > 2 else probs[:, -1]
        mag = net_t.abs()
        call_edge = torch.where(net_t > 0, net_t, mag * p_call)
        put_edge = torch.where(net_t < 0, -net_t, mag * p_put)
        return F.relu(call_edge), F.relu(put_edge)

    call_edge = torch.clamp(net_t, min=0.0)
    put_edge = torch.clamp(-net_t, min=0.0)
    return call_edge, put_edge


def pick_tradable_side(
    call_edge: float,
    put_edge: float,
    *,
    threshold: float = 0.015,
) -> Tuple[int, float, str]:
    """
    Return (direction, signed_alpha, reason).
    direction: +1 call, -1 put, 0 flat.
    """
    c = _f(call_edge)
    p = _f(put_edge)
    th = _f(threshold, 0.015)
    if c >= p and c >= th:
        return 1, c, f"argmax_call|c={c:.4f}|p={p:.4f}"
    if p > c and p >= th:
        return -1, -p, f"argmax_put|c={c:.4f}|p={p:.4f}"
    return 0, 0.0, f"below_threshold|c={c:.4f}|p={p:.4f}|th={th:.4f}"


def dual_edges_numpy(model_out: Mapping[str, Any], index: int = 0) -> Tuple[float, float]:
    """Numpy-friendly wrapper for inference loop."""
    if torch is None:
        net = _f(model_out.get("net_edge", 0.0))
        return max(0.0, net), max(0.0, -net)
    call_t, put_t = dual_edges_from_model_out(model_out)
    i = min(max(0, index), int(call_t.shape[0]) - 1) if call_t.numel() else 0
    return float(call_t[i].cpu().item()), float(put_t[i].cpu().item())


def enrich_ctx_regime(ctx: dict, cfg: Any = None) -> dict:
    """Attach day_type / micro_regime to strategy ctx (in-place)."""
    day = resolve_day_type(ctx)
    ctx["day_type"] = day.value
    ctx["micro_regime"] = resolve_micro_regime(ctx, cfg)
    ctx["qqq_day_roc"] = _f(ctx.get("qqq_day_roc", ctx.get("day_roc", 0.0)))
    return ctx


def regime_preferred_side(day_type: str) -> int:
    """+1 call bias, -1 put bias, 0 neutral."""
    dt = str(day_type or "").lower()
    if dt == DayType.TREND_UP.value:
        return 1
    if dt == DayType.TREND_DOWN.value:
        return -1
    return 0


def oracle_side_from_returns(call_ret: float, put_ret: float, *, min_edge: float = 0.0) -> int:
    """Phase 1 audit: hindsight best side at this minute."""
    c = _f(call_ret)
    p = _f(put_ret)
    if c >= p and c > min_edge:
        return 1
    if p > c and p > min_edge:
        return -1
    return 0

"""
Strategy context contract for V0 core ↔ OMS parity with S4 replay.

S4 builds per-symbol rows that eventually flow into ExecutionEngineV8._build_strategy_ctx
as ``item`` + ``opt_data`` + ``frame``. This module lists the canonical ctx keys that
V0 reads (directly or via defaults) so replay and live stay aligned.

See: production/baseline/S4_LIVE_PARITY_ROADMAP.md
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, MutableMapping, Tuple

# Keys produced by ExecutionEngineV8._build_strategy_ctx today; V0 strategy should
# not rely on undocumented extras for core decisions.
STRATEGY_CTX_V0_KEYS: Tuple[str, ...] = (
    "symbol",
    "time",
    "curr_ts",
    "price",
    "alpha",
    "alpha_z",
    "cs_alpha_z",
    "vol_z",
    "stock_roc",
    "event_prob",
    "macd_hist",
    "macd_hist_slope",
    "spy_roc",
    "qqq_roc",
    "index_trend",
    "position",
    "cooldown_until",
    "is_ready",
    "is_banned",
    "held_mins",
    "stock_iv",
    "holding",
    "curr_price",
    "curr_stock",
    "bid",
    "ask",
    "spread_divergence",
    "snap_roc",
    "global_regime_reversal_cnt",
    "regime_reversal_count",
    "is_volatile_regime",
    "regime_band",
    "regime_score",
    "state",
)


def validate_strategy_ctx_for_v0(
    ctx: Mapping[str, Any],
    *,
    require_state: bool = True,
) -> None:
    """
    Assert all canonical keys exist on ctx (value may be None where strategy allows).

    Raises:
        KeyError: missing key
        TypeError: ctx is not a mapping
    """
    if not isinstance(ctx, Mapping):
        raise TypeError(f"ctx must be a mapping, got {type(ctx)!r}")
    missing = [k for k in STRATEGY_CTX_V0_KEYS if k not in ctx]
    if missing:
        raise KeyError("strategy ctx missing keys: " + ", ".join(missing))
    if require_state and ctx.get("state") is None:
        raise ValueError("ctx['state'] is required for OMS parity (SymbolState)")


def normalize_spread_pct_cap_for_parity(cfg: Any) -> MutableMapping[str, Any]:
    """
    Document-only helper: return a dict of spread-related attrs from cfg for logging
    / audit that live and replay use the same caps (not mutates cfg).
    """
    out: Dict[str, Any] = {}
    for name in (
        "MAX_SPREAD_PCT_ENTRY",
        "MAX_SPREAD_PCT_ENTRY_CALL",
        "MAX_SPREAD_PCT_ENTRY_PUT",
        "MAX_SPREAD_PCT_EXIT",
        "MAX_SPREAD_DIVERGENCE",
    ):
        if hasattr(cfg, name):
            out[name] = getattr(cfg, name)
    module = getattr(type(cfg), "__module__", "")
    out["_config_module"] = module
    return out

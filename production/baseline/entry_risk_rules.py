from typing import Any, Dict


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value or 0.0)
    except Exception:
        return float(default)


def get_entry_min_option_price(cfg_or_value: Any, hard_floor: float = 0.05) -> float:
    """Return the effective minimum option price used for entry checks."""
    if hasattr(cfg_or_value, "MIN_OPTION_PRICE"):
        raw_value = getattr(cfg_or_value, "MIN_OPTION_PRICE", 0.0)
    else:
        raw_value = cfg_or_value
    return max(_to_float(raw_value, 0.0), _to_float(hard_floor, 0.05))


def get_entry_direction(alpha_z: Any) -> int:
    return 1 if _to_float(alpha_z, 0.0) >= 0.0 else -1


def get_entry_spread_threshold(curr_price: Any, alpha_z: Any, cfg: Any) -> float:
    price = _to_float(curr_price, 0.0)
    if price <= 0.5:
        dynamic_threshold = 0.20
    elif price >= 5.0:
        dynamic_threshold = 0.10
    else:
        dynamic_threshold = 0.20 - (price - 0.5) * (0.10 / 4.5)

    direction = get_entry_direction(alpha_z)
    fallback_cap = _to_float(getattr(cfg, "MAX_SPREAD_PCT_ENTRY", 0.10), 0.10)
    side_cap = (
        _to_float(getattr(cfg, "MAX_SPREAD_PCT_ENTRY_CALL", fallback_cap), fallback_cap)
        if direction >= 0
        else _to_float(getattr(cfg, "MAX_SPREAD_PCT_ENTRY_PUT", fallback_cap), fallback_cap)
    )
    return min(dynamic_threshold, side_cap)


def evaluate_entry_liquidity(
    *,
    bid: Any,
    ask: Any,
    curr_price: Any,
    alpha_z: Any,
    spread_divergence: Any,
    cfg: Any,
) -> Dict[str, Any]:
    """Evaluate shared entry price/liquidity guards across strategy and OMS."""
    bid_f = _to_float(bid, 0.0)
    ask_f = _to_float(ask, 0.0)
    curr_f = _to_float(curr_price, 0.0)
    direction = get_entry_direction(alpha_z)
    direction_label = "CALL" if direction >= 0 else "PUT"
    min_option_price = get_entry_min_option_price(cfg)

    decision: Dict[str, Any] = {
        "ok": False,
        "reason": "",
        "detail": "",
        "direction": direction,
        "direction_label": direction_label,
        "bid": bid_f,
        "ask": ask_f,
        "curr_price": curr_f,
        "effective_min_option_price": min_option_price,
        "spread_pct": 0.0,
        "spread_threshold": 0.0,
        "spread_divergence": _to_float(spread_divergence, 0.0),
    }

    if bid_f <= 0.01 or ask_f <= 0.01 or curr_f <= 0.01:
        decision["reason"] = "bidask_invalid"
        decision["detail"] = f"bid={bid_f:.3f}/ask={ask_f:.3f}/p={curr_f:.3f}"
        return decision

    if curr_f < min_option_price:
        decision["reason"] = "min_option_price"
        decision["detail"] = f"p={curr_f:.3f} < min_option_price={min_option_price:.3f}"
        return decision

    spread_threshold = get_entry_spread_threshold(curr_f, alpha_z, cfg)
    spread_pct = (ask_f - bid_f) / curr_f if curr_f > 0.0 else 0.0
    decision["spread_pct"] = spread_pct
    decision["spread_threshold"] = spread_threshold
    if spread_pct > spread_threshold:
        decision["reason"] = "spread_too_wide"
        decision["detail"] = (
            f"dir={direction_label} spread={spread_pct:.2%} > th={spread_threshold:.2%}"
        )
        return decision

    max_spread_divergence = _to_float(getattr(cfg, "MAX_SPREAD_DIVERGENCE", 0.02), 0.02)
    if decision["spread_divergence"] > max_spread_divergence:
        decision["reason"] = "spread_divergence"
        decision["detail"] = (
            f"div={decision['spread_divergence']:.4f} > {max_spread_divergence}"
        )
        return decision

    decision["ok"] = True
    decision["reason"] = "ok"
    decision["detail"] = (
        f"dir={direction_label} spread={spread_pct:.2%} ≤ {spread_threshold:.2%}"
    )
    return decision

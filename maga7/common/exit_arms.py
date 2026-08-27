"""Exit-arm snapshot + light health hints for live OMS / Dash.

Keeps toggles visible (enabled / windows / day trigger counts) without binding
every arm into the full path. Suggestions are advisory only — never auto-disable.
"""
from __future__ import annotations

from typing import Any

from maga7.common.hold_watchdog import hold_watchdog_from_trade
from maga7.common.option_trades import trade_toxic_from_trade

# Early-cut reasons that can false-cut winners if regime flips.
EARLY_CUT_REASONS = frozenset(
    {
        "TRADE_TOX",
        "TRADE_TOX_RECONNECT",
        "HOLD_SHOCK",
        "MTM_FLOOR",
        "MF_FLIP",
        "STREAK0",
    }
)


def build_exit_arms(
    trade: dict[str, Any] | None,
    *,
    reason_counts: dict[str, int] | None = None,
) -> dict[str, Any]:
    trade = trade or {}
    counts = {str(k): int(v) for k, v in (reason_counts or {}).items()}
    ttox = trade_toxic_from_trade(trade)
    hwd = hold_watchdog_from_trade(trade)
    day_circuit = trade.get("day_circuit")
    day_circuit_on = day_circuit not in (None, "", False)
    early = str(trade.get("early_exit_mode") or "").strip().lower()
    mae_on = early in {"mae_cut", "trade_mae"} or "mae_cut" in str(
        trade.get("exit_mode") or ""
    ).lower()

    def _n(*keys: str) -> int:
        return sum(int(counts.get(k, 0)) for k in keys)

    return {
        "trade_toxic": {
            "enabled": bool(ttox.enabled),
            "cut_ret": float(ttox.cut_ret),
            "mfe_bypass": float(ttox.mfe_bypass),
            "min_hold_seconds": int(ttox.min_hold_seconds),
            "max_cut_seconds": ttox.max_cut_seconds,
            "n_triggers": _n("TRADE_TOX", "TRADE_TOX_RECONNECT"),
        },
        "hold_watchdog": {
            "enabled": bool(hwd.enabled),
            "qqq_adverse_from_entry": float(hwd.qqq_adverse_from_entry),
            "min_hold_seconds": int(hwd.min_hold_seconds),
            "require_option_mtm_max": hwd.require_option_mtm_max,
            "n_triggers": _n("HOLD_SHOCK"),
        },
        "sl_tp": {
            "sl_mult": float(trade.get("sl_mult", 0.45) or 0.45),
            "tp_mult": float(trade.get("tp_mult", 1.6) or 1.6),
            "hold_minutes": int(trade.get("hold_minutes", 30) or 30),
            "n_sl": _n("SL"),
            "n_tp": _n("TP"),
        },
        "day_circuit": {
            "enabled": bool(day_circuit_on),
            "threshold": float(day_circuit) if day_circuit_on else None,
            "n_triggers": _n("DAY_CIRCUIT"),
        },
        "mae_cut": {
            "enabled": bool(mae_on),
            "note": "research REJECT; live evaluate_exits has no MAE_CUT branch",
            "n_triggers": _n("MAE_CUT"),
        },
    }


def build_exit_health(
    reason_counts: dict[str, int] | None,
    *,
    arms: dict[str, Any] | None = None,
) -> dict[str, Any]:
    counts = {str(k): int(v) for k, v in (reason_counts or {}).items()}
    n_close = sum(counts.values())
    n_early = sum(v for k, v in counts.items() if k in EARLY_CUT_REASONS)
    n_sl = int(counts.get("SL", 0))
    suggestions: list[str] = []
    if n_close >= 3 and n_early >= 2 and n_early >= max(2, int(0.6 * n_close)):
        suggestions.append(
            "early_cut_heavy: 早切占比偏高，复核 trade_toxic/hold_watchdog 是否误伤"
        )
    if n_sl >= 3 and n_early == 0 and n_close >= 3:
        suggestions.append(
            "sl_only: 多次硬 SL、无早切；考虑是否应开/收紧 toxic 或 hold_watchdog"
        )
    tox_n = int((arms or {}).get("trade_toxic", {}).get("n_triggers") or 0)
    if tox_n >= 3:
        suggestions.append(
            "trade_toxic_hot: 当日 TRADE_TOX≥3，建议对照误伤日后再决定降权"
        )
    return {
        "n_closes": n_close,
        "n_early_cut": n_early,
        "n_sl": n_sl,
        "closes_by_reason": dict(sorted(counts.items())),
        "suggestions": suggestions,
        "auto_disable": False,
        "note": "suggestions are advisory only; never auto-disable arms",
    }

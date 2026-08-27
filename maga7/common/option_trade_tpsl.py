"""Causal TP/SL exits on option trade-last paths (no fixed hold as primary exit).

Entry: buy at ``last*(1+slip)``.
MTM/exit: sell at ``last*(1-slip)``.
First hit of +tp or −sl wins; ``max_hold_sec`` is only a safety flatten
(session end / research cap), not the intended alpha exit.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from maga7.common.replay import to_ny


def simulate_trade_tpsl(
    ts_ns: np.ndarray,
    last: np.ndarray,
    entry_ts: pd.Timestamp,
    *,
    tp: float,
    sl: float,
    max_hold_sec: int = 900,
    slip: float = 0.01,
) -> dict[str, Any] | None:
    """Return exit ret/reason/hold, or None if no entry print within 5s."""
    t0 = int(to_ny(entry_ts).value)
    i0 = int(np.searchsorted(ts_ns, t0, side="left"))
    if i0 >= len(ts_ns):
        return None
    lag = (int(ts_ns[i0]) - t0) / 1e9
    if lag > 5:
        return None
    entry = float(last[i0]) * (1.0 + float(slip))
    if not np.isfinite(entry) or entry <= 0:
        return None
    sell_mult = 1.0 - float(slip)
    end_ns = int(ts_ns[i0]) + int(max_hold_sec) * 1_000_000_000
    i_end = int(np.searchsorted(ts_ns, end_ns, side="right") - 1)
    if i_end < i0:
        return None

    tp_v = float(tp)
    sl_v = float(sl)
    mfe = -1.0
    mae = 1.0
    exit_i = i_end
    reason = "max_hold"
    for k in range(i0 + 1, i_end + 1):
        px = float(last[k])
        if not np.isfinite(px) or px <= 0:
            continue
        ret = px * sell_mult / entry - 1.0
        if ret > mfe:
            mfe = ret
        if ret < mae:
            mae = ret
        if ret >= tp_v:
            exit_i = k
            reason = "tp"
            break
        if ret <= -sl_v:
            exit_i = k
            reason = "sl"
            break
    else:
        px = float(last[i_end])
        ret = px * sell_mult / entry - 1.0 if px > 0 else float("nan")
        mfe = max(mfe, ret) if np.isfinite(ret) else mfe
        mae = min(mae, ret) if np.isfinite(ret) else mae

    px_x = float(last[exit_i])
    ret = px_x * sell_mult / entry - 1.0
    hold = (int(ts_ns[exit_i]) - int(ts_ns[i0])) / 1e9
    return {
        "ret": float(ret),
        "reason": reason,
        "hold_sec": float(hold),
        "mfe": float(mfe if np.isfinite(mfe) else ret),
        "mae": float(mae if np.isfinite(mae) else ret),
        "entry_lag_sec": float(lag),
        "tp": tp_v,
        "sl": sl_v,
        "max_hold_sec": int(max_hold_sec),
    }


def simulate_trade_tpsl_confirm_abort(
    ts_ns: np.ndarray,
    last: np.ndarray,
    entry_ts: pd.Timestamp,
    *,
    tp: float,
    sl: float,
    max_hold_sec: int = 900,
    confirm_sec: int = 60,
    confirm_thr: float = 0.02,
    abort_thr: float = 0.10,
    on_timeout: str = "abort",
    slip: float = 0.01,
) -> dict[str, Any] | None:
    """Trade-last TP/SL with post-fill confirm-or-abort (causal)."""
    if on_timeout not in ("abort", "allow"):
        raise ValueError(f"on_timeout must be abort|allow, got {on_timeout!r}")
    t0 = int(to_ny(entry_ts).value)
    i0 = int(np.searchsorted(ts_ns, t0, side="left"))
    if i0 >= len(ts_ns):
        return None
    lag = (int(ts_ns[i0]) - t0) / 1e9
    if lag > 5:
        return None
    entry = float(last[i0]) * (1.0 + float(slip))
    if not np.isfinite(entry) or entry <= 0:
        return None
    sell_mult = 1.0 - float(slip)
    end_ns = int(ts_ns[i0]) + int(max_hold_sec) * 1_000_000_000
    i_end = int(np.searchsorted(ts_ns, end_ns, side="right") - 1)
    if i_end < i0:
        return None
    confirm_ns = int(ts_ns[i0]) + int(confirm_sec) * 1_000_000_000
    tp_v = float(tp)
    sl_v = float(sl)
    conf_v = float(confirm_thr)
    abort_v = float(abort_thr)
    confirmed = False
    mfe = -1.0
    mae = 1.0
    exit_i = i_end
    reason = "max_hold"
    for k in range(i0 + 1, i_end + 1):
        px = float(last[k])
        if not np.isfinite(px) or px <= 0:
            continue
        ret = px * sell_mult / entry - 1.0
        if ret > mfe:
            mfe = ret
        if ret < mae:
            mae = ret
        ts_k = int(ts_ns[k])
        if not confirmed:
            if ret >= conf_v:
                confirmed = True
            elif ret <= -abort_v:
                exit_i, reason = k, "early_abort"
                break
            elif ts_k >= confirm_ns:
                if on_timeout == "abort":
                    exit_i, reason = k, "confirm_abort"
                    break
                confirmed = True
        if ret >= tp_v:
            exit_i, reason = k, "tp"
            break
        if ret <= -sl_v:
            exit_i, reason = k, "sl"
            break
    else:
        px = float(last[i_end])
        ret = px * sell_mult / entry - 1.0 if px > 0 else float("nan")
        mfe = max(mfe, ret) if np.isfinite(ret) else mfe
        mae = min(mae, ret) if np.isfinite(ret) else mae

    px_x = float(last[exit_i])
    ret = px_x * sell_mult / entry - 1.0
    hold = (int(ts_ns[exit_i]) - int(ts_ns[i0])) / 1e9
    return {
        "ret": float(ret),
        "reason": reason,
        "hold_sec": float(hold),
        "mfe": float(mfe if np.isfinite(mfe) else ret),
        "mae": float(mae if np.isfinite(mae) else ret),
        "entry_lag_sec": float(lag),
        "confirmed": bool(confirmed),
        "tp": tp_v,
        "sl": sl_v,
        "max_hold_sec": int(max_hold_sec),
    }

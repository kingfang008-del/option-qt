"""Causal TP/SL / timer exits on option quote paths (FillSpec buy/sell).

Entry: first quote at/after signal with bid/ask, buy via FillSpec.
Exit: first of +tp, −sl, or ``timer_sec`` (scalp primary). Legacy callers may
pass ``max_hold_sec`` as a safety flatten (reason ``max_hold``).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from maga7.common.fills import FillSpec
from maga7.common.replay import to_ny


def spread_pct(bid: float, ask: float) -> float:
    mid = 0.5 * (float(bid) + float(ask))
    if not np.isfinite(mid) or mid <= 0:
        return float("inf")
    return float((float(ask) - float(bid)) / mid)


def entry_quote_row(
    path: pd.DataFrame,
    entry_ts: pd.Timestamp,
    *,
    max_lag_sec: float = 3.0,
    max_spread_pct: float = 0.15,
    min_mid: float = 0.05,
) -> dict[str, Any] | None:
    """First usable quote at/after signal satisfying lag/spread/mid gates."""
    if path is None or path.empty:
        return None
    t0 = to_ny(entry_ts)
    after = path[path["timestamp"] >= t0]
    if after.empty:
        return None
    r0 = after.iloc[0]
    ts = to_ny(r0["timestamp"])
    lag = (ts - t0).total_seconds()
    if lag > float(max_lag_sec):
        return None
    bid = float(r0["bid"])
    ask = float(r0["ask"])
    if not (np.isfinite(bid) and np.isfinite(ask) and ask > bid > 0):
        return None
    mid = 0.5 * (bid + ask)
    if mid < float(min_mid):
        return None
    sp = spread_pct(bid, ask)
    if sp > float(max_spread_pct):
        return None
    return {
        "entry_ts": ts,
        "bid": bid,
        "ask": ask,
        "mid": float(mid),
        "spread_pct": float(sp),
        "lag_sec": float(lag),
        "after": after,
    }


def simulate_quote_tpsl(
    path: pd.DataFrame,
    entry_ts: pd.Timestamp,
    *,
    tp: float,
    sl: float,
    max_hold_sec: int = 900,
    fill: FillSpec | None = None,
    max_lag_sec: float = 3.0,
    max_spread_pct: float = 0.15,
    min_mid: float = 0.05,
) -> dict[str, Any] | None:
    """Return exit ret/reason/hold, or None if entry gate fails."""
    fill = fill or FillSpec(entry_frac=0.75, exit_frac=0.75)
    ent = entry_quote_row(
        path,
        entry_ts,
        max_lag_sec=max_lag_sec,
        max_spread_pct=max_spread_pct,
        min_mid=min_mid,
    )
    if ent is None:
        return None
    after: pd.DataFrame = ent["after"]
    entry_px = fill.buy(ent["bid"], ent["ask"])
    if not np.isfinite(entry_px) or entry_px <= 0:
        return None
    t_entry = ent["entry_ts"]
    end = t_entry + pd.Timedelta(seconds=int(max_hold_sec))
    tp_v = float(tp)
    sl_v = float(sl)
    mfe = -1.0
    mae = 1.0
    reason = "max_hold"
    ret = float("nan")
    hold = 0.0
    exit_ts = t_entry

    for i in range(1, len(after)):
        r = after.iloc[i]
        ts = to_ny(r["timestamp"])
        if ts > end:
            prev = after.iloc[i - 1]
            px = fill.sell(float(prev["bid"]), float(prev["ask"]))
            ret = px / entry_px - 1.0
            hold = (to_ny(prev["timestamp"]) - t_entry).total_seconds()
            exit_ts = to_ny(prev["timestamp"])
            reason = "max_hold"
            break
        bid, ask = float(r["bid"]), float(r["ask"])
        if not (np.isfinite(bid) and np.isfinite(ask) and ask > bid > 0):
            continue
        px = fill.sell(bid, ask)
        cur = px / entry_px - 1.0
        if cur > mfe:
            mfe = cur
        if cur < mae:
            mae = cur
        if cur >= tp_v:
            ret, hold, reason, exit_ts = cur, (ts - t_entry).total_seconds(), "tp", ts
            break
        if cur <= -sl_v:
            ret, hold, reason, exit_ts = cur, (ts - t_entry).total_seconds(), "sl", ts
            break
    else:
        last = after.iloc[-1]
        px = fill.sell(float(last["bid"]), float(last["ask"]))
        ret = px / entry_px - 1.0
        hold = (to_ny(last["timestamp"]) - t_entry).total_seconds()
        exit_ts = to_ny(last["timestamp"])
        reason = "max_hold"
        mfe = max(mfe, ret) if np.isfinite(ret) else mfe
        mae = min(mae, ret) if np.isfinite(ret) else mae

    if not np.isfinite(ret):
        return None
    return {
        "ret": float(ret),
        "reason": reason,
        "hold_sec": float(hold),
        "mfe": float(mfe if np.isfinite(mfe) else ret),
        "mae": float(mae if np.isfinite(mae) else ret),
        "entry_lag_sec": float(ent["lag_sec"]),
        "entry_spread_pct": float(ent["spread_pct"]),
        "entry_mid": float(ent["mid"]),
        "entry_px": float(entry_px),
        "entry_ts": t_entry,
        "exit_ts": exit_ts,
        "tp": tp_v,
        "sl": sl_v,
        "max_hold_sec": int(max_hold_sec),
    }


def simulate_quote_scalp(
    path: pd.DataFrame,
    entry_ts: pd.Timestamp,
    *,
    tp: float,
    sl: float,
    timer_sec: int = 90,
    fill: FillSpec | None = None,
    max_lag_sec: float = 3.0,
    max_spread_pct: float = 0.15,
    min_mid: float = 0.05,
) -> dict[str, Any] | None:
    """1–2m scalp book: first of TP / SL / hard timer (reason ``timer``)."""
    out = simulate_quote_tpsl(
        path,
        entry_ts,
        tp=tp,
        sl=sl,
        max_hold_sec=int(timer_sec),
        fill=fill,
        max_lag_sec=max_lag_sec,
        max_spread_pct=max_spread_pct,
        min_mid=min_mid,
    )
    if out is None:
        return None
    if out["reason"] == "max_hold":
        out["reason"] = "timer"
    out["timer_sec"] = int(timer_sec)
    return out


def simulate_quote_tpsl_confirm_abort(
    path: pd.DataFrame,
    entry_ts: pd.Timestamp,
    *,
    tp: float,
    sl: float,
    max_hold_sec: int = 900,
    confirm_sec: int = 120,
    confirm_thr: float = 0.03,
    abort_thr: float | None = None,
    on_timeout: str = "abort",
    fill: FillSpec | None = None,
    max_lag_sec: float = 5.0,
    max_spread_pct: float = 0.15,
    min_mid: float = 0.05,
) -> dict[str, Any] | None:
    """Post-fill confirm-or-abort, then first-passage TP/SL.

    Causal rules after entry fill:

    - TP / SL always active.
    - Before confirmation: if ``abort_thr`` is set and mark <= −abort → ``early_abort``.
    - If mark reaches ``confirm_thr`` within ``confirm_sec`` → confirmed.
    - At ``confirm_sec`` without confirm: ``on_timeout=abort`` flattens
      (``confirm_abort``); ``allow`` continues under TP/SL only.
    """
    fill = fill or FillSpec(entry_frac=0.75, exit_frac=0.75)
    if on_timeout not in ("abort", "allow"):
        raise ValueError(f"on_timeout must be abort|allow, got {on_timeout!r}")
    ent = entry_quote_row(
        path,
        entry_ts,
        max_lag_sec=max_lag_sec,
        max_spread_pct=max_spread_pct,
        min_mid=min_mid,
    )
    if ent is None:
        return None
    after: pd.DataFrame = ent["after"]
    entry_px = fill.buy(ent["bid"], ent["ask"])
    if not np.isfinite(entry_px) or entry_px <= 0:
        return None
    t_entry = ent["entry_ts"]
    end = t_entry + pd.Timedelta(seconds=int(max_hold_sec))
    conf_deadline = t_entry + pd.Timedelta(seconds=int(confirm_sec))
    tp_v = float(tp)
    sl_v = float(sl)
    conf_thr = float(confirm_thr)
    abort_v = None if abort_thr is None else abs(float(abort_thr))
    mfe = -1.0
    mae = 1.0
    confirmed = conf_thr <= 0.0
    reason = "max_hold"
    ret = float("nan")
    hold = 0.0
    exit_ts = t_entry
    timed_out = False

    for i in range(1, len(after)):
        r = after.iloc[i]
        ts = to_ny(r["timestamp"])
        if ts > end:
            prev = after.iloc[i - 1]
            px = fill.sell(float(prev["bid"]), float(prev["ask"]))
            ret = px / entry_px - 1.0
            hold = (to_ny(prev["timestamp"]) - t_entry).total_seconds()
            exit_ts = to_ny(prev["timestamp"])
            reason = "max_hold"
            break
        bid, ask = float(r["bid"]), float(r["ask"])
        if not (np.isfinite(bid) and np.isfinite(ask) and ask > bid > 0):
            continue
        px = fill.sell(bid, ask)
        cur = px / entry_px - 1.0
        if cur > mfe:
            mfe = cur
        if cur < mae:
            mae = cur

        if cur >= tp_v:
            ret, hold, reason, exit_ts = cur, (ts - t_entry).total_seconds(), "tp", ts
            break
        if cur <= -sl_v:
            ret, hold, reason, exit_ts = cur, (ts - t_entry).total_seconds(), "sl", ts
            break

        if not confirmed:
            if abort_v is not None and cur <= -abort_v:
                ret, hold, reason, exit_ts = (
                    cur,
                    (ts - t_entry).total_seconds(),
                    "early_abort",
                    ts,
                )
                break
            if cur >= conf_thr:
                confirmed = True
            elif ts >= conf_deadline:
                if on_timeout == "abort":
                    ret, hold, reason, exit_ts = (
                        cur,
                        (ts - t_entry).total_seconds(),
                        "confirm_abort",
                        ts,
                    )
                    break
                confirmed = True
                timed_out = True
    else:
        last = after.iloc[-1]
        px = fill.sell(float(last["bid"]), float(last["ask"]))
        ret = px / entry_px - 1.0
        hold = (to_ny(last["timestamp"]) - t_entry).total_seconds()
        exit_ts = to_ny(last["timestamp"])
        reason = "max_hold"
        mfe = max(mfe, ret) if np.isfinite(ret) else mfe
        mae = min(mae, ret) if np.isfinite(ret) else mae

    if not np.isfinite(ret):
        return None
    return {
        "ret": float(ret),
        "reason": reason,
        "hold_sec": float(hold),
        "mfe": float(mfe if np.isfinite(mfe) else ret),
        "mae": float(mae if np.isfinite(mae) else ret),
        "confirmed": bool(confirmed),
        "timed_out_allow": bool(timed_out),
        "entry_lag_sec": float(ent["lag_sec"]),
        "entry_spread_pct": float(ent["spread_pct"]),
        "entry_mid": float(ent["mid"]),
        "entry_px": float(entry_px),
        "entry_ts": t_entry,
        "exit_ts": exit_ts,
        "tp": tp_v,
        "sl": sl_v,
        "max_hold_sec": int(max_hold_sec),
        "confirm_sec": int(confirm_sec),
        "confirm_thr": conf_thr,
        "abort_thr": None if abort_v is None else float(abort_v),
        "on_timeout": on_timeout,
    }

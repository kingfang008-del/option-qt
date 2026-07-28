"""Causal path-state exit simulator for offline stress gates.

Combines TP/SL with optional post-fill confirm/abort and peak-armed
giveback / BE / trail / ladder floors. Used by ``run_exit_stress_gate``.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Literal

import numpy as np
import pandas as pd

from maga7.common.fills import FillSpec
from maga7.common.option_quote_tpsl import entry_quote_row
from maga7.common.replay import to_ny

FloorMode = Literal["be", "trail", "ladder"]


@dataclass(frozen=True)
class ExitStressPolicy:
    name: str = "tpsl"
    tp: float = 0.15
    sl: float = 0.20
    max_hold_sec: int = 900
    # confirm-or-abort (pre-confirm window)
    confirm_enabled: bool = False
    confirm_sec: int = 60
    confirm_thr: float = 0.02
    abort_thr: float | None = 0.08
    on_timeout: str = "abort"  # abort | allow
    # peak giveback flatten
    giveback_arm: float | None = None
    giveback_thr: float | None = None
    giveback_green_only: bool = False
    # dynamic stop floor after peak arm
    floor_arm: float | None = None
    floor_mode: FloorMode | None = None
    floor_offset: float | None = None  # trail: peak−offset; ladder: absolute floor; be ignored
    # optional: only arm giveback/floor after N seconds
    arm_min_sec: float = 0.0


def policy_preset(name: str, *, tp: float = 0.15, sl: float = 0.20) -> ExitStressPolicy:
    """Named presets used by the stress gate scoreboard."""
    base = ExitStressPolicy(name=name, tp=tp, sl=sl)
    if name == "tpsl":
        return base
    if name == "hard_sl12":
        return replace(base, sl=0.12)
    if name == "gb08_p10":
        return replace(base, giveback_arm=0.10, giveback_thr=0.08)
    if name == "gb08_p08":
        return replace(base, giveback_arm=0.08, giveback_thr=0.08)
    if name == "gb12_p15":
        return replace(base, giveback_arm=0.15, giveback_thr=0.12)
    if name == "gb08_green":
        return replace(base, giveback_arm=0.08, giveback_thr=0.08, giveback_green_only=True)
    if name == "be_lock08":
        return replace(base, floor_arm=0.08, floor_mode="be", floor_offset=0.0)
    if name == "trail10_8":
        return replace(base, floor_arm=0.10, floor_mode="trail", floor_offset=0.08)
    if name == "ladder_08_03":
        return replace(base, floor_arm=0.08, floor_mode="ladder", floor_offset=0.03)
    if name == "fast_lad10_5_180":
        return replace(
            base,
            floor_arm=0.10,
            floor_mode="ladder",
            floor_offset=0.05,
            arm_min_sec=0.0,
            # arm only if peak hit within 180s — enforced in sim via deadline
        )
    if name == "ca_t60_c02_a08":
        return replace(
            base,
            confirm_enabled=True,
            confirm_sec=60,
            confirm_thr=0.02,
            abort_thr=0.08,
        )
    if name == "ca_gb08_p10":
        return replace(
            base,
            confirm_enabled=True,
            confirm_sec=60,
            confirm_thr=0.02,
            abort_thr=0.08,
            giveback_arm=0.10,
            giveback_thr=0.08,
        )
    raise ValueError(f"unknown exit stress preset: {name!r}")


def simulate_quote_exit_stress(
    path: pd.DataFrame,
    entry_ts: pd.Timestamp,
    policy: ExitStressPolicy,
    *,
    fill: FillSpec | None = None,
    max_lag_sec: float = 5.0,
    max_spread_pct: float = 0.15,
    min_mid: float = 0.05,
    # fast_lad: only arm ladder if peak>=arm within this many seconds (0=disabled)
    arm_within_sec: float | None = None,
) -> dict[str, Any] | None:
    """Replay one entry under a path-state exit policy.

    Priority each quote tick after fill:
      TP → confirm early_abort / confirm_abort → giveback → dynamic floor → SL → max_hold
    """
    fill = fill or FillSpec(entry_frac=0.75, exit_frac=0.75)
    if policy.on_timeout not in ("abort", "allow"):
        raise ValueError(f"on_timeout must be abort|allow, got {policy.on_timeout!r}")
    if policy.name == "fast_lad10_5_180" and arm_within_sec is None:
        arm_within_sec = 180.0

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
    end = t_entry + pd.Timedelta(seconds=int(policy.max_hold_sec))
    conf_deadline = t_entry + pd.Timedelta(seconds=int(policy.confirm_sec))
    arm_deadline = (
        None
        if arm_within_sec is None or float(arm_within_sec) <= 0
        else t_entry + pd.Timedelta(seconds=float(arm_within_sec))
    )

    tp_v = float(policy.tp)
    sl_v = float(policy.sl)
    floor = -sl_v
    mfe = -1.0
    mae = 1.0
    confirmed = (not policy.confirm_enabled) or float(policy.confirm_thr) <= 0.0
    armed = False
    arm_eligible = True
    timed_out = False
    reason = "max_hold"
    ret = float("nan")
    hold = 0.0
    exit_ts = t_entry
    abort_v = None if policy.abort_thr is None else abs(float(policy.abort_thr))

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
        hold_now = (ts - t_entry).total_seconds()
        if cur > mfe:
            mfe = cur
        if cur < mae:
            mae = cur

        # 1) hard TP always
        if cur >= tp_v:
            ret, hold, reason, exit_ts = cur, hold_now, "tp", ts
            break

        # 2) confirm / early abort while unconfirmed
        if policy.confirm_enabled and not confirmed:
            if abort_v is not None and cur <= -abort_v:
                ret, hold, reason, exit_ts = cur, hold_now, "early_abort", ts
                break
            if cur >= float(policy.confirm_thr):
                confirmed = True
            elif ts >= conf_deadline:
                if policy.on_timeout == "abort":
                    ret, hold, reason, exit_ts = cur, hold_now, "confirm_abort", ts
                    break
                confirmed = True
                timed_out = True

        # 3) arm eligibility window (fast ladder)
        if arm_deadline is not None and not armed and ts > arm_deadline and mfe < float(
            policy.floor_arm or 0.0
        ):
            arm_eligible = False

        can_arm = hold_now >= float(policy.arm_min_sec) and arm_eligible

        # 4) peak giveback
        if (
            can_arm
            and policy.giveback_arm is not None
            and policy.giveback_thr is not None
            and mfe >= float(policy.giveback_arm)
            and (mfe - cur) >= float(policy.giveback_thr)
        ):
            if (not policy.giveback_green_only) or cur >= 0.0:
                ret, hold, reason, exit_ts = cur, hold_now, "giveback", ts
                break

        # 5) dynamic floor
        if can_arm and policy.floor_arm is not None and policy.floor_mode is not None:
            if mfe >= float(policy.floor_arm):
                armed = True
                if policy.floor_mode == "be":
                    floor = max(floor, 0.0)
                elif policy.floor_mode == "trail":
                    floor = max(floor, mfe - float(policy.floor_offset or 0.0))
                elif policy.floor_mode == "ladder":
                    floor = max(floor, float(policy.floor_offset or 0.0))

        # 6) stop / floor
        if cur <= floor:
            if armed and policy.floor_mode == "be":
                reason = "be_stop"
            elif armed and policy.floor_mode == "trail":
                reason = "trail"
            elif armed and policy.floor_mode == "ladder":
                reason = "ladder_stop"
            else:
                reason = "sl"
            ret, hold, exit_ts = cur, hold_now, ts
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
        "confirmed": bool(confirmed),
        "timed_out_allow": bool(timed_out),
        "armed": bool(armed),
        "entry_lag_sec": float(ent["lag_sec"]),
        "entry_spread_pct": float(ent["spread_pct"]),
        "entry_mid": float(ent["mid"]),
        "entry_px": float(entry_px),
        "entry_ts": t_entry,
        "exit_ts": exit_ts,
        "policy": policy.name,
        "tp": tp_v,
        "sl": sl_v,
        "max_hold_sec": int(policy.max_hold_sec),
    }

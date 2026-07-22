"""Post-fill wave confirm with revocable ARMED → WAVE_ABORT.

Unlike pre-fill ``stock_path_confirm`` (touch-and-go gate), this watches the
underlying **after** fill:

1. Before arm: ``signed <= thr_neg`` → abort; ``signed >= thr_pos`` → arm.
2. Timeout without arm (``on_timeout=abort``) → abort.
3. After arm, until ``revoke_seconds``: ``signed <= thr_neg`` → **revoke** abort.

See ``docs/wave_confirm_spec.md``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


AbortReason = Literal["neg", "timeout", "revoke", ""]


@dataclass
class WaveAbortConfig:
    enabled: bool = False
    thr_pos: float = 0.0015
    thr_neg: float = -0.003
    max_wait_seconds: float = 300.0
    # Keep watching for revoke after fill (covers Jul21-style confirm-then-fade).
    revoke_seconds: float = 1800.0
    on_timeout: str = "abort"  # abort | allow
    # If set, also abort when option MTM <= this (stock may be flat).
    opt_mtm_max: float | None = None
    # Asymmetric revoke: default = thr_neg. Deeper (more negative) = fewer false revokes.
    thr_neg_revoke: float | None = None
    # If set, ARMED→revoke only when option MTM <= this as well.
    revoke_opt_mtm_max: float | None = None
    # If False, once armed the confirm window closes (no revoke).
    allow_revoke: bool = True
    # Restrict to directions, e.g. ("UP",) to avoid DN false revokes (Jul02 AMD).
    only_directions: tuple[str, ...] | None = None


@dataclass
class WaveAbortState:
    armed: bool = False
    done: bool = False  # window closed without abort


def wave_abort_from_trade(trade: dict[str, Any] | None) -> WaveAbortConfig:
    trade = trade or {}
    raw = trade.get("wave_abort")
    if raw is None:
        return WaveAbortConfig(enabled=False)
    if isinstance(raw, bool):
        return WaveAbortConfig(enabled=bool(raw))
    if not isinstance(raw, dict):
        return WaveAbortConfig(enabled=False)
    ot = str(raw.get("on_timeout", "abort") or "abort").strip().lower()
    if ot in {"block", "fail"}:
        ot = "abort"
    if ot in {"pass", "keep"}:
        ot = "allow"
    opt_mtm = raw.get("opt_mtm_max")
    thr_neg = float(raw.get("thr_neg", -0.003) if raw.get("thr_neg") is not None else -0.003)
    thr_rev = raw.get("thr_neg_revoke")
    rev_opt = raw.get("revoke_opt_mtm_max")
    only_raw = raw.get("only_directions")
    only_dirs: tuple[str, ...] | None = None
    if isinstance(only_raw, str) and only_raw.strip():
        only_dirs = tuple(x.strip().upper() for x in only_raw.split(",") if x.strip())
    elif isinstance(only_raw, (list, tuple)):
        only_dirs = tuple(str(x).strip().upper() for x in only_raw if str(x).strip())
    if only_dirs == ():
        only_dirs = None
    return WaveAbortConfig(
        enabled=bool(raw.get("enabled", False)),
        thr_pos=float(raw.get("thr_pos", 0.0015) if raw.get("thr_pos") is not None else 0.0015),
        thr_neg=thr_neg,
        max_wait_seconds=float(
            raw.get("max_wait_seconds", 300.0) if raw.get("max_wait_seconds") is not None else 300.0
        ),
        revoke_seconds=float(
            raw.get("revoke_seconds", 1800.0) if raw.get("revoke_seconds") is not None else 1800.0
        ),
        on_timeout=ot if ot in {"abort", "allow"} else "abort",
        opt_mtm_max=None if opt_mtm in (None, "", False) else float(opt_mtm),
        thr_neg_revoke=None if thr_rev in (None, "", False) else float(thr_rev),
        revoke_opt_mtm_max=None if rev_opt in (None, "", False) else float(rev_opt),
        allow_revoke=bool(raw.get("allow_revoke", True)),
        only_directions=only_dirs,
    )


def wave_abort_on_tick(
    state: WaveAbortState,
    *,
    cfg: WaveAbortConfig,
    held_seconds: float,
    stock_signed: float,
    opt_mtm: float | None = None,
) -> tuple[bool, AbortReason, WaveAbortState]:
    """Return ``(should_abort, reason, new_state)``."""
    if not cfg.enabled or state.done:
        return False, "", state
    st = WaveAbortState(armed=state.armed, done=state.done)
    thr_p = float(cfg.thr_pos)
    thr_n = float(cfg.thr_neg)
    wait = float(cfg.max_wait_seconds)
    rev = float(cfg.revoke_seconds)
    held = float(held_seconds)

    if held > rev + 1e-9:
        st.done = True
        return False, "", st

    if cfg.opt_mtm_max is not None and opt_mtm is not None:
        if float(opt_mtm) <= float(cfg.opt_mtm_max) and float(stock_signed) < thr_p:
            # option already dead and stock not confirming — abort
            if not st.armed or held <= rev:
                return True, "neg", st

    if not st.armed:
        if float(stock_signed) <= thr_n:
            return True, "neg", st
        if float(stock_signed) >= thr_p:
            st.armed = True
            if not cfg.allow_revoke:
                st.done = True
            return False, "", st
        if held + 1e-9 >= wait:
            mode = str(cfg.on_timeout or "abort").strip().lower()
            if mode in {"allow", "pass", "keep"}:
                st.done = True
                return False, "", st
            return True, "timeout", st
        return False, "", st

    # armed — revocable
    if not cfg.allow_revoke:
        st.done = True
        return False, "", st
    thr_r = float(cfg.thr_neg_revoke) if cfg.thr_neg_revoke is not None else thr_n
    if float(stock_signed) <= thr_r:
        if cfg.revoke_opt_mtm_max is not None:
            if opt_mtm is None or float(opt_mtm) > float(cfg.revoke_opt_mtm_max):
                return False, "", st
        return True, "revoke", st
    return False, "", st

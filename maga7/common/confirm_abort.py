"""Post-fill option-mark confirm-or-abort (AM_EXT satellite).

After fill, require mark MTM >= ``confirm_thr`` within ``confirm_sec``.
Otherwise flatten (``confirm_abort``). Optional ``abort_thr`` kills early
before confirm. TP/SL remain live in the OMS caller.

Research:
- ``research_am_confirm_abort_20260728`` best ``ca_t60_c0.02_a0.08_ext_1025``
  (apply only near EXT window open).
- ``research_am_segB_both_ddctrl_20260728`` promote ``both08_ca_up_only``:
  B both-dir FO@0.8%, CA only on UP (``only_dirs=["UP"]``, no entry clock cut).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import pandas as pd

ConfirmAbortReason = Literal["confirm_abort", "early_abort", ""]


@dataclass
class ConfirmAbortConfig:
    enabled: bool = False
    confirm_sec: float = 60.0
    confirm_thr: float = 0.02
    abort_thr: float | None = 0.08
    on_timeout: str = "abort"  # abort | allow
    # If set (HH:MM NY), only arm for entries strictly before this clock.
    # Research PASS used EXT@10:25 → ``10:26``. None = all entries in lane.
    only_entry_before: str | None = "10:26"
    # If set, only apply to these trade directions (e.g. ("UP",) = ca_up_only).
    only_dirs: tuple[str, ...] | None = None


@dataclass
class ConfirmAbortState:
    confirmed: bool = False
    done: bool = False  # window closed without abort (allow path)


def confirm_abort_from_raw(raw: Any) -> ConfirmAbortConfig:
    if raw is None:
        return ConfirmAbortConfig(enabled=False)
    if isinstance(raw, bool):
        return ConfirmAbortConfig(enabled=bool(raw))
    if not isinstance(raw, dict):
        return ConfirmAbortConfig(enabled=False)
    ot = str(raw.get("on_timeout", "abort") or "abort").strip().lower()
    if ot in {"block", "fail"}:
        ot = "abort"
    if ot in {"pass", "keep"}:
        ot = "allow"
    abort_raw = raw.get("abort_thr", 0.08)
    abort_thr: float | None
    if abort_raw is None or str(abort_raw).strip().lower() in {"", "none", "null"}:
        abort_thr = None
    else:
        abort_thr = abs(float(abort_raw))
    before = raw.get("only_entry_before", "10:26")
    if before is None or str(before).strip().lower() in {"", "none", "null"}:
        before_s: str | None = None
    else:
        before_s = str(before).strip()
    dirs_raw = raw.get("only_dirs") or raw.get("dirs")
    only_dirs: tuple[str, ...] | None = None
    if isinstance(dirs_raw, str):
        only_dirs = tuple(x.strip().upper() for x in dirs_raw.split(",") if x.strip())
    elif isinstance(dirs_raw, (list, tuple)):
        only_dirs = tuple(str(x).strip().upper() for x in dirs_raw if str(x).strip())
    if only_dirs == ():
        only_dirs = None
    return ConfirmAbortConfig(
        enabled=bool(raw.get("enabled", False)),
        confirm_sec=float(
            raw.get("confirm_sec", 60.0) if raw.get("confirm_sec") is not None else 60.0
        ),
        confirm_thr=float(
            raw.get("confirm_thr", 0.02) if raw.get("confirm_thr") is not None else 0.02
        ),
        abort_thr=abort_thr,
        on_timeout=ot if ot in {"abort", "allow"} else "abort",
        only_entry_before=before_s,
        only_dirs=only_dirs,
    )


def confirm_abort_applies(
    cfg: ConfirmAbortConfig,
    entry_ts: float | pd.Timestamp,
    *,
    direction: str | None = None,
) -> bool:
    if not cfg.enabled:
        return False
    if cfg.only_dirs is not None:
        d = str(direction or "").strip().upper()
        if d not in set(cfg.only_dirs):
            return False
    if not cfg.only_entry_before:
        return True
    try:
        ny = pd.Timestamp(float(entry_ts), unit="s", tz="UTC").tz_convert("America/New_York")
    except Exception:
        ny = pd.Timestamp(entry_ts)
        if ny.tzinfo is None:
            ny = ny.tz_localize("America/New_York")
        else:
            ny = ny.tz_convert("America/New_York")
    parts = str(cfg.only_entry_before).split(":")
    cut = int(parts[0]) * 60 + int(parts[1])
    return (ny.hour * 60 + ny.minute) < cut


def confirm_abort_on_tick(
    st: ConfirmAbortState,
    *,
    cfg: ConfirmAbortConfig,
    held_seconds: float,
    opt_mtm: float,
) -> tuple[bool, ConfirmAbortReason, ConfirmAbortState]:
    """Return (do_abort, reason, state). No-op once confirmed or done."""
    if not cfg.enabled or st.done or st.confirmed:
        return False, "", st
    if not (opt_mtm == opt_mtm):  # NaN
        return False, "", st

    if cfg.abort_thr is not None and float(opt_mtm) <= -abs(float(cfg.abort_thr)):
        st.done = True
        return True, "early_abort", st

    if float(opt_mtm) >= float(cfg.confirm_thr):
        st.confirmed = True
        st.done = True
        return False, "", st

    if float(held_seconds) >= float(cfg.confirm_sec):
        if cfg.on_timeout == "allow":
            st.confirmed = True
            st.done = True
            return False, "", st
        st.done = True
        return True, "confirm_abort", st

    return False, "", st

"""NY session phase + tape path layout for Mag7 live validation.

PRE / POST ticks are validation-only: they prove IB→Redis→disk plumbing
without polluting the RTH authority stream consumed by Scanner/OMS.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from maga7.common.bar_agg import RTH_END, RTH_START, NY, in_rth, to_ny_ts

Phase = str  # "PRE" | "RTH" | "POST"


def session_phase(ts: Any, *, trade_date: str | None = None) -> Phase:
    """Classify wall/exchange time into PRE / RTH / POST for one trade date."""
    if isinstance(ts, (int, float)) and not isinstance(ts, bool):
        # Connector publish clock is unix seconds; pd.Timestamp(int) is ns.
        t = pd.Timestamp(float(ts), unit="s", tz="UTC").tz_convert(NY)
    else:
        t = to_ny_ts(ts)
    if trade_date:
        day = pd.Timestamp(f"{trade_date} 12:00:00", tz=NY).date()
        if t.date() < day:
            return "PRE"
        if t.date() > day:
            return "POST"
    clock = t.time()
    if clock < RTH_START:
        return "PRE"
    if clock >= RTH_END:
        return "POST"
    return "RTH"


def tape_root(session_dir: Path) -> Path:
    return Path(session_dir) / "tape"


def tape_phase_dir(session_dir: Path, phase: Phase) -> Path:
    name = {"PRE": "pre", "RTH": "rth", "POST": "post"}.get(str(phase).upper(), "other")
    return tape_root(session_dir) / name


def tape_symbol_path(
    session_dir: Path,
    *,
    phase: Phase,
    symbol: str,
    trade_date: str,
) -> Path:
    return tape_phase_dir(session_dir, phase) / f"{str(symbol).upper()}_{trade_date}.jsonl"


def is_rth_authority_phase(phase: Phase) -> bool:
    return str(phase).upper() == "RTH"


__all__ = [
    "NY",
    "Phase",
    "in_rth",
    "is_rth_authority_phase",
    "session_phase",
    "tape_phase_dir",
    "tape_root",
    "tape_symbol_path",
    "to_ny_ts",
]

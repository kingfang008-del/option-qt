"""Overnight gap trap / degrade gate (causal).

Default: large gap *aligned with trade direction* (fav_gap).
UP-only degrade: ``dirs=["UP"]`` + ``mode=scale`` → only gap-up chase longs.
Optional ``require_adv_share``: only act when pre-entry adverse vol share is hot
(separates Jul20/21 AMD tox from Jul1 META winner in probe).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class OvernightGapGateConfig:
    enabled: bool = False
    max_fav_gap: float = 0.04
    mode: str = "block"  # block | scale
    scale: float = 0.5
    on_missing: str = "allow"  # allow | block
    dirs: tuple[str, ...] | None = None  # None = both; ("UP",) = gap-up chase only
    require_adv_share: float | None = None
    on_missing_adv: str = "pass"  # pass | act
    # When set with require_adv_share, measure adv at ts+lag (Jul20 heats in ~60s).
    lag_seconds: int = 0


@dataclass(frozen=True)
class OvernightGapDecision:
    allow: bool
    size_scale: float
    reason: str
    gap: float | None = None
    fav_gap: float | None = None
    adv_share: float | None = None


def parse_overnight_gap_gate(raw: Any) -> OvernightGapGateConfig:
    if not isinstance(raw, dict):
        return OvernightGapGateConfig(enabled=False)
    mode = str(raw.get("mode") or "block").strip().lower()
    if mode in {"reject", "hard", "skip"}:
        mode = "block"
    if mode in {"soft", "size", "half", "degrade"}:
        mode = "scale"
    if mode not in {"block", "scale"}:
        mode = "block"
    on_miss = str(raw.get("on_missing") or "allow").strip().lower()
    if on_miss not in {"allow", "block"}:
        on_miss = "allow"
    sc = max(0.0, min(1.0, float(raw.get("scale", 0.5) or 0.5)))
    dirs_raw = raw.get("dirs") or raw.get("directions")
    if raw.get("up_only") in (True, 1, "1", "true", "True", "yes"):
        dirs_raw = ["UP"]
    dirs: tuple[str, ...] | None = None
    if isinstance(dirs_raw, str):
        dirs = tuple(x.strip().upper() for x in dirs_raw.split(",") if x.strip())
    elif isinstance(dirs_raw, (list, tuple)):
        dirs = tuple(str(x).strip().upper() for x in dirs_raw if str(x).strip())
    if dirs == ():
        dirs = None
    adv_req = raw.get("require_adv_share", raw.get("min_adv_share"))
    adv_f = float(adv_req) if adv_req is not None else None
    if adv_f is not None:
        adv_f = max(0.0, min(1.0, adv_f))
    on_miss_adv = str(raw.get("on_missing_adv") or "pass").strip().lower()
    if on_miss_adv in {"allow", "skip", "ignore"}:
        on_miss_adv = "pass"
    if on_miss_adv in {"hot", "trigger", "degrade"}:
        on_miss_adv = "act"
    if on_miss_adv not in {"pass", "act"}:
        on_miss_adv = "pass"
    lag = max(0, int(raw.get("lag_seconds", 0) or 0))
    return OvernightGapGateConfig(
        enabled=bool(raw.get("enabled", False)),
        max_fav_gap=float(raw.get("max_fav_gap", raw.get("min_gap_up", 0.04)) or 0.04),
        mode=mode,
        scale=sc,
        on_missing=on_miss,
        dirs=dirs,
        require_adv_share=adv_f,
        on_missing_adv=on_miss_adv,
        lag_seconds=lag,
    )


def overnight_gap(
    stock_df: pd.DataFrame | None,
    *,
    date: str,
) -> float | None:
    """``day_open / prev_close - 1`` (causal at open)."""
    if stock_df is None or stock_df.empty:
        return None
    if "date" not in stock_df.columns or "open" not in stock_df.columns:
        return None
    day = stock_df[stock_df["date"].astype(str) == str(date)].sort_values("timestamp")
    if day.empty:
        return None
    prev = stock_df[stock_df["date"].astype(str) < str(date)].sort_values("timestamp")
    if prev.empty or "close" not in prev.columns:
        return None
    try:
        o = float(day.iloc[0]["open"])
        pc = float(prev.iloc[-1]["close"])
    except (TypeError, ValueError, IndexError):
        return None
    if o <= 0 or pc <= 0:
        return None
    return float(o / pc - 1.0)


def resolve_overnight_gap_gate(
    cfg: OvernightGapGateConfig,
    *,
    stock_df: pd.DataFrame | None,
    date: str,
    direction: str,
    adv_share: float | None = None,
) -> OvernightGapDecision:
    if not cfg.enabled:
        return OvernightGapDecision(True, 1.0, "off")
    d = str(direction or "").upper()
    if cfg.dirs is not None and d not in set(cfg.dirs):
        return OvernightGapDecision(True, 1.0, "dir_skip")
    gap = overnight_gap(stock_df, date=str(date))
    if gap is None:
        if cfg.on_missing == "block":
            return OvernightGapDecision(False, 0.0, "missing_gap")
        return OvernightGapDecision(True, 1.0, "missing_allow")
    fav = float(gap) if d == "UP" else float(-gap)
    adv_out = float(adv_share) if adv_share is not None and np.isfinite(adv_share) else None
    if fav + 1e-12 < float(cfg.max_fav_gap):
        return OvernightGapDecision(
            True, 1.0, "pass", gap=float(gap), fav_gap=fav, adv_share=adv_out
        )
    if cfg.require_adv_share is not None:
        if adv_out is None:
            if cfg.on_missing_adv == "pass":
                return OvernightGapDecision(
                    True,
                    1.0,
                    "gap_hot_adv_missing_pass",
                    gap=float(gap),
                    fav_gap=fav,
                    adv_share=None,
                )
        elif adv_out + 1e-12 < float(cfg.require_adv_share):
            return OvernightGapDecision(
                True,
                1.0,
                "gap_hot_adv_cool",
                gap=float(gap),
                fav_gap=fav,
                adv_share=adv_out,
            )
    if cfg.mode == "scale":
        return OvernightGapDecision(
            True,
            float(cfg.scale),
            f"degrade_gap>={cfg.max_fav_gap:g}",
            gap=float(gap),
            fav_gap=fav,
            adv_share=adv_out,
        )
    return OvernightGapDecision(
        False,
        0.0,
        f"block_gap>={cfg.max_fav_gap:g}",
        gap=float(gap),
        fav_gap=fav,
        adv_share=adv_out,
    )

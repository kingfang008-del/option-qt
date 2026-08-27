"""Causal CORE climate prior — soft size scale only (C2).

Uses 10:30-asof features (same family as ``router_dataset_v2``):
  - VIXY z high
  - Mag7 breadth mid (frac above open in a dead-zone)

Does **not** emit direction, does **not** block Rule-A, does **not** use
calendar labels as a live gate. Missing features → passthrough.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_ROUTER = Path("maga7/results/regime_router/router_dataset_v2.parquet")


@dataclass(frozen=True)
class ClimatePriorConfig:
    enabled: bool = False
    scale: float = 0.5
    vixy_z_min: float | None = 1.0
    breadth_mid_lo: float | None = 0.375
    breadth_mid_hi: float | None = 0.625
    combine: str = "or"  # or | and
    use_vixy: bool = True
    use_breadth_mid: bool = True
    missing: str = "passthrough"  # passthrough | skip
    dataset: str | None = None


def parse_climate_prior(raw: Any) -> ClimatePriorConfig:
    if not isinstance(raw, dict):
        return ClimatePriorConfig(enabled=False)
    vz = raw.get("vixy_z_min", 1.0)
    lo = raw.get("breadth_mid_lo", 0.375)
    hi = raw.get("breadth_mid_hi", 0.625)
    return ClimatePriorConfig(
        enabled=bool(raw.get("enabled", False)),
        scale=float(raw.get("scale", 0.5) or 0.5),
        vixy_z_min=(None if vz is None else float(vz)),
        breadth_mid_lo=(None if lo is None else float(lo)),
        breadth_mid_hi=(None if hi is None else float(hi)),
        combine=str(raw.get("combine") or "or").strip().lower(),
        use_vixy=bool(raw.get("use_vixy", True)),
        use_breadth_mid=bool(raw.get("use_breadth_mid", True)),
        missing=str(raw.get("missing") or "passthrough").strip().lower(),
        dataset=(str(raw["dataset"]) if raw.get("dataset") else None),
    )


def load_climate_day_table(
    path: Path | str | None,
    *,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame | None:
    p = Path(path or DEFAULT_ROUTER).expanduser()
    if not p.is_file():
        return None
    df = pd.read_parquet(p)
    if df is None or df.empty or "date" not in df.columns:
        return None
    out = df.copy()
    out["date"] = out["date"].astype(str)
    if start:
        out = out[out["date"] >= str(start)]
    if end:
        out = out[out["date"] <= str(end)]
    keep = [
        c
        for c in (
            "date",
            "vixy_z_1030",
            "mag7_frac_above_open",
            "breadth_dn_frac",
            "qqq_from_prev_1030",
        )
        if c in out.columns
    ]
    return out[keep].drop_duplicates("date") if keep else None


def _hit_vixy(cfg: ClimatePriorConfig, row: pd.Series) -> bool | None:
    if not cfg.use_vixy or cfg.vixy_z_min is None:
        return False
    z = pd.to_numeric(row.get("vixy_z_1030"), errors="coerce")
    if z is None or not np.isfinite(z):
        return None
    return bool(float(z) >= float(cfg.vixy_z_min))


def _hit_breadth_mid(cfg: ClimatePriorConfig, row: pd.Series) -> bool | None:
    if not cfg.use_breadth_mid:
        return False
    if cfg.breadth_mid_lo is None or cfg.breadth_mid_hi is None:
        return False
    br = pd.to_numeric(row.get("mag7_frac_above_open"), errors="coerce")
    if br is None or not np.isfinite(br):
        return None
    return bool(float(cfg.breadth_mid_lo) < float(br) < float(cfg.breadth_mid_hi))


def resolve_climate_prior(
    cfg: ClimatePriorConfig,
    *,
    date: str,
    day_table: pd.DataFrame | None,
) -> tuple[float, str]:
    """Return ``(size_mult, reason)``. ``0`` means skip (only if missing=skip)."""
    if not cfg.enabled:
        return 1.0, "climate_off"
    sc = max(0.0, min(1.0, float(cfg.scale)))
    if day_table is None or getattr(day_table, "empty", True):
        if cfg.missing == "skip":
            return 0.0, "climate_missing_skip"
        return 1.0, "climate_missing_passthrough"
    hit = day_table[day_table["date"].astype(str) == str(date)]
    if hit.empty:
        if cfg.missing == "skip":
            return 0.0, "climate_missing_skip"
        return 1.0, "climate_missing_passthrough"
    row = hit.iloc[0]
    flags: list[str] = []
    vixy = _hit_vixy(cfg, row)
    br = _hit_breadth_mid(cfg, row)
    known = []
    if cfg.use_vixy:
        if vixy is None:
            known.append(None)
        else:
            known.append(bool(vixy))
            if vixy:
                flags.append("vixy_high")
    if cfg.use_breadth_mid:
        if br is None:
            known.append(None)
        else:
            known.append(bool(br))
            if br:
                flags.append("breadth_mid")
    if any(x is None for x in known) and not flags:
        if cfg.missing == "skip":
            return 0.0, "climate_missing_skip"
        return 1.0, "climate_missing_passthrough"
    present = [bool(x) for x in known if x is not None]
    if not present:
        return 1.0, "climate_ok"
    if cfg.combine == "and":
        fire = all(present) and (None not in known)
    else:
        fire = any(present)
    if fire and sc < 1.0 - 1e-12:
        tag = "+".join(flags) or "hit"
        return sc, f"climate_scale:{tag}:{sc:.2f}"
    return 1.0, "climate_ok"

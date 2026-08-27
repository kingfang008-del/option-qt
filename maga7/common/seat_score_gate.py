"""Seat quality gate: low-score first fires do not consume TopK slots.

Walk ``all_first`` in time order; only candidates that pass the score may
reserve a daily seat (cap = ``signal.top_k``). Regime/peer blocks already
skip without consuming when slot accounting is on; this gate adds a
liquidity/quality skip of the same kind.

Narrow triggers (``when``):
  - ``always``: day armed every session
  - ``topk_weak``: arm only if some earliest-TopK fire fails the score at its own ts
  - ``morning`` / ``topk_weak_morning``: score only inside ``tod_start``–``tod_end``

``apply_to``:
  - ``all``: every candidate scored when active (harsh)
  - ``topk_members``: only earliest-TopK members can be rejected; later
    all_first backfills always pass (true “丢弃前两枪、后面补位”)

Does not force-close open positions (unlike ``displace_on_later``).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from maga7.common.signals import _cs_dollar_vol_rank


@dataclass(frozen=True)
class SeatScoreGateConfig:
    enabled: bool = False
    mode: str = "cs_dvol_max_rank"  # cs_dvol_max_rank | vol_z_min | fp_x_vz_min
    max_rank: int = 2
    min_vol_z: float = 1.5
    min_fp_x_vz: float = 0.03
    when: str = "always"  # always | topk_weak | morning | topk_weak_morning
    tod_start: str = "10:30"
    tod_end: str = "11:30"
    topk_weak_policy: str = "any"  # any | all
    apply_to: str = "all"  # all | topk_members


def parse_seat_score_gate(raw: Any) -> SeatScoreGateConfig:
    if not isinstance(raw, dict):
        return SeatScoreGateConfig(enabled=False)
    mode = str(raw.get("mode") or "cs_dvol_max_rank").strip().lower()
    when = str(raw.get("when") or "always").strip().lower()
    pol = str(raw.get("topk_weak_policy") or "any").strip().lower()
    apply_to = str(raw.get("apply_to") or "all").strip().lower()
    if apply_to in {"topk", "topk_only", "earliest_topk"}:
        apply_to = "topk_members"
    if apply_to not in {"all", "topk_members"}:
        apply_to = "all"
    return SeatScoreGateConfig(
        enabled=bool(raw.get("enabled", False)),
        mode=mode,
        max_rank=int(raw.get("max_rank", 2) or 2),
        min_vol_z=float(raw.get("min_vol_z", 1.5) or 1.5),
        min_fp_x_vz=float(raw.get("min_fp_x_vz", 0.03) or 0.03),
        when=when,
        tod_start=str(raw.get("tod_start") or "10:30"),
        tod_end=str(raw.get("tod_end") or "11:30"),
        topk_weak_policy=pol if pol in {"any", "all"} else "any",
        apply_to=apply_to,
    )


def _tod_hhmm(ts: pd.Timestamp) -> str:
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize("America/New_York")
    else:
        t = t.tz_convert("America/New_York")
    return f"{int(t.hour):02d}:{int(t.minute):02d}"


def _in_tod_window(ts: pd.Timestamp, start: str, end: str) -> bool:
    tod = _tod_hhmm(ts)
    return str(start) <= tod <= str(end)


def _vol_z_at(
    stock_by: dict[str, pd.DataFrame],
    *,
    symbol: str,
    date: str,
    asof_ts: pd.Timestamp,
) -> float | None:
    sdf = stock_by.get(str(symbol).upper()) or stock_by.get(str(symbol))
    if sdf is None or sdf.empty or "vol_z" not in sdf.columns:
        return None
    day = sdf[sdf["date"].astype(str) == str(date)]
    if day.empty:
        return None
    asof = pd.Timestamp(asof_ts)
    if asof.tzinfo is None:
        asof = asof.tz_localize("America/New_York")
    else:
        asof = asof.tz_convert("America/New_York")
    up = day[day["timestamp"] <= asof]
    if up.empty:
        return None
    v = up.iloc[-1].get("vol_z")
    try:
        return float(v) if pd.notna(v) else None
    except (TypeError, ValueError):
        return None


def seat_score_ok(
    cfg: SeatScoreGateConfig,
    *,
    stock_by: dict[str, pd.DataFrame],
    symbol: str,
    date: str,
    asof_ts: pd.Timestamp,
    from_prev: float | None = None,
    vol_z: float | None = None,
) -> tuple[bool, str, float | None]:
    """Return ``(ok, reason, score)``. ``ok=False`` → skip without consuming seat."""
    if not cfg.enabled:
        return True, "off", None
    mode = str(cfg.mode or "").strip().lower()
    if mode in {"cs_dvol_max_rank", "cs_dvol", "cs_rank", "liquidity"}:
        rank, dvol = _cs_dollar_vol_rank(
            stock_by, date=str(date), asof_ts=asof_ts, symbol=str(symbol)
        )
        if rank is None:
            return False, "cs_dvol_missing", None
        ok = int(rank) <= int(cfg.max_rank)
        return ok, (f"cs_rk{rank}" if ok else f"cs_rk{rank}>max{cfg.max_rank}"), float(rank)
    if mode in {"vol_z_min", "vol_z"}:
        if vol_z is None or not pd.notna(vol_z):
            return False, "vol_z_missing", None
        vz = float(vol_z)
        ok = vz >= float(cfg.min_vol_z)
        return ok, (f"vol_z={vz:.3f}" if ok else f"vol_z<{cfg.min_vol_z}"), vz
    if mode in {"fp_x_vz_min", "fp_x_vz", "fp*vz"}:
        if from_prev is None or vol_z is None or not pd.notna(from_prev) or not pd.notna(vol_z):
            return False, "fp_vz_missing", None
        sc = abs(float(from_prev)) * float(vol_z)
        ok = sc >= float(cfg.min_fp_x_vz)
        return ok, (f"fp_vz={sc:.4f}" if ok else f"fp_vz<{cfg.min_fp_x_vz}"), sc
    return True, f"unknown_mode:{mode}", None


def day_gate_armed(
    cfg: SeatScoreGateConfig,
    *,
    topk_day: pd.DataFrame,
    stock_by: dict[str, pd.DataFrame],
    date: str,
) -> tuple[bool, str]:
    """Whether the seat gate is armed for ``date`` under ``cfg.when``."""
    if not cfg.enabled:
        return False, "off"
    when = str(cfg.when or "always").strip().lower()
    if when in {"", "always", "all"}:
        return True, "always"
    needs_weak = when in {"topk_weak", "weak_topk", "topk_weak_morning", "morning_topk_weak"}
    if needs_weak:
        if topk_day is None or getattr(topk_day, "empty", True):
            return False, "topk_empty"
        fails = 0
        n = 0
        for r in topk_day.itertuples(index=False):
            n += 1
            fp = float(r.from_prev) if hasattr(r, "from_prev") and pd.notna(r.from_prev) else None
            vz = None
            if hasattr(r, "vol_z") and pd.notna(getattr(r, "vol_z")):
                try:
                    vz = float(r.vol_z)
                except (TypeError, ValueError):
                    vz = None
            if vz is None:
                vz = _vol_z_at(
                    stock_by, symbol=str(r.symbol), date=str(date), asof_ts=r.sig_ts
                )
            ok, _, _ = seat_score_ok(
                cfg,
                stock_by=stock_by,
                symbol=str(r.symbol),
                date=str(date),
                asof_ts=r.sig_ts,
                from_prev=fp,
                vol_z=vz,
            )
            if not ok:
                fails += 1
        if n <= 0:
            return False, "topk_empty"
        if cfg.topk_weak_policy == "all":
            armed = fails >= n
        else:
            armed = fails >= 1
        if not armed:
            return False, f"topk_strong fails={fails}/{n}"
        if when in {"topk_weak", "weak_topk"}:
            return True, f"topk_weak fails={fails}/{n}"
        # topk_weak_morning: armed flag set; per-signal morning filter applied later
        return True, f"topk_weak_morning fails={fails}/{n}"
    if when in {"morning", "am", "tod"}:
        return True, "morning"
    return True, f"unknown_when:{when}"


def candidate_gate_active(
    cfg: SeatScoreGateConfig,
    *,
    day_armed: bool,
    asof_ts: pd.Timestamp,
    is_topk_member: bool = True,
) -> bool:
    """Whether to apply ``seat_score_ok`` to this candidate given day arming + TOD."""
    if not cfg.enabled or not day_armed:
        return False
    if str(cfg.apply_to or "all") == "topk_members" and not is_topk_member:
        return False
    when = str(cfg.when or "always").strip().lower()
    if when in {"morning", "am", "tod", "topk_weak_morning", "morning_topk_weak"}:
        return _in_tod_window(asof_ts, cfg.tod_start, cfg.tod_end)
    return True

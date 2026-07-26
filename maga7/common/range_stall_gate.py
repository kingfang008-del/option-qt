"""Range-chase + pre-entry stall gate (causal).

``chase`` = session range position aligned with trade direction:
  - UP: ``(px-lo)/(hi-lo)``  (near day high)
  - DN: ``(hi-px)/(hi-lo)``  (near day low)

``pre5`` = favorable return over the prior ``pre_seconds`` into the signal clock.

Probe on remaining ≤−3% days: losers often sit at chase≥0.9 with flat/negative
pre5; winners with high chase usually still have positive pre5. Rule
``chase≥0.9 & pre5≤0`` proxy: weak≈1.04 / strong≈1.01 / zero false TP wins.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class RangeStallGateConfig:
    enabled: bool = False
    min_chase: float = 0.9
    max_pre5: float = 0.0
    pre_seconds: int = 300
    mode: str = "block"  # block | scale
    scale: float = 0.5
    dirs: tuple[str, ...] | None = None
    # If set, chase arm only fires when peer_align <= max_peer.
    max_peer: int | None = None
    # Optional second arm: peer_align <= peer_pre5_max_peer AND pre5 <= max_pre5
    # (no chase requirement) — covers 03-16 TSLA-style weak-peer stalls.
    peer_pre5_max_peer: int | None = None
    min_fav_from_open: float | None = None
    # Optional third arm: unanimous/crowded tape chase stall.
    # peer_align >= crowd_min_peer AND chase>=min_chase AND pre5<=crowd_max_pre5
    # AND fav_from_open>=crowd_min_fav_from_open (else min_fav_from_open). Covers
    # 02-06/02-25 peer=7; crowd fo can be looser than Arm A (02-18/03-12).
    crowd_min_peer: int | None = None
    crowd_max_pre5: float | None = None
    crowd_min_fav_from_open: float | None = None
    on_missing: str = "allow"  # allow | block


@dataclass(frozen=True)
class RangeStallDecision:
    allow: bool
    size_scale: float
    reason: str
    chase: float | None = None
    pre5: float | None = None
    fav_from_open: float | None = None


def parse_range_stall_gate(raw: Any) -> RangeStallGateConfig:
    if not isinstance(raw, dict):
        return RangeStallGateConfig(enabled=False)
    mode = str(raw.get("mode") or "block").strip().lower()
    if mode in {"reject", "hard", "skip"}:
        mode = "block"
    if mode in {"soft", "size", "half", "degrade"}:
        mode = "scale"
    if mode not in {"block", "scale"}:
        mode = "block"
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
    max_peer = raw.get("max_peer")
    max_peer_i = int(max_peer) if max_peer is not None else None
    pp = raw.get("peer_pre5_max_peer")
    pp_i = int(pp) if pp is not None else None
    mfo = raw.get("min_fav_from_open")
    mfo_f = float(mfo) if mfo is not None else None
    cmin = raw.get("crowd_min_peer")
    cmin_i = int(cmin) if cmin is not None else None
    cpre = raw.get("crowd_max_pre5")
    cpre_f = float(cpre) if cpre is not None else None
    cmfo = raw.get("crowd_min_fav_from_open")
    cmfo_f = float(cmfo) if cmfo is not None else None
    on_miss = str(raw.get("on_missing") or "allow").strip().lower()
    if on_miss not in {"allow", "block"}:
        on_miss = "allow"
    return RangeStallGateConfig(
        enabled=bool(raw.get("enabled", False)),
        min_chase=float(raw.get("min_chase", 0.9) or 0.9),
        max_pre5=float(raw.get("max_pre5", 0.0) or 0.0),
        pre_seconds=max(30, int(raw.get("pre_seconds", 300) or 300)),
        mode=mode,
        scale=max(0.0, min(1.0, float(raw.get("scale", 0.5) or 0.5))),
        dirs=dirs,
        max_peer=max_peer_i,
        peer_pre5_max_peer=pp_i,
        min_fav_from_open=mfo_f,
        crowd_min_peer=cmin_i,
        crowd_max_pre5=cpre_f,
        crowd_min_fav_from_open=cmfo_f,
        on_missing=on_miss,
    )


def session_chase_and_pre5(
    stock_df: pd.DataFrame | None,
    *,
    date: str,
    asof_ts: pd.Timestamp,
    direction: str,
    pre_seconds: int = 300,
) -> tuple[float | None, float | None, float | None]:
    """Return ``(chase, pre5_fav, from_open)`` or Nones if bars missing."""
    if stock_df is None or stock_df.empty:
        return None, None, None
    if "date" not in stock_df.columns or "timestamp" not in stock_df.columns:
        return None, None, None
    day = stock_df[stock_df["date"].astype(str) == str(date)].sort_values("timestamp")
    if day.empty:
        return None, None, None
    asof = pd.Timestamp(asof_ts)
    tz = day["timestamp"].dt.tz
    if asof.tzinfo is None and tz is not None:
        asof = asof.tz_localize(tz)
    elif asof.tzinfo is not None and tz is None:
        asof = asof.tz_localize(None)
    else:
        try:
            asof = asof.tz_convert(tz)
        except (TypeError, ValueError):
            pass
    before = day[day["timestamp"] <= asof]
    if before.empty:
        return None, None, None
    try:
        day_open = float(day.iloc[0]["open"])
        px = float(before.iloc[-1]["close"])
        hi = float(before["high"].max()) if "high" in before.columns else px
        lo = float(before["low"].min()) if "low" in before.columns else px
    except (TypeError, ValueError, IndexError):
        return None, None, None
    if not (day_open > 0 and px > 0):
        return None, None, None
    from_open = px / day_open - 1.0
    d = str(direction or "").upper()
    if hi > lo:
        rng = (px - lo) / (hi - lo)
    else:
        rng = 0.5
    chase = float(rng) if d == "UP" else float(1.0 - rng)
    t0 = asof - pd.Timedelta(seconds=int(pre_seconds))
    win = before[before["timestamp"] >= t0]
    pre5 = None
    if len(win) >= 2:
        try:
            p0 = float(win.iloc[0]["close"])
            p1 = float(win.iloc[-1]["close"])
            if p0 > 0:
                r = p1 / p0 - 1.0
                pre5 = float(r if d == "UP" else -r)
        except (TypeError, ValueError):
            pre5 = None
    return chase, pre5, float(from_open)


def resolve_range_stall_gate(
    cfg: RangeStallGateConfig,
    *,
    stock_df: pd.DataFrame | None,
    date: str,
    asof_ts: pd.Timestamp,
    direction: str,
    peer_n: int | None = None,
) -> RangeStallDecision:
    if not cfg.enabled:
        return RangeStallDecision(True, 1.0, "off")
    d = str(direction or "").upper()
    if cfg.dirs is not None and d not in set(cfg.dirs):
        return RangeStallDecision(True, 1.0, "dir_skip")
    chase, pre5, from_open = session_chase_and_pre5(
        stock_df,
        date=str(date),
        asof_ts=asof_ts,
        direction=d,
        pre_seconds=int(cfg.pre_seconds),
    )
    fav_fo = None
    if from_open is not None:
        fav_fo = float(from_open) if d == "UP" else float(-from_open)
    if chase is None or pre5 is None:
        if cfg.on_missing == "block":
            return RangeStallDecision(False, 0.0, "missing_bars")
        return RangeStallDecision(True, 1.0, "missing_allow")

    def _fire(reason: str) -> RangeStallDecision:
        if cfg.mode == "scale":
            return RangeStallDecision(
                True,
                float(cfg.scale),
                f"degrade_{reason}",
                chase=float(chase),
                pre5=float(pre5),
                fav_from_open=fav_fo,
            )
        return RangeStallDecision(
            False,
            0.0,
            f"block_{reason}",
            chase=float(chase),
            pre5=float(pre5),
            fav_from_open=fav_fo,
        )

    # Arm B: weak peer + stalled pre5 (no chase requirement).
    if cfg.peer_pre5_max_peer is not None and peer_n is not None:
        if int(peer_n) <= int(cfg.peer_pre5_max_peer) and pre5 - 1e-12 <= float(cfg.max_pre5):
            return _fire(f"peer<={cfg.peer_pre5_max_peer}&pre5<={cfg.max_pre5:g}")

    # Arm C: crowded/unanimous peer + chase stall (looser pre5 than Arm A).
    if cfg.crowd_min_peer is not None and peer_n is not None:
        crowd_pre = (
            float(cfg.crowd_max_pre5)
            if cfg.crowd_max_pre5 is not None
            else float(cfg.max_pre5)
        )
        crowd_ffo = (
            float(cfg.crowd_min_fav_from_open)
            if cfg.crowd_min_fav_from_open is not None
            else (float(cfg.min_fav_from_open) if cfg.min_fav_from_open is not None else None)
        )
        if (
            int(peer_n) >= int(cfg.crowd_min_peer)
            and chase + 1e-12 >= float(cfg.min_chase)
            and pre5 - 1e-12 <= crowd_pre
        ):
            if crowd_ffo is None or (
                fav_fo is not None and fav_fo + 1e-12 >= float(crowd_ffo)
            ):
                return _fire(
                    f"crowd>={cfg.crowd_min_peer}&chase>={cfg.min_chase:g}&pre5<={crowd_pre:g}"
                )

    # Arm A: range chase + stalled pre5.
    if cfg.max_peer is not None:
        if peer_n is None or int(peer_n) > int(cfg.max_peer):
            return RangeStallDecision(
                True, 1.0, "peer_skip", chase=float(chase), pre5=float(pre5), fav_from_open=fav_fo
            )
    if chase + 1e-12 < float(cfg.min_chase):
        return RangeStallDecision(
            True, 1.0, "chase_low", chase=float(chase), pre5=float(pre5), fav_from_open=fav_fo
        )
    if pre5 - 1e-12 > float(cfg.max_pre5):
        return RangeStallDecision(
            True, 1.0, "pre5_ok", chase=float(chase), pre5=float(pre5), fav_from_open=fav_fo
        )
    if cfg.min_fav_from_open is not None:
        if fav_fo is None or fav_fo + 1e-12 < float(cfg.min_fav_from_open):
            return RangeStallDecision(
                True,
                1.0,
                "ffo_short",
                chase=float(chase),
                pre5=float(pre5),
                fav_from_open=fav_fo,
            )
    return _fire(f"chase>={cfg.min_chase:g}&pre5<={cfg.max_pre5:g}")


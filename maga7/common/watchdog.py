"""Regime Watchdog — architecture layer above freeze baseline.

States
------
NORMAL   : baseline only (read-only freeze path)
DEGRADE  : apply soft expert overlay (size / scale) — protect baseline
HALT     : hard block entries (narrow toxic tape)
HUNT     : short-TTL opportunity slot (default empty / off)

Design rules
------------
1. Baseline engine is never rewritten — overlays only patch Mag7RegimeGate.cfg.
2. Priority: HALT > DEGRADE > HUNT > NORMAL.
3. HUNT cannot arm from HALT; by default HUNT is blocked while DEGRADE is active.
4. Every non-NORMAL state carries a TTL; expiry returns to NORMAL.
"""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from maga7.common.orb_open import (
    OrbOpenConfig,
    OrbSignal,
    WashoutReclaimConfig,
    count_open_washout,
    scan_orb_day,
    scan_washout_reclaim_day,
)
from maga7.common.signals import first_rule_a_day

NY = "America/New_York"

# Regime keys experts may temporarily override (must match replay).
WATCHDOG_REGIME_KEYS = (
    "scale_dn_if_qqq_above_open",
    "block_dn_if_qqq_above_open",
    "direction_size_scale",
    "block_directions",
)


class WatchdogState(str, Enum):
    NORMAL = "normal"
    DEGRADE = "degrade"
    HALT = "halt"
    HUNT = "hunt"


@dataclass(frozen=True)
class Overlay:
    """Patch applied on top of baseline regime cfg for the armed window."""

    expert_name: str | None = None
    regime_patch: dict[str, Any] = field(default_factory=dict)
    allow_baseline: bool = True
    allow_hunt: bool = False
    route_tag: str = "baseline"


@dataclass
class WatchdogDecision:
    state: WatchdogState
    overlay: Overlay
    reason: str
    asof: pd.Timestamp | None = None
    armed_until: pd.Timestamp | None = None
    expert: str | None = None

    def active_at(self, ts: pd.Timestamp) -> bool:
        if self.state == WatchdogState.NORMAL:
            return True
        if self.armed_until is None:
            return True
        return pd.Timestamp(ts) <= pd.Timestamp(self.armed_until)


@dataclass(frozen=True)
class HuntCandidate:
    """Short-window opportunity armed by Watchdog (not a Rule-A rewrite)."""

    symbol: str
    date: str
    direction: str
    sig_ts: pd.Timestamp
    armed_until: pd.Timestamp
    detector: str
    reason: str
    fractal_high: float | None = None
    wash_drop: float | None = None


def _session_upto(sdf: pd.DataFrame | None, date: str, asof: pd.Timestamp) -> pd.DataFrame:
    if sdf is None or sdf.empty:
        return pd.DataFrame()
    day = sdf[sdf["date"].astype(str) == str(date)].copy()
    if day.empty:
        return day
    ts = pd.to_datetime(day["timestamp"])
    if getattr(ts.dt, "tz", None) is None:
        ts = ts.dt.tz_localize("UTC").dt.tz_convert(NY)
    else:
        ts = ts.dt.tz_convert(NY)
    day["_ts"] = ts
    return day[day["_ts"] <= asof].sort_values("_ts")


def _prev_close(sdf: pd.DataFrame | None, date: str) -> float | None:
    if sdf is None or sdf.empty:
        return None
    prev = sdf[sdf["date"].astype(str) < str(date)]
    if prev.empty:
        return None
    last_date = str(prev["date"].astype(str).max())
    day = prev[prev["date"].astype(str) == last_date]
    if day.empty:
        return None
    px = float(day.sort_values("timestamp").iloc[-1]["close"])
    return px if np.isfinite(px) and px > 0 else None


def eval_router_rule(
    rule_name: str,
    *,
    date: str,
    stock_by: dict[str, pd.DataFrame],
    qqq_df: pd.DataFrame | None,
    symbols: list[str],
    asof_hhmm: str = "10:30",
    router_cfg: dict[str, Any] | None = None,
) -> str | None:
    """Causal morning rule → expert name or None (baseline).

    Shared by Watchdog and legacy ``regime_router`` path.
    """
    rule = str(rule_name or "").strip().lower()
    cfg = router_cfg if isinstance(router_cfg, dict) else {}
    asof = pd.Timestamp(f"{date} {asof_hhmm}", tz=NY)
    q_upto = _session_upto(qqq_df, date, asof)
    if q_upto.empty:
        return None
    q_open = float(q_upto.iloc[0]["open"] if "open" in q_upto.columns else q_upto.iloc[0]["close"])
    q_px = float(q_upto.iloc[-1]["close"])
    q_lo = float(
        pd.to_numeric(q_upto["low"] if "low" in q_upto.columns else q_upto["close"], errors="coerce").min()
    )
    if not np.isfinite(q_open) or q_open <= 0 or not np.isfinite(q_px) or not np.isfinite(q_lo) or q_lo <= 0:
        return None
    prev_c = _prev_close(qqq_df, date)
    open_vs_prev = (q_open / prev_c - 1.0) if prev_c else 0.0
    bounce = q_px / q_lo - 1.0
    above_open = q_px > q_open
    low_open_reclaim = bool(open_vs_prev < 0 and above_open)

    above_flags: list[float] = []
    for sym in symbols:
        upto = _session_upto(stock_by.get(sym), date, asof)
        if upto.empty:
            continue
        o = float(upto.iloc[0]["open"] if "open" in upto.columns else upto.iloc[0]["close"])
        px = float(upto.iloc[-1]["close"])
        if o > 0 and np.isfinite(px):
            above_flags.append(1.0 if px > o else 0.0)
    mag7_frac_above = float(np.mean(above_flags)) if above_flags else 1.0

    def _reclaim_hit(*, bounce_min: float, frac_max: float) -> bool:
        return bool(low_open_reclaim and bounce >= bounce_min and mag7_frac_above <= frac_max + 1e-12)

    def _washout_hit() -> bool:
        breadth_min = int(cfg.get("washout_breadth_min", 3) or 3)
        orb_cfg = OrbOpenConfig(
            wash_window_end=str(cfg.get("wash_window_end", "10:00")),
            wash_drop_min=float(cfg.get("wash_drop_min", 0.003) or 0.003),
            wash_min_bars=int(cfg.get("wash_min_bars", 3) or 3),
        )
        n, _ = count_open_washout(stock_by, date=date, symbols=symbols, cfg=orb_cfg)
        return n >= breadth_min

    washout_expert = str(cfg.get("washout_expert") or "washout_gate_dn").strip()

    if rule in {"reclaim_disp55", "low_open_reclaim_disp55"}:
        if _reclaim_hit(bounce_min=0.008, frac_max=0.55):
            return "rebound_trap_dn"
        return None
    if rule in {"reclaim_b015_disp", "reclaim_bounce015_disp65"}:
        if _reclaim_hit(bounce_min=0.015, frac_max=0.65):
            return "rebound_trap_dn"
        return None
    if rule in {"washout_breadth3", "washout_b3", "open_washout_breadth3", "washout_breadth"}:
        return washout_expert if _washout_hit() else None
    if rule in {"washout_and_reclaim", "washout_breadth_and_reclaim", "washout_reclaim"}:
        if _washout_hit() and _reclaim_hit(bounce_min=0.008, frac_max=0.55):
            return washout_expert
        return None
    if rule in {"washout_or_reclaim", "washout_breadth3_or_reclaim"}:
        if _washout_hit():
            return washout_expert
        if _reclaim_hit(bounce_min=0.008, frac_max=0.55):
            return "rebound_trap_dn"
        return None
    if rule in {"reclaim_or_washout", "reclaim_then_washout"}:
        if _reclaim_hit(bounce_min=0.008, frac_max=0.55):
            return "rebound_trap_dn"
        if _washout_hit():
            return washout_expert
        return None
    return None


def snapshot_regime(cfg: dict[str, Any]) -> dict[str, Any]:
    return {k: copy.deepcopy(cfg[k]) for k in WATCHDOG_REGIME_KEYS if k in cfg}


def restore_regime(cfg: dict[str, Any], snap: dict[str, Any]) -> None:
    for k in WATCHDOG_REGIME_KEYS:
        if k in snap:
            cfg[k] = copy.deepcopy(snap[k])
        elif k in cfg:
            del cfg[k]


def apply_overlay(cfg: dict[str, Any], overlay: Overlay | None) -> None:
    if not overlay or not overlay.regime_patch:
        return
    for k, v in overlay.regime_patch.items():
        cfg[k] = copy.deepcopy(v)


def apply_expert_dict(cfg: dict[str, Any], expert: dict[str, Any] | None) -> None:
    if not expert:
        return
    for k, v in (expert.get("regime") or {}).items():
        cfg[k] = copy.deepcopy(v)


def _load_experts(path: str | None, inline: dict | None) -> dict[str, dict]:
    experts: dict[str, dict] = {}
    if path:
        p = Path(str(path)).expanduser()
        if p.is_file():
            experts = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(inline, dict):
        experts.update(inline)
    return experts


def _ttl_until(asof: pd.Timestamp, ttl_minutes: int | None) -> pd.Timestamp | None:
    if ttl_minutes is None:
        return None  # rest of session / day-scoped in offline replay
    return asof + pd.Timedelta(minutes=int(ttl_minutes))


@dataclass
class WatchdogConfig:
    enabled: bool = False
    asof: str = "10:30"
    experts_path: str | None = None
    experts: dict[str, dict] = field(default_factory=dict)
    # degrade lane
    degrade_enabled: bool = True
    degrade_rule: str = "reclaim_disp55"
    degrade_expert: str = "rebound_trap_dn"
    degrade_ttl_minutes: int | None = None
    degrade_router_cfg: dict[str, Any] = field(default_factory=dict)
    # halt lane
    halt_enabled: bool = True
    halt_rule: str = "washout_and_reclaim"
    halt_expert: str = "washout_gate_halt"
    halt_ttl_minutes: int | None = None
    halt_router_cfg: dict[str, Any] = field(default_factory=dict)
    # hunt lane (arm → short TTL → budget; default off)
    hunter_enabled: bool = False
    hunter_ttl_minutes: int = 15
    hunter_max_entries_per_day: int = 1
    hunter_mutex_with_baseline: bool = True
    # symbol = block all baseline on hunted name; symbol_dir = only same direction
    hunter_mutex_scope: str = "symbol"
    # After a hunt, still allow one baseline entry in the *opposite* direction
    # (fixes wash→reclaim UP blocking later Rule-A DN on same name).
    hunter_allow_baseline_opposite: bool = False
    hunter_block_when_degrade: bool = True
    hunter_detector: str = "orb_fractal"  # orb_fractal | early_mf | washout_reclaim | off
    hunter_wash_drop_min: float = 0.005
    hunter_wash_window_end: str = "10:00"
    hunter_signal_deadline: str = "10:00"
    hunter_selloff_min_bars: int = 3
    hunter_hold_confirm_bars: int = 0
    hunter_require_breadth_min: int = 0  # 0 = no extra breadth gate beyond per-symbol washout
    hunter_skip_peer: bool = True
    hunter_skip_qqq_align: bool = False
    hunter_top_k: int = 1  # max hunt candidates injected per day (before fill budget)
    hunter_block_when_halt: bool = True  # architecture default; research may relax
    hunter_reclaim_level: str = "open"  # washout_reclaim: open | mid
    hunter_reclaim_buffer_pct: float = 0.0  # reclaim close must clear open*(1+buf)
    # early_mf detector (pre-Rule-A window)
    hunter_window_start: str = "09:45"
    hunter_window_end: str = "10:25"
    hunter_streak_min: int = 8
    hunter_streak_min_fast: int = 6
    hunter_from_prev_abs: float = 0.02
    hunter_vol_z_min: float = 1.0
    hunter_only_up: bool = False
    # Hunt-only trade / exit overrides (None = inherit baseline trade.*)
    hunter_hold_minutes: int | None = None
    hunter_hold_extend_minutes: int | None = None
    hunter_sl_mult: float | None = None
    hunter_tp_mult: float | None = None
    hunter_exit_mode: str | None = None
    hunter_early_exit_mode: str | None = None
    hunter_mae_cut_ret: float | None = None
    hunter_mae_cut_mfe_bypass: float | None = None
    hunter_mae_cut_min_hold_minutes: float | None = None
    hunter_position_frac: float | None = None
    # After a Hunt fill with ret <= this, halt remaining entries that day.
    # None / unset = off (default). Research only; not in peer3 freeze.
    hunter_day_circuit_ret: float | None = None
    # legacy oracle labels
    mode: str = "rule"  # rule | oracle
    labels: dict[str, str] = field(default_factory=dict)
    # map oracle day_type → state
    oracle_halt_types: tuple[str, ...] = ()
    oracle_degrade_types: tuple[str, ...] = ("rebound_trap_dn", "dn_toxic", "up_toxic")

    @classmethod
    def from_profile(cls, profile: dict[str, Any]) -> "WatchdogConfig":
        """Prefer enabled ``profile.watchdog``; else synthesize from ``regime_router``."""
        wd = profile.get("watchdog")
        if isinstance(wd, dict) and bool(wd.get("enabled", False)):
            return cls.from_dict(wd)
        rr = profile.get("regime_router")
        if isinstance(rr, dict) and bool(rr.get("enabled", False)):
            # Legacy bridge: single-rule router → degrade-only watchdog
            return cls.from_dict(
                {
                    "enabled": True,
                    "asof": rr.get("asof", "10:30"),
                    "mode": rr.get("mode", "rule"),
                    "experts_path": rr.get("experts_path"),
                    "experts": rr.get("experts") or {},
                    "labels_path": rr.get("labels_path") or rr.get("day_type_path"),
                    "labels": rr.get("labels") or {},
                    "degrade": {
                        "enabled": True,
                        "rule": rr.get("rule") or "reclaim_disp55",
                        "expert": None,  # filled by rule return
                        "ttl_minutes": rr.get("ttl_minutes"),
                        "router_cfg": {
                            k: v
                            for k, v in rr.items()
                            if k
                            not in {
                                "enabled",
                                "mode",
                                "rule",
                                "asof",
                                "experts_path",
                                "experts",
                                "labels",
                                "labels_path",
                                "day_type_path",
                            }
                        },
                    },
                    "halt": {"enabled": False},
                    "hunter": {"enabled": False},
                }
            )
        return cls(enabled=False)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "WatchdogConfig":
        raw = raw or {}
        deg = raw.get("degrade") if isinstance(raw.get("degrade"), dict) else {}
        halt = raw.get("halt") if isinstance(raw.get("halt"), dict) else {}
        hunt = raw.get("hunter") if isinstance(raw.get("hunter"), dict) else {}
        labels: dict[str, str] = {}
        lp = raw.get("labels_path") or raw.get("day_type_path")
        if lp:
            p = Path(str(lp)).expanduser()
            if p.is_file():
                if p.suffix.lower() == ".json":
                    blob = json.loads(p.read_text(encoding="utf-8"))
                    if isinstance(blob, dict):
                        labels = {str(k): str(v) for k, v in blob.items()}
                else:
                    df = pd.read_csv(p)
                    if "date" in df.columns and "day_type" in df.columns:
                        for r in df.itertuples(index=False):
                            labels[str(r.date)] = str(r.day_type)
        for k, v in (raw.get("labels") or {}).items():
            labels[str(k)] = str(v)

        deg_cfg = dict(deg.get("router_cfg") or {})
        # allow washout knobs at degrade/halt level
        for src, dst in ((deg, deg_cfg),):
            for k in (
                "wash_drop_min",
                "washout_breadth_min",
                "wash_window_end",
                "wash_min_bars",
                "washout_expert",
            ):
                if k in src and k not in dst:
                    dst[k] = src[k]
        halt_cfg = dict(halt.get("router_cfg") or {})
        for k in (
            "wash_drop_min",
            "washout_breadth_min",
            "wash_window_end",
            "wash_min_bars",
            "washout_expert",
        ):
            if k in halt and k not in halt_cfg:
                halt_cfg[k] = halt[k]
        if "washout_expert" not in halt_cfg and halt.get("expert"):
            halt_cfg["washout_expert"] = halt.get("expert")

        mode = str(raw.get("mode") or "rule").strip().lower()
        if mode in {"labels", "label", "oracle"}:
            mode = "oracle"
        elif mode in {"rule", "rules", "causal"}:
            mode = "rule"

        return cls(
            enabled=bool(raw.get("enabled", False)),
            asof=str(raw.get("asof") or "10:30"),
            experts_path=raw.get("experts_path"),
            experts=_load_experts(raw.get("experts_path"), raw.get("experts")),
            degrade_enabled=bool(deg.get("enabled", True)),
            degrade_rule=str(deg.get("rule") or "reclaim_disp55"),
            degrade_expert=str(deg.get("expert") or "rebound_trap_dn"),
            degrade_ttl_minutes=int(deg["ttl_minutes"]) if deg.get("ttl_minutes") is not None else None,
            degrade_router_cfg=deg_cfg,
            halt_enabled=bool(halt.get("enabled", False)),
            halt_rule=str(halt.get("rule") or "washout_and_reclaim"),
            halt_expert=str(halt.get("expert") or "washout_gate_halt"),
            halt_ttl_minutes=int(halt["ttl_minutes"]) if halt.get("ttl_minutes") is not None else None,
            halt_router_cfg=halt_cfg,
            hunter_enabled=bool(hunt.get("enabled", False)),
            hunter_ttl_minutes=int(hunt.get("ttl_minutes", 15) or 15),
            hunter_max_entries_per_day=int(hunt.get("max_entries_per_day", 1) or 1),
            hunter_mutex_with_baseline=bool(hunt.get("mutex_with_baseline", True)),
            hunter_mutex_scope=str(hunt.get("mutex_scope") or "symbol").strip().lower(),
            hunter_allow_baseline_opposite=bool(hunt.get("allow_baseline_opposite", False)),
            hunter_block_when_degrade=bool(hunt.get("block_when_degrade", True)),
            hunter_detector=str(hunt.get("detector") or "orb_fractal").strip().lower(),
            hunter_wash_drop_min=float(hunt.get("wash_drop_min", 0.005) or 0.005),
            hunter_wash_window_end=str(hunt.get("wash_window_end") or "10:00"),
            hunter_signal_deadline=str(hunt.get("signal_deadline") or "10:00"),
            hunter_selloff_min_bars=int(hunt.get("selloff_min_bars", 3) or 3),
            hunter_hold_confirm_bars=int(hunt.get("hold_confirm_bars", 0) or 0),
            hunter_require_breadth_min=int(hunt.get("require_breadth_min", 0) or 0),
            hunter_skip_peer=bool(hunt.get("skip_peer", True)),
            hunter_skip_qqq_align=bool(hunt.get("skip_qqq_align", False)),
            hunter_top_k=int(hunt.get("top_k", 1) or 1),
            hunter_block_when_halt=bool(hunt.get("block_when_halt", True)),
            hunter_reclaim_level=str(hunt.get("reclaim_level") or "open").strip().lower(),
            hunter_reclaim_buffer_pct=float(hunt.get("reclaim_buffer_pct", 0.0) or 0.0),
            hunter_window_start=str(hunt.get("window_start") or "09:45"),
            hunter_window_end=str(hunt.get("window_end") or "10:25"),
            hunter_streak_min=int(hunt.get("streak_min", 8) or 8),
            hunter_streak_min_fast=int(hunt.get("streak_min_fast", 6) or 6),
            hunter_from_prev_abs=float(hunt.get("from_prev_abs", 0.02) or 0.02),
            hunter_vol_z_min=float(hunt.get("vol_z_min", 1.0) or 1.0),
            hunter_only_up=bool(hunt.get("only_up", False)),
            hunter_hold_minutes=(
                int(hunt["hold_minutes"]) if hunt.get("hold_minutes") is not None else None
            ),
            hunter_hold_extend_minutes=(
                int(hunt["hold_extend_minutes"])
                if hunt.get("hold_extend_minutes") is not None
                else None
            ),
            hunter_sl_mult=(
                float(hunt["sl_mult"]) if hunt.get("sl_mult") is not None else None
            ),
            hunter_tp_mult=(
                float(hunt["tp_mult"]) if hunt.get("tp_mult") is not None else None
            ),
            hunter_exit_mode=(
                str(hunt.get("exit_mode")).strip() if hunt.get("exit_mode") is not None else None
            ),
            hunter_early_exit_mode=(
                str(hunt.get("early_exit_mode")).strip()
                if hunt.get("early_exit_mode") is not None
                else None
            ),
            hunter_mae_cut_ret=(
                float(hunt["mae_cut_ret"]) if hunt.get("mae_cut_ret") is not None else None
            ),
            hunter_mae_cut_mfe_bypass=(
                float(hunt["mae_cut_mfe_bypass"])
                if hunt.get("mae_cut_mfe_bypass") is not None
                else None
            ),
            hunter_mae_cut_min_hold_minutes=(
                float(hunt["mae_cut_min_hold_minutes"])
                if hunt.get("mae_cut_min_hold_minutes") is not None
                else None
            ),
            hunter_position_frac=(
                float(hunt["position_frac"]) if hunt.get("position_frac") is not None else None
            ),
            hunter_day_circuit_ret=(
                float(hunt["day_circuit_ret"])
                if hunt.get("day_circuit_ret") is not None
                else None
            ),
            mode=mode,
            labels=labels,
            oracle_halt_types=tuple(raw.get("oracle_halt_types") or ()),
            oracle_degrade_types=tuple(
                raw.get("oracle_degrade_types") or ("rebound_trap_dn", "dn_toxic", "up_toxic")
            ),
        )


def hunt_trade_overrides(cfg: WatchdogConfig) -> dict[str, Any]:
    """Sim/size kwargs that differ from baseline ``trade`` for Hunt fills only."""
    out: dict[str, Any] = {}
    if cfg.hunter_hold_minutes is not None:
        out["hold_minutes"] = int(cfg.hunter_hold_minutes)
    if cfg.hunter_hold_extend_minutes is not None:
        out["hold_extend_minutes"] = int(cfg.hunter_hold_extend_minutes)
    if cfg.hunter_sl_mult is not None:
        out["sl_mult"] = float(cfg.hunter_sl_mult)
    if cfg.hunter_tp_mult is not None:
        out["tp_mult"] = float(cfg.hunter_tp_mult)
    if cfg.hunter_exit_mode is not None:
        out["exit_mode"] = str(cfg.hunter_exit_mode)
    if cfg.hunter_early_exit_mode is not None:
        out["early_exit_mode"] = str(cfg.hunter_early_exit_mode)
    if cfg.hunter_mae_cut_ret is not None:
        out["mae_cut_ret"] = float(cfg.hunter_mae_cut_ret)
    if cfg.hunter_mae_cut_mfe_bypass is not None:
        out["mae_cut_mfe_bypass"] = float(cfg.hunter_mae_cut_mfe_bypass)
    if cfg.hunter_mae_cut_min_hold_minutes is not None:
        out["mae_cut_min_hold_minutes"] = float(cfg.hunter_mae_cut_min_hold_minutes)
    if cfg.hunter_position_frac is not None:
        out["position_frac"] = float(cfg.hunter_position_frac)
    return out


class RegimeWatchdog:
    """Day-scoped (offline) / bar-scoped (live) regime state machine."""

    def __init__(self, cfg: WatchdogConfig):
        self.cfg = cfg
        self.experts = dict(cfg.experts)
        self._day_decision: WatchdogDecision | None = None
        self._hunt_entries_today = 0
        self._current_date: str | None = None
        self.hunt_candidates: list[HuntCandidate] = []
        self.hunt_armed: bool = False

    @classmethod
    def from_profile(cls, profile: dict[str, Any]) -> "RegimeWatchdog | None":
        cfg = WatchdogConfig.from_profile(profile)
        if not cfg.enabled:
            return None
        return cls(cfg)

    def _expert_overlay(self, expert_name: str | None, *, state: WatchdogState, reason: str) -> Overlay:
        if not expert_name or expert_name in {"", "baseline", "ok", "other_loss", "wide_chop"}:
            return Overlay(route_tag="baseline", allow_baseline=True, allow_hunt=True)
        blob = self.experts.get(expert_name) or {}
        patch = copy.deepcopy(blob.get("regime") or {})
        allow_hunt = state == WatchdogState.HUNT
        if state == WatchdogState.DEGRADE and not self.cfg.hunter_block_when_degrade:
            allow_hunt = True
        if state == WatchdogState.HALT:
            allow_hunt = not bool(self.cfg.hunter_block_when_halt)
        return Overlay(
            expert_name=expert_name,
            regime_patch=patch,
            allow_baseline=state != WatchdogState.HALT,
            allow_hunt=allow_hunt,
            route_tag=expert_name,
        )

    def _decide_rule(
        self,
        *,
        date: str,
        stock_by: dict[str, pd.DataFrame],
        qqq_df: pd.DataFrame | None,
        symbols: list[str],
    ) -> WatchdogDecision:
        asof = pd.Timestamp(f"{date} {self.cfg.asof}", tz=NY)

        # 1) HALT lane (narrow toxic)
        if self.cfg.halt_enabled and self.cfg.halt_rule:
            hit = eval_router_rule(
                self.cfg.halt_rule,
                date=date,
                stock_by=stock_by,
                qqq_df=qqq_df,
                symbols=symbols,
                asof_hhmm=self.cfg.asof,
                router_cfg=self.cfg.halt_router_cfg,
            )
            # rule may return washout expert name; force configured halt expert
            if hit:
                expert = self.cfg.halt_expert or hit
                ov = self._expert_overlay(expert, state=WatchdogState.HALT, reason="halt_rule")
                return WatchdogDecision(
                    state=WatchdogState.HALT,
                    overlay=ov,
                    reason=f"halt:{self.cfg.halt_rule}",
                    asof=asof,
                    armed_until=_ttl_until(asof, self.cfg.halt_ttl_minutes),
                    expert=expert,
                )

        # 2) DEGRADE lane
        if self.cfg.degrade_enabled and self.cfg.degrade_rule:
            hit = eval_router_rule(
                self.cfg.degrade_rule,
                date=date,
                stock_by=stock_by,
                qqq_df=qqq_df,
                symbols=symbols,
                asof_hhmm=self.cfg.asof,
                router_cfg=self.cfg.degrade_router_cfg,
            )
            if hit:
                expert = hit if hit in self.experts else (self.cfg.degrade_expert or hit)
                ov = self._expert_overlay(expert, state=WatchdogState.DEGRADE, reason="degrade_rule")
                return WatchdogDecision(
                    state=WatchdogState.DEGRADE,
                    overlay=ov,
                    reason=f"degrade:{self.cfg.degrade_rule}",
                    asof=asof,
                    armed_until=_ttl_until(asof, self.cfg.degrade_ttl_minutes),
                    expert=expert,
                )

        # HUNT is planned separately (morning short window); baseline stays NORMAL here.
        return WatchdogDecision(
            state=WatchdogState.NORMAL,
            overlay=Overlay(route_tag="baseline", allow_baseline=True, allow_hunt=True),
            reason="normal",
            asof=asof,
            armed_until=None,
            expert=None,
        )

    def _hunt_allowed_under(self, baseline: WatchdogDecision) -> bool:
        if not self.cfg.hunter_enabled:
            return False
        if baseline.state == WatchdogState.HALT and self.cfg.hunter_block_when_halt:
            return False
        if baseline.state == WatchdogState.DEGRADE and self.cfg.hunter_block_when_degrade:
            return False
        return True

    def collect_hunt_candidates(
        self,
        *,
        date: str,
        stock_by: dict[str, pd.DataFrame],
        symbols: list[str],
        baseline: WatchdogDecision | None = None,
    ) -> list[HuntCandidate]:
        """Arm short-window hunters (causal). Empty if disabled / blocked / no detector hit."""
        baseline = baseline or self._day_decision
        if baseline is None or not self._hunt_allowed_under(baseline):
            return []
        det = str(self.cfg.hunter_detector or "").strip().lower()
        if det in {"", "off", "none", "false"}:
            return []

        if int(self.cfg.hunter_require_breadth_min or 0) > 0:
            n_wash, _ = count_open_washout(
                stock_by,
                date=date,
                symbols=symbols,
                cfg=OrbOpenConfig(
                    wash_window_end=self.cfg.hunter_wash_window_end,
                    wash_drop_min=float(self.cfg.hunter_wash_drop_min),
                ),
            )
            if n_wash < int(self.cfg.hunter_require_breadth_min):
                return []

        out: list[HuntCandidate] = []
        ttl = int(self.cfg.hunter_ttl_minutes)
        if det in {"orb_fractal", "orb", "orb_open"}:
            orb_cfg = OrbOpenConfig(
                wash_window_end=self.cfg.hunter_wash_window_end,
                wash_drop_min=float(self.cfg.hunter_wash_drop_min),
                selloff_min_bars=int(self.cfg.hunter_selloff_min_bars),
                hold_confirm_bars=int(self.cfg.hunter_hold_confirm_bars),
                signal_deadline=self.cfg.hunter_signal_deadline,
                only_up=True,
            )
            sigs: list[OrbSignal] = scan_orb_day(
                stock_by, date=date, symbols=symbols, cfg=orb_cfg
            )
            for sig in sigs:
                armed_until = pd.Timestamp(sig.sig_ts) + pd.Timedelta(minutes=ttl)
                out.append(
                    HuntCandidate(
                        symbol=sig.symbol,
                        date=sig.date,
                        direction=sig.direction,
                        sig_ts=pd.Timestamp(sig.sig_ts),
                        armed_until=armed_until,
                        detector="orb_fractal",
                        reason=sig.reason,
                        fractal_high=float(sig.fractal_high),
                        wash_drop=float(sig.wash_drop),
                    )
                )
        elif det in {"early_mf", "early_mf_fast", "mf_fast"}:
            # Pre-Rule-A window: same Rule-A mask with early_on_mf_fast, before 10:30.
            for sym in symbols:
                sdf = stock_by.get(sym)
                if sdf is None or sdf.empty:
                    continue
                day = sdf[sdf["date"].astype(str) == str(date)]
                if day.empty:
                    continue
                fire = first_rule_a_day(
                    day,
                    window_start=self.cfg.hunter_window_start,
                    window_end=self.cfg.hunter_window_end,
                    streak_min=int(self.cfg.hunter_streak_min),
                    from_prev_abs=float(self.cfg.hunter_from_prev_abs),
                    vol_z_min=float(self.cfg.hunter_vol_z_min),
                    early_on_mf_fast=True,
                    streak_min_fast=int(self.cfg.hunter_streak_min_fast),
                )
                if fire is None:
                    continue
                direction = str(fire["dir"])
                if self.cfg.hunter_only_up and direction != "UP":
                    continue
                sig_ts = pd.Timestamp(fire["sig_ts"])
                if getattr(sig_ts, "tzinfo", None) is None:
                    sig_ts = sig_ts.tz_localize(NY)
                else:
                    sig_ts = sig_ts.tz_convert(NY)
                out.append(
                    HuntCandidate(
                        symbol=str(sym),
                        date=str(date),
                        direction=direction,
                        sig_ts=sig_ts,
                        armed_until=sig_ts + pd.Timedelta(minutes=ttl),
                        detector="early_mf",
                        reason="early_mf_fast",
                    )
                )
        elif det in {"washout_reclaim", "wash_reclaim", "reclaim_open"}:
            wr_cfg = WashoutReclaimConfig(
                wash_window_end=self.cfg.hunter_wash_window_end,
                wash_drop_min=float(self.cfg.hunter_wash_drop_min),
                signal_deadline=self.cfg.hunter_signal_deadline,
                hold_confirm_bars=int(self.cfg.hunter_hold_confirm_bars),
                reclaim_level=str(self.cfg.hunter_reclaim_level or "open"),
                reclaim_buffer_pct=float(self.cfg.hunter_reclaim_buffer_pct or 0.0),
                only_up=True,
            )
            sigs = scan_washout_reclaim_day(
                stock_by, date=date, symbols=symbols, cfg=wr_cfg
            )
            for sig in sigs:
                if self.cfg.hunter_only_up and sig.direction != "UP":
                    continue
                armed_until = pd.Timestamp(sig.sig_ts) + pd.Timedelta(minutes=ttl)
                out.append(
                    HuntCandidate(
                        symbol=sig.symbol,
                        date=sig.date,
                        direction=sig.direction,
                        sig_ts=pd.Timestamp(sig.sig_ts),
                        armed_until=armed_until,
                        detector="washout_reclaim",
                        reason=sig.reason,
                        fractal_high=float(sig.fractal_high),
                        wash_drop=float(sig.wash_drop),
                    )
                )
        # earliest first, then cap top_k
        out.sort(key=lambda h: (h.sig_ts, h.symbol))
        k = max(1, int(self.cfg.hunter_top_k))
        return out[:k]

    def _decide_oracle(self, date: str) -> WatchdogDecision:
        asof = pd.Timestamp(f"{date} {self.cfg.asof}", tz=NY)
        day_type = str(self.cfg.labels.get(str(date), "baseline"))
        if day_type in self.cfg.oracle_halt_types:
            ov = self._expert_overlay(day_type, state=WatchdogState.HALT, reason="oracle_halt")
            return WatchdogDecision(
                state=WatchdogState.HALT,
                overlay=ov,
                reason=f"oracle:{day_type}",
                asof=asof,
                armed_until=_ttl_until(asof, self.cfg.halt_ttl_minutes),
                expert=day_type,
            )
        if day_type in self.cfg.oracle_degrade_types and day_type in self.experts:
            ov = self._expert_overlay(day_type, state=WatchdogState.DEGRADE, reason="oracle_degrade")
            return WatchdogDecision(
                state=WatchdogState.DEGRADE,
                overlay=ov,
                reason=f"oracle:{day_type}",
                asof=asof,
                armed_until=_ttl_until(asof, self.cfg.degrade_ttl_minutes),
                expert=day_type,
            )
        return WatchdogDecision(
            state=WatchdogState.NORMAL,
            overlay=Overlay(route_tag="baseline", allow_baseline=True, allow_hunt=True),
            reason="normal",
            asof=asof,
            expert=None,
        )

    def begin_day(
        self,
        date: str,
        *,
        stock_by: dict[str, pd.DataFrame],
        qqq_df: pd.DataFrame | None,
        symbols: list[str],
    ) -> WatchdogDecision:
        """Evaluate morning state once per session (offline replay).

        Also arms hunt candidates when the hunter lane is enabled and not blocked
        by HALT / DEGRADE policy. Baseline overlay is unchanged by hunt arming.
        """
        # Live may re-call begin_day as morning bars accumulate; only reset
        # Hunt budget on a true day change (offline calls once per date).
        if self._current_date != str(date):
            self._hunt_entries_today = 0
        self._current_date = str(date)
        self.hunt_candidates = []
        self.hunt_armed = False
        if self.cfg.mode == "oracle":
            self._day_decision = self._decide_oracle(str(date))
        else:
            self._day_decision = self._decide_rule(
                date=str(date),
                stock_by=stock_by,
                qqq_df=qqq_df,
                symbols=symbols,
            )
        self.hunt_candidates = self.collect_hunt_candidates(
            date=str(date),
            stock_by=stock_by,
            symbols=symbols,
            baseline=self._day_decision,
        )
        self.hunt_armed = bool(self.hunt_candidates)
        return self._day_decision

    def hunt_budget_remaining(self) -> int:
        return max(0, int(self.cfg.hunter_max_entries_per_day) - int(self._hunt_entries_today))

    def decision_at(self, ts: pd.Timestamp | None = None) -> WatchdogDecision:
        """Return active decision; expire to NORMAL if past TTL."""
        d = self._day_decision
        if d is None:
            return WatchdogDecision(
                state=WatchdogState.NORMAL,
                overlay=Overlay(route_tag="baseline", allow_baseline=True, allow_hunt=True),
                reason="uninitialized",
            )
        if ts is not None and d.armed_until is not None and not d.active_at(ts):
            return WatchdogDecision(
                state=WatchdogState.NORMAL,
                overlay=Overlay(route_tag="baseline", allow_baseline=True, allow_hunt=True),
                reason="ttl_expired",
                asof=d.asof,
                armed_until=d.armed_until,
                expert=None,
            )
        return d

    def apply_to_regime(self, regime_cfg: dict[str, Any], snap: dict[str, Any], *, ts=None) -> WatchdogDecision:
        """Restore baseline snap then apply active overlay."""
        restore_regime(regime_cfg, snap)
        d = self.decision_at(ts)
        if d.state != WatchdogState.NORMAL:
            apply_overlay(regime_cfg, d.overlay)
        return d

    def note_hunt_entry(self) -> bool:
        """Consume one hunt fill slot. Return False if budget exhausted."""
        if not self.cfg.hunter_enabled:
            return False
        if self._hunt_entries_today >= int(self.cfg.hunter_max_entries_per_day):
            return False
        self._hunt_entries_today += 1
        return True

    def summary_counts(self) -> dict[str, int]:
        return {
            "hunt_armed": int(self.hunt_armed),
            "hunt_candidates": int(len(self.hunt_candidates)),
            "hunt_entries": int(self._hunt_entries_today),
        }

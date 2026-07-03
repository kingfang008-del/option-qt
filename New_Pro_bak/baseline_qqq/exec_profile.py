"""
Execution profile routing: Path A (scalp_0dte) vs Path C (swing_1dte / auto_hybrid).

Phase 1 focuses on profile-specific hold/stop rails and replay shadow accounting.
Contract DTE selection at entry is recorded on the plan; dual-expiry IBKR locks land in Phase 2.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Optional

logger = logging.getLogger("ExecProfile")

from bidirectional_regime import DayType, resolve_day_type

# Exit knobs resolved per ExecMode (SCALP vs SWING).
_PROFILE_EXIT_KEYS = (
    "STOP_LOSS",
    "ABSOLUTE_STOP_LOSS",
    "MID_TIME_STOP_MINS",
    "MID_TIME_STOP_ROI",
    "TIME_STOP_MINS",
    "TIME_STOP_ROI",
    "TRAILING_TRIGGER_ROI",
    "TRAILING_KEEP_RATIO",
    "FLASH_PROTECT_TRIGGER",
    "FLASH_PROTECT_EXIT",
    "NO_MOMENTUM_MINS",
    "NO_MOMENTUM_MIN_MAX_ROI",
    "ZOMBIE_EXIT_MINS",
)


class ExecProfile(str, Enum):
    """Top-level runtime profile (env EXEC_PROFILE)."""

    SCALP_0DTE = "scalp_0dte"
    SWING_1DTE = "swing_1dte"
    AUTO_HYBRID = "auto_hybrid"
    MULTI_BAND = "multi_band"


class ExecBand(str, Enum):
    """Intraday roll leg within multi_band profile."""

    BAND1 = "BAND1"  # 错价/低价快打
    BAND2 = "BAND2"  # 趋势确认
    BAND3 = "BAND3"  # 尾段/epic


class ExecMode(str, Enum):
    """Resolved per-trade mode after routing."""

    SCALP = "SCALP"
    SWING = "SWING"


@dataclass(frozen=True)
class ExecPlan:
    profile: str
    mode: ExecMode
    target_dte: int
    hold_profile: str
    reason: str
    exec_band: str = ""


def parse_exec_profile(raw: Optional[str] = None) -> ExecProfile:
    text = str(raw if raw is not None else os.environ.get("EXEC_PROFILE", "auto_hybrid")).strip().lower()
    aliases = {
        "a": ExecProfile.SCALP_0DTE,
        "scalp": ExecProfile.SCALP_0DTE,
        "0dte": ExecProfile.SCALP_0DTE,
        "c": ExecProfile.AUTO_HYBRID,
        "hybrid": ExecProfile.AUTO_HYBRID,
        "auto": ExecProfile.AUTO_HYBRID,
        "swing": ExecProfile.SWING_1DTE,
        "1dte": ExecProfile.SWING_1DTE,
        "roll": ExecProfile.MULTI_BAND,
        "multi_band": ExecProfile.MULTI_BAND,
        "bands": ExecProfile.MULTI_BAND,
    }
    if text in aliases:
        return aliases[text]
    try:
        return ExecProfile(text)
    except ValueError as exc:
        raise ValueError(
            f"Unsupported EXEC_PROFILE={text!r}; expected scalp_0dte, swing_1dte, "
            f"auto_hybrid, multi_band"
        ) from exc


def _cfg_val(cfg: Any, name: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if hasattr(cfg, name):
        return getattr(cfg, name)
    return default


def profile_exit_overrides(cfg: Any, mode: ExecMode | str) -> Dict[str, Any]:
    """Map SCALP/SWING/BAND* mode to StrategyConfig overlay keys."""
    if isinstance(mode, ExecMode):
        mode_key = mode.value
    else:
        text = str(mode).upper()
        if text in {b.value for b in ExecBand}:
            mode_key = text
        else:
            mode_key = ExecMode(text).value
    prefix = mode_key
    out: Dict[str, Any] = {}
    for key in _PROFILE_EXIT_KEYS:
        prefixed = f"{prefix}_{key}"
        if hasattr(cfg, prefixed):
            out[key] = getattr(cfg, prefixed)
    return out


def effective_exit_param(cfg: Any, pos: Optional[Mapping[str, Any]], name: str) -> Any:
    """Read exit param with SCALP_/SWING_/BAND* overlay when pos carries exec_mode/exec_band."""
    band_raw = (pos or {}).get("exec_band", "")
    if band_raw:
        overrides = profile_exit_overrides(cfg, str(band_raw).upper())
        if name in overrides:
            return overrides[name]
    mode_raw = (pos or {}).get("exec_mode", ExecMode.SWING.value)
    try:
        mode = ExecMode(str(mode_raw).upper())
    except ValueError:
        mode = ExecMode.SWING
    overrides = profile_exit_overrides(cfg, mode)
    if name in overrides:
        return overrides[name]
    return getattr(cfg, name)


def _session_allows_scalp(ctx: Mapping[str, Any], cfg: Any) -> bool:
    t = ctx.get("time")
    if t is None:
        return True
    end_h = int(_cfg_val(cfg, "HYBRID_SCALP_SESSION_END_HOUR", 14))
    end_m = int(_cfg_val(cfg, "HYBRID_SCALP_SESSION_END_MINUTE", 30))
    if t.hour > end_h:
        return False
    if t.hour == end_h and t.minute >= end_m:
        return False
    start_h = int(_cfg_val(cfg, "START_HOUR", 9))
    start_m = int(_cfg_val(cfg, "START_MINUTE", 45))
    if t.hour < start_h:
        return False
    if t.hour == start_h and t.minute < start_m:
        return False
    return True


def _scalp_gates_pass(ctx: Mapping[str, Any], cfg: Any) -> tuple[bool, str]:
    alpha = abs(float(ctx.get("alpha_z", ctx.get("alpha", 0.0)) or 0.0))
    min_edge = float(_cfg_val(cfg, "HYBRID_SCALP_MIN_NET_EDGE", 0.030))
    if alpha < min_edge:
        return False, f"edge={alpha:.3f}<{min_edge:.3f}"

    if bool(ctx.get("is_volatile_regime", False)):
        return False, "volatile_regime"

    spread = float(ctx.get("options_vw_spread", 0.0) or 0.0)
    max_spread = float(_cfg_val(cfg, "HYBRID_SCALP_MAX_SPREAD", 0.08))
    if spread > max_spread > 0:
        return False, f"spread={spread:.2%}>{max_spread:.0%}"

    iv_mom = abs(float(ctx.get("options_iv_momentum", 0.0) or 0.0))
    iv_cap = float(_cfg_val(cfg, "FAST_GATE_IV_MOMENTUM_ABS_MAX", 0.50))
    if iv_mom > iv_cap:
        return False, f"iv_mom={iv_mom:.2f}>{iv_cap:.2f}"

    if not _session_allows_scalp(ctx, cfg):
        return False, "outside_scalp_session"

    spy = float(ctx.get("spy_roc", 0.0) or 0.0)
    qqq = float(ctx.get("qqq_roc", 0.0) or 0.0)
    min_idx = float(_cfg_val(cfg, "TREND_CORE_MIN_INDEX_ROC", 0.00015))
    direction = 1 if float(ctx.get("alpha", 0.0) or 0.0) >= 0 else -1
    if direction >= 0 and not (spy >= min_idx and qqq >= min_idx):
        return False, f"index_long_weak spy={spy:.4f} qqq={qqq:.4f}"
    if direction < 0 and not (spy <= -min_idx and qqq <= -min_idx):
        return False, f"index_short_weak spy={spy:.4f} qqq={qqq:.4f}"

    return True, "scalp_gates_ok"


def _band_price_tiers(cfg: Any) -> tuple[float, float]:
    """(band2_floor, band3_floor) in option mid price."""
    b2 = float(_cfg_val(cfg, "MULTI_BAND2_PRICE_FLOOR", 0.85))
    b3 = float(_cfg_val(cfg, "MULTI_BAND3_PRICE_FLOOR", 2.00))
    return b2, b3


def resolve_exec_band(
    ctx: Mapping[str, Any],
    cfg: Any,
    *,
    legs_today: int = 0,
) -> tuple[Optional[ExecBand], str]:
    """
    Pick BAND1/2/3 from option price + session leg count.
    Band1 = cheap dislocation; Band2 = trend; Band3 = epic extension.
    """
    max_legs = int(_cfg_val(cfg, "MULTI_BAND_MAX_LEGS_PER_DAY", 3))
    if legs_today >= max_legs:
        return None, f"max_legs={legs_today}>={max_legs}"

    price = float(ctx.get("curr_price", 0.0) or 0.0)
    if price <= 0.01:
        return None, "no_option_price"

    b2_floor, b3_floor = _band_price_tiers(cfg)
    snap = float(ctx.get("snap_roc", 0.0) or 0.0)
    alpha = float(ctx.get("alpha_z", ctx.get("alpha", 0.0)) or 0.0)
    direction = 1 if alpha >= 0 else -1

    if price < b2_floor:
        min_snap = float(_cfg_val(cfg, "MULTI_BAND1_MIN_SNAP_ROC", 0.0008))
        min_edge = float(_cfg_val(cfg, "MULTI_BAND1_MIN_NET_EDGE", 0.012))
        if direction >= 0 and snap < min_snap:
            return None, f"band1_snap={snap:.4f}<{min_snap}"
        if direction < 0 and snap > -min_snap:
            return None, f"band1_put_snap={snap:.4f}>{-min_snap}"
        if abs(alpha) < min_edge:
            return None, f"band1_edge={abs(alpha):.3f}<{min_edge}"
        return ExecBand.BAND1, f"price={price:.2f}<{b2_floor:.2f}|dislocation"

    if price >= b3_floor:
        min_roc = float(_cfg_val(cfg, "MULTI_BAND3_MIN_STOCK_ROC", 0.0004))
        stock_roc = float(ctx.get("stock_roc", 0.0) or 0.0)
        if direction >= 0 and stock_roc < min_roc:
            return None, f"band3_stock_roc={stock_roc:.4f}<{min_roc}"
        if direction < 0 and stock_roc > -min_roc:
            return None, f"band3_put_stock_roc={stock_roc:.4f}>{-min_roc}"
        return ExecBand.BAND3, f"price={price:.2f}>={b3_floor:.2f}|epic"

    return ExecBand.BAND2, f"{b2_floor:.2f}≤price={price:.2f}<{b3_floor:.2f}|trend"


def _band_to_mode(band: ExecBand) -> ExecMode:
    if band == ExecBand.BAND1:
        return ExecMode.SCALP
    if band == ExecBand.BAND3:
        return ExecMode.SWING
    return ExecMode.SWING


def _band_hold_profile(band: ExecBand) -> str:
    return {
        ExecBand.BAND1: "band1_dislocation",
        ExecBand.BAND2: "band2_trend",
        ExecBand.BAND3: "band3_epic",
    }[band]


def multi_band_roll_cooldown_seconds(cfg: Any, *, profitable: bool, reason: str) -> int:
    """Shorter cooldown after profitable roll leg; keep long cooldown on stops."""
    long_cd = int(_cfg_val(cfg, "COOLDOWN_MINUTES", 60)) * 60
    if not profitable:
        return long_cd
    reason_u = str(reason or "").upper()
    stop_tokens = (
        "HARD_STOP", "ABSOLUTE_STOP", "STOP_LOSS", "STOCK_STOP", "COND_STOP", "FLIP", "DIR_OPP",
    )
    if any(tok in reason_u for tok in stop_tokens):
        return long_cd
    return int(_cfg_val(cfg, "MULTI_BAND_ROLL_COOLDOWN_MINS", 8)) * 60


def resolve_effective_exec_profile(
    base: ExecProfile | str,
    ctx: Mapping[str, Any],
    cfg: Any,
    *,
    regime_routing_enabled: Optional[bool] = None,
) -> tuple[ExecProfile, str]:
    """
    Phase 4: map day_type → effective EXEC_PROFILE (when REGIME_ROUTING_ENABLED).
    """
    prof = base if isinstance(base, ExecProfile) else parse_exec_profile(str(base))
    if regime_routing_enabled is None:
        regime_routing_enabled = False

    if not regime_routing_enabled:
        return prof, f"regime_off|base={prof.value}"

    day_th = float(_cfg_val(cfg, "BIDIRECTIONAL_DAY_ROC_THRESHOLD", 0.0035))
    day_type = resolve_day_type(ctx, day_roc_threshold=day_th)

    if day_type == DayType.DISLOCATION:
        if prof != ExecProfile.MULTI_BAND:
            return ExecProfile.MULTI_BAND, f"regime→multi_band|day={day_type.value}|epic_upgrade"
        return prof, f"regime_keep|day={day_type.value}"

    if day_type == DayType.CHOP:
        if prof == ExecProfile.AUTO_HYBRID:
            return ExecProfile.SCALP_0DTE, f"regime→scalp|day={day_type.value}|chop_short"
        return prof, f"regime_keep|day={day_type.value}"

    if day_type == DayType.TREND_DOWN and prof == ExecProfile.SWING_1DTE:
        return ExecProfile.AUTO_HYBRID, f"regime→auto_hybrid|day={day_type.value}|put_flex"

    return prof, f"regime_keep|day={day_type.value}|base={prof.value}"


def resolve_exec_plan(
    profile: ExecProfile | str,
    ctx: Mapping[str, Any],
    cfg: Any,
    *,
    legs_today: int = 0,
    regime_routing_enabled: Optional[bool] = None,
) -> ExecPlan:
    """Resolve runtime profile + context into per-trade execution plan."""
    effective, regime_detail = resolve_effective_exec_profile(
        profile, ctx, cfg, regime_routing_enabled=regime_routing_enabled,
    )
    prof = effective

    if prof == ExecProfile.MULTI_BAND:
        band, detail = resolve_exec_band(ctx, cfg, legs_today=legs_today)
        if band is None:
            return ExecPlan(
                profile=prof.value,
                mode=ExecMode.SWING,
                target_dte=0,
                hold_profile="band_blocked",
                reason=f"multi_band_blocked|{detail}",
                exec_band="",
            )
        mode = _band_to_mode(band)
        dte = 0 if band == ExecBand.BAND1 else int(_cfg_val(cfg, "MULTI_BAND_SWING_DTE", 1))
        return ExecPlan(
            profile=prof.value,
            mode=mode,
            target_dte=dte,
            hold_profile=_band_hold_profile(band),
            reason=f"multi_band→{band.value}|{detail}|{regime_detail}",
            exec_band=band.value,
        )

    if prof == ExecProfile.SCALP_0DTE:
        return ExecPlan(
            profile=prof.value,
            mode=ExecMode.SCALP,
            target_dte=0,
            hold_profile="scalp_v1",
            reason=f"profile=scalp_0dte|{regime_detail}",
        )

    if prof == ExecProfile.SWING_1DTE:
        return ExecPlan(
            profile=prof.value,
            mode=ExecMode.SWING,
            target_dte=1,
            hold_profile="swing_v1",
            reason=f"profile=swing_1dte|{regime_detail}",
        )

    ok, detail = _scalp_gates_pass(ctx, cfg)
    if ok:
        return ExecPlan(
            profile=prof.value,
            mode=ExecMode.SCALP,
            target_dte=0,
            hold_profile="scalp_v1",
            reason=f"auto_hybrid→scalp|{detail}|{regime_detail}",
        )
    return ExecPlan(
        profile=prof.value,
        mode=ExecMode.SWING,
        target_dte=1,
        hold_profile="swing_v1",
        reason=f"auto_hybrid→swing|{detail}|{regime_detail}",
    )


def attach_exec_plan_to_signal(sig: MutableMapping[str, Any], plan: ExecPlan) -> None:
    meta = sig.setdefault("meta", {})
    meta["exec_profile"] = plan.profile
    meta["exec_mode"] = plan.mode.value
    meta["exec_dte"] = int(plan.target_dte)
    meta["hold_profile"] = plan.hold_profile
    meta["exec_route_reason"] = plan.reason
    meta["exec_band"] = plan.exec_band or ""
    sig["exec_mode"] = plan.mode.value
    sig["exec_dte"] = int(plan.target_dte)
    sig["hold_profile"] = plan.hold_profile
    sig["exec_band"] = plan.exec_band or ""


def holding_from_position_state(st: Any) -> Dict[str, Any]:
    """Build strategy holding dict including exec profile fields."""
    init_ctx = getattr(st, "init_ctx", None) or {}
    return {
        "entry_price": float(getattr(st, "entry_price", 0.0) or 0.0),
        "entry_stock": float(getattr(st, "entry_stock", 0.0) or 0.0),
        "entry_ts": float(getattr(st, "entry_ts", 0.0) or 0.0),
        "dir": int(getattr(st, "position", 0) or 0),
        "max_roi": float(getattr(st, "max_roi", 0.0) or 0.0),
        "entry_spy_roc": float(getattr(st, "entry_spy_roc", 0.0) or 0.0),
        "entry_index_trend": int(getattr(st, "entry_index_trend", 0) or 0),
        "exec_mode": str(getattr(st, "exec_mode", init_ctx.get("exec_mode", ExecMode.SWING.value))),
        "exec_dte": int(getattr(st, "exec_dte", init_ctx.get("exec_dte", 1)) or 1),
        "exec_profile": str(getattr(st, "exec_profile", init_ctx.get("exec_profile", ""))),
        "hold_profile": str(getattr(st, "hold_profile", init_ctx.get("hold_profile", ""))),
        "exec_band": str(getattr(st, "exec_band", init_ctx.get("exec_band", ""))),
        "init_ctx": dict(init_ctx) if isinstance(init_ctx, dict) else {},
    }


@dataclass
class _ShadowPosition:
    profile: str
    mode: str
    symbol: str
    direction: int
    entry_price: float
    entry_stock: float
    entry_ts: float
    max_roi: float = -1.0
    closed: bool = False
    exit_ts: float = 0.0
    exit_roi: float = 0.0
    exit_reason: str = ""


class ExecProfileReplayLedger:
    """
    Shadow portfolios for A/B compare during replay/live-dry.

    When EXEC_PROFILE_SHADOW_COMPARE=1, tracks SCALP and SWING legs on the same
    entry marks without placing extra orders.
    """

    def __init__(self, enabled: bool = False, output_path: Optional[Path] = None):
        self.enabled = bool(enabled)
        self.output_path = Path(output_path) if output_path else None
        self._open: Dict[str, _ShadowPosition] = {}
        self._closed: list[_ShadowPosition] = []
        self._events: list[dict] = []

    @classmethod
    def from_env(cls) -> "ExecProfileReplayLedger":
        flag = os.environ.get("EXEC_PROFILE_SHADOW_COMPARE", "0").strip().lower() in {
            "1", "true", "yes", "on",
        }
        out = os.environ.get("EXEC_PROFILE_SHADOW_OUTPUT", "").strip()
        path = Path(out) if out else None
        return cls(enabled=flag, output_path=path)

    def _key(self, profile: str, symbol: str) -> str:
        return f"{profile}:{symbol}".upper()

    def on_entry(
        self,
        *,
        symbol: str,
        direction: int,
        entry_price: float,
        entry_stock: float,
        entry_ts: float,
        primary_plan: ExecPlan,
        shadow_modes: Optional[list[str]] = None,
    ) -> None:
        if not self.enabled:
            return
        modes = shadow_modes or [ExecMode.SCALP.value, ExecMode.SWING.value]
        for mode in modes:
            profile_tag = f"shadow_{mode.lower()}"
            key = self._key(profile_tag, symbol)
            pos = _ShadowPosition(
                profile=profile_tag,
                mode=mode,
                symbol=symbol,
                direction=int(direction),
                entry_price=float(entry_price),
                entry_stock=float(entry_stock),
                entry_ts=float(entry_ts),
            )
            self._open[key] = pos
            self._events.append(
                {
                    "event": "shadow_entry",
                    "ts": entry_ts,
                    "symbol": symbol,
                    "mode": mode,
                    "primary_profile": primary_plan.profile,
                    "primary_mode": primary_plan.mode.value,
                    "entry_price": entry_price,
                    "entry_stock": entry_stock,
                }
            )

    def tick_exit_check(
        self,
        *,
        symbol: str,
        ctx: Mapping[str, Any],
        strategy: Any,
    ) -> None:
        if not self.enabled or strategy is None:
            return
        for key, pos in list(self._open.items()):
            if pos.closed or pos.symbol != symbol:
                continue
            holding = {
                "entry_price": pos.entry_price,
                "entry_stock": pos.entry_stock,
                "entry_ts": pos.entry_ts,
                "dir": pos.direction,
                "max_roi": pos.max_roi,
                "entry_spy_roc": 0.0,
                "entry_index_trend": 0,
                "exec_mode": pos.mode,
                "init_ctx": {"exec_mode": pos.mode},
            }
            ctx_copy = dict(ctx)
            ctx_copy["holding"] = holding
            curr_price = float(ctx_copy.get("curr_price", 0.0) or 0.0)
            if curr_price > 0 and pos.entry_price > 0:
                roi = (curr_price - pos.entry_price) / pos.entry_price
                pos.max_roi = max(pos.max_roi, roi)
                holding["max_roi"] = pos.max_roi
            sig = strategy.check_exit(ctx_copy)
            if sig and sig.get("action") == "SELL":
                pos.closed = True
                pos.exit_ts = float(ctx_copy.get("curr_ts", 0.0) or 0.0)
                pos.exit_roi = (
                    (curr_price - pos.entry_price) / pos.entry_price
                    if curr_price > 0 and pos.entry_price > 0
                    else 0.0
                )
                pos.exit_reason = str(sig.get("reason", ""))
                self._closed.append(pos)
                del self._open[key]
                self._events.append(
                    {
                        "event": "shadow_exit",
                        "ts": pos.exit_ts,
                        "symbol": symbol,
                        "mode": pos.mode,
                        "roi": pos.exit_roi,
                        "reason": pos.exit_reason,
                    }
                )

    def summary(self) -> Dict[str, Any]:
        by_mode: Dict[str, Dict[str, Any]] = {}
        for pos in self._closed:
            bucket = by_mode.setdefault(
                pos.mode,
                {"trades": 0, "wins": 0, "total_roi": 0.0, "reasons": {}},
            )
            bucket["trades"] += 1
            bucket["total_roi"] += pos.exit_roi
            if pos.exit_roi > 0:
                bucket["wins"] += 1
            bucket["reasons"][pos.exit_reason] = bucket["reasons"].get(pos.exit_reason, 0) + 1
        for mode, bucket in by_mode.items():
            n = max(1, bucket["trades"])
            bucket["avg_roi"] = bucket["total_roi"] / n
            bucket["win_rate"] = bucket["wins"] / n
        return {
            "enabled": self.enabled,
            "closed_trades": len(self._closed),
            "open_trades": len(self._open),
            "by_mode": by_mode,
        }

    def flush(self) -> None:
        if not self.enabled or not self.output_path:
            return
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"summary": self.summary(), "events": self._events}
        self.output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.info("ExecProfile shadow ledger written → %s", self.output_path)

"""Offline replay: TopK Rule-A → ATM/OTM contract → quote fills.

Contract modes:
  - day_lock (default): step1 open/day 4-bucket lock
  - signal_atm: re-pick ATM from day_iv at/before sig_ts (research)
"""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from maga7.common.contract_select import (
    DayIvChainCache,
    day_iv_path_as_quotes,
    lock_policy_from_profile,
    resolve_contract,
)
from maga7.common.open_lock import load_multidte_lock_index, resolve_open_lock_contract
from maga7.common.fills import FillSpec
from maga7.common.hold_watchdog import HoldWatchdogConfig, hold_watchdog_from_trade, qqq_adverse_from_entry
from maga7.common.position_size import (
    block_same_dir_after_win_enabled,
    is_symbol_dir_big_win,
    post_win_cooldown_action,
    post_win_cooldown_sessions,
    resolve_size_frac,
)
from maga7.common.option_trades import (
    TradeToxicConfig,
    load_option_trades,
    path_for_ticker_trades,
    prepare_quote_mark_arrays,
    prepare_trade_mark_arrays,
    trade_mtm_asof,
    trade_peak_mfe_asof,
    trade_toxic_from_trade,
)
from maga7.common.scale_in import (
    ScaleInConfig,
    blend_scale_in_ret,
    confirm_scale_in,
    scale_in_from_trade,
)
from maga7.common.event_calendar import resolve_event_blackout_plan
from maga7.common.corr_rewire import corr_rewire_asof, corr_rewire_from_trade
from maga7.common.adverse_vol_share import (
    AdverseVolShareConfig,
    EntryAdvVolConfig,
    adverse_vol_share_asof,
    adverse_vol_share_from_trade,
    entry_adv_vol_from_trade,
    prepare_stock_1s_arrays,
)
from maga7.common.dvol_size_scale import parse_dvol_size_scale, resolve_dvol_size_scale
from maga7.common.vrp_prior import (
    build_vrp_day_table,
    parse_vrp_size_scale,
    resolve_vrp_size_scale,
)
from maga7.common.from_open_gate import (
    parse_from_open_gate,
    resolve_from_open_gate,
    session_from_open,
)
from maga7.common.overnight_gap_gate import (
    parse_overnight_gap_gate,
    resolve_overnight_gap_gate,
)
from maga7.common.peer_gap_gate import (
    parse_peer_gap_gate,
    resolve_peer_gap_gate,
)
from maga7.common.range_stall_gate import (
    parse_range_stall_gate,
    resolve_range_stall_gate,
)
from maga7.common.dn_gap_stall_gate import (
    parse_dn_gap_stall_gate,
    resolve_dn_gap_stall_gate,
)
from maga7.common.up_gap_stall_gate import (
    parse_up_gap_stall_gate,
    resolve_up_gap_stall_gate,
)
from maga7.common.fo_lod_chase_gate import (
    parse_fo_lod_chase_gate,
    resolve_fo_lod_chase_gate,
)
from maga7.common.seat_score_gate import (
    candidate_gate_active,
    day_gate_armed,
    parse_seat_score_gate,
    seat_score_ok,
)
from maga7.common.delta_time_stop import (
    AdverseSoftConfig,
    DeltaTimeStopConfig,
    RoiTimeStopConfig,
    StockRevExitConfig,
    adverse_soft_from_trade,
    delta_time_stop_from_trade,
    morning_r5_scale_from_trade,
    roi_time_stop_from_trade,
    stock_rev_day_should_arm,
    stock_rev_exit_from_trade,
)
from maga7.common.wave_confirm import (
    WaveAbortConfig,
    WaveAbortState,
    wave_abort_from_trade,
    wave_abort_on_tick,
)
from maga7.common.path_fast_pack import (
    apply_path_fast_pack_overrides,
    path_fast_pack_day_should_arm,
    path_fast_pack_from_trade,
)
from maga7.common.ladder_active import (
    LadderActiveConfig,
    ladder_active_from_trade,
    ladder_day_should_arm,
)
from maga7.common.reentry import resolve_only_win_reenter
from maga7.common.trend_purity import (
    path_efficiency_features,
    trend_purity_score,
    trend_purity_size_scale,
)
from maga7.common.signals import (
    _rule_a_kwargs_from_cfg,
    all_rule_a_times,
    attach_mf_features,
    build_all_first_rule_a_signals,
    build_topk_signals,
    count_peer_align,
    load_stock_month_files,
    mf_idio_ok,
    price_efficiency_ok,
    resolve_mf_fast_window,
    rolling_idio_beta,
    sync_index,
    tod_mf_z_ok,
)
from maga7.common.lgbm_bouncer import load_lgbm_bouncer
from maga7.common.state_gate import load_state_gate
from maga7.common.chop_gate import load_chop_gate
from maga7.common.session_flow_gate import load_session_flow_gate
from maga7.common.tcn_gate import load_tcn_gate
from maga7.common.watchdog import (
    WATCHDOG_REGIME_KEYS,
    RegimeWatchdog,
    WatchdogState,
    apply_expert_dict,
    eval_router_rule,
    restore_regime,
    snapshot_regime,
)

# Regime keys that per-day experts may temporarily override.
_ROUTER_REGIME_KEYS = WATCHDOG_REGIME_KEYS


def _load_regime_router(
    profile: dict[str, Any],
) -> tuple[bool, dict[str, str], dict[str, dict], str, str]:
    """Return (enabled, date->day_type, experts, mode, rule_name).

    mode:
      - ``oracle`` / ``labels``: use labels map (research / upper bound)
      - ``rule``: evaluate causal morning rule each day (live-feasible)
    """
    raw = profile.get("regime_router")
    if not isinstance(raw, dict) or not bool(raw.get("enabled", False)):
        return False, {}, {}, "off", ""
    mode = str(raw.get("mode") or "oracle").strip().lower()
    if mode in {"labels", "label", "oracle"}:
        mode = "oracle"
    elif mode in {"rule", "rules", "causal"}:
        mode = "rule"
    else:
        mode = "oracle"
    rule_name = str(raw.get("rule") or raw.get("rule_name") or "reclaim_disp55").strip()
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
    experts: dict[str, dict] = {}
    ep = raw.get("experts_path")
    if ep:
        epth = Path(str(ep)).expanduser()
        if epth.is_file():
            experts = json.loads(epth.read_text(encoding="utf-8"))
    experts.update(raw.get("experts") or {})
    return True, labels, experts, mode, rule_name


def _eval_router_rule(
    rule_name: str,
    *,
    date: str,
    stock_by: dict[str, pd.DataFrame],
    qqq_df: pd.DataFrame | None,
    symbols: list[str],
    asof_hhmm: str = "10:30",
    router_cfg: dict[str, Any] | None = None,
) -> str | None:
    """Causal morning rule → expert name or None (baseline)."""
    return eval_router_rule(
        rule_name,
        date=date,
        stock_by=stock_by,
        qqq_df=qqq_df,
        symbols=symbols,
        asof_hhmm=asof_hhmm,
        router_cfg=router_cfg,
    )


def _router_snapshot_regime(cfg: dict[str, Any]) -> dict[str, Any]:
    return snapshot_regime(cfg)


def _router_restore_regime(cfg: dict[str, Any], snap: dict[str, Any]) -> None:
    restore_regime(cfg, snap)


def _router_apply_expert(cfg: dict[str, Any], expert: dict[str, Any] | None) -> None:
    apply_expert_dict(cfg, expert)


def _trade_flag(trade: dict[str, Any], key: str, default: bool = False) -> bool:
    raw = trade.get(key, default)
    if isinstance(raw, str):
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    return bool(raw)


def _displace_score(
    trade: dict[str, Any],
    *,
    from_prev: float | None,
    peer_n: int | None,
) -> float:
    """Causal strength score for later-signal displacement gates."""
    mode = str(trade.get("displace_score") or "none").strip().lower()
    if mode in {"", "none", "off", "any"}:
        return 1.0
    if mode in {"abs_from_prev", "from_prev", "fp"}:
        if from_prev is None or not np.isfinite(float(from_prev)):
            return 0.0
        return abs(float(from_prev))
    if mode in {"peer_n", "peer"}:
        return float(peer_n or 0)
    return 1.0


def _parse_commit_tod(raw: Any) -> str | None:
    if raw is None:
        return None
    s = str(raw).strip().lower()
    if s in {"", "none", "off", "false", "0"}:
        return None
    if ":" not in s:
        return None
    hh, mm = s.split(":", 1)
    return f"{int(hh):02d}:{int(mm):02d}"


def _topk_rank_score(
    mode: str,
    *,
    from_prev: float | None,
    peer_n: int | None,
) -> float:
    """Score for deferred TopK commit auction (higher wins)."""
    m = str(mode or "abs_from_prev").strip().lower()
    try:
        afp = abs(float(from_prev)) if from_prev is not None and np.isfinite(float(from_prev)) else 0.0
    except (TypeError, ValueError):
        afp = 0.0
    pn = float(peer_n or 0)
    if m in {"peer_n", "peer"}:
        return pn
    if m in {"peer_fp", "peer_x_fp", "peer*fp", "peer_times_fp"}:
        return pn * afp
    # default: abs_from_prev
    return afp


def entry_confirm_ok(
    stock_day: pd.DataFrame | None,
    *,
    direction: str,
    feature_ts,
    confirm_bars: int,
    mode: str = "mf",
) -> tuple[bool, pd.Timestamp | None, float | None, int, int]:
    """After Rule-A, require N more 1m bars still aligned before entry.

    Returns ``(ok, confirm_feature_ts, mf, streak_up, streak_dn)``.
    ``confirm_feature_ts = feature_ts + confirm_bars minutes`` (bar clock).
    """
    n = int(confirm_bars or 0)
    if n <= 0:
        return True, None, None, 0, 0
    confirm_ft = to_ny(feature_ts) + pd.Timedelta(minutes=n)
    day = _prepare_stock_day(stock_day)
    mf, su, sd = _stock_mf_at(day, confirm_ft)
    m = str(mode or "mf").strip().lower()
    dir_u = str(direction).upper()
    mf_ok = True
    streak_ok = True
    if m in {"mf", "mf10", "both", "mf_streak", "all"}:
        if mf is None:
            mf_ok = False
        elif dir_u == "UP":
            mf_ok = mf > 0
        elif dir_u == "DN":
            mf_ok = mf < 0
        else:
            mf_ok = False
    if m in {"streak", "both", "mf_streak", "all"}:
        if dir_u == "UP":
            streak_ok = su > 0
        elif dir_u == "DN":
            streak_ok = sd > 0
        else:
            streak_ok = False
    if m in {"streak"}:
        ok = streak_ok
    elif m in {"both", "mf_streak", "all"}:
        ok = mf_ok and streak_ok
    else:
        ok = mf_ok
    return ok, confirm_ft, mf, su, sd


def _hhmm_to_minutes(raw: Any) -> int | None:
    if raw is None:
        return None
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        return int(raw)
    s = str(raw).strip()
    if not s:
        return None
    if ":" in s:
        hh, mm = s.split(":", 1)
        return int(hh) * 60 + int(mm)
    return int(float(s))


def stock_path_confirm_ok(
    stock_day: pd.DataFrame | None,
    *,
    direction: str,
    entry_ts,
    thr_pos: float = 0.0015,
    thr_neg: float = -0.003,
    max_wait_seconds: int = 300,
    on_timeout: str = "block",
    asof_ts=None,
) -> tuple[bool, pd.Timestamp | None, str]:
    """Causal first-touch path confirm after the fill clock starts.

    From the latest close at/before ``entry_ts``, walk later 1m bars up to
    ``max_wait_seconds``. Keep if signed move reaches ``thr_pos`` before
    ``thr_neg``; cancel on adverse-first.

    ``on_timeout``:
      - ``block`` (default): no +thr_pos within wait → cancel
      - ``allow``: only cancel on adverse-first; timeout keeps original ``entry_ts``

    ``asof_ts`` (live / streaming): only bars at/before ``asof_ts`` are visible.
    If no decision yet and ``asof_ts < deadline``, returns
    ``(False, None, "pending")`` so the caller can wait. Offline batch leaves
    ``asof_ts=None`` and resolves timeout immediately from the full day.

    Returns ``(ok, confirm_bar_ts, reason)``. On ``allow`` timeout,
    ``confirm_bar_ts`` is None (caller keeps original fill clock).
    """
    day = _prepare_stock_day(stock_day)
    if day is None or day.empty or "_close" not in day.columns:
        return False, None, "no_stock"
    et = to_ny(entry_ts)
    i0 = _stock_bar_index(day, et)
    if i0 < 0:
        return False, None, "no_anchor"
    c0 = float(day["_close"].iloc[i0])
    if not np.isfinite(c0) or c0 <= 0:
        return False, None, "bad_anchor"
    dir_u = str(direction).upper()
    if dir_u not in {"UP", "DN"}:
        return False, None, "bad_dir"
    deadline = et + pd.Timedelta(seconds=int(max_wait_seconds))
    asof = to_ny(asof_ts) if asof_ts is not None else None
    thr_p = float(thr_pos)
    thr_n = float(thr_neg)
    for i in range(i0 + 1, len(day)):
        ts_i = to_ny(day["timestamp"].iloc[i])
        if ts_i > deadline:
            break
        if asof is not None and ts_i > asof:
            break
        px = float(day["_close"].iloc[i])
        if not np.isfinite(px) or px <= 0:
            continue
        signed = (px / c0 - 1.0) if dir_u == "UP" else (1.0 - px / c0)
        if signed >= thr_p:
            return True, ts_i, "pos"
        if signed <= thr_n:
            return False, ts_i, "neg"
    if asof is not None and asof < deadline:
        return False, None, "pending"
    mode = str(on_timeout or "block").strip().lower()
    if mode in {"allow", "pass", "keep"}:
        return True, None, "timeout_allow"
    return False, None, "timeout"

NY = "America/New_York"
BUCKET_MAP = {
    ("UP", "ATM"): 2,
    ("UP", "OTM"): 3,
    ("DN", "ATM"): 0,
    ("DN", "OTM"): 1,
}


def to_ny(ts) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        return t.tz_localize(NY)
    return t.tz_convert(NY)


def month_list(start: str, end: str) -> list[str]:
    idx = pd.period_range(start=start[:7], end=end[:7], freq="M")
    return [str(p) for p in idx]


def load_lock_index(lock_path: Path) -> dict[tuple[str, str], dict[int, str]]:
    lock = pd.read_parquet(lock_path)
    lock["contract"] = lock["contract_symbol"].astype(str).str.replace("O:", "", regex=False)
    out: dict[tuple[str, str], dict[int, str]] = {}
    for (sym, date), g in lock.groupby(["symbol", "date_str"]):
        out[(str(sym), str(date))] = {int(r.bucket_id): str(r.contract) for r in g.itertuples()}
    return out


def load_quotes(quote_root: Path, symbol: str, date: str) -> pd.DataFrame | None:
    p = quote_root / symbol / f"{symbol}_{date}.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    if df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize(NY)
    else:
        df["timestamp"] = df["timestamp"].dt.tz_convert(NY)
    return df


def path_for_ticker(qdf: pd.DataFrame | None, ticker: str) -> pd.DataFrame | None:
    if qdf is None or qdf.empty:
        return None
    t = str(ticker).replace("O:", "")
    sub = qdf[qdf["ticker"].astype(str).str.replace("O:", "", regex=False) == t].sort_values("timestamp")
    if sub.empty:
        return None
    return sub.drop_duplicates("timestamp", keep="last")


@dataclass
class SimResult:
    entry: float
    exit: float
    ret: float
    reason: str
    entry_ts: pd.Timestamp
    exit_ts: pd.Timestamp
    scale_in_added: bool = False
    scale_in_entry2: float | None = None
    scale_in_ts: pd.Timestamp | None = None
    scale_in_deployed_frac: float = 1.0
    adverse_soft_armed: bool = False
    adv_vol_armed: bool = False


def _stock_mf_at(stock_day: pd.DataFrame | None, t: pd.Timestamp) -> tuple[float | None, int, int]:
    """Latest causal mf10 / streaks at or before ``t``.

    Prefer precomputed columns ``_ts_ns`` / ``_mf`` / ``_su`` / ``_sd`` for speed.
    """
    if stock_day is None or stock_day.empty:
        return None, 0, 0
    if "_ts_ns" in stock_day.columns:
        ts_ns = stock_day["_ts_ns"].to_numpy()
        i = int(np.searchsorted(ts_ns, t.value, side="right") - 1)
        if i < 0:
            return None, 0, 0
        mf = float(stock_day["_mf"].iloc[i])
        if not np.isfinite(mf):
            return None, 0, 0
        return mf, int(stock_day["_su"].iloc[i]), int(stock_day["_sd"].iloc[i])
    ts = stock_day["timestamp"]
    mask = ts <= t
    if not mask.any():
        return None, 0, 0
    row = stock_day.loc[mask].iloc[-1]
    mf = float(row["mf10"]) if np.isfinite(row["mf10"]) else None
    su = int(row["streak_up"]) if "streak_up" in stock_day.columns else 0
    sd = int(row["streak_dn"]) if "streak_dn" in stock_day.columns else 0
    return mf, su, sd


def _prepare_stock_day(stock_day: pd.DataFrame | None) -> pd.DataFrame | None:
    if stock_day is None or stock_day.empty:
        return None
    day = stock_day.copy()
    day["timestamp"] = [to_ny(x) for x in day["timestamp"].tolist()]
    day = day.sort_values("timestamp")
    day["_ts_ns"] = [int(to_ny(x).value) for x in day["timestamp"].tolist()]
    if "mf10" not in day.columns and {"open", "high", "low", "close", "volume", "date"}.issubset(
        set(day.columns)
    ):
        from maga7.common.signals import attach_mf_features

        day = attach_mf_features(day)
    if "mf10" not in day.columns:
        day["mf10"] = np.nan
    day["_mf"] = day["mf10"].astype(float)
    if "mf_fast" in day.columns:
        day["_mf_fast"] = day["mf_fast"].astype(float)
    elif "mf_short" in day.columns:
        day["_mf_fast"] = day["mf_short"].astype(float)
    elif "net$" in day.columns:
        day["_mf_fast"] = day["net$"].astype(float).rolling(3, min_periods=3).sum()
    else:
        day["_mf_fast"] = np.nan
    day["_su"] = day["streak_up"].astype(int) if "streak_up" in day.columns else 0
    day["_sd"] = day["streak_dn"].astype(int) if "streak_dn" in day.columns else 0
    if "close" in day.columns:
        day["_close"] = day["close"].astype(float)
    if "low" in day.columns:
        day["_low"] = day["low"].astype(float)
    if "high" in day.columns:
        day["_high"] = day["high"].astype(float)
    if "net$" in day.columns:
        day["_net"] = day["net$"].astype(float)
        day["_net_csum"] = day["_net"].cumsum()
    return day


def _stock_bar_index(day: pd.DataFrame | None, t: pd.Timestamp) -> int:
    if day is None or day.empty or "_ts_ns" not in day.columns:
        return -1
    return int(np.searchsorted(day["_ts_ns"].to_numpy(), t.value, side="right") - 1)


def _stock_close_at(day: pd.DataFrame | None, t: pd.Timestamp) -> float | None:
    """Latest causal stock close at/before ``t`` (requires ``_prepare_stock_day``)."""
    i = _stock_bar_index(day, t)
    if i < 0 or day is None or "_close" not in day.columns:
        return None
    px = float(day["_close"].iloc[i])
    return px if np.isfinite(px) and px > 0 else None


def _session_structure_at(
    sdf: pd.DataFrame | None,
    *,
    date: str,
    asof_ts: pd.Timestamp,
) -> dict[str, float | bool | None] | None:
    """Causal session stats at asof: open, vwap, lod, px, above_open, above_vwap, bounce_from_lod."""
    if sdf is None or sdf.empty or "close" not in sdf.columns:
        return None
    day = sdf[sdf["date"].astype(str) == str(date)]
    if day.empty:
        return None
    day = day.sort_values("timestamp")
    asof = to_ny(asof_ts)
    # timestamps may already be tz-aware
    ts = day["timestamp"].map(to_ny)
    upto = day.loc[ts <= asof]
    if upto.empty:
        return None
    close = pd.to_numeric(upto["close"], errors="coerce")
    px = float(close.iloc[-1])
    if not np.isfinite(px):
        return None
    open_col = "open" if "open" in upto.columns else "close"
    o = pd.to_numeric(upto[open_col], errors="coerce").dropna()
    if o.empty:
        return None
    open0 = float(o.iloc[0])
    low_col = "low" if "low" in upto.columns else "close"
    lod = float(pd.to_numeric(upto[low_col], errors="coerce").min())
    vol = (
        pd.to_numeric(upto["volume"], errors="coerce").fillna(0.0)
        if "volume" in upto.columns
        else pd.Series(0.0, index=upto.index)
    )
    if float(vol.sum()) > 0:
        vwap = float((close * vol).sum() / vol.sum())
    else:
        vwap = float(close.expanding().mean().iloc[-1])
    if not np.isfinite(open0) or open0 <= 0 or not np.isfinite(lod) or lod <= 0 or not np.isfinite(vwap):
        return None
    bounce = (px / lod - 1.0) if lod > 0 else None
    return {
        "px": px,
        "open": open0,
        "vwap": vwap,
        "lod": lod,
        "above_open": bool(px > open0),
        "above_vwap": bool(px > vwap),
        "bounce_from_lod": float(bounce) if bounce is not None else None,
    }


def _above_session_open(
    sdf: pd.DataFrame | None,
    *,
    date: str,
    asof_ts: pd.Timestamp,
) -> bool | None:
    """True if last close at/before asof is strictly above that day's first open."""
    st = _session_structure_at(sdf, date=date, asof_ts=asof_ts)
    if st is None:
        return None
    return bool(st["above_open"])


def _structure_gate_blocks(
    sdf: pd.DataFrame | None,
    *,
    date: str,
    asof_ts: pd.Timestamp,
    direction: str,
    block_dn_if_above_open: bool = False,
    vwap_dir_lock: bool = False,
    block_dn_if_vwap_lod: bool = False,
    lod_bounce_min: float = 0.02,
) -> str | None:
    """Return block reason or None if allowed."""
    st = _session_structure_at(sdf, date=date, asof_ts=asof_ts)
    if st is None:
        return None
    d = str(direction).upper()
    if block_dn_if_above_open and d == "DN" and st["above_open"] is True:
        return "dn_above_open"
    if vwap_dir_lock:
        if d == "UP" and st["above_vwap"] is False:
            return "vwap_lock_up"
        if d == "DN" and st["above_vwap"] is True:
            return "vwap_lock_dn"
    if block_dn_if_vwap_lod and d == "DN":
        bounce = st.get("bounce_from_lod")
        if (
            st["above_vwap"] is True
            and bounce is not None
            and float(bounce) >= float(lod_bounce_min)
        ):
            return "dn_vwap_lod_bounce"
    return None


def _fav_mf(raw: float | None, direction: str) -> float | None:
    if raw is None or not np.isfinite(raw):
        return None
    return float(raw) if direction == "UP" else -float(raw)


def _cum_fav_flow(
    day: pd.DataFrame | None,
    entry_ts: pd.Timestamp,
    visible_at: pd.Timestamp,
    direction: str,
) -> float | None:
    """Post-entry cumulative favorable money flow (UP=inflow, DN=outflow)."""
    if day is None or day.empty or "_ts_ns" not in day.columns or "_net_csum" not in day.columns:
        return None
    if direction not in ("UP", "DN"):
        return None
    ts_ns = day["_ts_ns"].to_numpy()
    csum = day["_net_csum"].to_numpy()
    i0 = int(np.searchsorted(ts_ns, entry_ts.value, side="left"))
    i1 = int(np.searchsorted(ts_ns, visible_at.value, side="right") - 1)
    if i1 < i0 or i0 >= len(csum):
        return None
    prev = float(csum[i0 - 1]) if i0 > 0 else 0.0
    raw = float(csum[i1]) - prev
    if not np.isfinite(raw):
        return None
    return raw if direction == "UP" else -raw


def simulate_trade(
    path: pd.DataFrame | None,
    entry_ts,
    *,
    fill: FillSpec,
    tp_mult: float = 1.6,
    sl_mult: float = 0.4,
    hold_minutes: int = 30,
    direction: str | None = None,
    stock_day: pd.DataFrame | None = None,
    exit_mode: str | None = None,
    exit_mf_grace_seconds: int = 60,
    exit_min_hold_minutes: float | None = None,
    mtm_floor_ret: float | None = None,
    flow_cum_floor: float | None = None,
    stock_bar_delay_seconds: int = 0,
    trail_activate: float | None = None,
    trail_dd: float | None = None,
    hold_extend_minutes: int | None = None,
    hold_extend_mtm_min: float | None = None,
    hold_extend_require_mf: bool = True,
    hold_extend_require_stock: bool = False,
    hold_extend_stock_min: float = 0.0,
    hold_extend_min_peak_mfe: float | None = None,
    hold_extend_max_qqq_adverse: float | None = None,
    stale_cut_minutes: float | None = None,
    stale_cut_mtm_max: float = 0.0,
    stale_cut_stock_max: float = 0.0,
    force_exit_ts=None,
    early_exit_mode: str | None = None,
    mae_cut_ret: float | None = None,
    mae_cut_mfe_bypass: float | None = None,
    mae_cut_min_hold_minutes: float | None = None,
    mae_cut_only_dn: bool = False,
    mae_cut_require_mf_against: bool = False,
    dyn_max_hold_minutes: int | None = None,
    dyn_min_hold_minutes: float | None = None,
    dyn_trail_start_minutes: float | None = None,
    dyn_slope_lookback: int | None = None,
    dyn_fast_opp_bars: int | None = None,
    dyn_fast_pct: float | None = None,
    dyn_require_price_break: bool = True,
    qqq_day: pd.DataFrame | None = None,
    hold_watchdog: dict[str, Any] | HoldWatchdogConfig | None = None,
    scale_in: dict[str, Any] | ScaleInConfig | None = None,
    trade_path: pd.DataFrame | None = None,
    trade_toxic: dict[str, Any] | TradeToxicConfig | None = None,
    delta_time_stop: dict[str, Any] | DeltaTimeStopConfig | None = None,
    roi_time_stop: dict[str, Any] | RoiTimeStopConfig | None = None,
    adverse_soft: dict[str, Any] | AdverseSoftConfig | None = None,
    adverse_vol_share: dict[str, Any] | AdverseVolShareConfig | None = None,
    stock_rev_exit: dict[str, Any] | StockRevExitConfig | None = None,
    wave_abort: dict[str, Any] | WaveAbortConfig | None = None,
    stock_1s: pd.DataFrame | None = None,
    ladder_active: dict[str, Any] | LadderActiveConfig | None = None,
) -> SimResult | None:
    """Option path fill with TP/SL/time, optional stock-window or MTM trail exit.

    ``hold_watchdog``: optional mid-hold flatten when QQQ moves hard against the
    trade (see ``maga7.common.hold_watchdog``). Exit reason ``HOLD_SHOCK``.

    ``delta_time_stop``: if underlying signed move stays below threshold after
    ``check_seconds`` while option MTM is non-positive, exit ``DELTA_STOP``.

    ``stock_rev_exit``: after ``min_hold_minutes``, if signed stock from fill
    ``<= stock_max`` and option MTM ``<= opt_mtm_max``, exit ``STOCK_REV``.

    ``wave_abort``: post-fill revocable wave confirm → exit ``WAVE_ABORT``
    (see ``maga7.common.wave_confirm`` / ``docs/wave_confirm_spec.md``).

    ``adverse_soft``: after ``check_seconds``, if signed stock MAE is deep adverse
    and option MTM is non-positive, either tighten trade_toxic (``tox_tighten``)
    or flatten ``ADVERSE_SOFT``. Default OFF — softer than hard DELTA_STOP.

    ``adverse_vol_share`` + ``stock_1s``: short-window share of volume on
    adverse ticks; ``tox_tighten`` or flatten ``ADV_VOL``. Default OFF.

    ``roi_time_stop``: V0-style option ROI progress rails — at each ``mins``,
    if MTM ``< min_roi`` exit ``ROI_TIME{mins}``.

    ``scale_in``: optional split entry — first tranche at signal fill; on option
    MTM pullback + secondary factor confirm, add second tranche (see
    ``maga7.common.scale_in``). Blended ``ret`` keeps ``size_frac * ret`` valid.

    ``trade_path`` / ``trade_toxic``: mark toxic path on trade last prints
    (MFE bypass + cut); exit fill still uses quote ``sell_px`` → reason
    ``TRADE_TOX``. See ``maga7.common.option_trades``.

    ``exit_mode``:
      - ``None`` / ``none`` / ``rails``: TP/SL/T+hold only (default)
      - ``mf_flip``: exit when mf10 flips against ``direction`` (after grace)
      - ``mf_reversal``: same as mf_flip but default min-hold ~10m before watching
        (V-reversals often start ~10m after entry; avoids the old 60s grace trap)
      - ``streak_break``: exit when entry-dir streak resets and mf opposing
      - ``mtm_trail`` / ``trail``: after peak MTM ret >= ``trail_activate``,
        exit when ret falls by ``trail_dd`` from that peak (still respects TP/SL/T+hold)
      - ``mtm_floor`` / ``mtm_defend``: after min-hold (default 10m), exit if MTM ret
        <= ``mtm_floor_ret`` (default 0). Targets early V givebacks without shortening winners.
      - ``flow_die`` / ``cum_fav``: after min-hold (default 5m), exit if post-entry
        cumulative favorable stock net$ <= ``flow_cum_floor`` (default 0).
        UP uses inflow (net$); DN uses outflow (-net$).
      - ``flow_mtm`` / ``flow_soft``: same flow check, but only exit when option MTM
        ret also <= ``mtm_floor_ret`` (default 0). Softens false cuts on winners.
      - ``hold_extend`` / ``extend_hold``: at base ``hold_minutes``, if option MTM
        >= ``hold_extend_mtm_min`` (default 0) and (optional) mf10 still aligned,
        extend deadline to ``hold_extend_minutes`` (default 45). Rails still apply.
        Optional feature gates (default off): ``hold_extend_require_stock`` /
        ``hold_extend_stock_min``, ``hold_extend_min_peak_mfe``,
        ``hold_extend_max_qqq_adverse``. Mid-hold ``stale_cut_minutes`` can flatten
        losers early (MTM≤``stale_cut_mtm_max`` and stock≤``stale_cut_stock_max``)
        without abolishing the T30 clock for healthy trades.
      - ``mae_cut`` / ``toxic_cut``: after min-hold (default 5m), if peak MFE
        < ``mae_cut_mfe_bypass`` (default 0.20) and MTM ret <= -``mae_cut_ret``
        (default 0.25), exit ``MAE_CUT``. Cuts toxic reversals before SL (-60%)
        without clipping winners that already printed MFE. Optional
        ``mae_cut_only_dn``. Set ``mae_cut_require_mf_against`` to also require
        stock mf10 against the trade (avoids locking V-recovery troughs).
      - ``dyn_trail`` / ``mf_dual``: drop static T+hold; trail with fast/slow MF
        windows (default 3m/10m already on stock bars). Soft exits:
        FAST_REVERSAL (min-hold), MOM_EXHAUST (trail-start + fast pctile +
        slow slope), TREND_DEAD (slow fav MF < 0), hard ``dyn_max_hold_minutes``.
      - ``ladder_active`` / ``sec_active`` / ``hft_ladder``: second-level active
        management — hard ``max_hold_seconds``, stepped TP/SL rails, profit-stall,
        optional short-grace mf flip. Never passive T+30. See
        ``maga7.common.ladder_active`` / ``docs/sec_ladder_active_research.md``.

    Soft cuts may be **stacked** on extend via ``exit_mode="hold_extend+mtm_floor"``
    (or ``+mf_flip`` / ``+mf_reversal`` / ``+mae_cut``), or via ``early_exit_mode``.

    ``force_exit_ts``: if set, exit at the first quote at/after this time with
    reason ``DISPLACE`` (after TP/SL checks). Used by later-signal displacement.
    """
    if path is None or path.empty:
        return None
    entry_ts = to_ny(entry_ts)
    force_ts = to_ny(force_exit_ts) if force_exit_ts is not None else None
    after = path[path["timestamp"] >= entry_ts]
    if after.empty:
        return None
    bid0 = float(after.iloc[0]["bid"])
    ask0 = float(after.iloc[0]["ask"])
    entry = fill.buy(bid0, ask0)
    if not np.isfinite(entry) or entry <= 0:
        return None
    bids = after["bid"].astype(float).to_numpy()
    asks = after["ask"].astype(float).to_numpy()
    sell_px = fill.sell_series(bids, asks)
    ts_list = [to_ny(x) for x in after["timestamp"].tolist()]
    base_hold = int(hold_minutes)
    base_end = entry_ts + pd.Timedelta(minutes=base_hold)
    end_ts = base_end
    tp_lvl, sl_lvl = entry * tp_mult, entry * sl_mult
    reason, exit_px, exit_ts = f"T+{base_hold}", float(sell_px[-1]), ts_list[-1]
    mode = str(exit_mode or "none").strip().lower()
    early = str(early_exit_mode or "").strip().lower()
    if early and early not in {"", "none", "off"} and early not in mode:
        mode = f"{mode}+{early}" if mode not in {"", "none"} else early
    blob = mode.replace(",", "+").replace("|", "+")
    use_mf_flip = ("mf_flip" in blob or "mf_reversal" in blob) and direction in ("UP", "DN")
    use_streak = "streak_break" in blob and direction in ("UP", "DN")
    use_mf = use_mf_flip or use_streak
    use_trail = "mtm_trail" in blob or blob in {"trail"}
    use_floor = "mtm_floor" in blob or "mtm_defend" in blob
    use_flow = any(x in blob for x in ("flow_die", "cum_fav", "flow_mtm", "flow_soft")) and direction in (
        "UP",
        "DN",
    )
    use_extend = ("hold_extend" in blob or "extend_hold" in blob) and direction in ("UP", "DN")
    use_mae_cut = "mae_cut" in blob or "toxic_cut" in blob
    if use_mae_cut and bool(mae_cut_only_dn) and str(direction or "").upper() != "DN":
        use_mae_cut = False
    if isinstance(trade_toxic, TradeToxicConfig):
        ttox = trade_toxic
    elif isinstance(trade_toxic, dict):
        ttox = trade_toxic_from_trade({"trade_toxic": trade_toxic})
    else:
        ttox = TradeToxicConfig(enabled=False)
    use_trade_toxic = (
        bool(ttox.enabled)
        or "trade_toxic" in blob
        or "trade_mae" in blob
        or "toxic_trade" in blob
    )
    if isinstance(delta_time_stop, DeltaTimeStopConfig):
        dts = delta_time_stop
    elif isinstance(delta_time_stop, dict):
        dts = delta_time_stop_from_trade({"delta_time_stop": delta_time_stop})
    else:
        dts = DeltaTimeStopConfig(enabled=False)
    use_delta_stop = bool(dts.enabled) and direction in ("UP", "DN")
    delta_check_until = None
    delta_check_deadline = None
    if isinstance(roi_time_stop, RoiTimeStopConfig):
        rts = roi_time_stop
    elif isinstance(roi_time_stop, dict):
        rts = roi_time_stop_from_trade({"roi_time_stop": roi_time_stop})
    else:
        rts = RoiTimeStopConfig(enabled=False)
    use_roi_time = bool(rts.enabled) and bool(rts.rails)
    roi_fired: set[float] = set()
    if isinstance(adverse_soft, AdverseSoftConfig):
        adv = adverse_soft
    elif isinstance(adverse_soft, dict):
        adv = adverse_soft_from_trade({"adverse_soft": adverse_soft})
    else:
        adv = AdverseSoftConfig(enabled=False)
    use_adv_soft = bool(adv.enabled) and direction in ("UP", "DN")
    if isinstance(stock_rev_exit, StockRevExitConfig):
        srev = stock_rev_exit
    elif isinstance(stock_rev_exit, dict):
        srev = stock_rev_exit_from_trade({"stock_rev_exit": stock_rev_exit})
    else:
        srev = StockRevExitConfig(enabled=False)
    use_stock_rev = bool(srev.enabled) and direction in ("UP", "DN")
    stock_rev_until = None
    if isinstance(wave_abort, WaveAbortConfig):
        wabort = wave_abort
    elif isinstance(wave_abort, dict):
        wabort = wave_abort_from_trade({"wave_abort": wave_abort})
    else:
        wabort = WaveAbortConfig(enabled=False)
    _wa_dirs = getattr(wabort, "only_directions", None) or ("UP", "DN")
    use_wave_abort = bool(wabort.enabled) and direction in _wa_dirs
    wave_state = WaveAbortState()
    adv_check_until = None
    adv_armed = False
    stock_mae = 0.0
    stock_signed_now = 0.0
    opt_quote_peak = -np.inf
    if isinstance(adverse_vol_share, AdverseVolShareConfig):
        avs = adverse_vol_share
    elif isinstance(adverse_vol_share, dict):
        avs = adverse_vol_share_from_trade({"adverse_vol_share": adverse_vol_share})
    else:
        avs = AdverseVolShareConfig(enabled=False)
    use_adv_vol = bool(avs.enabled) and direction in ("UP", "DN")
    adv_vol_check_until = None
    adv_vol_armed = False
    stock_1s_arr = prepare_stock_1s_arrays(stock_1s) if use_adv_vol else None
    if use_adv_vol and stock_1s_arr is None:
        use_adv_vol = False
    use_dyn = ("dyn_trail" in blob or "mf_dual" in blob) and direction in ("UP", "DN")
    if isinstance(ladder_active, LadderActiveConfig):
        lac = ladder_active
    elif isinstance(ladder_active, dict):
        lac = ladder_active_from_trade({"ladder_active": ladder_active, "exit_mode": mode})
    else:
        lac = ladder_active_from_trade({"exit_mode": mode})
    use_ladder = bool(lac.enabled)
    if use_ladder and lac.mf_flip and direction in ("UP", "DN"):
        use_mf_flip = True
        use_mf = True
    # Ladder owns its own trail rails; don't double-arm classic mtm_trail / extend.
    if use_ladder:
        use_trail = False
        use_extend = False
    require_mtm = "flow_mtm" in blob or "flow_soft" in blob
    act = float(trail_activate) if trail_activate is not None else 0.20
    dd = float(trail_dd) if trail_dd is not None else 0.15
    ladder_trail_dd = 0.05
    ladder_trail_armed = False
    peak_ts: pd.Timestamp | None = None
    floor = float(mtm_floor_ret) if mtm_floor_ret is not None else 0.0
    flow_floor = float(flow_cum_floor) if flow_cum_floor is not None else 0.0
    mae_thr = float(mae_cut_ret) if mae_cut_ret is not None else 0.25
    mae_bypass = float(mae_cut_mfe_bypass) if mae_cut_mfe_bypass is not None else 0.20
    mae_thr = max(0.0, mae_thr)
    mae_bypass = max(0.0, mae_bypass)
    ext_hold = int(hold_extend_minutes) if hold_extend_minutes is not None else 45
    if ext_hold <= base_hold:
        ext_hold = base_hold
    ext_end = entry_ts + pd.Timedelta(minutes=ext_hold)
    ext_mtm_min = float(hold_extend_mtm_min) if hold_extend_mtm_min is not None else 0.0
    require_mf_align = bool(hold_extend_require_mf)
    require_stock_align = bool(hold_extend_require_stock)
    ext_stock_min = float(hold_extend_stock_min)
    ext_min_peak = (
        float(hold_extend_min_peak_mfe) if hold_extend_min_peak_mfe is not None else None
    )
    ext_max_qqq_adv = (
        float(hold_extend_max_qqq_adverse)
        if hold_extend_max_qqq_adverse is not None
        else None
    )
    use_stale_cut = (
        stale_cut_minutes is not None
        and float(stale_cut_minutes) > 0
        and direction in ("UP", "DN")
    )
    stale_until = (
        entry_ts + pd.Timedelta(minutes=float(stale_cut_minutes))
        if use_stale_cut
        else None
    )
    stale_mtm_max = float(stale_cut_mtm_max)
    stale_stock_max = float(stale_cut_stock_max)
    extended = False
    peak_ret = -np.inf
    trail_armed = False
    # dyn_trail replaces static T+hold with a hard max clock.
    dyn_max_m = int(dyn_max_hold_minutes) if dyn_max_hold_minutes is not None else 60
    dyn_min_m = float(dyn_min_hold_minutes) if dyn_min_hold_minutes is not None else 5.0
    dyn_start_m = float(dyn_trail_start_minutes) if dyn_trail_start_minutes is not None else 15.0
    dyn_lb = max(1, int(dyn_slope_lookback) if dyn_slope_lookback is not None else 3)
    dyn_opp_n = max(1, int(dyn_fast_opp_bars) if dyn_fast_opp_bars is not None else 2)
    dyn_pct = float(dyn_fast_pct) if dyn_fast_pct is not None else 20.0
    if use_dyn:
        end_ts = entry_ts + pd.Timedelta(minutes=dyn_max_m)
        reason = f"T+{dyn_max_m}"
    if isinstance(hold_watchdog, HoldWatchdogConfig):
        hwd = hold_watchdog
    elif isinstance(hold_watchdog, dict):
        hwd = hold_watchdog_from_trade({"hold_watchdog": hold_watchdog})
    else:
        hwd = HoldWatchdogConfig(enabled=False)
    use_hold_wd = bool(hwd.enabled) and direction in ("UP", "DN") and qqq_day is not None
    hold_wd_until = entry_ts + pd.Timedelta(seconds=int(hwd.min_hold_seconds))
    if isinstance(scale_in, ScaleInConfig):
        sic = scale_in
    elif isinstance(scale_in, dict):
        sic = scale_in_from_trade({"scale_in": scale_in})
    else:
        sic = ScaleInConfig(enabled=False)
    use_scale_in = bool(sic.enabled) and direction in ("UP", "DN")
    scale_grace = entry_ts + pd.Timedelta(seconds=int(sic.min_hold_seconds))
    scale_deadline = (
        entry_ts + pd.Timedelta(seconds=int(sic.max_wait_seconds))
        if use_scale_in and sic.max_wait_seconds is not None
        else None
    )
    entry2: float | None = None
    entry2_ts: pd.Timestamp | None = None
    # Anchor trade marks at the *quote fill* clock (first usable quote bar), not
    # signal ts. entry_confirm / quote gaps can leave 10m+ between sig and fill;
    # pre-fill prints at stale highs create phantom TRADE_TOX (e.g. AMD 05-15).
    fill_ts = ts_list[0]
    if use_ladder:
        ladder_end = fill_ts + pd.Timedelta(seconds=int(lac.max_hold_seconds))
        end_ts = ladder_end
        base_end = ladder_end
        reason = "SEC_MAX"
        if not lac.keep_outer_rails:
            tp_lvl = float("inf")
            sl_lvl = 0.0
    trade_mark = None
    trade_mark_source = "none"
    trade_peak_mfe = -np.inf
    trade_cut_until = fill_ts + pd.Timedelta(seconds=int(ttox.min_hold_seconds))
    trade_dig_since: pd.Timestamp | None = None
    trade_cut_deadline = (
        fill_ts + pd.Timedelta(seconds=int(ttox.max_cut_seconds))
        if use_trade_toxic and ttox.max_cut_seconds is not None and int(ttox.max_cut_seconds) > 0
        else None
    )
    use_div_mfe = (
        use_trade_toxic
        and ttox.div_mfe_bypass is not None
        and ttox.div_stock_adverse_max is not None
        and direction in ("UP", "DN")
    )
    if use_trade_toxic:
        trade_mark = prepare_trade_mark_arrays(trade_path, fill_ts)
        if trade_mark is not None:
            trade_mark_source = "prints"
        elif bool(ttox.quote_fallback):
            trade_mark = prepare_quote_mark_arrays(
                ts_list, sell_px, entry_px=float(entry), fill_ts=fill_ts
            )
            if trade_mark is not None:
                trade_mark_source = "quote"
        if trade_mark is None:
            use_trade_toxic = False
            use_div_mfe = False
            trade_mark_source = "none"
    # mf_reversal / mtm_floor: wait ~10m; flow_* / mae_cut: wait ~5m before soft exit.
    min_hold_m = exit_min_hold_minutes
    if min_hold_m is None and ("mf_reversal" in blob or use_floor):
        min_hold_m = 10.0
    if min_hold_m is None and use_flow:
        min_hold_m = 5.0
    if use_mae_cut:
        if mae_cut_min_hold_minutes is not None:
            min_hold_m = float(mae_cut_min_hold_minutes)
        elif min_hold_m is None:
            min_hold_m = 5.0
    grace_secs = int(exit_mf_grace_seconds)
    if min_hold_m is not None:
        grace_secs = max(grace_secs, int(float(min_hold_m) * 60))
    # Ladder: never inherit mf_reversal's ~10m grace on a seconds-scale book.
    if use_ladder and lac.mf_flip:
        grace_secs = int(lac.mf_grace_seconds)
    grace_until = (fill_ts if use_ladder else entry_ts) + pd.Timedelta(seconds=grace_secs)
    day = stock_day
    need_stock_day = (
        use_mf
        or use_flow
        or use_extend
        or use_dyn
        or use_ladder
        or (use_mae_cut and mae_cut_require_mf_against)
        or (use_scale_in and sic.confirm_mode not in {"always", "any", "never", "off", "none", "half_only"})
        or use_div_mfe
        or use_delta_stop
        or use_adv_soft
        or (use_adv_vol and bool(avs.require_stock_adverse))
        or use_stale_cut
        or require_stock_align
        or use_stock_rev
        or use_wave_abort
    )
    if need_stock_day and day is not None and not day.empty:
        day = _prepare_stock_day(day)
    stock_entry_px = (
        _stock_close_at(day, fill_ts)
        if (
            use_div_mfe
            or use_delta_stop
            or use_adv_soft
            or (use_adv_vol and bool(avs.require_stock_adverse))
            or use_stale_cut
            or require_stock_align
            or use_stock_rev
            or use_wave_abort
        )
        else None
    )
    if use_div_mfe and stock_entry_px is None:
        use_div_mfe = False
    if use_delta_stop:
        if stock_entry_px is None:
            use_delta_stop = False
        else:
            delta_check_until = fill_ts + pd.Timedelta(seconds=int(dts.check_seconds))
            if dts.max_seconds is not None and int(dts.max_seconds) > 0:
                delta_check_deadline = fill_ts + pd.Timedelta(seconds=int(dts.max_seconds))
    if use_adv_soft:
        if stock_entry_px is None:
            use_adv_soft = False
        else:
            adv_check_until = fill_ts + pd.Timedelta(seconds=int(adv.check_seconds))
    if use_stock_rev:
        if stock_entry_px is None:
            use_stock_rev = False
        else:
            stock_rev_until = fill_ts + pd.Timedelta(minutes=float(srev.min_hold_minutes))
    if use_wave_abort and stock_entry_px is None:
        use_wave_abort = False
    if use_adv_vol:
        adv_vol_check_until = fill_ts + pd.Timedelta(seconds=int(avs.check_seconds))
    entry_px_low: float | None = None
    entry_px_high: float | None = None
    opp_streak = 0
    last_dyn_bar = -1
    if use_dyn and day is not None and not day.empty:
        ei = _stock_bar_index(day, entry_ts)
        if ei >= 0:
            if "_low" in day.columns:
                v = float(day["_low"].iloc[ei])
                entry_px_low = v if np.isfinite(v) else None
            if "_high" in day.columns:
                v = float(day["_high"].iloc[ei])
                entry_px_high = v if np.isfinite(v) else None
    for i, p in enumerate(sell_px):
        t = ts_list[i]
        if not np.isfinite(p) or p <= 0:
            continue
        if p >= tp_lvl:
            reason, exit_px, exit_ts = "TP", float(p), t
            break
        if p <= sl_lvl:
            reason, exit_px, exit_ts = "SL", float(p), t
            break
        if force_ts is not None and t >= force_ts:
            reason, exit_px, exit_ts = "DISPLACE", float(p), t
            break
        if (
            use_scale_in
            and entry2 is None
            and t >= scale_grace
            and (scale_deadline is None or t <= scale_deadline)
        ):
            mtm_mark = float(p) / entry - 1.0
            if mtm_mark <= -float(sic.pullback_ret):
                mf_v, su_v, sd_v = None, 0, 0
                if sic.confirm_mode not in {"always", "any", "never", "off", "none", "half_only"}:
                    visible_at = t - pd.Timedelta(seconds=int(stock_bar_delay_seconds))
                    mf_v, su_v, sd_v = _stock_mf_at(day, visible_at)
                if confirm_scale_in(
                    mode=sic.confirm_mode,
                    direction=str(direction),
                    mf=mf_v,
                    streak_up=su_v,
                    streak_dn=sd_v,
                ):
                    e2 = fill.buy(float(bids[i]), float(asks[i]))
                    if np.isfinite(e2) and e2 > 0:
                        entry2 = float(e2)
                        entry2_ts = t
        if use_hold_wd and t >= hold_wd_until:
            fired, _signed = qqq_adverse_from_entry(
                qqq_day,
                entry_ts=entry_ts,
                now_ts=t,
                direction=str(direction),
                thresh=float(hwd.qqq_adverse_from_entry),
                bar_delay_seconds=int(stock_bar_delay_seconds),
            )
            if fired:
                cur_ret_h = float(p) / entry - 1.0
                mtm_gate = hwd.require_option_mtm_max
                if mtm_gate is None or cur_ret_h <= float(mtm_gate):
                    reason, exit_px, exit_ts = "HOLD_SHOCK", float(p), t
                    break
        # Ladder rails before time hard-cap so quote gaps past end_ts still honor SL/TP.
        cur_ret = float(p) / entry - 1.0
        if use_trail or use_mae_cut or use_ladder or use_extend or use_stale_cut:
            if cur_ret > peak_ret:
                peak_ret = cur_ret
                if use_ladder:
                    peak_ts = t
        if (
            use_stale_cut
            and stale_until is not None
            and t >= stale_until
            and (not extended)
            and cur_ret <= stale_mtm_max
        ):
            stock_signed = None
            if stock_entry_px is not None and day is not None and stock_entry_px > 0:
                visible_at = t - pd.Timedelta(seconds=int(stock_bar_delay_seconds))
                s_px = _stock_close_at(day, visible_at)
                if s_px is not None:
                    s_ret = s_px / stock_entry_px - 1.0
                    stock_signed = s_ret if str(direction).upper() == "UP" else -s_ret
            if stock_signed is not None and float(stock_signed) <= stale_stock_max:
                reason, exit_px, exit_ts = "STALE_CUT", float(p), t
                break
        if use_ladder:
            ladder_hit = False
            # SL rails: tightest (closest to 0) first.
            for rail in sorted(lac.sl_rails, key=lambda r: float(r.ret), reverse=True):
                if cur_ret <= float(rail.ret):
                    pct = int(round(abs(float(rail.ret)) * 100))
                    reason, exit_px, exit_ts = f"SL_LADDER{pct}", float(p), t
                    ladder_hit = True
                    break
            if not ladder_hit:
                for rail in lac.tp_rails:
                    if cur_ret < float(rail.ret):
                        continue
                    if rail.action == "exit":
                        pct = int(round(float(rail.ret) * 100))
                        reason, exit_px, exit_ts = f"TP_LADDER{pct}", float(p), t
                        ladder_hit = True
                        break
                    ladder_trail_armed = True
                    ladder_trail_dd = float(rail.trail_dd)
            if (
                not ladder_hit
                and ladder_trail_armed
                and np.isfinite(peak_ret)
                and cur_ret <= float(peak_ret) - ladder_trail_dd
            ):
                reason, exit_px, exit_ts = "TRAIL_LADDER", float(p), t
                ladder_hit = True
            if (
                not ladder_hit
                and np.isfinite(peak_ret)
                and float(peak_ret) >= float(lac.stall_min_peak)
                and peak_ts is not None
                and (t - peak_ts).total_seconds() >= float(lac.stall_seconds)
            ):
                reason, exit_px, exit_ts = "PROFIT_STALL", float(p), t
                ladder_hit = True
            if ladder_hit:
                break
        if t >= end_ts:
            if use_ladder:
                reason, exit_px, exit_ts = "SEC_MAX", float(p), t
                break
            if use_extend and (not extended) and end_ts == base_end and ext_hold > base_hold:
                mtm_ok = cur_ret >= ext_mtm_min
                mf_ok = True
                if require_mf_align:
                    visible_at = t - pd.Timedelta(seconds=int(stock_bar_delay_seconds))
                    mf, _, _ = _stock_mf_at(day, visible_at)
                    if mf is None:
                        mf_ok = False
                    elif direction == "UP":
                        mf_ok = mf > 0
                    else:
                        mf_ok = mf < 0
                stock_ok = True
                if require_stock_align:
                    stock_ok = False
                    if stock_entry_px is not None and day is not None and stock_entry_px > 0:
                        visible_at = t - pd.Timedelta(seconds=int(stock_bar_delay_seconds))
                        s_px = _stock_close_at(day, visible_at)
                        if s_px is not None:
                            s_ret = s_px / stock_entry_px - 1.0
                            signed = s_ret if str(direction).upper() == "UP" else -s_ret
                            stock_ok = float(signed) >= ext_stock_min
                peak_ok = True
                if ext_min_peak is not None:
                    peak_ok = bool(np.isfinite(peak_ret) and float(peak_ret) >= ext_min_peak)
                qqq_ok = True
                if ext_max_qqq_adv is not None and qqq_day is not None:
                    fired, _signed = qqq_adverse_from_entry(
                        qqq_day,
                        entry_ts=entry_ts,
                        now_ts=t,
                        direction=str(direction),
                        thresh=float(ext_max_qqq_adv),
                        bar_delay_seconds=int(stock_bar_delay_seconds),
                    )
                    qqq_ok = not bool(fired)
                if mtm_ok and mf_ok and stock_ok and peak_ok and qqq_ok:
                    extended = True
                    end_ts = ext_end
                    continue
                reason, exit_px, exit_ts = f"T+{base_hold}", float(p), t
                break
            reason, exit_px, exit_ts = f"T+{ext_hold if extended else base_hold}", float(p), t
            break
        if use_trail:
            if (not trail_armed) and peak_ret >= act:
                trail_armed = True
            if trail_armed and cur_ret <= peak_ret - dd:
                reason, exit_px, exit_ts = "TRAIL", float(p), t
                break
        if use_mae_cut and t >= grace_until:
            # Toxic path: never printed meaningful MFE, then dug to -mae_thr.
            if (not np.isfinite(peak_ret) or float(peak_ret) < mae_bypass) and cur_ret <= -mae_thr:
                mf_ok = True
                if mae_cut_require_mf_against and direction in ("UP", "DN"):
                    visible_at = t - pd.Timedelta(seconds=int(stock_bar_delay_seconds))
                    mf, _, _ = _stock_mf_at(day, visible_at)
                    if mf is None:
                        mf_ok = False
                    elif direction == "UP":
                        mf_ok = mf < 0
                    else:
                        mf_ok = mf > 0
                if mf_ok:
                    reason, exit_px, exit_ts = "MAE_CUT", float(p), t
                    break
        if use_adv_soft and stock_entry_px is not None and day is not None:
            if cur_ret > opt_quote_peak:
                opt_quote_peak = float(cur_ret)
            visible_at = t - pd.Timedelta(seconds=int(stock_bar_delay_seconds))
            s_px = _stock_close_at(day, visible_at)
            if s_px is not None and stock_entry_px > 0:
                s_ret = s_px / stock_entry_px - 1.0
                signed = s_ret if str(direction).upper() == "UP" else -s_ret
                stock_signed_now = float(signed)
                if signed < stock_mae:
                    stock_mae = float(signed)
            mfe_ok = True
            if adv.max_opt_mfe is not None:
                mfe_ok = (not np.isfinite(opt_quote_peak)) or (
                    float(opt_quote_peak) < float(adv.max_opt_mfe)
                )
            still_ok = True
            if bool(adv.require_still_adverse):
                still_ok = stock_signed_now <= float(adv.still_adverse_max)
            if (
                adv_check_until is not None
                and t >= adv_check_until
                and stock_mae <= -float(adv.adverse_mae)
                and cur_ret <= float(adv.opt_mtm_max)
                and mfe_ok
                and still_ok
            ):
                if adv.mode == "soft_exit":
                    reason, exit_px, exit_ts = "ADVERSE_SOFT", float(p), t
                    break
                if not adv_armed:
                    adv_armed = True
                    if bool(adv.extend_max_cut):
                        trade_cut_deadline = None
        if use_adv_vol and stock_1s_arr is not None and adv_vol_check_until is not None:
            if t >= adv_vol_check_until and cur_ret <= float(avs.opt_mtm_max):
                stock_ok = True
                if bool(avs.require_stock_adverse):
                    if stock_entry_px is None or day is None:
                        stock_ok = False
                    else:
                        visible_at = t - pd.Timedelta(seconds=int(stock_bar_delay_seconds))
                        s_px = _stock_close_at(day, visible_at)
                        if s_px is None or stock_entry_px <= 0:
                            stock_ok = False
                        else:
                            s_ret = s_px / stock_entry_px - 1.0
                            signed = s_ret if str(direction).upper() == "UP" else -s_ret
                            stock_ok = float(signed) <= float(avs.stock_adverse_max)
                if stock_ok:
                    share = adverse_vol_share_asof(
                        stock_1s_arr,
                        now_ts=t,
                        window_seconds=int(avs.window_seconds),
                        direction=str(direction),
                    )
                    if share is not None and float(share) >= float(avs.min_share):
                        if avs.mode == "soft_exit":
                            reason, exit_px, exit_ts = "ADV_VOL", float(p), t
                            break
                        if not adv_vol_armed:
                            adv_vol_armed = True
                            if bool(avs.extend_max_cut):
                                trade_cut_deadline = None
        if use_trade_toxic and trade_mark is not None:
            ts_ns, tpx, t_entry = trade_mark
            t_mtm = trade_mtm_asof(ts_ns, tpx, t_entry, t)
            t_peak = trade_peak_mfe_asof(ts_ns, tpx, t_entry, t)
            if t_peak is not None and t_peak > trade_peak_mfe:
                trade_peak_mfe = float(t_peak)
            in_cut_window = t >= trade_cut_until and (
                trade_cut_deadline is None or t <= trade_cut_deadline
            )
            cut_lim = float(ttox.cut_ret)
            if (
                trade_mark_source == "quote"
                and ttox.quote_fallback_cut_ret is not None
            ):
                cut_lim = float(ttox.quote_fallback_cut_ret)
            mfe_lim = float(ttox.mfe_bypass)
            if adv_armed:
                cut_lim = min(cut_lim, float(adv.tight_cut_ret))
                mfe_lim = min(mfe_lim, float(adv.tight_mfe_bypass))
            if adv_vol_armed:
                cut_lim = min(cut_lim, float(avs.tight_cut_ret))
                mfe_lim = min(mfe_lim, float(avs.tight_mfe_bypass))
            if use_div_mfe and stock_entry_px is not None and day is not None:
                visible_at = t - pd.Timedelta(seconds=int(stock_bar_delay_seconds))
                s_px = _stock_close_at(day, visible_at)
                if s_px is not None and stock_entry_px > 0:
                    s_ret = s_px / stock_entry_px - 1.0
                    # Positive when underlying moves against the option trade.
                    adverse = (-s_ret) if str(direction).upper() == "UP" else s_ret
                    if adverse < float(ttox.div_stock_adverse_max):
                        mfe_lim = max(mfe_lim, float(ttox.div_mfe_bypass))
            dig = (
                t_mtm is not None
                and float(trade_peak_mfe) < mfe_lim
                and float(t_mtm) <= -float(cut_lim)
            )
            if dig and in_cut_window:
                if trade_dig_since is None:
                    trade_dig_since = t
                persist_ok = (t - trade_dig_since).total_seconds() >= float(
                    ttox.persist_seconds or 0
                )
                qconf = ttox.quote_confirm_ret
                quote_ok = True if qconf is None else (float(p) / float(entry) - 1.0) <= -float(qconf)
                if persist_ok and quote_ok:
                    reason, exit_px, exit_ts = "TRADE_TOX", float(p), t
                    break
            else:
                trade_dig_since = None
        if use_delta_stop and stock_entry_px is not None and day is not None:
            in_delta_win = t >= delta_check_until and (
                delta_check_deadline is None or t <= delta_check_deadline
            )
            if in_delta_win and cur_ret <= float(dts.opt_mtm_max):
                visible_at = t - pd.Timedelta(seconds=int(stock_bar_delay_seconds))
                s_px = _stock_close_at(day, visible_at)
                if s_px is not None and stock_entry_px > 0:
                    s_ret = s_px / stock_entry_px - 1.0
                    signed = s_ret if str(direction).upper() == "UP" else -s_ret
                    if signed < float(dts.min_stock_move):
                        reason, exit_px, exit_ts = "DELTA_STOP", float(p), t
                        break
        if (
            use_stock_rev
            and stock_rev_until is not None
            and t >= stock_rev_until
            and stock_entry_px is not None
            and day is not None
            and cur_ret <= float(srev.opt_mtm_max)
        ):
            visible_at = t - pd.Timedelta(seconds=int(stock_bar_delay_seconds))
            s_px = _stock_close_at(day, visible_at)
            if s_px is not None and stock_entry_px > 0:
                s_ret = s_px / stock_entry_px - 1.0
                signed = s_ret if str(direction).upper() == "UP" else -s_ret
                if float(signed) <= float(srev.stock_max):
                    reason, exit_px, exit_ts = "STOCK_REV", float(p), t
                    break
        if (
            use_wave_abort
            and stock_entry_px is not None
            and day is not None
            and not wave_state.done
        ):
            visible_at = t - pd.Timedelta(seconds=int(stock_bar_delay_seconds))
            s_px = _stock_close_at(day, visible_at)
            if s_px is not None and stock_entry_px > 0:
                s_ret = s_px / stock_entry_px - 1.0
                signed = s_ret if str(direction).upper() == "UP" else -s_ret
                held_s = (t - fill_ts).total_seconds()
                do_abort, wreason, wave_state = wave_abort_on_tick(
                    wave_state,
                    cfg=wabort,
                    held_seconds=float(held_s),
                    stock_signed=float(signed),
                    opt_mtm=float(cur_ret),
                )
                if do_abort:
                    reason, exit_px, exit_ts = "WAVE_ABORT", float(p), t
                    break
        if use_roi_time:
            held_m = (t - fill_ts).total_seconds() / 60.0
            hit_roi = False
            for mins, min_roi in rts.rails:
                key = float(mins)
                if key in roi_fired:
                    continue
                if held_m + 1e-9 < key:
                    break
                roi_fired.add(key)
                if cur_ret < float(min_roi):
                    reason, exit_px, exit_ts = f"ROI_TIME{int(mins)}", float(p), t
                    hit_roi = True
                    break
            if hit_roi:
                break
        if use_floor and t >= grace_until and cur_ret <= floor:
            reason, exit_px, exit_ts = "MTM_FLOOR", float(p), t
            break
        if use_flow and t >= grace_until:
            visible_at = t - pd.Timedelta(seconds=int(stock_bar_delay_seconds))
            cum = _cum_fav_flow(day, entry_ts, visible_at, str(direction))
            if cum is not None and cum <= flow_floor:
                if (not require_mtm) or cur_ret <= floor:
                    reason, exit_px, exit_ts = "FLOW_MTM" if require_mtm else "FLOW_DIE", float(p), t
                    break
        if use_mf and t >= grace_until:
            visible_at = t - pd.Timedelta(seconds=int(stock_bar_delay_seconds))
            mf, su, sd = _stock_mf_at(day, visible_at)
            if mf is None:
                continue
            if use_mf_flip:
                if direction == "UP" and mf < 0:
                    reason, exit_px, exit_ts = "MF_FLIP", float(p), t
                    break
                if direction == "DN" and mf > 0:
                    reason, exit_px, exit_ts = "MF_FLIP", float(p), t
                    break
            elif use_streak:
                if direction == "UP" and su == 0 and mf <= 0:
                    reason, exit_px, exit_ts = "STREAK0", float(p), t
                    break
                if direction == "DN" and sd == 0 and mf >= 0:
                    reason, exit_px, exit_ts = "STREAK0", float(p), t
                    break
        if use_dyn and day is not None and not day.empty:
            visible_at = t - pd.Timedelta(seconds=int(stock_bar_delay_seconds))
            bi = _stock_bar_index(day, visible_at)
            if bi < 0 or bi == last_dyn_bar:
                continue
            last_dyn_bar = bi
            held_m = (t - entry_ts).total_seconds() / 60.0
            mf_slow = float(day["_mf"].iloc[bi])
            mf_fast = float(day["_mf_fast"].iloc[bi]) if "_mf_fast" in day.columns else float("nan")
            fav_slow = _fav_mf(mf_slow, str(direction))
            fav_fast = _fav_mf(mf_fast, str(direction))
            slow_slope = None
            if bi >= dyn_lb:
                prev = float(day["_mf"].iloc[bi - dyn_lb])
                if np.isfinite(prev) and np.isfinite(mf_slow):
                    # Favorable slope: UP uses raw Δmf10; DN flips.
                    raw_slope = mf_slow - prev
                    slow_slope = raw_slope if direction == "UP" else -raw_slope
            if fav_fast is not None and fav_fast < 0:
                opp_streak += 1
            else:
                opp_streak = 0
            # Early reverse cut (situation C / 防线二)
            if held_m >= dyn_min_m and opp_streak >= dyn_opp_n and slow_slope is not None and slow_slope < 0:
                px_break = True
                if dyn_require_price_break and "_close" in day.columns:
                    px = float(day["_close"].iloc[bi])
                    if direction == "UP":
                        px_break = entry_px_low is not None and np.isfinite(px) and px < entry_px_low
                    else:
                        px_break = entry_px_high is not None and np.isfinite(px) and px > entry_px_high
                if px_break:
                    reason, exit_px, exit_ts = "FAST_REVERSAL", float(p), t
                    break
            # Profit / trend soft exits after trail-start (防线一 + trend dead)
            if held_m >= dyn_start_m and fav_slow is not None:
                if fav_slow < 0:
                    reason, exit_px, exit_ts = "TREND_DEAD", float(p), t
                    break
                if fav_fast is not None and slow_slope is not None and slow_slope < 0:
                    hist = day["_mf_fast"].iloc[: bi + 1].astype(float).to_numpy()
                    if direction == "DN":
                        hist = -hist
                    hist = hist[np.isfinite(hist)]
                    thr = None
                    if hist.size >= 5:
                        thr = float(np.nanpercentile(hist, dyn_pct))
                    weak = fav_fast <= (thr if thr is not None else 0.0)
                    if weak:
                        reason, exit_px, exit_ts = "MOM_EXHAUST", float(p), t
                        break
    ret = float(exit_px) / entry - 1.0
    scale_added = False
    deployed = 1.0
    if use_scale_in:
        ret, deployed, scale_added = blend_scale_in_ret(
            entry1=entry,
            entry2=entry2,
            exit_px=float(exit_px),
            first_frac=float(sic.first_frac),
            add_frac=float(sic.add_frac),
        )
    return SimResult(
        entry=entry,
        exit=exit_px,
        ret=ret,
        reason=reason,
        entry_ts=ts_list[0],
        exit_ts=exit_ts,
        scale_in_added=bool(scale_added),
        scale_in_entry2=entry2 if scale_added else None,
        scale_in_ts=entry2_ts if scale_added else None,
        scale_in_deployed_frac=float(deployed),
        adverse_soft_armed=bool(adv_armed),
        adv_vol_armed=bool(adv_vol_armed),
    )


def run_offline_replay(
    profile: dict[str, Any],
    *,
    scheme: str = "single",
    stock_by: dict[str, pd.DataFrame] | None = None,
    regime_gate=None,
) -> dict[str, Any]:
    """
    scheme:
      - single: one entry per TopK symbol/day
      - m5: reentry max N with cooldown
      - m5_circuit: + day circuit

    stock_by: optional preloaded feature frames (skip disk load) for sensitivity grids.
    regime_gate: optional prebuilt Mag7RegimeGate (skip rebuild).
    """
    paths = profile["_paths"]
    sig_cfg = profile["signal"]
    trade = profile["trade"]
    fill = FillSpec(
        entry_frac=float(profile["fill"].get("entry_frac", 0.8)),
        exit_frac=float(profile["fill"].get("exit_frac", 0.8)),
    )
    start = profile["date_range"]["start"]
    end = profile["date_range"]["end"]
    months = month_list(start, end)
    symbols = list(profile["symbols"])
    exclude_raw = trade.get("symbol_exclude") or sig_cfg.get("symbol_exclude") or []
    if isinstance(exclude_raw, str):
        exclude_raw = [exclude_raw]
    symbol_exclude = {str(x).upper() for x in exclude_raw}
    if symbol_exclude:
        symbols = [s for s in symbols if str(s).upper() not in symbol_exclude]
    symbol_size_scale = {
        str(k).upper(): float(v)
        for k, v in (trade.get("symbol_size_scale") or {}).items()
    }
    dvol_size_cfg = parse_dvol_size_scale(trade.get("dvol_size_scale"))
    n_dvol_size_boost = 0
    vrp_size_cfg = parse_vrp_size_scale(trade.get("vrp_size_scale"))
    n_vrp_size_scale = 0
    n_vrp_skip = 0
    from_open_cfg = parse_from_open_gate(trade.get("from_open_gate"))
    n_from_open_block = 0
    n_from_open_scale = 0
    overnight_gap_cfg = parse_overnight_gap_gate(trade.get("overnight_gap_gate"))
    n_overnight_gap_block = 0
    n_overnight_gap_scale = 0
    peer_gap_cfg = parse_peer_gap_gate(trade.get("peer_gap_gate"))
    n_peer_gap_block = 0
    n_peer_gap_scale = 0
    range_stall_cfg = parse_range_stall_gate(trade.get("range_stall_gate"))
    n_range_stall_block = 0
    n_range_stall_scale = 0
    dn_gap_stall_cfg = parse_dn_gap_stall_gate(trade.get("dn_gap_stall_gate"))
    n_dn_gap_stall_block = 0
    n_dn_gap_stall_scale = 0
    up_gap_stall_cfg = parse_up_gap_stall_gate(trade.get("up_gap_stall_gate"))
    n_up_gap_stall_block = 0
    n_up_gap_stall_scale = 0
    fo_lod_chase_cfg = parse_fo_lod_chase_gate(trade.get("fo_lod_chase_gate"))
    n_fo_lod_chase_block = 0
    n_fo_lod_chase_scale = 0
    money = str(trade.get("moneyness", "ATM"))

    mf_idio_mode = str(sig_cfg.get("mf_idio_mode") or "off").strip().lower()
    mf_idio_on = mf_idio_mode not in {"", "off", "none", "false", "0"}
    mf_idio_min_frac = float(sig_cfg.get("mf_idio_min_frac", 0.0) or 0.0)
    mf_idio_beta_days = int(sig_cfg.get("mf_idio_beta_days", 5) or 5)
    mf_idio_beta_on = str(sig_cfg.get("mf_idio_beta_on") or "ret").strip().lower()
    mf_idio_block_missing = bool(sig_cfg.get("mf_idio_block_missing", False))
    # block = reject entry (default); scale = keep entry, cut size_frac
    mf_idio_action = str(sig_cfg.get("mf_idio_action") or "block").strip().lower()
    if mf_idio_action not in {"block", "scale", "size", "half"}:
        mf_idio_action = "block"
    if mf_idio_action in {"size", "half"}:
        mf_idio_action = "scale"
    mf_idio_scale = float(sig_cfg.get("mf_idio_scale", 0.5) or 0.5)
    mf_idio_scale = max(0.0, min(mf_idio_scale, 1.0))
    # Only arm the gate after N consecutive losing days (None/0 = always on).
    _als = sig_cfg.get("mf_idio_after_loss_streak")
    mf_idio_after_loss_streak = int(_als) if _als is not None else None
    # Prior sessions for beta / TCN window need bars before ``start``.
    load_start = start
    tcn_cfg_peek = (profile.get("tcn_gate") or (profile.get("signal") or {}).get("tcn_gate") or {})
    tcn_enabled_peek = bool(tcn_cfg_peek.get("enabled", False))
    router_peek = profile.get("regime_router") or {}
    watchdog_peek = profile.get("watchdog") or {}
    router_rule_peek = bool(router_peek.get("enabled", False)) and str(
        router_peek.get("mode") or "oracle"
    ).strip().lower() in {"rule", "rules", "causal"}
    watchdog_rule_peek = bool(watchdog_peek.get("enabled", False))
    if mf_idio_on or tcn_enabled_peek or router_rule_peek or watchdog_rule_peek:
        lookback_days = max(14, mf_idio_beta_days * 3 if mf_idio_on else 14)
        load_start = (pd.Timestamp(start) - pd.Timedelta(days=lookback_days)).strftime("%Y-%m-%d")
        months = month_list(load_start, end)

    if stock_by is None:
        stock_by = {}
        # Load excluded names too if they remain in peer_symbols (breadth only).
        load_syms = list(dict.fromkeys(list(symbols) + list(sig_cfg.get("peer_symbols") or [])))
        # corr_rewire needs QQQ + Mag7 1m history (parsed later; peek trade flag).
        _cr_peek = bool((trade.get("corr_rewire") or {}).get("enabled")) if isinstance(trade.get("corr_rewire"), dict) else False
        _chop_peek = bool((profile.get("chop_gate") or {}).get("enabled")) if isinstance(profile.get("chop_gate"), dict) else False
        _sfg_peek = (
            bool((profile.get("session_flow_gate") or {}).get("enabled"))
            if isinstance(profile.get("session_flow_gate"), dict)
            else False
        )
        if (
            mf_idio_on
            or tcn_enabled_peek
            or router_rule_peek
            or watchdog_rule_peek
            or _cr_peek
            or _chop_peek
            or _sfg_peek
        ) and "QQQ" not in {str(s).upper() for s in load_syms}:
            load_syms.append("QQQ")
        if _sfg_peek and "VIXY" not in {str(s).upper() for s in load_syms}:
            load_syms.append("VIXY")
        for sym in load_syms:
            raw = load_stock_month_files(paths["stock_root"], sym, months)
            if raw.empty:
                continue
            # clip date range (keep lookback when residual-MF gate is on)
            raw = raw[(raw["date"] >= load_start) & (raw["date"] <= end)]
            stock_by[sym] = attach_mf_features(
                raw,
                mf_window=int(sig_cfg.get("mf_window", 10)),
                vol_ma_window=int(sig_cfg.get("vol_ma_window", 20)),
                mf_fast_window=resolve_mf_fast_window(sig_cfg),
            )

    # TopK only among tradeable symbols — ignore ref frames (QQQ/VIXY) that may
    # be present in stock_by for regime / peer helpers.
    # Signal universe stays inside the requested date_range even if lookback bars remain.
    trade_stock = {}
    for s in symbols:
        sdf = stock_by.get(s)
        if sdf is None or getattr(sdf, "empty", True):
            continue
        trade_stock[s] = sdf[(sdf["date"] >= start) & (sdf["date"] <= end)].copy()
    all_first = build_all_first_rule_a_signals(trade_stock, sig_cfg)
    top2 = build_topk_signals(trade_stock, sig_cfg)
    displace_on = _trade_flag(trade, "displace_on_later", False)
    # TopK blocked-backfill: walk all first-Rule-A in time order, but only keep up to
    # top_k *successful fills* per day. Regime/peer/quote blocks do not consume a slot
    # (unlike raw earliest-TopK, where a later-blocked #2 permanently wastes the seat).
    topk_backfill = _trade_flag(trade, "topk_backfill_on_block", False) or _trade_flag(
        sig_cfg, "topk_backfill_on_block", False
    )
    commit_tod = _parse_commit_tod(
        trade.get("topk_commit_tod", sig_cfg.get("topk_commit_tod"))
    )
    commit_rank = str(
        trade.get("topk_rank") or sig_cfg.get("topk_rank") or "abs_from_prev"
    ).strip().lower()
    post_commit_fill = _trade_flag(trade, "topk_post_commit_fill", True)
    if commit_tod is not None:
        # Deferred auction needs the full first-Rule-A universe, not earliest-TopK.
        event_sigs = all_first
        displace_universe = "all_first"
    else:
        displace_universe = str(
            trade.get("displace_universe") or ("all_first" if displace_on else "topk")
        ).strip().lower()
        if displace_universe in {"all_first", "all", "universe", "first_rule_a"}:
            event_sigs = all_first
        else:
            event_sigs = top2
    # Research-only: expand event universe without kicking (free slots after exits).
    if _trade_flag(trade, "late_signal_universe_all_first", False):
        event_sigs = all_first
        displace_universe = "all_first"
    topk_mf_backfill = _trade_flag(trade, "topk_mf_backfill_on_block", False) or _trade_flag(
        sig_cfg, "topk_mf_backfill_on_block", False
    )
    if topk_mf_backfill and commit_tod is None:
        # Stricter than time-all_first backfill: try earliest TopK first; only then
        # remaining first-Rule-A ordered by multi-factor score (research).
        from maga7.common.multifactor_rank import MultiFactorConfig, order_events_topk_then_mf

        topk_backfill = True
        event_sigs = order_events_topk_then_mf(
            top2,
            all_first,
            stock_by,
            symbols=list(symbols),
            cfg=MultiFactorConfig(),
        )
        displace_universe = "topk_mf_backfill"
    elif topk_backfill and commit_tod is None:
        event_sigs = all_first
        displace_universe = "all_first_backfill"
    seat_gate_cfg = parse_seat_score_gate(trade.get("seat_score_gate"))
    n_seat_score_skip = 0
    if seat_gate_cfg.enabled and commit_tod is None:
        # Quality seat gate needs the full first-Rule-A stream + fill-cap accounting.
        event_sigs = all_first
        displace_universe = "all_first_seat_score"
        topk_backfill = True
    contract_mode = str(trade.get("contract_mode", "day_lock")).lower()
    quote_source = str(trade.get("quote_source", "1s")).lower()  # 1s | day_iv | auto
    half_spread = float(trade.get("day_iv_half_spread_frac", 0.01))
    prefer_dte, allowed_dte = lock_policy_from_profile(profile)
    clear_otm = trade.get("clear_otm_ban_0dte_pct", None)
    clear_otm_thresh = float(clear_otm) if clear_otm is not None else None
    max_entry_otm_raw = trade.get("max_entry_abs_otm_pct", None)
    max_entry_abs_otm = (
        float(max_entry_otm_raw) if max_entry_otm_raw is not None else None
    )

    # open_lock / open_ladder use multi-DTE map; day_lock / signal_atm use flat 4-bucket map
    lock_path = paths.get("open_locked_map") if contract_mode in ("open_lock", "open", "open_ladder", "open_lock_ladder") else None
    lock_path = lock_path or paths["locked_map"]
    multi_lock_idx = None
    lock_idx = None
    ladder = contract_mode in ("open_ladder", "open_lock_ladder") or bool(trade.get("open_ladder"))
    from maga7.common.open_lock import resolve_otm_rungs

    otm_rungs = resolve_otm_rungs(profile, default=2 if ladder else 1)
    if contract_mode in ("open_lock", "open", "open_ladder", "open_lock_ladder"):
        multi_lock_idx = load_multidte_lock_index(lock_path)
    else:
        lock_idx = load_lock_index(lock_path)

    quote_root = paths["quote_1s_root"]
    quote_cache: dict[tuple[str, str], pd.DataFrame | None] = {}
    trades_root = paths.get("option_trades_root")
    trade_cache: dict[tuple[str, str], pd.DataFrame | None] = {}
    stock_1s_root = paths.get("stock_1s_root")
    stock_1s_cache: dict[tuple[str, str], pd.DataFrame] = {}
    avs_cfg_early = adverse_vol_share_from_trade(trade)
    entry_avs_cfg = entry_adv_vol_from_trade(trade)
    # Also load 1s when overnight_gap_gate needs adv-share confirm.
    _og_needs_1s = bool(
        getattr(overnight_gap_cfg, "enabled", False)
        and getattr(overnight_gap_cfg, "require_adv_share", None) is not None
    )
    use_adv_vol_global = (
        bool(avs_cfg_early.enabled) or bool(entry_avs_cfg.enabled) or _og_needs_1s
    )
    n_entry_adv_vol_block = 0
    n_entry_adv_vol_scale = 0
    ttox_cfg = trade_toxic_from_trade(trade)
    use_trade_toxic_global = bool(ttox_cfg.enabled) or str(
        trade.get("early_exit_mode") or ""
    ).strip().lower() in {"trade_toxic", "trade_mae", "toxic_trade"}
    day_iv_root = paths.get("day_iv_root")
    chain_cache = DayIvChainCache(day_iv_root) if day_iv_root else None
    n_signal_atm = 0
    n_day_lock_fb = 0
    n_skip0_clear_otm = 0
    n_skip_max_entry_otm = 0
    n_quote_1s = 0
    n_quote_day_iv = 0
    n_trade_path = 0
    n_trade_path_miss = 0

    from maga7.common.regime import Mag7RegimeGate

    if regime_gate is None:
        regime_gate = Mag7RegimeGate.from_profile(profile, months=months)
    # Causal 1s path: preagg stock_root may be missing; rebuild QQQ/VIXY regime
    # from the provided stock_by so Watchdog Hunt can arm (begin_day requires gate).
    if regime_gate is None and stock_by:
        qqq = stock_by.get("QQQ")
        if qqq is not None and not getattr(qqq, "empty", True):
            from maga7.common.stock_1s import regime_gate_from_1s

            regime_gate = regime_gate_from_1s(profile, stock_by)
    # Watchdog is the architecture layer; legacy regime_router bridges into it.
    watchdog = RegimeWatchdog.from_profile(profile)
    router_on, router_labels, router_experts, router_mode, router_rule = _load_regime_router(
        profile
    )
    # If watchdog synthesized from regime_router, prefer watchdog path only.
    if watchdog is not None:
        router_on = False
    router_regime_snap = (
        _router_snapshot_regime(regime_gate.cfg)
        if (watchdog is not None or router_on) and regime_gate is not None
        else {}
    )
    router_asof = str(
        (profile.get("watchdog") or {}).get("asof")
        or (profile.get("regime_router") or {}).get("asof")
        or "10:30"
    )
    n_router_expert_days = 0
    router_day_counts: dict[str, int] = {}
    watchdog_state_counts: dict[str, int] = {}
    n_watchdog_days = 0
    n_hunt_signals = 0
    n_hunt_trades = 0
    n_hunt_budget_skip = 0
    n_hunt_mutex_skip = 0
    n_hunt_day_circuit = 0
    n_regime_block = 0
    n_regime_scale = 0
    n_peer_block = 0
    n_mf_idio_block = 0
    n_mf_idio_scale = 0
    n_si_block = 0
    n_pe_block = 0
    n_tod_z_block = 0
    n_dn_above_open_block = 0
    n_vwap_lock_block = 0
    n_dn_vwap_lod_block = 0
    block_dn_if_above_open = bool(sig_cfg.get("block_dn_if_above_open", False)) or bool(
        trade.get("block_dn_if_above_open", False)
    )
    vwap_dir_lock = bool(sig_cfg.get("vwap_dir_lock", False)) or bool(
        trade.get("vwap_dir_lock", False)
    )
    block_dn_if_vwap_lod = bool(sig_cfg.get("block_dn_if_vwap_lod", False)) or bool(
        trade.get("block_dn_if_vwap_lod", False)
    )
    lod_bounce_min = float(
        sig_cfg.get("lod_bounce_min", trade.get("lod_bounce_min", 0.02)) or 0.02
    )
    peer_align_min = sig_cfg.get("peer_align_min")
    peer_align_min_i = int(peer_align_min) if peer_align_min is not None else None
    peer_align_mode = str(sig_cfg.get("peer_align_mode", "mf10")).strip().lower()
    peer_symbols = list(sig_cfg.get("peer_symbols") or profile.get("symbols") or [])
    qqq_frame = stock_by.get("QQQ")
    mf_idio_beta_cache: dict[tuple[str, str], float | None] = {}
    tcn_gate = load_tcn_gate(profile)
    tcn_on = bool(getattr(tcn_gate, "cfg", None) and tcn_gate.cfg.enabled)
    n_tcn_block = 0
    n_tcn_scale = 0
    n_tcn_skip_regime = 0
    state_gate = load_state_gate(profile)
    state_gate_on = bool(getattr(state_gate, "cfg", None) and state_gate.cfg.enabled)
    n_state_gate_block = 0
    n_state_gate_scale = 0
    state_gate_day_counts: dict[str, int] = {}
    chop_gate = load_chop_gate(profile)
    chop_gate_on = bool(getattr(chop_gate, "cfg", None) and chop_gate.cfg.enabled)
    n_chop_gate_block = 0
    n_chop_gate_scale = 0
    chop_gate_day_counts: dict[str, int] = {}
    session_flow_gate = load_session_flow_gate(profile)
    session_flow_gate_on = bool(
        getattr(session_flow_gate, "cfg", None) and session_flow_gate.cfg.enabled
    )
    n_session_flow_block = 0
    n_session_flow_scale = 0
    session_flow_day_counts: dict[str, int] = {}
    lgbm_bouncer = load_lgbm_bouncer(profile)
    lgbm_on = bool(getattr(lgbm_bouncer, "cfg", None) and lgbm_bouncer.cfg.enabled)
    n_lgbm_block = 0
    n_lgbm_scale = 0
    # Ensure QQQ frame exists for tcn/lgbm/router-rule / chop_gate channels.
    need_qqq = (
        tcn_on
        or lgbm_on
        or (router_on and router_mode == "rule")
        or (watchdog is not None)
        or bool(hold_watchdog_from_trade(trade).enabled)
        or chop_gate_on
        or session_flow_gate_on
        or bool(vrp_size_cfg.enabled)
    )
    if need_qqq and qqq_frame is None and stock_by.get("QQQ") is None:
        try:
            raw_q = load_stock_month_files(paths["stock_root"], "QQQ", months)
            if not raw_q.empty:
                raw_q = raw_q[(raw_q["date"] >= load_start) & (raw_q["date"] <= end)]
                stock_by["QQQ"] = attach_mf_features(
                    raw_q,
                    mf_window=int(sig_cfg.get("mf_window", 10)),
                    vol_ma_window=int(sig_cfg.get("vol_ma_window", 20)),
                    mf_fast_window=resolve_mf_fast_window(sig_cfg),
                )
                qqq_frame = stock_by["QQQ"]
        except Exception:
            qqq_frame = None
    elif need_qqq and qqq_frame is None and stock_by.get("QQQ") is not None:
        qqq_frame = stock_by["QQQ"]

    if session_flow_gate_on and stock_by.get("VIXY") is None:
        try:
            raw_v = load_stock_month_files(paths["stock_root"], "VIXY", months)
            if not raw_v.empty:
                raw_v = raw_v[(raw_v["date"] >= load_start) & (raw_v["date"] <= end)]
                stock_by["VIXY"] = attach_mf_features(
                    raw_v,
                    mf_window=int(sig_cfg.get("mf_window", 10)),
                    vol_ma_window=int(sig_cfg.get("vol_ma_window", 20)),
                    mf_fast_window=resolve_mf_fast_window(sig_cfg),
                )
        except Exception:
            pass

    vrp_day_table = None
    if vrp_size_cfg.enabled:
        try:
            vrp_day_table = build_vrp_day_table(
                qqq_df=qqq_frame if qqq_frame is not None else stock_by.get("QQQ"),
                start=str(start),
                end=str(end),
                cfg=vrp_size_cfg,
                stock_1s_root=paths.get("stock_1s_root"),
            )
        except Exception:
            vrp_day_table = None

    def _mf_idio_armed(loss_streak_n: int) -> bool:
        if not mf_idio_on:
            return False
        if mf_idio_after_loss_streak is None or int(mf_idio_after_loss_streak) <= 0:
            return True
        return int(loss_streak_n) >= int(mf_idio_after_loss_streak)

    def _mf_idio_fail(
        sym: str,
        direction: str,
        feature_ts: pd.Timestamp,
        date: str,
        *,
        loss_streak_n: int,
    ) -> bool:
        """True when residual-MF gate fails (and is armed)."""
        if not _mf_idio_armed(loss_streak_n):
            return False
        key = (str(sym).upper(), str(date))
        if key not in mf_idio_beta_cache and mf_idio_mode not in {"diff_pos", "diff", "mf_diff"}:
            mf_idio_beta_cache[key] = rolling_idio_beta(
                stock_by.get(sym),
                qqq_frame,
                asof_date=str(date),
                n_days=mf_idio_beta_days,
                on=mf_idio_beta_on,
            )
        beta = mf_idio_beta_cache.get(key)
        ok, _meta = mf_idio_ok(
            stock_by.get(sym),
            qqq_frame,
            date=str(date),
            asof_ts=feature_ts,
            direction=str(direction),
            mode=mf_idio_mode,
            min_frac=mf_idio_min_frac,
            beta_days=mf_idio_beta_days,
            beta_on=mf_idio_beta_on,
            beta=beta,
            block_missing=mf_idio_block_missing,
        )
        return not ok

    def _mf_idio_allows_entry(
        sym: str,
        direction: str,
        feature_ts: pd.Timestamp,
        date: str,
        *,
        loss_streak_n: int,
    ) -> bool:
        nonlocal n_mf_idio_block
        if mf_idio_action == "scale":
            return True
        if _mf_idio_fail(sym, direction, feature_ts, date, loss_streak_n=loss_streak_n):
            n_mf_idio_block += 1
            return False
        return True

    def _mf_idio_size_mult(
        sym: str,
        direction: str,
        feature_ts: pd.Timestamp,
        date: str,
        *,
        loss_streak_n: int,
    ) -> float:
        nonlocal n_mf_idio_scale
        if mf_idio_action != "scale":
            return 1.0
        if _mf_idio_fail(sym, direction, feature_ts, date, loss_streak_n=loss_streak_n):
            n_mf_idio_scale += 1
            return float(mf_idio_scale)
        return 1.0

    def _tcn_regime_kwargs(dec_reg: Any | None) -> dict[str, float | None]:
        if dec_reg is None:
            return {"regime_vixy_z": None, "regime_qqq_fp": None}
        vz = getattr(dec_reg, "vixy_z", None)
        qfp = getattr(dec_reg, "qqq_from_prev", None)
        return {
            "regime_vixy_z": float(vz) if vz is not None and np.isfinite(vz) else None,
            "regime_qqq_fp": float(qfp) if qfp is not None and np.isfinite(qfp) else None,
        }

    def _tcn_allows_entry(
        sym: str,
        direction: str,
        feature_ts: pd.Timestamp,
        dec_reg: Any | None = None,
    ) -> bool:
        nonlocal n_tcn_block, n_tcn_skip_regime
        if not tcn_on or tcn_gate.cfg.action != "block":
            return True
        dec_t = tcn_gate.decide(
            symbol=str(sym),
            direction=str(direction),
            asof_ts=feature_ts,
            stock_df=stock_by.get(sym),
            qqq_df=qqq_frame,
            **_tcn_regime_kwargs(dec_reg),
        )
        if dec_t.reason == "tcn_skip_regime":
            n_tcn_skip_regime += 1
            return True
        if not dec_t.allow:
            n_tcn_block += 1
            return False
        return True

    def _tcn_size_mult(
        sym: str,
        direction: str,
        feature_ts: pd.Timestamp,
        dec_reg: Any | None = None,
    ) -> tuple[float, float | None, str]:
        nonlocal n_tcn_scale, n_tcn_skip_regime
        if not tcn_on:
            return 1.0, None, "off"
        dec_t = tcn_gate.decide(
            symbol=str(sym),
            direction=str(direction),
            asof_ts=feature_ts,
            stock_df=stock_by.get(sym),
            qqq_df=qqq_frame,
            **_tcn_regime_kwargs(dec_reg),
        )
        if dec_t.reason == "tcn_skip_regime":
            n_tcn_skip_regime += 1
            return 1.0, dec_t.p, dec_t.reason
        if tcn_gate.cfg.action == "block":
            return (1.0 if dec_t.allow else 0.0), dec_t.p, dec_t.reason
        scale = float(dec_t.size_scale)
        if scale < 1.0 - 1e-12:
            n_tcn_scale += 1
        return scale, dec_t.p, dec_t.reason

    def _lgbm_allows_entry(
        sym: str,
        direction: str,
        feature_ts: pd.Timestamp,
    ) -> bool:
        nonlocal n_lgbm_block
        if not lgbm_on or lgbm_bouncer.cfg.action != "block":
            return True
        dec_l = lgbm_bouncer.decide(
            symbol=str(sym),
            direction=str(direction),
            asof_ts=feature_ts,
            stock_df=stock_by.get(sym),
            qqq_df=qqq_frame,
        )
        if not dec_l.allow:
            n_lgbm_block += 1
            return False
        return True

    def _lgbm_size_mult(
        sym: str,
        direction: str,
        feature_ts: pd.Timestamp,
    ) -> tuple[float, float | None, str]:
        nonlocal n_lgbm_scale, n_lgbm_block
        if not lgbm_on:
            return 1.0, None, "off"
        dec_l = lgbm_bouncer.decide(
            symbol=str(sym),
            direction=str(direction),
            asof_ts=feature_ts,
            stock_df=stock_by.get(sym),
            qqq_df=qqq_frame,
        )
        if lgbm_bouncer.cfg.action == "block":
            if not dec_l.allow:
                n_lgbm_block += 1
                return 0.0, dec_l.p, dec_l.reason
            return 1.0, dec_l.p, dec_l.reason
        scale = float(dec_l.size_scale)
        if scale < 1.0 - 1e-12:
            n_lgbm_scale += 1
        return scale, dec_l.p, dec_l.reason

    si_min = sig_cfg.get("si_min")
    si_min_f = float(si_min) if si_min is not None else None
    pe_min_ratio = sig_cfg.get("pe_min_ratio")
    pe_min_ratio_f = float(pe_min_ratio) if pe_min_ratio is not None else None
    pe_window = int(sig_cfg.get("pe_window", 10))
    pe_lookback = int(sig_cfg.get("pe_lookback_bars", 780))
    tod_z_min = sig_cfg.get("tod_mf_z_min")
    tod_z_min_f = float(tod_z_min) if tod_z_min is not None else None
    tod_z_lookback = int(sig_cfg.get("tod_mf_z_lookback_days", 20))
    confirm_bars_raw = trade.get("entry_confirm_bars", sig_cfg.get("entry_confirm_bars"))
    confirm_bars_n = int(confirm_bars_raw) if confirm_bars_raw is not None else 0
    confirm_mode = str(
        trade.get("entry_confirm_mode") or sig_cfg.get("entry_confirm_mode") or "mf"
    ).strip().lower()
    # Optional: only apply confirm on given weekdays (0=Mon..4=Fri).
    confirm_wd_raw = trade.get("entry_confirm_weekdays", sig_cfg.get("entry_confirm_weekdays"))
    confirm_weekdays: set[int] | None = None
    if confirm_wd_raw is not None:
        if isinstance(confirm_wd_raw, str):
            confirm_weekdays = {
                int(x.strip()) for x in confirm_wd_raw.split(",") if str(x).strip() != ""
            }
        else:
            confirm_weekdays = {int(x) for x in confirm_wd_raw}
    confirm_dir_raw = trade.get(
        "entry_confirm_directions", sig_cfg.get("entry_confirm_directions")
    )
    confirm_directions: set[str] | None = None
    if confirm_dir_raw is not None:
        if isinstance(confirm_dir_raw, str):
            confirm_directions = {
                x.strip().upper()
                for x in confirm_dir_raw.split(",")
                if str(x).strip() != ""
            }
        else:
            confirm_directions = {str(x).upper() for x in confirm_dir_raw}
    max_fp_raw = trade.get("max_from_prev_abs", sig_cfg.get("max_from_prev_abs"))
    max_from_prev_abs = float(max_fp_raw) if max_fp_raw is not None else None
    n_confirm_block = 0
    n_max_fp_block = 0
    spc_raw = trade.get("stock_path_confirm") or {}
    if not isinstance(spc_raw, dict):
        spc_raw = {}
    spc_on = bool(spc_raw.get("enabled", False))
    spc_thr_pos = float(spc_raw.get("thr_pos", 0.0015) or 0.0015)
    spc_thr_neg = float(spc_raw.get("thr_neg", -0.003) or -0.003)
    spc_max_wait = int(spc_raw.get("max_wait_seconds", 300) or 300)
    spc_on_timeout = str(spc_raw.get("on_timeout", "block") or "block").strip().lower()
    # When False, path confirm is a veto only (no fill delay on +thr_pos).
    spc_delay_on_pos = bool(spc_raw.get("delay_on_pos", True))
    spc_tod_start = _hhmm_to_minutes(spc_raw.get("tod_start", "10:31"))
    spc_tod_end = _hhmm_to_minutes(spc_raw.get("tod_end", "11:00"))
    spc_wd_raw = spc_raw.get("weekdays")
    spc_weekdays: set[int] | None = None
    if spc_wd_raw is not None:
        if isinstance(spc_wd_raw, str):
            spc_weekdays = {
                int(x.strip()) for x in spc_wd_raw.split(",") if str(x).strip() != ""
            }
        else:
            spc_weekdays = {int(x) for x in spc_wd_raw}
    n_stock_path_confirm_block = 0
    n_stock_path_confirm_ok = 0
    cr_cfg = corr_rewire_from_trade(trade)
    cr_on = cr_cfg is not None
    n_corr_rewire_scale = 0
    n_corr_rewire_block = 0
    n_corr_rewire_days = 0
    purity_on = _trade_flag(trade, "trend_purity_sizing", False)
    purity_fp_ref = float(trade.get("trend_purity_fp_ref", 0.025) or 0.025)
    purity_features = str(
        trade.get("trend_purity_features") or "momentum"
    ).strip().lower()
    purity_path_window = int(trade.get("trend_purity_path_window", 20) or 20)
    purity_adverse_ref = float(trade.get("trend_purity_adverse_ref", 0.005) or 0.005)
    n_purity_skip = 0
    n_purity_scaled = 0

    def get_q(sym: str, date: str):
        k = (sym, date)
        if k not in quote_cache:
            quote_cache[k] = load_quotes(quote_root, sym, date)
        return quote_cache[k]

    def get_path(sym: str, date: str, ticker: str) -> tuple[pd.DataFrame | None, str]:
        """Resolve quote path: prefer 1s when requested/available, else day_iv synthetic."""
        if quote_source in ("1s", "auto"):
            path = path_for_ticker(get_q(sym, date), ticker)
            if path is not None and not path.empty:
                return path, "1s"
            if quote_source == "1s":
                return None, "none"
        if quote_source in ("day_iv", "auto") and chain_cache is not None:
            path = day_iv_path_as_quotes(chain_cache.get(sym, date), ticker, half_spread_frac=half_spread)
            if path is not None and not path.empty:
                return path, "day_iv"
        return None, "none"

    def get_trade_path(sym: str, date: str, ticker: str) -> pd.DataFrame | None:
        if not use_trade_toxic_global or trades_root is None:
            return None
        k = (sym, date)
        if k not in trade_cache:
            trade_cache[k] = load_option_trades(trades_root, sym, date)
        return path_for_ticker_trades(trade_cache[k], ticker)

    def get_stock_1s(sym: str, date: str) -> pd.DataFrame | None:
        if not use_adv_vol_global or stock_1s_root is None:
            return None
        from maga7.common.bar_agg import load_stock_1s_day

        k = (sym, date)
        if k not in stock_1s_cache:
            stock_1s_cache[k] = load_stock_1s_day(stock_1s_root, sym, date)
        df = stock_1s_cache[k]
        return df if df is not None and not df.empty else None

    def spot_at(sym: str, date: str, ts) -> float | None:
        sdf = stock_by.get(sym)
        if sdf is None or sdf.empty:
            return None
        ts = to_ny(ts)
        day = sdf[sdf["date"] == date]
        if day.empty:
            return None
        bar = day[day["timestamp"] <= ts].tail(1)
        if bar.empty:
            return None
        px = float(bar.iloc[0]["close"])
        return px if np.isfinite(px) and px > 0 else None

    pos = float(trade.get("position_frac", 0.25))
    cooldown = int(trade.get("cooldown_minutes", 5))
    max_n = int(trade.get("max_entries_per_symbol", 5))
    # Honor trade.day_circuit whenever set (single / m5 / m5_circuit).
    # Legacy gate required "circuit" in scheme name; that left peer3 single
    # unprotected despite profile day_circuit=-0.05. Opt out with null / false.
    circuit_raw = trade.get("day_circuit", None)
    if circuit_raw is False or circuit_raw is None:
        circuit = None
    else:
        circuit = float(circuit_raw)
    only_win = resolve_only_win_reenter(trade)
    use_reentry = scheme.startswith("m5")
    n_size_full = 0
    n_circuit_halt = 0
    n_size_split = 0
    n_skip_max_concurrent = 0
    n_post_win_skip = 0
    n_post_win_scale = 0
    n_same_dir_win_block = 0
    sizing_mode_seen = None
    block_same_dir = block_same_dir_after_win_enabled(trade)
    prev_big_win_dirs: set[tuple[str, str]] = set()

    reg_cfg = profile.get("regime") or {}
    # After N consecutive losing days, skip the next session (flat day resets streak).
    day_loss_halt = reg_cfg.get("day_loss_streak_halt")
    day_loss_halt_n = int(day_loss_halt) if day_loss_halt is not None else None
    # Event calendar blackout (FOMC / mega earnings / giant IPO).
    # Prefer regime.*; trade.* keys accepted as alias for research profiles.
    event_cfg = {
        **{k: reg_cfg[k] for k in (
            "event_calendar_block",
            "event_calendar",
            "event_dates",
            "event_blackout_sessions",
            "event_symbol_blackout",
        ) if k in reg_cfg},
        **{k: trade[k] for k in (
            "event_calendar_block",
            "event_calendar",
            "event_dates",
            "event_blackout_sessions",
            "event_symbol_blackout",
        ) if k in trade},
    }
    session_dates = sorted(str(d) for d in event_sigs["date"].unique()) if len(event_sigs) else []
    # Also include stock calendar so +N sessions expand beyond TopK-only days.
    for sdf in stock_by.values():
        if sdf is not None and not getattr(sdf, "empty", True) and "date" in sdf.columns:
            session_dates.extend(str(x) for x in sdf["date"].unique())
    session_dates = sorted(d for d in set(session_dates) if start <= d <= end)
    event_plan = resolve_event_blackout_plan(event_cfg, session_dates=session_dates)
    event_blackout = set(event_plan.full_days)
    n_event_block = 0
    n_event_symbol_block = 0
    n_displace = 0
    n_displace_skip_score = 0
    n_commit_days = 0
    n_commit_pool = 0
    n_commit_selected = 0
    n_commit_post_fill = 0
    n_topk_backfill = 0
    n_topk_backfill_cap = 0
    top2_keys = {
        (str(r.date), str(r.symbol).upper(), str(r.dir).upper())
        for r in top2.itertuples(index=False)
    } if topk_backfill and len(top2) else set()

    trades: list[dict[str, Any]] = []
    eq = 100.0
    peak = 100.0
    maxdd = 0.0
    daily_rows = []
    n_day_halt = 0
    loss_streak = 0
    prev_day_ret: float | None = None
    post_win_left = 0
    post_win_sessions = post_win_cooldown_sessions(trade)
    post_win_thr = trade.get("post_win_cooldown_day_ret", 0.10)
    bar_delay_seconds = int(trade.get("bar_availability_delay_seconds", 0) or 0)
    bar_delay = pd.Timedelta(seconds=bar_delay_seconds)
    displace_ratio = float(trade.get("displace_min_score_ratio", 1.0) or 1.0)
    sim_kwargs_common = dict(
        fill=fill,
        tp_mult=float(trade.get("tp_mult", 1.6)),
        sl_mult=float(trade.get("sl_mult", 0.4)),
        hold_minutes=int(trade.get("hold_minutes", 30)),
        exit_mf_grace_seconds=int(trade.get("exit_mf_grace_seconds", 60)),
        exit_min_hold_minutes=trade.get("exit_min_hold_minutes"),
        mtm_floor_ret=trade.get("mtm_floor_ret"),
        flow_cum_floor=trade.get("flow_cum_floor"),
        stock_bar_delay_seconds=bar_delay_seconds,
        trail_activate=trade.get("trail_activate"),
        trail_dd=trade.get("trail_dd"),
        hold_extend_minutes=trade.get("hold_extend_minutes"),
        hold_extend_mtm_min=trade.get("hold_extend_mtm_min"),
        hold_extend_require_mf=bool(trade.get("hold_extend_require_mf", True)),
        hold_extend_require_stock=bool(trade.get("hold_extend_require_stock", False)),
        hold_extend_stock_min=float(trade.get("hold_extend_stock_min", 0.0) or 0.0),
        hold_extend_min_peak_mfe=trade.get("hold_extend_min_peak_mfe"),
        hold_extend_max_qqq_adverse=trade.get("hold_extend_max_qqq_adverse"),
        stale_cut_minutes=trade.get("stale_cut_minutes"),
        stale_cut_mtm_max=float(trade.get("stale_cut_mtm_max", 0.0) or 0.0),
        stale_cut_stock_max=float(trade.get("stale_cut_stock_max", 0.0) or 0.0),
        early_exit_mode=trade.get("early_exit_mode"),
        mae_cut_ret=trade.get("mae_cut_ret"),
        mae_cut_mfe_bypass=trade.get("mae_cut_mfe_bypass"),
        mae_cut_min_hold_minutes=trade.get("mae_cut_min_hold_minutes"),
        mae_cut_only_dn=bool(trade.get("mae_cut_only_dn", False)),
        mae_cut_require_mf_against=bool(trade.get("mae_cut_require_mf_against", False)),
        dyn_max_hold_minutes=trade.get("dyn_max_hold_minutes"),
        dyn_min_hold_minutes=trade.get("dyn_min_hold_minutes"),
        dyn_trail_start_minutes=trade.get("dyn_trail_start_minutes"),
        dyn_slope_lookback=trade.get("dyn_slope_lookback"),
        dyn_fast_opp_bars=trade.get("dyn_fast_opp_bars"),
        dyn_fast_pct=trade.get("dyn_fast_pct"),
        dyn_require_price_break=bool(trade.get("dyn_require_price_break", True)),
        hold_watchdog=hold_watchdog_from_trade(trade),
        scale_in=scale_in_from_trade(trade),
        trade_toxic=ttox_cfg,
        delta_time_stop=delta_time_stop_from_trade(trade),
        roi_time_stop=roi_time_stop_from_trade(trade),
        adverse_soft=adverse_soft_from_trade(trade),
        adverse_vol_share=adverse_vol_share_from_trade(trade),
        stock_rev_exit=stock_rev_exit_from_trade(trade),
        wave_abort=wave_abort_from_trade(trade),
        ladder_active=ladder_active_from_trade(trade),
    )
    hwd_cfg = sim_kwargs_common["hold_watchdog"]
    dts_cfg = sim_kwargs_common["delta_time_stop"]
    rts_cfg = sim_kwargs_common["roi_time_stop"]
    adv_cfg = sim_kwargs_common["adverse_soft"]
    avs_cfg = sim_kwargs_common["adverse_vol_share"]
    lac_cfg_global = sim_kwargs_common["ladder_active"]
    ladder_fallback_exit = str(
        trade.get("ladder_fallback_exit_mode") or "hold_extend"
    ).strip() or "hold_extend"
    n_ladder_days = 0
    n_ladder_fallback_days = 0
    n_stock_rev_days = 0
    n_stock_rev_off_days = 0
    fast_pack_cfg = path_fast_pack_from_trade(trade)
    n_fast_pack_days = 0
    n_fast_pack_off_days = 0
    n_delta_stop = 0
    n_roi_time_stop = 0
    n_adverse_soft = 0
    n_adverse_soft_armed = 0
    n_adv_vol = 0
    n_adv_vol_armed = 0
    morn_r5_cfg = morning_r5_scale_from_trade(trade)
    n_morn_r5_scale = 0

    # Day loop keys: Rule-A dates ∪ (optional) Hunt calendar dates.
    # Without the Hunt pad, single-name sleeves miss washout_reclaim days that
    # have no Rule-A fire (groupby never yields empty days). Mag7 May–Jul had
    # zero such days; QQQ-only does.
    _sig_cols = list(event_sigs.columns) if len(event_sigs) else [
        "date",
        "symbol",
        "dir",
        "sig_ts",
        "spot",
        "rank",
    ]
    _by_date: dict[str, pd.DataFrame] = {}
    if len(event_sigs):
        for _d, _g in event_sigs.groupby("date", sort=True):
            _by_date[str(_d)] = _g
    _loop_dates = sorted(_by_date.keys())
    if watchdog is not None and bool(getattr(watchdog.cfg, "hunter_enabled", False)):
        _cal: set[str] = set()
        for _s in symbols:
            _sdf = trade_stock.get(_s)
            if _sdf is None or getattr(_sdf, "empty", True):
                continue
            for _dd in _sdf["date"].astype(str).unique():
                _ds = str(_dd)
                if start <= _ds <= end:
                    _cal.add(_ds)
        _loop_dates = sorted(set(_loop_dates) | _cal)
    _empty_day = pd.DataFrame(columns=_sig_cols)

    for date in _loop_dates:
        day_sigs = _by_date.get(str(date), _empty_day)
        day_seat_armed = False
        day_seat_arm_reason = "off"
        if seat_gate_cfg.enabled:
            topk_day = (
                top2[top2["date"].astype(str) == str(date)].copy()
                if len(top2)
                else top2
            )
            day_seat_armed, day_seat_arm_reason = day_gate_armed(
                seat_gate_cfg,
                topk_day=topk_day,
                stock_by=stock_by,
                date=str(date),
            )
        # Watchdog / legacy router: restore baseline then apply day's overlay.
        day_route = "baseline"
        day_watchdog_state = WatchdogState.NORMAL.value
        day_watchdog_reason = "off"
        day_ladder_on = bool(getattr(lac_cfg_global, "enabled", False))
        srev_cfg_global = sim_kwargs_common.get("stock_rev_exit")
        day_stock_rev_on = bool(getattr(srev_cfg_global, "enabled", False))
        day_fast_pack_on = False
        day_exit_mode_base = str(trade.get("exit_mode") or trade.get("stock_exit") or "none")
        if watchdog is not None and regime_gate is not None:
            wd_dec = watchdog.begin_day(
                str(date),
                stock_by=stock_by,
                qqq_df=qqq_frame if qqq_frame is not None else stock_by.get("QQQ"),
                symbols=symbols,
            )
            watchdog.apply_to_regime(regime_gate.cfg, router_regime_snap)
            day_watchdog_state = wd_dec.state.value
            day_watchdog_reason = wd_dec.reason
            day_route = wd_dec.overlay.route_tag or "baseline"
            if wd_dec.state != WatchdogState.NORMAL:
                n_watchdog_days += 1
                n_router_expert_days += 1
                watchdog_state_counts[day_watchdog_state] = int(
                    watchdog_state_counts.get(day_watchdog_state, 0)
                ) + 1
                if wd_dec.expert:
                    router_day_counts[str(wd_dec.expert)] = int(
                        router_day_counts.get(str(wd_dec.expert), 0)
                    ) + 1
        elif router_on and regime_gate is not None:
            _router_restore_regime(regime_gate.cfg, router_regime_snap)
            if router_mode == "rule":
                hit = _eval_router_rule(
                    router_rule,
                    date=str(date),
                    stock_by=stock_by,
                    qqq_df=qqq_frame if qqq_frame is not None else stock_by.get("QQQ"),
                    symbols=symbols,
                    asof_hhmm=router_asof,
                    router_cfg=profile.get("regime_router")
                    if isinstance(profile.get("regime_router"), dict)
                    else {},
                )
                day_route = str(hit or "baseline")
            else:
                day_route = str(router_labels.get(str(date), "baseline"))
            if day_route not in {"", "baseline", "ok", "other_loss", "wide_chop"} and day_route in router_experts:
                _router_apply_expert(regime_gate.cfg, router_experts.get(day_route))
                n_router_expert_days += 1
                router_day_counts[day_route] = int(router_day_counts.get(day_route, 0)) + 1
                day_watchdog_state = WatchdogState.DEGRADE.value
                day_watchdog_reason = f"legacy_router:{day_route}"
            else:
                day_route = "baseline"
        # Conditional ladder: arm only on mixed_wash_up / prevention days.
        if bool(getattr(lac_cfg_global, "enabled", False)):
            prev_cfg = {}
            if isinstance(profile.get("watchdog"), dict):
                prev_cfg = (profile["watchdog"].get("prevention") or {}) if isinstance(
                    profile["watchdog"].get("prevention"), dict
                ) else {}
            day_ladder_on = ladder_day_should_arm(
                lac_cfg_global,
                date=str(date),
                stock_by=stock_by,
                qqq_df=qqq_frame if qqq_frame is not None else stock_by.get("QQQ"),
                symbols=list(symbols),
                asof=str(prev_cfg.get("asof") or (profile.get("watchdog") or {}).get("asof") or "10:30"),
                washout_breadth_min=int(prev_cfg.get("washout_breadth_min", 3) or 3),
                wash_drop_min=float(prev_cfg.get("wash_drop_min", 0.008) or 0.008),
                frac_above_min=float(prev_cfg.get("frac_above_min", 0.35) or 0.35),
                frac_above_max=float(prev_cfg.get("frac_above_max", 0.70) or 0.70),
            )
            if day_ladder_on:
                n_ladder_days += 1
            else:
                n_ladder_fallback_days += 1
        # Conditional STOCK_REV: arm only on mixed_wash_up days when when≠always.
        if bool(getattr(srev_cfg_global, "enabled", False)):
            prev_cfg = {}
            if isinstance(profile.get("watchdog"), dict):
                prev_cfg = (profile["watchdog"].get("prevention") or {}) if isinstance(
                    profile["watchdog"].get("prevention"), dict
                ) else {}
            day_stock_rev_on = stock_rev_day_should_arm(
                srev_cfg_global,
                date=str(date),
                stock_by=stock_by,
                qqq_df=qqq_frame if qqq_frame is not None else stock_by.get("QQQ"),
                symbols=list(symbols),
                asof=str(prev_cfg.get("asof") or (profile.get("watchdog") or {}).get("asof") or "10:30"),
                washout_breadth_min=int(prev_cfg.get("washout_breadth_min", 3) or 3),
                wash_drop_min=float(prev_cfg.get("wash_drop_min", 0.008) or 0.008),
                frac_above_min=float(prev_cfg.get("frac_above_min", 0.35) or 0.35),
                frac_above_max=float(prev_cfg.get("frac_above_max", 0.70) or 0.70),
            )
            if day_stock_rev_on:
                n_stock_rev_days += 1
            else:
                n_stock_rev_off_days += 1
        if bool(getattr(fast_pack_cfg, "enabled", False)):
            prev_cfg = {}
            if isinstance(profile.get("watchdog"), dict):
                prev_cfg = (profile["watchdog"].get("prevention") or {}) if isinstance(
                    profile["watchdog"].get("prevention"), dict
                ) else {}
            day_fast_pack_on = path_fast_pack_day_should_arm(
                fast_pack_cfg,
                date=str(date),
                stock_by=stock_by,
                qqq_df=qqq_frame if qqq_frame is not None else stock_by.get("QQQ"),
                symbols=list(symbols),
                asof=str(
                    fast_pack_cfg.asof
                    or prev_cfg.get("asof")
                    or (profile.get("watchdog") or {}).get("asof")
                    or "10:30"
                ),
                washout_breadth_min=int(
                    fast_pack_cfg.washout_breadth_min
                    if fast_pack_cfg.washout_breadth_min is not None
                    else prev_cfg.get("washout_breadth_min", 3)
                    or 3
                ),
                wash_drop_min=float(prev_cfg.get("wash_drop_min", 0.008) or 0.008),
                frac_above_min=float(prev_cfg.get("frac_above_min", 0.35) or 0.35),
                frac_above_max=float(prev_cfg.get("frac_above_max", 0.70) or 0.70),
            )
            if day_fast_pack_on:
                n_fast_pack_days += 1
            else:
                n_fast_pack_off_days += 1
        day_state_dec = None
        if state_gate_on:
            day_state_dec = state_gate.begin_day(
                str(date),
                stock_by=stock_by,
                qqq_df=qqq_frame if qqq_frame is not None else stock_by.get("QQQ"),
                symbols=list(symbols),
            )
            st_name = str(day_state_dec.state or "unknown")
            state_gate_day_counts[st_name] = int(state_gate_day_counts.get(st_name, 0)) + 1
        day_chop_dec = None
        if chop_gate_on:
            day_chop_dec = chop_gate.begin_day(
                str(date),
                stock_by=stock_by,
                qqq_df=qqq_frame if qqq_frame is not None else stock_by.get("QQQ"),
                symbols=list(symbols),
            )
            ch_name = str(day_chop_dec.state or "unknown")
            chop_gate_day_counts[ch_name] = int(chop_gate_day_counts.get(ch_name, 0)) + 1
        if session_flow_gate_on:
            day_sfg = session_flow_gate.begin_day(
                str(date),
                stock_by=stock_by,
                qqq_df=qqq_frame if qqq_frame is not None else stock_by.get("QQQ"),
                vixy_df=stock_by.get("VIXY"),
                symbols=list(symbols),
            )
            sf_name = str(day_sfg.state or "unknown")
            session_flow_day_counts[sf_name] = int(session_flow_day_counts.get(sf_name, 0)) + 1
        syms = list(day_sigs.sort_values("sig_ts")["symbol"].unique())
        # Keep concurrent cap tied to configured top_k even when universe expands.
        n_sym = max(int(sig_cfg.get("top_k", 2)), 1)
        day_start = eq
        halt = False
        # Day-level Mag7–QQQ corr rewire (causal asof 10:30); default OFF.
        day_cr = None
        day_cr_scale = 1.0
        if cr_on and cr_cfg is not None:
            asof_hhmm = str(cr_cfg.get("asof", "10:30") or "10:30")
            try:
                hh, mm = asof_hhmm.split(":")
                asof_cr = pd.Timestamp(f"{date} {int(hh):02d}:{int(mm):02d}:00", tz=NY)
            except Exception:
                asof_cr = pd.Timestamp(f"{date} 10:30:00", tz=NY)
            cr_syms = cr_cfg.get("symbols") or symbols
            day_cr = corr_rewire_asof(
                stock_by,
                asof_ts=asof_cr,
                symbols=list(cr_syms),
                event_bars=int(cr_cfg.get("event_bars", 60) or 60),
                calm_bars=int(cr_cfg.get("calm_bars", 180) or 180),
                min_bars=int(cr_cfg.get("min_bars", 30) or 30),
                edge_threshold=float(cr_cfg.get("edge_threshold", 0.5) or 0.5),
                rewire_min=(
                    float(cr_cfg["rewire_min"])
                    if cr_cfg.get("rewire_min") is not None
                    else None
                ),
                rho_event_min=(
                    float(cr_cfg["rho_event_min"])
                    if cr_cfg.get("rho_event_min") is not None
                    else None
                ),
                rho_event_max=(
                    float(cr_cfg["rho_event_max"])
                    if cr_cfg.get("rho_event_max") is not None
                    else None
                ),
                edge_density_min=(
                    float(cr_cfg["edge_density_min"])
                    if cr_cfg.get("edge_density_min") is not None
                    else None
                ),
                action=str(cr_cfg.get("action", "scale") or "scale"),
                scale=float(cr_cfg.get("scale", 0.5) or 0.5),
            )
            day_cr_scale = float(day_cr.size_scale)
            if day_cr.trigger:
                n_corr_rewire_days += 1
        skip_day = bool(day_loss_halt_n is not None and loss_streak >= day_loss_halt_n)
        if cr_on and day_cr is not None and day_cr.trigger and day_cr_scale <= 0.0:
            skip_day = True
            n_corr_rewire_block += 1
        post_win_mode, post_win_scale = post_win_cooldown_action(
            trade, prev_day_ret=prev_day_ret, cooldown_left=post_win_left
        )
        if post_win_mode == "skip":
            skip_day = True
            n_post_win_skip += 1
        if str(date) in event_blackout:
            skip_day = True
            n_event_block += 1
        day_sym_block = event_plan.symbols_blocked_on(str(date))
        if skip_day:
            n_day_halt += 1

        if not use_reentry:
            events = []
            for r in day_sigs.itertuples(index=False):
                if str(r.symbol).upper() in day_sym_block:
                    n_event_symbol_block += 1
                    continue
                fp = getattr(r, "from_prev", None)
                try:
                    fp_f = float(fp) if fp is not None and pd.notna(fp) else None
                except (TypeError, ValueError):
                    fp_f = None
                events.append(
                    (to_ny(r.sig_ts) + bar_delay, to_ny(r.sig_ts), r.symbol, r.dir, fp_f)
                )
        else:
            events = []
            rule_kw = _rule_a_kwargs_from_cfg(sig_cfg)
            for r in day_sigs.itertuples(index=False):
                if str(r.symbol).upper() in day_sym_block:
                    n_event_symbol_block += 1
                    continue
                fp = getattr(r, "from_prev", None)
                try:
                    fp_f = float(fp) if fp is not None and pd.notna(fp) else None
                except (TypeError, ValueError):
                    fp_f = None
                for ts0 in all_rule_a_times(
                    stock_by[r.symbol][stock_by[r.symbol]["date"] == date],
                    r.dir,
                    **rule_kw,
                ):
                    feature_ts = to_ny(ts0)
                    events.append((feature_ts + bar_delay, feature_ts, r.symbol, r.dir, fp_f))
            events.sort(key=lambda x: x[0])

        if skip_day:
            events = []

        # Normalize baseline events to 6-tuple: + source
        events = [
            (e[0], e[1], e[2], e[3], e[4], "baseline") if len(e) == 5 else e for e in events
        ]

        # Watchdog Hunter: inject short-window armed candidates (before Rule-A chronologically).
        day_hunt_symbols: set[str] = set()
        day_hunt_dirs: set[tuple[str, str]] = set()
        day_hunt_n = 0
        if watchdog is not None and watchdog.hunt_armed and not skip_day:
            for hc in watchdog.hunt_candidates:
                if str(hc.symbol).upper() in day_sym_block:
                    n_event_symbol_block += 1
                    continue
                n_hunt_signals += 1
                entry_ts = to_ny(hc.sig_ts) + bar_delay
                # Drop if entry would fall outside arm TTL (after delay).
                if entry_ts > to_ny(hc.armed_until):
                    continue
                events.append(
                    (entry_ts, to_ny(hc.sig_ts), hc.symbol, hc.direction, None, "hunt")
                )
                if hc.symbol not in syms:
                    syms.append(hc.symbol)
            events.sort(key=lambda x: (x[0], str(x[5]), str(x[2])))

        # Deferred TopK commit: collect pre-commit fires → rank → enter at commit clock.
        if commit_tod is not None and events and not use_reentry:
            commit_ts = pd.Timestamp(f"{date} {commit_tod}", tz=NY)
            commit_entry = commit_ts + bar_delay
            pre: list[dict[str, Any]] = []
            post: list[tuple] = []
            for ev in events:
                ts, feature_ts, sym, direction, sig_from_prev = ev[0], ev[1], ev[2], ev[3], ev[4]
                src0 = ev[5] if len(ev) > 5 else "baseline"
                if src0 == "hunt":
                    # Hunt entries are immediate (not deferred TopK auction).
                    post.append((ts, feature_ts, sym, direction, sig_from_prev, "hunt"))
                    continue
                if block_same_dir and (str(sym).upper(), str(direction).upper()) in prev_big_win_dirs:
                    n_same_dir_win_block += 1
                    continue
                if regime_gate is not None:
                    dec0 = regime_gate.check(direction, feature_ts)
                    if not dec0.allow:
                        n_regime_block += 1
                        continue
                reason0 = _structure_gate_blocks(
                    stock_by.get(sym),
                    date=str(date),
                    asof_ts=feature_ts,
                    direction=str(direction),
                    block_dn_if_above_open=block_dn_if_above_open,
                    vwap_dir_lock=vwap_dir_lock,
                    block_dn_if_vwap_lod=block_dn_if_vwap_lod,
                    lod_bounce_min=lod_bounce_min,
                )
                if reason0 == "dn_above_open":
                    n_dn_above_open_block += 1
                    continue
                if reason0 in {"vwap_lock_up", "vwap_lock_dn"}:
                    n_vwap_lock_block += 1
                    continue
                if reason0 == "dn_vwap_lod_bounce":
                    n_dn_vwap_lod_block += 1
                    continue
                peer_n0 = None
                if peer_align_min_i is not None and peer_align_min_i > 0:
                    peer_n0 = count_peer_align(
                        stock_by,
                        date=str(date),
                        asof_ts=feature_ts,
                        direction=str(direction),
                        peer_symbols=peer_symbols,
                        mode=peer_align_mode,
                        streak_min=int(sig_cfg.get("streak_min", 8)),
                    )
                    if peer_n0 < peer_align_min_i:
                        n_peer_block += 1
                        continue
                if not _mf_idio_allows_entry(
                    sym, direction, feature_ts, str(date), loss_streak_n=loss_streak
                ):
                    continue
                if not _tcn_allows_entry(sym, direction, feature_ts, dec0 if regime_gate is not None else None):
                    continue
                if not _lgbm_allows_entry(sym, direction, feature_ts):
                    continue
                score = _topk_rank_score(
                    commit_rank, from_prev=sig_from_prev, peer_n=peer_n0
                )
                item = {
                    "ts": ts,
                    "feature_ts": feature_ts,
                    "sym": sym,
                    "direction": direction,
                    "sig_from_prev": sig_from_prev,
                    "peer_n": peer_n0,
                    "score": score,
                    "source": "baseline",
                }
                if feature_ts <= commit_ts:
                    pre.append(item)
                else:
                    post.append((ts, feature_ts, sym, direction, sig_from_prev, "baseline"))
            pre.sort(
                key=lambda c: (-float(c["score"]), to_ny(c["feature_ts"]), str(c["sym"]))
            )
            winners = pre[:n_sym]
            n_commit_days += 1
            n_commit_pool += len(pre)
            n_commit_selected += len(winners)
            new_events: list[tuple] = []
            for c in winners:
                # Enter at commit clock (causal: selection uses only info ≤ commit).
                new_events.append(
                    (
                        commit_entry,
                        c["feature_ts"],
                        c["sym"],
                        c["direction"],
                        c["sig_from_prev"],
                        "baseline",
                    )
                )
            if post_commit_fill:
                won = {c["sym"] for c in winners}
                for ev in post:
                    if ev[2] in won:
                        continue
                    new_events.append(ev)
                    n_commit_post_fill += 1
            events = new_events

        last_exit = {s: None for s in syms}
        last_win = {s: True for s in syms}
        n_done = {s: 0 for s in syms}
        open_until = {s: None for s in syms}
        day_big_win_dirs: set[tuple[str, str]] = set()
        # Slots = filtered TopK seats (regime/peer cleared). Blocks do not consume;
        # quote/sim failure still consumes so we do not cascade into late noise.
        day_slots = 0
        day_slot_cap = int(n_sym) if topk_backfill and not use_reentry else None

        for ev in events:
            ts, feature_ts, sym, direction, sig_from_prev = ev[0], ev[1], ev[2], ev[3], ev[4]
            event_source = ev[5] if len(ev) > 5 else "baseline"
            is_hunt = event_source == "hunt"
            if halt:
                break
            if day_slot_cap is not None and day_slots >= day_slot_cap and not is_hunt:
                n_topk_backfill_cap += 1
                continue
            if is_hunt and watchdog is not None and watchdog.hunt_budget_remaining() <= 0:
                n_hunt_budget_skip += 1
                continue
            # Mutex vs prior hunt: default same-symbol; research may use symbol_dir.
            if (
                (not is_hunt)
                and watchdog is not None
                and watchdog.cfg.hunter_mutex_with_baseline
            ):
                scope = str(getattr(watchdog.cfg, "hunter_mutex_scope", "symbol") or "symbol").lower()
                if scope in {"symbol_dir", "dir", "same_dir"}:
                    mutex_hit = (str(sym), str(direction).upper()) in day_hunt_dirs
                else:
                    mutex_hit = str(sym) in day_hunt_symbols
                if mutex_hit:
                    n_hunt_mutex_skip += 1
                    continue
            if block_same_dir and (str(sym).upper(), str(direction).upper()) in prev_big_win_dirs:
                n_same_dir_win_block += 1
                continue
            if sym not in n_done:
                n_done[sym] = 0
                last_exit[sym] = None
                last_win[sym] = True
                open_until[sym] = None
            if use_reentry:
                if n_done[sym] >= max_n:
                    continue
                if open_until[sym] is not None and ts < open_until[sym]:
                    continue
                if last_exit[sym] is not None and ts < last_exit[sym] + pd.Timedelta(minutes=cooldown):
                    continue
                if only_win and n_done[sym] > 0 and not last_win[sym]:
                    continue
            else:
                if n_done[sym] >= 1:
                    allow_opp = (
                        (not is_hunt)
                        and watchdog is not None
                        and bool(getattr(watchdog.cfg, "hunter_allow_baseline_opposite", False))
                        and str(sym) in day_hunt_symbols
                        and (str(sym), str(direction).upper()) not in day_hunt_dirs
                        and any(s == str(sym) and d != str(direction).upper() for s, d in day_hunt_dirs)
                    )
                    if not allow_opp:
                        continue

            if regime_gate is not None and not (
                is_hunt and watchdog is not None and watchdog.cfg.hunter_skip_qqq_align
            ):
                dec = regime_gate.check(direction, feature_ts)
                if not dec.allow:
                    n_regime_block += 1
                    continue
            else:
                dec = None

            # State gate: day-regime veto / scale (never predicts Call/Put).
            state_entry_scale = 1.0
            state_entry_tag = None
            if state_gate_on:
                sg_dec = state_gate.decide_entry(str(direction))
                state_entry_tag = f"{sg_dec.state}:{sg_dec.reason}"
                if not sg_dec.allow:
                    n_state_gate_block += 1
                    continue
                state_entry_scale = float(sg_dec.size_scale)

            # Chop gate: calm-range / mixed-tape overlay (parallel to state_gate).
            chop_entry_scale = 1.0
            chop_entry_tag = None
            if chop_gate_on:
                cg_dec = chop_gate.decide_entry(str(direction))
                chop_entry_tag = f"{cg_dec.state}:{cg_dec.reason}"
                if not cg_dec.allow:
                    n_chop_gate_block += 1
                    continue
                chop_entry_scale = float(cg_dec.size_scale)

            # Session cumflow + QQQ/VIXY chop gate (proactive leader filter).
            sfg_entry_scale = 1.0
            sfg_entry_tag = None
            if session_flow_gate_on:
                sfg_dec = session_flow_gate.decide_entry(
                    symbol=str(sym),
                    direction=str(direction),
                    asof_ts=feature_ts,
                )
                sfg_entry_tag = f"{sfg_dec.state}:{sfg_dec.reason}"
                if not sfg_dec.allow:
                    n_session_flow_block += 1
                    continue
                sfg_entry_scale = float(sfg_dec.size_scale)

            reason_s = _structure_gate_blocks(
                stock_by.get(sym),
                date=str(date),
                asof_ts=feature_ts,
                direction=str(direction),
                block_dn_if_above_open=block_dn_if_above_open,
                vwap_dir_lock=vwap_dir_lock,
                block_dn_if_vwap_lod=block_dn_if_vwap_lod,
                lod_bounce_min=lod_bounce_min,
            )
            if reason_s == "dn_above_open":
                n_dn_above_open_block += 1
                continue
            if reason_s in {"vwap_lock_up", "vwap_lock_dn"}:
                n_vwap_lock_block += 1
                continue
            if reason_s == "dn_vwap_lod_bounce":
                n_dn_vwap_lod_block += 1
                continue

            peer_n = None
            si_val = None
            pe_val = None
            pe_ma_val = None
            skip_peer = bool(
                is_hunt and watchdog is not None and watchdog.cfg.hunter_skip_peer
            )
            if (not skip_peer) and peer_align_min_i is not None and peer_align_min_i > 0:
                peer_n = count_peer_align(
                    stock_by,
                    date=str(date),
                    asof_ts=feature_ts,
                    direction=str(direction),
                    peer_symbols=peer_symbols,
                    mode=peer_align_mode,
                    streak_min=int(sig_cfg.get("streak_min", 8)),
                )
                if peer_n < peer_align_min_i:
                    n_peer_block += 1
                    continue

            if not is_hunt and not _mf_idio_allows_entry(
                sym, direction, feature_ts, str(date), loss_streak_n=loss_streak
            ):
                continue
            if not is_hunt and not _tcn_allows_entry(sym, direction, feature_ts, dec):
                continue
            if not is_hunt and not _lgbm_allows_entry(sym, direction, feature_ts):
                continue

            if si_min_f is not None:
                si_val = sync_index(
                    stock_by,
                    date=str(date),
                    asof_ts=feature_ts,
                    peer_symbols=peer_symbols,
                )
                # Require SI aligned with trade direction and |SI| >= si_min
                if si_val is None or abs(si_val) < si_min_f:
                    n_si_block += 1
                    continue
                if (direction == "UP" and si_val < 0) or (direction == "DN" and si_val > 0):
                    n_si_block += 1
                    continue

            if pe_min_ratio_f is not None:
                allow_pe, pe_val, pe_ma_val = price_efficiency_ok(
                    stock_by.get(sym),
                    asof_ts=feature_ts,
                    direction=str(direction),
                    window=pe_window,
                    min_ratio=pe_min_ratio_f,
                    lookback_bars=pe_lookback,
                )
                if not allow_pe:
                    n_pe_block += 1
                    continue

            tod_z_val = None
            if tod_z_min_f is not None:
                allow_z, tod_z_val = tod_mf_z_ok(
                    stock_by.get(sym),
                    asof_ts=feature_ts,
                    direction=str(direction),
                    lookback_days=tod_z_lookback,
                    z_min=tod_z_min_f,
                )
                if not allow_z:
                    n_tod_z_block += 1
                    continue

            # Chase cap: skip Rule-A already extended beyond max |from_prev|.
            if max_from_prev_abs is not None and sig_from_prev is not None:
                try:
                    if abs(float(sig_from_prev)) > float(max_from_prev_abs) + 1e-12:
                        n_max_fp_block += 1
                        continue
                except (TypeError, ValueError):
                    pass

            # Session-open extension (from_open): hard block before seating.
            sig_from_open = None
            from_open_size_mult = 1.0
            if from_open_cfg.enabled:
                sig_from_open = session_from_open(
                    stock_by.get(sym), date=str(date), asof_ts=feature_ts
                )
                fo_act, fo_mult, sig_from_open = resolve_from_open_gate(
                    from_open_cfg,
                    from_open=sig_from_open,
                    direction=str(direction),
                )
                if fo_act == "block":
                    n_from_open_block += 1
                    continue
                if fo_act == "scale":
                    from_open_size_mult = float(fo_mult)

            # Overnight gap trap: large open gap aligned with direction.
            overnight_gap_size_mult = 1.0
            if overnight_gap_cfg.enabled:
                og_adv = None
                og_lag = int(getattr(overnight_gap_cfg, "lag_seconds", 0) or 0)
                og_ts = to_ny(ts) + pd.Timedelta(seconds=og_lag) if og_lag > 0 else ts
                if getattr(overnight_gap_cfg, "require_adv_share", None) is not None:
                    og_adv = adverse_vol_share_asof(
                        prepare_stock_1s_arrays(get_stock_1s(sym, date)),
                        now_ts=og_ts,
                        window_seconds=120,
                        direction=str(direction),
                    )
                og = resolve_overnight_gap_gate(
                    overnight_gap_cfg,
                    stock_df=stock_by.get(sym),
                    date=str(date),
                    direction=str(direction),
                    adv_share=og_adv,
                )
                if not og.allow:
                    n_overnight_gap_block += 1
                    continue
                if abs(float(og.size_scale) - 1.0) > 1e-12:
                    overnight_gap_size_mult = float(og.size_scale)
                    n_overnight_gap_scale += 1
                    # Causal: waited for adv confirm → enter at measurement clock.
                    if og_lag > 0:
                        ts = og_ts

            # Weak-peer + medium overnight gap (SL/TOX stall: 04-08 / 02-18).
            peer_gap_size_mult = 1.0
            if peer_gap_cfg.enabled:
                if sig_from_open is None:
                    sig_from_open = session_from_open(
                        stock_by.get(sym), date=str(date), asof_ts=feature_ts
                    )
                pg = resolve_peer_gap_gate(
                    peer_gap_cfg,
                    stock_df=stock_by.get(sym),
                    date=str(date),
                    direction=str(direction),
                    peer_n=peer_n,
                    from_open=sig_from_open,
                )
                if not pg.allow:
                    n_peer_gap_block += 1
                    continue
                if abs(float(pg.size_scale) - 1.0) > 1e-12:
                    peer_gap_size_mult = float(pg.size_scale)
                    n_peer_gap_scale += 1

            # DN gap mid-extension stall (02-17 GOOGL): measure at feature clock.
            dn_gap_stall_size_mult = 1.0
            if dn_gap_stall_cfg.enabled:
                dgs = resolve_dn_gap_stall_gate(
                    dn_gap_stall_cfg,
                    stock_df=stock_by.get(sym),
                    date=str(date),
                    asof_ts=feature_ts,
                    direction=str(direction),
                    peer_n=peer_n,
                )
                if not dgs.allow:
                    n_dn_gap_stall_block += 1
                    continue
                if abs(float(dgs.size_scale) - 1.0) > 1e-12:
                    dn_gap_stall_size_mult = float(dgs.size_scale)
                    n_dn_gap_stall_scale += 1

            # UP gap early stall (06-11 TSLA): feature clock — entry fo/chase drift.
            up_gap_stall_size_mult = 1.0
            if up_gap_stall_cfg.enabled:
                ugs = resolve_up_gap_stall_gate(
                    up_gap_stall_cfg,
                    stock_df=stock_by.get(sym),
                    date=str(date),
                    asof_ts=feature_ts,
                    direction=str(direction),
                )
                if not ugs.allow:
                    n_up_gap_stall_block += 1
                    continue
                if abs(float(ugs.size_scale) - 1.0) > 1e-12:
                    up_gap_stall_size_mult = float(ugs.size_scale)
                    n_up_gap_stall_scale += 1

            # Large fo + LOD/HOD chase (07-24 TSLA): feature clock.
            fo_lod_chase_size_mult = 1.0
            if fo_lod_chase_cfg.enabled:
                flc = resolve_fo_lod_chase_gate(
                    fo_lod_chase_cfg,
                    stock_df=stock_by.get(sym),
                    date=str(date),
                    asof_ts=feature_ts,
                    direction=str(direction),
                )
                if not flc.allow:
                    n_fo_lod_chase_block += 1
                    continue
                if abs(float(flc.size_scale) - 1.0) > 1e-12:
                    fo_lod_chase_size_mult = float(flc.size_scale)
                    n_fo_lod_chase_scale += 1

            range_stall_size_mult = 1.0

            # Entry confirm: wait N bars after Rule-A; keep peer/regime at fire time.
            confirm_ft = None
            confirm_mf = None
            use_confirm = confirm_bars_n > 0
            if use_confirm and confirm_weekdays is not None:
                try:
                    wd0 = int(pd.Timestamp(str(date)).weekday())
                except Exception:
                    wd0 = -1
                use_confirm = wd0 in confirm_weekdays
            if use_confirm and confirm_directions is not None:
                use_confirm = str(direction).upper() in confirm_directions
            if use_confirm:
                sdf_c = stock_by.get(sym)
                stock_day_c = None
                if sdf_c is not None and not sdf_c.empty:
                    stock_day_c = sdf_c[sdf_c["date"] == date]
                ok_c, confirm_ft, confirm_mf, _, _ = entry_confirm_ok(
                    stock_day_c,
                    direction=str(direction),
                    feature_ts=feature_ts,
                    confirm_bars=confirm_bars_n,
                    mode=confirm_mode,
                )
                if not ok_c:
                    n_confirm_block += 1
                    continue
                # Shift fill clock to confirm bar + availability delay.
                ts = to_ny(confirm_ft) + bar_delay

            # Stock path confirm: wait for +thr_pos before -thr_neg (causal delay).
            path_confirm_ts = None
            path_confirm_reason = None
            if spc_on:
                use_spc = True
                if spc_weekdays is not None:
                    try:
                        wd_s = int(pd.Timestamp(str(date)).weekday())
                    except Exception:
                        wd_s = -1
                    use_spc = wd_s in spc_weekdays
                if use_spc and spc_tod_start is not None and spc_tod_end is not None:
                    hm = int(to_ny(ts).hour) * 60 + int(to_ny(ts).minute)
                    use_spc = spc_tod_start <= hm <= spc_tod_end
                if use_spc:
                    sdf_p = stock_by.get(sym)
                    stock_day_p = None
                    if sdf_p is not None and not sdf_p.empty:
                        stock_day_p = sdf_p[sdf_p["date"] == date]
                    ok_p, path_confirm_ts, path_confirm_reason = stock_path_confirm_ok(
                        stock_day_p,
                        direction=str(direction),
                        entry_ts=ts,
                        thr_pos=spc_thr_pos,
                        thr_neg=spc_thr_neg,
                        max_wait_seconds=spc_max_wait,
                        on_timeout=spc_on_timeout,
                    )
                    if not ok_p:
                        n_stock_path_confirm_block += 1
                        continue
                    n_stock_path_confirm_ok += 1
                    if (
                        spc_delay_on_pos
                        and path_confirm_ts is not None
                        and path_confirm_reason == "pos"
                    ):
                        ts = to_ny(path_confirm_ts) + bar_delay

            # Entry-side adverse volume share (1s): block or scale before fill.
            entry_adv_share = None
            entry_adv_scale_pending = 1.0
            if bool(entry_avs_cfg.enabled):
                use_eavs = True
                dirs_allow = getattr(entry_avs_cfg, "dirs", None)
                if dirs_allow and str(direction).upper() not in set(dirs_allow):
                    use_eavs = False
                if entry_avs_cfg.tod_start and entry_avs_cfg.tod_end:
                    tod0 = _hhmm_to_minutes(entry_avs_cfg.tod_start)
                    tod1 = _hhmm_to_minutes(entry_avs_cfg.tod_end)
                    hm = int(to_ny(ts).hour) * 60 + int(to_ny(ts).minute)
                    if tod0 is not None and tod1 is not None:
                        use_eavs = use_eavs and (tod0 <= hm <= tod1)
                if use_eavs:
                    lag_s = int(getattr(entry_avs_cfg, "lag_seconds", 0) or 0)
                    eavs_ts = to_ny(ts) + pd.Timedelta(seconds=lag_s) if lag_s > 0 else ts
                    arr = prepare_stock_1s_arrays(get_stock_1s(sym, date))
                    share = adverse_vol_share_asof(
                        arr,
                        now_ts=eavs_ts,
                        window_seconds=int(entry_avs_cfg.window_seconds),
                        direction=str(direction),
                    )
                    entry_adv_share = share
                    hot = share is not None and float(share) >= float(entry_avs_cfg.max_share)
                    missing = share is None
                    if missing and entry_avs_cfg.on_missing == "block":
                        n_entry_adv_vol_block += 1
                        continue
                    if hot:
                        if entry_avs_cfg.action == "block":
                            n_entry_adv_vol_block += 1
                            continue
                        if float(entry_avs_cfg.scale) < 1.0 - 1e-12:
                            entry_adv_scale_pending = float(entry_avs_cfg.scale)
                    # Causal: if we waited for lag to measure share, enter at that clock.
                    if lag_s > 0:
                        ts = eavs_ts

            # Range-chase + pre5 stall: measure at final entry clock (after confirms).
            # feature_ts often still has positive pre5; stall appears by entry_ts.
            if range_stall_cfg.enabled:
                rs = resolve_range_stall_gate(
                    range_stall_cfg,
                    stock_df=stock_by.get(sym),
                    date=str(date),
                    asof_ts=ts,
                    direction=str(direction),
                    peer_n=peer_n,
                )
                if not rs.allow:
                    n_range_stall_block += 1
                    continue
                if abs(float(rs.size_scale) - 1.0) > 1e-12:
                    range_stall_size_mult = float(rs.size_scale)
                    n_range_stall_scale += 1

            # Seat quality gate: low score skips without consuming the daily TopK slot.
            _is_topk_member = (
                (str(date), str(sym).upper(), str(direction).upper()) in top2_keys
                if top2_keys
                else False
            )
            if (
                seat_gate_cfg.enabled
                and (not is_hunt)
                and candidate_gate_active(
                    seat_gate_cfg,
                    day_armed=day_seat_armed,
                    asof_ts=feature_ts,
                    is_topk_member=_is_topk_member,
                )
            ):
                vz_bar = None
                sdf_g = stock_by.get(sym)
                if sdf_g is not None and not sdf_g.empty and "vol_z" in sdf_g.columns:
                    day_g = sdf_g[sdf_g["date"].astype(str) == str(date)]
                    if not day_g.empty:
                        up_g = day_g[day_g["timestamp"] <= to_ny(feature_ts)]
                        if not up_g.empty:
                            vz_bar = up_g.iloc[-1].get("vol_z")
                            try:
                                vz_bar = float(vz_bar) if pd.notna(vz_bar) else None
                            except (TypeError, ValueError):
                                vz_bar = None
                ok_seat, seat_reason, _seat_sc = seat_score_ok(
                    seat_gate_cfg,
                    stock_by=stock_by,
                    symbol=str(sym),
                    date=str(date),
                    asof_ts=feature_ts,
                    from_prev=float(sig_from_prev) if sig_from_prev is not None else None,
                    vol_z=vz_bar,
                )
                if not ok_seat:
                    n_seat_score_skip += 1
                    continue

            # Reserve filtered-TopK seat after gates; quote/sim miss still consumes.
            if day_slot_cap is not None:
                day_slots += 1

            buckets = lock_idx.get((sym, date)) if lock_idx is not None else None
            day_ticker = None
            if buckets:
                bid = BUCKET_MAP[(direction, money)]
                day_ticker = buckets.get(bid)

            spot = spot_at(sym, date, feature_ts)
            ticker = None
            pick_dte = None
            pick_source = None

            if contract_mode in ("open_lock", "open", "open_ladder", "open_lock_ladder"):
                by_dte = multi_lock_idx.get((sym, date)) if multi_lock_idx else None
                ticker, pick_dte, pick_source = resolve_open_lock_contract(
                    by_dte,
                    direction=direction,
                    moneyness=money,
                    spot=spot,
                    prefer_dte=prefer_dte,
                    allowed_dte=allowed_dte,
                    clear_otm_thresh=clear_otm_thresh,
                    ladder=ladder,
                    otm_rungs=otm_rungs,
                )
                if pick_source and "skip0_clear_otm" in str(pick_source):
                    n_skip0_clear_otm += 1
                if not ticker:
                    continue
            elif contract_mode == "signal_atm":
                chain = chain_cache.get(sym, date) if chain_cache is not None else None
                pick = resolve_contract(
                    mode="signal_atm",
                    chain=chain,
                    date=str(date),
                    direction=direction,
                    sig_ts=ts,
                    spot=spot,
                    day_lock_ticker=day_ticker,
                    prefer_dte=prefer_dte,
                    allowed_dte=allowed_dte,
                    fallback_day_lock=True,
                )
                if pick is None:
                    continue
                if pick.source == "signal_atm":
                    n_signal_atm += 1
                elif pick.source == "day_lock_fallback":
                    n_day_lock_fb += 1
                ticker = pick.ticker
                pick_dte = pick.dte if pick.dte >= 0 else None
                pick_source = pick.source
                if (
                    clear_otm_thresh is not None
                    and pick_dte == 0
                    and spot is not None
                    and pick.strike
                    and np.isfinite(pick.strike)
                ):
                    from maga7.common.open_lock import is_clearly_otm

                    if is_clearly_otm(direction, float(spot), float(pick.strike), thresh=clear_otm_thresh):
                        pick2 = resolve_contract(
                            mode="signal_atm",
                            chain=chain,
                            date=str(date),
                            direction=direction,
                            sig_ts=ts,
                            spot=spot,
                            day_lock_ticker=day_ticker,
                            prefer_dte=1,
                            allowed_dte=[d for d in allowed_dte if int(d) >= 1] or [1, 2],
                            fallback_day_lock=False,
                        )
                        if pick2 is not None:
                            n_skip0_clear_otm += 1
                            ticker = pick2.ticker
                            pick_dte = pick2.dte if pick2.dte >= 0 else None
                            pick_source = "signal_atm_skip0_clear_otm"
            else:
                # day_lock (lookahead research default)
                if not day_ticker:
                    continue
                ticker = day_ticker
                pick_source = "day_lock"
                if clear_otm_thresh is not None and spot is not None:
                    from maga7.common.open_lock import is_clearly_otm, strike_from_occ
                    from maga7.common.contract_select import trading_dte
                    import re as _re

                    m = _re.search(r"(\d{6})[CP]", str(ticker).replace("O:", ""))
                    if m:
                        exp = pd.Timestamp("20" + m.group(1)).date()
                        dte0 = trading_dte(exp, date)
                        k = strike_from_occ(ticker)
                        if dte0 == 0 and is_clearly_otm(direction, float(spot), k, thresh=clear_otm_thresh):
                            n_skip0_clear_otm += 1
                            continue
                        pick_dte = dte0

            if max_entry_abs_otm is not None and spot is not None and ticker:
                from maga7.common.open_lock import is_clearly_otm, strike_from_occ

                k_ent = strike_from_occ(str(ticker))
                if np.isfinite(k_ent) and is_clearly_otm(
                    direction,
                    float(spot),
                    float(k_ent),
                    thresh=float(max_entry_abs_otm),
                ):
                    n_skip_max_entry_otm += 1
                    continue

            path, qsrc = get_path(sym, date, ticker)
            if qsrc == "1s":
                n_quote_1s += 1
            elif qsrc == "day_iv":
                n_quote_day_iv += 1
            sdf = stock_by.get(sym)
            stock_day = None
            if sdf is not None and not sdf.empty:
                stock_day = sdf[sdf["date"] == date]
            qqq_day = None
            if bool(getattr(hwd_cfg, "enabled", False)):
                qdf = stock_by.get("QQQ")
                if qdf is not None and not getattr(qdf, "empty", True):
                    qqq_day = qdf[qdf["date"].astype(str) == str(date)]
            exit_mode = day_exit_mode_base
            sim_kw = dict(sim_kwargs_common)
            if bool(getattr(lac_cfg_global, "enabled", False)) and not day_ladder_on:
                # Non-toxic day: fall back to extend rails (never unmanaged).
                exit_mode = ladder_fallback_exit
                sim_kw["ladder_active"] = LadderActiveConfig(enabled=False)
            elif bool(getattr(lac_cfg_global, "enabled", False)) and day_ladder_on:
                exit_mode = "ladder_active"
                sim_kw["ladder_active"] = lac_cfg_global
            if bool(getattr(srev_cfg_global, "enabled", False)) and not day_stock_rev_on:
                sim_kw["stock_rev_exit"] = StockRevExitConfig(enabled=False)
            elif bool(getattr(srev_cfg_global, "enabled", False)) and day_stock_rev_on:
                sim_kw["stock_rev_exit"] = srev_cfg_global
            if day_fast_pack_on:
                sim_kw.update(
                    apply_path_fast_pack_overrides(
                        hold_minutes=int(sim_kw.get("hold_minutes") or 30),
                        trail_activate=sim_kw.get("trail_activate"),
                        trail_dd=sim_kw.get("trail_dd"),
                        stock_rev=sim_kw.get("stock_rev_exit"),
                        pack=fast_pack_cfg,
                    )
                )
            hunt_pos_frac = None
            if is_hunt and watchdog is not None:
                from maga7.common.watchdog import hunt_trade_overrides

                hov = hunt_trade_overrides(watchdog.cfg)
                hunt_pos_frac = hov.pop("position_frac", None)
                if "exit_mode" in hov:
                    exit_mode = str(hov.pop("exit_mode"))
                sim_kw.update(hov)
            tpath = get_trade_path(sym, date, ticker)
            if use_trade_toxic_global:
                if tpath is not None and not tpath.empty:
                    n_trade_path += 1
                else:
                    n_trade_path_miss += 1
            sim_kw["trade_path"] = tpath
            if use_adv_vol_global:
                sim_kw["stock_1s"] = get_stock_1s(sym, date)
            sim = simulate_trade(
                path,
                ts,
                direction=direction,
                stock_day=stock_day,
                exit_mode=exit_mode,
                qqq_day=qqq_day,
                **sim_kw,
            )
            if sim is None:
                continue
            if str(sim.reason).upper() == "DELTA_STOP":
                n_delta_stop += 1
            if str(sim.reason).upper().startswith("ROI_TIME"):
                n_roi_time_stop += 1
            if str(sim.reason).upper() == "ADVERSE_SOFT":
                n_adverse_soft += 1
            if bool(getattr(sim, "adverse_soft_armed", False)):
                n_adverse_soft_armed += 1
            if str(sim.reason).upper() == "ADV_VOL":
                n_adv_vol += 1
            if bool(getattr(sim, "adv_vol_armed", False)):
                n_adv_vol_armed += 1
            size_frac, sizing_mode, n_conc, allow, size_reason = resolve_size_frac(
                trade,
                top_k=n_sym,
                open_until=open_until,
                symbol=sym,
                entry_ts=sim.entry_ts,
            )
            if hunt_pos_frac is not None and float(hunt_pos_frac) > 0:
                size_frac = float(hunt_pos_frac)
                size_reason = f"{size_reason}+hunt_frac:{float(hunt_pos_frac):.2f}"
            if not allow and displace_on:
                # Kick oldest still-open position if later signal clears score gate.
                open_syms = [
                    s
                    for s, until in open_until.items()
                    if s != sym and until is not None and until > sim.entry_ts
                ]
                victims: list[tuple[pd.Timestamp, int, str]] = []
                for s in open_syms:
                    for i in range(len(trades) - 1, -1, -1):
                        row_i = trades[i]
                        if str(row_i.get("date")) != str(date) or row_i.get("symbol") != s:
                            continue
                        if to_ny(row_i["exit_ts"]) > sim.entry_ts:
                            victims.append((to_ny(row_i["entry_ts"]), i, s))
                        break
                victims.sort(key=lambda x: x[0])
                displaced = False
                if victims:
                    _, vic_i, vic_sym = victims[0]
                    old_row = trades[vic_i]
                    new_score = _displace_score(
                        trade, from_prev=sig_from_prev, peer_n=peer_n
                    )
                    old_score = _displace_score(
                        trade,
                        from_prev=old_row.get("sig_from_prev"),
                        peer_n=old_row.get("peer_align_n"),
                    )
                    score_mode = str(trade.get("displace_score") or "none").strip().lower()
                    score_ok = True
                    if score_mode not in {"", "none", "off", "any"}:
                        score_ok = new_score + 1e-12 >= displace_ratio * float(old_score)
                    if not score_ok:
                        n_displace_skip_score += 1
                    else:
                        vic_path, _ = get_path(vic_sym, date, old_row["ticker"])
                        sdf_v = stock_by.get(vic_sym)
                        stock_day_v = None
                        if sdf_v is not None and not sdf_v.empty:
                            stock_day_v = sdf_v[sdf_v["date"] == date]
                        sim_v_kw = dict(sim_kwargs_common)
                        sim_v_kw["trade_path"] = get_trade_path(
                            vic_sym, date, str(old_row["ticker"])
                        )
                        if use_adv_vol_global:
                            sim_v_kw["stock_1s"] = get_stock_1s(vic_sym, date)
                        if bool(getattr(srev_cfg_global, "enabled", False)) and not day_stock_rev_on:
                            sim_v_kw["stock_rev_exit"] = StockRevExitConfig(enabled=False)
                        elif bool(getattr(srev_cfg_global, "enabled", False)) and day_stock_rev_on:
                            sim_v_kw["stock_rev_exit"] = srev_cfg_global
                        if day_fast_pack_on:
                            sim_v_kw.update(
                                apply_path_fast_pack_overrides(
                                    hold_minutes=int(sim_v_kw.get("hold_minutes") or 30),
                                    trail_activate=sim_v_kw.get("trail_activate"),
                                    trail_dd=sim_v_kw.get("trail_dd"),
                                    stock_rev=sim_v_kw.get("stock_rev_exit"),
                                    pack=fast_pack_cfg,
                                )
                            )
                        sim_v = simulate_trade(
                            vic_path,
                            old_row["entry_ts"],
                            direction=str(old_row.get("dir")),
                            stock_day=stock_day_v,
                            exit_mode=exit_mode,
                            force_exit_ts=sim.entry_ts,
                            **sim_v_kw,
                        )
                        if sim_v is not None and str(sim_v.reason) == "DISPLACE":
                            old_sf = float(old_row["size_frac"])
                            old_ret = float(old_row["ret"])
                            factor = 1.0 + old_sf * old_ret
                            if factor > 1e-12:
                                eq /= factor
                            eq *= 1.0 + old_sf * float(sim_v.ret)
                            peak = max(peak, eq)
                            maxdd = min(maxdd, eq / peak - 1.0)
                            old_row["exit"] = sim_v.exit
                            old_row["ret"] = sim_v.ret
                            old_row["reason"] = sim_v.reason
                            old_row["exit_ts"] = sim_v.exit_ts
                            old_row["displaced_by"] = sym
                            old_row["displaced_at"] = sim.entry_ts
                            open_until[vic_sym] = sim_v.exit_ts
                            last_exit[vic_sym] = sim_v.exit_ts
                            last_win[vic_sym] = sim_v.ret > 0
                            n_displace += 1
                            displaced = True
                            size_frac, sizing_mode, n_conc, allow, size_reason = resolve_size_frac(
                                trade,
                                top_k=n_sym,
                                open_until=open_until,
                                symbol=sym,
                                entry_ts=sim.entry_ts,
                            )
                            size_reason = f"{size_reason}+displace"
                if not displaced:
                    n_skip_max_concurrent += 1
                    continue
            elif not allow:
                n_skip_max_concurrent += 1
                continue
            if not allow:
                n_skip_max_concurrent += 1
                continue
            if post_win_mode == "scale" and post_win_scale < 1.0:
                size_frac = float(size_frac) * float(post_win_scale)
                size_reason = f"{size_reason}+post_win_scale"
                n_post_win_scale += 1
            if dec is not None and float(getattr(dec, "size_scale", 1.0) or 1.0) < 1.0:
                size_frac = float(size_frac) * float(dec.size_scale)
                size_reason = f"{size_reason}+regime_scale"
                n_regime_scale += 1
            if state_gate_on and float(state_entry_scale) < 1.0 - 1e-12:
                size_frac = float(size_frac) * float(state_entry_scale)
                size_reason = f"{size_reason}+state_gate:{state_entry_scale:.2f}"
                n_state_gate_scale += 1
            if chop_gate_on and float(chop_entry_scale) < 1.0 - 1e-12:
                size_frac = float(size_frac) * float(chop_entry_scale)
                size_reason = f"{size_reason}+chop_gate:{chop_entry_scale:.2f}"
                n_chop_gate_scale += 1
            if session_flow_gate_on and abs(float(sfg_entry_scale) - 1.0) > 1e-12:
                size_frac = float(size_frac) * float(sfg_entry_scale)
                tag = sfg_entry_tag or f"scale:{sfg_entry_scale:.2f}"
                size_reason = f"{size_reason}+session_flow:{tag}"
                n_session_flow_scale += 1
            if cr_on and day_cr_scale < 1.0 - 1e-12:
                size_frac = float(size_frac) * float(day_cr_scale)
                size_reason = f"{size_reason}+corr_rewire:{day_cr_scale:.2f}"
                n_corr_rewire_scale += 1
            if morn_r5_cfg is not None:
                et = to_ny(sim.entry_ts)
                hm = int(et.hour) * 60 + int(et.minute)
                tod0 = _hhmm_to_minutes(morn_r5_cfg.get("tod_start", "10:31"))
                tod1 = _hhmm_to_minutes(morn_r5_cfg.get("tod_end", "11:00"))
                if tod0 is not None and tod1 is not None and tod0 <= hm <= tod1:
                    lb = max(1, int(morn_r5_cfg.get("lookback_bars", 5) or 5))
                    min_s = float(morn_r5_cfg.get("min_signed_ret", 0.0005) or 0.0005)
                    sc = float(morn_r5_cfg.get("scale", 0.5) or 0.5)
                    sc = max(0.0, min(sc, 1.0))
                    sdf_r = stock_by.get(sym)
                    g_r = None
                    if sdf_r is not None and not sdf_r.empty:
                        g_r = _prepare_stock_day(sdf_r[sdf_r["date"] == date])
                    if g_r is not None and "_close" in g_r.columns:
                        i1 = _stock_bar_index(g_r, et)
                        i0 = max(0, i1 - lb)
                        if i1 > i0:
                            c0 = float(g_r["_close"].iloc[i0])
                            c1 = float(g_r["_close"].iloc[i1])
                            if np.isfinite(c0) and c0 > 0 and np.isfinite(c1):
                                raw = c1 / c0 - 1.0
                                signed = raw if str(direction).upper() == "UP" else -raw
                                if signed < min_s and sc < 1.0 - 1e-12:
                                    size_frac = float(size_frac) * sc
                                    size_reason = f"{size_reason}+morn_r5:{sc:.2f}"
                                    n_morn_r5_scale += 1
            if float(entry_adv_scale_pending) < 1.0 - 1e-12:
                size_frac = float(size_frac) * float(entry_adv_scale_pending)
                size_reason = f"{size_reason}+entry_adv_vol:{float(entry_adv_scale_pending):.2f}"
                n_entry_adv_vol_scale += 1
            idio_mult = _mf_idio_size_mult(
                sym, direction, feature_ts, str(date), loss_streak_n=loss_streak
            )
            if idio_mult < 1.0 - 1e-12:
                size_frac = float(size_frac) * float(idio_mult)
                size_reason = f"{size_reason}+mf_idio_scale:{idio_mult:.2f}"
            tcn_mult, tcn_p, tcn_reason = _tcn_size_mult(sym, direction, feature_ts, dec)
            if tcn_mult <= 0.0:
                n_tcn_block += 1
                continue
            if tcn_mult < 1.0 - 1e-12:
                size_frac = float(size_frac) * float(tcn_mult)
                size_reason = f"{size_reason}+tcn_scale:{tcn_mult:.2f}"
            lgbm_mult, lgbm_p, lgbm_reason = _lgbm_size_mult(sym, direction, feature_ts)
            if lgbm_mult <= 0.0:
                continue
            if lgbm_mult < 1.0 - 1e-12:
                size_frac = float(size_frac) * float(lgbm_mult)
                size_reason = f"{size_reason}+lgbm_scale:{lgbm_mult:.2f}"
            sym_scale = float(symbol_size_scale.get(str(sym).upper(), 1.0))
            if sym_scale <= 0.0:
                continue
            if abs(sym_scale - 1.0) > 1e-12:
                size_frac = float(size_frac) * sym_scale
                size_reason = f"{size_reason}+sym_scale:{sym_scale:.2f}"
            if dvol_size_cfg.enabled:
                # Prefer full universe (incl. peers/ref) for causal cs rank.
                dvol_scale, dvol_rank, _dvol = resolve_dvol_size_scale(
                    dvol_size_cfg,
                    stock_by=stock_by,
                    symbol=str(sym),
                    date=str(date),
                    asof_ts=feature_ts,
                )
                if abs(dvol_scale - 1.0) > 1e-12:
                    size_frac = float(size_frac) * float(dvol_scale)
                    size_reason = (
                        f"{size_reason}+dvol_size:{dvol_scale:.2f}"
                        + (f"(rk{dvol_rank})" if dvol_rank is not None else "")
                    )
                    if dvol_scale > 1.0 + 1e-12:
                        n_dvol_size_boost += 1
            if vrp_size_cfg.enabled:
                vrp_scale, vrp_reason = resolve_vrp_size_scale(
                    vrp_size_cfg, date=str(date), day_table=vrp_day_table
                )
                if vrp_scale <= 0.0:
                    n_vrp_skip += 1
                    continue
                if abs(float(vrp_scale) - 1.0) > 1e-12:
                    size_frac = float(size_frac) * float(vrp_scale)
                    size_reason = f"{size_reason}+{vrp_reason}"
                    n_vrp_size_scale += 1
            if from_open_cfg.enabled and abs(float(from_open_size_mult) - 1.0) > 1e-12:
                size_frac = float(size_frac) * float(from_open_size_mult)
                size_reason = f"{size_reason}+from_open:{from_open_size_mult:.2f}"
            if overnight_gap_cfg.enabled and abs(float(overnight_gap_size_mult) - 1.0) > 1e-12:
                size_frac = float(size_frac) * float(overnight_gap_size_mult)
                size_reason = f"{size_reason}+overnight_gap:{overnight_gap_size_mult:.2f}"
            if peer_gap_cfg.enabled and abs(float(peer_gap_size_mult) - 1.0) > 1e-12:
                size_frac = float(size_frac) * float(peer_gap_size_mult)
                size_reason = f"{size_reason}+peer_gap:{peer_gap_size_mult:.2f}"
            if dn_gap_stall_cfg.enabled and abs(float(dn_gap_stall_size_mult) - 1.0) > 1e-12:
                size_frac = float(size_frac) * float(dn_gap_stall_size_mult)
                size_reason = f"{size_reason}+dn_gap_stall:{dn_gap_stall_size_mult:.2f}"
            if up_gap_stall_cfg.enabled and abs(float(up_gap_stall_size_mult) - 1.0) > 1e-12:
                size_frac = float(size_frac) * float(up_gap_stall_size_mult)
                size_reason = f"{size_reason}+up_gap_stall:{up_gap_stall_size_mult:.2f}"
            if fo_lod_chase_cfg.enabled and abs(float(fo_lod_chase_size_mult) - 1.0) > 1e-12:
                size_frac = float(size_frac) * float(fo_lod_chase_size_mult)
                size_reason = f"{size_reason}+fo_lod_chase:{fo_lod_chase_size_mult:.2f}"
            if range_stall_cfg.enabled and abs(float(range_stall_size_mult) - 1.0) > 1e-12:
                size_frac = float(size_frac) * float(range_stall_size_mult)
                size_reason = f"{size_reason}+range_stall:{range_stall_size_mult:.2f}"
                n_from_open_scale += 1
            purity_score = None
            purity_parts = None
            purity_scale = None
            if purity_on:
                # mf/streak/SI at fire time (causal purity; no EOD).
                sdf_p = stock_by.get(sym)
                stock_day_p = None
                if sdf_p is not None and not sdf_p.empty:
                    stock_day_p = sdf_p[sdf_p["date"] == date]
                mf_p, su_p, sd_p = _stock_mf_at(_prepare_stock_day(stock_day_p), feature_ts)
                streak_p = su_p if str(direction).upper() == "UP" else sd_p
                si_p = si_val
                if si_p is None:
                    si_p = sync_index(
                        stock_by,
                        date=str(date),
                        asof_ts=feature_ts,
                        peer_symbols=peer_symbols,
                    )
                qfp = float(dec.qqq_from_prev) if dec is not None and dec.qqq_from_prev is not None else None
                path_feats = path_efficiency_features(
                    stock_day_p,
                    asof_ts=feature_ts,
                    direction=str(direction),
                    window=purity_path_window,
                )
                purity_score, purity_parts = trend_purity_score(
                    direction=str(direction),
                    from_prev=sig_from_prev,
                    peer_n=peer_n,
                    peer_min=int(peer_align_min_i or 0),
                    peer_universe=max(len(peer_symbols), 1),
                    mf10=mf_p,
                    streak=streak_p,
                    streak_min=int(sig_cfg.get("streak_min", 8)),
                    qqq_from_prev=qfp,
                    si=si_p,
                    fp_ref=purity_fp_ref,
                    features=purity_features,
                    path_eff=path_feats.get("path_eff"),
                    range_eff=path_feats.get("range_eff"),
                    dir_frac=path_feats.get("dir_frac"),
                    adverse=path_feats.get("adverse"),
                    adverse_ref=purity_adverse_ref,
                )
                purity_scale, purity_tag = trend_purity_size_scale(purity_score, trade)
                if purity_scale <= 0.0:
                    n_purity_skip += 1
                    continue
                if purity_scale < 1.0 - 1e-12:
                    size_frac = float(size_frac) * float(purity_scale)
                    size_reason = f"{size_reason}+{purity_tag}:{purity_scale:.2f}"
                    n_purity_scaled += 1
            sizing_mode_seen = sizing_mode
            if n_conc <= 1:
                n_size_full += 1
            else:
                n_size_split += 1
            eq *= 1.0 + size_frac * sim.ret
            peak = max(peak, eq)
            maxdd = min(maxdd, eq / peak - 1.0)
            n_done[sym] += 1
            last_exit[sym] = sim.exit_ts
            open_until[sym] = sim.exit_ts
            last_win[sym] = sim.ret > 0
            from maga7.common.open_lock import strike_from_occ

            row = {
                "date": date,
                "symbol": sym,
                "dir": direction,
                "moneyness": money,
                "ticker": ticker,
                "day_lock_ticker": day_ticker,
                "contract_source": pick_source,
                "quote_source": qsrc,
                "sig_spot": spot,
                "sig_strike": strike_from_occ(ticker),
                "sig_dte": pick_dte,
                "sig_ts": ts,
                "feature_ts": feature_ts,
                "n_in_day": n_done[sym],
                "entry": sim.entry,
                "exit": sim.exit,
                "ret": sim.ret,
                "reason": sim.reason,
                "entry_ts": sim.entry_ts,
                "exit_ts": sim.exit_ts,
                "size_frac": size_frac,
                "n_concurrent": n_conc,
                "position_sizing": sizing_mode,
                "size_reason": size_reason,
                "scale_in_added": bool(getattr(sim, "scale_in_added", False)),
                "scale_in_entry2": getattr(sim, "scale_in_entry2", None),
                "scale_in_ts": getattr(sim, "scale_in_ts", None),
                "scale_in_deployed_frac": float(getattr(sim, "scale_in_deployed_frac", 1.0) or 1.0),
                "adverse_soft_armed": bool(getattr(sim, "adverse_soft_armed", False)),
                "adv_vol_armed": bool(getattr(sim, "adv_vol_armed", False)),
                "entry_adv_vol_share": float(entry_adv_share) if entry_adv_share is not None else None,
                "tcn_p": tcn_p,
                "tcn_reason": tcn_reason,
                "route": day_route,
                "watchdog_state": day_watchdog_state,
                "watchdog_reason": day_watchdog_reason,
                "event_source": event_source,
            }
            if is_hunt:
                row["route"] = "hunt"
                row["watchdog_state"] = WatchdogState.HUNT.value
                hunt_det = (
                    str(watchdog.cfg.hunter_detector)
                    if watchdog is not None
                    else "hunt"
                )
                row["watchdog_reason"] = f"hunt:{hunt_det}"
                if watchdog is not None:
                    watchdog.note_hunt_entry()
                day_hunt_symbols.add(str(sym))
                day_hunt_dirs.add((str(sym), str(direction).upper()))
                day_hunt_n += 1
                n_hunt_trades += 1
                h_circ = getattr(watchdog.cfg, "hunter_day_circuit_ret", None) if watchdog else None
                if h_circ is not None and float(sim.ret) <= float(h_circ):
                    if not halt:
                        n_hunt_day_circuit += 1
                    halt = True
                    row["hunt_day_circuit"] = True
            if dec is not None:
                row["regime_qqq_fp"] = dec.qqq_from_prev
                row["regime_vix_rev"] = dec.vix_reversal
                row["regime_vixy_z"] = dec.vixy_z
            if peer_n is not None:
                row["peer_align_n"] = int(peer_n)
            if si_val is not None:
                row["si"] = float(si_val)
            if pe_val is not None:
                row["pe"] = float(pe_val)
            if tod_z_val is not None:
                row["tod_mf_z"] = float(tod_z_val)
            if pe_ma_val is not None:
                row["pe_ma"] = float(pe_ma_val)
            if sig_from_prev is not None:
                row["sig_from_prev"] = float(sig_from_prev)
            if sig_from_open is not None:
                row["sig_from_open"] = float(sig_from_open)
            if confirm_ft is not None:
                row["confirm_ts"] = confirm_ft
                row["entry_confirm_bars"] = int(confirm_bars_n)
                if confirm_mf is not None:
                    row["confirm_mf"] = float(confirm_mf)
            if path_confirm_ts is not None:
                row["stock_path_confirm_ts"] = path_confirm_ts
                row["stock_path_confirm_reason"] = path_confirm_reason
            if day_cr is not None:
                row["corr_rewire"] = day_cr.rewire
                row["corr_rho_event"] = day_cr.rho_event
                row["corr_rho_calm"] = day_cr.rho_calm
                row["corr_edge_density"] = day_cr.edge_density
                row["corr_rewire_reason"] = day_cr.reason
                row["corr_rewire_scale"] = float(day_cr_scale)
            if purity_score is not None:
                row["trend_purity"] = float(purity_score)
                if purity_scale is not None:
                    row["trend_purity_scale"] = float(purity_scale)
                if purity_parts is not None:
                    for pk, pv in purity_parts.items():
                        row[f"purity_{pk}"] = float(pv)
            if topk_backfill:
                key = (str(date), str(sym).upper(), str(direction).upper())
                is_backfill = key not in top2_keys
                row["topk_backfill"] = bool(is_backfill)
                if is_backfill:
                    n_topk_backfill += 1
            trades.append(row)
            if block_same_dir and is_symbol_dir_big_win(
                ret=float(sim.ret), reason=sim.reason, trade=trade
            ):
                day_big_win_dirs.add((str(sym).upper(), str(direction).upper()))
            if circuit is not None and (eq / day_start - 1.0) <= float(circuit):
                if not halt:
                    n_circuit_halt += 1
                halt = True

        day_ret = eq / day_start - 1.0
        daily_rows.append(
            {
                "date": date,
                "equity": eq,
                "day_ret": day_ret,
                "n": sum(n_done.values()),
                "day_halt": bool(skip_day or halt),
                "route": day_route if (watchdog is not None or router_on) else None,
                "watchdog_state": day_watchdog_state if watchdog is not None or router_on else None,
                "watchdog_reason": day_watchdog_reason if watchdog is not None or router_on else None,
                "hunt_armed": bool(watchdog.hunt_armed) if watchdog is not None else False,
                "n_hunt": int(day_hunt_n),
            }
        )
        if day_ret < 0:
            loss_streak += 1
        else:
            loss_streak = 0
        prev_day_ret = float(day_ret)
        prev_big_win_dirs = day_big_win_dirs
        if (
            post_win_thr is not None
            and str(trade.get("post_win_cooldown_mode") or "off").lower()
            not in {"", "off", "none", "false", "0"}
            and day_ret >= float(post_win_thr)
        ):
            post_win_left = int(post_win_sessions)
        elif post_win_left > 0:
            post_win_left -= 1

    trades_df = pd.DataFrame(trades)
    daily_df = pd.DataFrame(daily_rows)
    summary = {
        "scheme": scheme,
        "moneyness": money,
        "fill_frac": fill.entry_frac,
        "n_trades": int(len(trades_df)),
        "n_days": int(daily_df["date"].nunique()) if len(daily_df) else 0,
        "total_ret": float(eq / 100.0 - 1.0),
        "maxdd": float(maxdd),
        "day_win": float((daily_df["day_ret"] > 0).mean()) if len(daily_df) else float("nan"),
        "trade_win": float((trades_df["ret"] > 0).mean()) if len(trades_df) else float("nan"),
        "trade_exp": float(trades_df["ret"].mean()) if len(trades_df) else float("nan"),
        "start_equity": 100.0,
        "end_equity": float(eq),
        "n_signals_topk": int(len(top2)),
        "topk_backfill_on_block": bool(topk_backfill),
        "n_topk_backfill": int(n_topk_backfill),
        "n_topk_backfill_cap_skip": int(n_topk_backfill_cap),
        "seat_score_gate_enabled": bool(seat_gate_cfg.enabled),
        "seat_score_gate_mode": str(seat_gate_cfg.mode) if seat_gate_cfg.enabled else None,
        "seat_score_gate_when": str(seat_gate_cfg.when) if seat_gate_cfg.enabled else None,
        "n_seat_score_skip": int(n_seat_score_skip),
        "displace_universe": displace_universe,
        "n_regime_block": int(n_regime_block),
        "n_regime_scale": int(n_regime_scale),
        "n_dn_above_open_block": int(n_dn_above_open_block),
        "n_vwap_lock_block": int(n_vwap_lock_block),
        "n_dn_vwap_lod_block": int(n_dn_vwap_lod_block),
        "block_dn_if_above_open": bool(block_dn_if_above_open),
        "vwap_dir_lock": bool(vwap_dir_lock),
        "block_dn_if_vwap_lod": bool(block_dn_if_vwap_lod),
        "lod_bounce_min": float(lod_bounce_min) if block_dn_if_vwap_lod else None,
        "block_dn_if_qqq_above_open": bool(
            (profile.get("regime") or {}).get("block_dn_if_qqq_above_open", False)
        ),
        "n_event_block": int(n_event_block),
        "n_event_symbol_block": int(n_event_symbol_block),
        "event_blackout_dates": sorted(event_blackout),
        "event_symbol_blackout": {
            d: sorted(s) for d, s in sorted(event_plan.symbol_days.items())
        },
        "n_peer_block": int(n_peer_block),
        "n_mf_idio_block": int(n_mf_idio_block),
        "n_mf_idio_scale": int(n_mf_idio_scale),
        "mf_idio_mode": mf_idio_mode if mf_idio_on else None,
        "mf_idio_action": mf_idio_action if mf_idio_on else None,
        "mf_idio_scale": mf_idio_scale if mf_idio_on and mf_idio_action == "scale" else None,
        "mf_idio_after_loss_streak": mf_idio_after_loss_streak if mf_idio_on else None,
        "tcn_gate_enabled": bool(tcn_on),
        "tcn_gate_action": tcn_gate.cfg.action if tcn_on else None,
        "tcn_gate_p_min": tcn_gate.cfg.p_min if tcn_on else None,
        "n_tcn_block": int(n_tcn_block),
        "n_tcn_scale": int(n_tcn_scale),
        "n_tcn_skip_regime": int(n_tcn_skip_regime),
        "lgbm_bouncer_enabled": bool(lgbm_on),
        "lgbm_bouncer_action": lgbm_bouncer.cfg.action if lgbm_on else None,
        "lgbm_bouncer_p_min": lgbm_bouncer.cfg.p_min if lgbm_on else None,
        "n_lgbm_block": int(n_lgbm_block),
        "n_lgbm_scale": int(n_lgbm_scale),
        "regime_router_enabled": bool(router_on),
        "regime_router_mode": router_mode if router_on else None,
        "regime_router_rule": router_rule if router_on and router_mode == "rule" else None,
        "n_router_expert_days": int(n_router_expert_days),
        "router_day_counts": dict(router_day_counts),
        "watchdog_enabled": bool(watchdog is not None),
        "watchdog_mode": watchdog.cfg.mode if watchdog is not None else None,
        "n_watchdog_days": int(n_watchdog_days),
        "watchdog_state_counts": dict(watchdog_state_counts),
        "hunter_enabled": bool(watchdog.cfg.hunter_enabled) if watchdog is not None else False,
        "n_hunt_signals": int(n_hunt_signals),
        "n_hunt_trades": int(n_hunt_trades),
        "n_hunt_day_circuit": int(n_hunt_day_circuit),
        "n_hunt_budget_skip": int(n_hunt_budget_skip),
        "n_hunt_mutex_skip": int(n_hunt_mutex_skip),
        "mf_idio_min_frac": mf_idio_min_frac if mf_idio_on and mf_idio_mode in {"frac", "min_frac", "fraction"} else None,
        "mf_idio_beta_days": mf_idio_beta_days if mf_idio_on else None,
        "mf_idio_beta_on": mf_idio_beta_on if mf_idio_on else None,
        "n_si_block": int(n_si_block),
        "n_pe_block": int(n_pe_block),
        "n_tod_z_block": int(n_tod_z_block),
        "n_confirm_block": int(n_confirm_block),
        "max_from_prev_abs": max_from_prev_abs,
        "n_max_fp_block": int(n_max_fp_block),
        "from_open_gate_enabled": bool(from_open_cfg.enabled),
        "from_open_gate_max_abs": float(from_open_cfg.max_abs) if from_open_cfg.enabled else None,
        "from_open_gate_mode": from_open_cfg.mode if from_open_cfg.enabled else None,
        "n_from_open_block": int(n_from_open_block),
        "n_from_open_scale": int(n_from_open_scale),
        "overnight_gap_gate_enabled": bool(overnight_gap_cfg.enabled),
        "overnight_gap_max_fav": float(overnight_gap_cfg.max_fav_gap) if overnight_gap_cfg.enabled else None,
        "n_overnight_gap_block": int(n_overnight_gap_block),
        "n_overnight_gap_scale": int(n_overnight_gap_scale),
        "peer_gap_gate_enabled": bool(peer_gap_cfg.enabled),
        "n_peer_gap_block": int(n_peer_gap_block),
        "n_peer_gap_scale": int(n_peer_gap_scale),
        "range_stall_gate_enabled": bool(range_stall_cfg.enabled),
        "n_range_stall_block": int(n_range_stall_block),
        "n_range_stall_scale": int(n_range_stall_scale),
        "dn_gap_stall_gate_enabled": bool(dn_gap_stall_cfg.enabled),
        "n_dn_gap_stall_block": int(n_dn_gap_stall_block),
        "n_dn_gap_stall_scale": int(n_dn_gap_stall_scale),
        "up_gap_stall_gate_enabled": bool(up_gap_stall_cfg.enabled),
        "n_up_gap_stall_block": int(n_up_gap_stall_block),
        "n_up_gap_stall_scale": int(n_up_gap_stall_scale),
        "fo_lod_chase_gate_enabled": bool(fo_lod_chase_cfg.enabled),
        "n_fo_lod_chase_block": int(n_fo_lod_chase_block),
        "n_fo_lod_chase_scale": int(n_fo_lod_chase_scale),
        "entry_confirm_bars": int(confirm_bars_n) if confirm_bars_n > 0 else None,
        "entry_confirm_mode": confirm_mode if confirm_bars_n > 0 else None,
        "stock_path_confirm_enabled": bool(spc_on),
        "stock_path_confirm_thr_pos": spc_thr_pos if spc_on else None,
        "stock_path_confirm_thr_neg": spc_thr_neg if spc_on else None,
        "stock_path_confirm_max_wait_seconds": spc_max_wait if spc_on else None,
        "n_stock_path_confirm_block": int(n_stock_path_confirm_block),
        "n_stock_path_confirm_ok": int(n_stock_path_confirm_ok),
        "state_gate_enabled": bool(state_gate_on),
        "n_state_gate_block": int(n_state_gate_block),
        "n_state_gate_scale": int(n_state_gate_scale),
        "state_gate_day_counts": dict(state_gate_day_counts),
        "chop_gate_enabled": bool(chop_gate_on),
        "chop_gate_mode": chop_gate.cfg.mode if chop_gate_on else None,
        "n_chop_gate_block": int(n_chop_gate_block),
        "n_chop_gate_scale": int(n_chop_gate_scale),
        "chop_gate_day_counts": dict(chop_gate_day_counts),
        "session_flow_gate_enabled": bool(session_flow_gate_on),
        "session_flow_gate_when": session_flow_gate.cfg.when if session_flow_gate_on else None,
        "n_session_flow_block": int(n_session_flow_block),
        "n_session_flow_scale": int(n_session_flow_scale),
        "session_flow_day_counts": dict(session_flow_day_counts),
        "corr_rewire_enabled": bool(cr_on),
        "n_corr_rewire_days": int(n_corr_rewire_days),
        "n_corr_rewire_scale": int(n_corr_rewire_scale),
        "n_corr_rewire_block": int(n_corr_rewire_block),
        "trend_purity_sizing": bool(purity_on),
        "trend_purity_features": purity_features if purity_on else None,
        "n_purity_skip": int(n_purity_skip),
        "n_purity_scaled": int(n_purity_scaled),
        "peer_align_min": peer_align_min_i,
        "peer_align_mode": peer_align_mode if peer_align_min_i is not None else None,
        "si_min": si_min_f,
        "pe_min_ratio": pe_min_ratio_f,
        "tod_mf_z_min": tod_z_min_f,
        "n_day_halt": int(n_day_halt),
        "n_circuit_halt": int(n_circuit_halt),
        "day_circuit": circuit,
        "n_post_win_skip": int(n_post_win_skip),
        "n_post_win_scale": int(n_post_win_scale),
        "post_win_cooldown_mode": str(trade.get("post_win_cooldown_mode") or "off"),
        "post_win_cooldown_day_ret": trade.get("post_win_cooldown_day_ret"),
        "block_same_dir_after_win": bool(block_same_dir),
        "n_same_dir_win_block": int(n_same_dir_win_block),
        "regime_enabled": bool(regime_gate is not None),
        "day_loss_streak_halt": day_loss_halt_n,
        "contract_mode": contract_mode,
        "quote_source_mode": quote_source,
        "n_signal_atm": int(n_signal_atm),
        "n_day_lock_fallback": int(n_day_lock_fb),
        "n_skip0_clear_otm": int(n_skip0_clear_otm),
        "n_skip_max_entry_otm": int(n_skip_max_entry_otm),
        "max_entry_abs_otm_pct": max_entry_abs_otm,
        "n_quote_1s": int(n_quote_1s),
        "n_quote_day_iv": int(n_quote_day_iv),
        "clear_otm_ban_0dte_pct": clear_otm_thresh,
        "reentry_mode": str(trade.get("reentry_mode") or "").strip() or None,
        "only_win_reenter": bool(only_win),
        "position_frac": pos,
        "position_sizing": sizing_mode_seen or str(trade.get("position_sizing") or "concurrent"),
        "max_concurrent_positions": int(trade.get("max_concurrent_positions") or n_sym),
        "dvol_size_scale_enabled": bool(dvol_size_cfg.enabled),
        "dvol_size_scale_max": float(dvol_size_cfg.max_scale) if dvol_size_cfg.enabled else None,
        "n_dvol_size_boost": int(n_dvol_size_boost),
        "vrp_size_scale_enabled": bool(vrp_size_cfg.enabled),
        "vrp_size_scale_mode": vrp_size_cfg.mode if vrp_size_cfg.enabled else None,
        "n_vrp_size_scale": int(n_vrp_size_scale),
        "n_vrp_skip": int(n_vrp_skip),
        "n_size_full": int(n_size_full),
        "n_size_split": int(n_size_split),
        "n_skip_max_concurrent": int(n_skip_max_concurrent),
        "n_displace": int(n_displace),
        "n_displace_skip_score": int(n_displace_skip_score),
        "displace_on_later": bool(displace_on),
        "displace_universe": displace_universe,
        "displace_score": str(trade.get("displace_score") or "none"),
        "n_signals_all_first": int(len(all_first)),
        "topk_commit_tod": commit_tod,
        "topk_rank": commit_rank if commit_tod else None,
        "topk_post_commit_fill": bool(post_commit_fill) if commit_tod else None,
        "n_commit_days": int(n_commit_days),
        "n_commit_pool": int(n_commit_pool),
        "n_commit_selected": int(n_commit_selected),
        "n_commit_post_fill_events": int(n_commit_post_fill),
        "exit_mode": str(trade.get("exit_mode") or trade.get("stock_exit") or "none"),
        "early_exit_mode": trade.get("early_exit_mode"),
        "mae_cut_ret": trade.get("mae_cut_ret")
        if str(trade.get("early_exit_mode") or "").lower() in {"mae_cut", "toxic_cut"}
        or "mae_cut" in str(trade.get("exit_mode") or "").lower()
        else None,
        "mae_cut_only_dn": bool(trade.get("mae_cut_only_dn", False))
        if trade.get("early_exit_mode") or "mae_cut" in str(trade.get("exit_mode") or "").lower()
        else None,
        "n_mae_cut": int((trades_df["reason"] == "MAE_CUT").sum())
        if len(trades_df) and "reason" in trades_df.columns
        else 0,
        "scale_in_enabled": bool(scale_in_from_trade(trade).enabled),
        "n_scale_in_added": int(trades_df["scale_in_added"].sum())
        if len(trades_df) and "scale_in_added" in trades_df.columns
        else 0,
        "trade_toxic_enabled": bool(use_trade_toxic_global),
        "trade_toxic_cut_ret": float(ttox_cfg.cut_ret) if use_trade_toxic_global else None,
        "trade_toxic_mfe_bypass": float(ttox_cfg.mfe_bypass) if use_trade_toxic_global else None,
        "delta_time_stop_enabled": bool(getattr(dts_cfg, "enabled", False)),
        "n_delta_stop": int(n_delta_stop),
        "roi_time_stop_enabled": bool(getattr(rts_cfg, "enabled", False)),
        "n_roi_time_stop": int(n_roi_time_stop),
        "ladder_active_enabled": bool(getattr(lac_cfg_global, "enabled", False)),
        "ladder_when": str(getattr(lac_cfg_global, "when", "always") or "always")
        if getattr(lac_cfg_global, "enabled", False)
        else None,
        "n_ladder_days": int(n_ladder_days),
        "n_ladder_fallback_days": int(n_ladder_fallback_days),
        "n_stock_rev_days": int(n_stock_rev_days),
        "n_stock_rev_off_days": int(n_stock_rev_off_days),
        "stock_rev_when": str(
            getattr(sim_kwargs_common.get("stock_rev_exit"), "when", "off") or "off"
        ),
        "path_fast_pack_enabled": bool(getattr(fast_pack_cfg, "enabled", False)),
        "path_fast_pack_when": str(getattr(fast_pack_cfg, "when", "off") or "off")
        if getattr(fast_pack_cfg, "enabled", False)
        else None,
        "n_fast_pack_days": int(n_fast_pack_days),
        "n_fast_pack_off_days": int(n_fast_pack_off_days),
        "morning_r5_scale_enabled": bool(morn_r5_cfg is not None),
        "n_morn_r5_scale": int(n_morn_r5_scale),
        "adverse_soft_enabled": bool(getattr(adv_cfg, "enabled", False)),
        "adverse_soft_mode": str(getattr(adv_cfg, "mode", "")) if getattr(adv_cfg, "enabled", False) else None,
        "n_adverse_soft": int(n_adverse_soft),
        "n_adverse_soft_armed": int(n_adverse_soft_armed),
        "adverse_vol_share_enabled": bool(getattr(avs_cfg, "enabled", False)),
        "n_adv_vol": int(n_adv_vol),
        "n_adv_vol_armed": int(n_adv_vol_armed),
        "entry_adv_vol_enabled": bool(entry_avs_cfg.enabled),
        "n_entry_adv_vol_block": int(n_entry_adv_vol_block),
        "n_entry_adv_vol_scale": int(n_entry_adv_vol_scale),
        "n_trade_path": int(n_trade_path),
        "n_trade_path_miss": int(n_trade_path_miss),
        "n_trade_tox": int((trades_df["reason"] == "TRADE_TOX").sum())
        if len(trades_df) and "reason" in trades_df.columns
        else 0,
    }
    return {"summary": summary, "trades": trades_df, "daily": daily_df, "topk": top2}

"""Offline replay: TopK Rule-A → ATM/OTM contract → quote fills.

Contract modes:
  - day_lock (default): step1 open/day 4-bucket lock
  - signal_atm: re-pick ATM from day_iv at/before sig_ts (research)
"""
from __future__ import annotations

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
from maga7.common.position_size import (
    block_same_dir_after_win_enabled,
    is_symbol_dir_big_win,
    post_win_cooldown_action,
    post_win_cooldown_sessions,
    resolve_size_frac,
)
from maga7.common.event_calendar import resolve_event_blackout
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
    day["_mf"] = day["mf10"].astype(float)
    day["_su"] = day["streak_up"].astype(int) if "streak_up" in day.columns else 0
    day["_sd"] = day["streak_dn"].astype(int) if "streak_dn" in day.columns else 0
    if "net$" in day.columns:
        day["_net"] = day["net$"].astype(float)
        day["_net_csum"] = day["_net"].cumsum()
    return day


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
    force_exit_ts=None,
    early_exit_mode: str | None = None,
) -> SimResult | None:
    """Option path fill with TP/SL/time, optional stock-window or MTM trail exit.

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

    Soft cuts may be **stacked** on extend via ``exit_mode="hold_extend+mtm_floor"``
    (or ``+mf_flip`` / ``+mf_reversal``), or via ``early_exit_mode``.

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
    require_mtm = "flow_mtm" in blob or "flow_soft" in blob
    act = float(trail_activate) if trail_activate is not None else 0.20
    dd = float(trail_dd) if trail_dd is not None else 0.15
    floor = float(mtm_floor_ret) if mtm_floor_ret is not None else 0.0
    flow_floor = float(flow_cum_floor) if flow_cum_floor is not None else 0.0
    ext_hold = int(hold_extend_minutes) if hold_extend_minutes is not None else 45
    if ext_hold <= base_hold:
        ext_hold = base_hold
    ext_end = entry_ts + pd.Timedelta(minutes=ext_hold)
    ext_mtm_min = float(hold_extend_mtm_min) if hold_extend_mtm_min is not None else 0.0
    require_mf_align = bool(hold_extend_require_mf)
    extended = False
    peak_ret = -np.inf
    trail_armed = False
    # mf_reversal / mtm_floor: wait ~10m; flow_*: wait ~5m before soft exit.
    min_hold_m = exit_min_hold_minutes
    if min_hold_m is None and ("mf_reversal" in blob or use_floor):
        min_hold_m = 10.0
    if min_hold_m is None and use_flow:
        min_hold_m = 5.0
    grace_secs = int(exit_mf_grace_seconds)
    if min_hold_m is not None:
        grace_secs = max(grace_secs, int(float(min_hold_m) * 60))
    grace_until = entry_ts + pd.Timedelta(seconds=grace_secs)
    day = stock_day
    if (use_mf or use_flow or use_extend) and day is not None and not day.empty:
        day = _prepare_stock_day(day)
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
        if t >= end_ts:
            if use_extend and (not extended) and end_ts == base_end and ext_hold > base_hold:
                cur_ret = float(p) / entry - 1.0
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
                if mtm_ok and mf_ok:
                    extended = True
                    end_ts = ext_end
                    continue
                reason, exit_px, exit_ts = f"T+{base_hold}", float(p), t
                break
            reason, exit_px, exit_ts = f"T+{ext_hold if extended else base_hold}", float(p), t
            break
        cur_ret = float(p) / entry - 1.0
        if use_trail:
            if cur_ret > peak_ret:
                peak_ret = cur_ret
            if (not trail_armed) and peak_ret >= act:
                trail_armed = True
            if trail_armed and cur_ret <= peak_ret - dd:
                reason, exit_px, exit_ts = "TRAIL", float(p), t
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
    return SimResult(
        entry=entry,
        exit=exit_px,
        ret=exit_px / entry - 1.0,
        reason=reason,
        entry_ts=ts_list[0],
        exit_ts=exit_ts,
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
    # Prior sessions for beta need bars before ``start``.
    load_start = start
    if mf_idio_on:
        load_start = (pd.Timestamp(start) - pd.Timedelta(days=max(14, mf_idio_beta_days * 3))).strftime(
            "%Y-%m-%d"
        )
        months = month_list(load_start, end)

    if stock_by is None:
        stock_by = {}
        # Load excluded names too if they remain in peer_symbols (breadth only).
        load_syms = list(dict.fromkeys(list(symbols) + list(sig_cfg.get("peer_symbols") or [])))
        if mf_idio_on and "QQQ" not in {str(s).upper() for s in load_syms}:
            load_syms.append("QQQ")
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
    contract_mode = str(trade.get("contract_mode", "day_lock")).lower()
    quote_source = str(trade.get("quote_source", "1s")).lower()  # 1s | day_iv | auto
    half_spread = float(trade.get("day_iv_half_spread_frac", 0.01))
    prefer_dte, allowed_dte = lock_policy_from_profile(profile)
    clear_otm = trade.get("clear_otm_ban_0dte_pct", None)
    clear_otm_thresh = float(clear_otm) if clear_otm is not None else None

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
    day_iv_root = paths.get("day_iv_root")
    chain_cache = DayIvChainCache(day_iv_root) if day_iv_root else None
    n_signal_atm = 0
    n_day_lock_fb = 0
    n_skip0_clear_otm = 0
    n_quote_1s = 0
    n_quote_day_iv = 0

    from maga7.common.regime import Mag7RegimeGate

    if regime_gate is None:
        regime_gate = Mag7RegimeGate.from_profile(profile, months=months)
    n_regime_block = 0
    n_regime_scale = 0
    n_peer_block = 0
    n_mf_idio_block = 0
    n_mf_idio_scale = 0
    n_si_block = 0
    n_pe_block = 0
    n_tod_z_block = 0
    peer_align_min = sig_cfg.get("peer_align_min")
    peer_align_min_i = int(peer_align_min) if peer_align_min is not None else None
    peer_align_mode = str(sig_cfg.get("peer_align_mode", "mf10")).strip().lower()
    peer_symbols = list(sig_cfg.get("peer_symbols") or profile.get("symbols") or [])
    qqq_frame = stock_by.get("QQQ")
    mf_idio_beta_cache: dict[tuple[str, str], float | None] = {}

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
    n_confirm_block = 0
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
        ) if k in reg_cfg},
        **{k: trade[k] for k in (
            "event_calendar_block",
            "event_calendar",
            "event_dates",
            "event_blackout_sessions",
        ) if k in trade},
    }
    session_dates = sorted(str(d) for d in event_sigs["date"].unique()) if len(event_sigs) else []
    # Also include stock calendar so +N sessions expand beyond TopK-only days.
    for sdf in stock_by.values():
        if sdf is not None and not getattr(sdf, "empty", True) and "date" in sdf.columns:
            session_dates.extend(str(x) for x in sdf["date"].unique())
    session_dates = sorted(d for d in set(session_dates) if start <= d <= end)
    event_blackout = resolve_event_blackout(event_cfg, session_dates=session_dates)
    n_event_block = 0
    n_displace = 0
    n_displace_skip_score = 0
    n_commit_days = 0
    n_commit_pool = 0
    n_commit_selected = 0
    n_commit_post_fill = 0

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
        early_exit_mode=trade.get("early_exit_mode"),
    )

    for date, day_sigs in event_sigs.groupby("date", sort=True):
        syms = list(day_sigs.sort_values("sig_ts")["symbol"].unique())
        # Keep concurrent cap tied to configured top_k even when universe expands.
        n_sym = max(int(sig_cfg.get("top_k", 2)), 1)
        day_start = eq
        halt = False
        skip_day = bool(day_loss_halt_n is not None and loss_streak >= day_loss_halt_n)
        post_win_mode, post_win_scale = post_win_cooldown_action(
            trade, prev_day_ret=prev_day_ret, cooldown_left=post_win_left
        )
        if post_win_mode == "skip":
            skip_day = True
            n_post_win_skip += 1
        if str(date) in event_blackout:
            skip_day = True
            n_event_block += 1
        if skip_day:
            n_day_halt += 1

        if not use_reentry:
            events = []
            for r in day_sigs.itertuples(index=False):
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

        # Deferred TopK commit: collect pre-commit fires → rank → enter at commit clock.
        if commit_tod is not None and events and not use_reentry:
            commit_ts = pd.Timestamp(f"{date} {commit_tod}", tz=NY)
            commit_entry = commit_ts + bar_delay
            pre: list[dict[str, Any]] = []
            post: list[tuple] = []
            for ts, feature_ts, sym, direction, sig_from_prev in events:
                if block_same_dir and (str(sym).upper(), str(direction).upper()) in prev_big_win_dirs:
                    n_same_dir_win_block += 1
                    continue
                if regime_gate is not None:
                    dec0 = regime_gate.check(direction, feature_ts)
                    if not dec0.allow:
                        n_regime_block += 1
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
                }
                if feature_ts <= commit_ts:
                    pre.append(item)
                else:
                    post.append((ts, feature_ts, sym, direction, sig_from_prev))
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

        for ts, feature_ts, sym, direction, sig_from_prev in events:
            if halt:
                break
            if block_same_dir and (str(sym).upper(), str(direction).upper()) in prev_big_win_dirs:
                n_same_dir_win_block += 1
                continue
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
                    continue

            if regime_gate is not None:
                dec = regime_gate.check(direction, feature_ts)
                if not dec.allow:
                    n_regime_block += 1
                    continue
            else:
                dec = None

            peer_n = None
            si_val = None
            pe_val = None
            pe_ma_val = None
            if peer_align_min_i is not None and peer_align_min_i > 0:
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

            if not _mf_idio_allows_entry(
                sym, direction, feature_ts, str(date), loss_streak_n=loss_streak
            ):
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

            # Entry confirm: wait N bars after Rule-A; keep peer/regime at fire time.
            confirm_ft = None
            confirm_mf = None
            if confirm_bars_n > 0:
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

            path, qsrc = get_path(sym, date, ticker)
            if qsrc == "1s":
                n_quote_1s += 1
            elif qsrc == "day_iv":
                n_quote_day_iv += 1
            sdf = stock_by.get(sym)
            stock_day = None
            if sdf is not None and not sdf.empty:
                stock_day = sdf[sdf["date"] == date]
            exit_mode = str(trade.get("exit_mode") or trade.get("stock_exit") or "none")
            sim = simulate_trade(
                path,
                ts,
                direction=direction,
                stock_day=stock_day,
                exit_mode=exit_mode,
                **sim_kwargs_common,
            )
            if sim is None:
                continue
            size_frac, sizing_mode, n_conc, allow, size_reason = resolve_size_frac(
                trade,
                top_k=n_sym,
                open_until=open_until,
                symbol=sym,
                entry_ts=sim.entry_ts,
            )
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
                        sim_v = simulate_trade(
                            vic_path,
                            old_row["entry_ts"],
                            direction=str(old_row.get("dir")),
                            stock_day=stock_day_v,
                            exit_mode=exit_mode,
                            force_exit_ts=sim.entry_ts,
                            **sim_kwargs_common,
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
            idio_mult = _mf_idio_size_mult(
                sym, direction, feature_ts, str(date), loss_streak_n=loss_streak
            )
            if idio_mult < 1.0 - 1e-12:
                size_frac = float(size_frac) * float(idio_mult)
                size_reason = f"{size_reason}+mf_idio_scale:{idio_mult:.2f}"
            sym_scale = float(symbol_size_scale.get(str(sym).upper(), 1.0))
            if sym_scale <= 0.0:
                continue
            if abs(sym_scale - 1.0) > 1e-12:
                size_frac = float(size_frac) * sym_scale
                size_reason = f"{size_reason}+sym_scale:{sym_scale:.2f}"
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
            }
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
            if confirm_ft is not None:
                row["confirm_ts"] = confirm_ft
                row["entry_confirm_bars"] = int(confirm_bars_n)
                if confirm_mf is not None:
                    row["confirm_mf"] = float(confirm_mf)
            if purity_score is not None:
                row["trend_purity"] = float(purity_score)
                if purity_scale is not None:
                    row["trend_purity_scale"] = float(purity_scale)
                if purity_parts is not None:
                    for pk, pv in purity_parts.items():
                        row[f"purity_{pk}"] = float(pv)
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
        "n_regime_block": int(n_regime_block),
        "n_regime_scale": int(n_regime_scale),
        "n_event_block": int(n_event_block),
        "event_blackout_dates": sorted(event_blackout),
        "n_peer_block": int(n_peer_block),
        "n_mf_idio_block": int(n_mf_idio_block),
        "n_mf_idio_scale": int(n_mf_idio_scale),
        "mf_idio_mode": mf_idio_mode if mf_idio_on else None,
        "mf_idio_action": mf_idio_action if mf_idio_on else None,
        "mf_idio_scale": mf_idio_scale if mf_idio_on and mf_idio_action == "scale" else None,
        "mf_idio_after_loss_streak": mf_idio_after_loss_streak if mf_idio_on else None,
        "mf_idio_min_frac": mf_idio_min_frac if mf_idio_on and mf_idio_mode in {"frac", "min_frac", "fraction"} else None,
        "mf_idio_beta_days": mf_idio_beta_days if mf_idio_on else None,
        "mf_idio_beta_on": mf_idio_beta_on if mf_idio_on else None,
        "n_si_block": int(n_si_block),
        "n_pe_block": int(n_pe_block),
        "n_tod_z_block": int(n_tod_z_block),
        "n_confirm_block": int(n_confirm_block),
        "entry_confirm_bars": int(confirm_bars_n) if confirm_bars_n > 0 else None,
        "entry_confirm_mode": confirm_mode if confirm_bars_n > 0 else None,
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
        "n_quote_1s": int(n_quote_1s),
        "n_quote_day_iv": int(n_quote_day_iv),
        "clear_otm_ban_0dte_pct": clear_otm_thresh,
        "reentry_mode": str(trade.get("reentry_mode") or "").strip() or None,
        "only_win_reenter": bool(only_win),
        "position_frac": pos,
        "position_sizing": sizing_mode_seen or str(trade.get("position_sizing") or "concurrent"),
        "max_concurrent_positions": int(trade.get("max_concurrent_positions") or n_sym),
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
    }
    return {"summary": summary, "trades": trades_df, "daily": daily_df, "topk": top2}

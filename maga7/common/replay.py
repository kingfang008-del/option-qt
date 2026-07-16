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
from maga7.common.position_size import resolve_size_frac
from maga7.common.reentry import resolve_only_win_reenter
from maga7.common.signals import (
    all_rule_a_times,
    attach_mf_features,
    build_topk_signals,
    count_peer_align,
    price_efficiency_ok,
    sync_index,
    load_stock_month_files,
)

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
    """
    if path is None or path.empty:
        return None
    entry_ts = to_ny(entry_ts)
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
    end_ts = entry_ts + pd.Timedelta(minutes=hold_minutes)
    tp_lvl, sl_lvl = entry * tp_mult, entry * sl_mult
    reason, exit_px, exit_ts = "T+30", float(sell_px[-1]), ts_list[-1]
    mode = str(exit_mode or "none").strip().lower()
    use_mf = mode in ("mf_flip", "mf_reversal", "streak_break") and direction in ("UP", "DN")
    use_trail = mode in ("mtm_trail", "trail")
    use_floor = mode in ("mtm_floor", "mtm_defend")
    use_flow = mode in ("flow_die", "cum_fav", "flow_mtm", "flow_soft") and direction in ("UP", "DN")
    require_mtm = mode in ("flow_mtm", "flow_soft")
    act = float(trail_activate) if trail_activate is not None else 0.20
    dd = float(trail_dd) if trail_dd is not None else 0.15
    floor = float(mtm_floor_ret) if mtm_floor_ret is not None else 0.0
    flow_floor = float(flow_cum_floor) if flow_cum_floor is not None else 0.0
    peak_ret = -np.inf
    trail_armed = False
    # mf_reversal / mtm_floor: wait ~10m; flow_*: wait ~5m before soft exit.
    min_hold_m = exit_min_hold_minutes
    if min_hold_m is None and mode in ("mf_reversal", "mtm_floor", "mtm_defend"):
        min_hold_m = 10.0
    if min_hold_m is None and mode in ("flow_die", "cum_fav", "flow_mtm", "flow_soft"):
        min_hold_m = 5.0
    grace_secs = int(exit_mf_grace_seconds)
    if min_hold_m is not None:
        grace_secs = max(grace_secs, int(float(min_hold_m) * 60))
    grace_until = entry_ts + pd.Timedelta(seconds=grace_secs)
    day = stock_day
    if (use_mf or use_flow) and day is not None and not day.empty:
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
        if t >= end_ts:
            reason, exit_px, exit_ts = "T+30", float(p), t
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
            if mode in ("mf_flip", "mf_reversal"):
                if direction == "UP" and mf < 0:
                    reason, exit_px, exit_ts = "MF_FLIP", float(p), t
                    break
                if direction == "DN" and mf > 0:
                    reason, exit_px, exit_ts = "MF_FLIP", float(p), t
                    break
            elif mode == "streak_break":
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
    money = str(trade.get("moneyness", "ATM"))

    if stock_by is None:
        stock_by = {}
        for sym in symbols:
            raw = load_stock_month_files(paths["stock_root"], sym, months)
            if raw.empty:
                continue
            # clip date range
            raw = raw[(raw["date"] >= start) & (raw["date"] <= end)]
            stock_by[sym] = attach_mf_features(
                raw,
                mf_window=int(sig_cfg.get("mf_window", 10)),
                vol_ma_window=int(sig_cfg.get("vol_ma_window", 20)),
                mf_confirm_bars=int(sig_cfg.get("mf_confirm_bars", 3)),
            )

    top2 = build_topk_signals(stock_by, sig_cfg)
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
    n_peer_block = 0
    n_si_block = 0
    n_pe_block = 0
    peer_align_min = sig_cfg.get("peer_align_min")
    peer_align_min_i = int(peer_align_min) if peer_align_min is not None else None
    peer_align_mode = str(sig_cfg.get("peer_align_mode", "mf10")).strip().lower()
    peer_symbols = list(sig_cfg.get("peer_symbols") or profile.get("symbols") or [])
    si_min = sig_cfg.get("si_min")
    si_min_f = float(si_min) if si_min is not None else None
    pe_min_ratio = sig_cfg.get("pe_min_ratio")
    pe_min_ratio_f = float(pe_min_ratio) if pe_min_ratio is not None else None
    pe_window = int(sig_cfg.get("pe_window", 10))
    pe_lookback = int(sig_cfg.get("pe_lookback_bars", 780))

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
    circuit = trade.get("day_circuit", None)
    only_win = resolve_only_win_reenter(trade)
    use_reentry = scheme.startswith("m5")
    use_circuit = "circuit" in scheme
    if not use_circuit:
        circuit = None
    n_size_full = 0
    n_size_split = 0
    n_skip_max_concurrent = 0
    sizing_mode_seen = None

    reg_cfg = profile.get("regime") or {}
    # After N consecutive losing days, skip the next session (flat day resets streak).
    day_loss_halt = reg_cfg.get("day_loss_streak_halt")
    day_loss_halt_n = int(day_loss_halt) if day_loss_halt is not None else None

    trades: list[dict[str, Any]] = []
    eq = 100.0
    peak = 100.0
    maxdd = 0.0
    daily_rows = []
    n_day_halt = 0
    loss_streak = 0
    bar_delay_seconds = int(trade.get("bar_availability_delay_seconds", 0) or 0)
    bar_delay = pd.Timedelta(seconds=bar_delay_seconds)

    for date, day_sigs in top2.groupby("date", sort=True):
        syms = list(day_sigs.sort_values("sig_ts")["symbol"].unique())
        n_sym = max(int(sig_cfg.get("top_k", 2)), 1)
        day_start = eq
        halt = False
        skip_day = bool(day_loss_halt_n is not None and loss_streak >= day_loss_halt_n)
        if skip_day:
            n_day_halt += 1

        if not use_reentry:
            events = [
                (to_ny(r.sig_ts) + bar_delay, to_ny(r.sig_ts), r.symbol, r.dir)
                for r in day_sigs.itertuples(index=False)
            ]
        else:
            events = []
            for r in day_sigs.itertuples(index=False):
                for ts in all_rule_a_times(
                    stock_by[r.symbol][stock_by[r.symbol]["date"] == date],
                    r.dir,
                    window_start=str(sig_cfg.get("window_start", "10:30")),
                    window_end=str(sig_cfg.get("window_end", "14:00")),
                    streak_min=int(sig_cfg.get("streak_min", 8)),
                    from_prev_abs=float(sig_cfg.get("from_prev_abs", 0.02)),
                    vol_z_min=float(sig_cfg.get("vol_z_min", 1.0)),
                ):
                    feature_ts = to_ny(ts)
                    events.append((feature_ts + bar_delay, feature_ts, r.symbol, r.dir))
            events.sort(key=lambda x: x[0])

        if skip_day:
            events = []

        last_exit = {s: None for s in syms}
        last_win = {s: True for s in syms}
        n_done = {s: 0 for s in syms}
        open_until = {s: None for s in syms}

        for ts, feature_ts, sym, direction in events:
            if halt:
                break
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
                fill=fill,
                tp_mult=float(trade.get("tp_mult", 1.6)),
                sl_mult=float(trade.get("sl_mult", 0.4)),
                hold_minutes=int(trade.get("hold_minutes", 30)),
                direction=direction,
                stock_day=stock_day,
                exit_mode=exit_mode,
                exit_mf_grace_seconds=int(trade.get("exit_mf_grace_seconds", 60)),
                exit_min_hold_minutes=trade.get("exit_min_hold_minutes"),
                mtm_floor_ret=trade.get("mtm_floor_ret"),
                flow_cum_floor=trade.get("flow_cum_floor"),
                stock_bar_delay_seconds=bar_delay_seconds,
                trail_activate=trade.get("trail_activate"),
                trail_dd=trade.get("trail_dd"),
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
            if not allow:
                n_skip_max_concurrent += 1
                continue
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
            if pe_ma_val is not None:
                row["pe_ma"] = float(pe_ma_val)
            trades.append(row)
            if circuit is not None and (eq / day_start - 1.0) <= float(circuit):
                halt = True

        day_ret = eq / day_start - 1.0
        daily_rows.append(
            {
                "date": date,
                "equity": eq,
                "day_ret": day_ret,
                "n": sum(n_done.values()),
                "day_halt": bool(skip_day),
            }
        )
        if day_ret < 0:
            loss_streak += 1
        else:
            loss_streak = 0

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
        "n_peer_block": int(n_peer_block),
        "n_si_block": int(n_si_block),
        "n_pe_block": int(n_pe_block),
        "peer_align_min": peer_align_min_i,
        "peer_align_mode": peer_align_mode if peer_align_min_i is not None else None,
        "si_min": si_min_f,
        "pe_min_ratio": pe_min_ratio_f,
        "n_day_halt": int(n_day_halt),
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
        "exit_mode": str(trade.get("exit_mode") or trade.get("stock_exit") or "none"),
    }
    return {"summary": summary, "trades": trades_df, "daily": daily_df, "topk": top2}

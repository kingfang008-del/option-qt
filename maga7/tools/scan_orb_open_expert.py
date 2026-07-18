#!/usr/bin/env python3
"""Scan opening washout + ORB fractal-high break; estimate open_ladder option PnL.

Research-only. Does **not** mutate freeze. Default expert stays off.

Outputs under ``--out``:
  - signals.csv / trades.csv
  - scoreboard.csv / summary.json
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from maga7.common.config import load_profile
from maga7.common.fills import FillSpec
from maga7.common.open_lock import load_multidte_lock_index, resolve_open_lock_contract, resolve_otm_rungs
from maga7.common.orb_open import OrbOpenConfig, OrbSignal, scan_orb_day
from maga7.common.replay import load_quotes, month_list, path_for_ticker, simulate_trade, to_ny
from maga7.common.signals import attach_mf_features, load_stock_month_files


def _spot_at(sdf: pd.DataFrame | None, asof_ts) -> float | None:
    if sdf is None or sdf.empty:
        return None
    asof = to_ny(asof_ts)
    ts = pd.to_datetime(sdf["timestamp"])
    if getattr(ts.dt, "tz", None) is None:
        ts = ts.dt.tz_localize("UTC").dt.tz_convert("America/New_York")
    else:
        ts = ts.dt.tz_convert("America/New_York")
    upto = sdf.loc[ts <= asof]
    if upto.empty:
        return None
    px = float(upto.iloc[-1]["close"])
    return px if np.isfinite(px) and px > 0 else None


def _under_ret(sdf: pd.DataFrame | None, entry_ts, *, hold_minutes: int, direction: str) -> float | None:
    if sdf is None or sdf.empty:
        return None
    entry_ts = to_ny(entry_ts)
    end_ts = entry_ts + pd.Timedelta(minutes=int(hold_minutes))
    ts = pd.to_datetime(sdf["timestamp"])
    if getattr(ts.dt, "tz", None) is None:
        ts = ts.dt.tz_localize("UTC").dt.tz_convert("America/New_York")
    else:
        ts = ts.dt.tz_convert("America/New_York")
    day = sdf.assign(_ts=ts)
    e = day[day["_ts"] <= entry_ts].tail(1)
    x = day[day["_ts"] <= end_ts].tail(1)
    if e.empty or x.empty:
        return None
    p0, p1 = float(e.iloc[0]["close"]), float(x.iloc[0]["close"])
    if not (np.isfinite(p0) and np.isfinite(p1) and p0 > 0):
        return None
    r = p1 / p0 - 1.0
    return float(r if direction == "UP" else -r)


def _portfolio_day(
    day_trades: list[dict],
    *,
    position_frac: float,
    max_concurrent: int,
    cooldown_minutes: int,
) -> list[dict]:
    """Greedy by entry_ts; size = position_frac / active (concurrent mode)."""
    if not day_trades:
        return []
    rows = sorted(day_trades, key=lambda r: (r["entry_ts"], r["symbol"]))
    open_pos: list[tuple[pd.Timestamp, str]] = []  # (exit_ts, symbol)
    last_exit: dict[str, pd.Timestamp] = {}
    out: list[dict] = []
    for tr in rows:
        et = to_ny(tr["entry_ts"])
        xt = to_ny(tr["exit_ts"])
        sym = str(tr["symbol"])
        open_pos = [(x, s) for x, s in open_pos if x > et]
        if any(s == sym for _, s in open_pos):
            continue
        if sym in last_exit and (et - last_exit[sym]).total_seconds() < cooldown_minutes * 60:
            continue
        if len(open_pos) >= int(max_concurrent):
            continue
        n_active = len(open_pos) + 1
        size = float(position_frac) / float(n_active)
        row = dict(tr)
        row["size"] = size
        row["pnl_frac"] = float(tr["ret"]) * size
        out.append(row)
        open_pos.append((xt, sym))
        last_exit[sym] = xt
    return out


def _window_stats(tr: pd.DataFrame, start: str, end: str) -> dict:
    sub = tr[(tr["date"] >= start) & (tr["date"] <= end)].copy()
    if sub.empty:
        return {
            "start": start,
            "end": end,
            "n_trades": 0,
            "n_days": 0,
            "sum_pnl": 0.0,
            "cum_ret": 0.0,
            "win_rate": None,
            "avg_ret": None,
            "max_dd": None,
        }
    daily = sub.groupby("date", as_index=False)["pnl_frac"].sum().sort_values("date")
    equity = daily["pnl_frac"].cumsum()
    peak = equity.cummax()
    dd = equity - peak
    return {
        "start": start,
        "end": end,
        "n_trades": int(len(sub)),
        "n_days": int(daily["date"].nunique()),
        "sum_pnl": float(sub["pnl_frac"].sum()),
        "cum_ret": float(equity.iloc[-1]) if len(equity) else 0.0,
        "win_rate": float((sub["ret"] > 0).mean()),
        "avg_ret": float(sub["ret"].mean()),
        "max_dd": float(dd.min()) if len(dd) else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--profile",
        default="maga7/CONFIG/strategy_profiles/single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1.json",
    )
    ap.add_argument("--start-date", default="2025-07-01")
    ap.add_argument("--end-date", default="2026-07-17")
    ap.add_argument("--wash-drop-min", type=float, default=0.003)
    ap.add_argument("--wash-window-end", default="10:00")
    ap.add_argument("--signal-deadline", default="10:00")
    ap.add_argument("--selloff-min-bars", type=int, default=3)
    ap.add_argument("--hold-confirm-bars", type=int, default=0)
    ap.add_argument("--top-k-day", type=int, default=2, help="Max ORB entries per day (portfolio)")
    ap.add_argument(
        "--out",
        default="maga7/results/orb_open_expert/scan_2025-07_2026-07-17",
    )
    args = ap.parse_args()

    prof = load_profile(args.profile)
    paths = prof["_paths"]
    symbols = list(prof["symbols"])
    trade = prof.get("trade") or {}
    fill_cfg = prof.get("fill") or {}
    months = month_list(args.start_date, args.end_date)
    cfg = OrbOpenConfig(
        wash_window_end=args.wash_window_end,
        wash_drop_min=float(args.wash_drop_min),
        selloff_min_bars=int(args.selloff_min_bars),
        hold_confirm_bars=int(args.hold_confirm_bars),
        signal_deadline=args.signal_deadline,
        only_up=True,
    )

    print(f"loading stock {args.start_date}..{args.end_date} symbols={symbols}", flush=True)
    stock_by: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        raw = load_stock_month_files(paths["stock_root"], sym, months)
        if raw.empty:
            print(f"  warn: empty stock for {sym}", flush=True)
            continue
        raw = raw[(raw["date"] >= args.start_date) & (raw["date"] <= args.end_date)]
        stock_by[sym] = attach_mf_features(
            raw,
            mf_window=int(prof["signal"].get("mf_window", 10)),
            vol_ma_window=int(prof["signal"].get("vol_ma_window", 20)),
        )

    dates = sorted(
        {
            d
            for sdf in stock_by.values()
            for d in sdf["date"].astype(str).unique().tolist()
            if args.start_date <= d <= args.end_date
        }
    )
    print(f"scanning ORB on {len(dates)} sessions …", flush=True)

    signals: list[OrbSignal] = []
    for date in dates:
        signals.extend(scan_orb_day(stock_by, date=date, symbols=symbols, cfg=cfg))
    print(f"ORB fires: {len(signals)}", flush=True)

    lock_path = Path(paths.get("open_locked_map") or paths.get("locked_map")).expanduser()
    multi_idx = load_multidte_lock_index(lock_path) if lock_path.is_file() else {}
    quote_root = Path(paths["quote_1s_root"]).expanduser()
    otm_rungs = resolve_otm_rungs(prof, default=5)
    prefer_dte = int((prof.get("lock") or {}).get("prefer_dte", 0))
    allowed_dte = list((prof.get("lock") or {}).get("allowed_dte") or [0, 1, 2])
    clear_otm = float(trade.get("clear_otm_ban_0dte_pct", 0.01) or 0.01)
    fill = FillSpec(
        entry_frac=float(fill_cfg.get("entry_frac", 0.8)),
        exit_frac=float(fill_cfg.get("exit_frac", 0.8)),
    )
    bar_delay = int(trade.get("bar_availability_delay_seconds", 60) or 0)
    hold_minutes = int(trade.get("hold_minutes", 30))
    exit_mode = trade.get("exit_mode")
    hold_extend_minutes = trade.get("hold_extend_minutes")
    hold_extend_mtm_min = trade.get("hold_extend_mtm_min")
    hold_extend_require_mf = bool(trade.get("hold_extend_require_mf", False))
    tp_mult = float(trade.get("tp_mult", 1.6))
    sl_mult = float(trade.get("sl_mult", 0.4))
    pos_frac = float(trade.get("position_frac", 0.2))
    max_conc = min(int(trade.get("max_concurrent_positions", 2)), int(args.top_k_day))
    cooldown = int(trade.get("cooldown_minutes", 5))

    quote_cache: dict[tuple[str, str], pd.DataFrame | None] = {}
    raw_trades: list[dict] = []
    n_opt = n_under = n_miss = 0

    for i, sig in enumerate(signals):
        if (i + 1) % 50 == 0:
            print(f"  priced {i + 1}/{len(signals)}", flush=True)
        sdf = stock_by.get(sig.symbol)
        entry_ts = to_ny(sig.sig_ts) + pd.Timedelta(seconds=bar_delay)
        spot = _spot_at(sdf, entry_ts)
        by_dte = multi_idx.get((sig.symbol, sig.date))
        ticker, dte, src = resolve_open_lock_contract(
            by_dte,
            direction=sig.direction,
            moneyness="ATM",
            spot=spot,
            prefer_dte=prefer_dte,
            allowed_dte=allowed_dte,
            clear_otm_thresh=clear_otm,
            ladder=True,
            otm_rungs=otm_rungs,
        )
        ret = None
        reason = "NO_PATH"
        exit_ts = entry_ts + pd.Timedelta(minutes=hold_minutes)
        entry_px = exit_px = np.nan
        path_src = "none"
        if ticker:
            qkey = (sig.symbol, sig.date)
            if qkey not in quote_cache:
                quote_cache[qkey] = load_quotes(quote_root, sig.symbol, sig.date)
            path = path_for_ticker(quote_cache[qkey], ticker)
            stock_day = None
            if sdf is not None:
                stock_day = sdf[sdf["date"].astype(str) == sig.date]
            sim = simulate_trade(
                path,
                entry_ts,
                fill=fill,
                tp_mult=tp_mult,
                sl_mult=sl_mult,
                hold_minutes=hold_minutes,
                direction=sig.direction,
                stock_day=stock_day,
                exit_mode=exit_mode,
                hold_extend_minutes=int(hold_extend_minutes) if hold_extend_minutes is not None else None,
                hold_extend_mtm_min=float(hold_extend_mtm_min)
                if hold_extend_mtm_min is not None
                else None,
                hold_extend_require_mf=hold_extend_require_mf,
                stock_bar_delay_seconds=bar_delay,
            )
            if sim is not None:
                ret = float(sim.ret)
                reason = str(sim.reason)
                exit_ts = sim.exit_ts
                entry_px = float(sim.entry)
                exit_px = float(sim.exit)
                path_src = "option"
                n_opt += 1
        if ret is None:
            ur = _under_ret(sdf, entry_ts, hold_minutes=hold_minutes, direction=sig.direction)
            if ur is None:
                n_miss += 1
                continue
            ret = float(ur)
            reason = "UNDER_PROXY"
            path_src = "underlying"
            n_under += 1

        raw_trades.append(
            {
                "date": sig.date,
                "symbol": sig.symbol,
                "dir": sig.direction,
                "sig_ts": str(to_ny(sig.sig_ts)),
                "entry_ts": entry_ts,
                "exit_ts": exit_ts,
                "ticker": ticker,
                "dte": dte,
                "lock_source": src if ticker else None,
                "path_src": path_src,
                "open_px": sig.open_px,
                "wash_low": sig.wash_low,
                "wash_drop": sig.wash_drop,
                "fractal_high": sig.fractal_high,
                "break_px": sig.break_px,
                "entry": entry_px,
                "exit": exit_px,
                "ret": ret,
                "reason": reason,
                "orb_reason": sig.reason,
            }
        )

    # portfolio per day (top-k / concurrent)
    by_date: dict[str, list[dict]] = {}
    for tr in raw_trades:
        by_date.setdefault(str(tr["date"]), []).append(tr)
    port: list[dict] = []
    for date in sorted(by_date):
        # earliest first; cap candidates before portfolio greed
        cands = sorted(by_date[date], key=lambda r: (r["entry_ts"], r["symbol"]))[: max(max_conc * 3, max_conc)]
        port.extend(
            _portfolio_day(
                cands,
                position_frac=pos_frac,
                max_concurrent=max_conc,
                cooldown_minutes=cooldown,
            )
        )

    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    sig_rows = [
        {
            "date": s.date,
            "symbol": s.symbol,
            "direction": s.direction,
            "sig_ts": str(to_ny(s.sig_ts)),
            "open_px": s.open_px,
            "wash_low": s.wash_low,
            "wash_drop": s.wash_drop,
            "fractal_high": s.fractal_high,
            "selloff_end_ts": str(to_ny(s.selloff_end_ts)),
            "break_px": s.break_px,
            "reason": s.reason,
        }
        for s in signals
    ]
    sig_df = pd.DataFrame(sig_rows)
    sig_df.to_csv(out_dir / "signals.csv", index=False)

    tr_df = pd.DataFrame(port)
    if not tr_df.empty:
        tr_df["entry_ts"] = tr_df["entry_ts"].map(lambda x: str(to_ny(x)))
        tr_df["exit_ts"] = tr_df["exit_ts"].map(lambda x: str(to_ny(x)))
        tr_df["date"] = tr_df["date"].astype(str)
    tr_df.to_csv(out_dir / "trades.csv", index=False)

    # also dump all priced signals (pre-portfolio) for research
    raw_df = pd.DataFrame(raw_trades)
    if not raw_df.empty:
        raw_df["entry_ts"] = raw_df["entry_ts"].map(lambda x: str(to_ny(x)))
        raw_df["exit_ts"] = raw_df["exit_ts"].map(lambda x: str(to_ny(x)))
    raw_df.to_csv(out_dir / "trades_raw.csv", index=False)

    windows = [
        ("full", args.start_date, args.end_date),
        ("feb_apr", "2026-02-01", "2026-04-30"),
        ("may_jul", "2026-05-01", "2026-07-17"),
    ]
    board = [_window_stats(tr_df if not tr_df.empty else pd.DataFrame(columns=["date", "pnl_frac", "ret"]), s, e) for name, s, e in windows]
    for b, (name, _, _) in zip(board, windows):
        b["window"] = name
    board_df = pd.DataFrame(board)
    board_df.to_csv(out_dir / "scoreboard.csv", index=False)

    summary = {
        "profile": args.profile,
        "orb_cfg": asdict(cfg),
        "start_date": args.start_date,
        "end_date": args.end_date,
        "n_sessions": len(dates),
        "n_signals": len(signals),
        "n_priced_option": n_opt,
        "n_priced_underlying": n_under,
        "n_miss": n_miss,
        "n_portfolio_trades": int(len(tr_df)),
        "position_frac": pos_frac,
        "max_concurrent": max_conc,
        "bar_delay_seconds": bar_delay,
        "exit_mode": exit_mode,
        "scoreboard": board,
        "note": "ORB expert research scan; freeze untouched. OFI tick expert deferred (no aggressor tape).",
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()

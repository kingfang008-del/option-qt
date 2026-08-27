#!/usr/bin/env python3
"""Morning sec-MF: enter on window signal, exit on window trend reverse (not fixed hold).

Exit modes (second clock, same mf_window_sec as entry):
  - mf_flip: mf sign flips against entry direction after min_hold
  - streak_break: entry-dir streak resets to 0 while mf opposing
  - either: whichever comes first

Still prices open_ladder ATM with 1s quotes. Research only.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from maga7.common.bar_agg import load_stock_1s_day
from maga7.common.config import load_profile
from maga7.common.fills import FillSpec
from maga7.common.open_lock import load_multidte_lock_index, resolve_open_lock_contract, resolve_otm_rungs
from maga7.common.replay import load_quotes, month_list, path_for_ticker, simulate_trade, to_ny
from maga7.common.sec_mf import attach_sec_mf_features
from maga7.common.signals import attach_mf_features, load_stock_month_files
from maga7.tools.run_morning_sec_option_fill import (
    DEFAULT_CANDIDATES,
    _equity_stats,
    _filter_events,
    _portfolio_day,
    _spot_at,
)
from maga7.tools.scan_morning_sec_edge import _prior_close, _bdates

FREEZE = (
    ROOT
    / "maga7"
    / "CONFIG"
    / "strategy_profiles"
    / "single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1.json"
)


def _resolve_trend_exit(
    feat: pd.DataFrame,
    *,
    entry_ts: pd.Timestamp,
    direction: str,
    mode: str,
    min_hold_sec: int,
    max_hold_sec: int,
) -> tuple[pd.Timestamp | None, str]:
    """Return (exit_ts, reason) on second-level trend reverse."""
    if feat is None or feat.empty:
        return None, "NO_FEAT"
    entry_ts = to_ny(entry_ts)
    ts = pd.DatetimeIndex(pd.to_datetime(feat["timestamp"]))
    if ts.tz is None:
        ts = ts.tz_localize("America/New_York")
    else:
        ts = ts.tz_convert("America/New_York")
    mf = feat["mf"].to_numpy(dtype=np.float64)
    su = feat["streak_up"].to_numpy(dtype=np.int32)
    sd = feat["streak_dn"].to_numpy(dtype=np.int32)

    start = entry_ts + pd.Timedelta(seconds=int(min_hold_sec))
    hard = entry_ts + pd.Timedelta(seconds=int(max_hold_sec))
    # search from first bar >= start
    i0 = int(ts.searchsorted(start, side="left"))
    i_hard = int(ts.searchsorted(hard, side="left"))
    if i0 >= len(ts):
        return None, "NO_BARS"
    i_end = min(len(ts) - 1, max(i0, i_hard))

    for i in range(i0, i_end + 1):
        v = mf[i]
        if not np.isfinite(v):
            continue
        flip = (direction == "UP" and v < 0) or (direction == "DN" and v > 0)
        streak0 = (direction == "UP" and su[i] == 0) or (direction == "DN" and sd[i] == 0)
        if mode == "mf_flip" and flip:
            return ts[i], "SEC_MF_FLIP"
        if mode == "streak_break" and streak0 and flip:
            return ts[i], "SEC_STREAK_BREAK"
        if mode == "either":
            if flip:
                return ts[i], "SEC_MF_FLIP"
            if streak0 and ((direction == "UP" and v <= 0) or (direction == "DN" and v >= 0)):
                return ts[i], "SEC_STREAK_BREAK"
    # hit max hold
    if i_hard < len(ts):
        return ts[i_hard], "SEC_MAX_HOLD"
    return ts[i_end], "SEC_MAX_HOLD"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", default=str(FREEZE))
    ap.add_argument("--events-tag", default="research_morn_sec_edge_feb_jul")
    ap.add_argument("--tag", default="research_morn_sec_trend_exit_feb_jul")
    ap.add_argument("--candidate", default="w100_s20_h180_fp005_vz1_p0")
    ap.add_argument("--modes", default="mf_flip,streak_break,either")
    ap.add_argument("--min-hold-sec", default="15,30,60")
    ap.add_argument("--max-hold-sec", default="300,600")
    ap.add_argument("--position-frac", type=float, default=0.10)
    ap.add_argument("--max-concurrent", type=int, default=2)
    ap.add_argument("--cooldown-minutes", type=int, default=5)
    ap.add_argument("--toxic", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args()

    prof = load_profile(args.profile)
    paths = prof["_paths"]
    trade = prof.get("trade") or {}
    fill_cfg = prof.get("fill") or {}
    results_dir = Path(paths["results_dir"])
    events_path = results_dir / args.events_tag / "events.parquet"
    events = pd.read_parquet(events_path) if events_path.is_file() else pd.read_csv(events_path.with_suffix(".csv"))

    cand = next((c for c in DEFAULT_CANDIDATES if c["name"] == args.candidate), None)
    if cand is None:
        raise SystemExit(f"unknown candidate {args.candidate}")
    sigs = _filter_events(events, cand)
    print(f"candidate={cand['name']} signals={len(sigs)} W={cand['mf_window_sec']}", flush=True)

    lock_path = Path(paths.get("open_locked_map") or paths.get("locked_map")).expanduser()
    multi_idx = load_multidte_lock_index(lock_path) if lock_path.is_file() else {}
    quote_root = Path(paths["quote_1s_root"]).expanduser()
    stock_root = Path(paths["stock_root"]).expanduser()
    stock_1s_root = Path(paths["stock_1s_root"]).expanduser()
    otm_rungs = resolve_otm_rungs(prof, default=3)
    prefer_dte = int((prof.get("lock") or {}).get("prefer_dte", 0))
    allowed_dte = list((prof.get("lock") or {}).get("allowed_dte") or [0, 1, 2])
    clear_otm = float(trade.get("clear_otm_ban_0dte_pct", 0.01) or 0.01)
    fill = FillSpec(
        entry_frac=float(fill_cfg.get("entry_frac", 0.75)),
        exit_frac=float(fill_cfg.get("exit_frac", 0.75)),
    )
    tp = float(trade.get("tp_mult", 1.6))
    sl = float(trade.get("sl_mult", 0.45))
    toxic = (trade.get("trade_toxic") or {}) if args.toxic else {"enabled": False}

    dates_all = _bdates(str(sigs["date"].min()), str(sigs["date"].max()))
    months = month_list(str(sigs["date"].min()), str(sigs["date"].max()))
    symbols = sorted(sigs["symbol"].astype(str).unique())
    stock_by: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        raw = load_stock_month_files(stock_root, sym, months)
        if not raw.empty:
            stock_by[sym] = attach_mf_features(raw)

    # cache sec features per (sym, date)
    feat_cache: dict[tuple[str, str], pd.DataFrame] = {}
    W = int(cand["mf_window_sec"])

    def get_feat(sym: str, date: str) -> pd.DataFrame:
        key = (sym, date)
        if key in feat_cache:
            return feat_cache[key]
        day = load_stock_1s_day(stock_1s_root, sym, date)
        if day.empty:
            feat_cache[key] = pd.DataFrame()
            return feat_cache[key]
        # morning + buffer for exits into late morning
        ts = pd.to_datetime(day["timestamp"])
        if getattr(ts.dt, "tz", None) is None:
            ts = ts.dt.tz_localize("America/New_York")
        else:
            ts = ts.dt.tz_convert("America/New_York")
        day = day.copy()
        day["timestamp"] = ts
        t = day["timestamp"].dt.time
        buf = day[(t >= pd.Timestamp("09:30").time()) & (t < pd.Timestamp("11:30").time())]
        prev = _prior_close(stock_1s_root, sym, date, dates_all)
        feat_cache[key] = attach_sec_mf_features(
            buf, mf_window_sec=W, vol_ma_sec=max(300, W * 2), prev_close=prev
        )
        return feat_cache[key]

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    min_holds = [int(x) for x in args.min_hold_sec.split(",") if x.strip()]
    max_holds = [int(x) for x in args.max_hold_sec.split(",") if x.strip()]

    out_root = results_dir / args.tag
    out_root.mkdir(parents=True, exist_ok=True)
    quote_cache: dict[tuple[str, str], pd.DataFrame | None] = {}
    score_rows: list[dict[str, Any]] = []

    for mode in modes:
        for min_h in min_holds:
            for max_h in max_holds:
                if min_h >= max_h:
                    continue
                name = f"{cand['name']}_{mode}_min{min_h}_max{max_h}"
                print(f"=== {name} ===", flush=True)
                raw_trades: list[dict] = []
                n_miss = 0
                hold_secs: list[float] = []
                exit_reasons: dict[str, int] = {}
                for _, row in sigs.iterrows():
                    sym = str(row["symbol"])
                    date = str(row["date"])
                    direction = str(row["dir"])
                    entry_ts = to_ny(row["ts"])
                    feat = get_feat(sym, date)
                    exit_ts, exit_why = _resolve_trend_exit(
                        feat,
                        entry_ts=entry_ts,
                        direction=direction,
                        mode=mode,
                        min_hold_sec=min_h,
                        max_hold_sec=max_h,
                    )
                    if exit_ts is None:
                        n_miss += 1
                        continue
                    sdf = stock_by.get(sym)
                    spot = _spot_at(sdf, entry_ts)
                    ticker, dte, src = resolve_open_lock_contract(
                        multi_idx.get((sym, date)),
                        direction=direction,
                        moneyness="ATM",
                        spot=spot,
                        prefer_dte=prefer_dte,
                        allowed_dte=allowed_dte,
                        clear_otm_thresh=clear_otm,
                        ladder=True,
                        otm_rungs=otm_rungs,
                    )
                    if not ticker:
                        n_miss += 1
                        continue
                    qkey = (sym, date)
                    if qkey not in quote_cache:
                        quote_cache[qkey] = load_quotes(quote_root, sym, date)
                    path = path_for_ticker(quote_cache[qkey], ticker)
                    stock_day = sdf[sdf["date"].astype(str) == date] if sdf is not None else None
                    hold_minutes = max(1, int(np.ceil(max_h / 60.0)))
                    sim = simulate_trade(
                        path,
                        entry_ts,
                        fill=fill,
                        tp_mult=tp,
                        sl_mult=sl,
                        hold_minutes=hold_minutes,
                        direction=direction,
                        stock_day=stock_day,
                        exit_mode=None,
                        force_exit_ts=exit_ts,
                        trade_toxic=toxic,
                        stock_bar_delay_seconds=0,
                    )
                    if sim is None:
                        n_miss += 1
                        continue
                    # prefer sim reason if TP/SL hit earlier; else annotate trend reason
                    reason = str(sim.reason)
                    if reason == "DISPLACE":
                        reason = exit_why
                    held = (to_ny(sim.exit_ts) - entry_ts).total_seconds()
                    hold_secs.append(held)
                    exit_reasons[reason] = exit_reasons.get(reason, 0) + 1
                    raw_trades.append(
                        {
                            "date": date,
                            "symbol": sym,
                            "dir": direction,
                            "entry_ts": entry_ts,
                            "exit_ts": sim.exit_ts,
                            "trend_exit_ts": exit_ts,
                            "ticker": ticker,
                            "dte": dte,
                            "ret": float(sim.ret),
                            "reason": reason,
                            "held_sec": held,
                            "entry": float(sim.entry),
                            "exit": float(sim.exit),
                        }
                    )

                by_day: dict[str, list[dict]] = {}
                for tr in raw_trades:
                    by_day.setdefault(str(tr["date"]), []).append(tr)
                sized: list[dict] = []
                for _, rows in sorted(by_day.items()):
                    sized.extend(
                        _portfolio_day(
                            rows,
                            position_frac=float(args.position_frac),
                            max_concurrent=int(args.max_concurrent),
                            cooldown_minutes=int(args.cooldown_minutes),
                        )
                    )
                trdf = pd.DataFrame(sized)
                stats = _equity_stats(trdf)
                row = {
                    "variant": name,
                    "mode": mode,
                    "min_hold_sec": min_h,
                    "max_hold_sec": max_h,
                    "n_signals": int(len(sigs)),
                    "n_fills": int(len(raw_trades)),
                    "n_miss": int(n_miss),
                    "held_sec_p50": float(np.median(hold_secs)) if hold_secs else None,
                    "held_sec_mean": float(np.mean(hold_secs)) if hold_secs else None,
                    "exit_reasons": exit_reasons,
                    **stats,
                }
                sub = out_root / name
                sub.mkdir(parents=True, exist_ok=True)
                if not trdf.empty:
                    trdf.to_csv(sub / "trades.csv", index=False)
                pd.DataFrame(raw_trades).to_csv(sub / "trades_raw.csv", index=False)
                (sub / "summary.json").write_text(
                    json.dumps(row, indent=2, default=str),
                    encoding="utf-8",
                )
                score_rows.append(row)
                print(
                    json.dumps(
                        {
                            k: row[k]
                            for k in (
                                "variant",
                                "n_fills",
                                "trade_win",
                                "exp",
                                "total_ret",
                                "maxdd",
                                "held_sec_p50",
                                "exit_reasons",
                            )
                        },
                        indent=2,
                        default=str,
                    ),
                    flush=True,
                )

    board = pd.DataFrame([{k: v for k, v in r.items() if k != "exit_reasons"} | {"exit_reasons": json.dumps(r.get("exit_reasons") or {})} for r in score_rows])
    board.to_csv(out_root / "scoreboard.csv", index=False)
    (out_root / "scoreboard.json").write_text(json.dumps(score_rows, indent=2, default=str), encoding="utf-8")
    print("\n=== trend-exit scoreboard ===")
    cols = ["variant", "n_fills", "trade_win", "exp", "total_ret", "maxdd", "held_sec_p50", "day_win"]
    show = board[[c for c in cols if c in board.columns]].sort_values("total_ret", ascending=False)
    print(show.to_string(index=False))
    print(f"wrote {out_root}")


if __name__ == "__main__":
    main()

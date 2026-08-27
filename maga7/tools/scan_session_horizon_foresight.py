#!/usr/bin/env python3
"""Foresight horizon scan for AM 09:30–10:00 and midday 12:30–13:30.

Uses ``/mnt/s990/new_option_data_s3_trades`` (1s last) because quote books often
lack early RTH. Entry direction is causal stock lookback on **1s closes**
(default ``stock_1s_root``); choice of hold length uses foresight (oracle best
exit inside each candidate H, plus clock exit at H).

Do **not** use left-labeled 1m closes for direction (up to ~59s look-ahead).

Example:
  PYTHONPATH=. python -m maga7.tools.scan_session_horizon_foresight \\
    --start-date 2026-05-01 --end-date 2026-07-22 \\
    --stock-source 1s --tag research_session_horizon_foresight_1s_may_jul
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
from maga7.common.open_lock import (
    load_multidte_lock_index,
    resolve_open_lock_contract,
    resolve_otm_rungs,
)
from maga7.common.option_trades import load_option_trades
from maga7.common.replay import month_list, to_ny
from maga7.common.signals import load_stock_month_files

NY = "America/New_York"
DEFAULT_TRADES = Path("/mnt/s990/new_option_data_s3_trades")

SESSIONS = (
    ("AM_0930_1000", "09:30", "10:00"),
    ("CORE_1030_1130", "10:30", "11:30"),
    ("MID_1230_1330", "12:30", "13:30"),
)


def _bdates(start: str, end: str) -> list[str]:
    return [d.strftime("%Y-%m-%d") for d in pd.bdate_range(start, end)]


def _ts_ns(ts: pd.Timestamp) -> int:
    t = to_ny(ts)
    return int(t.value)


def _stock_arrays(day: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    ts = pd.to_datetime(day["timestamp"], utc=True).astype("int64").to_numpy()
    px = day["close"].astype(float).to_numpy()
    order = np.argsort(ts)
    return ts[order], px[order]


def _spot_at_arr(ts_ns: np.ndarray, px: np.ndarray, t: pd.Timestamp) -> float | None:
    if len(ts_ns) == 0:
        return None
    i = int(np.searchsorted(ts_ns, _ts_ns(t), side="right") - 1)
    if i < 0:
        return None
    v = float(px[i])
    return v if np.isfinite(v) and v > 0 else None


def _stock_dir_arr(
    ts_ns: np.ndarray,
    px: np.ndarray,
    t: pd.Timestamp,
    lookback_sec: int,
    min_abs: float,
    *,
    max_stale_sec: float = 5.0,
) -> tuple[str | None, float]:
    """Causal direction: last print ≤ t vs last print ≤ t−lookback."""
    if len(ts_ns) < 2:
        return None, np.nan
    t_ns = _ts_ns(t)
    t0_ns = t_ns - int(lookback_sec) * 1_000_000_000
    i1 = int(np.searchsorted(ts_ns, t_ns, side="right") - 1)
    i0 = int(np.searchsorted(ts_ns, t0_ns, side="right") - 1)
    if i1 < 0 or i0 < 0:
        return None, np.nan
    stale = int(max_stale_sec * 1_000_000_000)
    if abs(int(ts_ns[i1]) - t_ns) > stale or abs(int(ts_ns[i0]) - t0_ns) > stale:
        return None, np.nan
    a, b = float(px[i0]), float(px[i1])
    if a <= 0 or b <= 0 or not np.isfinite(a) or not np.isfinite(b):
        return None, np.nan
    sr = b / a - 1.0
    if abs(sr) < float(min_abs):
        return None, sr
    return ("UP" if b > a else "DN"), sr


def _load_stock_1s_arrays(
    stock_1s_root: Path,
    symbols: list[str],
    dates: list[str],
) -> dict[str, pd.DataFrame]:
    """Per-symbol RTH 1s frames with date column (close used for direction)."""
    out: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        frames: list[pd.DataFrame] = []
        for date in dates:
            raw = load_stock_1s_day(stock_1s_root, sym, date)
            if raw.empty:
                continue
            d = raw.copy()
            d["date"] = date
            frames.append(d)
        if not frames:
            continue
        sdf = pd.concat(frames, ignore_index=True)
        sdf["timestamp"] = pd.to_datetime(sdf["timestamp"])
        if sdf["timestamp"].dt.tz is None:
            sdf["timestamp"] = sdf["timestamp"].dt.tz_localize(NY)
        else:
            sdf["timestamp"] = sdf["timestamp"].dt.tz_convert(NY)
        out[sym] = sdf
        print(f"  stock1s {sym} n={len(sdf)} days={sdf['date'].nunique()}", flush=True)
    return out


def _paths_by_ticker(tday: pd.DataFrame) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """ticker -> (ts_ns sorted, last)."""
    px_col = "c" if "c" in tday.columns else ("price" if "price" in tday.columns else None)
    if px_col is None:
        return {}
    tick = tday["ticker"].astype(str).str.replace("O:", "", regex=False)
    ts = pd.to_datetime(tday["timestamp"], utc=True).astype("int64").to_numpy()
    last = tday[px_col].astype(float).to_numpy()
    order = np.lexsort((ts, tick.to_numpy()))
    tick_s = tick.to_numpy()[order]
    ts = ts[order]
    last = last[order]
    # drop non-positive / non-finite, then last-per-(ticker, ts)
    ok = np.isfinite(last) & (last > 0)
    tick_s, ts, last = tick_s[ok], ts[ok], last[ok]
    if len(ts) == 0:
        return {}
    # keep last of consecutive (ticker, ts) duplicates
    keep = np.ones(len(ts), dtype=bool)
    keep[:-1] = ~((tick_s[:-1] == tick_s[1:]) & (ts[:-1] == ts[1:]))
    tick_s, ts, last = tick_s[keep], ts[keep], last[keep]
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    # split by ticker runs
    cuts = np.flatnonzero(tick_s[1:] != tick_s[:-1]) + 1
    starts = np.r_[0, cuts]
    ends = np.r_[cuts, len(tick_s)]
    for a, b in zip(starts, ends):
        out[str(tick_s[a])] = (ts[a:b], last[a:b])
    return out


def _fwd_trade_rets_arr(
    ts_ns: np.ndarray,
    last: np.ndarray,
    entry_ts: pd.Timestamp,
    horizons: list[int],
    *,
    slip: float = 0.01,
) -> list[dict[str, Any]]:
    """Clock + oracle rets on trade last path. Buy = last*(1+slip), sell=last*(1-slip)."""
    t0 = _ts_ns(entry_ts)
    i0 = int(np.searchsorted(ts_ns, t0, side="left"))
    if i0 >= len(ts_ns):
        return []
    lag = (ts_ns[i0] - t0) / 1e9
    if lag > 5:
        return []
    entry = float(last[i0]) * (1.0 + float(slip))
    if not np.isfinite(entry) or entry <= 0:
        return []
    out = []
    sell_mult = 1.0 - float(slip)
    for h in horizons:
        end_ns = ts_ns[i0] + int(h) * 1_000_000_000
        i1 = int(np.searchsorted(ts_ns, end_ns, side="right") - 1)
        if i1 < i0:
            continue
        win = last[i0 : i1 + 1]
        sells = win * sell_mult
        clock = float(sells[-1] / entry - 1.0)
        best_i = int(np.nanargmax(sells))
        oracle = float(sells[best_i] / entry - 1.0)
        hold = float((ts_ns[i0 + best_i] - ts_ns[i0]) / 1e9)
        mfe = float(np.nanmax(win) / entry - 1.0)
        mae = float(np.nanmin(win) / entry - 1.0)
        out.append(
            {
                "horizon_sec": int(h),
                "clock_ret": clock,
                "oracle_ret": oracle,
                "oracle_hold_sec": hold,
                "mfe": mfe,
                "mae": mae,
                "n_prints": int(i1 - i0 + 1),
                "entry_lag_sec": float(lag),
            }
        )
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--profile",
        default=(
            "maga7/CONFIG/strategy_profiles/"
            "single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1.json"
        ),
    )
    ap.add_argument("--start-date", required=True)
    ap.add_argument("--end-date", required=True)
    ap.add_argument("--tag", default="research_session_horizon_foresight")
    ap.add_argument("--trades-root", default=str(DEFAULT_TRADES))
    ap.add_argument("--stride-sec", type=int, default=120)
    ap.add_argument("--lookback-sec", type=int, default=60)
    ap.add_argument("--horizons", default="30,60,90,120,180,300,450,600,900")
    ap.add_argument("--prefer-dte", type=int, default=0)
    ap.add_argument("--allowed-dte", default="0,1,2")
    ap.add_argument("--slip", type=float, default=0.01)
    ap.add_argument("--min-abs-stock-ret", type=float, default=0.0005)
    ap.add_argument(
        "--stock-source",
        choices=["1s", "1m"],
        default="1s",
        help="1s = causal closes from stock_1s_root (default). 1m = LEGACY left-label (leaky).",
    )
    ap.add_argument(
        "--stock-1s-root",
        default="",
        help="Override paths.stock_1s_root (default /mnt/s990/data/raw_1s/stocks).",
    )
    ap.add_argument(
        "--sessions",
        default="",
        help="Comma subset of session names; empty = all SESSIONS.",
    )
    args = ap.parse_args(argv)

    prof = load_profile(args.profile)
    paths = prof["_paths"]
    symbols = list(prof.get("symbols") or [])
    horizons = [int(x) for x in args.horizons.split(",") if x.strip()]
    allowed_dte = [int(x) for x in args.allowed_dte.split(",") if x.strip()]
    trades_root = Path(args.trades_root)
    if str(args.sessions).strip():
        want = {x.strip() for x in str(args.sessions).split(",") if x.strip()}
        active_sessions = tuple(s for s in SESSIONS if s[0] in want)
        if not active_sessions:
            raise SystemExit(f"no sessions matched {want}; known={[s[0] for s in SESSIONS]}")
    else:
        active_sessions = SESSIONS

    dates = _bdates(args.start_date, args.end_date)
    stock_by: dict[str, pd.DataFrame] = {}
    if args.stock_source == "1s":
        stock_1s_root = Path(
            args.stock_1s_root
            or paths.get("stock_1s_root")
            or "/mnt/s990/data/raw_1s/stocks"
        ).expanduser()
        print(
            f"loading stock 1s (causal) {stock_1s_root} {args.start_date}..{args.end_date}",
            flush=True,
        )
        stock_by = _load_stock_1s_arrays(stock_1s_root, symbols, dates)
    else:
        months = month_list(args.start_date, args.end_date)
        print(
            f"WARNING: loading LEGACY 1m left-label stock {args.start_date}..{args.end_date}",
            flush=True,
        )
        for sym in symbols:
            raw = load_stock_month_files(Path(paths["stock_root"]).expanduser(), sym, months)
            if raw.empty:
                continue
            sdf = raw[(raw["date"] >= args.start_date) & (raw["date"] <= args.end_date)].copy()
            sdf["timestamp"] = pd.to_datetime(sdf["timestamp"])
            if sdf["timestamp"].dt.tz is None:
                sdf["timestamp"] = sdf["timestamp"].dt.tz_localize(NY)
            else:
                sdf["timestamp"] = sdf["timestamp"].dt.tz_convert(NY)
            stock_by[sym] = sdf
            print(f"  stock1m {sym} n={len(sdf)}", flush=True)

    lock_path = Path(paths.get("open_locked_map") or paths.get("locked_map")).expanduser()
    multi_idx = load_multidte_lock_index(lock_path) if lock_path.is_file() else {}
    otm_rungs = resolve_otm_rungs(prof, default=3)

    rows: list[dict[str, Any]] = []
    n_miss_trades = n_miss_lock = n_miss_dir = 0
    stride = pd.Timedelta(seconds=int(args.stride_sec))
    lb = max(int(args.lookback_sec), int(args.stride_sec))

    for di, date in enumerate(dates):
        if di % 5 == 0 or di == 0:
            print(f"[day] {date} ({di+1}/{len(dates)}) rows={len(rows)}", flush=True)
        for sym in symbols:
            sdf = stock_by.get(sym)
            if sdf is None:
                continue
            day = sdf[sdf["date"].astype(str) == date]
            if day.empty:
                continue
            tday = load_option_trades(trades_root, sym, date)
            if tday is None or tday.empty:
                n_miss_trades += 1
                continue
            trade_paths = _paths_by_ticker(tday)
            if not trade_paths:
                n_miss_trades += 1
                continue
            ts_ns, px = _stock_arrays(day)
            by_dte = multi_idx.get((sym, date))
            for sess_name, s0, s1 in active_sessions:
                t_start = pd.Timestamp(f"{date} {s0}:00", tz=NY)
                t_end = pd.Timestamp(f"{date} {s1}:00", tz=NY)
                t = t_start + pd.Timedelta(seconds=lb)
                while t < t_end:
                    direction, sr = _stock_dir_arr(
                        ts_ns, px, t, args.lookback_sec, float(args.min_abs_stock_ret)
                    )
                    if direction is None:
                        n_miss_dir += 1
                        t += stride
                        continue
                    spot = _spot_at_arr(ts_ns, px, t)
                    ticker, dte, _src = resolve_open_lock_contract(
                        by_dte,
                        direction=direction,
                        moneyness="ATM",
                        spot=spot,
                        prefer_dte=int(args.prefer_dte),
                        allowed_dte=allowed_dte,
                        clear_otm_thresh=0.01,
                        ladder=True,
                        otm_rungs=otm_rungs,
                    )
                    if not ticker:
                        n_miss_lock += 1
                        t += stride
                        continue
                    key = str(ticker).replace("O:", "")
                    arr = trade_paths.get(key)
                    if arr is None:
                        n_miss_trades += 1
                        t += stride
                        continue
                    pts, plast = arr
                    for fr in _fwd_trade_rets_arr(
                        pts, plast, t, horizons, slip=float(args.slip)
                    ):
                        rows.append(
                            {
                                "date": date,
                                "symbol": sym,
                                "session": sess_name,
                                "dir": direction,
                                "entry_ts": str(t),
                                "ticker": ticker,
                                "dte": dte,
                                "stock_ret_lb": sr,
                                **fr,
                            }
                        )
                    t += stride

    out = Path(paths["results_dir"]) / args.tag
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    if df.empty:
        print("no rows", flush=True)
        (out / "summary.json").write_text(
            json.dumps({"n": 0, "n_miss_trades": n_miss_trades}, indent=2),
            encoding="utf-8",
        )
        return 1
    df.to_parquet(out / "events.parquet", index=False)
    df.to_csv(out / "events.csv", index=False)

    # scoreboard: mean clock/oracle by session × horizon
    score = (
        df.groupby(["session", "horizon_sec"], as_index=False)
        .agg(
            n=("clock_ret", "size"),
            clock_mean=("clock_ret", "mean"),
            clock_med=("clock_ret", "median"),
            clock_win=("clock_ret", lambda s: float((s > 0).mean())),
            oracle_mean=("oracle_ret", "mean"),
            oracle_med=("oracle_ret", "median"),
            oracle_hold_p50=("oracle_hold_sec", "median"),
            oracle_hold_p75=("oracle_hold_sec", lambda s: float(np.nanpercentile(s, 75))),
            mfe_mean=("mfe", "mean"),
            mae_mean=("mae", "mean"),
        )
        .sort_values(["session", "horizon_sec"])
    )
    score.to_csv(out / "scoreboard.csv", index=False)

    # best H per session by clock_mean / oracle_mean
    picks = []
    for sess, g in score.groupby("session"):
        best_clock = g.loc[g["clock_mean"].idxmax()]
        best_oracle = g.loc[g["oracle_mean"].idxmax()]
        # among H with clock_win>=0.5, pick max clock_mean
        ok = g[g["clock_win"] >= 0.48]
        best_ok = ok.loc[ok["clock_mean"].idxmax()] if len(ok) else best_clock
        picks.append(
            {
                "session": sess,
                "best_clock_H": int(best_clock["horizon_sec"]),
                "best_clock_mean": float(best_clock["clock_mean"]),
                "best_clock_win": float(best_clock["clock_win"]),
                "best_oracle_H": int(best_oracle["horizon_sec"]),
                "best_oracle_mean": float(best_oracle["oracle_mean"]),
                "recommend_H": int(best_ok["horizon_sec"]),
                "recommend_clock_mean": float(best_ok["clock_mean"]),
                "recommend_win": float(best_ok["clock_win"]),
                "oracle_hold_p50_at_recH": float(
                    g.loc[g["horizon_sec"] == best_ok["horizon_sec"], "oracle_hold_p50"].iloc[0]
                ),
                "oracle_hold_p75_at_recH": float(
                    g.loc[g["horizon_sec"] == best_ok["horizon_sec"], "oracle_hold_p75"].iloc[0]
                ),
            }
        )

    summary = {
        "start": args.start_date,
        "end": args.end_date,
        "trades_root": str(trades_root),
        "stock_source": args.stock_source,
        "sessions": [s[0] for s in active_sessions],
        "horizons": horizons,
        "stride_sec": args.stride_sec,
        "lookback_sec": args.lookback_sec,
        "slip": args.slip,
        "n_rows": int(len(df)),
        "n_miss_trades": int(n_miss_trades),
        "n_miss_lock": int(n_miss_lock),
        "n_miss_dir": int(n_miss_dir),
        "picks": picks,
        "note": (
            "Foresight for HOLD length only. Entry dir = causal stock lookback "
            f"({args.stock_source}). Pricing = option trade last ± slip. "
            "Do not compound opportunity fills as capacity-unlimited."
        ),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    (out / "picks.json").write_text(json.dumps(picks, indent=2), encoding="utf-8")

    print("\n=== scoreboard (clock_mean) ===", flush=True)
    print(
        score.pivot(index="horizon_sec", columns="session", values="clock_mean").to_string(),
        flush=True,
    )
    print("\n=== picks ===", flush=True)
    print(json.dumps(picks, indent=2), flush=True)
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Scan Mag7 for second-level launch-slope edges (steep first impulse).

Research-only, independent of 10:30 Rule-A freeze. Stock-path forward returns.

Windows (default both):
  - open: 09:30–10:30
  - mid:  10:30–11:00

Example:
  python -m maga7.tools.scan_morning_launch_slope \\
    --start-date 2026-05-01 --end-date 2026-07-17 \\
    --tag research_launch_slope_may_jul
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from maga7.common.bar_agg import load_stock_1s_day
from maga7.common.config import load_profile
from maga7.common.launch_slope import attach_launch_slope_features, launch_edges
from maga7.common.sec_mf import forward_returns

NY = "America/New_York"
DEFAULT_SYMBOLS = ("NVDA", "TSLA", "AAPL", "AMZN", "META", "MSFT", "AMD", "GOOGL")
FREEZE = (
    ROOT
    / "maga7"
    / "CONFIG"
    / "strategy_profiles"
    / "single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1.json"
)

# (name, signal_start, signal_end, label_end)
DEFAULT_SESSIONS = (
    ("open_0930_1030", "09:30", "10:30", "11:00"),
    ("mid_1030_1100", "10:30", "11:00", "11:30"),
)


@dataclass
class Event:
    date: str
    symbol: str
    dir: str
    ts: pd.Timestamp
    session: str
    slope_sec: int
    abs_ret_min: float
    ret_k: float
    from_prev: float
    vol_z: float
    mf: float
    mf_ok: bool
    peer_n: int
    entry_px: float
    rets: dict[int, float]


def _bdates(start: str, end: str) -> list[str]:
    return [d.strftime("%Y-%m-%d") for d in pd.bdate_range(start, end)]


def _slice(df: pd.DataFrame, *, start: str, end: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    ts = pd.to_datetime(df["timestamp"])
    if getattr(ts.dt, "tz", None) is None:
        ts = ts.dt.tz_localize(NY)
    else:
        ts = ts.dt.tz_convert(NY)
    out = df.copy()
    out["timestamp"] = ts
    t = out["timestamp"].dt.time
    lo = pd.Timestamp(start).time()
    hi = pd.Timestamp(end).time()
    return out[(t >= lo) & (t < hi)].sort_values("timestamp")


def _prior_close(stock_1s_root: Path, symbol: str, date: str, dates: list[str]) -> float | None:
    try:
        i = dates.index(date)
    except ValueError:
        i = -1
    for j in range(i - 1, -1, -1):
        prev = load_stock_1s_day(stock_1s_root, symbol, dates[j])
        if prev is None or prev.empty:
            continue
        c = float(prev["close"].iloc[-1])
        if c > 0:
            return c
    return None


def _peer_count(
    asof_ts: pd.Timestamp,
    direction: str,
    mf_by_sym: dict[str, tuple[pd.DatetimeIndex, np.ndarray]],
    *,
    self_sym: str,
) -> int:
    n = 0
    for sym, (idx, mf) in mf_by_sym.items():
        if sym == self_sym or len(idx) == 0:
            continue
        pos = idx.searchsorted(asof_ts, side="right") - 1
        if pos < 0:
            continue
        v = mf[pos]
        if not np.isfinite(v) or v == 0:
            continue
        if direction == "UP" and v > 0:
            n += 1
        elif direction == "DN" and v < 0:
            n += 1
    return n


def collect_day_events(
    *,
    stock_1s_root: Path,
    date: str,
    dates: list[str],
    symbols: Iterable[str],
    slope_secs: list[int],
    abs_ret_mins: list[float],
    horizons: list[int],
    sessions: list[tuple[str, str, str, str]],
    peak_lookback_sec: int = 60,
    mf_window_sec: int = 60,
    require_local_peak: bool = True,
    first_only: bool = True,
) -> list[Event]:
    max_h = max(horizons) if horizons else 0
    # Load enough buffer for farthest label_end
    label_ends = [s[3] for s in sessions]
    load_end = max(label_ends)
    raw: dict[str, pd.DataFrame] = {}
    prev_close: dict[str, float | None] = {}
    for sym in symbols:
        day = load_stock_1s_day(stock_1s_root, sym, date)
        if day is None or day.empty:
            continue
        buf = _slice(day, start="09:30", end=load_end)
        if buf.empty:
            continue
        raw[sym] = buf
        prev_close[sym] = _prior_close(stock_1s_root, sym, date, dates)
    if not raw:
        return []

    events: list[Event] = []
    for slope_k in slope_secs:
        feat_by: dict[str, pd.DataFrame] = {}
        mf_by_sym: dict[str, tuple[pd.DatetimeIndex, np.ndarray]] = {}
        for sym, buf in raw.items():
            f = attach_launch_slope_features(
                buf,
                slope_sec=int(slope_k),
                peak_lookback_sec=int(peak_lookback_sec),
                prev_close=prev_close.get(sym),
                mf_window_sec=int(mf_window_sec),
            )
            if f.empty:
                continue
            feat_by[sym] = f
            if "mf" in f.columns:
                mf_by_sym[sym] = (
                    pd.DatetimeIndex(f["timestamp"]),
                    f["mf"].to_numpy(dtype=np.float64),
                )

        for sess_name, sig_start, sig_end, _label_end in sessions:
            for sym, f in feat_by.items():
                ts = pd.DatetimeIndex(f["timestamp"])
                close = f["close"].to_numpy(dtype=np.float64)
                fwd = {h: forward_returns(close, h) for h in horizons}
                t_time = ts.time
                lo = pd.Timestamp(sig_start).time()
                hi = pd.Timestamp(sig_end).time()
                sig_mask = (t_time >= lo) & (t_time < hi) & np.isfinite(f["ret_k"].to_numpy())

                for thr in abs_ret_mins:
                    for direction in ("UP", "DN"):
                        edges = launch_edges(
                            f,
                            direction=direction,
                            abs_ret_min=float(thr),
                            require_local_peak=require_local_peak,
                        )
                        edges = edges[sig_mask[edges]]
                        if len(edges) == 0:
                            continue
                        idxs = [int(edges[0])] if first_only else [int(x) for x in edges]
                        for i in idxs:
                            if all(not np.isfinite(fwd[h][i]) for h in horizons):
                                continue
                            rets: dict[int, float] = {}
                            for h in horizons:
                                r = fwd[h][i]
                                if not np.isfinite(r):
                                    continue
                                rets[h] = float(r if direction == "UP" else -r)
                            if not rets:
                                continue
                            asof = ts[i]
                            mf_v = (
                                float(f["mf"].iloc[i])
                                if "mf" in f.columns and np.isfinite(f["mf"].iloc[i])
                                else float("nan")
                            )
                            mf_ok = False
                            if np.isfinite(mf_v):
                                mf_ok = (direction == "UP" and mf_v > 0) or (
                                    direction == "DN" and mf_v < 0
                                )
                            peer = _peer_count(asof, direction, mf_by_sym, self_sym=sym)
                            vz = float(f["vol_z"].iloc[i]) if np.isfinite(f["vol_z"].iloc[i]) else float("nan")
                            events.append(
                                Event(
                                    date=date,
                                    symbol=sym,
                                    dir=direction,
                                    ts=asof,
                                    session=sess_name,
                                    slope_sec=int(slope_k),
                                    abs_ret_min=float(thr),
                                    ret_k=float(f["ret_k"].iloc[i]),
                                    from_prev=float(f["from_prev"].iloc[i]),
                                    vol_z=vz,
                                    mf=mf_v,
                                    mf_ok=bool(mf_ok),
                                    peer_n=int(peer),
                                    entry_px=float(close[i]),
                                    rets=rets,
                                )
                            )
    return events


def scoreboard(
    events: list[Event],
    *,
    horizons: list[int],
    from_prev_mins: list[float],
    vol_z_mins: list[float],
    peer_mins: list[int],
    mf_confirm: list[int],
) -> pd.DataFrame:
    rows = []
    if not events:
        return pd.DataFrame()
    by_key: dict[tuple[str, int, float], list[Event]] = {}
    for ev in events:
        by_key.setdefault((ev.session, ev.slope_sec, ev.abs_ret_min), []).append(ev)

    for (sess, slope_k, thr), group in by_key.items():
        for fp in from_prev_mins:
            for vz in vol_z_mins:
                for peer in peer_mins:
                    for mfc in mf_confirm:
                        for H in horizons:
                            picked = []
                            for ev in group:
                                if H not in ev.rets:
                                    continue
                                if ev.dir == "UP" and ev.from_prev < fp:
                                    continue
                                if ev.dir == "DN" and ev.from_prev > -fp:
                                    continue
                                if np.isfinite(ev.vol_z) and ev.vol_z < vz:
                                    continue
                                if ev.peer_n < peer:
                                    continue
                                if mfc and not ev.mf_ok:
                                    continue
                                picked.append(ev.rets[H])
                            if len(picked) < 8:
                                continue
                            arr = np.asarray(picked, dtype=np.float64)
                            rows.append(
                                {
                                    "session": sess,
                                    "slope_sec": slope_k,
                                    "abs_ret_min": thr,
                                    "horizon_sec": H,
                                    "from_prev_min": fp,
                                    "vol_z_min": vz,
                                    "peer_min": peer,
                                    "mf_confirm": int(mfc),
                                    "n": int(len(arr)),
                                    "win": float((arr > 0).mean()),
                                    "exp": float(arr.mean()),
                                    "med": float(np.median(arr)),
                                    "sum": float(arr.sum()),
                                    "p05": float(np.quantile(arr, 0.05)),
                                    "p95": float(np.quantile(arr, 0.95)),
                                    "sharpe_ish": float(arr.mean() / arr.std())
                                    if arr.std() > 1e-12
                                    else 0.0,
                                }
                            )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values(["exp", "n"], ascending=[False, False]).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", default=str(FREEZE))
    ap.add_argument("--start-date", default="2026-05-01")
    ap.add_argument("--end-date", default="2026-07-17")
    ap.add_argument("--tag", default="research_launch_slope_may_jul")
    ap.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    ap.add_argument("--slope-secs", default="3,5,10")
    ap.add_argument("--abs-ret-mins", default="0.0008,0.0012,0.002,0.003")
    ap.add_argument("--horizons", default="15,30,60,120,180")
    ap.add_argument("--from-prev", default="0,0.003,0.005")
    ap.add_argument("--vol-z", default="0,1.0,1.5")
    ap.add_argument("--peer", default="0,2,3")
    ap.add_argument("--mf-confirm", default="0,1", help="0=off 1=require sec mf same dir")
    ap.add_argument("--peak-lookback", type=int, default=60)
    ap.add_argument("--mf-window-sec", type=int, default=60)
    ap.add_argument("--no-local-peak", action="store_true")
    ap.add_argument("--all-edges", action="store_true", help="keep every edge (default: first/day/sym/dir/cell)")
    ap.add_argument("--sessions", default="open_0930_1030,mid_1030_1100")
    ap.add_argument("--max-days", type=int, default=0)
    args = ap.parse_args()

    profile = load_profile(args.profile)
    stock_1s = Path(profile["_paths"]["stock_1s_root"])
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    slope_secs = [int(x) for x in args.slope_secs.split(",") if x.strip()]
    abs_ret_mins = [float(x) for x in args.abs_ret_mins.split(",") if x.strip()]
    horizons = [int(x) for x in args.horizons.split(",") if x.strip()]
    from_prevs = [float(x) for x in args.from_prev.split(",") if x.strip()]
    vol_zs = [float(x) for x in args.vol_z.split(",") if x.strip()]
    peers = [int(x) for x in args.peer.split(",") if x.strip()]
    mf_conf = [int(x) for x in args.mf_confirm.split(",") if x.strip()]

    sess_map = {s[0]: s for s in DEFAULT_SESSIONS}
    sessions = []
    for name in args.sessions.split(","):
        name = name.strip()
        if name in sess_map:
            sessions.append(sess_map[name])
        else:
            raise SystemExit(f"unknown session {name}; choose {list(sess_map)}")

    dates = _bdates(args.start_date, args.end_date)
    if args.max_days and args.max_days > 0:
        dates = dates[: int(args.max_days)]

    out_dir = Path(profile["_paths"]["results_dir"]) / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    all_events: list[Event] = []
    for i, date in enumerate(dates, 1):
        evs = collect_day_events(
            stock_1s_root=stock_1s,
            date=date,
            dates=dates,
            symbols=symbols,
            slope_secs=slope_secs,
            abs_ret_mins=abs_ret_mins,
            horizons=horizons,
            sessions=sessions,
            peak_lookback_sec=int(args.peak_lookback),
            mf_window_sec=int(args.mf_window_sec),
            require_local_peak=not bool(args.no_local_peak),
            first_only=not bool(args.all_edges),
        )
        all_events.extend(evs)
        if i % 10 == 0 or i == len(dates):
            print(f"[{i}/{len(dates)}] {date} events_total={len(all_events)}", flush=True)

    long_rows = []
    for ev in all_events:
        for h, r in ev.rets.items():
            long_rows.append(
                {
                    "date": ev.date,
                    "symbol": ev.symbol,
                    "dir": ev.dir,
                    "ts": str(ev.ts),
                    "session": ev.session,
                    "slope_sec": ev.slope_sec,
                    "abs_ret_min": ev.abs_ret_min,
                    "ret_k": ev.ret_k,
                    "from_prev": ev.from_prev,
                    "vol_z": ev.vol_z,
                    "mf": ev.mf,
                    "mf_ok": ev.mf_ok,
                    "peer_n": ev.peer_n,
                    "entry_px": ev.entry_px,
                    "horizon_sec": h,
                    "fwd_ret_signed": r,
                }
            )
    events_df = pd.DataFrame(long_rows)
    if len(events_df):
        events_df.to_parquet(out_dir / "events.parquet", index=False)
        events_df.to_csv(out_dir / "events.csv", index=False)
    else:
        events_df.to_csv(out_dir / "events.csv", index=False)

    board = scoreboard(
        all_events,
        horizons=horizons,
        from_prev_mins=from_prevs,
        vol_z_mins=vol_zs,
        peer_mins=peers,
        mf_confirm=mf_conf,
    )
    board.to_csv(out_dir / "scoreboard.csv", index=False)
    pos = board[board["exp"] > 0].copy() if not board.empty else board
    top = pos.head(40) if not pos.empty else board.head(40)
    top.to_csv(out_dir / "scoreboard_top.csv", index=False)

    # Per-session best
    sess_best = {}
    if not board.empty:
        for sess in board["session"].unique():
            sub = board[board["session"] == sess].sort_values("exp", ascending=False)
            sess_best[str(sess)] = sub.head(5).to_dict(orient="records")

    summary = {
        "start": args.start_date,
        "end": args.end_date,
        "n_dates": len(dates),
        "symbols": symbols,
        "sessions": [s[0] for s in sessions],
        "slope_secs": slope_secs,
        "abs_ret_mins": abs_ret_mins,
        "n_events_struct": len(all_events),
        "n_event_horizon_rows": int(len(events_df)),
        "n_grid_cells": int(len(board)),
        "n_positive_exp_cells": int((board["exp"] > 0).sum()) if not board.empty else 0,
        "best_overall": top.head(8).to_dict(orient="records") if not top.empty else [],
        "best_by_session": sess_best,
        "note": (
            "Stock-path signed forward return after launch-slope edge. "
            "Not option PnL; not freeze Rule-A. Local-peak = ret_k is causal "
            "rolling max/min over peak_lookback. Next: option-fill on top cells."
        ),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    print("\n=== launch-slope scoreboard (top by exp) ===")
    if top.empty:
        print("no cells")
    else:
        cols = [
            "session",
            "slope_sec",
            "abs_ret_min",
            "horizon_sec",
            "from_prev_min",
            "vol_z_min",
            "peer_min",
            "mf_confirm",
            "n",
            "win",
            "exp",
            "sum",
            "sharpe_ish",
        ]
        print(top[cols].head(25).to_string(index=False))
    print(json.dumps({k: summary[k] for k in summary if k not in {"best_overall", "best_by_session"}}, indent=2))
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()

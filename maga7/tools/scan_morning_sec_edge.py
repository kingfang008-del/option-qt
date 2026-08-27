#!/usr/bin/env python3
"""Scan Mag7 morning (09:30–10:30) for second-level MF edge — stock path only.

Independent of the 10:30 Rule-A freeze book. Goal: does a second-window
money-flow proxy have positive forward stock expectancy in the open hour?

- Features: same net$ formula as 1m Rule-A, rolling window in **seconds**
- Labels: forward stock return over H seconds (no option fill in this pass)
- Output: grid scoreboard sorted by expectancy (not vs baseline)

Example:
  python -m maga7.tools.scan_morning_sec_edge \\
    --start-date 2026-02-01 --end-date 2026-07-17 \\
    --tag research_morn_sec_edge_feb_jul
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
from maga7.common.sec_mf import attach_sec_mf_features, forward_returns

NY = "America/New_York"
DEFAULT_SYMBOLS = ("NVDA", "TSLA", "AAPL", "AMZN", "META", "MSFT", "AMD", "GOOGL")
FREEZE = (
    ROOT
    / "maga7"
    / "CONFIG"
    / "strategy_profiles"
    / "single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1.json"
)


@dataclass
class Event:
    date: str
    symbol: str
    dir: str
    ts: pd.Timestamp
    mf_window_sec: int
    streak_min: int
    from_prev: float
    vol_z: float
    peer_n: int
    entry_px: float
    rets: dict[int, float]  # horizon -> fwd ret (signed for dir)


def _bdates(start: str, end: str) -> list[str]:
    return [d.strftime("%Y-%m-%d") for d in pd.bdate_range(start, end)]


def _morning_slice(df: pd.DataFrame, *, start: str, end: str) -> pd.DataFrame:
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
    """Last close of previous available 1s day (overnight reference)."""
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


def _rising_edges(streak: np.ndarray, smin: int) -> np.ndarray:
    """Indices where streak first reaches smin."""
    smin = int(smin)
    if len(streak) == 0:
        return np.array([], dtype=np.int64)
    hit = streak >= smin
    prev = np.concatenate([[False], hit[:-1]])
    return np.flatnonzero(hit & ~prev)


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
        # last mf at or before asof
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
    windows: list[int],
    streak_mins: list[int],
    horizons: list[int],
    signal_end: str = "10:30",
    label_end: str = "11:00",
) -> list[Event]:
    max_h = max(horizons)
    # Load through label_end so forward returns near 10:30 still resolve.
    raw: dict[str, pd.DataFrame] = {}
    prev_close: dict[str, float | None] = {}
    for sym in symbols:
        day = load_stock_1s_day(stock_1s_root, sym, date)
        if day.empty:
            continue
        # need full morning+buffer for labels; features only fire before signal_end
        buf = _morning_slice(day, start="09:30", end=label_end)
        if buf.empty:
            continue
        raw[sym] = buf
        prev_close[sym] = _prior_close(stock_1s_root, sym, date, dates)

    if not raw:
        return []

    events: list[Event] = []
    for W in windows:
        feat: dict[str, pd.DataFrame] = {}
        mf_by_sym: dict[str, tuple[pd.DatetimeIndex, np.ndarray]] = {}
        for sym, buf in raw.items():
            prev = prev_close.get(sym)
            f = attach_sec_mf_features(buf, mf_window_sec=W, vol_ma_sec=max(300, W * 2), prev_close=prev)
            if f.empty:
                continue
            feat[sym] = f
            mf_by_sym[sym] = (pd.DatetimeIndex(f["timestamp"]), f["mf"].to_numpy(dtype=np.float64))

        for sym, f in feat.items():
            ts = pd.DatetimeIndex(f["timestamp"])
            close = f["close"].to_numpy(dtype=np.float64)
            fwd = {h: forward_returns(close, h) for h in horizons}
            # truncate signal window
            sig_mask = ts.time < pd.Timestamp(signal_end).time()
            # also need warmup
            sig_mask = sig_mask & np.isfinite(f["mf"].to_numpy(dtype=np.float64))

            for smin in streak_mins:
                for direction, streak_col in (("UP", "streak_up"), ("DN", "streak_dn")):
                    edges = _rising_edges(f[streak_col].to_numpy(), smin)
                    edges = edges[sig_mask[edges]]
                    if len(edges) == 0:
                        continue
                    # first fire per symbol/dir/day/W/smin
                    i = int(edges[0])
                    # require enough future for max horizon if possible; else skip thin tail
                    if i + max_h >= len(close):
                        # still keep if any horizon available
                        if all(not np.isfinite(fwd[h][i]) for h in horizons):
                            continue
                    rets = {}
                    for h in horizons:
                        r = fwd[h][i]
                        if not np.isfinite(r):
                            continue
                        rets[h] = float(r if direction == "UP" else -r)
                    if not rets:
                        continue
                    asof = ts[i]
                    peer = _peer_count(asof, direction, mf_by_sym, self_sym=sym)
                    events.append(
                        Event(
                            date=date,
                            symbol=sym,
                            dir=direction,
                            ts=asof,
                            mf_window_sec=int(W),
                            streak_min=int(smin),
                            from_prev=float(f["from_prev"].iloc[i]),
                            vol_z=float(f["vol_z"].iloc[i]) if np.isfinite(f["vol_z"].iloc[i]) else float("nan"),
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
) -> pd.DataFrame:
    rows = []
    if not events:
        return pd.DataFrame()
    # group by structural keys first
    by_key: dict[tuple[int, int], list[Event]] = {}
    for ev in events:
        by_key.setdefault((ev.mf_window_sec, ev.streak_min), []).append(ev)

    for (W, S), group in by_key.items():
        for fp in from_prev_mins:
            for vz in vol_z_mins:
                for peer in peer_mins:
                    for H in horizons:
                        picked = []
                        for ev in group:
                            if H not in ev.rets:
                                continue
                            # direction-aware from_prev
                            if ev.dir == "UP" and ev.from_prev < fp:
                                continue
                            if ev.dir == "DN" and ev.from_prev > -fp:
                                continue
                            if np.isfinite(ev.vol_z) and ev.vol_z < vz:
                                continue
                            if ev.peer_n < peer:
                                continue
                            picked.append(ev.rets[H])
                        if len(picked) < 8:
                            continue
                        arr = np.asarray(picked, dtype=np.float64)
                        rows.append(
                            {
                                "mf_window_sec": W,
                                "streak_min": S,
                                "horizon_sec": H,
                                "from_prev_min": fp,
                                "vol_z_min": vz,
                                "peer_min": peer,
                                "n": int(len(arr)),
                                "win": float((arr > 0).mean()),
                                "exp": float(arr.mean()),
                                "med": float(np.median(arr)),
                                "sum": float(arr.sum()),
                                "p05": float(np.quantile(arr, 0.05)),
                                "p95": float(np.quantile(arr, 0.95)),
                                "sharpe_ish": float(arr.mean() / arr.std()) if arr.std() > 1e-12 else 0.0,
                            }
                        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values(["exp", "n"], ascending=[False, False]).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", default=str(FREEZE))
    ap.add_argument("--start-date", default="2026-02-01")
    ap.add_argument("--end-date", default="2026-07-17")
    ap.add_argument("--tag", default="research_morn_sec_edge_feb_jul")
    ap.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    ap.add_argument("--windows", default="30,60,100,180,300")
    ap.add_argument("--streaks", default="20,40,60,100")
    ap.add_argument("--horizons", default="30,60,120,180,300")
    ap.add_argument("--from-prev", default="0,0.003,0.005,0.01")
    ap.add_argument("--vol-z", default="0,1.0,1.5")
    ap.add_argument("--peer", default="0,2,3")
    ap.add_argument("--max-days", type=int, default=0, help="debug cap; 0=all")
    args = ap.parse_args()

    profile = load_profile(args.profile)
    stock_1s = Path(profile["_paths"]["stock_1s_root"])
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    windows = [int(x) for x in args.windows.split(",") if x.strip()]
    streaks = [int(x) for x in args.streaks.split(",") if x.strip()]
    horizons = [int(x) for x in args.horizons.split(",") if x.strip()]
    from_prevs = [float(x) for x in getattr(args, "from_prev").split(",") if x.strip()]
    vol_zs = [float(x) for x in getattr(args, "vol_z").split(",") if x.strip()]
    peers = [int(x) for x in args.peer.split(",") if x.strip()]

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
            windows=windows,
            streak_mins=streaks,
            horizons=horizons,
        )
        all_events.extend(evs)
        if i % 10 == 0 or i == len(dates):
            print(f"[{i}/{len(dates)}] {date} events_total={len(all_events)}", flush=True)

    # persist raw event table (long: one row per event x horizon)
    long_rows = []
    for ev in all_events:
        for h, r in ev.rets.items():
            long_rows.append(
                {
                    "date": ev.date,
                    "symbol": ev.symbol,
                    "dir": ev.dir,
                    "ts": str(ev.ts),
                    "mf_window_sec": ev.mf_window_sec,
                    "streak_min": ev.streak_min,
                    "from_prev": ev.from_prev,
                    "vol_z": ev.vol_z,
                    "peer_n": ev.peer_n,
                    "entry_px": ev.entry_px,
                    "horizon_sec": h,
                    "fwd_ret_signed": r,
                }
            )
    events_df = pd.DataFrame(long_rows)
    events_df.to_parquet(out_dir / "events.parquet", index=False)
    events_df.to_csv(out_dir / "events.csv", index=False)

    board = scoreboard(
        all_events,
        horizons=horizons,
        from_prev_mins=from_prevs,
        vol_z_mins=vol_zs,
        peer_mins=peers,
    )
    board.to_csv(out_dir / "scoreboard.csv", index=False)

    # Top positive cells + summary
    pos = board[board["exp"] > 0].copy() if not board.empty else board
    top = pos.head(30) if not pos.empty else board.head(30)
    top.to_csv(out_dir / "scoreboard_top.csv", index=False)

    summary = {
        "start": args.start_date,
        "end": args.end_date,
        "n_dates": len(dates),
        "symbols": symbols,
        "n_events_struct": len(all_events),
        "n_event_horizon_rows": int(len(events_df)),
        "n_grid_cells": int(len(board)),
        "n_positive_exp_cells": int((board["exp"] > 0).sum()) if not board.empty else 0,
        "best": top.head(5).to_dict(orient="records") if not top.empty else [],
        "note": (
            "Stock-path only (signed forward return). "
            "Not option PnL; not compared to 10:30 freeze book. "
            "Positive exp here = candidate space, still needs option-fill validation."
        ),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n=== morning sec-edge scoreboard (top by exp) ===")
    if top.empty:
        print("no cells")
    else:
        cols = [
            "mf_window_sec",
            "streak_min",
            "horizon_sec",
            "from_prev_min",
            "vol_z_min",
            "peer_min",
            "n",
            "win",
            "exp",
            "sum",
            "sharpe_ish",
        ]
        print(top[cols].head(20).to_string(index=False))
    print(json.dumps({k: summary[k] for k in summary if k != "best"}, indent=2))
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()

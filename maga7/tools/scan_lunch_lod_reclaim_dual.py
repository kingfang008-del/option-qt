#!/usr/bin/env python3
"""Lunch LOD-reclaim UP sleeve (research) — V-reversal hour 12:30–13:30.

Causal trigger on Mag7 1m bars (full session fed so LOD/open are session-true):

  wash:   (day_open - session_lo) / day_open ≥ min_wash
  bounce: (px - session_lo) / day_open ≥ min_bounce
  room:   chase_up = (px-lo)/(hi-lo) ≤ max_chase   (avoid HOD chase)
  optional: session LOD printed **before** window_start (true V from AM low)

First hit per symbol×day → ATM call · trades last±slip · TP/SL dual windows
(may_jul09 / jul10_23). Independent of Rule-A; CORE owns 10:30–14:00.

Example:
  PYTHONPATH=. python -m maga7.tools.scan_lunch_lod_reclaim_dual \\
    --tag research_lunch_lod_reclaim_v1
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

from maga7.common.config import load_profile
from maga7.common.open_lock import (
    load_multidte_lock_index,
    resolve_open_lock_contract,
    resolve_otm_rungs,
)
from maga7.common.option_trade_tpsl import simulate_trade_tpsl
from maga7.common.option_trades import load_option_trades
from maga7.common.replay import month_list, to_ny
from maga7.common.signals import load_stock_month_files
from maga7.common.stock_1s import session_dates
from maga7.tools.run_morning_sec_option_fill import _portfolio_day
from maga7.tools.scan_am_certainty_morph_tpsl import _ok, _stats
from maga7.tools.scan_session_horizon_foresight import _paths_by_ticker, _spot_at_arr, _stock_arrays

NY = "America/New_York"
DEFAULT_TRADES = Path("/mnt/s990/new_option_data_s3_trades")
PROFILE = (
    "maga7/CONFIG/strategy_profiles/"
    "single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1.json"
)
WINDOWS = (
    ("may_jul09", "2026-05-01", "2026-07-09"),
    ("jul10_23", "2026-07-10", "2026-07-23"),
)


def _window_of(date: str) -> str | None:
    for name, a, b in WINDOWS:
        if a <= date <= b:
            return name
    return None


def _hhmm_min(hhmm: str) -> int:
    p = str(hhmm).split(":")
    return int(p[0]) * 60 + int(p[1])


def _spot_from_1m(day: pd.DataFrame, ts: pd.Timestamp) -> float | None:
    if day is None or day.empty:
        return None
    t = to_ny(ts)
    sub = day[pd.to_datetime(day["timestamp"]) <= t]
    if sub.empty:
        return None
    px = float(sub.iloc[-1]["close"])
    return px if px > 0 else None


def detect_lod_reclaim(
    day1m: pd.DataFrame,
    *,
    window_start: str,
    window_end: str,
    min_wash: float,
    min_bounce: float,
    max_chase: float,
    require_lod_before_window: bool,
) -> dict[str, Any] | None:
    """Return first reclaim arm in window, or None."""
    if day1m is None or day1m.empty:
        return None
    day = day1m.sort_values("timestamp").copy()
    day["timestamp"] = pd.to_datetime(day["timestamp"], utc=True).dt.tz_convert(NY)
    w0 = _hhmm_min(window_start)
    w1 = _hhmm_min(window_end)

    day_open = None
    hi = None
    lo = None
    lod_ts = None
    for r in day.itertuples(index=False):
        ts = pd.Timestamp(r.timestamp)
        o, h, l, c = float(r.open), float(r.high), float(r.low), float(r.close)
        if not (o > 0 and h > 0 and l > 0 and c > 0):
            continue
        if day_open is None:
            day_open = o
            hi, lo, lod_ts = h, l, ts
        else:
            if h > hi:
                hi = h
            if l < lo:
                lo = l
                lod_ts = ts
        hm = ts.hour * 60 + ts.minute
        if hm < w0 or hm >= w1:
            continue
        wash = (day_open - lo) / day_open
        bounce = (c - lo) / day_open
        if hi > lo:
            chase = (c - lo) / (hi - lo)
        else:
            chase = 0.5
        if wash + 1e-12 < float(min_wash):
            continue
        if bounce + 1e-12 < float(min_bounce):
            continue
        if chase > float(max_chase) + 1e-12:
            continue
        if require_lod_before_window and lod_ts is not None:
            lod_hm = lod_ts.hour * 60 + lod_ts.minute
            if lod_hm >= w0:
                continue
        return {
            "arm_ts": ts,
            "dir": "UP",
            "day_open": float(day_open),
            "px": float(c),
            "session_hi": float(hi),
            "session_lo": float(lo),
            "lod_ts": str(lod_ts) if lod_ts is not None else None,
            "wash": float(wash),
            "bounce": float(bounce),
            "chase": float(chase),
            "from_open": float(c / day_open - 1.0),
        }
    return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", default=PROFILE)
    ap.add_argument("--tag", default="research_lunch_lod_reclaim_v1")
    ap.add_argument("--trades-root", default=str(DEFAULT_TRADES))
    ap.add_argument("--window-start", default="12:30")
    ap.add_argument("--window-end", default="13:30")
    ap.add_argument("--min-wash", default="0.005,0.008,0.01,0.012")
    ap.add_argument("--min-bounce", default="0.003,0.005,0.008,0.01")
    ap.add_argument("--max-chase", default="0.85,0.90,0.95")
    ap.add_argument(
        "--lod-before",
        default="0,1",
        help="0=allow LOD inside window; 1=require LOD before window_start",
    )
    ap.add_argument("--tp", default="0.10,0.15,0.20")
    ap.add_argument("--sl", default="0.15,0.20,0.25")
    ap.add_argument("--max-hold-sec", type=int, default=900)
    ap.add_argument("--slip", type=float, default=0.01)
    ap.add_argument("--position-frac", type=float, default=0.10)
    ap.add_argument("--max-concurrent", type=int, default=2)
    ap.add_argument("--cooldown-minutes", type=float, default=10.0)
    ap.add_argument("--min-n", type=int, default=8)
    ap.add_argument("--min-day-win", type=float, default=0.55)
    args = ap.parse_args(argv)

    washes = [float(x) for x in str(args.min_wash).split(",") if x.strip()]
    bounces = [float(x) for x in str(args.min_bounce).split(",") if x.strip()]
    chases = [float(x) for x in str(args.max_chase).split(",") if x.strip()]
    lod_befores = [bool(int(x)) for x in str(args.lod_before).split(",") if x.strip()]
    tps = [float(x) for x in str(args.tp).split(",") if x.strip()]
    sls = [float(x) for x in str(args.sl).split(",") if x.strip()]
    w_start, w_end = str(args.window_start), str(args.window_end)

    prof = load_profile(args.profile)
    paths = prof["_paths"]
    symbols = list(prof.get("symbols") or [])
    stock_root = Path(paths["stock_root"])
    stock_1s = Path(paths.get("stock_1s_root") or "/mnt/s990/data/raw_1s/stocks").expanduser()
    trades_root = Path(args.trades_root)
    lock = load_multidte_lock_index(Path(paths["open_locked_map"]).expanduser())
    otm = resolve_otm_rungs(prof, default=3)
    out = Path(paths["results_dir"]) / args.tag
    out.mkdir(parents=True, exist_ok=True)

    start_all = min(w[1] for w in WINDOWS)
    end_all = max(w[2] for w in WINDOWS)
    dates = [d for d in session_dates(start_all, end_all) if start_all <= d <= end_all]
    months = month_list(start_all, end_all)
    print(
        f"lunch LOD-reclaim UP {w_start}..{w_end} {start_all}..{end_all} days={len(dates)}",
        flush=True,
    )

    stock_by: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        sdf = load_stock_month_files(stock_root, sym, months)
        if sdf is not None and not sdf.empty:
            stock_by[sym] = sdf

    # Trigger grid keys (without tp/sl)
    trig_keys: list[tuple[float, float, float, bool]] = [
        (w, b, c, lb) for w in washes for b in bounces for c in chases for lb in lod_befores
        if b <= w + 1e-12  # bounce cannot exceed wash depth meaningfully for reclaim
    ]
    print(f"trigger combos={len(trig_keys)} × tp/sl cells later", flush=True)

    arms: list[dict[str, Any]] = []
    for di, date in enumerate(dates):
        if di % 10 == 0:
            print(f"[day] {date} ({di+1}/{len(dates)}) arms={len(arms)}", flush=True)
        for sym in symbols:
            sdf = stock_by.get(sym)
            if sdf is None:
                continue
            day1m = sdf[sdf["date"].astype(str) == date]
            if day1m.empty:
                continue
            by_dte = lock.get((sym, date))
            if not by_dte:
                continue
            tday = load_option_trades(trades_root, sym, date)
            if tday is None or tday.empty:
                continue
            tpaths = _paths_by_ticker(tday)
            if not tpaths:
                continue
            day1s = None
            try:
                from maga7.common.bar_agg import load_stock_1s_day

                day1s = load_stock_1s_day(stock_1s, sym, date)
            except Exception:
                day1s = None
            ts_ns = px_arr = None
            if day1s is not None and not day1s.empty:
                ts_ns, px_arr = _stock_arrays(day1s)

            for min_wash, min_bounce, max_chase, lod_before in trig_keys:
                hit = detect_lod_reclaim(
                    day1m,
                    window_start=w_start,
                    window_end=w_end,
                    min_wash=min_wash,
                    min_bounce=min_bounce,
                    max_chase=max_chase,
                    require_lod_before_window=lod_before,
                )
                if hit is None:
                    continue
                arm_ts = to_ny(hit["arm_ts"])
                spot = None
                if ts_ns is not None and px_arr is not None:
                    spot = _spot_at_arr(ts_ns, px_arr, arm_ts)
                if spot is None:
                    spot = _spot_from_1m(day1m, arm_ts)
                ticker, dte, _ = resolve_open_lock_contract(
                    by_dte,
                    direction="UP",
                    moneyness="ATM",
                    spot=spot,
                    prefer_dte=0,
                    allowed_dte=[0, 1, 2],
                    clear_otm_thresh=0.01,
                    ladder=True,
                    otm_rungs=otm,
                )
                if not ticker:
                    continue
                arr = tpaths.get(str(ticker).replace("O:", ""))
                if arr is None:
                    continue
                arms.append(
                    {
                        "date": date,
                        "symbol": sym,
                        "dir": "UP",
                        "min_wash": min_wash,
                        "min_bounce": min_bounce,
                        "max_chase": max_chase,
                        "lod_before": int(lod_before),
                        "arm_ts": arm_ts,
                        "wash": hit["wash"],
                        "bounce": hit["bounce"],
                        "chase": hit["chase"],
                        "from_open": hit["from_open"],
                        "lod_ts": hit["lod_ts"],
                        "ticker": ticker,
                        "dte": dte,
                        "pts": arr[0],
                        "plast": arr[1],
                    }
                )

    print(f"arms={len(arms)}; scoring…", flush=True)
    if arms:
        pd.DataFrame(
            [{k: v for k, v in a.items() if k not in {"pts", "plast"}} for a in arms]
        ).to_csv(out / "arms.csv", index=False)

    cells: list[dict[str, Any]] = []
    for min_wash, min_bounce, max_chase, lod_before in trig_keys:
        for tp in tps:
            for sl in sls:
                cells.append(
                    {
                        "name": (
                            f"lodrec_w{min_wash}_b{min_bounce}_c{max_chase}"
                            f"_lb{int(lod_before)}_tp{tp}_sl{sl}"
                        ),
                        "min_wash": min_wash,
                        "min_bounce": min_bounce,
                        "max_chase": max_chase,
                        "lod_before": int(lod_before),
                        "tp": tp,
                        "sl": sl,
                    }
                )

    dual_pass: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []

    for ci, cell in enumerate(cells):
        if ci % 40 == 0:
            print(f"[cell] {ci+1}/{len(cells)} dual={len(dual_pass)}", flush=True)
        win_raw: dict[str, list[dict]] = {w[0]: [] for w in WINDOWS}
        for arm in arms:
            if float(arm["min_wash"]) != float(cell["min_wash"]):
                continue
            if float(arm["min_bounce"]) != float(cell["min_bounce"]):
                continue
            if float(arm["max_chase"]) != float(cell["max_chase"]):
                continue
            if int(arm["lod_before"]) != int(cell["lod_before"]):
                continue
            wname = _window_of(str(arm["date"]))
            if wname is None:
                continue
            entry_ts = arm["arm_ts"]
            sim = simulate_trade_tpsl(
                arm["pts"],
                arm["plast"],
                entry_ts,
                tp=float(cell["tp"]),
                sl=float(cell["sl"]),
                max_hold_sec=int(args.max_hold_sec),
                slip=float(args.slip),
            )
            if sim is None or not np.isfinite(sim["ret"]):
                continue
            et = to_ny(entry_ts)
            xt = et + pd.Timedelta(seconds=float(sim["hold_sec"]))
            win_raw[wname].append(
                {
                    "date": arm["date"],
                    "symbol": arm["symbol"],
                    "dir": "UP",
                    "entry_ts": et,
                    "exit_ts": xt,
                    "ret": float(sim["ret"]),
                    "exit_reason": str(sim["reason"]),
                    "hold_sec": float(sim["hold_sec"]),
                    "wash": arm["wash"],
                    "bounce": arm["bounce"],
                    "size": float(args.position_frac),
                }
            )

        win_stats: dict[str, dict[str, Any]] = {}
        sized_all: list[dict] = []
        for wname, _, _ in WINDOWS:
            raw = win_raw[wname]
            by_d: dict[str, list] = {}
            for r in raw:
                by_d.setdefault(str(r["date"]), []).append(r)
            sized: list[dict] = []
            for _, rs in sorted(by_d.items()):
                sized.extend(
                    _portfolio_day(
                        sorted(rs, key=lambda x: (x["entry_ts"], x["symbol"])),
                        position_frac=float(args.position_frac),
                        max_concurrent=int(args.max_concurrent),
                        cooldown_minutes=float(args.cooldown_minutes),
                    )
                )
            win_stats[wname] = _stats(sized)
            sized_all.extend(sized)

        both = True
        for wname, _, _ in WINDOWS:
            mn = int(args.min_n)
            if wname == "jul10_23":
                mn = min(mn, 6)
            if not _ok(win_stats[wname], min_n=mn, min_day_win=float(args.min_day_win)):
                both = False
                break

        row: dict[str, Any] = {**cell, "dual_pass": both}
        for wname, _, _ in WINDOWS:
            for k, v in win_stats[wname].items():
                row[f"{wname}_{k}"] = v
        score_rows.append(row)
        if both:
            dual_pass.append(row)
            print(
                f"  *** DUAL {cell['name']} "
                f"MJ n={row.get('may_jul09_n')} mean={row.get('may_jul09_mean'):+.3f} "
                f"J10 n={row.get('jul10_23_n')} mean={row.get('jul10_23_mean'):+.3f}",
                flush=True,
            )
            pd.DataFrame(sized_all).to_csv(out / f"trades_{cell['name']}.csv", index=False)

    sb = pd.DataFrame(score_rows)
    sb.to_csv(out / "scoreboard.csv", index=False)
    dual_pass = sorted(
        dual_pass,
        key=lambda r: (
            float(r.get("may_jul09_add") or 0) + float(r.get("jul10_23_add") or 0)
        ),
        reverse=True,
    )
    (out / "dual_pass.json").write_text(json.dumps(dual_pass, indent=2, default=str), encoding="utf-8")

    # Best near-miss / diagnostic
    if len(sb):
        sb2 = sb.copy()
        sb2["score"] = sb2.get("may_jul09_add", 0).fillna(0) + sb2.get("jul10_23_add", 0).fillna(0)
        top = sb2.sort_values(["dual_pass", "score"], ascending=[False, False]).head(15)
    else:
        top = sb

    verdict = {
        "rule": "lunch LOD-reclaim UP 12:30–13:30",
        "window": [w_start, w_end],
        "n_cells": len(score_rows),
        "n_arms": len(arms),
        "dual_pass_n": len(dual_pass),
        "dual_pass_top": dual_pass[:10],
        "core_overlap": "Rule-A 10:30–14:00; research overlay only",
        "verdict": "PASS" if dual_pass else "REJECT",
    }
    (out / "verdict.json").write_text(json.dumps(verdict, indent=2, default=str), encoding="utf-8")
    print("\n=== VERDICT ===", flush=True)
    print(
        f"dual_pass_n={len(dual_pass)} / cells={len(score_rows)} arms={len(arms)}",
        flush=True,
    )
    if len(top):
        cols = [
            c
            for c in [
                "name",
                "dual_pass",
                "lod_before",
                "min_wash",
                "min_bounce",
                "max_chase",
                "may_jul09_n",
                "may_jul09_mean",
                "may_jul09_day_win",
                "jul10_23_n",
                "jul10_23_mean",
                "jul10_23_day_win",
            ]
            if c in top.columns
        ]
        print(top[cols].to_string(index=False), flush=True)
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

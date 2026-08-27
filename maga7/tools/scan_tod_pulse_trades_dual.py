#!/usr/bin/env python3
"""Time-of-day pulse sleeve dual (reuse AM FO/LB rules on arbitrary TOD window).

Default research target: lunch **12:30–13:30** (V-reversal hour) vs AM champion
logic (FO |fav_from_open| / LB lookback → ATM option TP/SL on trades last±slip).

Notes
-----
- Feeds **full-session** 1m bars so FO uses true day-open (same as AM scout).
  Lunch FO often fires on the first in-window bar if morning already extended
  ("stale FO") — scored separately in diagnostics.
- CORE Rule-A owns 10:30–14:00; lunch overlaps. This tool is research-only
  (independent book), not a live wire proposal.

Example:
  PYTHONPATH=. python -m maga7.tools.scan_tod_pulse_trades_dual \\
    --window-start 12:30 --window-end 13:30 --tag research_lunch_pulse_dual \\
    --dirs DN,UP --arms FO,LB
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

from maga7.common.am_pulse_scout import am_pulse_decision_ts, parse_am_pulse_scout, scan_day
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


def _spot_from_1m(day: pd.DataFrame, ts: pd.Timestamp) -> float | None:
    if day is None or day.empty:
        return None
    t = to_ny(ts)
    sub = day[pd.to_datetime(day["timestamp"]) <= t]
    if sub.empty:
        return None
    px = float(sub.iloc[-1]["close"])
    return px if px > 0 else None


def _hhmm_min(hhmm: str) -> int:
    p = str(hhmm).split(":")
    return int(p[0]) * 60 + int(p[1])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", default=PROFILE)
    ap.add_argument("--tag", default="research_lunch_pulse_dual")
    ap.add_argument("--trades-root", default=str(DEFAULT_TRADES))
    ap.add_argument("--window-start", default="12:30")
    ap.add_argument("--window-end", default="13:30")
    ap.add_argument("--session-tag", default="LUNCH_1230_1330")
    ap.add_argument("--dirs", default="DN,UP")
    ap.add_argument("--arms", default="FO,LB")
    ap.add_argument("--fo-thr", default="0.008,0.01,0.012,0.015")
    ap.add_argument("--lb-thr", default="0.006,0.008,0.01")
    ap.add_argument("--lookback-bars", type=int, default=2)
    ap.add_argument("--tp", default="0.10,0.15,0.20")
    ap.add_argument("--sl", default="0.15,0.20,0.25")
    ap.add_argument("--max-hold-sec", type=int, default=900)
    ap.add_argument(
        "--bar-delay-sec",
        type=int,
        default=60,
        help="decision_ts = feature_ts + delay (left-labeled 1m availability)",
    )
    ap.add_argument("--slip", type=float, default=0.01)
    ap.add_argument("--position-frac", type=float, default=0.10)
    ap.add_argument("--max-concurrent", type=int, default=2)
    ap.add_argument("--cooldown-minutes", type=float, default=10.0)
    ap.add_argument("--min-n", type=int, default=8)
    ap.add_argument("--min-day-win", type=float, default=0.55)
    # Champion-only smoke: skip full grid
    ap.add_argument(
        "--champions-only",
        action="store_true",
        help="Only score FO@0.8% TP15/SL20 and LB@0.8% TP15/SL20 (both dirs).",
    )
    args = ap.parse_args(argv)

    w_start = str(args.window_start)
    w_end = str(args.window_end)
    session = str(args.session_tag)
    dirs = {x.strip().upper() for x in args.dirs.split(",") if x.strip()}
    want_arms = {x.strip().upper() for x in args.arms.split(",") if x.strip()}
    fo_thrs = [float(x) for x in args.fo_thr.split(",") if x.strip()]
    lb_thrs = [float(x) for x in args.lb_thr.split(",") if x.strip()]
    tps = [float(x) for x in args.tp.split(",") if x.strip()]
    sls = [float(x) for x in args.sl.split(",") if x.strip()]
    if args.champions_only:
        fo_thrs, lb_thrs, tps, sls = [0.008], [0.008], [0.15], [0.20]
    bar_delay_sec = max(0, int(args.bar_delay_sec))

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
        f"tod_pulse {session} {w_start}..{w_end} {start_all}..{end_all} "
        f"days={len(dates)} dirs={sorted(dirs)} arms={sorted(want_arms)}",
        flush=True,
    )

    stock_by_sym: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        sdf = load_stock_month_files(stock_root, sym, months)
        if sdf is not None and not sdf.empty:
            stock_by_sym[sym] = sdf

    arms: list[dict[str, Any]] = []
    diag_rows: list[dict[str, Any]] = []
    win0 = _hhmm_min(w_start)

    for di, date in enumerate(dates):
        if di % 10 == 0:
            print(f"[day] {date} ({di+1}/{len(dates)}) arms={len(arms)}", flush=True)
        for sym in symbols:
            sdf = stock_by_sym.get(sym)
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
            ts_ns = px = None
            if day1s is not None and not day1s.empty:
                ts_ns, px = _stock_arrays(day1s)

            def _emit(a, thr: float, arm_name: str) -> None:
                arm_ts = to_ny(pd.Timestamp(a.ts))
                decision_ts = am_pulse_decision_ts(
                    arm_ts, delay_seconds=bar_delay_sec
                )
                hm = arm_ts.hour * 60 + arm_ts.minute
                stale = bool(arm_name == "FO" and hm == win0)
                spot = None
                if ts_ns is not None and px is not None:
                    spot = _spot_at_arr(ts_ns, px, arm_ts)
                if spot is None:
                    spot = _spot_from_1m(day1m, arm_ts)
                ticker, dte, _ = resolve_open_lock_contract(
                    by_dte,
                    direction=a.dir,
                    moneyness="ATM",
                    spot=spot,
                    prefer_dte=0,
                    allowed_dte=[0, 1, 2],
                    clear_otm_thresh=0.01,
                    ladder=True,
                    otm_rungs=otm,
                )
                if not ticker:
                    return
                arr = tpaths.get(str(ticker).replace("O:", ""))
                if arr is None:
                    return
                arms.append(
                    {
                        "date": date,
                        "symbol": sym,
                        "dir": a.dir,
                        "arm": arm_name,
                        "thr": float(thr),
                        "lookback_bars": int(args.lookback_bars),
                        "session": session,
                        "arm_ts": arm_ts,
                        "decision_ts": decision_ts,
                        "fav_from_open": float(a.fav_from_open),
                        "lookback_ret": a.lookback_ret,
                        "chase": a.chase,
                        "stale_fo_window_open": stale,
                        "ticker": ticker,
                        "dte": dte,
                        "pts": arr[0],
                        "plast": arr[1],
                    }
                )
                diag_rows.append(
                    {
                        "date": date,
                        "symbol": sym,
                        "dir": a.dir,
                        "arm": arm_name,
                        "thr": float(thr),
                        "arm_ts": str(arm_ts),
                        "decision_ts": str(decision_ts),
                        "fav_from_open": float(a.fav_from_open),
                        "lookback_ret": a.lookback_ret,
                        "chase": a.chase,
                        "stale_fo_window_open": stale,
                    }
                )

            if "FO" in want_arms:
                for thr in fo_thrs:
                    cfg = parse_am_pulse_scout(
                        {
                            "enabled": True,
                            "window_start": w_start,
                            "window_end": w_end,
                            "min_fav_from_open": thr,
                            "lookback_bars": int(args.lookback_bars),
                            "min_lookback_ret": 0.99,
                            "dirs": sorted(dirs),
                            "max_alerts_per_symbol": 1,
                        }
                    )
                    for a in scan_day(day1m, date=date, symbol=sym, cfg=cfg):
                        if a.arm == "FO" and a.dir in dirs:
                            _emit(a, thr, "FO")
            if "LB" in want_arms:
                for thr in lb_thrs:
                    cfg = parse_am_pulse_scout(
                        {
                            "enabled": True,
                            "window_start": w_start,
                            "window_end": w_end,
                            "min_fav_from_open": 0.99,
                            "lookback_bars": int(args.lookback_bars),
                            "min_lookback_ret": thr,
                            "dirs": sorted(dirs),
                            "max_alerts_per_symbol": 1,
                        }
                    )
                    for a in scan_day(day1m, date=date, symbol=sym, cfg=cfg):
                        if a.arm == "LB" and a.dir in dirs:
                            _emit(a, thr, "LB")

    print(f"arms={len(arms)}; scoring cells…", flush=True)
    if diag_rows:
        pd.DataFrame(diag_rows).to_csv(out / "arms_diag.csv", index=False)
        ddf = pd.DataFrame(diag_rows)
        fo = ddf[ddf.arm == "FO"]
        stale_rate = float(fo["stale_fo_window_open"].mean()) if len(fo) else None
        arm_dir = {}
        if len(ddf):
            for (arm_k, dir_k), n in ddf.groupby(["arm", "dir"]).size().items():
                arm_dir[f"{arm_k}:{dir_k}"] = int(n)
        (out / "diagnostics.json").write_text(
            json.dumps(
                {
                    "session": session,
                    "window": [w_start, w_end],
                    "n_arm_rows": len(ddf),
                    "n_fo": int((ddf.arm == "FO").sum()),
                    "n_lb": int((ddf.arm == "LB").sum()),
                    "fo_stale_window_open_rate": stale_rate,
                    "dir_counts": {
                        str(k): int(v) for k, v in ddf.groupby("dir").size().items()
                    }
                    if len(ddf)
                    else {},
                    "arm_dir": arm_dir,
                },
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        print(
            f"diag FO stale@window_open={stale_rate} "
            f"n_fo={(ddf.arm=='FO').sum()} n_lb={(ddf.arm=='LB').sum()}",
            flush=True,
        )

    cells: list[dict[str, Any]] = []
    if "FO" in want_arms:
        for thr in fo_thrs:
            for tp in tps:
                for sl in sls:
                    cells.append(
                        {
                            "name": f"pulse_FO_t{thr}_tp{tp}_sl{sl}",
                            "arm": "FO",
                            "thr": thr,
                            "tp": tp,
                            "sl": sl,
                        }
                    )
    if "LB" in want_arms:
        for thr in lb_thrs:
            for tp in tps:
                for sl in sls:
                    cells.append(
                        {
                            "name": f"pulse_LB_t{thr}_lb{args.lookback_bars}_tp{tp}_sl{sl}",
                            "arm": "LB",
                            "thr": thr,
                            "tp": tp,
                            "sl": sl,
                        }
                    )

    dual_pass: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []

    for ci, cell in enumerate(cells):
        if ci % 20 == 0:
            print(f"[cell] {ci+1}/{len(cells)} dual_so_far={len(dual_pass)}", flush=True)
        win_raw: dict[str, list[dict]] = {w[0]: [] for w in WINDOWS}
        for arm in arms:
            if str(arm["arm"]) != str(cell["arm"]):
                continue
            if float(arm["thr"]) != float(cell["thr"]):
                continue
            wname = _window_of(str(arm["date"]))
            if wname is None:
                continue
            entry_ts = arm["decision_ts"]
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
                    "dir": arm["dir"],
                    "session": arm["session"],
                    "feature_ts": to_ny(arm["arm_ts"]),
                    "entry_ts": et,
                    "exit_ts": xt,
                    "ret": float(sim["ret"]),
                    "exit_reason": str(sim["reason"]),
                    "hold_sec": float(sim["hold_sec"]),
                    "stale_fo_window_open": bool(arm.get("stale_fo_window_open")),
                    "size": float(args.position_frac),
                }
            )

        win_stats: dict[str, dict[str, Any]] = {}
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
            st = _stats(sized)
            stale_n = int(sum(1 for r in sized if r.get("stale_fo_window_open")))
            st["stale_fo_n"] = stale_n
            st["stale_fo_frac"] = stale_n / max(1, len(sized))
            if sized:
                tdf = pd.DataFrame(sized)
                for d in ("UP", "DN"):
                    g = tdf[tdf["dir"] == d]
                    if len(g):
                        st[f"n_{d}"] = int(len(g))
                        st[f"mean_{d}"] = float(g["ret"].mean())
            win_stats[wname] = st

        both = True
        for wname, _, _ in WINDOWS:
            mn = int(args.min_n)
            if wname == "jul10_23":
                mn = min(mn, 6)
            if not _ok(win_stats[wname], min_n=mn, min_day_win=float(args.min_day_win)):
                both = False
                break

        row: dict[str, Any] = {
            "name": cell["name"],
            "arm": cell["arm"],
            "thr": cell["thr"],
            "tp": cell["tp"],
            "sl": cell["sl"],
            "session": session,
            "window_start": w_start,
            "window_end": w_end,
            "dual_pass": both,
        }
        for wname, _, _ in WINDOWS:
            for k, v in win_stats[wname].items():
                row[f"{wname}_{k}"] = v
        score_rows.append(row)
        if both:
            dual_pass.append(row)
            print(
                f"  *** DUAL PASS {cell['name']} "
                f"MJ09 n={row.get('may_jul09_n')} mean={row.get('may_jul09_mean')} "
                f"J10 n={row.get('jul10_23_n')} mean={row.get('jul10_23_mean')}",
                flush=True,
            )

    sb = pd.DataFrame(score_rows)
    sb.to_csv(out / "scoreboard.csv", index=False)
    (out / "dual_pass.json").write_text(json.dumps(dual_pass, indent=2, default=str), encoding="utf-8")

    # Highlight AM-champion twin on this TOD
    champ = "pulse_FO_t0.008_tp0.15_sl0.2"
    champ_row = next((r for r in score_rows if r["name"] == champ), None)
    am_ref = {
        "name": champ,
        "am_may_jul09": {"n": 38, "mean": 0.145, "day_win": 0.90},
        "am_jul10_23": {"n": 6, "mean": 0.233, "day_win": 1.00},
        "note": "AM champion from docs/am_pulse_scout.md (DN-only quote dual).",
    }
    verdict = {
        "session": session,
        "window": [w_start, w_end],
        "bar_delay_sec": int(bar_delay_sec),
        "entry_anchor": "decision_ts=feature_ts+bar_delay_sec",
        "n_cells": len(score_rows),
        "dual_pass_n": len(dual_pass),
        "dual_pass_names": [r["name"] for r in dual_pass[:30]],
        "am_champion_twin": champ_row,
        "am_reference": am_ref,
        "core_overlap": "10:30–14:00 Rule-A owns this clock; lunch is research overlay only.",
        "verdict": (
            "PASS_LIKE_AM"
            if champ_row and champ_row.get("dual_pass")
            else (
                "PARTIAL"
                if dual_pass
                else "REJECT_NO_AM_FEATURE"
            )
        ),
    }
    (out / "verdict.json").write_text(json.dumps(verdict, indent=2, default=str), encoding="utf-8")
    print("\n=== VERDICT ===", flush=True)
    print(json.dumps(verdict, indent=2, default=str)[:2500], flush=True)
    if len(sb):
        top = sb.sort_values(
            ["dual_pass", "may_jul09_mean"] if "may_jul09_mean" in sb.columns else ["dual_pass"],
            ascending=[False, False],
        ).head(12)
        cols = [
            c
            for c in [
                "name",
                "dual_pass",
                "may_jul09_n",
                "may_jul09_mean",
                "may_jul09_day_win",
                "may_jul09_stale_fo_frac",
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

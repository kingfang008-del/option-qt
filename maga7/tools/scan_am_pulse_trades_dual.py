#!/usr/bin/env python3
"""A-window AM pulse scout → trades last±slip TP/SL dual-window.

Defaults come from profile ``am_pulse`` (LOCK: 09:30–10:30, flatten 10:45):
  FO — first |fav_from_open| ≥ thr
  LB — first |ret over lookback_bars| ≥ thr

Independent of Rule-A. Promote only after quote FillSpec dual.

Example:
  PYTHONPATH=. python -m maga7.tools.scan_am_pulse_trades_dual \\
    --tag research_am_pulse_trades_dual --dirs DN
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

from maga7.common.am_pulse_scout import (
    load_am_pulse_lane_cfg,
    parse_am_pulse_scout,
    scan_day,
)
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
SESSION = "AM_0930_1030"


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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", default=PROFILE)
    ap.add_argument("--tag", default="research_am_pulse_trades_dual")
    ap.add_argument("--trades-root", default=str(DEFAULT_TRADES))
    ap.add_argument("--dirs", default="", help="Empty = profile am_pulse directions")
    ap.add_argument("--arms", default="FO,LB", help="FO and/or LB")
    ap.add_argument("--fo-thr", default="0.008,0.01,0.012,0.015")
    ap.add_argument("--lb-thr", default="0.006,0.008,0.01")
    ap.add_argument("--lookback-bars", type=int, default=2)
    ap.add_argument("--tp", default="0.10,0.15,0.20,0.25")
    ap.add_argument("--sl", default="0.12,0.15,0.20,0.25")
    ap.add_argument("--max-hold-sec", type=int, default=0, help="0 = profile am_pulse value")
    ap.add_argument("--slip", type=float, default=0.01)
    ap.add_argument("--position-frac", type=float, default=0.10)
    ap.add_argument("--max-concurrent", type=int, default=2)
    ap.add_argument("--cooldown-minutes", type=float, default=10.0)
    ap.add_argument("--min-n", type=int, default=8)
    ap.add_argument("--min-day-win", type=float, default=0.55)
    args = ap.parse_args(argv)

    prof = load_profile(args.profile)
    lane_cfg = load_am_pulse_lane_cfg(prof, "am_pulse")
    dirs_spec = args.dirs or ",".join(lane_cfg.get("dirs") or ["DN", "UP"])
    dirs = {x.strip().upper() for x in dirs_spec.split(",") if x.strip()}
    want_arms = {x.strip().upper() for x in args.arms.split(",") if x.strip()}
    fo_thrs = [float(x) for x in args.fo_thr.split(",") if x.strip()]
    lb_thrs = [float(x) for x in args.lb_thr.split(",") if x.strip()]
    tps = [float(x) for x in args.tp.split(",") if x.strip()]
    sls = [float(x) for x in args.sl.split(",") if x.strip()]
    window_start = str(lane_cfg.get("window_start") or "09:30")
    window_end = str(lane_cfg.get("window_end") or "10:30")
    flatten_before = str(lane_cfg.get("flatten_before") or "").strip()
    max_fo = float(lane_cfg.get("max_fav_from_open", 0.0) or 0.0)
    max_hold_sec = (
        int(args.max_hold_sec)
        if int(args.max_hold_sec) > 0
        else int(lane_cfg.get("max_hold_sec", 900) or 900)
    )
    prefer_dte = int(lane_cfg.get("prefer_dte", 0) or 0)
    allowed_raw = lane_cfg.get("allowed_dte") or (prof.get("lock") or {}).get(
        "allowed_dte"
    ) or [0, 1, 2]
    allowed_dte = [int(x) for x in allowed_raw]
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
        f"am_pulse trades {start_all}..{end_all} days={len(dates)} "
        f"dirs={sorted(dirs)} arms={sorted(want_arms)}",
        flush=True,
    )

    stock_by_sym: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        sdf = load_stock_month_files(stock_root, sym, months)
        if sdf is not None and not sdf.empty:
            stock_by_sym[sym] = sdf

    arms: list[dict[str, Any]] = []
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
            # Optional 1s spot for finer ATM; fallback to 1m close at arm.
            day1s = None
            try:
                from maga7.common.bar_agg import load_stock_1s_day

                day1s = load_stock_1s_day(stock_1s, sym, date)
            except Exception:
                day1s = None
            ts_ns = px = None
            if day1s is not None and not day1s.empty:
                ts_ns, px = _stock_arrays(day1s)

            # FO grid
            if "FO" in want_arms:
                for thr in fo_thrs:
                    cfg = parse_am_pulse_scout(
                        {
                            "enabled": True,
                            "window_start": window_start,
                            "window_end": window_end,
                            "min_fav_from_open": thr,
                            "max_fav_from_open": max_fo,
                            "lookback_bars": int(args.lookback_bars),
                            "min_lookback_ret": 0.99,  # disable LB for this pass
                            "dirs": sorted(dirs),
                            "max_alerts_per_symbol": 1,
                        }
                    )
                    for a in scan_day(day1m, date=date, symbol=sym, cfg=cfg):
                        if a.arm != "FO" or a.dir not in dirs:
                            continue
                        arm_ts = to_ny(pd.Timestamp(a.ts))
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
                            prefer_dte=prefer_dte,
                            allowed_dte=allowed_dte,
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
                                "dir": a.dir,
                                "arm": "FO",
                                "thr": float(thr),
                                "lookback_bars": int(args.lookback_bars),
                                "session": SESSION,
                                "arm_ts": arm_ts,
                                "fav_from_open": float(a.fav_from_open),
                                "lookback_ret": a.lookback_ret,
                                "ticker": ticker,
                                "dte": dte,
                                "pts": arr[0],
                                "plast": arr[1],
                            }
                        )

            # LB grid
            if "LB" in want_arms:
                for thr in lb_thrs:
                    cfg = parse_am_pulse_scout(
                        {
                            "enabled": True,
                            "window_start": window_start,
                            "window_end": window_end,
                            "min_fav_from_open": 0.99,  # disable FO
                            "max_fav_from_open": max_fo,
                            "lookback_bars": int(args.lookback_bars),
                            "min_lookback_ret": thr,
                            "dirs": sorted(dirs),
                            "max_alerts_per_symbol": 1,
                        }
                    )
                    for a in scan_day(day1m, date=date, symbol=sym, cfg=cfg):
                        if a.arm != "LB" or a.dir not in dirs:
                            continue
                        arm_ts = to_ny(pd.Timestamp(a.ts))
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
                            prefer_dte=prefer_dte,
                            allowed_dte=allowed_dte,
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
                                "dir": a.dir,
                                "arm": "LB",
                                "thr": float(thr),
                                "lookback_bars": int(args.lookback_bars),
                                "session": SESSION,
                                "arm_ts": arm_ts,
                                "fav_from_open": float(a.fav_from_open),
                                "lookback_ret": a.lookback_ret,
                                "ticker": ticker,
                                "dte": dte,
                                "pts": arr[0],
                                "plast": arr[1],
                            }
                        )

    print(f"arms={len(arms)}; scoring cells…", flush=True)
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
                            "lookback_bars": int(args.lookback_bars),
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
                            "lookback_bars": int(args.lookback_bars),
                            "tp": tp,
                            "sl": sl,
                        }
                    )

    dual_pass: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []
    trade_dump: dict[str, pd.DataFrame] = {}

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
            entry_ts = arm["arm_ts"]
            hold_sec = max_hold_sec
            if flatten_before:
                flat_ts = pd.Timestamp(
                    f"{arm['date']} {flatten_before}", tz=NY
                )
                hold_sec = min(
                    hold_sec,
                    max(
                        1,
                        int((flat_ts - to_ny(entry_ts)).total_seconds()),
                    ),
                )
            sim = simulate_trade_tpsl(
                arm["pts"],
                arm["plast"],
                entry_ts,
                tp=float(cell["tp"]),
                sl=float(cell["sl"]),
                max_hold_sec=hold_sec,
                slip=float(args.slip),
            )
            if sim is None or not np.isfinite(sim["ret"]):
                continue
            et = to_ny(entry_ts)
            win_raw[wname].append(
                {
                    "date": arm["date"],
                    "symbol": arm["symbol"],
                    "dir": arm["dir"],
                    "session": arm["session"],
                    "entry_ts": str(et),
                    "exit_ts": str(et + pd.Timedelta(seconds=sim["hold_sec"])),
                    "ticker": arm["ticker"],
                    "dte": arm["dte"],
                    "ret": sim["ret"],
                    "exit_reason": sim["reason"],
                    "hold_sec": sim["hold_sec"],
                    "fav_from_open": arm.get("fav_from_open"),
                    "lookback_ret": arm.get("lookback_ret"),
                    "cell": cell["name"],
                    "event_source": "am_pulse_sleeve",
                    "window": wname,
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
            st = _stats(sized)
            if sized:
                tdf = pd.DataFrame(sized)
                for d in ("UP", "DN"):
                    g = tdf[tdf["dir"] == d]
                    if len(g):
                        st[f"n_{d}"] = int(len(g))
                        st[f"mean_{d}"] = float(g["ret"].mean())
            win_stats[wname] = st
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
            trade_dump[cell["name"]] = pd.DataFrame(sized_all)
            print(
                f"  *** DUAL PASS {cell['name']} "
                f"MJ09 n={row.get('may_jul09_n')} mean={row.get('may_jul09_mean'):+.3f} "
                f"J10 n={row.get('jul10_23_n')} mean={row.get('jul10_23_mean'):+.3f}",
                flush=True,
            )

    score = pd.DataFrame(score_rows)
    score.to_csv(out / "scoreboard.csv", index=False)
    dual_pass = sorted(
        dual_pass,
        key=lambda r: (
            float(r.get("may_jul09_add") or 0) + float(r.get("jul10_23_add") or 0)
        ),
        reverse=True,
    )
    for i, p in enumerate(dual_pass[:15]):
        name = p["name"]
        if name in trade_dump and len(trade_dump[name]):
            trade_dump[name].to_csv(out / f"trades_dual{i:02d}_{name}.csv", index=False)

    summary = {
        "expert_kind": "am_pulse_sleeve",
        "isolation": "independent of Mag7 Rule-A; signal_end=10:25 CORE mutex",
        "session": SESSION,
        "windows": [list(w) for w in WINDOWS],
        "pricing": "option_trades_last_slip",
        "dirs": sorted(dirs),
        "arms": sorted(want_arms),
        "n_arms": int(len(arms)),
        "n_cells": int(len(cells)),
        "dual_pass_n": int(len(dual_pass)),
        "verdict": "PASS" if dual_pass else "REJECT",
        "champion": dual_pass[0] if dual_pass else None,
        "note": "Trades dual only — promote after quote FillSpec dual.",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    (out / "dual_pass.json").write_text(
        json.dumps(dual_pass[:50], indent=2, default=str), encoding="utf-8"
    )
    # champions list for quote tool
    champs = [
        {
            "name": p["name"],
            "arm": p["arm"],
            "thr": p["thr"],
            "lookback_bars": p.get("lookback_bars", args.lookback_bars),
            "tp": p["tp"],
            "sl": p["sl"],
        }
        for p in dual_pass[:12]
    ]
    (out / "champions.json").write_text(json.dumps(champs, indent=2), encoding="utf-8")

    print("\n=== verdict", summary["verdict"], "dual_pass_n=", len(dual_pass), flush=True)
    if dual_pass:
        c = dual_pass[0]
        print(
            f"champion {c['name']}: "
            f"MJ09 n={c.get('may_jul09_n')} mean={c.get('may_jul09_mean')} "
            f"day_win={c.get('may_jul09_day_win')} | "
            f"J10 n={c.get('jul10_23_n')} mean={c.get('jul10_23_mean')} "
            f"day_win={c.get('jul10_23_day_win')}",
            flush=True,
        )
    elif not score.empty:
        score["_sum"] = score["may_jul09_add"].fillna(0) + score["jul10_23_add"].fillna(0)
        near = score.sort_values("_sum", ascending=False).head(12)
        cols = [
            c
            for c in [
                "name",
                "may_jul09_n",
                "may_jul09_mean",
                "may_jul09_day_win",
                "may_jul09_add",
                "jul10_23_n",
                "jul10_23_mean",
                "jul10_23_day_win",
                "jul10_23_add",
            ]
            if c in near.columns
        ]
        print(near[cols].to_string(index=False), flush=True)
    print(f"wrote {out}", flush=True)
    return 0 if dual_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())

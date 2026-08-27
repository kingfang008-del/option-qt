#!/usr/bin/env python3
"""Independent impulse scout (侦查兵): fast dump/rally → ATM option TP/SL.

Isolated from Mag7 Rule-A / Hunt. No baseline-miss filter.
Arm: first time in session ``|stock_ret over lookback| >= thr``.
Enter immediately on arm (trades last ± slip).

Dual windows: may_jul09 / jul10_23 (same gate family as certainty morph).

Example:
  PYTHONPATH=. python -m maga7.tools.scan_impulse_scout_tpsl \\
    --tag research_impulse_scout_tpsl_dual
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
from maga7.common.option_trade_tpsl import simulate_trade_tpsl
from maga7.common.option_trades import load_option_trades
from maga7.common.replay import to_ny
from maga7.common.stock_1s import session_dates
from maga7.tools.run_morning_sec_option_fill import _portfolio_day
from maga7.tools.scan_am_certainty_morph_tpsl import _ok, _stats
from maga7.tools.scan_session_horizon_foresight import (
    _paths_by_ticker,
    _spot_at_arr,
    _stock_arrays,
    _stock_dir_arr,
)

NY = "America/New_York"
DEFAULT_TRADES = Path("/mnt/s990/new_option_data_s3_trades")
PROFILE = (
    "maga7/CONFIG/strategy_profiles/"
    "single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1.json"
)

SESSIONS = (
    ("AM_0935_1030", "09:35", "10:30"),
    ("CORE_1030_1200", "10:30", "12:00"),
    ("MID_1200_1400", "12:00", "14:00"),
    ("PM_1400_1530", "14:00", "15:30"),
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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", default=PROFILE)
    ap.add_argument("--tag", default="research_impulse_scout_tpsl_dual")
    ap.add_argument("--trades-root", default=str(DEFAULT_TRADES))
    ap.add_argument(
        "--sessions",
        default="AM_0935_1030,CORE_1030_1200,MID_1200_1400,PM_1400_1530",
    )
    ap.add_argument("--dirs", default="DN,UP")
    ap.add_argument("--thr", default="0.003,0.005,0.008")
    ap.add_argument("--lookback-sec", default="60,120")
    ap.add_argument("--tp", default="0.15,0.20,0.25")
    ap.add_argument("--sl", default="0.12,0.15,0.20")
    ap.add_argument("--max-hold-sec", type=int, default=900)
    ap.add_argument("--slip", type=float, default=0.01)
    ap.add_argument("--stride-sec", type=int, default=30)
    ap.add_argument("--position-frac", type=float, default=0.10)
    ap.add_argument("--max-concurrent", type=int, default=2)
    ap.add_argument("--cooldown-minutes", type=float, default=10.0)
    ap.add_argument("--min-n", type=int, default=8)
    ap.add_argument("--min-day-win", type=float, default=0.55)
    args = ap.parse_args(argv)

    want_sess = {x.strip() for x in args.sessions.split(",") if x.strip()}
    sessions = tuple(s for s in SESSIONS if s[0] in want_sess)
    dirs = {x.strip().upper() for x in args.dirs.split(",") if x.strip()}
    thrs = [float(x) for x in args.thr.split(",") if x.strip()]
    lookbacks = [int(x) for x in args.lookback_sec.split(",") if x.strip()]
    tps = [float(x) for x in args.tp.split(",") if x.strip()]
    sls = [float(x) for x in args.sl.split(",") if x.strip()]

    prof = load_profile(args.profile)
    paths = prof["_paths"]
    symbols = list(prof.get("symbols") or [])
    stock_1s = Path(paths.get("stock_1s_root") or "/mnt/s990/data/raw_1s/stocks").expanduser()
    trades_root = Path(args.trades_root)
    lock = load_multidte_lock_index(Path(paths["open_locked_map"]).expanduser())
    otm = resolve_otm_rungs(prof, default=3)
    out = Path(paths["results_dir"]) / args.tag
    out.mkdir(parents=True, exist_ok=True)

    start_all = min(w[1] for w in WINDOWS)
    end_all = max(w[2] for w in WINDOWS)
    dates = session_dates(start_all, end_all)
    print(
        f"impulse scout {start_all}..{end_all} days={len(dates)} "
        f"sess={[s[0] for s in sessions]} dirs={sorted(dirs)}",
        flush=True,
    )

    arms: list[dict[str, Any]] = []
    for di, date in enumerate(dates):
        if di % 10 == 0:
            print(f"[day] {date} ({di+1}/{len(dates)}) arms={len(arms)}", flush=True)
        for sym in symbols:
            day = load_stock_1s_day(stock_1s, sym, date)
            if day is None or day.empty:
                continue
            tday = load_option_trades(trades_root, sym, date)
            if tday is None or tday.empty:
                continue
            tpaths = _paths_by_ticker(tday)
            if not tpaths:
                continue
            ts_ns, px = _stock_arrays(day)
            by_dte = lock.get((sym, date))
            if not by_dte:
                continue
            for sess_name, s0, s1 in sessions:
                for lb in lookbacks:
                    t_start = pd.Timestamp(f"{date} {s0}:00", tz=NY) + pd.Timedelta(
                        seconds=int(lb)
                    )
                    t_end = pd.Timestamp(f"{date} {s1}:00", tz=NY)
                    fired: set[tuple[str, float, int]] = set()
                    t = t_start
                    stride = pd.Timedelta(seconds=int(args.stride_sec))
                    while t < t_end:
                        for thr in thrs:
                            direction, sr = _stock_dir_arr(
                                ts_ns, px, t, int(lb), float(thr)
                            )
                            if direction is None or direction not in dirs:
                                continue
                            key = (direction, float(thr), int(lb))
                            if key in fired:
                                continue
                            spot = _spot_at_arr(ts_ns, px, t)
                            ticker, dte, _ = resolve_open_lock_contract(
                                by_dte,
                                direction=direction,
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
                            fired.add(key)
                            arms.append(
                                {
                                    "date": date,
                                    "symbol": sym,
                                    "dir": direction,
                                    "thr": float(thr),
                                    "lookback_sec": int(lb),
                                    "session": sess_name,
                                    "arm_ts": to_ny(t),
                                    "stock_ret_lb": float(sr),
                                    "ticker": ticker,
                                    "dte": dte,
                                    "pts": arr[0],
                                    "plast": arr[1],
                                }
                            )
                        t += stride

    print(f"arms={len(arms)}; scoring cells…", flush=True)
    cells: list[dict[str, Any]] = []
    for thr in thrs:
        for lb in lookbacks:
            for tp in tps:
                for sl in sls:
                    cells.append(
                        {
                            "name": f"imp_t{thr}_lb{lb}_tp{tp}_sl{sl}",
                            "morph": "impulse",
                            "thr": thr,
                            "lookback_sec": lb,
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
            if float(arm["thr"]) != float(cell["thr"]):
                continue
            if int(arm["lookback_sec"]) != int(cell["lookback_sec"]):
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
                    "stock_ret_lb": arm["stock_ret_lb"],
                    "cell": cell["name"],
                    "event_source": "impulse_scout",
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
                for sess in tdf["session"].unique():
                    g = tdf[tdf.session == sess]
                    st[f"n_{sess}"] = int(len(g))
                    st[f"mean_{sess}"] = float(g["ret"].mean())
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
        "expert_kind": "impulse_scout",
        "isolation": "independent of Mag7 Rule-A / Hunt; no baseline-miss filter",
        "sessions": [s[0] for s in sessions],
        "windows": [list(w) for w in WINDOWS],
        "pricing": "option_trades_last_slip",
        "n_arms": int(len(arms)),
        "n_cells": int(len(cells)),
        "dual_pass_n": int(len(dual_pass)),
        "verdict": "PASS" if dual_pass else "REJECT",
        "champion": dual_pass[0] if dual_pass else None,
        "note": (
            "Scout: first |ret_lb|>=thr in session → immediate ATM option entry. "
            "Promote only after quote FillSpec dual (not done in this trades scan)."
        ),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    (out / "dual_pass.json").write_text(
        json.dumps(dual_pass[:50], indent=2, default=str), encoding="utf-8"
    )
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
    else:
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

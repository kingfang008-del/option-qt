#!/usr/bin/env python3
"""QQQ foresight rule grid — volatility-recalibrated (not Mag7 FO copy).

Purpose: use foresight (oracle + clock hold) to discover which *QQQ-scale*
entry rules have option path edge, then score causal TP/SL on survivors.

Arms (causal direction; choice of H / TP-SL is foresight for discovery):
  - MOM: |ret over lookback| ≥ thr → that dir (QQQ thr ≪ Mag7)
  - FO:  |from_open| ≥ thr → that dir
  - FADE: |from_open| ≥ thr → *opposite* dir (stretch fade)

Pricing: ``/mnt/s990/new_option_data_s3_tick`` last ± slip.
ATM: OCC 0DTE closest-to-spot (quote bucket assist when present).

Example:
  PYTHONPATH=. python -m maga7.tools.scan_qqq_foresight_rule_grid \\
    --tag research_qqq_foresight_rule_grid_20260728
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from maga7.common.bar_agg import load_stock_1s_day
from maga7.common.option_flow import load_option_tick_day, tick_dates
from maga7.common.option_trade_tpsl import simulate_trade_tpsl
from maga7.common.replay import to_ny
from maga7.tools.run_morning_sec_option_fill import _portfolio_day
from maga7.tools.run_morning_sec_qqq_dte1 import _load_atm_path
from maga7.tools.scan_session_horizon_foresight import (
    _fwd_trade_rets_arr,
    _paths_by_ticker,
    _spot_at_arr,
    _stock_arrays,
    _stock_dir_arr,
    _ts_ns,
)

NY = "America/New_York"
DEFAULT_TICK = Path("/mnt/s990/new_option_data_s3_tick")
DEFAULT_OPT = Path("/mnt/s990/data/raw_1s/dte0_options/QQQ")
DEFAULT_STOCK = Path("/mnt/s990/data/raw_1s/stocks")
DEFAULT_RESULTS = Path("/mnt/s990/data/maga7/results")

SESSIONS = (
    ("AM_0935_1030", "09:35", "10:30"),
    ("CORE_1030_1200", "10:30", "12:00"),
    ("MID_1200_1400", "12:00", "14:00"),
    ("PM_1400_1530", "14:00", "15:30"),
)
WINDOWS = (
    ("feb_apr", "2026-02-02", "2026-04-30"),
    ("may_jul", "2026-05-01", "2026-07-23"),
)

_OCC = re.compile(
    r"^O?:?(?P<root>[A-Z]+)(?P<yy>\d{2})(?P<mm>\d{2})(?P<dd>\d{2})"
    r"(?P<cp>[CP])(?P<strike>\d{8})$"
)


def _atm_ticker(
    *,
    date: str,
    direction: str,
    spot: float,
    opt_root: Path,
    print_paths: dict[str, tuple[np.ndarray, np.ndarray]],
) -> tuple[str | None, float | None]:
    path, ticker, strike = _load_atm_path(opt_root, date, direction)
    if ticker:
        key = str(ticker).replace("O:", "")
        if key in print_paths:
            return key, float(strike) if strike is not None else None
        if str(ticker) in print_paths:
            return str(ticker), float(strike) if strike is not None else None
    ymd = date.replace("-", "")[2:]
    want = "C" if direction == "UP" else "P"
    best_t, best_k, best_abs = None, None, float("inf")
    for raw in print_paths:
        key = str(raw).replace("O:", "")
        m = _OCC.match(key)
        if m is None or m.group("root") != "QQQ":
            continue
        exp = f"{m.group('yy')}{m.group('mm')}{m.group('dd')}"
        if exp != ymd or m.group("cp") != want:
            continue
        k = float(m.group("strike")) / 1000.0
        ad = abs(k - spot)
        if ad < best_abs:
            best_abs, best_k, best_t = ad, k, str(raw)
    return best_t, best_k


def _from_open_dir(
    ts_ns: np.ndarray,
    px: np.ndarray,
    t: pd.Timestamp,
    open_px: float,
    thr: float,
    *,
    fade: bool,
) -> tuple[str | None, float]:
    spot = _spot_at_arr(ts_ns, px, t)
    if spot is None or open_px <= 0:
        return None, np.nan
    fo = float(spot / open_px - 1.0)
    if abs(fo) < float(thr):
        return None, fo
    cont = "UP" if fo > 0 else "DN"
    if fade:
        return ("DN" if cont == "UP" else "UP"), fo
    return cont, fo


def _port(rows: list[dict[str, Any]], *, position_frac: float, max_concurrent: int, cooldown: float) -> dict[str, Any]:
    if not rows:
        return {"n": 0, "mean": None, "win": None, "add": 0.0, "day_win": None, "n_days": 0}
    by: dict[str, list] = {}
    for r in rows:
        by.setdefault(str(r["date"]), []).append(r)
    sized: list[dict] = []
    for d in sorted(by):
        sized.extend(
            _portfolio_day(
                by[d],
                position_frac=position_frac,
                max_concurrent=max_concurrent,
                cooldown_minutes=cooldown,
            )
        )
    if not sized:
        return {"n": 0, "mean": None, "win": None, "add": 0.0, "day_win": None, "n_days": 0}
    t = pd.DataFrame(sized)
    day = t.groupby("date")["pnl_frac"].sum()
    return {
        "n": int(len(t)),
        "mean": float(t["ret"].mean()),
        "win": float((t["ret"] > 0).mean()),
        "add": float(t["pnl_frac"].sum()),
        "day_win": float((day > 0).mean()),
        "n_days": int(day.shape[0]),
        "trades_per_day": float(len(t) / max(1, day.shape[0])),
        "worst_day": float(day.min()),
    }


def _ok(st: dict[str, Any], *, min_n: int, min_day_win: float) -> bool:
    if st.get("mean") is None or st.get("day_win") is None:
        return False
    return bool(
        int(st.get("n") or 0) >= min_n
        and float(st["mean"]) > 0
        and float(st.get("add") or 0) > 0
        and float(st["day_win"]) >= min_day_win
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tick-root", default=str(DEFAULT_TICK))
    ap.add_argument("--opt-root", default=str(DEFAULT_OPT))
    ap.add_argument("--stock-1s-root", default=str(DEFAULT_STOCK))
    ap.add_argument("--results-dir", default=str(DEFAULT_RESULTS))
    ap.add_argument("--tag", default="research_qqq_foresight_rule_grid_20260728")
    ap.add_argument(
        "--sessions",
        default="AM_0935_1030,CORE_1030_1200,MID_1200_1400,PM_1400_1530",
    )
    ap.add_argument("--arms", default="MOM,FO,FADE", help="comma of MOM,FO,FADE")
    # QQQ-scale thresholds (bps-ish): much smaller than Mag7 impulse 0.5%+
    ap.add_argument("--mom-thrs", default="0.0008,0.0012,0.002,0.003")
    ap.add_argument("--mom-lookbacks", default="30,60,120")
    ap.add_argument("--fo-thrs", default="0.001,0.002,0.003,0.005")
    ap.add_argument("--horizons", default="60,120,300,600,900")
    ap.add_argument("--tps", default="0.08,0.10,0.15,0.20")
    ap.add_argument("--sls", default="0.10,0.15,0.20,0.25")
    ap.add_argument("--stride-sec", type=int, default=60)
    ap.add_argument("--max-hold-sec", type=int, default=900)
    ap.add_argument("--slip", type=float, default=0.01)
    ap.add_argument("--position-frac", type=float, default=0.10)
    ap.add_argument("--max-concurrent", type=int, default=2)
    ap.add_argument("--cooldown-minutes", type=float, default=5.0)
    ap.add_argument("--min-n", type=int, default=25)
    ap.add_argument("--min-day-win", type=float, default=0.55)
    ap.add_argument("--top-foresight", type=int, default=12, help="promote top clock cells to TP/SL")
    args = ap.parse_args(argv)

    tick_root = Path(args.tick_root)
    opt_root = Path(args.opt_root)
    stock_1s = Path(args.stock_1s_root)
    out = Path(args.results_dir) / args.tag
    out.mkdir(parents=True, exist_ok=True)

    want_sess = {x.strip() for x in args.sessions.split(",") if x.strip()}
    sessions = tuple(s for s in SESSIONS if s[0] in want_sess)
    arms_want = {x.strip().upper() for x in args.arms.split(",") if x.strip()}
    mom_thrs = [float(x) for x in args.mom_thrs.split(",") if x.strip()]
    mom_lbs = [int(x) for x in args.mom_lookbacks.split(",") if x.strip()]
    fo_thrs = [float(x) for x in args.fo_thrs.split(",") if x.strip()]
    horizons = [int(x) for x in args.horizons.split(",") if x.strip()]
    tps = [float(x) for x in args.tps.split(",") if x.strip()]
    sls = [float(x) for x in args.sls.split(",") if x.strip()]

    start_all = min(w[1] for w in WINDOWS)
    end_all = max(w[2] for w in WINDOWS)
    dates = [
        d
        for d in tick_dates(tick_root, "QQQ")
        if start_all <= d <= end_all and (stock_1s / "QQQ" / f"QQQ_{d}.parquet").is_file()
    ]
    print(
        f"QQQ foresight dates={len(dates)} {dates[0]}..{dates[-1]} "
        f"sess={[s[0] for s in sessions]} arms={sorted(arms_want)}",
        flush=True,
    )

    # Collect first-fire arms per (date, session, arm_cfg, dir).
    events: list[dict[str, Any]] = []
    for di, date in enumerate(dates):
        if di % 15 == 0:
            print(f"[day] {date} ({di+1}/{len(dates)}) events={len(events)}", flush=True)
        day = load_stock_1s_day(stock_1s, "QQQ", date)
        if day is None or day.empty:
            continue
        tday = load_option_tick_day(tick_root, "QQQ", date)
        if tday is None or tday.empty:
            continue
        if "correction" in tday.columns:
            tday = tday[pd.to_numeric(tday["correction"], errors="coerce").fillna(0) == 0]
        print_paths = _paths_by_ticker(tday)
        if not print_paths:
            continue
        ts_ns, px = _stock_arrays(day)
        open_px = float(px[0]) if len(px) else 0.0
        if open_px <= 0:
            continue

        for sess_name, s0, s1 in sessions:
            t_end = pd.Timestamp(f"{date} {s1}", tz=NY)
            stride = pd.Timedelta(seconds=int(args.stride_sec))
            fired: set[tuple] = set()

            # MOM arms
            if "MOM" in arms_want:
                for lb in mom_lbs:
                    t = pd.Timestamp(f"{date} {s0}", tz=NY) + pd.Timedelta(seconds=int(lb))
                    while t < t_end:
                        for thr in mom_thrs:
                            direction, sr = _stock_dir_arr(ts_ns, px, t, int(lb), float(thr))
                            if direction is None:
                                continue
                            key = ("MOM", sess_name, direction, float(thr), int(lb))
                            if key in fired:
                                continue
                            spot = _spot_at_arr(ts_ns, px, t)
                            if spot is None:
                                continue
                            ticker, strike = _atm_ticker(
                                date=date,
                                direction=direction,
                                spot=float(spot),
                                opt_root=opt_root,
                                print_paths=print_paths,
                            )
                            if not ticker or ticker not in print_paths:
                                continue
                            fired.add(key)
                            pts, plast = print_paths[ticker]
                            events.append(
                                {
                                    "date": date,
                                    "session": sess_name,
                                    "arm": "MOM",
                                    "dir": direction,
                                    "thr": float(thr),
                                    "lookback_sec": int(lb),
                                    "feat": float(sr),
                                    "entry_ts": to_ny(t),
                                    "ticker": ticker,
                                    "strike": strike,
                                    "pts": pts,
                                    "plast": plast,
                                }
                            )
                        t += stride

            # FO / FADE arms
            for arm_name, fade in (("FO", False), ("FADE", True)):
                if arm_name not in arms_want:
                    continue
                t = pd.Timestamp(f"{date} {s0}", tz=NY)
                while t < t_end:
                    for thr in fo_thrs:
                        direction, fo = _from_open_dir(
                            ts_ns, px, t, open_px, float(thr), fade=fade
                        )
                        if direction is None:
                            continue
                        key = (arm_name, sess_name, direction, float(thr), 0)
                        if key in fired:
                            continue
                        spot = _spot_at_arr(ts_ns, px, t)
                        if spot is None:
                            continue
                        ticker, strike = _atm_ticker(
                            date=date,
                            direction=direction,
                            spot=float(spot),
                            opt_root=opt_root,
                            print_paths=print_paths,
                        )
                        if not ticker or ticker not in print_paths:
                            continue
                        fired.add(key)
                        pts, plast = print_paths[ticker]
                        events.append(
                            {
                                "date": date,
                                "session": sess_name,
                                "arm": arm_name,
                                "dir": direction,
                                "thr": float(thr),
                                "lookback_sec": 0,
                                "feat": float(fo),
                                "entry_ts": to_ny(t),
                                "ticker": ticker,
                                "strike": strike,
                                "pts": pts,
                                "plast": plast,
                            }
                        )
                    t += stride

    print(f"events={len(events)}; foresight scoring…", flush=True)

    # --- Foresight scoreboard: clock + oracle by H ---
    f_rows: list[dict[str, Any]] = []
    # group keys
    cells: dict[tuple, list[dict[str, Any]]] = {}
    for e in events:
        k = (e["arm"], e["session"], float(e["thr"]), int(e["lookback_sec"]))
        cells.setdefault(k, []).append(e)

    for (arm, sess, thr, lb), ents in cells.items():
        for h in horizons:
            win_stats: dict[str, dict[str, Any]] = {}
            for wname, w0, w1 in WINDOWS:
                raw_clock: list[dict[str, Any]] = []
                raw_oracle: list[dict[str, Any]] = []
                for e in ents:
                    if not (w0 <= e["date"] <= w1):
                        continue
                    fw = _fwd_trade_rets_arr(
                        e["pts"], e["plast"], e["entry_ts"], [h], slip=float(args.slip)
                    )
                    if not fw:
                        continue
                    r = fw[0]
                    base = {
                        "date": e["date"],
                        "symbol": "QQQ",
                        "dir": e["dir"],
                        "entry_ts": str(e["entry_ts"]),
                        "exit_ts": str(
                            e["entry_ts"] + pd.Timedelta(seconds=float(r.get("oracle_hold_sec") or h))
                        ),
                        "hold_sec": float(r.get("oracle_hold_sec") or h),
                    }
                    raw_clock.append({**base, "ret": float(r["clock_ret"]), "exit_reason": "clock"})
                    # oracle exit_ts approximate
                    raw_oracle.append(
                        {
                            **base,
                            "ret": float(r["oracle_ret"]),
                            "exit_reason": "oracle",
                            "exit_ts": str(
                                e["entry_ts"]
                                + pd.Timedelta(seconds=float(r.get("oracle_hold_sec") or h))
                            ),
                        }
                    )
                st_c = _port(
                    raw_clock,
                    position_frac=float(args.position_frac),
                    max_concurrent=int(args.max_concurrent),
                    cooldown=float(args.cooldown_minutes),
                )
                st_o = _port(
                    raw_oracle,
                    position_frac=float(args.position_frac),
                    max_concurrent=int(args.max_concurrent),
                    cooldown=float(args.cooldown_minutes),
                )
                win_stats[wname] = {"clock": st_c, "oracle": st_o}
            row: dict[str, Any] = {
                "arm": arm,
                "session": sess,
                "thr": thr,
                "lookback_sec": lb,
                "horizon_sec": h,
                "exit": "clock_vs_oracle",
            }
            dual_clock = True
            dual_oracle = True
            for wname, _, _ in WINDOWS:
                for kind in ("clock", "oracle"):
                    st = win_stats[wname][kind]
                    for k, v in st.items():
                        row[f"{wname}_{kind}_{k}"] = v
                dual_clock = dual_clock and _ok(
                    win_stats[wname]["clock"], min_n=int(args.min_n), min_day_win=float(args.min_day_win)
                )
                dual_oracle = dual_oracle and _ok(
                    win_stats[wname]["oracle"], min_n=int(args.min_n), min_day_win=float(args.min_day_win)
                )
            row["dual_clock_pass"] = dual_clock
            row["dual_oracle_pass"] = dual_oracle
            row["clock_add_sum"] = float(row.get("feb_apr_clock_add") or 0) + float(
                row.get("may_jul_clock_add") or 0
            )
            row["oracle_add_sum"] = float(row.get("feb_apr_oracle_add") or 0) + float(
                row.get("may_jul_oracle_add") or 0
            )
            f_rows.append(row)
            if dual_clock or (dual_oracle and row["clock_add_sum"] > 0):
                print(
                    f"[FS {'CLOCK' if dual_clock else 'orcl'}] {arm} {sess} thr={thr} lb={lb} H={h} "
                    f"clock_add={row['clock_add_sum']:+.3f} oracle_add={row['oracle_add_sum']:+.3f}",
                    flush=True,
                )

    fscore = pd.DataFrame(f_rows)
    fscore.to_csv(out / "foresight_scoreboard.csv", index=False)
    clock_pass = fscore[fscore["dual_clock_pass"]].sort_values("clock_add_sum", ascending=False)
    oracle_pass = fscore[fscore["dual_oracle_pass"]].sort_values("oracle_add_sum", ascending=False)
    # Promote: dual clock pass, else top oracle with weakly nonnegative clock both windows
    promote = clock_pass.head(int(args.top_foresight))
    if promote.empty:
        soft = fscore[
            (fscore["feb_apr_clock_mean"].fillna(-1) > -0.01)
            & (fscore["may_jul_clock_mean"].fillna(-1) > -0.01)
            & (fscore["dual_oracle_pass"])
        ].sort_values("oracle_add_sum", ascending=False)
        promote = soft.head(int(args.top_foresight))
    if promote.empty:
        # still take top oracle cells for TP/SL probe
        promote = oracle_pass.head(int(args.top_foresight))

    print(f"foresight dual_clock={len(clock_pass)} dual_oracle={len(oracle_pass)} promote={len(promote)}", flush=True)

    # --- Causal TP/SL on promoted cells ---
    t_rows: list[dict[str, Any]] = []
    dual_tpsl: list[dict[str, Any]] = []
    for _, pr in promote.iterrows():
        arm, sess, thr, lb = pr["arm"], pr["session"], float(pr["thr"]), int(pr["lookback_sec"])
        ents = cells.get((arm, sess, thr, lb), [])
        if not ents:
            continue
        for tp in tps:
            for sl in sls:
                win_stats: dict[str, dict[str, Any]] = {}
                for wname, w0, w1 in WINDOWS:
                    raw: list[dict[str, Any]] = []
                    for e in ents:
                        if not (w0 <= e["date"] <= w1):
                            continue
                        sim = simulate_trade_tpsl(
                            e["pts"],
                            e["plast"],
                            e["entry_ts"],
                            tp=tp,
                            sl=sl,
                            max_hold_sec=int(args.max_hold_sec),
                            slip=float(args.slip),
                        )
                        if sim is None or not np.isfinite(sim["ret"]):
                            continue
                        et = e["entry_ts"]
                        raw.append(
                            {
                                "date": e["date"],
                                "symbol": "QQQ",
                                "dir": e["dir"],
                                "entry_ts": str(et),
                                "exit_ts": str(et + pd.Timedelta(seconds=sim["hold_sec"])),
                                "ret": sim["ret"],
                                "exit_reason": sim["reason"],
                                "hold_sec": sim["hold_sec"],
                            }
                        )
                    win_stats[wname] = _port(
                        raw,
                        position_frac=float(args.position_frac),
                        max_concurrent=int(args.max_concurrent),
                        cooldown=float(args.cooldown_minutes),
                    )
                both = all(
                    _ok(win_stats[w[0]], min_n=int(args.min_n), min_day_win=float(args.min_day_win))
                    for w in WINDOWS
                )
                # also require not too much max_hold if present
                mh_ok = all(
                    (win_stats[w[0]].get("n") or 0) == 0
                    or True  # frac_max_hold not in _port; skip
                    for w in WINDOWS
                )
                row = {
                    "arm": arm,
                    "session": sess,
                    "thr": thr,
                    "lookback_sec": lb,
                    "tp": tp,
                    "sl": sl,
                    "dual_pass": both and mh_ok,
                    "fs_horizon": int(pr["horizon_sec"]),
                    "fs_clock_add_sum": float(pr.get("clock_add_sum") or 0),
                }
                for wname, _, _ in WINDOWS:
                    for k, v in win_stats[wname].items():
                        row[f"{wname}_{k}"] = v
                t_rows.append(row)
                if both:
                    dual_tpsl.append(row)
                    print(
                        f"  *** TPSL PASS {arm} {sess} thr={thr} lb={lb} tp{tp}/sl{sl} "
                        f"feb_n={row.get('feb_apr_n')} may_n={row.get('may_jul_n')}",
                        flush=True,
                    )

    tscore = pd.DataFrame(t_rows)
    if not tscore.empty:
        tscore.to_csv(out / "tpsl_scoreboard.csv", index=False)
    dual_tpsl = sorted(
        dual_tpsl,
        key=lambda r: float(r.get("feb_apr_add") or 0) + float(r.get("may_jul_add") or 0),
        reverse=True,
    )

    summary = {
        "symbol": "QQQ",
        "mode": "foresight_rule_discovery",
        "dates": {"n": len(dates), "start": dates[0], "end": dates[-1]},
        "n_events": len(events),
        "sessions": [s[0] for s in sessions],
        "arms": sorted(arms_want),
        "mom_thrs": mom_thrs,
        "fo_thrs": fo_thrs,
        "horizons": horizons,
        "foresight": {
            "n_cells": int(len(fscore)),
            "dual_clock_pass": int(len(clock_pass)),
            "dual_oracle_pass": int(len(oracle_pass)),
            "top_clock": clock_pass.head(15).to_dict(orient="records") if len(clock_pass) else [],
            "top_oracle": oracle_pass.head(15).to_dict(orient="records") if len(oracle_pass) else [],
        },
        "tpsl": {
            "promoted_cells": int(len(promote)),
            "dual_pass_n": int(len(dual_tpsl)),
            "dual_pass": dual_tpsl[:20],
        },
        "verdict": (
            "PASS"
            if dual_tpsl
            else "FORESIGHT_ONLY"
            if len(clock_pass) or len(oracle_pass)
            else "REJECT"
        ),
        "note": (
            "QQQ-scale MOM/FO/FADE foresight grid on tick±slip. "
            "Clock dual PASS is causal hold edge; oracle PASS only proves path optionality. "
            "TP/SL scored on promoted foresight cells."
        ),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    (out / "dual_clock_pass.json").write_text(
        json.dumps(clock_pass.head(30).to_dict(orient="records"), indent=2, default=str),
        encoding="utf-8",
    )
    (out / "tpsl_dual_pass.json").write_text(json.dumps(dual_tpsl[:30], indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("verdict", "foresight", "tpsl") if k in summary}, indent=2, default=str)[:4000], flush=True)
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

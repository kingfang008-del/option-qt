#!/usr/bin/env python3
"""PM pulse window C research: 12:30+ FO both, DTE bakeoff, quote FillSpec.

Re-opens the abandoned lunch FO question with A/B-aligned quote fills and an
explicit **1DTE-only** book (plus 0 / 0+1 controls).

Cells
-----
- windows: 12:30–13:30 / 12:30–14:00 / 12:30–15:00
- FO thr 0.8%, dirs DN+UP, TP15/SL20, FillSpec 0.75, lag≤5, sp≤15%
- fo_mode:
    * any   — first in-window bar with |FO|≥thr (continuation / often stale)
    * fresh — only if |FO| was still <thr on the last bar before window_start
- dte_mode: dte1 / dte0 / dte01

Portfolio: position_frac=0.20, max_concurrent=2, cooldown 10m (same as A/B lock).

Example:
  PYTHONPATH=. python -m maga7.tools.run_pm_pulse_C_accept \\
    --tag research_pm_pulse_C_20260728
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

from maga7.common.am_pulse_scout import AmPulseScoutConfig, scan_day
from maga7.common.config import load_profile
from maga7.common.fills import FillSpec
from maga7.common.open_lock import (
    load_multidte_lock_index,
    resolve_open_lock_contract,
    resolve_otm_rungs,
)
from maga7.common.option_quote_tpsl import entry_quote_row, simulate_quote_tpsl
from maga7.common.replay import load_quotes, month_list, path_for_ticker, to_ny
from maga7.common.signals import load_stock_month_files
from maga7.common.stock_1s import session_dates
from maga7.tools.run_morning_sec_option_fill import _portfolio_day
from maga7.tools.scan_am_delayed_confirm_quote_dual import _prep_path

PROFILE = (
    "maga7/CONFIG/strategy_profiles/"
    "single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1.json"
)
NY = "America/New_York"
EVAL_WINDOWS = (
    ("feb_mar", "2026-02-01", "2026-03-31"),
    ("may_jul", "2026-05-01", "2026-07-23"),
)
FO_THR = 0.008
TP = 0.15
SL = 0.20
MAX_HOLD = 900
POS = 0.20
MAX_CONCURRENT = 2
COOLDOWN_MIN = 10.0


def _hhmm_min(hhmm: str) -> int:
    p = str(hhmm).split(":")
    return int(p[0]) * 60 + int(p[1])


def _fo_before_window(day1m: pd.DataFrame, window_start: str) -> float | None:
    """|from_open| on last 1m bar strictly before window_start."""
    if day1m is None or day1m.empty:
        return None
    day = day1m.sort_values("timestamp")
    open_px = float(day.iloc[0]["open"])
    if open_px <= 0:
        return None
    w0 = _hhmm_min(window_start)
    ts = pd.to_datetime(day["timestamp"])
    if getattr(ts.dt, "tz", None) is None:
        ts = ts.dt.tz_localize(NY)
    else:
        ts = ts.dt.tz_convert(NY)
    hm = ts.dt.hour * 60 + ts.dt.minute
    pre = day.loc[hm < w0]
    if pre.empty:
        return None
    px = float(pre.iloc[-1]["close"])
    return abs(px / open_px - 1.0)


def _stats_book(tr: pd.DataFrame) -> dict[str, Any]:
    if tr is None or tr.empty:
        return {
            "n": 0,
            "win": None,
            "add": 0.0,
            "mean": None,
            "compound": 0.0,
            "mult": 1.0,
            "maxdd": 0.0,
            "day_win": None,
            "n_days": 0,
            "n_up": 0,
            "n_dn": 0,
            "add_up": 0.0,
            "add_dn": 0.0,
            "fresh_n": 0,
            "stale_n": 0,
        }
    d = tr.groupby("date")["pnl_frac"].sum().sort_index()
    eq = (1.0 + d).cumprod()
    peak = eq.cummax()
    dd = eq / peak - 1.0
    up = tr[tr["dir"] == "UP"]
    dn = tr[tr["dir"] == "DN"]
    return {
        "n": int(len(tr)),
        "win": float((tr["ret"] > 0).mean()),
        "add": float(d.sum()),
        "mean": float(tr["ret"].mean()),
        "compound": float(eq.iloc[-1] - 1.0),
        "mult": float(eq.iloc[-1]),
        "maxdd": float(dd.min()) if len(dd) else 0.0,
        "day_win": float((d > 0).mean()) if len(d) else None,
        "n_days": int(len(d)),
        "n_up": int(len(up)),
        "n_dn": int(len(dn)),
        "add_up": float(up["pnl_frac"].sum()) if len(up) else 0.0,
        "add_dn": float(dn["pnl_frac"].sum()) if len(dn) else 0.0,
        "fresh_n": int(tr["fresh"].sum()) if "fresh" in tr.columns else 0,
        "stale_n": int((~tr["fresh"].astype(bool)).sum()) if "fresh" in tr.columns else 0,
    }


def _verdict(may: dict[str, Any], feb: dict[str, Any]) -> str:
    """Loose accept: both windows mean>0, may day_win≥0.55, may n≥15, maxdd>-25%."""
    if not may["n"] or may["mean"] is None:
        return "FAIL"
    if may["mean"] <= 0 or (feb["mean"] is not None and feb["mean"] <= 0 and feb["n"] >= 8):
        return "FAIL"
    if may["day_win"] is None or may["day_win"] < 0.55:
        return "FAIL"
    if may["n"] < 15:
        return "THIN"
    if may["maxdd"] < -0.25:
        return "FAIL"
    # Stronger than old lunch (~+8% may mean / weak jul): want may mean≥0.08 and feb mean≥0
    if may["mean"] >= 0.08 and (feb["mean"] is None or feb["mean"] >= 0):
        return "PASS"
    return "WEAK"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", default="research_pm_pulse_C_20260728")
    ap.add_argument("--profile", default=PROFILE)
    ap.add_argument("--start-date", default="2026-02-01")
    ap.add_argument("--end-date", default="2026-07-23")
    ap.add_argument(
        "--windows",
        default="12:30:13:30,12:30:14:00,12:30:15:00",
        help="comma list of start:end",
    )
    args = ap.parse_args(argv)

    prof = load_profile(args.profile)
    paths = prof["_paths"]
    stock_root = Path(paths["stock_root"])
    quote_root = Path(paths["quote_1s_root"])
    lock_path = Path(paths["open_locked_map"])
    out_dir = Path(paths["results_dir"]) / str(args.tag)
    out_dir.mkdir(parents=True, exist_ok=True)

    symbols = [str(s).upper() for s in (prof.get("symbols") or [])]
    months = month_list(args.start_date, args.end_date)
    dates = [
        d
        for d in session_dates(args.start_date, args.end_date)
        if args.start_date <= d <= args.end_date
    ]
    print(f"[init] dates={len(dates)} symbols={len(symbols)} months={months}", flush=True)

    lock = load_multidte_lock_index(lock_path)
    otm = resolve_otm_rungs(prof)
    fill = FillSpec(entry_frac=0.75, exit_frac=0.75)

    stock_by: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        stock_by[sym] = load_stock_month_files(stock_root, sym, months)
        print(f"  stock {sym}: {len(stock_by[sym])}", flush=True)

    win_specs: list[tuple[str, str, str]] = []
    for chunk in str(args.windows).split(","):
        parts = chunk.strip().split(":")
        if len(parts) != 4:
            raise SystemExit(f"bad window {chunk!r}; want HH:MM:HH:MM")
        ws, we = f"{parts[0]}:{parts[1]}", f"{parts[2]}:{parts[3]}"
        name = f"C_{ws.replace(':','')}_{we.replace(':','')}"
        win_specs.append((name, ws, we))

    dte_modes = {
        "dte1": (1, [1]),
        "dte0": (0, [0]),
        "dte01": (1, [0, 1]),
    }
    fo_modes = ("any", "fresh")

    # Collect raw fills per (window, fo_mode, dte_mode)
    books: dict[tuple[str, str, str], list[dict[str, Any]]] = {
        (w[0], fo, dte): [] for w in win_specs for fo in fo_modes for dte in dte_modes
    }

    for di, date in enumerate(dates):
        if di % 15 == 0:
            print(f"[day] {date} ({di+1}/{len(dates)})", flush=True)
        for sym in symbols:
            sdf = stock_by.get(sym)
            if sdf is None or sdf.empty:
                continue
            day1m = sdf[sdf["date"].astype(str) == date]
            if day1m.empty:
                continue
            by_dte = lock.get((sym, date))
            if not by_dte:
                continue
            qday = _prep_path(load_quotes(quote_root, sym, date))
            if qday is None or qday.empty:
                continue

            for wname, ws, we in win_specs:
                fo_pre = _fo_before_window(day1m, ws)
                cfg = AmPulseScoutConfig(
                    enabled=True,
                    window_start=ws,
                    window_end=we,
                    min_fav_from_open=FO_THR,
                    lookback_bars=2,
                    min_lookback_ret=0.99,
                    dirs=("DN", "UP"),
                    max_alerts_per_symbol=1,
                    rth_open_only=True,
                )
                alerts = scan_day(day1m, date=date, symbol=sym, cfg=cfg)
                for a in alerts:
                    if a.arm != "FO":
                        continue
                    is_fresh = fo_pre is None or float(fo_pre) + 1e-12 < FO_THR
                    for fo_mode in fo_modes:
                        if fo_mode == "fresh" and not is_fresh:
                            continue
                        arm_ts = to_ny(pd.Timestamp(a.ts))
                        spot = float(a.px)
                        for dte_name, (prefer, allowed) in dte_modes.items():
                            ticker, dte, _ = resolve_open_lock_contract(
                                by_dte,
                                direction=a.dir,
                                moneyness="ATM",
                                spot=spot,
                                prefer_dte=int(prefer),
                                allowed_dte=list(allowed),
                                clear_otm_thresh=0.01,
                                ladder=True,
                                otm_rungs=otm,
                            )
                            if not ticker:
                                continue
                            # Strict: if dte1 mode, require actual dte==1
                            if dte_name == "dte1" and int(dte) != 1:
                                continue
                            if dte_name == "dte0" and int(dte) != 0:
                                continue
                            path = _prep_path(path_for_ticker(qday, ticker))
                            if path is None or path.empty:
                                continue
                            sim = simulate_quote_tpsl(
                                path,
                                arm_ts,
                                tp=TP,
                                sl=SL,
                                max_hold_sec=MAX_HOLD,
                                fill=fill,
                                max_lag_sec=5.0,
                                max_spread_pct=0.15,
                                min_mid=0.05,
                            )
                            if sim is None:
                                continue
                            books[(wname, fo_mode, dte_name)].append(
                                {
                                    "date": date,
                                    "symbol": sym,
                                    "dir": a.dir,
                                    "entry_ts": str(sim.get("entry_ts") or arm_ts),
                                    "exit_ts": str(sim.get("exit_ts") or ""),
                                    "ticker": ticker,
                                    "dte": int(dte),
                                    "ret": float(sim["ret"]),
                                    "exit_reason": str(sim.get("reason") or ""),
                                    "hold_sec": float(sim.get("hold_sec") or 0.0),
                                    "fav_from_open": float(a.fav_from_open),
                                    "fo_pre": float(fo_pre) if fo_pre is not None else np.nan,
                                    "fresh": bool(is_fresh),
                                    "window": wname,
                                    "fo_mode": fo_mode,
                                    "dte_mode": dte_name,
                                    "entry_mid": float(sim.get("entry_mid") or np.nan)
                                    if sim.get("entry_mid") is not None
                                    else (
                                        float(sim["entry_px"])
                                        if sim.get("entry_px") is not None
                                        else np.nan
                                    ),
                                }
                            )

    score_rows: list[dict[str, Any]] = []
    for key, rows in books.items():
        wname, fo_mode, dte_name = key
        raw = pd.DataFrame(rows)
        tag = f"{wname}__{fo_mode}__{dte_name}"
        if raw.empty:
            score_rows.append(
                {
                    "name": tag,
                    "window": wname,
                    "fo_mode": fo_mode,
                    "dte_mode": dte_name,
                    "verdict": "EMPTY",
                    "may_n": 0,
                    "feb_n": 0,
                }
            )
            continue
        # Portfolio size
        raw = raw.sort_values(["date", "entry_ts", "symbol"]).reset_index(drop=True)
        port_parts: list[pd.DataFrame] = []
        for date, g in raw.groupby("date", sort=True):
            day_rows = g.to_dict(orient="records")
            sized = _portfolio_day(
                day_rows,
                position_frac=POS,
                max_concurrent=MAX_CONCURRENT,
                cooldown_minutes=int(COOLDOWN_MIN),
            )
            if sized:
                port_parts.append(pd.DataFrame(sized))
        book = pd.concat(port_parts, ignore_index=True) if port_parts else raw.assign(
            size=POS, pnl_frac=raw["ret"].astype(float) * POS
        )
        raw.to_csv(out_dir / f"raw_{tag}.csv", index=False)
        book.to_csv(out_dir / f"book_{tag}.csv", index=False)

        may = _stats_book(
            book[(book["date"] >= "2026-05-01") & (book["date"] <= "2026-07-23")]
        )
        feb = _stats_book(
            book[(book["date"] >= "2026-02-01") & (book["date"] <= "2026-03-31")]
        )
        verd = _verdict(may, feb)
        row = {
            "name": tag,
            "window": wname,
            "fo_mode": fo_mode,
            "dte_mode": dte_name,
            "verdict": verd,
            **{f"may_{k}": v for k, v in may.items()},
            **{f"feb_{k}": v for k, v in feb.items()},
        }
        score_rows.append(row)
        print(
            f"[{tag}] may n={may['n']} mean={may['mean']} mult={may['mult']:.2f} "
            f"maxdd={may['maxdd']:.2%} feb_mean={feb['mean']} → {verd}",
            flush=True,
        )

    sb = pd.DataFrame(score_rows).sort_values(
        by=["verdict", "may_mult", "may_mean"],
        ascending=[True, False, False],
        key=lambda s: s.map({"PASS": 0, "WEAK": 1, "THIN": 2, "FAIL": 3, "EMPTY": 4})
        if s.name == "verdict"
        else s,
    )
    # Stable sort workaround
    order = {"PASS": 0, "WEAK": 1, "THIN": 2, "FAIL": 3, "EMPTY": 4}
    sb["_o"] = sb["verdict"].map(order)
    sb = sb.sort_values(["_o", "may_mult", "may_mean"], ascending=[True, False, False]).drop(
        columns=["_o"]
    )
    sb.to_csv(out_dir / "scoreboard.csv", index=False)

    promote = sb[sb["verdict"] == "PASS"]["name"].tolist()
    weak = sb[sb["verdict"] == "WEAK"]["name"].tolist()
    # Highlight user target: 1DTE
    dte1 = sb[sb["dte_mode"] == "dte1"].copy()
    summary = {
        "tag": args.tag,
        "note": (
            "PM pulse C re-scan after abandoned lunch FO. Quote FillSpec aligned with A/B. "
            "Primary ask: 12:30+ 1DTE-only. fo_mode=any includes morning-extended names; "
            "fresh requires first FO cross inside the window."
        ),
        "fo_thr": FO_THR,
        "tp": TP,
        "sl": SL,
        "pos": POS,
        "max_concurrent": MAX_CONCURRENT,
        "windows": win_specs,
        "promote": promote,
        "weak": weak,
        "best_dte1": dte1.head(5).to_dict(orient="records") if len(dte1) else [],
        "scoreboard": sb.to_dict(orient="records"),
        "vs_lunch_trades_ref": {
            "tag": "research_lunch_pulse_dual_v1",
            "cell": "FO@0.8 tp15/sl20 12:30-13:30 trades",
            "may_mean_approx": 0.081,
            "jul_mean_approx": 0.016,
            "status": "ABANDONED_weak_vs_AM",
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"out": str(out_dir), "promote": promote, "weak": weak[:8]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

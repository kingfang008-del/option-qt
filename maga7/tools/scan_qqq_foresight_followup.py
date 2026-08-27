#!/usr/bin/env python3
"""QQQ follow-ups from foresight grid:

1) MID FO≥0.3% continuation — quote FillSpec TP/SL dual accept
2) AM MOM 30s@0.08% — tick (+quote when present) TP/SL / trail probe

Example:
  PYTHONPATH=. python -m maga7.tools.scan_qqq_foresight_followup \\
    --tag research_qqq_foresight_followup_20260728
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
from maga7.common.fills import FillSpec
from maga7.common.option_flow import load_option_tick_day, tick_dates
from maga7.common.option_quote_exit_stress import ExitStressPolicy, simulate_quote_exit_stress
from maga7.common.option_quote_tpsl import entry_quote_row, simulate_quote_tpsl
from maga7.common.option_trade_tpsl import simulate_trade_tpsl
from maga7.common.replay import to_ny
from maga7.tools.run_morning_sec_option_fill import _portfolio_day
from maga7.tools.run_morning_sec_qqq_dte1 import _discover_option_dates, _load_atm_path
from maga7.tools.scan_morning_sec_edge import _morning_slice
from maga7.tools.scan_session_horizon_foresight import (
    _paths_by_ticker,
    _spot_at_arr,
    _stock_arrays,
    _stock_dir_arr,
)

NY = "America/New_York"
DEFAULT_OPT = Path("/mnt/s990/data/raw_1s/dte0_options/QQQ")
DEFAULT_TICK = Path("/mnt/s990/new_option_data_s3_tick")
DEFAULT_STOCK = Path("/mnt/s990/data/raw_1s/stocks")
DEFAULT_RESULTS = Path("/mnt/s990/data/maga7/results")

QUOTE_WINDOWS = (
    ("jan_mar", "2026-01-02", "2026-03-31"),
    ("may_jun", "2026-05-01", "2026-06-30"),
)
TICK_WINDOWS = (
    ("feb_apr", "2026-02-02", "2026-04-30"),
    ("may_jul", "2026-05-01", "2026-07-23"),
)

_OCC = re.compile(
    r"^O?:?(?P<root>[A-Z]+)(?P<yy>\d{2})(?P<mm>\d{2})(?P<dd>\d{2})"
    r"(?P<cp>[CP])(?P<strike>\d{8})$"
)


def _port(rows: list[dict[str, Any]], *, position_frac: float = 0.10, max_concurrent: int = 2, cooldown: float = 5.0) -> dict[str, Any]:
    if not rows:
        return {"n": 0, "mean": None, "win": None, "add": 0.0, "day_win": None, "frac_tp": None, "frac_sl": None, "frac_max_hold": None}
    by: dict[str, list] = {}
    for r in rows:
        by.setdefault(str(r["date"]), []).append(r)
    sized: list[dict] = []
    for d in sorted(by):
        sized.extend(
            _portfolio_day(by[d], position_frac=position_frac, max_concurrent=max_concurrent, cooldown_minutes=cooldown)
        )
    if not sized:
        return {"n": 0, "mean": None, "win": None, "add": 0.0, "day_win": None, "frac_tp": None, "frac_sl": None, "frac_max_hold": None}
    t = pd.DataFrame(sized)
    day = t.groupby("date")["pnl_frac"].sum()
    reasons = pd.Series([r.get("exit_reason") for r in sized])
    return {
        "n": int(len(t)),
        "mean": float(t["ret"].mean()),
        "win": float((t["ret"] > 0).mean()),
        "add": float(t["pnl_frac"].sum()),
        "day_win": float((day > 0).mean()),
        "worst_day": float(day.min()),
        "frac_tp": float((reasons == "tp").mean()) if len(reasons) else None,
        "frac_sl": float((reasons.isin(["sl", "trail", "be", "ladder"])).mean()) if len(reasons) else None,
        "frac_max_hold": float((reasons == "max_hold").mean()) if len(reasons) else None,
        "n_days": int(day.shape[0]),
    }


def _ok(st: dict[str, Any], *, min_n: int, min_day_win: float = 0.55) -> bool:
    if st.get("mean") is None or st.get("day_win") is None:
        return False
    mh = st.get("frac_max_hold")
    return bool(
        int(st.get("n") or 0) >= min_n
        and float(st["mean"]) > 0
        and float(st.get("add") or 0) > 0
        and float(st["day_win"]) >= min_day_win
        and (mh is None or float(mh) <= 0.50)
    )


def _atm_print(
    *,
    date: str,
    direction: str,
    spot: float,
    opt_root: Path,
    print_paths: dict[str, tuple[np.ndarray, np.ndarray]],
) -> tuple[str | None, float | None]:
    _p, ticker, strike = _load_atm_path(opt_root, date, direction)
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


def _stock_bundle(stock_1s: Path, date: str) -> dict[str, Any] | None:
    day = load_stock_1s_day(stock_1s, "QQQ", date)
    buf = _morning_slice(day, start="09:30", end="16:00")
    if buf.empty:
        return None
    ts_ns, px = _stock_arrays(buf)
    return {"ts_ns": ts_ns, "px": px, "open": float(px[0])}


def _collect_mid_fo(
    dates: list[str],
    *,
    stock_1s: Path,
    opt_root: Path,
    fo_min: float,
    session: tuple[str, str] = ("12:00", "14:00"),
    stride_sec: int = 60,
    book: str,
    tick_root: Path | None = None,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for di, date in enumerate(dates):
        if di % 20 == 0:
            print(f"[MID/{book}] {date} ({di+1}/{len(dates)}) n={len(out)}", flush=True)
        sb = _stock_bundle(stock_1s, date)
        if sb is None:
            continue
        ts_ns, px, open_px = sb["ts_ns"], sb["px"], sb["open"]
        qpath_cache: dict[str, Any] = {}
        print_paths = None
        if book == "tick":
            assert tick_root is not None
            tday = load_option_tick_day(tick_root, "QQQ", date)
            if tday is None or tday.empty:
                continue
            if "correction" in tday.columns:
                tday = tday[pd.to_numeric(tday["correction"], errors="coerce").fillna(0) == 0]
            print_paths = _paths_by_ticker(tday)
            if not print_paths:
                continue

        fired = False
        t = pd.Timestamp(f"{date} {session[0]}", tz=NY)
        t_end = pd.Timestamp(f"{date} {session[1]}", tz=NY)
        stride = pd.Timedelta(seconds=stride_sec)
        while t < t_end and not fired:
            spot = _spot_at_arr(ts_ns, px, t)
            if spot is None or open_px <= 0:
                t += stride
                continue
            fo = float(spot / open_px - 1.0)
            if abs(fo) < fo_min:
                t += stride
                continue
            direction = "UP" if fo > 0 else "DN"
            if book == "quote":
                if direction not in qpath_cache:
                    qpath_cache[direction] = _load_atm_path(opt_root, date, direction)
                path, ticker, strike = qpath_cache[direction]
                if path is None or path.empty:
                    t += stride
                    continue
                out.append(
                    {
                        "date": date,
                        "dir": direction,
                        "from_open": fo,
                        "entry_ts": to_ny(t),
                        "ticker": ticker,
                        "strike": strike,
                        "path": path,
                    }
                )
            else:
                assert print_paths is not None
                ticker, strike = _atm_print(
                    date=date, direction=direction, spot=float(spot), opt_root=opt_root, print_paths=print_paths
                )
                if not ticker or ticker not in print_paths:
                    t += stride
                    continue
                pts, plast = print_paths[ticker]
                out.append(
                    {
                        "date": date,
                        "dir": direction,
                        "from_open": fo,
                        "entry_ts": to_ny(t),
                        "ticker": ticker,
                        "strike": strike,
                        "pts": pts,
                        "plast": plast,
                    }
                )
            fired = True
            t += stride
    return out


def _collect_am_mom(
    dates: list[str],
    *,
    stock_1s: Path,
    opt_root: Path,
    thr: float,
    lookback_sec: int,
    session: tuple[str, str] = ("09:35", "10:30"),
    stride_sec: int = 30,
    book: str,
    tick_root: Path | None = None,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for di, date in enumerate(dates):
        if di % 20 == 0:
            print(f"[AM_MOM/{book}] {date} ({di+1}/{len(dates)}) n={len(out)}", flush=True)
        sb = _stock_bundle(stock_1s, date)
        if sb is None:
            continue
        ts_ns, px = sb["ts_ns"], sb["px"]
        print_paths = None
        if book == "tick":
            assert tick_root is not None
            tday = load_option_tick_day(tick_root, "QQQ", date)
            if tday is None or tday.empty:
                continue
            if "correction" in tday.columns:
                tday = tday[pd.to_numeric(tday["correction"], errors="coerce").fillna(0) == 0]
            print_paths = _paths_by_ticker(tday)
            if not print_paths:
                continue
        qpath_cache: dict[str, Any] = {}
        fired_dirs: set[str] = set()
        t = pd.Timestamp(f"{date} {session[0]}", tz=NY) + pd.Timedelta(seconds=lookback_sec)
        t_end = pd.Timestamp(f"{date} {session[1]}", tz=NY)
        stride = pd.Timedelta(seconds=stride_sec)
        while t < t_end and len(fired_dirs) < 2:
            direction, sr = _stock_dir_arr(ts_ns, px, t, lookback_sec, thr)
            if direction is None or direction in fired_dirs:
                t += stride
                continue
            spot = _spot_at_arr(ts_ns, px, t)
            if spot is None:
                t += stride
                continue
            if book == "quote":
                if direction not in qpath_cache:
                    qpath_cache[direction] = _load_atm_path(opt_root, date, direction)
                path, ticker, strike = qpath_cache[direction]
                if path is None or path.empty:
                    t += stride
                    continue
                out.append(
                    {
                        "date": date,
                        "dir": direction,
                        "feat": float(sr),
                        "entry_ts": to_ny(t),
                        "ticker": ticker,
                        "strike": strike,
                        "path": path,
                    }
                )
            else:
                assert print_paths is not None
                ticker, strike = _atm_print(
                    date=date, direction=direction, spot=float(spot), opt_root=opt_root, print_paths=print_paths
                )
                if not ticker or ticker not in print_paths:
                    t += stride
                    continue
                pts, plast = print_paths[ticker]
                out.append(
                    {
                        "date": date,
                        "dir": direction,
                        "feat": float(sr),
                        "entry_ts": to_ny(t),
                        "ticker": ticker,
                        "strike": strike,
                        "pts": pts,
                        "plast": plast,
                    }
                )
            fired_dirs.add(direction)
            t += stride
    return out


def _score_tpsl(
    *,
    name: str,
    book: str,
    entries: list[dict[str, Any]],
    windows: tuple[tuple[str, str, str], ...],
    tps: list[float],
    sls: list[float],
    fill: FillSpec,
    slip: float,
    max_hold_sec: int,
    max_spread: float,
    max_lag: float,
    min_mid: float,
    min_n: int,
    trails: list[tuple[float, float]] | None = None,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    dual: list[dict[str, Any]] = []

    def _sim_one(e: dict[str, Any], tp: float, sl: float, trail: tuple[float, float] | None) -> dict[str, Any] | None:
        if book == "quote":
            if trail is not None:
                pol = ExitStressPolicy(
                    tp=tp,
                    sl=sl,
                    max_hold_sec=max_hold_sec,
                    floor_arm=trail[0],
                    floor_mode="trail",
                    floor_offset=trail[1],
                )
                sim = simulate_quote_exit_stress(
                    e["path"],
                    e["entry_ts"],
                    policy=pol,
                    fill=fill,
                    max_lag_sec=max_lag,
                    max_spread_pct=max_spread,
                    min_mid=min_mid,
                )
            else:
                sim = simulate_quote_tpsl(
                    e["path"],
                    e["entry_ts"],
                    tp=tp,
                    sl=sl,
                    max_hold_sec=max_hold_sec,
                    fill=fill,
                    max_lag_sec=max_lag,
                    max_spread_pct=max_spread,
                    min_mid=min_mid,
                )
            if sim is None:
                return None
            reason = sim.get("reason") or sim.get("exit_reason")
            hold = float(sim.get("hold_sec") or 0)
            ret = float(sim["ret"])
        else:
            if trail is not None:
                return None  # trail only on quote
            sim = simulate_trade_tpsl(
                e["pts"], e["plast"], e["entry_ts"], tp=tp, sl=sl, max_hold_sec=max_hold_sec, slip=slip
            )
            if sim is None:
                return None
            reason = sim["reason"]
            hold = float(sim["hold_sec"])
            ret = float(sim["ret"])
        if not np.isfinite(ret):
            return None
        et = e["entry_ts"]
        return {
            "date": e["date"],
            "symbol": "QQQ",
            "dir": e["dir"],
            "entry_ts": str(et),
            "exit_ts": str(et + pd.Timedelta(seconds=hold)),
            "ret": ret,
            "exit_reason": reason,
            "hold_sec": hold,
        }

    configs: list[tuple[float, float, tuple[float, float] | None, str]] = []
    for tp in tps:
        for sl in sls:
            configs.append((tp, sl, None, "tpsl"))
    if trails and book == "quote":
        for tp in tps:
            for sl in sls:
                for tr in trails:
                    configs.append((tp, sl, tr, f"trail_{tr[0]}_{tr[1]}"))

    for tp, sl, trail, mode in configs:
        win_stats: dict[str, dict[str, Any]] = {}
        for wname, w0, w1 in windows:
            raw: list[dict[str, Any]] = []
            for e in entries:
                if not (w0 <= e["date"] <= w1):
                    continue
                if book == "quote":
                    probe = entry_quote_row(
                        e["path"], e["entry_ts"], max_lag_sec=max_lag, max_spread_pct=max_spread, min_mid=min_mid
                    )
                    if probe is None:
                        continue
                one = _sim_one(e, tp, sl, trail)
                if one is None:
                    continue
                raw.append(one)
            win_stats[wname] = _port(raw)
            st = win_stats[wname]
            if st.get("n", 0) >= 15:
                print(
                    f"[{name}/{book}/{mode} tp{tp}/sl{sl} {wname}] "
                    f"n={st['n']} mean={st['mean']} add={st['add']:+.3f} day_win={st['day_win']}",
                    flush=True,
                )
        both = all(_ok(win_stats[w[0]], min_n=min_n) for w in windows)
        row: dict[str, Any] = {
            "rule": name,
            "book": book,
            "mode": mode,
            "tp": tp,
            "sl": sl,
            "trail_arm": None if trail is None else trail[0],
            "trail_dd": None if trail is None else trail[1],
            "dual_pass": both,
        }
        for wname, _, _ in windows:
            for k, v in win_stats[wname].items():
                row[f"{wname}_{k}"] = v
        rows.append(row)
        if both:
            dual.append(row)
            print(f"  *** DUAL PASS {name}/{book}/{mode} tp{tp}/sl{sl}", flush=True)

    dual.sort(
        key=lambda r: sum(float(r.get(f"{w[0]}_add") or 0) for w in windows),
        reverse=True,
    )
    return pd.DataFrame(rows), dual


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--opt-root", default=str(DEFAULT_OPT))
    ap.add_argument("--tick-root", default=str(DEFAULT_TICK))
    ap.add_argument("--stock-1s-root", default=str(DEFAULT_STOCK))
    ap.add_argument("--results-dir", default=str(DEFAULT_RESULTS))
    ap.add_argument("--tag", default="research_qqq_foresight_followup_20260728")
    ap.add_argument("--mid-fo", type=float, default=0.003)
    ap.add_argument("--am-thr", type=float, default=0.0008)
    ap.add_argument("--am-lookback", type=int, default=30)
    ap.add_argument("--tps", default="0.10,0.15,0.20")
    ap.add_argument("--sls", default="0.15,0.20,0.25")
    ap.add_argument("--max-spread", type=float, default=0.15)
    ap.add_argument("--max-lag", type=float, default=2.0)
    ap.add_argument("--min-mid", type=float, default=0.05)
    ap.add_argument("--max-hold-sec", type=int, default=900)
    ap.add_argument("--slip", type=float, default=0.01)
    ap.add_argument("--min-n-quote", type=int, default=20)
    ap.add_argument("--min-n-tick", type=int, default=20)
    args = ap.parse_args(argv)

    opt_root = Path(args.opt_root)
    tick_root = Path(args.tick_root)
    stock_1s = Path(args.stock_1s_root)
    out = Path(args.results_dir) / args.tag
    out.mkdir(parents=True, exist_ok=True)
    fill = FillSpec(0.75, 0.75)
    tps = [float(x) for x in args.tps.split(",") if x.strip()]
    sls = [float(x) for x in args.sls.split(",") if x.strip()]
    trails = [(0.10, 0.08), (0.15, 0.08), (0.20, 0.10)]

    # Quote calendar
    q_dates = [
        d
        for d in _discover_option_dates(opt_root, "2026-01-02", "2026-06-30")
        if (stock_1s / "QQQ" / f"QQQ_{d}.parquet").is_file()
    ]
    t_dates = [
        d
        for d in tick_dates(tick_root, "QQQ")
        if "2026-02-02" <= d <= "2026-07-23" and (stock_1s / "QQQ" / f"QQQ_{d}.parquet").is_file()
    ]
    print(f"quote_days={len(q_dates)} tick_days={len(t_dates)}", flush=True)

    summary: dict[str, Any] = {"tag": args.tag, "rules": {}}

    # ---- 1) MID FO quote + tick confirm ----
    mid_q = _collect_mid_fo(
        q_dates, stock_1s=stock_1s, opt_root=opt_root, fo_min=float(args.mid_fo), book="quote"
    )
    mid_t = _collect_mid_fo(
        t_dates,
        stock_1s=stock_1s,
        opt_root=opt_root,
        fo_min=float(args.mid_fo),
        book="tick",
        tick_root=tick_root,
    )
    print(f"MID FO entries quote={len(mid_q)} tick={len(mid_t)}", flush=True)

    mq_score, mq_dual = _score_tpsl(
        name="MID_FO",
        book="quote",
        entries=mid_q,
        windows=QUOTE_WINDOWS,
        tps=tps,
        sls=sls,
        fill=fill,
        slip=float(args.slip),
        max_hold_sec=int(args.max_hold_sec),
        max_spread=float(args.max_spread),
        max_lag=float(args.max_lag),
        min_mid=float(args.min_mid),
        min_n=int(args.min_n_quote),
        trails=None,
    )
    mt_score, mt_dual = _score_tpsl(
        name="MID_FO",
        book="tick",
        entries=mid_t,
        windows=TICK_WINDOWS,
        tps=tps,
        sls=sls,
        fill=fill,
        slip=float(args.slip),
        max_hold_sec=int(args.max_hold_sec),
        max_spread=float(args.max_spread),
        max_lag=float(args.max_lag),
        min_mid=float(args.min_mid),
        min_n=int(args.min_n_tick),
    )
    if not mq_score.empty:
        mq_score.to_csv(out / "mid_fo_quote_scoreboard.csv", index=False)
    if not mt_score.empty:
        mt_score.to_csv(out / "mid_fo_tick_scoreboard.csv", index=False)
    summary["rules"]["MID_FO"] = {
        "fo_min": float(args.mid_fo),
        "session": "12:00-14:00",
        "quote": {"n_entries": len(mid_q), "dual_pass_n": len(mq_dual), "dual_pass": mq_dual[:10], "verdict": "PASS" if mq_dual else "REJECT"},
        "tick": {"n_entries": len(mid_t), "dual_pass_n": len(mt_dual), "dual_pass": mt_dual[:10], "verdict": "PASS" if mt_dual else "REJECT"},
    }

    # ---- 2) AM MOM tick + quote (+ trail on quote) ----
    am_q = _collect_am_mom(
        q_dates,
        stock_1s=stock_1s,
        opt_root=opt_root,
        thr=float(args.am_thr),
        lookback_sec=int(args.am_lookback),
        book="quote",
    )
    am_t = _collect_am_mom(
        t_dates,
        stock_1s=stock_1s,
        opt_root=opt_root,
        thr=float(args.am_thr),
        lookback_sec=int(args.am_lookback),
        book="tick",
        tick_root=tick_root,
    )
    print(f"AM MOM entries quote={len(am_q)} tick={len(am_t)}", flush=True)

    aq_score, aq_dual = _score_tpsl(
        name="AM_MOM30",
        book="quote",
        entries=am_q,
        windows=QUOTE_WINDOWS,
        tps=tps,
        sls=sls,
        fill=fill,
        slip=float(args.slip),
        max_hold_sec=int(args.max_hold_sec),
        max_spread=float(args.max_spread),
        max_lag=float(args.max_lag),
        min_mid=float(args.min_mid),
        min_n=int(args.min_n_quote),
        trails=trails,
    )
    at_score, at_dual = _score_tpsl(
        name="AM_MOM30",
        book="tick",
        entries=am_t,
        windows=TICK_WINDOWS,
        tps=tps,
        sls=sls,
        fill=fill,
        slip=float(args.slip),
        max_hold_sec=int(args.max_hold_sec),
        max_spread=float(args.max_spread),
        max_lag=float(args.max_lag),
        min_mid=float(args.min_mid),
        min_n=int(args.min_n_tick),
    )
    if not aq_score.empty:
        aq_score.to_csv(out / "am_mom_quote_scoreboard.csv", index=False)
    if not at_score.empty:
        at_score.to_csv(out / "am_mom_tick_scoreboard.csv", index=False)
    summary["rules"]["AM_MOM30"] = {
        "thr": float(args.am_thr),
        "lookback_sec": int(args.am_lookback),
        "session": "09:35-10:30",
        "quote": {"n_entries": len(am_q), "dual_pass_n": len(aq_dual), "dual_pass": aq_dual[:10], "verdict": "PASS" if aq_dual else "REJECT"},
        "tick": {"n_entries": len(am_t), "dual_pass_n": len(at_dual), "dual_pass": at_dual[:10], "verdict": "PASS" if at_dual else "REJECT"},
    }

    mid_ok = bool(mq_dual) and bool(mt_dual)
    am_ok = bool(aq_dual) or bool(at_dual)
    summary["verdict"] = (
        "PASS_BOTH"
        if mid_ok and am_ok
        else "PASS_MID"
        if mid_ok
        else "PASS_AM"
        if am_ok
        else "PARTIAL_TICK_OR_QUOTE"
        if (mq_dual or mt_dual or aq_dual or at_dual)
        else "REJECT"
    )
    summary["note"] = (
        "MID FO quote+tick both required for MID promote; "
        "AM MOM accepts either book PASS as research candidate (trail only on quote)."
    )
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str)[:5000], flush=True)
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

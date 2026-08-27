#!/usr/bin/env python3
"""QQQ multi-clock open_cont — full-session, both dirs, multi-fire dual accept.

Relaxes the 09:45 one-shot sleeve:
  - clocks across RTH (default every 15m 09:35–15:15)
  - UP→CALL / DN→PUT whenever |from_open| ≥ fo_min
  - same-day multi-entry (max_concurrent > 1; same-symbol allowed)
  - lower FO grid (default 0.05%/0.1%/0.2%)

Books:
  - quote FillSpec on ``dte0_options/QQQ`` (ends ~2026-06-30)
  - tick last±slip on ``new_option_data_s3_tick`` (→2026-07-23)

Example:
  PYTHONPATH=. python -m maga7.tools.scan_qqq_multi_clock_tpsl \\
    --tag research_qqq_multi_clock_20260728
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
from maga7.common.option_quote_tpsl import entry_quote_row, simulate_quote_tpsl
from maga7.common.option_trade_tpsl import simulate_trade_tpsl
from maga7.common.replay import to_ny
from maga7.tools.run_morning_sec_qqq_dte1 import _discover_option_dates, _load_atm_path
from maga7.tools.scan_morning_sec_edge import _morning_slice
from maga7.tools.scan_session_horizon_foresight import _paths_by_ticker

NY = "America/New_York"
DEFAULT_OPT = Path("/mnt/s990/data/raw_1s/dte0_options/QQQ")
DEFAULT_TICK = Path("/mnt/s990/new_option_data_s3_tick")
DEFAULT_STOCK = Path("/mnt/s990/data/raw_1s/stocks")
DEFAULT_RESULTS = Path("/mnt/s990/data/maga7/results")

WINDOWS = (
    ("jan_mar", "2026-01-02", "2026-03-31"),
    ("may_jul", "2026-05-01", "2026-07-23"),
)

_OCC = re.compile(
    r"^O?:?(?P<root>[A-Z]+)(?P<yy>\d{2})(?P<mm>\d{2})(?P<dd>\d{2})"
    r"(?P<cp>[CP])(?P<strike>\d{8})$"
)


def _hhmm_range(start: str, end: str, stride_min: int) -> list[str]:
    def _m(hhmm: str) -> int:
        h, m = hhmm.split(":")
        return int(h) * 60 + int(m)

    out: list[str] = []
    t = _m(start)
    e = _m(end)
    while t <= e:
        out.append(f"{t // 60:02d}:{t % 60:02d}")
        t += int(stride_min)
    return out


def _portfolio_day_multi(
    day_trades: list[dict[str, Any]],
    *,
    position_frac: float,
    max_concurrent: int,
    cooldown_minutes: float,
) -> list[dict[str, Any]]:
    """Same-symbol multi-seat portfolio (QQQ can stack)."""
    if not day_trades:
        return []
    rows = sorted(day_trades, key=lambda r: (pd.Timestamp(r["entry_ts"]), str(r.get("dir") or "")))
    open_pos: list[pd.Timestamp] = []
    last_exit: pd.Timestamp | None = None
    out: list[dict[str, Any]] = []
    for tr in rows:
        et = to_ny(tr["entry_ts"])
        xt = to_ny(tr["exit_ts"])
        open_pos = [x for x in open_pos if x > et]
        if last_exit is not None and (et - last_exit).total_seconds() < float(cooldown_minutes) * 60.0:
            continue
        if len(open_pos) >= int(max_concurrent):
            continue
        n_active = len(open_pos) + 1
        size = float(position_frac) / float(n_active)
        row = dict(tr)
        row["size"] = size
        row["pnl_frac"] = float(tr["ret"]) * size
        out.append(row)
        open_pos.append(xt)
        last_exit = xt
    return out


def _port(
    rows: list[dict[str, Any]],
    *,
    position_frac: float,
    max_concurrent: int,
    cooldown_minutes: float,
) -> dict[str, Any]:
    if not rows:
        return {
            "n": 0,
            "mean": None,
            "win": None,
            "add": 0.0,
            "day_win": None,
            "frac_tp": None,
            "frac_sl": None,
            "frac_max_hold": None,
            "hold_p50": None,
            "n_days": 0,
            "trades_per_day": None,
        }
    by: dict[str, list] = {}
    for r in rows:
        by.setdefault(str(r["date"]), []).append(r)
    sized: list[dict] = []
    for d in sorted(by):
        sized.extend(
            _portfolio_day_multi(
                by[d],
                position_frac=position_frac,
                max_concurrent=max_concurrent,
                cooldown_minutes=cooldown_minutes,
            )
        )
    if not sized:
        return {
            "n": 0,
            "mean": None,
            "win": None,
            "add": 0.0,
            "day_win": None,
            "frac_tp": None,
            "frac_sl": None,
            "frac_max_hold": None,
            "hold_p50": None,
            "n_days": 0,
            "trades_per_day": None,
        }
    t = pd.DataFrame(sized)
    t["pnl_frac"] = t["ret"].astype(float) * t["size"].astype(float)
    day = t.groupby("date")["pnl_frac"].sum()
    reasons = pd.Series([r.get("exit_reason") for r in sized])
    n_days = int(day.shape[0])
    return {
        "n": int(len(t)),
        "mean": float(t["ret"].mean()),
        "win": float((t["ret"] > 0).mean()),
        "add": float(t["pnl_frac"].sum()),
        "day_win": float((day > 0).mean()),
        "red_days": int((day < 0).sum()),
        "worst_day": float(day.min()),
        "frac_tp": float((reasons == "tp").mean()) if len(reasons) else None,
        "frac_sl": float((reasons == "sl").mean()) if len(reasons) else None,
        "frac_max_hold": float((reasons == "max_hold").mean()) if len(reasons) else None,
        "hold_p50": float(pd.Series([r.get("hold_sec") for r in sized]).median()),
        "n_days": n_days,
        "trades_per_day": float(len(t) / n_days) if n_days else None,
    }


def _ok(st: dict[str, Any], *, min_n: int) -> bool:
    mean = st.get("mean")
    day_win = st.get("day_win")
    mh = st.get("frac_max_hold")
    add = st.get("add")
    if mean is None or day_win is None or mh is None or add is None:
        return False
    return bool(
        int(st.get("n") or 0) >= min_n
        and float(mean) > 0
        and float(add) > 0
        and float(day_win) >= 0.55
        and float(mh) <= 0.50
    )


def _atm_from_prints(
    paths: dict[str, tuple[np.ndarray, np.ndarray]],
    *,
    date: str,
    direction: str,
    spot: float,
) -> tuple[str | None, float | None]:
    if not paths or not np.isfinite(spot) or spot <= 0:
        return None, None
    ymd = date.replace("-", "")[2:]
    want_cp = "C" if direction == "UP" else "P"
    best_t = None
    best_k = None
    best_abs = float("inf")
    for raw in paths:
        key = str(raw).replace("O:", "")
        m = _OCC.match(key)
        if m is None or m.group("root") != "QQQ":
            continue
        exp = f"{m.group('yy')}{m.group('mm')}{m.group('dd')}"
        if exp != ymd or m.group("cp") != want_cp:
            continue
        k = float(m.group("strike")) / 1000.0
        ad = abs(k - spot)
        if ad < best_abs:
            best_abs = ad
            best_k = k
            best_t = str(raw)
    return best_t, best_k


def _resolve_atm(
    *,
    date: str,
    direction: str,
    spot: float,
    opt_root: Path,
    print_paths: dict[str, tuple[np.ndarray, np.ndarray]] | None,
) -> tuple[str | None, float | None, Any, str]:
    """Return ticker, strike, quote_path_or_None, source."""
    qpath, qticker, qstrike = _load_atm_path(opt_root, date, direction)
    if print_paths is not None:
        if qticker:
            key = str(qticker).replace("O:", "")
            if key in print_paths:
                return key, float(qstrike) if qstrike is not None else None, qpath, "dte0_bucket"
            if str(qticker) in print_paths:
                return str(qticker), float(qstrike) if qstrike is not None else None, qpath, "dte0_bucket"
        t2, k2 = _atm_from_prints(print_paths, date=date, direction=direction, spot=spot)
        if t2:
            return t2, k2, qpath, "prints_occ"
        return None, None, qpath, "miss"
    if qpath is not None and not qpath.empty and qticker:
        return str(qticker), float(qstrike) if qstrike is not None else None, qpath, "dte0_bucket"
    return None, None, None, "miss"


def _stock_day(stock_1s: Path, date: str) -> dict[str, Any] | None:
    day = load_stock_1s_day(stock_1s, "QQQ", date)
    buf = _morning_slice(day, start="09:30", end="16:00")
    if buf.empty:
        return None
    ts = pd.DatetimeIndex(pd.to_datetime(buf["timestamp"]))
    if ts.tz is None:
        ts = ts.tz_localize(NY, ambiguous="infer")
    else:
        ts = ts.tz_convert(NY)
    close = buf["close"].astype(float).to_numpy()
    return {"ts": ts, "close": close, "open": float(close[0])}


def _build_raw_entries(
    *,
    dates: list[str],
    clocks: list[str],
    stock_1s: Path,
    opt_root: Path,
    tick_root: Path | None,
    book: str,
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for di, date in enumerate(dates):
        if di % 20 == 0:
            print(f"[{book}] day {date} ({di+1}/{len(dates)}) entries={len(entries)}", flush=True)
        sd = _stock_day(stock_1s, date)
        if sd is None:
            continue
        ts, close, open_px = sd["ts"], sd["close"], sd["open"]
        print_paths = None
        path_cache: dict[str, tuple[Any, Any, Any, str]] = {}
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

        for clock in clocks:
            t0 = pd.Timestamp(f"{date} {clock}", tz=NY)
            i = int(ts.searchsorted(t0, side="left"))
            if i >= len(close) - 1:
                continue
            if abs((ts[i] - t0).total_seconds()) > 5:
                continue
            spot = float(close[i])
            from_open = float((spot - open_px) / open_px) if open_px else 0.0
            if from_open == 0.0:
                continue
            direction = "UP" if from_open > 0 else "DN"
            if direction not in path_cache:
                path_cache[direction] = _resolve_atm(
                    date=date,
                    direction=direction,
                    spot=spot,
                    opt_root=opt_root,
                    print_paths=print_paths,
                )
            ticker, strike, qpath, src = path_cache[direction]
            # Refresh OCC ATM with current spot for tick book (strike drifts intraday).
            if book == "tick" and print_paths is not None:
                t2, k2 = _atm_from_prints(print_paths, date=date, direction=direction, spot=spot)
                if t2:
                    ticker, strike, src = t2, k2, "prints_occ"
            if book == "quote":
                if qpath is None or getattr(qpath, "empty", True):
                    continue
                entries.append(
                    {
                        "date": date,
                        "clock": clock,
                        "dir": direction,
                        "from_open": from_open,
                        "entry_ts": to_ny(ts[i]),
                        "ticker": ticker,
                        "strike": strike,
                        "atm_source": src,
                        "path": qpath,
                    }
                )
            else:
                if not ticker or print_paths is None or ticker not in print_paths:
                    continue
                pts, plast = print_paths[ticker]
                entries.append(
                    {
                        "date": date,
                        "clock": clock,
                        "dir": direction,
                        "from_open": from_open,
                        "entry_ts": to_ny(ts[i]),
                        "ticker": ticker,
                        "strike": strike,
                        "atm_source": src,
                        "pts": pts,
                        "plast": plast,
                    }
                )
    return entries


def _simulate_entries(
    *,
    book: str,
    entries: list[dict[str, Any]],
    tp: float,
    sl: float,
    max_sp: float | None,
    max_lag: float | None,
    max_hold_sec: int,
    fill: FillSpec,
    slip: float,
    min_mid: float,
) -> list[dict[str, Any]]:
    """One TP/SL pass over all clock hits (FO filtered later)."""
    out: list[dict[str, Any]] = []
    for e in entries:
        if book == "quote":
            assert max_sp is not None and max_lag is not None
            probe = entry_quote_row(
                e["path"],
                e["entry_ts"],
                max_lag_sec=max_lag,
                max_spread_pct=max_sp,
                min_mid=min_mid,
            )
            if probe is None:
                continue
            if float(probe["spread_pct"]) > max_sp or float(probe["lag_sec"]) > max_lag:
                continue
            sim = simulate_quote_tpsl(
                e["path"],
                e["entry_ts"],
                tp=tp,
                sl=sl,
                max_hold_sec=max_hold_sec,
                fill=fill,
                max_lag_sec=max_lag,
                max_spread_pct=max_sp,
                min_mid=min_mid,
            )
        else:
            sim = simulate_trade_tpsl(
                e["pts"],
                e["plast"],
                e["entry_ts"],
                tp=tp,
                sl=sl,
                max_hold_sec=max_hold_sec,
                slip=slip,
            )
        if sim is None or not np.isfinite(sim["ret"]):
            continue
        et = e["entry_ts"]
        hold = float(sim.get("hold_sec") or 0.0)
        out.append(
            {
                "date": e["date"],
                "symbol": "QQQ",
                "dir": e["dir"],
                "clock": e["clock"],
                "entry_ts": str(et),
                "exit_ts": str(et + pd.Timedelta(seconds=hold)),
                "ticker": e.get("ticker"),
                "ret": sim["ret"],
                "exit_reason": sim.get("reason") or sim.get("exit_reason"),
                "hold_sec": hold,
                "from_open": float(e["from_open"]),
            }
        )
    return out


def _score_book(
    *,
    book: str,
    entries: list[dict[str, Any]],
    fo_mins: list[float],
    tps: list[float],
    sls: list[float],
    spreads: list[float],
    lags: list[float],
    max_hold_sec: int,
    fill: FillSpec,
    slip: float,
    min_mid: float,
    position_frac: float,
    concurrents: list[int],
    cooldowns: list[float],
    min_n: int,
    windows: list[tuple[str, str, str]],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    score_rows: list[dict[str, Any]] = []
    dual_ok: list[dict[str, Any]] = []

    if book == "quote":
        gate_iter = [(sp, lg) for sp in spreads for lg in lags]
    else:
        gate_iter = [(None, None)]

    for max_sp, max_lag in gate_iter:
        for tp in tps:
            for sl in sls:
                print(
                    f"[{book}] simulate tp{tp}/sl{sl} sp={max_sp} lag={max_lag} "
                    f"hits={len(entries)}",
                    flush=True,
                )
                filled = _simulate_entries(
                    book=book,
                    entries=entries,
                    tp=tp,
                    sl=sl,
                    max_sp=max_sp,
                    max_lag=max_lag,
                    max_hold_sec=max_hold_sec,
                    fill=fill,
                    slip=slip,
                    min_mid=min_mid,
                )
                print(f"[{book}] filled={len(filled)}", flush=True)
                for fo in fo_mins:
                    ents_fo = [e for e in filled if abs(float(e["from_open"])) >= fo]
                    for c in concurrents:
                        for cd in cooldowns:
                            win_stats: dict[str, dict[str, Any]] = {}
                            for wname, w0, w1 in windows:
                                raw = [e for e in ents_fo if w0 <= e["date"] <= w1]
                                st = _port(
                                    raw,
                                    position_frac=position_frac,
                                    max_concurrent=c,
                                    cooldown_minutes=cd,
                                )
                                win_stats[wname] = st
                                if st.get("n", 0) >= 20:
                                    print(
                                        f"[{book} fo≥{fo} c={c} cd={cd} {wname}] "
                                        f"n={st['n']} mean={st['mean']} add={st['add']:+.3f} "
                                        f"day_win={st['day_win']} tpd={st['trades_per_day']}",
                                        flush=True,
                                    )
                            both = all(_ok(win_stats[w[0]], min_n=min_n) for w in windows)
                            row: dict[str, Any] = {
                                "book": book,
                                "from_open_min": fo,
                                "tp": tp,
                                "sl": sl,
                                "max_spread_pct": max_sp,
                                "max_lag_sec": max_lag,
                                "max_concurrent": c,
                                "cooldown_minutes": cd,
                                "position_frac": position_frac,
                                "dual_pass": both,
                            }
                            for wname, _, _ in windows:
                                for k, v in win_stats[wname].items():
                                    row[f"{wname}_{k}"] = v
                            score_rows.append(row)
                            if both:
                                dual_ok.append(row)
                                print(
                                    f"  *** DUAL PASS {book} fo≥{fo} tp{tp}/sl{sl} "
                                    f"c={c} cd={cd}",
                                    flush=True,
                                )

    dual_ok.sort(
        key=lambda r: float(r.get("may_jul_add") or 0) + float(r.get("jan_mar_add") or 0),
        reverse=True,
    )
    return pd.DataFrame(score_rows), dual_ok


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--opt-root", default=str(DEFAULT_OPT))
    ap.add_argument("--tick-root", default=str(DEFAULT_TICK))
    ap.add_argument("--stock-1s-root", default=str(DEFAULT_STOCK))
    ap.add_argument("--results-dir", default=str(DEFAULT_RESULTS))
    ap.add_argument("--tag", default="research_qqq_multi_clock_20260728")
    ap.add_argument("--clock-start", default="09:35")
    ap.add_argument("--clock-end", default="15:15")
    ap.add_argument("--stride-min", type=int, default=15)
    ap.add_argument("--from-open-mins", default="0.0005,0.001,0.002")
    ap.add_argument("--tps", default="0.10")
    ap.add_argument("--sls", default="0.25")
    ap.add_argument("--max-spreads", default="0.15")
    ap.add_argument("--max-lags", default="2")
    ap.add_argument("--min-mid", type=float, default=0.05)
    ap.add_argument("--max-hold-sec", type=int, default=900)
    ap.add_argument("--entry-frac", type=float, default=0.75)
    ap.add_argument("--exit-frac", type=float, default=0.75)
    ap.add_argument("--slip", type=float, default=0.01)
    ap.add_argument("--position-frac", type=float, default=0.10)
    ap.add_argument("--max-concurrent", default="2,3")
    ap.add_argument("--cooldowns", default="0,5,15")
    ap.add_argument("--min-n", type=int, default=30)
    ap.add_argument("--books", default="quote,tick", help="comma: quote,tick")
    ap.add_argument("--end-date", default="2026-07-23")
    args = ap.parse_args(argv)

    opt_root = Path(args.opt_root)
    tick_root = Path(args.tick_root)
    stock_1s = Path(args.stock_1s_root)
    out = Path(args.results_dir) / args.tag
    out.mkdir(parents=True, exist_ok=True)

    clocks = _hhmm_range(args.clock_start, args.clock_end, int(args.stride_min))
    fo_mins = [float(x) for x in args.from_open_mins.split(",") if x.strip()]
    tps = [float(x) for x in args.tps.split(",") if x.strip()]
    sls = [float(x) for x in args.sls.split(",") if x.strip()]
    spreads = [float(x) for x in args.max_spreads.split(",") if x.strip()]
    lags = [float(x) for x in args.max_lags.split(",") if x.strip()]
    concurrents = [int(x) for x in args.max_concurrent.split(",") if x.strip()]
    cooldowns = [float(x) for x in args.cooldowns.split(",") if x.strip()]
    books = [x.strip() for x in args.books.split(",") if x.strip()]
    fill = FillSpec(entry_frac=float(args.entry_frac), exit_frac=float(args.exit_frac))

    windows = [(w[0], w[1], min(w[2], str(args.end_date)) if w[0] == "may_jul" else w[2]) for w in WINDOWS]
    # Quote book ends 06-30 — clip may_jul for quote scoring later.
    print(
        f"clocks={len(clocks)} {clocks[0]}..{clocks[-1]} fo={fo_mins} "
        f"c={concurrents} cd={cooldowns} books={books}",
        flush=True,
    )

    all_scores: list[pd.DataFrame] = []
    summary_books: dict[str, Any] = {}

    for book in books:
        if book == "quote":
            q_end = "2026-06-30"
            dates = [
                d
                for d in _discover_option_dates(opt_root, windows[0][1], q_end)
                if (stock_1s / "QQQ" / f"QQQ_{d}.parquet").is_file()
            ]
            book_windows = [
                (windows[0][0], windows[0][1], windows[0][2]),
                ("may_jul", "2026-05-01", q_end),
            ]
            entries = _build_raw_entries(
                dates=dates,
                clocks=clocks,
                stock_1s=stock_1s,
                opt_root=opt_root,
                tick_root=None,
                book="quote",
            )
        elif book == "tick":
            all_tick = tick_dates(tick_root, "QQQ")
            dates = [
                d
                for d in all_tick
                if windows[0][1] <= d <= str(args.end_date)
                and (stock_1s / "QQQ" / f"QQQ_{d}.parquet").is_file()
            ]
            book_windows = list(windows)
            entries = _build_raw_entries(
                dates=dates,
                clocks=clocks,
                stock_1s=stock_1s,
                opt_root=opt_root,
                tick_root=tick_root,
                book="tick",
            )
        else:
            raise SystemExit(f"unknown book {book}")

        print(f"[{book}] dates={len(dates)} raw_clock_hits={len(entries)}", flush=True)
        score_df, book_dual = _score_book(
            book=book,
            entries=entries,
            fo_mins=fo_mins,
            tps=tps,
            sls=sls,
            spreads=spreads,
            lags=lags,
            max_hold_sec=int(args.max_hold_sec),
            fill=fill,
            slip=float(args.slip),
            min_mid=float(args.min_mid),
            position_frac=float(args.position_frac),
            concurrents=concurrents,
            cooldowns=cooldowns,
            min_n=int(args.min_n),
            windows=book_windows,
        )
        if not score_df.empty:
            score_df.to_csv(out / f"scoreboard_{book}.csv", index=False)
            all_scores.append(score_df)
        book_dual.sort(
            key=lambda r: float(r.get("may_jul_add") or 0) + float(r.get("jan_mar_add") or 0),
            reverse=True,
        )
        summary_books[book] = {
            "dates": {"n": len(dates), "start": dates[0] if dates else None, "end": dates[-1] if dates else None},
            "raw_clock_hits": len(entries),
            "n_dual_pass": len(book_dual),
            "verdict": "PASS" if book_dual else "REJECT",
            "dual_pass": book_dual[:20],
        }
        print(f"[{book}] dual_pass={len(book_dual)} verdict={summary_books[book]['verdict']}", flush=True)

    if all_scores:
        pd.concat(all_scores, ignore_index=True).to_csv(out / "scoreboard_all.csv", index=False)

    summary = {
        "symbol": "QQQ",
        "entry": "multi_clock_from_open",
        "clocks": clocks,
        "from_open_mins": fo_mins,
        "tps": tps,
        "sls": sls,
        "max_concurrent": concurrents,
        "cooldowns": cooldowns,
        "position_frac": float(args.position_frac),
        "min_n": int(args.min_n),
        "books": summary_books,
        "verdict": (
            "PASS"
            if all(summary_books.get(b, {}).get("verdict") == "PASS" for b in books)
            else "PARTIAL"
            if any(summary_books.get(b, {}).get("verdict") == "PASS" for b in books)
            else "REJECT"
        ),
        "note": (
            "Full-session multi-fire QQQ: both CALL/PUT by from_open sign; "
            "same-day stacking allowed via max_concurrent; FO grid lowered."
        ),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("verdict", "books") if k in summary}, indent=2, default=str), flush=True)
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

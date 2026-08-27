#!/usr/bin/env python3
"""QQQ open_cont × print-last TP/SL dual-window accept (AM clocks).

Stock direction from causal 1s ``from_open`` at clock (default 09:45).
Pricing: ``/mnt/s990/new_option_data_s3_trades`` last ± slip (covers Jul).
Raw ``/mnt/s990/new_option_data_s3_tick`` is also accepted (``price`` column).
ATM: prefer dte0 bucket ticker when available; else OCC 0DTE closest-to-spot
from the print file (needed after 2026-06-30 when quote dte0 ends).

Dual PASS (both windows):
  mean>0, add>0, day_win≥0.55, n≥15, frac_max_hold≤0.50

Example:
  PYTHONPATH=. python -m maga7.tools.scan_qqq_open_cont_trades_tpsl \\
    --tag research_qqq_open_cont_trades_tpsl_dual
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
from maga7.common.option_trade_tpsl import simulate_trade_tpsl
from maga7.common.option_trades import load_option_trades
from maga7.common.replay import to_ny
from maga7.tools.run_morning_sec_option_fill import _portfolio_day
from maga7.tools.run_morning_sec_qqq_dte1 import _load_atm_path
from maga7.tools.scan_morning_sec_edge import _morning_slice
from maga7.tools.scan_session_horizon_foresight import _paths_by_ticker

NY = "America/New_York"
DEFAULT_TRADES = Path("/mnt/s990/new_option_data_s3_trades")
DEFAULT_OPT = Path("/mnt/s990/data/raw_1s/dte0_options/QQQ")
DEFAULT_STOCK = Path("/mnt/s990/data/raw_1s/stocks")
DEFAULT_RESULTS = Path("/mnt/s990/data/maga7/results")

WINDOWS = (
    ("jan_mar", "2026-01-02", "2026-03-31"),
    ("may_jul", "2026-05-01", "2026-07-22"),
)

_OCC = re.compile(
    r"^O?:?(?P<root>[A-Z]+)(?P<yy>\d{2})(?P<mm>\d{2})(?P<dd>\d{2})"
    r"(?P<cp>[CP])(?P<strike>\d{8})$"
)


def _port(rows: list[dict[str, Any]]) -> dict[str, Any]:
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
        }
    by: dict[str, list] = {}
    for r in rows:
        by.setdefault(str(r["date"]), []).append(r)
    sized: list[dict] = []
    for d in sorted(by):
        sized.extend(
            _portfolio_day(by[d], position_frac=0.10, max_concurrent=1, cooldown_minutes=0.0)
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
        }
    t = pd.DataFrame(sized)
    t["pnl_frac"] = t["ret"].astype(float) * t["size"].astype(float)
    day = t.groupby("date")["pnl_frac"].sum()
    reasons = pd.Series([r.get("exit_reason") for r in sized])
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


def _atm_ticker_from_trades(
    trade_paths: dict[str, tuple[np.ndarray, np.ndarray]],
    *,
    date: str,
    direction: str,
    spot: float,
) -> tuple[str | None, float | None]:
    """0DTE ATM: same-day OCC expiry, CP by dir, strike closest to spot."""
    if not trade_paths or not np.isfinite(spot) or spot <= 0:
        return None, None
    ymd = date.replace("-", "")[2:]  # YYMMDD
    want_cp = "C" if direction == "UP" else "P"
    best_t: str | None = None
    best_k: float | None = None
    best_abs = float("inf")
    for raw in trade_paths:
        key = str(raw).replace("O:", "")
        m = _OCC.match(key)
        if m is None:
            continue
        if m.group("root") != "QQQ":
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
    trade_paths: dict[str, tuple[np.ndarray, np.ndarray]],
) -> tuple[str | None, float | None, str]:
    """Return (ticker_key, strike, source)."""
    path, ticker, strike = _load_atm_path(opt_root, date, direction)
    if ticker:
        key = str(ticker).replace("O:", "")
        if key in trade_paths:
            return key, float(strike) if strike is not None else None, "dte0_bucket"
        if str(ticker) in trade_paths:
            return str(ticker), float(strike) if strike is not None else None, "dte0_bucket"
    t2, k2 = _atm_ticker_from_trades(trade_paths, date=date, direction=direction, spot=spot)
    if t2:
        return t2, k2, "trades_occ"
    return None, None, "miss"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trades-root", default=str(DEFAULT_TRADES))
    ap.add_argument("--opt-root", default=str(DEFAULT_OPT), help="dte0 quote for ATM id when present")
    ap.add_argument("--stock-1s-root", default=str(DEFAULT_STOCK))
    ap.add_argument("--results-dir", default=str(DEFAULT_RESULTS))
    ap.add_argument("--tag", default="research_qqq_open_cont_trades_tpsl_dual")
    ap.add_argument("--clocks", default="09:45", help="comma clocks in AM, e.g. 09:40,09:45,09:50")
    ap.add_argument("--from-open-mins", default="0,0.002,0.003,0.005")
    ap.add_argument("--tps", default="0.05,0.10,0.15,0.20")
    ap.add_argument("--sls", default="0.10,0.15,0.25")
    ap.add_argument("--slip", type=float, default=0.01)
    ap.add_argument("--max-hold-sec", type=int, default=900)
    ap.add_argument("--min-n", type=int, default=15)
    ap.add_argument(
        "--end-date",
        default=None,
        help="Override the late-window end date (for newly arrived print days)",
    )
    ap.add_argument(
        "--data-kind",
        choices=("auto", "trades", "tick"),
        default="auto",
        help="Label the print source; auto infers it from the root name",
    )
    ap.add_argument(
        "--include-fade-null",
        action="store_true",
        help="Also score fade (opposite of from_open) as open-drift null",
    )
    args = ap.parse_args(argv)

    trades_root = Path(args.trades_root)
    opt_root = Path(args.opt_root)
    stock_1s = Path(args.stock_1s_root)
    out = Path(args.results_dir) / args.tag
    out.mkdir(parents=True, exist_ok=True)

    clocks = [x.strip() for x in args.clocks.split(",") if x.strip()]
    fo_mins = [float(x) for x in args.from_open_mins.split(",") if x.strip()]
    tps = [float(x) for x in args.tps.split(",") if x.strip()]
    sls = [float(x) for x in args.sls.split(",") if x.strip()]
    modes = ["cont"] + (["fade"] if args.include_fade_null else [])
    data_kind = str(args.data_kind)
    if data_kind == "auto":
        data_kind = "tick" if "tick" in trades_root.name.lower() else "trades"
    windows = list(WINDOWS)
    if args.end_date:
        windows[-1] = (windows[-1][0], windows[-1][1], str(args.end_date))

    start_all = min(w[1] for w in windows)
    end_all = max(w[2] for w in windows)
    # Prefer trades calendar (includes Jul); intersect stock.
    trade_dates: list[str] = []
    for p in sorted((trades_root / "QQQ").glob("QQQ_*.parquet")):
        d = p.stem.split("_", 1)[1]
        if start_all <= d <= end_all:
            trade_dates.append(d)
    dates = [d for d in trade_dates if (stock_1s / "QQQ" / f"QQQ_{d}.parquet").is_file()]
    if not dates:
        raise SystemExit("no overlapping QQQ stock+trades days")
    print(
        f"QQQ open_cont trades dual dates={len(dates)} {dates[0]}..{dates[-1]} "
        f"clocks={clocks} modes={modes}",
        flush=True,
    )

    # Build candidate entries once.
    entries: list[dict[str, Any]] = []
    n_miss = 0
    src_counts: dict[str, int] = {}
    for di, date in enumerate(dates):
        if di % 20 == 0:
            print(f"[day] {date} ({di+1}/{len(dates)}) entries={len(entries)} miss={n_miss}", flush=True)
        day = load_stock_1s_day(stock_1s, "QQQ", date)
        buf = _morning_slice(day, start="09:30", end="16:00")
        if buf.empty:
            n_miss += 1
            continue
        ts = pd.DatetimeIndex(pd.to_datetime(buf["timestamp"]))
        if ts.tz is None:
            ts = ts.tz_localize(NY, ambiguous="infer")
        else:
            ts = ts.tz_convert(NY)
        close = buf["close"].astype(float).to_numpy()
        open_px = float(close[0])
        tday = load_option_trades(trades_root, "QQQ", date)
        if tday is None or tday.empty:
            n_miss += 1
            continue
        trade_paths = _paths_by_ticker(tday)
        if not trade_paths:
            n_miss += 1
            continue

        for clock in clocks:
            t0 = pd.Timestamp(f"{date} {clock}", tz=NY)
            i = int(ts.searchsorted(t0, side="left"))
            if i >= len(close) - 1:
                continue
            # require print within 5s of clock
            if abs((ts[i] - t0).total_seconds()) > 5:
                continue
            spot = float(close[i])
            from_open = float((spot - open_px) / open_px) if open_px else 0.0
            if from_open == 0.0:
                continue
            cont_dir = "UP" if from_open > 0 else "DN"
            for mode in modes:
                direction = cont_dir if mode == "cont" else ("DN" if cont_dir == "UP" else "UP")
                ticker, strike, src = _resolve_atm(
                    date=date,
                    direction=direction,
                    spot=spot,
                    opt_root=opt_root,
                    trade_paths=trade_paths,
                )
                if not ticker or ticker not in trade_paths:
                    n_miss += 1
                    continue
                src_counts[src] = src_counts.get(src, 0) + 1
                pts, plast = trade_paths[ticker]
                entries.append(
                    {
                        "date": date,
                        "clock": clock,
                        "mode": mode,
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

    print(
        f"entries={len(entries)} miss_days_or_atm≈{n_miss} atm_src={src_counts}",
        flush=True,
    )

    score_rows: list[dict[str, Any]] = []
    dual_pass_list: list[dict[str, Any]] = []
    trade_dump: dict[str, pd.DataFrame] = {}

    for clock in clocks:
        for mode in modes:
            for fo in fo_mins:
                for tp in tps:
                    for sl in sls:
                        win_stats: dict[str, dict[str, Any]] = {}
                        win_raw: dict[str, list[dict[str, Any]]] = {}
                        for wname, w0, w1 in windows:
                            raw: list[dict[str, Any]] = []
                            for e in entries:
                                if e["clock"] != clock or e["mode"] != mode:
                                    continue
                                if not (w0 <= e["date"] <= w1):
                                    continue
                                if abs(float(e["from_open"])) < fo:
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
                                        "exit_ts": str(
                                            et + pd.Timedelta(seconds=sim["hold_sec"])
                                        ),
                                        "ticker": e["ticker"],
                                        "ret": sim["ret"],
                                        "exit_reason": sim["reason"],
                                        "hold_sec": sim["hold_sec"],
                                        "from_open": e["from_open"],
                                        "window": wname,
                                        "mode": mode,
                                        "clock": clock,
                                    }
                                )
                            st = _port(raw)
                            win_stats[wname] = st
                            win_raw[wname] = raw
                            if st.get("n", 0) >= 10:
                                print(
                                    f"[{clock} {mode} fo≥{fo} tp{tp}/sl{sl} {wname}] "
                                    f"n={st['n']} mean={st['mean']} add={st['add']:+.3f} "
                                    f"day_win={st['day_win']}",
                                    flush=True,
                                )

                        both = _ok(win_stats["jan_mar"], min_n=int(args.min_n)) and _ok(
                            win_stats["may_jul"], min_n=int(args.min_n)
                        )
                        row: dict[str, Any] = {
                            "clock": clock,
                            "mode": mode,
                            "from_open_min": fo,
                            "tp": tp,
                            "sl": sl,
                            "max_hold_sec": int(args.max_hold_sec),
                            "slip": float(args.slip),
                            "dual_pass": both,
                        }
                        for wname, _, _ in windows:
                            for k, v in win_stats[wname].items():
                                row[f"{wname}_{k}"] = v
                        score_rows.append(row)
                        if both:
                            key = f"{clock}|{mode}|fo{fo}|tp{tp}|sl{sl}"
                            dual_pass_list.append(row)
                            trade_dump[key] = pd.DataFrame(
                                win_raw["jan_mar"] + win_raw["may_jul"]
                            )
                            print(f"  *** DUAL PASS {key}", flush=True)

    score = pd.DataFrame(score_rows)
    score.to_csv(out / "scoreboard.csv", index=False)
    dual_pass_list = sorted(
        dual_pass_list,
        key=lambda r: (
            float(r.get("may_jul_add") or 0) + float(r.get("jan_mar_add") or 0)
        ),
        reverse=True,
    )
    for i, p in enumerate(dual_pass_list[:12]):
        key = f"{p['clock']}|{p['mode']}|fo{p['from_open_min']}|tp{p['tp']}|sl{p['sl']}"
        if key in trade_dump:
            trade_dump[key].to_csv(
                out
                / (
                    f"trades_dual{i}_{p['mode']}_{p['clock'].replace(':', '')}"
                    f"_fo{p['from_open_min']}_tp{p['tp']}_sl{p['sl']}.csv"
                ),
                index=False,
            )

    verdict = "PASS" if dual_pass_list else "REJECT"
    # cont-only champions for headline
    cont_pass = [r for r in dual_pass_list if r.get("mode") == "cont"]
    fade_pass = [r for r in dual_pass_list if r.get("mode") == "fade"]
    summary = {
        "symbol": "QQQ",
        "entry": "open_cont",
        "session": "AM_clocks",
        "pricing": f"{data_kind}_last",
        "print_root": str(trades_root),
        "slip": float(args.slip),
        "exit": "tp_sl_first_passage_trade_last",
        "dates": {"n": len(dates), "start": dates[0], "end": dates[-1]},
        "atm_source_counts": src_counts,
        "windows": [list(w) for w in windows],
        "clocks": clocks,
        "modes": modes,
        "n_entries": int(len(entries)),
        "dual_pass_n": int(len(dual_pass_list)),
        "dual_pass_cont_n": int(len(cont_pass)),
        "dual_pass_fade_n": int(len(fade_pass)),
        "verdict": verdict,
        "dual_pass": dual_pass_list[:30],
        "note": (
            f"QQQ open continuation on {data_kind}-last ± slip. "
            "fade mode is open-drift null when --include-fade-null. "
            "Not a quote-executable promote by itself."
        ),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    (out / "dual_pass.json").write_text(
        json.dumps(dual_pass_list[:30], indent=2, default=str), encoding="utf-8"
    )
    print(f"\n=== dual PASS ({len(dual_pass_list)}) cont={len(cont_pass)} "
          f"fade={len(fade_pass)} verdict={verdict} ===", flush=True)
    print(json.dumps(dual_pass_list[:12], indent=2, default=str), flush=True)
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

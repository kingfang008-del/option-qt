#!/usr/bin/env python3
"""QQQ 0DTE open_cont (09:45) + quote FillSpec TP/SL dual-window accept.

Entry: at clock, |from_open| ≥ fo_min → direction; ATM 0DTE path.
Gates: quote lag / spread / mid. Exit: first-passage TP/SL (no clock primary).

Example:
  PYTHONPATH=. python -m maga7.tools.scan_qqq_open_cont_quote_tpsl \\
    --tag research_qqq_open_cont_quote_tpsl_dual
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
from maga7.common.fills import FillSpec
from maga7.common.option_quote_tpsl import entry_quote_row, simulate_quote_tpsl
from maga7.common.replay import to_ny
from maga7.tools.run_morning_sec_option_fill import _portfolio_day
from maga7.tools.run_morning_sec_qqq_dte1 import _discover_option_dates, _load_atm_path
from maga7.tools.scan_morning_sec_edge import _bdates, _morning_slice, _prior_close

NY = "America/New_York"
DEFAULT_OPT = Path("/mnt/s990/data/raw_1s/dte0_options/QQQ")
DEFAULT_STOCK = Path("/mnt/s990/data/raw_1s/stocks")
DEFAULT_RESULTS = Path("/mnt/s990/data/maga7/results")

WINDOWS = (
    ("jan_mar", "2026-01-02", "2026-03-31"),
    ("may_jul", "2026-05-01", "2026-07-22"),
)


def _port(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"n": 0, "mean": None, "win": None, "add": 0.0, "day_win": None}
    by: dict[str, list] = {}
    for r in rows:
        by.setdefault(str(r["date"]), []).append(r)
    sized: list[dict] = []
    for d in sorted(by):
        sized.extend(
            _portfolio_day(by[d], position_frac=0.10, max_concurrent=1, cooldown_minutes=0.0)
        )
    if not sized:
        return {"n": 0, "mean": None, "win": None, "add": 0.0, "day_win": None}
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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--opt-root", default=str(DEFAULT_OPT))
    ap.add_argument("--stock-1s-root", default=str(DEFAULT_STOCK))
    ap.add_argument("--results-dir", default=str(DEFAULT_RESULTS))
    ap.add_argument("--tag", default="research_qqq_open_cont_quote_tpsl_dual")
    ap.add_argument("--clock", default="09:45")
    ap.add_argument("--from-open-mins", default="0,0.002,0.003,0.005")
    ap.add_argument("--tps", default="0.10,0.15,0.20,0.30")
    ap.add_argument("--sls", default="0.10,0.15,0.25")
    ap.add_argument("--max-spreads", default="0.05,0.08,0.10,0.15")
    ap.add_argument("--max-lags", default="2,3,5")
    ap.add_argument("--min-mid", type=float, default=0.05)
    ap.add_argument("--max-hold-sec", type=int, default=900)
    ap.add_argument("--entry-frac", type=float, default=0.75)
    ap.add_argument("--exit-frac", type=float, default=0.75)
    args = ap.parse_args(argv)

    opt_root = Path(args.opt_root)
    stock_1s = Path(args.stock_1s_root)
    out = Path(args.results_dir) / args.tag
    out.mkdir(parents=True, exist_ok=True)
    fill = FillSpec(entry_frac=float(args.entry_frac), exit_frac=float(args.exit_frac))

    fo_mins = [float(x) for x in args.from_open_mins.split(",") if x.strip()]
    tps = [float(x) for x in args.tps.split(",") if x.strip()]
    sls = [float(x) for x in args.sls.split(",") if x.strip()]
    spreads = [float(x) for x in args.max_spreads.split(",") if x.strip()]
    lags = [float(x) for x in args.max_lags.split(",") if x.strip()]
    clock = str(args.clock)

    # Load all days covering both windows.
    start_all = min(w[1] for w in WINDOWS)
    end_all = max(w[2] for w in WINDOWS)
    dates = [
        d
        for d in _discover_option_dates(opt_root, start_all, end_all)
        if (stock_1s / "QQQ" / f"QQQ_{d}.parquet").is_file()
    ]
    if not dates:
        raise SystemExit("no overlapping QQQ days")
    all_bd = _bdates(dates[0], dates[-1])
    print(f"dates={len(dates)} {dates[0]}..{dates[-1]}", flush=True)

    day_cache: dict[str, dict[str, Any]] = {}
    for date in dates:
        day = load_stock_1s_day(stock_1s, "QQQ", date)
        buf = _morning_slice(day, start="09:30", end="16:00")
        if buf.empty:
            continue
        ts = pd.DatetimeIndex(pd.to_datetime(buf["timestamp"]))
        if ts.tz is None:
            ts = ts.tz_localize(NY, ambiguous="infer")
        else:
            ts = ts.tz_convert(NY)
        close = buf["close"].astype(float).to_numpy()
        day_cache[date] = {
            "ts": ts,
            "close": close,
            "open": float(close[0]),
            "prev": _prior_close(stock_1s, "QQQ", date, all_bd),
        }

    path_cache: dict[tuple[str, str], Any] = {}

    def get_path(date: str, direction: str):
        key = (date, direction)
        if key not in path_cache:
            path_cache[key] = _load_atm_path(opt_root, date, direction)
        return path_cache[key]

    # Collect entries for all fo_mins (widest later filtered).
    entries_by_fo: dict[float, list[dict[str, Any]]] = {fo: [] for fo in fo_mins}
    for date, d in day_cache.items():
        ts, close, open_px = d["ts"], d["close"], d["open"]
        t0 = pd.Timestamp(f"{date} {clock}", tz=NY)
        i = int(ts.searchsorted(t0, side="left"))
        if i >= len(close) - 1:
            continue
        from_open = float((close[i] - open_px) / open_px) if open_px else 0.0
        direction = "UP" if from_open > 0 else "DN"
        path, ticker, strike = get_path(date, direction)
        if path is None or path.empty or strike is None:
            continue
        entry_ts = to_ny(ts[i])
        probe = entry_quote_row(
            path,
            entry_ts,
            max_lag_sec=max(lags),
            max_spread_pct=max(spreads),
            min_mid=float(args.min_mid),
        )
        if probe is None:
            continue
        base = {
            "date": date,
            "dir": direction,
            "from_open": from_open,
            "entry_ts": entry_ts,
            "path": path,
            "ticker": ticker,
            "strike": float(strike),
            "entry_spread_pct": probe["spread_pct"],
            "entry_lag_sec": probe["lag_sec"],
        }
        for fo in fo_mins:
            if abs(from_open) >= fo:
                entries_by_fo[fo].append(base)
    for fo, ents in entries_by_fo.items():
        print(f"fo>={fo}: entries={len(ents)}", flush=True)

    score_rows: list[dict[str, Any]] = []
    for wname, w0, w1 in WINDOWS:
        for fo in fo_mins:
            ents = [e for e in entries_by_fo[fo] if w0 <= e["date"] <= w1]
            for max_sp in spreads:
                for max_lag in lags:
                    for tp in tps:
                        for sl in sls:
                            raw: list[dict[str, Any]] = []
                            for e in ents:
                                if float(e["entry_spread_pct"]) > max_sp:
                                    continue
                                if float(e["entry_lag_sec"]) > max_lag:
                                    continue
                                sim = simulate_quote_tpsl(
                                    e["path"],
                                    e["entry_ts"],
                                    tp=tp,
                                    sl=sl,
                                    max_hold_sec=int(args.max_hold_sec),
                                    fill=fill,
                                    max_lag_sec=max_lag,
                                    max_spread_pct=max_sp,
                                    min_mid=float(args.min_mid),
                                )
                                if sim is None or not np.isfinite(sim["ret"]):
                                    continue
                                raw.append(
                                    {
                                        "date": e["date"],
                                        "symbol": "QQQ",
                                        "dir": e["dir"],
                                        "entry_ts": str(sim["entry_ts"]),
                                        "exit_ts": str(sim["exit_ts"]),
                                        "ticker": e["ticker"],
                                        "ret": sim["ret"],
                                        "exit_reason": sim["reason"],
                                        "hold_sec": sim["hold_sec"],
                                    }
                                )
                            st = _port(raw)
                            score_rows.append(
                                {
                                    "window": wname,
                                    "start": w0,
                                    "end": w1,
                                    "clock": clock,
                                    "from_open_min": fo,
                                    "max_spread_pct": max_sp,
                                    "max_lag_sec": max_lag,
                                    "tp": tp,
                                    "sl": sl,
                                    "n_entries": int(len(ents)),
                                    **st,
                                }
                            )
                            if st.get("n", 0) >= 15:
                                print(
                                    f"[{wname} fo≥{fo} sp≤{max_sp} lag≤{max_lag} "
                                    f"tp{tp}/sl{sl}] n={st['n']} mean={st['mean']} "
                                    f"add={st['add']:+.3f} day_win={st['day_win']}",
                                    flush=True,
                                )

    score = pd.DataFrame(score_rows)
    score.to_csv(out / "scoreboard.csv", index=False)

    picks = []
    if len(score):
        ok = score[
            (score["mean"].fillna(-1) > 0)
            & (score["add"].fillna(0) > 0)
            & (score["day_win"].fillna(0) >= 0.55)
            & (score["n"].fillna(0) >= 20)
            & (score["frac_max_hold"].fillna(1) <= 0.50)
        ].sort_values(["window", "add"], ascending=[True, False])
        picks = ok.to_dict(orient="records")

    dual_ok: list[dict[str, Any]] = []
    if len(score):
        keys = ["from_open_min", "max_spread_pct", "max_lag_sec", "tp", "sl"]
        # Merge (not MultiIndex) — float keys like 0.002 break set_index intersection.
        a = score[score["window"] == "may_jul"][keys + ["n", "mean", "add", "day_win", "frac_max_hold"]].copy()
        b = score[score["window"] == "jan_mar"][keys + ["n", "mean", "add", "day_win", "frac_max_hold"]].copy()
        m = a.merge(b, on=keys, suffixes=("_mj", "_jm"))
        for _, r in m.iterrows():
            if (
                float(r["n_mj"]) >= 20
                and float(r["n_jm"]) >= 20
                and float(r["mean_mj"]) > 0
                and float(r["mean_jm"]) > 0
                and float(r["add_mj"]) > 0
                and float(r["add_jm"]) > 0
                and float(r["day_win_mj"]) >= 0.55
                and float(r["day_win_jm"]) >= 0.55
                and float(r["frac_max_hold_mj"]) <= 0.50
                and float(r["frac_max_hold_jm"]) <= 0.50
            ):
                dual_ok.append(
                    {
                        "from_open_min": float(r["from_open_min"]),
                        "max_spread_pct": float(r["max_spread_pct"]),
                        "max_lag_sec": float(r["max_lag_sec"]),
                        "tp": float(r["tp"]),
                        "sl": float(r["sl"]),
                        "may_jul_n": int(r["n_mj"]),
                        "may_jul_mean": float(r["mean_mj"]),
                        "may_jul_add": float(r["add_mj"]),
                        "may_jul_day_win": float(r["day_win_mj"]),
                        "jan_mar_n": int(r["n_jm"]),
                        "jan_mar_mean": float(r["mean_jm"]),
                        "jan_mar_add": float(r["add_jm"]),
                        "jan_mar_day_win": float(r["day_win_jm"]),
                        "add_sum": float(r["add_mj"]) + float(r["add_jm"]),
                    }
                )
        dual_ok.sort(key=lambda x: x["add_sum"], reverse=True)

    summary = {
        "symbol": "QQQ",
        "entry": "open_cont_0945",
        "book": "quote_fill_tpsl",
        "windows": [list(w) for w in WINDOWS],
        "n_score_rows": int(len(score)),
        "n_picks_any_window": int(len(picks)),
        "n_dual_window_pass": int(len(dual_ok)),
        "dual_window_pass": dual_ok[:20],
        "picks_any_window": picks[:30],
        "top_by_add": (
            score.sort_values("add", ascending=False).head(25).to_dict(orient="records")
            if len(score)
            else []
        ),
        "verdict": "PASS" if dual_ok else "REJECT",
        "note": (
            "QQQ 0DTE ATM open_cont at 09:45; quote FillSpec TP/SL; no clock primary. "
            "Prior path_exit Feb–Jun used clock/trail — this is the accept gate."
        ),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    (out / "dual_pass.json").write_text(json.dumps(dual_ok[:30], indent=2, default=str), encoding="utf-8")
    (out / "picks.json").write_text(json.dumps(picks[:30], indent=2, default=str), encoding="utf-8")

    print(f"\n=== dual-window PASS ({len(dual_ok)}) verdict={summary['verdict']} ===", flush=True)
    print(json.dumps(dual_ok[:10], indent=2, default=str), flush=True)
    if len(score):
        cols = [
            "window",
            "from_open_min",
            "max_spread_pct",
            "max_lag_sec",
            "tp",
            "sl",
            "n",
            "mean",
            "add",
            "day_win",
            "frac_tp",
            "frac_sl",
            "frac_max_hold",
        ]
        print("\n=== top by add ===", flush=True)
        print(score.sort_values("add", ascending=False)[cols].head(20).to_string(index=False), flush=True)
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

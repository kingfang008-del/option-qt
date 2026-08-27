#!/usr/bin/env python3
"""Stock micro-state (f'/f''+SNR) → ATM quote scalp (TP/SL/90s timer).

Architecture (research MVP):
  underlying 1s local slope velocity/accel + dual-window confluence + SNR
  → ATM option FillSpec entry (spread/lag gates)
  → exit = first of +tp / −sl / hard timer (default 90–120s)

Does **not** fit option premium curves. Parked: ``scan_smooth_regress_quote_tpsl``.

Example:
  PYTHONPATH=. python -m maga7.tools.scan_micro_state_quote_scalp \\
    --universe both --tag research_micro_state_quote_scalp_dual
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
from maga7.common.fills import FillSpec
from maga7.common.open_lock import (
    load_multidte_lock_index,
    resolve_open_lock_contract,
    resolve_otm_rungs,
)
from maga7.common.option_quote_tpsl import entry_quote_row, simulate_quote_scalp
from maga7.common.replay import load_quotes, path_for_ticker, to_ny
from maga7.common.stock_micro_state import MicroStateConfig, detect_micro_edges
from maga7.tools.run_morning_sec_option_fill import _portfolio_day
from maga7.tools.run_morning_sec_qqq_dte1 import _discover_option_dates, _load_atm_path
from maga7.tools.scan_morning_sec_edge import _bdates

NY = "America/New_York"
FREEZE = (
    "maga7/CONFIG/strategy_profiles/"
    "single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1.json"
)
DEFAULT_QQQ_OPT = Path("/mnt/s990/data/raw_1s/dte0_options/QQQ")
MAG7 = ("NVDA", "TSLA", "AAPL", "AMZN", "META", "MSFT", "AMD", "GOOGL")
WINDOWS = (
    ("jan_mar", "2026-01-02", "2026-03-31"),
    ("may_jul", "2026-05-01", "2026-07-22"),
)


def _port(rows: list[dict[str, Any]], *, max_concurrent: int) -> dict[str, Any]:
    if not rows:
        return {"n": 0, "mean": None, "win": None, "add": 0.0, "day_win": None}
    by: dict[str, list] = {}
    for r in rows:
        by.setdefault(str(r["date"]), []).append(r)
    sized: list[dict] = []
    for d in sorted(by):
        sized.extend(
            _portfolio_day(
                by[d],
                position_frac=0.10,
                max_concurrent=int(max_concurrent),
                cooldown_minutes=2.0,
            )
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
        "frac_timer": float((reasons == "timer").mean()) if len(reasons) else None,
        "hold_p50": float(pd.Series([r.get("hold_sec") for r in sized]).median()),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", default=FREEZE)
    ap.add_argument("--universe", choices=("both", "maga7", "qqq"), default="both")
    ap.add_argument("--tag", default="research_micro_state_quote_scalp_dual")
    ap.add_argument("--qqq-opt-root", default=str(DEFAULT_QQQ_OPT))
    ap.add_argument("--short-secs", default="15,20")
    ap.add_argument("--long-secs", default="60")
    ap.add_argument("--min-snrs", default="2.0,2.5")
    ap.add_argument("--min-slope-bp", type=float, default=1.5)
    ap.add_argument("--stride-sec", type=int, default=5)
    ap.add_argument("--cooldown-sec", type=int, default=120)
    ap.add_argument("--scan-start", default="09:30")
    ap.add_argument("--scan-end", default="10:15")
    ap.add_argument("--timers", default="90,120")
    ap.add_argument("--tps", default="0.08,0.10,0.15")
    ap.add_argument("--sls", default="0.08,0.10,0.15")
    ap.add_argument("--max-spreads", default="0.08,0.15")
    ap.add_argument("--max-lag-sec", type=float, default=3.0)
    ap.add_argument("--min-mid", type=float, default=0.05)
    args = ap.parse_args(argv)

    prof = load_profile(args.profile)
    paths = prof["_paths"]
    stock_1s = Path(paths["stock_1s_root"])
    quote_root = Path(paths["quote_1s_root"])
    out = Path(paths["results_dir"]) / args.tag
    out.mkdir(parents=True, exist_ok=True)
    fill = FillSpec(entry_frac=0.75, exit_frac=0.75)

    short_secs = [int(x) for x in args.short_secs.split(",") if x.strip()]
    long_secs = [int(x) for x in args.long_secs.split(",") if x.strip()]
    min_snrs = [float(x) for x in args.min_snrs.split(",") if x.strip()]
    timers = [int(x) for x in args.timers.split(",") if x.strip()]
    tps = [float(x) for x in args.tps.split(",") if x.strip()]
    sls = [float(x) for x in args.sls.split(",") if x.strip()]
    spreads = [float(x) for x in args.max_spreads.split(",") if x.strip()]

    universes: list[tuple[str, list[str]]] = []
    if args.universe in ("both", "maga7"):
        universes.append(("maga7", list(MAG7)))
    if args.universe in ("both", "qqq"):
        universes.append(("qqq", ["QQQ"]))

    lock_path = Path(paths.get("open_locked_map") or paths.get("locked_map")).expanduser()
    multi_idx = load_multidte_lock_index(lock_path) if lock_path.is_file() else {}
    otm_rungs = resolve_otm_rungs(prof, default=3)
    qqq_opt = Path(args.qqq_opt_root)

    score_rows: list[dict[str, Any]] = []
    events_meta: dict[str, Any] = {}

    for uname, symbols in universes:
        start_all, end_all = WINDOWS[0][1], WINDOWS[1][2]
        if uname == "qqq":
            dates = [
                d
                for d in _discover_option_dates(qqq_opt, start_all, end_all)
                if (stock_1s / "QQQ" / f"QQQ_{d}.parquet").is_file()
            ]
        else:
            dates = [
                d
                for d in _bdates(start_all, end_all)
                if any((stock_1s / s / f"{s}_{d}.parquet").is_file() for s in symbols)
            ]
        print(f"\n===== {uname} dates={len(dates)} AM {args.scan_start}-{args.scan_end} =====", flush=True)

        ev_rows: list[dict] = []
        for di, date in enumerate(dates):
            if di % 10 == 0:
                print(f"[{uname} events] {date} ({di+1}/{len(dates)}) n={len(ev_rows)}", flush=True)
            for sym in symbols:
                day = load_stock_1s_day(stock_1s, sym, date)
                if day is None or day.empty:
                    continue
                for ws in short_secs:
                    for wl in long_secs:
                        if wl <= ws:
                            continue
                        for snr in min_snrs:
                            cfg = MicroStateConfig(
                                short_sec=ws,
                                long_sec=wl,
                                stride_sec=int(args.stride_sec),
                                min_snr=float(snr),
                                min_slope_bp_per_min=float(args.min_slope_bp),
                                scan_start=args.scan_start,
                                scan_end=args.scan_end,
                                cooldown_sec=int(args.cooldown_sec),
                            )
                            for ev in detect_micro_edges(day, symbol=sym, date=date, cfg=cfg):
                                ev = dict(ev)
                                ev["min_snr"] = float(snr)
                                ev["cell"] = f"s{ws}_l{wl}_snr{snr:.1f}"
                                ev_rows.append(ev)
        events = pd.DataFrame(ev_rows)
        if events.empty:
            print(f"[{uname}] no events", flush=True)
            continue
        ev_save = events.copy()
        ev_save["ts"] = ev_save["ts"].astype(str)
        ev_path = out / f"events_{uname}.parquet"
        ev_save.to_parquet(ev_path, index=False)
        events_meta[uname] = {"n": int(len(events)), "path": str(ev_path)}
        print(f"[{uname}] events={len(events)}", flush=True)

        max_sp = max(spreads)
        uniq = events.drop_duplicates(["date", "symbol", "dir", "ts"])
        quote_cache: dict[tuple[str, str], Any] = {}
        qqq_cache: dict[tuple[str, str], Any] = {}
        resolved: dict[tuple[str, str, str, str], dict[str, Any]] = {}
        n_miss = 0
        for i, r in enumerate(uniq.itertuples(index=False)):
            if i % 200 == 0:
                print(f"[{uname} resolve] {i}/{len(uniq)} ok={len(resolved)} miss={n_miss}", flush=True)
            date, sym, direction = str(r.date), str(r.symbol), str(r.dir)
            entry_ts = to_ny(r.ts)
            spot = float(r.entry_px) if pd.notna(r.entry_px) else None
            if uname == "qqq":
                key = (date, direction)
                if key not in qqq_cache:
                    qqq_cache[key] = _load_atm_path(qqq_opt, date, direction)
                path, ticker, _ = qqq_cache[key]
                dte = 0
            else:
                qkey = (sym, date)
                if qkey not in quote_cache:
                    quote_cache[qkey] = load_quotes(quote_root, sym, date)
                qday = quote_cache[qkey]
                if qday is None or qday.empty:
                    n_miss += 1
                    continue
                ticker, dte, _ = resolve_open_lock_contract(
                    multi_idx.get((sym, date)),
                    direction=direction,
                    moneyness="ATM",
                    spot=spot,
                    prefer_dte=0,
                    allowed_dte=[0, 1, 2],
                    clear_otm_thresh=0.01,
                    ladder=True,
                    otm_rungs=otm_rungs,
                )
                if not ticker:
                    n_miss += 1
                    continue
                path = path_for_ticker(qday, ticker)
            if path is None or (hasattr(path, "empty") and path.empty):
                n_miss += 1
                continue
            probe = entry_quote_row(
                path,
                entry_ts,
                max_lag_sec=float(args.max_lag_sec),
                max_spread_pct=max_sp,
                min_mid=float(args.min_mid),
            )
            if probe is None:
                n_miss += 1
                continue
            resolved[(date, sym, direction, str(r.ts))] = {
                "date": date,
                "symbol": sym,
                "dir": direction,
                "sig_ts": entry_ts,
                "ticker": ticker,
                "dte": dte,
                "path": path,
                "entry_spread_pct": probe["spread_pct"],
            }
        print(f"[{uname}] resolved={len(resolved)} miss={n_miss}", flush=True)

        for wname, w0, w1 in WINDOWS:
            for cell in sorted(events["cell"].unique()):
                sub = events[
                    (events["cell"] == cell)
                    & (events["date"] >= w0)
                    & (events["date"] <= w1)
                ]
                for max_sp_i in spreads:
                    for timer in timers:
                        for tp in tps:
                            for sl in sls:
                                raw: list[dict[str, Any]] = []
                                for r in sub.itertuples(index=False):
                                    k = (str(r.date), str(r.symbol), str(r.dir), str(r.ts))
                                    f = resolved.get(k)
                                    if f is None or float(f["entry_spread_pct"]) > max_sp_i:
                                        continue
                                    sim = simulate_quote_scalp(
                                        f["path"],
                                        f["sig_ts"],
                                        tp=tp,
                                        sl=sl,
                                        timer_sec=int(timer),
                                        fill=fill,
                                        max_lag_sec=float(args.max_lag_sec),
                                        max_spread_pct=max_sp_i,
                                        min_mid=float(args.min_mid),
                                    )
                                    if sim is None or not np.isfinite(sim["ret"]):
                                        continue
                                    raw.append(
                                        {
                                            "date": f["date"],
                                            "symbol": f["symbol"],
                                            "dir": f["dir"],
                                            "entry_ts": str(sim["entry_ts"]),
                                            "exit_ts": str(sim["exit_ts"]),
                                            "ticker": f["ticker"],
                                            "ret": sim["ret"],
                                            "exit_reason": sim["reason"],
                                            "hold_sec": sim["hold_sec"],
                                        }
                                    )
                                st = _port(raw, max_concurrent=1 if uname == "qqq" else 2)
                                score_rows.append(
                                    {
                                        "universe": uname,
                                        "window": wname,
                                        "cell": cell,
                                        "max_spread_pct": max_sp_i,
                                        "timer_sec": timer,
                                        "tp": tp,
                                        "sl": sl,
                                        "n_signals": int(len(sub)),
                                        **st,
                                    }
                                )
                                if st.get("n", 0) >= 20:
                                    print(
                                        f"[{uname} {wname} {cell} t{timer} tp{tp}/sl{sl} sp≤{max_sp_i}] "
                                        f"n={st['n']} mean={st['mean']} add={st['add']:+.3f} "
                                        f"day_win={st['day_win']} timer%={st.get('frac_timer')}",
                                        flush=True,
                                    )

    score = pd.DataFrame(score_rows)
    score.to_csv(out / "scoreboard.csv", index=False)

    dual_ok: list[dict[str, Any]] = []
    if len(score):
        keys = ["universe", "cell", "max_spread_pct", "timer_sec", "tp", "sl"]
        cols = keys + ["n", "mean", "add", "day_win", "frac_timer", "hold_p50"]
        a = score[score["window"] == "may_jul"][cols].copy()
        b = score[score["window"] == "jan_mar"][cols].copy()
        m = a.merge(b, on=keys, suffixes=("_mj", "_jm"))
        for _, r in m.iterrows():
            # Scalp gate: hold should be short; timer not required majority.
            if (
                float(r["n_mj"]) >= 20
                and float(r["n_jm"]) >= 20
                and float(r["mean_mj"]) > 0
                and float(r["mean_jm"]) > 0
                and float(r["add_mj"]) > 0
                and float(r["add_jm"]) > 0
                and float(r["day_win_mj"]) >= 0.55
                and float(r["day_win_jm"]) >= 0.55
                and float(r.get("hold_p50_mj") or 999) <= 150
                and float(r.get("hold_p50_jm") or 999) <= 150
            ):
                dual_ok.append(
                    {
                        "universe": r["universe"],
                        "cell": r["cell"],
                        "max_spread_pct": float(r["max_spread_pct"]),
                        "timer_sec": int(r["timer_sec"]),
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
        "architecture": "stock_micro_state → ATM_quote → TP/SL/timer_scalp",
        "parked": ["scan_smooth_regress_quote_tpsl", "option premium curve fitting"],
        "session": f"{args.scan_start}-{args.scan_end}",
        "events": events_meta,
        "n_score_rows": int(len(score)),
        "n_dual_window_pass": int(len(dual_ok)),
        "dual_by_universe": {
            u: sum(1 for d in dual_ok if d["universe"] == u) for u, _ in universes
        },
        "dual_window_pass": dual_ok[:40],
        "top_by_add": (
            score.sort_values("add", ascending=False).head(25).to_dict(orient="records")
            if len(score)
            else []
        ),
        "verdict": "PASS" if dual_ok else "REJECT",
        "note": (
            "No option-curve fitting. Stock velocity/accel+SNR dual-window confluence; "
            "quote FillSpec; hard timer 90–120s as scalp primary exit alongside TP/SL."
        ),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    (out / "dual_pass.json").write_text(json.dumps(dual_ok[:40], indent=2, default=str), encoding="utf-8")
    print(f"\n=== dual PASS ({len(dual_ok)}) verdict={summary['verdict']} ===", flush=True)
    print(json.dumps(dual_ok[:12], indent=2, default=str), flush=True)
    if len(score):
        cols = [
            "universe",
            "window",
            "cell",
            "timer_sec",
            "tp",
            "sl",
            "max_spread_pct",
            "n",
            "mean",
            "add",
            "day_win",
            "frac_timer",
            "hold_p50",
        ]
        print("\n=== top by add ===", flush=True)
        print(score.sort_values("add", ascending=False)[cols].head(20).to_string(index=False), flush=True)
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Causal AM pulse search on 1s trailing VWAP (10/20/30s), not 1m close.

Feature: fav_from_open = vwap_w / RTH_open - 1, sampled every ``--sample-sec``.
Decision clock: delay=0 (prints ≤ feature_ts already known).

Stage 1 — trades last±slip dual (may_jul09 / jul10_23).
Stage 2 — quote FillSpec dual on trades dual_pass cells (sp/lag grid).

Example:
  PYTHONPATH=. python -m maga7.tools.scan_am_pulse_vwap_1s_dual \\
    --lane am_pulse --tag research_am_vwap1s_A_search
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
    am_pulse_decision_ts,
    load_am_pulse_lane_cfg,
    parse_am_pulse_scout,
    scan_day_1s_vwap,
)
from maga7.common.bar_agg import load_stock_1s_day
from maga7.common.config import load_profile
from maga7.common.fills import FillSpec
from maga7.common.open_lock import (
    load_multidte_lock_index,
    resolve_open_lock_contract,
    resolve_otm_rungs,
)
from maga7.common.option_quote_tpsl import entry_quote_row, simulate_quote_tpsl
from maga7.common.option_trade_tpsl import simulate_trade_tpsl
from maga7.common.option_trades import load_option_trades
from maga7.common.replay import load_quotes, path_for_ticker, to_ny
from maga7.common.session_1s_features import prepare_day_arrays
from maga7.common.stock_1s import session_dates
from maga7.tools.run_morning_sec_option_fill import _portfolio_day
from maga7.tools.scan_am_delayed_confirm_quote_dual import _ok, _prep_path, _stats
from maga7.tools.scan_session_horizon_foresight import _paths_by_ticker, _spot_at_arr, _stock_arrays

NY = "America/New_York"
PROFILE = (
    "maga7/CONFIG/strategy_profiles/"
    "single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1.json"
)
DEFAULT_TRADES = Path("/mnt/s990/new_option_data_s3_trades")
WINDOWS = (
    ("may_jul09", "2026-05-01", "2026-07-09"),
    ("jul10_23", "2026-07-10", "2026-07-23"),
)


def _parse_floats(spec: str) -> list[float]:
    return [float(x) for x in str(spec).split(",") if x.strip()]


def _parse_ints(spec: str) -> list[int]:
    return [int(x) for x in str(spec).split(",") if x.strip()]


def _window_of(date: str) -> str | None:
    for name, a, b in WINDOWS:
        if a <= date <= b:
            return name
    return None


def _cell_name(
    *,
    win: int,
    thr: float,
    max_fo: float,
    agree: tuple[int, ...],
    accel: bool,
    tp: float,
    sl: float,
) -> str:
    if agree:
        mode = "agree"
    elif accel:
        mode = "accel"
    else:
        mode = "solo"
    mf = f"_max{max_fo:g}" if max_fo > 0 else ""
    return f"vwap{win}_{mode}_t{thr:g}{mf}_tp{tp:g}_sl{sl:g}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", default=PROFILE)
    ap.add_argument("--lane", choices=("am_pulse", "am_pulse_extension"), default="am_pulse")
    ap.add_argument("--tag", default="research_am_vwap1s_search")
    ap.add_argument("--trades-root", default=str(DEFAULT_TRADES))
    ap.add_argument("--dirs", default="")
    ap.add_argument("--vwap-wins", default="10,20,30")
    ap.add_argument("--fo-thr", default="0.005,0.006,0.008,0.01,0.012")
    ap.add_argument("--max-fo", default="0,0.015,0.02")
    ap.add_argument("--sample-sec", type=int, default=10)
    ap.add_argument("--with-agree", action="store_true", help="Also test 10+20+30 agree cells")
    ap.add_argument("--with-accel", action="store_true", help="Also test vwap_fast>=vwap_primary accel")
    ap.add_argument("--tp", default="0.10,0.15,0.20")
    ap.add_argument("--sl", default="0.15,0.20,0.25")
    ap.add_argument("--max-hold-sec", type=int, default=900)
    ap.add_argument("--slip", type=float, default=0.01)
    ap.add_argument("--max-spreads", default="0.15")
    ap.add_argument("--max-lags", default="5")
    ap.add_argument("--position-frac", type=float, default=0.10)
    ap.add_argument("--max-concurrent", type=int, default=2)
    ap.add_argument("--cooldown-minutes", type=float, default=10.0)
    ap.add_argument("--min-n", type=int, default=8)
    ap.add_argument("--min-day-win", type=float, default=0.55)
    ap.add_argument("--skip-quote", action="store_true")
    ap.add_argument("--quote-top", type=int, default=40, help="Max trades dual_pass cells to quote")
    args = ap.parse_args(argv)

    prof = load_profile(args.profile)
    lane_cfg = load_am_pulse_lane_cfg(prof, args.lane)
    window_start = str(lane_cfg.get("window_start") or "09:30")
    window_end = str(lane_cfg.get("window_end") or "10:30")
    flatten_before = str(lane_cfg.get("flatten_before") or "").strip()
    dirs_spec = args.dirs or ",".join(lane_cfg.get("dirs") or ["DN", "UP"])
    dirs = {x.strip().upper() for x in dirs_spec.split(",") if x.strip()}
    prefer_dte = int(lane_cfg.get("prefer_dte", 0) or 0)
    allowed_raw = lane_cfg.get("allowed_dte") or (prof.get("lock") or {}).get(
        "allowed_dte"
    ) or [0, 1, 2]
    allowed_dte = [int(x) for x in allowed_raw]
    min_mid = float(lane_cfg.get("min_mid", 0.05) or 0.05)

    wins = _parse_ints(args.vwap_wins)
    thrs = _parse_floats(args.fo_thr)
    max_fos = _parse_floats(args.max_fo)
    tps = _parse_floats(args.tp)
    sls = _parse_floats(args.sl)
    spreads = _parse_floats(args.max_spreads)
    lags = _parse_floats(args.max_lags)
    sample_sec = max(1, int(args.sample_sec))

    # Probe keys: (win, thr, max_fo, agree_tuple, accel)
    probes: list[tuple[int, float, float, tuple[int, ...], bool]] = []
    for w in wins:
        for thr in thrs:
            for mf in max_fos:
                probes.append((w, thr, mf, (), False))
    if args.with_agree:
        agree = tuple(sorted(set(wins)))
        primary = max(agree) if agree else 30
        for thr in thrs:
            for mf in max_fos:
                probes.append((primary, thr, mf, agree, False))
    if args.with_accel:
        for w in wins:
            if w <= 10:
                continue
            for thr in thrs:
                for mf in max_fos:
                    probes.append((w, thr, mf, (), True))

    cells: list[dict[str, Any]] = []
    for w, thr, mf, agree, accel in probes:
        for tp in tps:
            for sl in sls:
                cells.append(
                    {
                        "name": _cell_name(
                            win=w,
                            thr=thr,
                            max_fo=mf,
                            agree=agree,
                            accel=accel,
                            tp=tp,
                            sl=sl,
                        ),
                        "vwap_win_sec": w,
                        "thr": thr,
                        "max_fo": mf,
                        "agree": list(agree),
                        "accel": bool(accel),
                        "tp": tp,
                        "sl": sl,
                    }
                )

    paths = prof["_paths"]
    symbols = list(prof.get("symbols") or [])
    stock_1s = Path(paths.get("stock_1s_root") or "/mnt/s990/data/raw_1s/stocks").expanduser()
    trades_root = Path(args.trades_root)
    quote_root = Path(paths["quote_1s_root"])
    lock = load_multidte_lock_index(Path(paths["open_locked_map"]).expanduser())
    otm = resolve_otm_rungs(prof, default=3)
    out = Path(paths["results_dir"]) / args.tag
    out.mkdir(parents=True, exist_ok=True)

    start_all = min(w[1] for w in WINDOWS)
    end_all = max(w[2] for w in WINDOWS)
    dates = [d for d in session_dates(start_all, end_all) if start_all <= d <= end_all]
    print(
        f"am_vwap1s {args.lane} {window_start}..{window_end} "
        f"{start_all}..{end_all} probes={len(probes)} cells={len(cells)} "
        f"dirs={sorted(dirs)} sample={sample_sec}s delay=0",
        flush=True,
    )

    # Cache alerts per probe key to avoid rescanning 1s for every tp/sl.
    arms_by_probe: dict[tuple, list[dict[str, Any]]] = {p: [] for p in probes}

    for di, date in enumerate(dates):
        if di % 10 == 0:
            n_arms = sum(len(v) for v in arms_by_probe.values())
            print(f"[day] {date} ({di+1}/{len(dates)}) arms={n_arms}", flush=True)
        for sym in symbols:
            by_dte = lock.get((sym, date))
            if not by_dte:
                continue
            day1s = load_stock_1s_day(stock_1s, sym, date)
            if day1s is None or day1s.empty:
                continue
            tday = load_option_trades(trades_root, sym, date)
            if tday is None or tday.empty:
                continue
            tpaths = _paths_by_ticker(tday)
            if not tpaths:
                continue
            ts_ns, px = _stock_arrays(day1s)
            # Official open: first RTH 1s open
            day_open = float(day1s.iloc[0]["open"]) if "open" in day1s.columns else float(
                day1s.iloc[0]["close"]
            )
            arr = prepare_day_arrays(day1s)
            vwap_cache: dict[int, Any] = {}

            seen_cfg: dict[tuple, list] = {}
            for probe in probes:
                w, thr, mf, agree, accel = probe
                cfg = parse_am_pulse_scout(
                    {
                        "enabled": True,
                        "feature_mode": "vwap_1s",
                        "window_start": window_start,
                        "window_end": window_end,
                        "min_fav_from_open": thr,
                        "max_fav_from_open": mf,
                        "vwap_win_sec": w,
                        "vwap_agree_wins": list(agree),
                        "vwap_accel": bool(accel),
                        "sample_every_sec": sample_sec,
                        "dirs": sorted(dirs),
                        "max_alerts_per_symbol": 1,
                        "min_lookback_ret": 0.99,
                    }
                )
                alerts = scan_day_1s_vwap(
                    day1s,
                    date=date,
                    symbol=sym,
                    cfg=cfg,
                    day_open=day_open,
                    arr=arr,
                    vwap_cache=vwap_cache,
                )
                seen_cfg[probe] = alerts

            for probe, alerts in seen_cfg.items():
                w, thr, mf, agree, accel = probe
                for a in alerts:
                    if a.dir not in dirs:
                        continue
                    feature_ts = to_ny(pd.Timestamp(a.ts))
                    decision_ts = am_pulse_decision_ts(feature_ts, delay_seconds=0)
                    spot = _spot_at_arr(ts_ns, px, decision_ts) if ts_ns is not None else None
                    if spot is None:
                        spot = float(a.px)
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
                    arms_by_probe[probe].append(
                        {
                            "date": date,
                            "symbol": sym,
                            "dir": a.dir,
                            "arm": "FO",
                            "thr": float(thr),
                            "max_fo": float(mf),
                            "vwap_win_sec": int(w),
                            "agree": list(agree),
                            "feature_ts": feature_ts,
                            "decision_ts": decision_ts,
                            "fav_from_open": float(a.fav_from_open),
                            "ticker": ticker,
                            "dte": dte,
                            "pts": arr[0],
                            "plast": arr[1],
                            "day1s": day1s,
                        }
                    )

    n_arms = sum(len(v) for v in arms_by_probe.values())
    print(f"arms_resolvable={n_arms}", flush=True)

    # --- Stage 1: trades dual ---
    score_rows: list[dict[str, Any]] = []
    dual_pass: list[dict[str, Any]] = []
    for cell in cells:
        probe = (
            int(cell["vwap_win_sec"]),
            float(cell["thr"]),
            float(cell["max_fo"]),
            tuple(int(x) for x in cell["agree"]),
            bool(cell.get("accel", False)),
        )
        arms = arms_by_probe.get(probe) or []
        win_raw: dict[str, list] = {w[0]: [] for w in WINDOWS}
        for arm in arms:
            wname = _window_of(str(arm["date"]))
            if wname is None:
                continue
            sim = simulate_trade_tpsl(
                arm["pts"],
                arm["plast"],
                arm["decision_ts"],
                tp=float(cell["tp"]),
                sl=float(cell["sl"]),
                max_hold_sec=int(args.max_hold_sec),
                slip=float(args.slip),
            )
            if sim is None or not np.isfinite(sim["ret"]):
                continue
            et = to_ny(arm["decision_ts"])
            xt = et + pd.Timedelta(seconds=float(sim["hold_sec"]))
            if flatten_before:
                # hard flatten clock HH:MM
                parts = flatten_before.split(":")
                flat = et.normalize() + pd.Timedelta(
                    hours=int(parts[0]), minutes=int(parts[1])
                )
                if xt > flat:
                    # trim hold — approximate by re-sim not available; clip ret via hold
                    pass
            win_raw[wname].append(
                {
                    "date": arm["date"],
                    "symbol": arm["symbol"],
                    "dir": arm["dir"],
                    "entry_ts": et,
                    "exit_ts": xt,
                    "ret": float(sim["ret"]),
                    "exit_reason": str(sim["reason"]),
                    "hold_sec": float(sim["hold_sec"]),
                    "size": float(args.position_frac),
                }
            )
        win_stats = {}
        both = True
        for wname, _, _ in WINDOWS:
            rows = win_raw[wname]
            if not rows:
                win_stats[wname] = _stats([])
                both = False
                continue
            sized = _portfolio_day(
                sorted(rows, key=lambda x: (x["entry_ts"], x["symbol"])),
                position_frac=float(args.position_frac),
                max_concurrent=int(args.max_concurrent),
                cooldown_minutes=float(args.cooldown_minutes),
            )
            st = _stats(sized)
            win_stats[wname] = st
            if not _ok(st, min_n=int(args.min_n), min_day_win=float(args.min_day_win)):
                both = False
        row = {
            "name": cell["name"],
            "vwap_win_sec": cell["vwap_win_sec"],
            "thr": cell["thr"],
            "max_fo": cell["max_fo"],
            "agree": cell["agree"],
            "accel": bool(cell.get("accel", False)),
            "tp": cell["tp"],
            "sl": cell["sl"],
            "dual_pass": both,
            "feature_mode": "vwap_1s",
            "bar_delay_sec": 0,
            "n_sig": len(arms),
        }
        for wname, _, _ in WINDOWS:
            for k, v in win_stats[wname].items():
                row[f"{wname}_{k}"] = v
        score_rows.append(row)
        if both:
            dual_pass.append(row)
            print(
                f"  *** TRADES DUAL PASS {cell['name']} "
                f"MJ n={row.get('may_jul09_n')} mean={row.get('may_jul09_mean'):.4f} "
                f"J10 n={row.get('jul10_23_n')} mean={row.get('jul10_23_mean'):.4f}",
                flush=True,
            )

    sb = pd.DataFrame(score_rows)
    sb.to_csv(out / "trades_scoreboard.csv", index=False)
    (out / "trades_dual_pass.json").write_text(
        json.dumps(dual_pass, indent=2, default=str), encoding="utf-8"
    )
    print(f"trades dual_pass_n={len(dual_pass)} / {len(cells)}", flush=True)

    quote_pass: list[dict[str, Any]] = []
    quote_rows: list[dict[str, Any]] = []
    if not args.skip_quote and dual_pass:
        # Prefer higher MJ mean among dual_pass
        ranked = sorted(
            dual_pass,
            key=lambda r: (float(r.get("may_jul09_mean") or -9), float(r.get("jul10_23_mean") or -9)),
            reverse=True,
        )[: max(1, int(args.quote_top))]
        fill = FillSpec(entry_frac=0.75, exit_frac=0.75)
        print(f"quote stage cells={len(ranked)} sp={spreads} lag={lags}", flush=True)

        # Rebuild arms with quote paths for needed probes only
        need_probes = {
            (
                int(c["vwap_win_sec"]),
                float(c["thr"]),
                float(c["max_fo"]),
                tuple(int(x) for x in c["agree"]),
                bool(c.get("accel", False)),
            )
            for c in ranked
        }
        q_arms: dict[tuple, list[dict[str, Any]]] = {p: [] for p in need_probes}
        for di, date in enumerate(dates):
            if di % 10 == 0:
                print(f"[quote-day] {date} ({di+1}/{len(dates)})", flush=True)
            for sym in symbols:
                by_dte = lock.get((sym, date))
                if not by_dte:
                    continue
                day1s = load_stock_1s_day(stock_1s, sym, date)
                if day1s is None or day1s.empty:
                    continue
                qday = _prep_path(load_quotes(quote_root, sym, date))
                if qday is None or qday.empty:
                    continue
                ts_ns, px = _stock_arrays(day1s)
                day_open = float(day1s.iloc[0]["open"]) if "open" in day1s.columns else float(
                    day1s.iloc[0]["close"]
                )
                arr = prepare_day_arrays(day1s)
                vwap_cache: dict[int, Any] = {}
                for probe in need_probes:
                    w, thr, mf, agree, accel = probe
                    cfg = parse_am_pulse_scout(
                        {
                            "enabled": True,
                            "feature_mode": "vwap_1s",
                            "window_start": window_start,
                            "window_end": window_end,
                            "min_fav_from_open": thr,
                            "max_fav_from_open": mf,
                            "vwap_win_sec": w,
                            "vwap_agree_wins": list(agree),
                            "vwap_accel": bool(accel),
                            "sample_every_sec": sample_sec,
                            "dirs": sorted(dirs),
                            "max_alerts_per_symbol": 1,
                            "min_lookback_ret": 0.99,
                        }
                    )
                    for a in scan_day_1s_vwap(
                        day1s,
                        date=date,
                        symbol=sym,
                        cfg=cfg,
                        day_open=day_open,
                        arr=arr,
                        vwap_cache=vwap_cache,
                    ):
                        if a.dir not in dirs:
                            continue
                        feature_ts = to_ny(pd.Timestamp(a.ts))
                        decision_ts = am_pulse_decision_ts(feature_ts, delay_seconds=0)
                        spot = (
                            _spot_at_arr(ts_ns, px, decision_ts)
                            if ts_ns is not None
                            else float(a.px)
                        )
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
                        path = _prep_path(path_for_ticker(qday, ticker))
                        if path is None or path.empty:
                            continue
                        probe_q = entry_quote_row(
                            path,
                            decision_ts,
                            max_lag_sec=max(lags),
                            max_spread_pct=max(spreads),
                            min_mid=min_mid,
                        )
                        if probe_q is None:
                            continue
                        q_arms[probe].append(
                            {
                                "date": date,
                                "symbol": sym,
                                "dir": a.dir,
                                "decision_ts": decision_ts,
                                "path": path,
                                "ticker": ticker,
                            }
                        )

        for cell in ranked:
            probe = (
                int(cell["vwap_win_sec"]),
                float(cell["thr"]),
                float(cell["max_fo"]),
                tuple(int(x) for x in cell["agree"]),
                bool(cell.get("accel", False)),
            )
            for max_sp in spreads:
                for max_lag in lags:
                    name = f"{cell['name']}_sp{max_sp}_lag{max_lag}"
                    win_raw: dict[str, list] = {w[0]: [] for w in WINDOWS}
                    n_fill = 0
                    for arm in q_arms.get(probe) or []:
                        wname = _window_of(str(arm["date"]))
                        if wname is None:
                            continue
                        eq = entry_quote_row(
                            arm["path"],
                            arm["decision_ts"],
                            max_lag_sec=float(max_lag),
                            max_spread_pct=float(max_sp),
                            min_mid=min_mid,
                        )
                        if eq is None:
                            continue
                        sim = simulate_quote_tpsl(
                            arm["path"],
                            arm["decision_ts"],
                            tp=float(cell["tp"]),
                            sl=float(cell["sl"]),
                            max_hold_sec=int(args.max_hold_sec),
                            fill=fill,
                            max_lag_sec=float(max_lag),
                            max_spread_pct=float(max_sp),
                            min_mid=min_mid,
                        )
                        if sim is None or not np.isfinite(sim.get("ret", np.nan)):
                            continue
                        n_fill += 1
                        et = to_ny(sim["entry_ts"])
                        xt = to_ny(sim["exit_ts"])
                        win_raw[wname].append(
                            {
                                "date": arm["date"],
                                "symbol": arm["symbol"],
                                "dir": arm["dir"],
                                "entry_ts": et,
                                "exit_ts": xt,
                                "ret": float(sim["ret"]),
                                "exit_reason": str(sim["reason"]),
                                "hold_sec": float(sim["hold_sec"]),
                                "size": float(args.position_frac),
                            }
                        )
                    win_stats = {}
                    both = True
                    for wname, _, _ in WINDOWS:
                        rows = win_raw[wname]
                        if not rows:
                            win_stats[wname] = _stats([])
                            both = False
                            continue
                        sized = _portfolio_day(
                            sorted(rows, key=lambda x: (x["entry_ts"], x["symbol"])),
                            position_frac=float(args.position_frac),
                            max_concurrent=int(args.max_concurrent),
                            cooldown_minutes=float(args.cooldown_minutes),
                        )
                        st = _stats(sized)
                        win_stats[wname] = st
                        if not _ok(
                            st, min_n=int(args.min_n), min_day_win=float(args.min_day_win)
                        ):
                            both = False
                    qrow = {
                        "name": name,
                        "base": cell["name"],
                        "vwap_win_sec": cell["vwap_win_sec"],
                        "thr": cell["thr"],
                        "max_fo": cell["max_fo"],
                        "agree": cell["agree"],
                        "accel": bool(cell.get("accel", False)),
                        "tp": cell["tp"],
                        "sl": cell["sl"],
                        "max_spread_pct": max_sp,
                        "max_lag_sec": max_lag,
                        "dual_pass": both,
                        "n_fill": n_fill,
                        "feature_mode": "vwap_1s",
                        "bar_delay_sec": 0,
                        "entry_anchor": "decision_ts=feature_ts",
                    }
                    for wname, _, _ in WINDOWS:
                        for k, v in win_stats[wname].items():
                            qrow[f"{wname}_{k}"] = v
                    quote_rows.append(qrow)
                    if both:
                        quote_pass.append(qrow)
                        print(
                            f"  *** QUOTE DUAL PASS {name} "
                            f"MJ n={qrow.get('may_jul09_n')} mean={qrow.get('may_jul09_mean'):.4f} "
                            f"J10 n={qrow.get('jul10_23_n')} mean={qrow.get('jul10_23_mean'):.4f}",
                            flush=True,
                        )

        pd.DataFrame(quote_rows).to_csv(out / "quote_scoreboard.csv", index=False)
        (out / "quote_dual_pass.json").write_text(
            json.dumps(quote_pass, indent=2, default=str), encoding="utf-8"
        )

    champ = quote_pass[0] if quote_pass else (dual_pass[0] if dual_pass else None)
    if quote_pass:
        champ = max(
            quote_pass,
            key=lambda r: (
                float(r.get("may_jul09_mean") or -9),
                float(r.get("jul10_23_mean") or -9),
            ),
        )
    elif dual_pass:
        champ = max(
            dual_pass,
            key=lambda r: (
                float(r.get("may_jul09_mean") or -9),
                float(r.get("jul10_23_mean") or -9),
            ),
        )

    verdict = {
        "lane": args.lane,
        "feature_mode": "vwap_1s",
        "entry_anchor": "decision_ts=feature_ts (delay=0)",
        "sample_every_sec": sample_sec,
        "window": [window_start, window_end],
        "dirs": sorted(dirs),
        "trades_dual_pass_n": len(dual_pass),
        "quote_dual_pass_n": len(quote_pass),
        "champion": champ,
        "verdict": (
            "QUOTE_PASS"
            if quote_pass
            else ("TRADES_PASS_QUOTE_PENDING" if dual_pass and args.skip_quote else "REJECT")
        ),
    }
    (out / "summary.json").write_text(json.dumps(verdict, indent=2, default=str), encoding="utf-8")
    print("\n=== VERDICT ===", flush=True)
    print(json.dumps(verdict, indent=2, default=str)[:3000], flush=True)
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

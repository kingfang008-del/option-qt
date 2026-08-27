#!/usr/bin/env python3
"""AM v2 Step2: ≤3 signal bakeoff — quote FillSpec is the only promotion mark.

Candidates (reuse parts, no legacy narrative):
  1) pulse_fo08_causal_full   FO≥0.8% cap1.5%, 09:30–11:30, decision=+60s
  2) pulse_fo08_causal_post10 FO same, window 10:00–11:30 (quote-healthy)
  3) launch_s3_r002_cd120     1s launch slope k=3 |ret|≥0.2%, cd=120s

Fixed exit for bakeoff: TP15/SL20/h900, FillSpec 0.75, lag≤5, spread≤15%.
trade-last reported as diagnostic only — cannot promote alone.

PASS: dual-window quote econ (disc&blind compound>0, min n).
FAIL: none → stay on Step2, try next signal family (do not loosen mark).

Example:
  PYTHONPATH=. python -m maga7.tools.run_am_v2_step2_signal_bakeoff \\
    --tag research_am_v2_step2_signal_bakeoff
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
    parse_am_pulse_scout,
    scan_day,
)
from maga7.common.bar_agg import load_stock_1s_day
from maga7.common.config import load_profile
from maga7.common.fills import FillSpec
from maga7.common.launch_slope import attach_launch_slope_features, launch_edges
from maga7.common.open_lock import (
    load_multidte_lock_index,
    resolve_open_lock_contract,
    resolve_otm_rungs,
)
from maga7.common.option_quote_tpsl import entry_quote_row, simulate_quote_tpsl
from maga7.common.option_trade_tpsl import simulate_trade_tpsl
from maga7.common.option_trades import load_option_trades
from maga7.common.replay import load_quotes, path_for_ticker, to_ny
from maga7.common.stock_1s import load_symbol_1s_bars, session_dates
from maga7.tools.run_morning_sec_option_fill import _portfolio_day
from maga7.tools.scan_am_pocket_regime_ladder_v2 import _window_of
from maga7.tools.scan_am_pocket_risk_optimize import _equity_stats
from maga7.tools.scan_session_horizon_foresight import _paths_by_ticker, _spot_at_arr

PROFILE = "maga7/CONFIG/strategy_profiles/am_v2_executable_path_v1.json"
SPINE = (
    "maga7/CONFIG/strategy_profiles/"
    "single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1.json"
)
NY = "America/New_York"
WINDOWS = (
    ("may_jul09", "2026-05-01", "2026-07-09"),
    ("jul10_23", "2026-07-10", "2026-07-23"),
)
BDAYS = 60


def _in_hhmm(ts: pd.Timestamp, start: str, end: str) -> bool:
    t = to_ny(ts)
    hm = t.hour * 60 + t.minute

    def _m(hhmm: str) -> int:
        a, b = hhmm.split(":")
        return int(a) * 60 + int(b)

    return _m(start) <= hm < _m(end)


def collect_pulse(
    *,
    name: str,
    dates: list[str],
    symbols: list[str],
    stock_by_sym: dict[str, pd.DataFrame],
    stock_1s: Path,
    window_start: str,
    window_end: str,
    delay_sec: int = 60,
    min_fav_from_open: float = 0.008,
    max_fav_from_open: float = 0.015,
) -> list[dict[str, Any]]:
    cfg = parse_am_pulse_scout(
        {
            "enabled": True,
            "window_start": window_start,
            "window_end": window_end,
            "min_fav_from_open": float(min_fav_from_open),
            "max_fav_from_open": float(max_fav_from_open),
            "lookback_bars": 2,
            "min_lookback_ret": 0.99,
            "dirs": ["DN", "UP"],
            "max_alerts_per_symbol": 1,
            "rth_open_only": True,
        }
    )
    out: list[dict[str, Any]] = []
    for date in dates:
        w = _window_of(date)
        if w is None:
            continue
        for sym in symbols:
            sdf = stock_by_sym.get(sym)
            if sdf is None:
                continue
            day1m = sdf[sdf["date"].astype(str) == date]
            if day1m.empty:
                continue
            day1s = load_stock_1s_day(stock_1s, sym, date)
            ts_ns = px = None
            if day1s is not None and not day1s.empty:
                arr = day1s.sort_values("timestamp")
                ts = pd.to_datetime(arr["timestamp"])
                if getattr(ts.dt, "tz", None) is None:
                    ts = ts.dt.tz_localize(NY)
                else:
                    ts = ts.dt.tz_convert(NY)
                ts_ns = ts.astype("int64").to_numpy()
                px = arr["close"].to_numpy(dtype=float)
            for a in scan_day(day1m, date=date, symbol=sym, cfg=cfg):
                feat_ts = to_ny(pd.Timestamp(a.ts))
                dec_ts = am_pulse_decision_ts(feat_ts, delay_seconds=delay_sec)
                spot = _spot_at_arr(ts_ns, px, feat_ts) if ts_ns is not None else None
                out.append(
                    {
                        "signal": name,
                        "date": date,
                        "symbol": sym,
                        "dir": str(a.dir),
                        "feature_ts": feat_ts,
                        "decision_ts": dec_ts,
                        "spot": float(spot) if spot is not None else None,
                        "window": w,
                    }
                )
    return out


def collect_launch(
    *,
    name: str,
    dates: list[str],
    symbols: list[str],
    stock_1s: Path,
    window_start: str,
    window_end: str,
    slope_sec: int = 3,
    abs_ret_min: float = 0.002,
    cooldown_sec: int = 120,
    dirs: tuple[str, ...] | list[str] | None = None,
) -> list[dict[str, Any]]:
    want = tuple(str(d).upper() for d in (dirs or ("UP", "DN")))
    out: list[dict[str, Any]] = []
    for date in dates:
        w = _window_of(date)
        if w is None:
            continue
        for sym in symbols:
            raw = load_stock_1s_day(stock_1s, sym, date)
            if raw is None or raw.empty:
                continue
            feat = attach_launch_slope_features(raw, slope_sec=slope_sec, peak_lookback_sec=60)
            if feat.empty:
                continue
            ts_col = feat["timestamp"]
            events: list[tuple[pd.Timestamp, str, int]] = []
            for d in want:
                for i in launch_edges(feat, direction=d, abs_ret_min=abs_ret_min):
                    t = to_ny(pd.Timestamp(ts_col.iloc[int(i)]))
                    if _in_hhmm(t, window_start, window_end):
                        events.append((t, d, int(i)))
            events.sort(key=lambda x: x[0])
            last_fire: pd.Timestamp | None = None
            for t, d, i in events:
                if last_fire is not None and (t - last_fire).total_seconds() < float(cooldown_sec):
                    continue
                last_fire = t
                out.append(
                    {
                        "signal": name,
                        "date": date,
                        "symbol": sym,
                        "dir": d,
                        "feature_ts": t,
                        "decision_ts": t,  # 1s causal — tradeable at print
                        "spot": float(feat["close"].iloc[i]),
                        "window": w,
                    }
                )
    return out


def score_signals(
    signals: list[dict[str, Any]],
    *,
    lock: dict,
    otm: list,
    quote_root: Path,
    trades_root: Path,
    fill: FillSpec,
    tp: float,
    sl: float,
    max_hold: int,
    max_lag: float,
    max_spread: float,
    min_mid: float,
    position_frac: float,
    max_concurrent: int,
    cooldown_minutes: float,
) -> dict[str, Any]:
    qcache: dict[tuple[str, str], pd.DataFrame | None] = {}
    tcache: dict[tuple[str, str], dict] = {}

    def qday(sym: str, date: str):
        k = (sym, date)
        if k not in qcache:
            qcache[k] = load_quotes(quote_root, sym, date)
        return qcache[k]

    def tpaths(sym: str, date: str):
        k = (sym, date)
        if k not in tcache:
            tday = load_option_trades(trades_root, sym, date)
            tcache[k] = _paths_by_ticker(tday) if tday is not None and not tday.empty else {}
        return tcache[k]

    quote_raw: list[dict] = []
    trade_raw: list[dict] = []
    n_sig = len(signals)
    n_no_contract = n_no_quote = n_gate_fail = 0

    for s in signals:
        date, sym, direction = s["date"], s["symbol"], s["dir"]
        et = to_ny(s["decision_ts"])
        by_dte = lock.get((sym, date))
        if not by_dte:
            n_no_contract += 1
            continue
        ticker, dte, _ = resolve_open_lock_contract(
            by_dte,
            direction=direction,
            moneyness="ATM",
            spot=s.get("spot"),
            prefer_dte=0,
            allowed_dte=(0, 1, 2),
            clear_otm_thresh=0.01,
            ladder=True,
            otm_rungs=otm,
        )
        if not ticker:
            n_no_contract += 1
            continue
        key = str(ticker).replace("O:", "")
        qd = qday(sym, date)
        path = path_for_ticker(qd, key) if qd is not None else None
        if path is None or path.empty:
            n_no_quote += 1
            continue
        sim_q = simulate_quote_tpsl(
            path,
            et,
            tp=tp,
            sl=sl,
            max_hold_sec=max_hold,
            fill=fill,
            max_lag_sec=max_lag,
            max_spread_pct=max_spread,
            min_mid=min_mid,
        )
        if sim_q is None or not np.isfinite(sim_q.get("ret", np.nan)):
            n_gate_fail += 1
            continue
        hold = float(sim_q["hold_sec"])
        entry_fill_ts = to_ny(sim_q.get("entry_ts", et))
        quote_raw.append(
            {
                "date": date,
                "symbol": sym,
                "dir": direction,
                "entry_ts": entry_fill_ts,
                "exit_ts": entry_fill_ts + pd.Timedelta(seconds=hold),
                "ticker": key,
                "ret": float(sim_q["ret"]),
                "exit_reason": str(sim_q["reason"]),
                "hold_sec": hold,
                "window": s["window"],
                "mark": "quote",
            }
        )
        # diagnostic trade-last from decision_ts
        arr = tpaths(sym, date).get(key)
        if arr is not None:
            sim_t = simulate_trade_tpsl(
                arr[0], arr[1], et, tp=tp, sl=sl, max_hold_sec=max_hold, slip=0.01
            )
            if sim_t is not None and np.isfinite(sim_t.get("ret", np.nan)):
                ht = float(sim_t["hold_sec"])
                trade_raw.append(
                    {
                        "date": date,
                        "symbol": sym,
                        "dir": direction,
                        "entry_ts": et,
                        "exit_ts": et + pd.Timedelta(seconds=ht),
                        "ticker": key,
                        "ret": float(sim_t["ret"]),
                        "exit_reason": str(sim_t["reason"]),
                        "hold_sec": ht,
                        "window": s["window"],
                        "mark": "trade",
                    }
                )

    def _book(raw: list[dict]) -> dict[str, Any]:
        win_stats = {}
        sized_all = []
        for wname, _, _ in WINDOWS:
            wr = [t for t in raw if t["window"] == wname]
            by_d: dict[str, list] = {}
            for t in wr:
                by_d.setdefault(str(t["date"]), []).append(t)
            sized = []
            for _, rs in sorted(by_d.items()):
                sized.extend(
                    _portfolio_day(
                        sorted(rs, key=lambda x: (x["entry_ts"], x["symbol"])),
                        position_frac=position_frac,
                        max_concurrent=max_concurrent,
                        cooldown_minutes=cooldown_minutes,
                    )
                )
            ste = _equity_stats(pd.DataFrame(sized)) if sized else {"n": 0, "compound": 0.0}
            win_stats[wname] = ste
            sized_all.extend(sized)
        if not sized_all:
            return {
                "n": 0,
                "tpd": 0.0,
                "trade_win": None,
                "mean_ret": None,
                "disc_compound": 0.0,
                "blind_compound": 0.0,
                "disc_n": 0,
                "blind_n": 0,
                "econ_dual": False,
            }
        rr = np.array([t["ret"] for t in sized_all], dtype=float)
        disc = float(win_stats["may_jul09"].get("compound") or 0)
        blind = float(win_stats["jul10_23"].get("compound") or 0)
        n_d = int(win_stats["may_jul09"].get("n") or 0)
        n_b = int(win_stats["jul10_23"].get("n") or 0)
        return {
            "n": len(sized_all),
            "tpd": len(sized_all) / float(BDAYS),
            "trade_win": float((rr > 0).mean()),
            "mean_ret": float(rr.mean()),
            "med_ret": float(np.median(rr)),
            "disc_compound": disc,
            "blind_compound": blind,
            "disc_n": n_d,
            "blind_n": n_b,
            "disc_maxdd": float(win_stats["may_jul09"].get("maxdd") or 0),
            "econ_dual": bool(n_d >= 8 and n_b >= 3 and disc > 0 and blind > 0),
            "frac_tp": float(np.mean([t["exit_reason"] == "tp" for t in sized_all])),
            "hold_p50": float(np.median([t["hold_sec"] for t in sized_all])),
        }

    qbook = _book(quote_raw)
    tbook = _book(trade_raw)
    return {
        "n_signals": n_sig,
        "n_quote_fills": qbook["n"],
        "fill_rate": (qbook["n"] / n_sig) if n_sig else 0.0,
        "n_no_contract": n_no_contract,
        "n_no_quote": n_no_quote,
        "n_gate_fail": n_gate_fail,
        "quote": qbook,
        "trade_diag": tbook,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", default=PROFILE)
    ap.add_argument("--spine", default=SPINE)
    ap.add_argument("--tag", default="research_am_v2_step2_signal_bakeoff")
    ap.add_argument("--trades-root", default="/mnt/s990/new_option_data_s3_trades")
    ap.add_argument("--tp", type=float, default=0.15)
    ap.add_argument("--sl", type=float, default=0.20)
    ap.add_argument("--max-hold-sec", type=int, default=900)
    ap.add_argument("--max-lag-sec", type=float, default=5.0)
    ap.add_argument("--max-spread-pct", type=float, default=0.15)
    ap.add_argument("--min-mid", type=float, default=0.05)
    ap.add_argument("--position-frac", type=float, default=0.10)
    ap.add_argument("--max-concurrent", type=int, default=5)
    ap.add_argument("--cooldown-minutes", type=float, default=1.0)
    ap.add_argument("--start-date", default="2026-05-01")
    ap.add_argument("--end-date", default="2026-07-23")
    ap.add_argument("--max-days", type=int, default=0)
    args = ap.parse_args(argv)

    v2 = load_profile(args.profile)
    spine = load_profile(args.spine)
    paths = spine["_paths"]
    out = Path(paths["results_dir"]) / args.tag
    out.mkdir(parents=True, exist_ok=True)

    stock_1s = Path(paths["stock_1s_root"])
    quote_root = Path(paths["quote_1s_root"])
    trades_root = Path(args.trades_root)
    lock = load_multidte_lock_index(Path(paths["open_locked_map"]).expanduser())
    otm = resolve_otm_rungs(spine, default=3)
    symbols = list(v2.get("symbols") or spine.get("symbols") or [])
    dates = [
        d
        for d in session_dates(args.start_date, args.end_date)
        if args.start_date <= d <= args.end_date and _window_of(d) is not None
    ]
    if int(args.max_days) > 0:
        dates = dates[: int(args.max_days)]

    print(f"am_v2 step2 bakeoff days={len(dates)} syms={len(symbols)}", flush=True)
    print("loading 1s→1m bars…", flush=True)
    stock_by_sym: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        sdf = load_symbol_1s_bars(stock_1s, sym, dates, bar_seconds=60)
        if sdf is not None and not sdf.empty:
            stock_by_sym[sym] = sdf
            print(f"  {sym}: bars={len(sdf)}", flush=True)

    fill = FillSpec(entry_frac=0.75, exit_frac=0.75)
    candidates: list[tuple[str, list[dict]]] = []

    print("collect pulse_fo08_causal_full…", flush=True)
    candidates.append(
        (
            "pulse_fo08_causal_full",
            collect_pulse(
                name="pulse_fo08_causal_full",
                dates=dates,
                symbols=symbols,
                stock_by_sym=stock_by_sym,
                stock_1s=stock_1s,
                window_start="09:30",
                window_end="11:30",
            ),
        )
    )
    print("collect pulse_fo08_causal_post10…", flush=True)
    candidates.append(
        (
            "pulse_fo08_causal_post10",
            collect_pulse(
                name="pulse_fo08_causal_post10",
                dates=dates,
                symbols=symbols,
                stock_by_sym=stock_by_sym,
                stock_1s=stock_1s,
                window_start="10:00",
                window_end="11:30",
            ),
        )
    )
    print("collect launch_s3_r002_cd120…", flush=True)
    candidates.append(
        (
            "launch_s3_r002_cd120",
            collect_launch(
                name="launch_s3_r002_cd120",
                dates=dates,
                symbols=symbols,
                stock_1s=stock_1s,
                window_start="09:30",
                window_end="11:30",
            ),
        )
    )

    rows = []
    for name, sigs in candidates:
        print(f"score {name}: signals={len(sigs)}", flush=True)
        st = score_signals(
            sigs,
            lock=lock,
            otm=otm,
            quote_root=quote_root,
            trades_root=trades_root,
            fill=fill,
            tp=float(args.tp),
            sl=float(args.sl),
            max_hold=int(args.max_hold_sec),
            max_lag=float(args.max_lag_sec),
            max_spread=float(args.max_spread_pct),
            min_mid=float(args.min_mid),
            position_frac=float(args.position_frac),
            max_concurrent=int(args.max_concurrent),
            cooldown_minutes=float(args.cooldown_minutes),
        )
        q = st["quote"]
        t = st["trade_diag"]
        row = {
            "signal": name,
            "n_signals": st["n_signals"],
            "fill_rate": st["fill_rate"],
            "n_gate_fail": st["n_gate_fail"],
            "n_no_quote": st["n_no_quote"],
            "quote_n": q["n"],
            "quote_tpd": q["tpd"],
            "quote_win": q["trade_win"],
            "quote_mean": q["mean_ret"],
            "quote_disc": q["disc_compound"],
            "quote_blind": q["blind_compound"],
            "quote_econ": q["econ_dual"],
            "trade_n": t["n"],
            "trade_mean": t["mean_ret"],
            "trade_disc": t["disc_compound"],
            "trade_blind": t["blind_compound"],
            "trade_econ": t["econ_dual"],
        }
        rows.append(row)
        print(
            f"  quote n={q['n']} tpd={q['tpd']:.2f} win={q['trade_win']} "
            f"mean={q['mean_ret']} disc={q['disc_compound']:+.3f} "
            f"blind={q['blind_compound']:+.3f} econ={q['econ_dual']}",
            flush=True,
        )

    sb = pd.DataFrame(rows)
    sb.to_csv(out / "scoreboard.csv", index=False)

    econ = sb[sb.quote_econ == True].copy()  # noqa: E712
    econ = econ.sort_values(["quote_mean", "quote_disc"], ascending=[False, False])
    promote = "NONE"
    best = None
    if len(econ):
        best = econ.iloc[0].to_dict()
        promote = f"STEP2_{best['signal']}"
    elif len(sb):
        # best by quote mean among any positive disc (soft)
        soft = sb[(sb.quote_disc > 0) | (sb.quote_blind > 0)].copy()
        if len(soft):
            soft = soft.sort_values("quote_mean", ascending=False)
            best = soft.iloc[0].to_dict()

    summary = {
        "protocol": "am_v2_step2_signal_bakeoff",
        "step": 2,
        "promotion_mark": "quote_FillSpec",
        "exit": f"tp{args.tp}_sl{args.sl}_h{args.max_hold_sec}",
        "gate": {
            "max_lag_sec": float(args.max_lag_sec),
            "max_spread_pct": float(args.max_spread_pct),
            "min_mid": float(args.min_mid),
        },
        "n_econ_quote": int(len(econ)),
        "promote": promote,
        "best_quote_econ": best if promote != "NONE" else None,
        "scoreboard": sb.to_dict(orient="records"),
        "pass": bool(promote != "NONE"),
        "next_step": 3 if promote != "NONE" else "2b_try_other_signals_or_windows",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    lines = [
        "# AM v2 Step2 — Signal bakeoff",
        "",
        f"- exit: TP{args.tp:g}/SL{args.sl:g}/h{args.max_hold_sec}",
        f"- mark: **quote FillSpec** (trade-last diagnostic only)",
        f"- promote: **{promote}**",
        f"- pass: **{summary['pass']}** → next `{summary['next_step']}`",
        "",
        "## Scoreboard",
        "",
    ]
    try:
        lines.append(sb.to_markdown(index=False))
    except Exception:
        lines.append(sb.to_string(index=False))
    (out / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")

    print("\n=== SCOREBOARD ===", flush=True)
    print(sb.to_string(index=False), flush=True)
    print(json.dumps({"promote": promote, "pass": summary["pass"], "next": summary["next_step"]}, indent=2))
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

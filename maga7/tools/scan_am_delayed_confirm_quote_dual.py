#!/usr/bin/env python3
"""Quote FillSpec dual-window accept for AM delayed-confirm champions.

Companion to ``scan_am_delayed_confirm_tpsl``. Same AM hard-cut, 1s DN signal,
confirm wait + causal gates; pricing is quote FillSpec (executable).

Default cells = trades dual-pass champions from
``research_am_delayed_confirm_tpsl_dual``.

Windows (structure split, match trades tool):
  may_jul09 = 2026-05-01..2026-07-09
  jul10_23  = 2026-07-10..2026-07-23

Dual PASS: mean>0, add>0, day_win≥0.55, n≥min_n (jul10 min_n capped at 8),
frac_max_hold≤0.50.

Example:
  PYTHONPATH=. python -m maga7.tools.scan_am_delayed_confirm_quote_dual \\
    --tag research_am_delayed_confirm_quote_dual
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
from maga7.common.option_quote_tpsl import entry_quote_row, simulate_quote_tpsl
from maga7.common.replay import load_quotes, path_for_ticker, to_ny
from maga7.common.stock_1s import session_dates
from maga7.tools.run_morning_sec_option_fill import _portfolio_day
from maga7.tools.scan_session_horizon_foresight import (
    _spot_at_arr,
    _stock_arrays,
    _stock_dir_arr,
)

NY = "America/New_York"
PROFILE = (
    "maga7/CONFIG/strategy_profiles/"
    "single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1.json"
)
SIGNAL_START = "09:30"
SIGNAL_END = "10:00"

WINDOWS = (
    ("may_jul09", "2026-05-01", "2026-07-09"),
    ("jul10_23", "2026-07-10", "2026-07-23"),
)

# Trades dual-pass champions (+ none control).
DEFAULT_CELLS = (
    {"name": "champ_no_adv_c15", "thr": 0.003, "confirm_sec": 15, "gate": "no_adv", "tp": 0.15, "sl": 0.12},
    {"name": "champ_stock_opt_c60", "thr": 0.003, "confirm_sec": 60, "gate": "stock_cont+opt_green", "tp": 0.20, "sl": 0.12},
    {"name": "champ_opt_green_c60", "thr": 0.003, "confirm_sec": 60, "gate": "opt_green", "tp": 0.20, "sl": 0.12},
    {"name": "ctrl_none_c15", "thr": 0.003, "confirm_sec": 15, "gate": "none", "tp": 0.15, "sl": 0.12},
    {"name": "ctrl_none_c60", "thr": 0.003, "confirm_sec": 60, "gate": "none", "tp": 0.20, "sl": 0.12},
)


def _prep_path(path: pd.DataFrame | None) -> pd.DataFrame | None:
    if path is None or path.empty:
        return None
    out = path.copy()
    ts = pd.to_datetime(out["timestamp"])
    if getattr(ts.dt, "tz", None) is None:
        ts = ts.dt.tz_localize(NY, ambiguous="infer")
    else:
        ts = ts.dt.tz_convert(NY)
    out["timestamp"] = ts
    return out.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)


def _stock_signed_ret(
    ts_ns: np.ndarray,
    px: np.ndarray,
    t0: pd.Timestamp,
    t1: pd.Timestamp,
    direction: str,
) -> float | None:
    a = _spot_at_arr(ts_ns, px, t0)
    b = _spot_at_arr(ts_ns, px, t1)
    if a is None or b is None or a <= 0 or b <= 0:
        return None
    raw = b / a - 1.0
    return float(raw if direction == "UP" else -raw)


def _quote_mark_ret(
    path: pd.DataFrame,
    sig_ts: pd.Timestamp,
    asof_ts: pd.Timestamp,
    *,
    fill: FillSpec,
    max_lag_sec: float,
    max_spread_pct: float,
    min_mid: float,
) -> float | None:
    """Synthetic: buy at sig quote, mark sell at last usable quote ≤ asof."""
    ent = entry_quote_row(
        path,
        sig_ts,
        max_lag_sec=max_lag_sec,
        max_spread_pct=max_spread_pct,
        min_mid=min_mid,
    )
    if ent is None:
        return None
    entry_px = fill.buy(ent["bid"], ent["ask"])
    if not np.isfinite(entry_px) or entry_px <= 0:
        return None
    asof = to_ny(asof_ts)
    upto = path[(path["timestamp"] >= ent["entry_ts"]) & (path["timestamp"] <= asof)]
    if upto.empty:
        return None
    r1 = upto.iloc[-1]
    bid, ask = float(r1["bid"]), float(r1["ask"])
    if not (np.isfinite(bid) and np.isfinite(ask) and ask > bid > 0):
        return None
    exit_px = fill.sell(bid, ask)
    return float(exit_px / entry_px - 1.0)


def _quote_no_adverse(
    path: pd.DataFrame,
    sig_ts: pd.Timestamp,
    asof_ts: pd.Timestamp,
    *,
    fill: FillSpec,
    max_lag_sec: float,
    max_spread_pct: float,
    min_mid: float,
    pos_thr: float,
    adv_thr: float,
) -> bool:
    ent = entry_quote_row(
        path,
        sig_ts,
        max_lag_sec=max_lag_sec,
        max_spread_pct=max_spread_pct,
        min_mid=min_mid,
    )
    if ent is None:
        return False
    entry_px = fill.buy(ent["bid"], ent["ask"])
    if not np.isfinite(entry_px) or entry_px <= 0:
        return False
    asof = to_ny(asof_ts)
    win = path[(path["timestamp"] > ent["entry_ts"]) & (path["timestamp"] <= asof)]
    for _, r in win.iterrows():
        bid, ask = float(r["bid"]), float(r["ask"])
        if not (np.isfinite(bid) and np.isfinite(ask) and ask > bid > 0):
            continue
        ret = fill.sell(bid, ask) / entry_px - 1.0
        if ret >= float(pos_thr):
            return True
        if ret <= -abs(float(adv_thr)):
            return False
    return True  # clean wait


def _stats(sized: list[dict]) -> dict[str, Any]:
    if not sized:
        return {
            "n": 0,
            "mean": None,
            "win": None,
            "add": 0.0,
            "day_win": None,
            "n_days": 0,
            "frac_tp": None,
            "frac_sl": None,
            "frac_max_hold": None,
            "hold_p50": None,
            "resolve_frac": None,
        }
    t = pd.DataFrame(sized)
    if "pnl_frac" not in t.columns:
        t["pnl_frac"] = t["ret"].astype(float) * t["size"].astype(float)
    day = t.groupby("date")["pnl_frac"].sum()
    reasons = t["exit_reason"].astype(str)
    return {
        "n": int(len(t)),
        "mean": float(t["ret"].mean()),
        "win": float((t["ret"] > 0).mean()),
        "add": float(t["pnl_frac"].sum()),
        "day_win": float((day > 0).mean()),
        "n_days": int(day.shape[0]),
        "red_days": int((day < 0).sum()),
        "worst_day": float(day.min()),
        "frac_tp": float((reasons == "tp").mean()),
        "frac_sl": float((reasons == "sl").mean()),
        "frac_max_hold": float((reasons == "max_hold").mean()),
        "hold_p50": float(t["hold_sec"].median()),
    }


def _ok(st: dict[str, Any], *, min_n: int, min_day_win: float) -> bool:
    mean, day_win, add, mh = (
        st.get("mean"),
        st.get("day_win"),
        st.get("add"),
        st.get("frac_max_hold"),
    )
    if mean is None or day_win is None or add is None or mh is None:
        return False
    return bool(
        int(st.get("n") or 0) >= min_n
        and float(mean) > 0
        and float(add) > 0
        and float(day_win) >= float(min_day_win)
        and float(mh) <= 0.50
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", default=PROFILE)
    ap.add_argument("--tag", default="research_am_delayed_confirm_quote_dual")
    ap.add_argument("--max-spreads", default="0.08,0.10,0.15")
    ap.add_argument("--max-lags", default="2,3")
    ap.add_argument("--min-mid", type=float, default=0.05)
    ap.add_argument("--adv-thr", type=float, default=0.05)
    ap.add_argument("--pos-thr", type=float, default=0.05)
    ap.add_argument("--stock-cont-min", type=float, default=0.0)
    ap.add_argument("--max-hold-sec", type=int, default=900)
    ap.add_argument("--entry-frac", type=float, default=0.75)
    ap.add_argument("--exit-frac", type=float, default=0.75)
    ap.add_argument("--position-frac", type=float, default=0.10)
    ap.add_argument("--max-concurrent", type=int, default=2)
    ap.add_argument("--cooldown-minutes", type=float, default=5.0)
    ap.add_argument("--min-n", type=int, default=10)
    ap.add_argument("--min-day-win", type=float, default=0.55)
    ap.add_argument("--lookback-sec", type=int, default=60)
    ap.add_argument("--stride-sec", type=int, default=60)
    args = ap.parse_args(argv)

    spreads = [float(x) for x in args.max_spreads.split(",") if x.strip()]
    lags = [float(x) for x in args.max_lags.split(",") if x.strip()]
    fill = FillSpec(entry_frac=float(args.entry_frac), exit_frac=float(args.exit_frac))

    prof = load_profile(args.profile)
    paths = prof["_paths"]
    symbols = list(prof.get("symbols") or [])
    stock_1s = Path(paths.get("stock_1s_root") or "/mnt/s990/data/raw_1s/stocks").expanduser()
    quote_root = Path(paths["quote_1s_root"])
    lock = load_multidte_lock_index(Path(paths["open_locked_map"]).expanduser())
    otm = resolve_otm_rungs(prof, default=3)
    out = Path(paths["results_dir"]) / args.tag
    out.mkdir(parents=True, exist_ok=True)

    start_all = min(w[1] for w in WINDOWS)
    end_all = max(w[2] for w in WINDOWS)
    dates = session_dates(start_all, end_all)
    print(
        f"AM delayed-confirm QUOTE dual {start_all}..{end_all} days={len(dates)} "
        f"cells={len(DEFAULT_CELLS)} sp={spreads} lag={lags}",
        flush=True,
    )

    # Build DN thr=0.003 first-fire signals once
    thr_need = sorted({float(c["thr"]) for c in DEFAULT_CELLS})
    signals: list[dict[str, Any]] = []
    stock_cache: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    quote_cache: dict[tuple[str, str], pd.DataFrame | None] = {}
    n_miss_q = n_miss_lock = 0

    for di, date in enumerate(dates):
        if di % 10 == 0:
            print(f"[day] {date} ({di+1}/{len(dates)}) sigs={len(signals)}", flush=True)
        for sym in symbols:
            day = load_stock_1s_day(stock_1s, sym, date)
            if day is None or day.empty:
                continue
            qday = load_quotes(quote_root, sym, date)
            qday = _prep_path(qday)
            quote_cache[(sym, date)] = qday
            if qday is None or qday.empty:
                n_miss_q += 1
                continue
            ts_ns, px = _stock_arrays(day)
            stock_cache[(date, sym)] = (ts_ns, px)
            by_dte = lock.get((sym, date))
            if not by_dte:
                n_miss_lock += 1
                continue
            t0 = pd.Timestamp(f"{date} {SIGNAL_START}:00", tz=NY) + pd.Timedelta(
                seconds=int(args.lookback_sec)
            )
            t1 = pd.Timestamp(f"{date} {SIGNAL_END}:00", tz=NY)
            fired: set[tuple[str, float]] = set()
            t = t0
            stride = pd.Timedelta(seconds=int(args.stride_sec))
            while t < t1:
                for thr in thr_need:
                    direction, sr = _stock_dir_arr(
                        ts_ns, px, t, int(args.lookback_sec), float(thr)
                    )
                    if direction != "DN":
                        continue
                    key = (direction, float(thr))
                    if key in fired:
                        continue
                    spot = _spot_at_arr(ts_ns, px, t)
                    ticker, dte, _ = resolve_open_lock_contract(
                        by_dte,
                        direction="DN",
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
                    path = path_for_ticker(qday, ticker)
                    path = _prep_path(path)
                    if path is None or path.empty:
                        continue
                    # wide probe at signal for resolvability
                    probe = entry_quote_row(
                        path,
                        t,
                        max_lag_sec=max(lags),
                        max_spread_pct=max(spreads),
                        min_mid=float(args.min_mid),
                    )
                    if probe is None:
                        continue
                    fired.add(key)
                    signals.append(
                        {
                            "date": date,
                            "symbol": sym,
                            "dir": "DN",
                            "thr": float(thr),
                            "sig_ts": to_ny(t),
                            "stock_ret_lb": float(sr),
                            "ticker": ticker,
                            "dte": dte,
                            "path": path,
                            "probe_spread": float(probe["spread_pct"]),
                            "probe_lag": float(probe["lag_sec"]),
                        }
                    )
                t += stride

    print(
        f"signals_resolvable={len(signals)} miss_quote_days≈{n_miss_q} "
        f"miss_lock≈{n_miss_lock}",
        flush=True,
    )

    def window_of(date: str) -> str | None:
        for wname, a, b in WINDOWS:
            if a <= date <= b:
                return wname
        return None

    score_rows: list[dict[str, Any]] = []
    dual_pass: list[dict[str, Any]] = []
    trade_dump: dict[str, pd.DataFrame] = {}

    for cell in DEFAULT_CELLS:
        thr = float(cell["thr"])
        conf = int(cell["confirm_sec"])
        gate = str(cell["gate"])
        tp = float(cell["tp"])
        sl = float(cell["sl"])
        need_stock = "stock_cont" in gate
        need_green = "opt_green" in gate or gate == "all"
        need_noadv = "no_adv" in gate or gate == "all"
        for max_sp in spreads:
            for max_lag in lags:
                win_raw: dict[str, list[dict]] = {w[0]: [] for w in WINDOWS}
                n_sig = n_block = n_fill = 0
                for s in signals:
                    if float(s["thr"]) != thr:
                        continue
                    wname = window_of(str(s["date"]))
                    if wname is None:
                        continue
                    n_sig += 1
                    if float(s["probe_spread"]) > max_sp or float(s["probe_lag"]) > max_lag:
                        n_block += 1
                        continue
                    sig_ts = to_ny(s["sig_ts"])
                    entry_ts = sig_ts + pd.Timedelta(seconds=conf)
                    sess_end = pd.Timestamp(f"{s['date']} {SIGNAL_END}:00", tz=NY)
                    if entry_ts > sess_end + pd.Timedelta(seconds=30):
                        n_block += 1
                        continue
                    path = s["path"]
                    st_arr = stock_cache.get((str(s["date"]), str(s["symbol"])))
                    if st_arr is None:
                        n_block += 1
                        continue
                    sts, spx = st_arr

                    if need_stock:
                        sret = _stock_signed_ret(sts, spx, sig_ts, entry_ts, "DN")
                        if sret is None or sret < float(args.stock_cont_min):
                            n_block += 1
                            continue
                    if need_green:
                        mret = _quote_mark_ret(
                            path,
                            sig_ts,
                            entry_ts,
                            fill=fill,
                            max_lag_sec=max_lag,
                            max_spread_pct=max_sp,
                            min_mid=float(args.min_mid),
                        )
                        if mret is None or mret <= 0:
                            n_block += 1
                            continue
                    if need_noadv:
                        if not _quote_no_adverse(
                            path,
                            sig_ts,
                            entry_ts,
                            fill=fill,
                            max_lag_sec=max_lag,
                            max_spread_pct=max_sp,
                            min_mid=float(args.min_mid),
                            pos_thr=float(args.pos_thr),
                            adv_thr=float(args.adv_thr),
                        ):
                            n_block += 1
                            continue

                    sim = simulate_quote_tpsl(
                        path,
                        entry_ts,
                        tp=tp,
                        sl=sl,
                        max_hold_sec=int(args.max_hold_sec),
                        fill=fill,
                        max_lag_sec=max_lag,
                        max_spread_pct=max_sp,
                        min_mid=float(args.min_mid),
                    )
                    if sim is None or not np.isfinite(sim["ret"]):
                        n_block += 1
                        continue
                    n_fill += 1
                    win_raw[wname].append(
                        {
                            "date": s["date"],
                            "symbol": s["symbol"],
                            "dir": "DN",
                            "sig_ts": str(sig_ts),
                            "entry_ts": str(sim["entry_ts"]),
                            "exit_ts": str(sim["exit_ts"]),
                            "ticker": s["ticker"],
                            "dte": s["dte"],
                            "ret": sim["ret"],
                            "exit_reason": sim["reason"],
                            "hold_sec": sim["hold_sec"],
                            "entry_spread_pct": sim.get("entry_spread_pct"),
                            "entry_lag_sec": sim.get("entry_lag_sec"),
                            "cell": cell["name"],
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
                    st["n_raw"] = int(len(raw))
                    st["resolve_frac"] = (
                        float(len(raw) / n_sig) if n_sig else None
                    )  # rough; overwritten below per window better as fill/sig
                    win_stats[wname] = st
                    sized_all.extend(sized)

                both = True
                for wname, _, _ in WINDOWS:
                    mn = int(args.min_n)
                    if wname == "jul10_23":
                        mn = min(mn, 8)
                    if not _ok(
                        win_stats[wname],
                        min_n=mn,
                        min_day_win=float(args.min_day_win),
                    ):
                        both = False
                        break

                row: dict[str, Any] = {
                    "cell": cell["name"],
                    "thr": thr,
                    "confirm_sec": conf,
                    "gate": gate,
                    "tp": tp,
                    "sl": sl,
                    "max_spread_pct": max_sp,
                    "max_lag_sec": max_lag,
                    "dual_pass": both,
                    "n_sig": n_sig,
                    "n_block": n_block,
                    "n_fill": n_fill,
                    "fill_frac": float(n_fill / n_sig) if n_sig else None,
                }
                for wname, _, _ in WINDOWS:
                    for k, v in win_stats[wname].items():
                        row[f"{wname}_{k}"] = v
                score_rows.append(row)
                if both:
                    key = f"{cell['name']}_sp{max_sp}_lag{max_lag}"
                    dual_pass.append(row)
                    trade_dump[key] = pd.DataFrame(sized_all)
                    print(
                        f"  *** QUOTE DUAL PASS {key} "
                        f"MJ09 add={row.get('may_jul09_add'):+} "
                        f"J10 add={row.get('jul10_23_add'):+}",
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
        key = f"{p['cell']}_sp{p['max_spread_pct']}_lag{p['max_lag_sec']}"
        if key in trade_dump and len(trade_dump[key]):
            trade_dump[key].to_csv(out / f"trades_dual{i:02d}_{key}.csv", index=False)

    # Compare vs trades champions summary if present
    trades_sum = Path(paths["results_dir"]) / "research_am_delayed_confirm_tpsl_dual" / "summary.json"
    trades_verdict = None
    if trades_sum.is_file():
        trades_verdict = json.loads(trades_sum.read_text()).get("verdict")

    summary = {
        "session": "AM_0930_1000",
        "book": "quote_fill_tpsl",
        "complements": "research_am_delayed_confirm_tpsl_dual",
        "trades_verdict": trades_verdict,
        "windows": [list(w) for w in WINDOWS],
        "cells": DEFAULT_CELLS,
        "n_signals_resolvable": int(len(signals)),
        "n_score_rows": int(len(score)),
        "dual_pass_n": int(len(dual_pass)),
        "verdict": "PASS" if dual_pass else "REJECT",
        "champion": dual_pass[0] if dual_pass else None,
        "decision": (
            "AM_DELAYED_CONFIRM_QUOTE_PASS"
            if dual_pass
            else "AM_DELAYED_CONFIRM_TRADES_PASS_QUOTE_REJECT"
        ),
        "note": (
            "Gates/confirm evaluated on quote FillSpec marks; entry at confirm_ts. "
            "Same dual windows as trades delayed-confirm tool."
        ),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    (out / "dual_pass.json").write_text(
        json.dumps(dual_pass[:50], indent=2, default=str), encoding="utf-8"
    )

    print("\n=== QUOTE verdict", summary["verdict"], summary["decision"], flush=True)
    if dual_pass:
        print(json.dumps(dual_pass[0], indent=2, default=str), flush=True)
    elif not score.empty:
        score["_sum"] = score["may_jul09_add"].fillna(0) + score["jul10_23_add"].fillna(0)
        near = score.sort_values("_sum", ascending=False).head(10)
        cols = [
            c
            for c in [
                "cell",
                "max_spread_pct",
                "max_lag_sec",
                "fill_frac",
                "may_jul09_n",
                "may_jul09_mean",
                "may_jul09_win",
                "may_jul09_day_win",
                "may_jul09_add",
                "jul10_23_n",
                "jul10_23_mean",
                "jul10_23_win",
                "jul10_23_day_win",
                "jul10_23_add",
            ]
            if c in near.columns
        ]
        print("top by sum add (no dual):", flush=True)
        print(near[cols].to_string(index=False), flush=True)
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""AM 09:30–10:00 delayed-confirm gate + trade-last TP/SL (causal).

Signal: causal 1s stock lookback direction with ``|ret| >= thr`` in
``[09:30, 10:00)``. Do **not** fill at signal time. Wait ``confirm_sec``, then
require causal confirmations before entry:

  - ``stock_cont``: spot still moves in signal direction over the wait
  - ``opt_green``: option trade-last mark > 0 at confirm clock (after slip)
  - ``no_adv``: within wait, option does not touch ``-adv_thr`` before
    ``+pos_thr`` (causal walk; same spirit as path autopsy adverse-first)

Exit: first-passage TP/SL on option trades (``max_hold_sec`` safety only).

Dual windows (structure split):
  may_jul09 = 2026-05-01..2026-07-09
  jul10_23  = 2026-07-10..2026-07-23

Example:
  PYTHONPATH=. python -m maga7.tools.scan_am_delayed_confirm_tpsl \\
    --tag research_am_delayed_confirm_tpsl_dual
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
from maga7.common.open_lock import (
    load_multidte_lock_index,
    resolve_open_lock_contract,
    resolve_otm_rungs,
)
from maga7.common.option_trade_tpsl import simulate_trade_tpsl
from maga7.common.option_trades import load_option_trades
from maga7.common.replay import to_ny
from maga7.common.stock_1s import session_dates
from maga7.tools.run_morning_sec_option_fill import _portfolio_day
from maga7.tools.scan_session_horizon_foresight import (
    _paths_by_ticker,
    _spot_at_arr,
    _stock_arrays,
    _stock_dir_arr,
)

NY = "America/New_York"
DEFAULT_TRADES = Path("/mnt/s990/new_option_data_s3_trades")
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
        }
    by: dict[str, list] = {}
    for r in rows:
        by.setdefault(str(r["date"]), []).append(r)
    sized: list[dict] = []
    for d in sorted(by):
        sized.extend(
            _portfolio_day(
                by[d],
                position_frac=float(position_frac),
                max_concurrent=int(max_concurrent),
                cooldown_minutes=float(cooldown_minutes),
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
        }
    t = pd.DataFrame(sized)
    if "pnl_frac" not in t.columns:
        t["pnl_frac"] = t["ret"].astype(float) * t["size"].astype(float)
    day = t.groupby("date")["pnl_frac"].sum()
    reasons = t["exit_reason"].astype(str) if "exit_reason" in t.columns else pd.Series(dtype=str)
    return {
        "n": int(len(t)),
        "mean": float(t["ret"].mean()),
        "win": float((t["ret"] > 0).mean()),
        "add": float(t["pnl_frac"].sum()),
        "day_win": float((day > 0).mean()),
        "n_days": int(day.shape[0]),
        "red_days": int((day < 0).sum()),
        "worst_day": float(day.min()),
        "frac_tp": float((reasons == "tp").mean()) if len(reasons) else None,
        "frac_sl": float((reasons == "sl").mean()) if len(reasons) else None,
        "frac_max_hold": float((reasons == "max_hold").mean()) if len(reasons) else None,
        "hold_p50": float(t["hold_sec"].median()) if "hold_sec" in t.columns else None,
    }


def _ok(st: dict[str, Any], *, min_n: int, min_day_win: float) -> bool:
    mean, day_win, add = st.get("mean"), st.get("day_win"), st.get("add")
    if mean is None or day_win is None or add is None:
        return False
    return bool(
        int(st.get("n") or 0) >= min_n
        and float(mean) > 0
        and float(add) > 0
        and float(day_win) >= float(min_day_win)
    )


def _opt_mark_ret(
    ts_ns: np.ndarray,
    last: np.ndarray,
    entry_ts: pd.Timestamp,
    asof_ts: pd.Timestamp,
    *,
    slip: float,
) -> float | None:
    """Causal option mark ret from entry print → last print ≤ asof (sell slip)."""
    t0 = int(to_ny(entry_ts).value)
    i0 = int(np.searchsorted(ts_ns, t0, side="left"))
    if i0 >= len(ts_ns):
        return None
    if (int(ts_ns[i0]) - t0) / 1e9 > 5:
        return None
    entry = float(last[i0]) * (1.0 + float(slip))
    if not np.isfinite(entry) or entry <= 0:
        return None
    t1 = int(to_ny(asof_ts).value)
    i1 = int(np.searchsorted(ts_ns, t1, side="right") - 1)
    if i1 < i0:
        return None
    return float(last[i1] * (1.0 - float(slip)) / entry - 1.0)


def _opt_no_adverse(
    ts_ns: np.ndarray,
    last: np.ndarray,
    entry_ts: pd.Timestamp,
    asof_ts: pd.Timestamp,
    *,
    slip: float,
    pos_thr: float,
    adv_thr: float,
) -> tuple[bool, str]:
    """Walk prints in (entry, asof]; fail if -adv before +pos."""
    t0 = int(to_ny(entry_ts).value)
    i0 = int(np.searchsorted(ts_ns, t0, side="left"))
    if i0 >= len(ts_ns):
        return False, "no_entry"
    if (int(ts_ns[i0]) - t0) / 1e9 > 5:
        return False, "entry_lag"
    entry = float(last[i0]) * (1.0 + float(slip))
    if not np.isfinite(entry) or entry <= 0:
        return False, "bad_entry"
    t1 = int(to_ny(asof_ts).value)
    i1 = int(np.searchsorted(ts_ns, t1, side="right") - 1)
    sell_m = 1.0 - float(slip)
    saw_pos = False
    for k in range(i0 + 1, i1 + 1):
        ret = float(last[k]) * sell_m / entry - 1.0
        if not np.isfinite(ret):
            continue
        if ret >= float(pos_thr):
            saw_pos = True
            return True, "pos_first"
        if ret <= -abs(float(adv_thr)):
            return False, "adv_first"
    if saw_pos:
        return True, "pos_first"
    # no adverse touch within wait → pass (soft)
    return True, "clean_wait"


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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", default=PROFILE)
    ap.add_argument("--tag", default="research_am_delayed_confirm_tpsl_dual")
    ap.add_argument("--trades-root", default=str(DEFAULT_TRADES))
    ap.add_argument("--dirs", default="DN", help="Comma list UP,DN")
    ap.add_argument("--thr", default="0.003,0.005")
    ap.add_argument("--confirm-sec", default="15,30,60")
    ap.add_argument(
        "--gates",
        default="none,stock_cont,opt_green,no_adv,stock_cont+opt_green,stock_cont+no_adv,all",
        help="Confirm gate combos (none = fill at signal+confirm_sec with no filter)",
    )
    ap.add_argument("--stock-cont-min", type=float, default=0.0,
                    help="Min signed stock ret over confirm wait (DN favor >0)")
    ap.add_argument("--adv-thr", type=float, default=0.05)
    ap.add_argument("--pos-thr", type=float, default=0.05)
    ap.add_argument("--tp", default="0.15,0.20,0.25")
    ap.add_argument("--sl", default="0.10,0.12,0.15")
    ap.add_argument("--max-hold-sec", type=int, default=900)
    ap.add_argument("--slip", type=float, default=0.01)
    ap.add_argument("--stride-sec", type=int, default=60)
    ap.add_argument("--lookback-sec", type=int, default=60)
    ap.add_argument("--position-frac", type=float, default=0.10)
    ap.add_argument("--max-concurrent", type=int, default=2)
    ap.add_argument("--cooldown-minutes", type=float, default=5.0)
    ap.add_argument("--max-per-symbol-day", type=int, default=1)
    ap.add_argument("--min-n", type=int, default=10)
    ap.add_argument("--min-day-win", type=float, default=0.55)
    args = ap.parse_args(argv)

    dirs = {x.strip().upper() for x in args.dirs.split(",") if x.strip()}
    thrs = [float(x) for x in args.thr.split(",") if x.strip()]
    confirms = [int(x) for x in args.confirm_sec.split(",") if x.strip()]
    gates = [x.strip() for x in args.gates.split(",") if x.strip()]
    tps = [float(x) for x in args.tp.split(",") if x.strip()]
    sls = [float(x) for x in args.sl.split(",") if x.strip()]

    prof = load_profile(args.profile)
    paths = prof["_paths"]
    symbols = list(prof.get("symbols") or [])
    stock_1s = Path(paths.get("stock_1s_root") or "/mnt/s990/data/raw_1s/stocks").expanduser()
    trades_root = Path(args.trades_root)
    lock = load_multidte_lock_index(Path(paths["open_locked_map"]).expanduser())
    otm = resolve_otm_rungs(prof, default=3)
    out = Path(paths["results_dir"]) / args.tag
    out.mkdir(parents=True, exist_ok=True)

    start_all = min(w[1] for w in WINDOWS)
    end_all = max(w[2] for w in WINDOWS)
    dates = session_dates(start_all, end_all)
    print(
        f"AM delayed-confirm scan {start_all}..{end_all} days={len(dates)} "
        f"dirs={sorted(dirs)} thr={thrs} confirm={confirms} gates={gates}",
        flush=True,
    )

    # Collect raw signals once: (date,sym,dir,sig_ts,thr, ticker,dte, pts,plast, stock_ts,stock_px)
    # Store stock arrays per (date,sym) and trade path per signal.
    signals: list[dict[str, Any]] = []
    stock_cache: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    n_miss_lock = n_miss_trades = 0

    for di, date in enumerate(dates):
        if di % 10 == 0:
            print(f"[day] {date} ({di+1}/{len(dates)}) sigs={len(signals)}", flush=True)
        for sym in symbols:
            day = load_stock_1s_day(stock_1s, sym, date)
            if day is None or day.empty:
                continue
            tday = load_option_trades(trades_root, sym, date)
            if tday is None or tday.empty:
                n_miss_trades += 1
                continue
            tpaths = _paths_by_ticker(tday)
            if not tpaths:
                n_miss_trades += 1
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
            # first fire per (sym, dir, thr) per day
            fired: set[tuple[str, float]] = set()
            t = t0
            stride = pd.Timedelta(seconds=int(args.stride_sec))
            while t < t1:
                for thr in thrs:
                    direction, sr = _stock_dir_arr(
                        ts_ns, px, t, int(args.lookback_sec), float(thr)
                    )
                    if direction is None or direction not in dirs:
                        continue
                    key = (direction, float(thr))
                    if key in fired:
                        continue
                    spot = _spot_at_arr(ts_ns, px, t)
                    ticker, dte, _src = resolve_open_lock_contract(
                        by_dte,
                        direction=direction,
                        moneyness="ATM",
                        spot=spot,
                        prefer_dte=0,
                        allowed_dte=[0, 1, 2],
                        clear_otm_thresh=0.01,
                        ladder=True,
                        otm_rungs=otm,
                    )
                    if not ticker:
                        n_miss_lock += 1
                        continue
                    arr = tpaths.get(str(ticker).replace("O:", ""))
                    if arr is None:
                        n_miss_trades += 1
                        continue
                    fired.add(key)
                    signals.append(
                        {
                            "date": date,
                            "symbol": sym,
                            "dir": direction,
                            "thr": float(thr),
                            "sig_ts": t,
                            "stock_ret_lb": float(sr),
                            "ticker": ticker,
                            "dte": dte,
                            "pts": arr[0],
                            "plast": arr[1],
                        }
                    )
                t += stride

    print(
        f"signals={len(signals)} miss_lock≈{n_miss_lock} miss_trades≈{n_miss_trades}",
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

    for thr in thrs:
        for conf in confirms:
            for gate in gates:
                need_stock = "stock_cont" in gate or gate == "all"
                need_green = "opt_green" in gate or gate == "all"
                need_noadv = "no_adv" in gate or gate == "all"
                for tp in tps:
                    for sl in sls:
                        # Only score meaningful TP/SL pairs (tp >= sl loosely ok; skip weird)
                        win_raw: dict[str, list[dict[str, Any]]] = {
                            w[0]: [] for w in WINDOWS
                        }
                        n_sig = n_block = n_fill = 0
                        for s in signals:
                            if float(s["thr"]) != float(thr):
                                continue
                            wname = window_of(str(s["date"]))
                            if wname is None:
                                continue
                            n_sig += 1
                            sig_ts = to_ny(s["sig_ts"])
                            entry_ts = sig_ts + pd.Timedelta(seconds=int(conf))
                            # confirm must still be before session end + small buffer
                            sess_end = pd.Timestamp(
                                f"{s['date']} {SIGNAL_END}:00", tz=NY
                            )
                            if entry_ts > sess_end + pd.Timedelta(seconds=30):
                                n_block += 1
                                continue
                            st_arr = stock_cache.get((str(s["date"]), str(s["symbol"])))
                            if st_arr is None:
                                n_block += 1
                                continue
                            sts, spx = st_arr
                            pts, plast = s["pts"], s["plast"]

                            if need_stock:
                                sret = _stock_signed_ret(
                                    sts, spx, sig_ts, entry_ts, str(s["dir"])
                                )
                                if sret is None or sret < float(args.stock_cont_min):
                                    n_block += 1
                                    continue
                            if need_green:
                                mret = _opt_mark_ret(
                                    pts, plast, sig_ts, entry_ts, slip=float(args.slip)
                                )
                                # mark from signal→confirm using signal as synthetic entry
                                # For entry-at-confirm we need green vs signal fill level:
                                if mret is None or mret <= 0:
                                    n_block += 1
                                    continue
                            if need_noadv:
                                ok_na, _why = _opt_no_adverse(
                                    pts,
                                    plast,
                                    sig_ts,
                                    entry_ts,
                                    slip=float(args.slip),
                                    pos_thr=float(args.pos_thr),
                                    adv_thr=float(args.adv_thr),
                                )
                                if not ok_na:
                                    n_block += 1
                                    continue

                            sim = simulate_trade_tpsl(
                                pts,
                                plast,
                                entry_ts,
                                tp=float(tp),
                                sl=float(sl),
                                max_hold_sec=int(args.max_hold_sec),
                                slip=float(args.slip),
                            )
                            if sim is None or not np.isfinite(sim["ret"]):
                                n_block += 1
                                continue
                            n_fill += 1
                            et = to_ny(entry_ts)
                            win_raw[wname].append(
                                {
                                    "date": s["date"],
                                    "symbol": s["symbol"],
                                    "dir": s["dir"],
                                    "sig_ts": str(sig_ts),
                                    "entry_ts": str(et),
                                    "exit_ts": str(
                                        et + pd.Timedelta(seconds=sim["hold_sec"])
                                    ),
                                    "ticker": s["ticker"],
                                    "dte": s["dte"],
                                    "ret": sim["ret"],
                                    "exit_reason": sim["reason"],
                                    "hold_sec": sim["hold_sec"],
                                    "stock_ret_lb": s["stock_ret_lb"],
                                    "confirm_sec": conf,
                                    "gate": gate,
                                    "window": wname,
                                }
                            )

                        # per-symbol-day cap then portfolio
                        win_stats: dict[str, dict[str, Any]] = {}
                        sized_all: list[dict] = []
                        for wname, _, _ in WINDOWS:
                            raw = win_raw[wname]
                            # first chronologically already; cap per symbol day
                            picked: list[dict] = []
                            by_d: dict[str, list] = {}
                            for r in raw:
                                by_d.setdefault(str(r["date"]), []).append(r)
                            max_sym = int(args.max_per_symbol_day)
                            for d in sorted(by_d):
                                rows = sorted(
                                    by_d[d],
                                    key=lambda r: (str(r["entry_ts"]), str(r["symbol"])),
                                )
                                for r in rows:
                                    if max_sym > 0:
                                        nsym = sum(
                                            1
                                            for x in picked
                                            if x["date"] == d and x["symbol"] == r["symbol"]
                                        )
                                        if nsym >= max_sym:
                                            continue
                                    picked.append(r)
                            sized: list[dict] = []
                            by2: dict[str, list] = {}
                            for r in picked:
                                by2.setdefault(str(r["date"]), []).append(r)
                            for _, rs in sorted(by2.items()):
                                sized.extend(
                                    _portfolio_day(
                                        rs,
                                        position_frac=float(args.position_frac),
                                        max_concurrent=int(args.max_concurrent),
                                        cooldown_minutes=float(args.cooldown_minutes),
                                    )
                                )
                            if sized:
                                tdf = pd.DataFrame(sized)
                                if "pnl_frac" not in tdf.columns:
                                    tdf["pnl_frac"] = (
                                        tdf["ret"].astype(float) * tdf["size"].astype(float)
                                    )
                                day = tdf.groupby("date")["pnl_frac"].sum()
                                reasons = tdf["exit_reason"].astype(str)
                                st = {
                                    "n": int(len(tdf)),
                                    "mean": float(tdf["ret"].mean()),
                                    "win": float((tdf["ret"] > 0).mean()),
                                    "add": float(tdf["pnl_frac"].sum()),
                                    "day_win": float((day > 0).mean()),
                                    "n_days": int(day.shape[0]),
                                    "red_days": int((day < 0).sum()),
                                    "worst_day": float(day.min()),
                                    "frac_tp": float((reasons == "tp").mean()),
                                    "frac_sl": float((reasons == "sl").mean()),
                                    "frac_max_hold": float((reasons == "max_hold").mean()),
                                    "hold_p50": float(tdf["hold_sec"].median()),
                                }
                                sized_all.extend(sized)
                            win_stats[wname] = st

                        # jul10 has few days — allow lower min_n there via per-window check
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
                            "thr": thr,
                            "confirm_sec": conf,
                            "gate": gate,
                            "tp": tp,
                            "sl": sl,
                            "dirs": ",".join(sorted(dirs)),
                            "dual_pass": both,
                            "n_sig_eval": n_sig,
                            "n_block": n_block,
                            "n_fill": n_fill,
                        }
                        for wname, _, _ in WINDOWS:
                            for k, v in win_stats[wname].items():
                                row[f"{wname}_{k}"] = v
                        score_rows.append(row)
                        if both:
                            key = f"thr{thr}_c{conf}_{gate}_tp{tp}_sl{sl}"
                            dual_pass.append(row)
                            trade_dump[key] = pd.DataFrame(sized_all)
                            print(
                                f"  *** DUAL PASS {key} "
                                f"MJ09 add={row.get('may_jul09_add'):+} "
                                f"win={row.get('may_jul09_win')} "
                                f"J10 add={row.get('jul10_23_add'):+} "
                                f"win={row.get('jul10_23_win')}",
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
    for i, p in enumerate(dual_pass[:20]):
        key = f"thr{p['thr']}_c{p['confirm_sec']}_{p['gate']}_tp{p['tp']}_sl{p['sl']}"
        if key in trade_dump and len(trade_dump[key]):
            trade_dump[key].to_csv(out / f"trades_dual{i:02d}_{key}.csv", index=False)

    # Top by each window even if not dual
    def top_for(w: str, k: int = 10) -> list[dict]:
        col = f"{w}_add"
        if score.empty or col not in score.columns:
            return []
        sub = score[score[f"{w}_mean"].fillna(-1) > 0].copy()
        sub = sub.sort_values(col, ascending=False).head(k)
        return sub.to_dict(orient="records")

    summary = {
        "session": "AM_0930_1000",
        "pricing": str(trades_root),
        "slip": float(args.slip),
        "exit": "tp_sl_first_passage_after_delayed_entry",
        "windows": [list(w) for w in WINDOWS],
        "dirs": sorted(dirs),
        "thrs": thrs,
        "confirm_secs": confirms,
        "gates": gates,
        "n_signals": int(len(signals)),
        "n_score_rows": int(len(score)),
        "dual_pass_n": int(len(dual_pass)),
        "verdict": "PASS" if dual_pass else "REJECT",
        "champion": dual_pass[0] if dual_pass else None,
        "top_may_jul09": top_for("may_jul09"),
        "top_jul10_23": top_for("jul10_23"),
        "note": (
            "Entry delayed by confirm_sec; gates are causal on stock 1s / option "
            "trade-last path between sig and entry. Dual = both windows mean>0, "
            "add>0, day_win>=min_day_win, n>=min_n (jul10 min_n capped at 8)."
        ),
    }
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    (out / "dual_pass.json").write_text(
        json.dumps(dual_pass[:50], indent=2, default=str), encoding="utf-8"
    )

    print("\n=== verdict", summary["verdict"], "dual_pass_n=", len(dual_pass), flush=True)
    if dual_pass:
        print("champion:", json.dumps(dual_pass[0], indent=2, default=str), flush=True)
    else:
        # show best near-misses
        if not score.empty:
            score["_sum_add"] = score["may_jul09_add"].fillna(0) + score[
                "jul10_23_add"
            ].fillna(0)
            near = score.sort_values("_sum_add", ascending=False).head(8)
            cols = [
                c
                for c in [
                    "thr",
                    "confirm_sec",
                    "gate",
                    "tp",
                    "sl",
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

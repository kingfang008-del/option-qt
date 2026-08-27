#!/usr/bin/env python3
"""Certainty morphologies: chase option pump / sync stock+option (causal).

Replaces early AM sniping. Two families (trade-last ± slip, TP/SL exit):

  A) ``opt_chase``
     Arm on causal 1s stock ``|ret_lb|>=thr`` in session.
     Do not fill yet. From arm mark, wait until option trade-last reaches
     ``+chase`` **before** ``-abort`` (within ``wait_sec``). Enter then.

  B) ``sync``
     At clock t: stock dir ``|ret_lb|>=thr`` AND stock still favorable over
     last ``sync_stock_sec`` AND option mark over last ``sync_opt_sec`` > 0.
     Enter immediately (certainty already in the tape).

Sessions: AM 09:30–10:00 and CORE 10:30–11:30.
Dual windows: may_jul09 / jul10_23.

Example:
  PYTHONPATH=. python -m maga7.tools.scan_am_certainty_morph_tpsl \\
    --tag research_certainty_morph_tpsl_dual
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

SESSIONS = (
    ("AM_0930_1000", "09:30", "10:00"),
    ("CORE_1030_1130", "10:30", "11:30"),
)
WINDOWS = (
    ("may_jul09", "2026-05-01", "2026-07-09"),
    ("jul10_23", "2026-07-10", "2026-07-23"),
)


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
    mean, day_win, add = st.get("mean"), st.get("day_win"), st.get("add")
    if mean is None or day_win is None or add is None:
        return False
    return bool(
        int(st.get("n") or 0) >= min_n
        and float(mean) > 0
        and float(add) > 0
        and float(day_win) >= float(min_day_win)
    )


def _entry_idx(ts_ns: np.ndarray, t: pd.Timestamp) -> int | None:
    t0 = int(to_ny(t).value)
    i0 = int(np.searchsorted(ts_ns, t0, side="left"))
    if i0 >= len(ts_ns):
        return None
    if (int(ts_ns[i0]) - t0) / 1e9 > 5:
        return None
    return i0


def _chase_entry(
    ts_ns: np.ndarray,
    last: np.ndarray,
    arm_ts: pd.Timestamp,
    *,
    chase: float,
    abort: float,
    wait_sec: int,
    slip: float,
) -> tuple[pd.Timestamp, str] | None:
    """From arm fill level, wait for +chase before -abort within wait_sec."""
    i0 = _entry_idx(ts_ns, arm_ts)
    if i0 is None:
        return None
    entry = float(last[i0]) * (1.0 + float(slip))
    if not np.isfinite(entry) or entry <= 0:
        return None
    sell_m = 1.0 - float(slip)
    end_ns = int(ts_ns[i0]) + int(wait_sec) * 1_000_000_000
    i_end = int(np.searchsorted(ts_ns, end_ns, side="right") - 1)
    for k in range(i0 + 1, i_end + 1):
        px = float(last[k])
        if not np.isfinite(px) or px <= 0:
            continue
        ret = px * sell_m / entry - 1.0
        if ret <= -abs(float(abort)):
            return None  # aborted
        if ret >= float(chase):
            # enter on this print (chase trigger)
            return to_ny(pd.Timestamp(ts_ns[k], unit="ns", tz="UTC")), "chase"
    return None


def _opt_ret_window(
    ts_ns: np.ndarray,
    last: np.ndarray,
    t: pd.Timestamp,
    lookback_sec: int,
    *,
    slip: float,
) -> float | None:
    """Option ret from last print ≤ t−lb to last print ≤ t (sell/buy slip)."""
    t1 = int(to_ny(t).value)
    t0 = t1 - int(lookback_sec) * 1_000_000_000
    i1 = int(np.searchsorted(ts_ns, t1, side="right") - 1)
    i0 = int(np.searchsorted(ts_ns, t0, side="right") - 1)
    if i0 < 0 or i1 < 0 or i1 <= i0:
        return None
    # stale guards
    if abs(int(ts_ns[i1]) - t1) > 5_000_000_000:
        return None
    if abs(int(ts_ns[i0]) - t0) > 5_000_000_000:
        return None
    a = float(last[i0]) * (1.0 + float(slip))
    b = float(last[i1]) * (1.0 - float(slip))
    if a <= 0 or b <= 0:
        return None
    return float(b / a - 1.0)


def _stock_signed(
    ts_ns: np.ndarray,
    px: np.ndarray,
    t: pd.Timestamp,
    lookback_sec: int,
    direction: str,
) -> float | None:
    direction2, sr = _stock_dir_arr(
        ts_ns, px, t, lookback_sec, 0.0
    )  # always return sr
    # recompute signed
    t_ns = int(to_ny(t).value)
    t0_ns = t_ns - int(lookback_sec) * 1_000_000_000
    i1 = int(np.searchsorted(ts_ns, t_ns, side="right") - 1)
    i0 = int(np.searchsorted(ts_ns, t0_ns, side="right") - 1)
    if i0 < 0 or i1 < 0:
        return None
    a, b = float(px[i0]), float(px[i1])
    if a <= 0 or b <= 0:
        return None
    raw = b / a - 1.0
    return float(raw if direction == "UP" else -raw)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", default=PROFILE)
    ap.add_argument("--tag", default="research_certainty_morph_tpsl_dual")
    ap.add_argument("--trades-root", default=str(DEFAULT_TRADES))
    ap.add_argument("--sessions", default="AM_0930_1000,CORE_1030_1130")
    ap.add_argument("--dirs", default="DN,UP")
    ap.add_argument("--thr", default="0.003,0.005")
    ap.add_argument("--morphs", default="opt_chase,sync")
    ap.add_argument("--chase", default="0.05,0.08,0.10,0.15")
    ap.add_argument("--abort", default="0.05,0.08")
    ap.add_argument("--wait-sec", default="120,180,300")
    ap.add_argument("--sync-stock-sec", default="30,60")
    ap.add_argument("--sync-opt-sec", default="15,30")
    ap.add_argument("--tp", default="0.15,0.20,0.25")
    ap.add_argument("--sl", default="0.10,0.12,0.15")
    ap.add_argument("--max-hold-sec", type=int, default=900)
    ap.add_argument("--slip", type=float, default=0.01)
    ap.add_argument("--stride-sec", type=int, default=60)
    ap.add_argument("--lookback-sec", type=int, default=60)
    ap.add_argument("--position-frac", type=float, default=0.10)
    ap.add_argument("--max-concurrent", type=int, default=2)
    ap.add_argument("--cooldown-minutes", type=float, default=5.0)
    ap.add_argument("--min-n", type=int, default=10)
    ap.add_argument("--min-day-win", type=float, default=0.55)
    args = ap.parse_args(argv)

    want_sess = {x.strip() for x in args.sessions.split(",") if x.strip()}
    sessions = tuple(s for s in SESSIONS if s[0] in want_sess)
    dirs = {x.strip().upper() for x in args.dirs.split(",") if x.strip()}
    thrs = [float(x) for x in args.thr.split(",") if x.strip()]
    morphs = [x.strip() for x in args.morphs.split(",") if x.strip()]
    chases = [float(x) for x in args.chase.split(",") if x.strip()]
    aborts = [float(x) for x in args.abort.split(",") if x.strip()]
    waits = [int(x) for x in args.wait_sec.split(",") if x.strip()]
    sync_stock = [int(x) for x in args.sync_stock_sec.split(",") if x.strip()]
    sync_opt = [int(x) for x in args.sync_opt_sec.split(",") if x.strip()]
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
        f"certainty morph {start_all}..{end_all} days={len(dates)} "
        f"sess={[s[0] for s in sessions]} morphs={morphs}",
        flush=True,
    )

    # Collect arm candidates + resolved option paths once
    arms: list[dict[str, Any]] = []
    for di, date in enumerate(dates):
        if di % 10 == 0:
            print(f"[day] {date} ({di+1}/{len(dates)}) arms={len(arms)}", flush=True)
        for sym in symbols:
            day = load_stock_1s_day(stock_1s, sym, date)
            if day is None or day.empty:
                continue
            tday = load_option_trades(trades_root, sym, date)
            if tday is None or tday.empty:
                continue
            tpaths = _paths_by_ticker(tday)
            if not tpaths:
                continue
            ts_ns, px = _stock_arrays(day)
            by_dte = lock.get((sym, date))
            if not by_dte:
                continue
            for sess_name, s0, s1 in sessions:
                t_start = pd.Timestamp(f"{date} {s0}:00", tz=NY) + pd.Timedelta(
                    seconds=int(args.lookback_sec)
                )
                t_end = pd.Timestamp(f"{date} {s1}:00", tz=NY)
                fired: set[tuple[str, float]] = set()
                t = t_start
                stride = pd.Timedelta(seconds=int(args.stride_sec))
                while t < t_end:
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
                        ticker, dte, _ = resolve_open_lock_contract(
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
                            continue
                        arr = tpaths.get(str(ticker).replace("O:", ""))
                        if arr is None:
                            continue
                        fired.add(key)
                        arms.append(
                            {
                                "date": date,
                                "symbol": sym,
                                "dir": direction,
                                "thr": float(thr),
                                "session": sess_name,
                                "arm_ts": to_ny(t),
                                "stock_ret_lb": float(sr),
                                "ticker": ticker,
                                "dte": dte,
                                "pts": arr[0],
                                "plast": arr[1],
                                "sts": ts_ns,
                                "spx": px,
                                "sess_end": t_end,
                            }
                        )
                    t += stride

    print(f"arms={len(arms)}; scoring morphs…", flush=True)

    def window_of(date: str) -> str | None:
        for wname, a, b in WINDOWS:
            if a <= date <= b:
                return wname
        return None

    # Build cell grid
    cells: list[dict[str, Any]] = []
    if "opt_chase" in morphs:
        for thr in thrs:
            for chase in chases:
                for abort in aborts:
                    if abort > chase:
                        continue
                    for wait in waits:
                        for tp in tps:
                            for sl in sls:
                                cells.append(
                                    {
                                        "name": f"chase_t{thr}_c{chase}_a{abort}_w{wait}_tp{tp}_sl{sl}",
                                        "morph": "opt_chase",
                                        "thr": thr,
                                        "chase": chase,
                                        "abort": abort,
                                        "wait_sec": wait,
                                        "tp": tp,
                                        "sl": sl,
                                    }
                                )
    if "sync" in morphs:
        for thr in thrs:
            for ss in sync_stock:
                for so in sync_opt:
                    for tp in tps:
                        for sl in sls:
                            cells.append(
                                {
                                    "name": f"sync_t{thr}_ss{ss}_so{so}_tp{tp}_sl{sl}",
                                    "morph": "sync",
                                    "thr": thr,
                                    "sync_stock_sec": ss,
                                    "sync_opt_sec": so,
                                    "tp": tp,
                                    "sl": sl,
                                }
                            )

    print(f"cells={len(cells)}", flush=True)
    score_rows: list[dict[str, Any]] = []
    dual_pass: list[dict[str, Any]] = []
    trade_dump: dict[str, pd.DataFrame] = {}

    for ci, cell in enumerate(cells):
        if ci % 40 == 0:
            print(f"[cell] {ci+1}/{len(cells)} dual_so_far={len(dual_pass)}", flush=True)
        win_raw: dict[str, list[dict]] = {w[0]: [] for w in WINDOWS}
        # also by session for reporting
        for arm in arms:
            if float(arm["thr"]) != float(cell["thr"]):
                continue
            wname = window_of(str(arm["date"]))
            if wname is None:
                continue
            pts, plast = arm["pts"], arm["plast"]
            entry_ts: pd.Timestamp | None = None
            reason_e = None

            if cell["morph"] == "opt_chase":
                hit = _chase_entry(
                    pts,
                    plast,
                    arm["arm_ts"],
                    chase=float(cell["chase"]),
                    abort=float(cell["abort"]),
                    wait_sec=int(cell["wait_sec"]),
                    slip=float(args.slip),
                )
                if hit is None:
                    continue
                entry_ts, reason_e = hit
                if entry_ts > arm["sess_end"] + pd.Timedelta(seconds=60):
                    continue
            else:  # sync: arm_ts is candidate; require sync conditions at arm
                entry_ts = arm["arm_ts"]
                s_signed = _stock_signed(
                    arm["sts"],
                    arm["spx"],
                    entry_ts,
                    int(cell["sync_stock_sec"]),
                    str(arm["dir"]),
                )
                if s_signed is None or s_signed < 0:
                    continue
                o_ret = _opt_ret_window(
                    pts,
                    plast,
                    entry_ts,
                    int(cell["sync_opt_sec"]),
                    slip=float(args.slip),
                )
                if o_ret is None or o_ret <= 0:
                    continue
                reason_e = "sync"

            sim = simulate_trade_tpsl(
                pts,
                plast,
                entry_ts,
                tp=float(cell["tp"]),
                sl=float(cell["sl"]),
                max_hold_sec=int(args.max_hold_sec),
                slip=float(args.slip),
            )
            if sim is None or not np.isfinite(sim["ret"]):
                continue
            et = to_ny(entry_ts)
            win_raw[wname].append(
                {
                    "date": arm["date"],
                    "symbol": arm["symbol"],
                    "dir": arm["dir"],
                    "session": arm["session"],
                    "arm_ts": str(arm["arm_ts"]),
                    "entry_ts": str(et),
                    "exit_ts": str(et + pd.Timedelta(seconds=sim["hold_sec"])),
                    "ticker": arm["ticker"],
                    "dte": arm["dte"],
                    "ret": sim["ret"],
                    "exit_reason": sim["reason"],
                    "hold_sec": sim["hold_sec"],
                    "entry_reason": reason_e,
                    "stock_ret_lb": arm["stock_ret_lb"],
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
                # one per symbol first chronologically already in arm fire;
                # portfolio concurrent
                sized.extend(
                    _portfolio_day(
                        sorted(rs, key=lambda x: (x["entry_ts"], x["symbol"])),
                        position_frac=float(args.position_frac),
                        max_concurrent=int(args.max_concurrent),
                        cooldown_minutes=float(args.cooldown_minutes),
                    )
                )
            st = _stats(sized)
            # session breakdown
            if sized:
                tdf = pd.DataFrame(sized)
                for sess in tdf["session"].unique():
                    g = tdf[tdf.session == sess]
                    st[f"n_{sess}"] = int(len(g))
                    st[f"mean_{sess}"] = float(g["ret"].mean())
            win_stats[wname] = st
            sized_all.extend(sized)

        both = True
        for wname, _, _ in WINDOWS:
            mn = int(args.min_n)
            if wname == "jul10_23":
                mn = min(mn, 8)
            if not _ok(win_stats[wname], min_n=mn, min_day_win=float(args.min_day_win)):
                both = False
                break

        row: dict[str, Any] = {
            **{k: cell[k] for k in cell},
            "dual_pass": both,
        }
        for wname, _, _ in WINDOWS:
            for k, v in win_stats[wname].items():
                row[f"{wname}_{k}"] = v
        score_rows.append(row)
        if both:
            dual_pass.append(row)
            trade_dump[cell["name"]] = pd.DataFrame(sized_all)
            print(
                f"  *** DUAL PASS {cell['name']} "
                f"MJ09 n={row.get('may_jul09_n')} add={row.get('may_jul09_add'):+} "
                f"win={row.get('may_jul09_win')} "
                f"J10 n={row.get('jul10_23_n')} add={row.get('jul10_23_add'):+}",
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
        name = p["name"]
        if name in trade_dump and len(trade_dump[name]):
            trade_dump[name].to_csv(out / f"trades_dual{i:02d}_{name}.csv", index=False)

    summary = {
        "morphs": morphs,
        "sessions": [s[0] for s in sessions],
        "windows": [list(w) for w in WINDOWS],
        "pricing": "option_trades_last_slip",
        "n_arms": int(len(arms)),
        "n_cells": int(len(cells)),
        "dual_pass_n": int(len(dual_pass)),
        "verdict": "PASS" if dual_pass else "REJECT",
        "champion": dual_pass[0] if dual_pass else None,
        "note": (
            "opt_chase: wait option +chase before -abort after stock arm. "
            "sync: stock+option both already favorable at entry clock."
        ),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    (out / "dual_pass.json").write_text(
        json.dumps(dual_pass[:80], indent=2, default=str), encoding="utf-8"
    )

    print("\n=== verdict", summary["verdict"], "dual_pass_n=", len(dual_pass), flush=True)
    if dual_pass:
        c = dual_pass[0]
        print(
            f"champion {c['name']}: "
            f"MJ09 n={c.get('may_jul09_n')} mean={c.get('may_jul09_mean')} "
            f"win={c.get('may_jul09_win')} day_win={c.get('may_jul09_day_win')} add={c.get('may_jul09_add')} | "
            f"J10 n={c.get('jul10_23_n')} mean={c.get('jul10_23_mean')} "
            f"win={c.get('jul10_23_win')} day_win={c.get('jul10_23_day_win')} add={c.get('jul10_23_add')}",
            flush=True,
        )
        # top 5 morph breakdown
        for p in dual_pass[:8]:
            print(
                f"  {p['morph']} {p['name']}: "
                f"MJ09 add={p.get('may_jul09_add'):+.3f} win={p.get('may_jul09_win'):.0%} "
                f"J10 add={p.get('jul10_23_add'):+.3f} win={p.get('jul10_23_win'):.0%}",
                flush=True,
            )
    else:
        if not score.empty:
            score["_sum"] = score["may_jul09_add"].fillna(0) + score["jul10_23_add"].fillna(0)
            near = score.sort_values("_sum", ascending=False).head(12)
            cols = [
                c
                for c in [
                    "morph",
                    "name",
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
            print(near[cols].to_string(index=False), flush=True)
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

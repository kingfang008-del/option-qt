#!/usr/bin/env python3
"""Option 1s put-flow scout (DN): put/call volume proxy → ATM put TP/SL.

Uses existing ``/mnt/s990/new_option_data_s3_trades`` (1s OHLCV aggregates).
No aggressor side. Independent of Mag7 Rule-A / Hunt.

Arm (per session, first fire):
  put_share >= τ  AND  put_vol_z >= z  AND  put_v >= min_v
  optional: stock_ret_lb <= max_stock_ret (stock dump confirm)

Dual windows: may_jul09 / jul10_23.

Example:
  PYTHONPATH=. python -m maga7.tools.scan_option_flow_scout_tpsl \\
    --tag research_option_flow_scout_dn_tpsl_dual
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
from maga7.common.option_flow import first_put_flow_dn_in_window, prepare_option_flow_day
from maga7.common.option_trade_tpsl import simulate_trade_tpsl
from maga7.common.option_trades import load_option_trades
from maga7.common.replay import to_ny
from maga7.common.stock_1s import session_dates
from maga7.tools.run_morning_sec_option_fill import _portfolio_day
from maga7.tools.scan_am_certainty_morph_tpsl import _ok, _stats
from maga7.tools.scan_session_horizon_foresight import (
    _paths_by_ticker,
    _spot_at_arr,
    _stock_arrays,
)

NY = "America/New_York"
DEFAULT_TRADES = Path("/mnt/s990/new_option_data_s3_trades")
PROFILE = (
    "maga7/CONFIG/strategy_profiles/"
    "single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1.json"
)

SESSIONS = (
    ("AM_0935_1030", "09:35", "10:30"),
    ("CORE_1030_1200", "10:30", "12:00"),
    ("MID_1200_1400", "12:00", "14:00"),
    ("PM_1400_1530", "14:00", "15:30"),
)
WINDOWS = (
    ("may_jul09", "2026-05-01", "2026-07-09"),
    ("jul10_23", "2026-07-10", "2026-07-23"),
)

# (name, max_stock_ret or None = no stock filter)
STOCK_MODES = (
    ("flow_only", None),
    ("stk_m3", -0.003),
    ("stk_m5", -0.005),
)


def _window_of(date: str) -> str | None:
    for name, a, b in WINDOWS:
        if a <= date <= b:
            return name
    return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", default=PROFILE)
    ap.add_argument("--tag", default="research_option_flow_scout_dn_tpsl_dual")
    ap.add_argument("--trades-root", default=str(DEFAULT_TRADES))
    ap.add_argument(
        "--sessions",
        default="AM_0935_1030,CORE_1030_1200,MID_1200_1400,PM_1400_1530",
    )
    ap.add_argument("--flow-sec", default="60,120")
    ap.add_argument("--put-share", default="0.55,0.60,0.65")
    ap.add_argument("--put-vol-z", default="1.5,2.0,3.0")
    ap.add_argument("--min-put-v", default="200,500")
    ap.add_argument(
        "--stock-modes",
        default="flow_only,stk_m3,stk_m5",
        help="flow_only|stk_m3|stk_m5",
    )
    ap.add_argument("--stock-lb-sec", type=int, default=120)
    ap.add_argument("--tp", default="0.15,0.20,0.25")
    ap.add_argument("--sl", default="0.15,0.20")
    ap.add_argument("--max-hold-sec", type=int, default=900)
    ap.add_argument("--slip", type=float, default=0.01)
    ap.add_argument("--stride-sec", type=int, default=15)
    ap.add_argument("--position-frac", type=float, default=0.10)
    ap.add_argument("--max-concurrent", type=int, default=2)
    ap.add_argument("--cooldown-minutes", type=float, default=10.0)
    ap.add_argument("--min-n", type=int, default=8)
    ap.add_argument("--min-day-win", type=float, default=0.55)
    args = ap.parse_args(argv)

    want_sess = {x.strip() for x in args.sessions.split(",") if x.strip()}
    sessions = tuple(s for s in SESSIONS if s[0] in want_sess)
    flow_secs = [int(x) for x in args.flow_sec.split(",") if x.strip()]
    shares = [float(x) for x in args.put_share.split(",") if x.strip()]
    zs = [float(x) for x in args.put_vol_z.split(",") if x.strip()]
    min_vs = [float(x) for x in args.min_put_v.split(",") if x.strip()]
    tps = [float(x) for x in args.tp.split(",") if x.strip()]
    sls = [float(x) for x in args.sl.split(",") if x.strip()]
    want_sm = {x.strip() for x in args.stock_modes.split(",") if x.strip()}
    stock_modes = [m for m in STOCK_MODES if m[0] in want_sm]

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
        f"option_flow DN scout {start_all}..{end_all} days={len(dates)} "
        f"stock_modes={[m[0] for m in stock_modes]}",
        flush=True,
    )

    arm_specs: list[dict[str, Any]] = []
    for fw in flow_secs:
        for sh in shares:
            for z in zs:
                for mv in min_vs:
                    for sm_name, max_sr in stock_modes:
                        arm_specs.append(
                            {
                                "flow_sec": int(fw),
                                "put_share": float(sh),
                                "put_vol_z": float(z),
                                "min_put_v": float(mv),
                                "stock_mode": sm_name,
                                "max_stock_ret": max_sr,
                                "spec_key": (
                                    f"put_f{fw}_sh{sh}_z{z}_v{int(mv)}_{sm_name}"
                                ),
                            }
                        )

    arms: list[dict[str, Any]] = []
    for di, date in enumerate(dates):
        if di % 10 == 0:
            print(f"[day] {date} ({di+1}/{len(dates)}) arms={len(arms)}", flush=True)
        for sym in symbols:
            tday = load_option_trades(trades_root, sym, date)
            flow = prepare_option_flow_day(tday)
            if flow is None:
                continue
            tpaths = _paths_by_ticker(tday)
            if not tpaths:
                continue
            day = load_stock_1s_day(stock_1s, sym, date)
            if day is None or day.empty:
                ts_ns = px = None
            else:
                ts_ns, px = _stock_arrays(day)
            by_dte = lock.get((sym, date))
            if not by_dte:
                continue
            for sess_name, s0, s1 in sessions:
                warm = max(max(flow_secs), int(args.stock_lb_sec), 120)
                t_start = pd.Timestamp(f"{date} {s0}:00", tz=NY) + pd.Timedelta(
                    seconds=int(warm)
                )
                t_end = pd.Timestamp(f"{date} {s1}:00", tz=NY)
                for spec in arm_specs:
                    hit = first_put_flow_dn_in_window(
                        flow,
                        t_start=t_start,
                        t_end=t_end,
                        window_sec=int(spec["flow_sec"]),
                        min_put_share=float(spec["put_share"]),
                        min_put_vol_z=float(spec["put_vol_z"]),
                        min_put_v=float(spec["min_put_v"]),
                        stock_ts_ns=ts_ns,
                        stock_px=px,
                        stock_lb_sec=int(args.stock_lb_sec),
                        max_stock_ret=spec["max_stock_ret"],
                        stride_sec=int(args.stride_sec),
                    )
                    if hit is None:
                        continue
                    t_arm, arm = hit
                    spot = (
                        _spot_at_arr(ts_ns, px, t_arm)
                        if ts_ns is not None and px is not None
                        else None
                    )
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
                    arr = tpaths.get(str(ticker).replace("O:", ""))
                    if arr is None:
                        continue
                    arms.append(
                        {
                            "date": date,
                            "symbol": sym,
                            "dir": "DN",
                            "session": sess_name,
                            "arm_ts": to_ny(t_arm),
                            "ticker": ticker,
                            "dte": dte,
                            "pts": arr[0],
                            "plast": arr[1],
                            "obs_put_share": arm.put_share,
                            "obs_put_vol_z": arm.put_vol_z,
                            "obs_put_v": arm.put_v,
                            "stock_ret_lb": arm.stock_ret_lb,
                            "flow_sec": int(spec["flow_sec"]),
                            "min_put_v": float(spec["min_put_v"]),
                            "stock_mode": str(spec["stock_mode"]),
                            "spec_key": str(spec["spec_key"]),
                        }
                    )

    print(f"arms={len(arms)}; scoring cells…", flush=True)
    cells: list[dict[str, Any]] = []
    for spec in arm_specs:
        for tp in tps:
            for sl in sls:
                cells.append(
                    {
                        "name": f"{spec['spec_key']}_tp{tp}_sl{sl}",
                        "tp": float(tp),
                        "sl": float(sl),
                        **{
                            k: spec[k]
                            for k in (
                                "flow_sec",
                                "put_share",
                                "put_vol_z",
                                "min_put_v",
                                "stock_mode",
                                "spec_key",
                            )
                        },
                    }
                )

    dual_pass: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []
    trade_dump: dict[str, pd.DataFrame] = {}

    for ci, cell in enumerate(cells):
        if ci % 50 == 0:
            print(f"[cell] {ci+1}/{len(cells)} dual_so_far={len(dual_pass)}", flush=True)
        win_raw: dict[str, list[dict]] = {w[0]: [] for w in WINDOWS}
        for arm in arms:
            if arm["spec_key"] != cell["spec_key"]:
                continue
            wname = _window_of(str(arm["date"]))
            if wname is None:
                continue
            entry_ts = arm["arm_ts"]
            sim = simulate_trade_tpsl(
                arm["pts"],
                arm["plast"],
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
                    "dir": "DN",
                    "session": arm["session"],
                    "entry_ts": str(et),
                    "exit_ts": str(et + pd.Timedelta(seconds=sim["hold_sec"])),
                    "ticker": arm["ticker"],
                    "dte": arm["dte"],
                    "ret": sim["ret"],
                    "exit_reason": sim["reason"],
                    "hold_sec": sim["hold_sec"],
                    "obs_put_share": arm.get("obs_put_share", arm.get("put_share")),
                    "obs_put_vol_z": arm.get("obs_put_vol_z", arm.get("put_vol_z")),
                    "stock_mode": arm["stock_mode"],
                    "cell": cell["name"],
                    "event_source": "option_flow_scout",
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
                mn = min(mn, 6)
            if not _ok(win_stats[wname], min_n=mn, min_day_win=float(args.min_day_win)):
                both = False
                break

        row: dict[str, Any] = {**cell, "dual_pass": both}
        for wname, _, _ in WINDOWS:
            for k, v in win_stats[wname].items():
                row[f"{wname}_{k}"] = v
        score_rows.append(row)
        if both:
            dual_pass.append(row)
            trade_dump[cell["name"]] = pd.DataFrame(sized_all)
            print(
                f"  *** DUAL PASS {cell['name']} "
                f"MJ09 n={row.get('may_jul09_n')} mean={row.get('may_jul09_mean'):+.3f} "
                f"J10 n={row.get('jul10_23_n')} mean={row.get('jul10_23_mean'):+.3f}",
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
        name = p["name"]
        if name in trade_dump and len(trade_dump[name]):
            trade_dump[name].to_csv(out / f"trades_dual{i:02d}_{name}.csv", index=False)

    summary = {
        "expert_kind": "option_flow_scout",
        "isolation": "independent of Mag7 Rule-A / Hunt",
        "data_note": (
            "Option 1s put/call volume share + put vol z from existing trades aggs; "
            "no aggressor. Pricing = option trades last±slip."
        ),
        "sessions": [s[0] for s in sessions],
        "windows": [list(w) for w in WINDOWS],
        "pricing": "option_trades_last_slip",
        "n_arms": int(len(arms)),
        "n_cells": int(len(cells)),
        "dual_pass_n": int(len(dual_pass)),
        "verdict": "PASS" if dual_pass else "REJECT",
        "champion": dual_pass[0] if dual_pass else None,
        "note": "Promote only after quote FillSpec dual PASS.",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    (out / "dual_pass.json").write_text(
        json.dumps(dual_pass[:50], indent=2, default=str), encoding="utf-8"
    )
    print("\n=== verdict", summary["verdict"], "dual_pass_n=", len(dual_pass), flush=True)
    if dual_pass:
        c = dual_pass[0]
        print(
            f"champion {c['name']}: "
            f"MJ09 n={c.get('may_jul09_n')} mean={c.get('may_jul09_mean')} "
            f"day_win={c.get('may_jul09_day_win')} | "
            f"J10 n={c.get('jul10_23_n')} mean={c.get('jul10_23_mean')} "
            f"day_win={c.get('jul10_23_day_win')}",
            flush=True,
        )
    elif len(score):
        score["_sum"] = score["may_jul09_add"].fillna(0) + score["jul10_23_add"].fillna(0)
        near = score.sort_values("_sum", ascending=False).head(15)
        cols = [
            c
            for c in [
                "name",
                "stock_mode",
                "may_jul09_n",
                "may_jul09_mean",
                "may_jul09_day_win",
                "jul10_23_n",
                "jul10_23_mean",
                "jul10_23_day_win",
            ]
            if c in near.columns
        ]
        print(near[cols].to_string(index=False), flush=True)
    print(f"wrote {out}", flush=True)
    return 0 if dual_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())

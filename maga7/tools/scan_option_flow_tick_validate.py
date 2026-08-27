#!/usr/bin/env python3
"""Validate put-flow scout on downloaded option *tick* prints (recent days).

Uses ``/mnt/s990/new_option_data_s3_tick`` (S3_DATA_KIND=tick).
Single-window scoreboard on available tick dates (default Jul10–23).
Pricing: tick last±slip on ATM put path (same book as signal).

Not a dual-window promote gate — smoke / OOS pocket check only.

Example:
  PYTHONPATH=. python -m maga7.tools.scan_option_flow_tick_validate \\
    --tag research_option_flow_tick_validate_jul10_23
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
from maga7.common.option_flow import (
    DEFAULT_TICK_ROOT,
    iter_put_flow_dn_in_window,
    load_option_tick_day,
    prepare_option_flow_day,
    tick_dates,
)
from maga7.common.option_trade_tpsl import simulate_trade_tpsl
from maga7.common.replay import to_ny
from maga7.tools.run_morning_sec_option_fill import _portfolio_day
from maga7.tools.scan_am_certainty_morph_tpsl import _ok, _stats
from maga7.tools.scan_session_horizon_foresight import (
    _paths_by_ticker,
    _spot_at_arr,
    _stock_arrays,
)

NY = "America/New_York"
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
STOCK_MODES = (
    ("flow_only", None),
    ("stk_m3", -0.003),
    ("stk_m5", -0.005),
)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", default=PROFILE)
    ap.add_argument("--tag", default="research_option_flow_tick_validate_jul10_23")
    ap.add_argument("--tick-root", default=str(DEFAULT_TICK_ROOT))
    ap.add_argument("--start-date", default="")
    ap.add_argument("--end-date", default="")
    ap.add_argument(
        "--sessions",
        default="AM_0935_1030,CORE_1030_1200,MID_1200_1400,PM_1400_1530",
    )
    ap.add_argument("--flow-sec", default="60,120")
    ap.add_argument("--put-share", default="0.55,0.60,0.65")
    ap.add_argument("--put-vol-z", default="1.5,2.0,3.0")
    ap.add_argument("--min-put-v", default="200,500,1000")
    ap.add_argument("--stock-modes", default="flow_only,stk_m3,stk_m5")
    ap.add_argument("--stock-lb-sec", type=int, default=120)
    ap.add_argument("--tp", default="0.15,0.20,0.25")
    ap.add_argument("--sl", default="0.15,0.20")
    ap.add_argument("--max-hold-sec", type=int, default=900)
    ap.add_argument("--slip", type=float, default=0.01)
    ap.add_argument(
        "--stride-sec",
        type=int,
        default=5,
        help="Signal scan stride in seconds (Mag7 opportunistic).",
    )
    ap.add_argument(
        "--rearm-gap-sec",
        type=int,
        default=60,
        help="Min gap between consecutive arms on same symbol/session (0=every stride).",
    )
    ap.add_argument(
        "--fire-mode",
        default="rising",
        choices=("rising", "pulse", "hold", "first"),
        help="rising=False→True edge; pulse=edge or new z/v impulse; hold=gate-on spam; first=1/session.",
    )
    ap.add_argument(
        "--pulse-z-delta",
        type=float,
        default=0.5,
        help="For fire-mode=pulse: require put_vol_z >= last_z + delta.",
    )
    ap.add_argument(
        "--pulse-v-mult",
        type=float,
        default=1.25,
        help="For fire-mode=pulse: or put_v >= last_v * mult.",
    )
    ap.add_argument(
        "--first-fire-only",
        action="store_true",
        help="Alias for --fire-mode first.",
    )
    ap.add_argument("--position-frac", type=float, default=0.10)
    ap.add_argument("--max-concurrent", type=int, default=4)
    ap.add_argument("--cooldown-minutes", type=float, default=1.0)
    ap.add_argument("--min-n", type=int, default=6)
    ap.add_argument("--min-day-win", type=float, default=0.55)
    args = ap.parse_args(argv)

    tick_root = Path(args.tick_root)
    dates = tick_dates(tick_root)
    if args.start_date:
        dates = [d for d in dates if d >= args.start_date]
    if args.end_date:
        dates = [d for d in dates if d <= args.end_date]
    if not dates:
        print(f"no tick dates under {tick_root}", flush=True)
        return 2

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
    lock = load_multidte_lock_index(Path(paths["open_locked_map"]).expanduser())
    otm = resolve_otm_rungs(prof, default=3)
    out = Path(paths["results_dir"]) / args.tag
    out.mkdir(parents=True, exist_ok=True)

    fire_mode = "first" if args.first_fire_only else str(args.fire_mode)
    print(
        f"option_flow TICK validate {dates[0]}..{dates[-1]} days={len(dates)} "
        f"root={tick_root} fire_mode={fire_mode} stride={args.stride_sec}s "
        f"rearm={args.rearm_gap_sec}s stock_modes={[m[0] for m in stock_modes]}",
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
                                "spec_key": f"put_f{fw}_sh{sh}_z{z}_v{int(mv)}_{sm_name}",
                            }
                        )

    arms: list[dict[str, Any]] = []
    n_tick_ok = 0
    for di, date in enumerate(dates):
        if di % 2 == 0:
            print(f"[day] {date} ({di+1}/{len(dates)}) arms={len(arms)}", flush=True)
        for sym in symbols:
            tday = load_option_tick_day(tick_root, sym, date)
            flow = prepare_option_flow_day(tday)
            if flow is None:
                continue
            n_tick_ok += 1
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
                    hits = iter_put_flow_dn_in_window(
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
                        rearm_gap_sec=int(args.rearm_gap_sec),
                        fire_mode=fire_mode,
                        pulse_z_delta=float(args.pulse_z_delta),
                        pulse_v_mult=float(args.pulse_v_mult),
                    )
                    for t_arm, arm in hits:
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
                                "flow_source": flow.get("source"),
                                "flow_sec": int(spec["flow_sec"]),
                                "min_put_v": float(spec["min_put_v"]),
                                "stock_mode": str(spec["stock_mode"]),
                                "spec_key": str(spec["spec_key"]),
                            }
                        )

    print(f"sym_days_with_flow={n_tick_ok} arms={len(arms)}; scoring…", flush=True)
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

    score_rows: list[dict[str, Any]] = []
    win_pass: list[dict[str, Any]] = []
    trade_dump: dict[str, pd.DataFrame] = {}

    for ci, cell in enumerate(cells):
        if ci % 40 == 0:
            print(f"[cell] {ci+1}/{len(cells)} pass_so_far={len(win_pass)}", flush=True)
        raw_all: list[dict] = []
        for arm in arms:
            if arm["spec_key"] != cell["spec_key"]:
                continue
            sim = simulate_trade_tpsl(
                arm["pts"],
                arm["plast"],
                arm["arm_ts"],
                tp=float(cell["tp"]),
                sl=float(cell["sl"]),
                max_hold_sec=int(args.max_hold_sec),
                slip=float(args.slip),
            )
            if sim is None or not np.isfinite(sim["ret"]):
                continue
            et = to_ny(arm["arm_ts"])
            raw_all.append(
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
                    "obs_put_share": arm["obs_put_share"],
                    "obs_put_vol_z": arm["obs_put_vol_z"],
                    "stock_mode": arm["stock_mode"],
                    "cell": cell["name"],
                    "event_source": "option_flow_tick",
                }
            )

        by_d: dict[str, list] = {}
        for r in raw_all:
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
            for sym, g in tdf.groupby("symbol"):
                st[f"n_{sym}"] = int(len(g))
                st[f"mean_{sym}"] = float(g["ret"].mean())

        ok = _ok(st, min_n=int(args.min_n), min_day_win=float(args.min_day_win))
        row = {**cell, "window_pass": ok, **{f"jul10_23_{k}": v for k, v in st.items()}}
        # also flat keys for readability
        for k, v in st.items():
            row[k] = v
        score_rows.append(row)
        if ok:
            win_pass.append(row)
            trade_dump[cell["name"]] = pd.DataFrame(sized)
            print(
                f"  *** WINDOW PASS {cell['name']} "
                f"n={row.get('n')} mean={row.get('mean'):+.3f} "
                f"day_win={row.get('day_win')}",
                flush=True,
            )

    score = pd.DataFrame(score_rows)
    score.to_csv(out / "scoreboard.csv", index=False)
    win_pass = sorted(
        win_pass,
        key=lambda r: float(r.get("add") or 0),
        reverse=True,
    )
    for i, p in enumerate(win_pass[:15]):
        name = p["name"]
        if name in trade_dump and len(trade_dump[name]):
            trade_dump[name].to_csv(out / f"trades_pass{i:02d}_{name}.csv", index=False)

    summary = {
        "expert_kind": "option_flow_tick_validate",
        "tick_root": str(tick_root),
        "dates": dates,
        "n_dates": len(dates),
        "fire_mode": fire_mode,
        "stride_sec": int(args.stride_sec),
        "rearm_gap_sec": int(args.rearm_gap_sec),
        "max_concurrent": int(args.max_concurrent),
        "cooldown_minutes": float(args.cooldown_minutes),
        "pricing": "option_tick_last_slip",
        "n_arms": int(len(arms)),
        "n_cells": int(len(cells)),
        "window_pass_n": int(len(win_pass)),
        "verdict": "VALIDATE_PASS" if win_pass else "VALIDATE_REJECT",
        "champion": win_pass[0] if win_pass else None,
        "note": (
            "Single-window check on available tick days only. "
            "Default fire_mode=rising (new episode), not hold-spam. "
            "Not dual-window promote; May–Jul9 not covered by current tick download."
        ),
        "gate": {"min_n": args.min_n, "min_day_win": args.min_day_win},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    (out / "window_pass.json").write_text(
        json.dumps(win_pass[:50], indent=2, default=str), encoding="utf-8"
    )
    print("\n=== verdict", summary["verdict"], "window_pass_n=", len(win_pass), flush=True)
    if win_pass:
        c = win_pass[0]
        print(
            f"champion {c['name']}: n={c.get('n')} mean={c.get('mean')} "
            f"day_win={c.get('day_win')} add={c.get('add')}",
            flush=True,
        )
    elif len(score):
        score["_sum"] = score["add"].fillna(0)
        near = score.sort_values("_sum", ascending=False).head(15)
        cols = [
            c
            for c in [
                "name",
                "stock_mode",
                "n",
                "mean",
                "day_win",
                "add",
                "n_days",
            ]
            if c in near.columns
        ]
        print(near[cols].to_string(index=False), flush=True)
    print(f"wrote {out}", flush=True)
    return 0 if win_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Overlay stock activity / MF gates on LOCK AM Pulse (FO@0.8% A+B).

Keeps the existing AM FO trigger; at decision_ts (feature+60s) apply causal
1s stock gates inspired by activity→MF research, then score dual windows.

Baseline cell: FO thr=0.008, tp15/sl20/h900 (profile LOCK), both dirs.
Overlays: mf/ret/vol align with FO direction; optional shorter hold / PP.

Example:
  PYTHONPATH=. python -m maga7.tools.scan_am_pulse_activity_overlay \\
    --tag research_am_pulse_activity_overlay
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from maga7.common.am_pulse_scout import (
    am_pulse_decision_ts,
    load_am_pulse_lane_cfg,
    parse_am_pulse_scout,
    scan_day,
)
from maga7.common.bar_agg import load_stock_1s_day
from maga7.common.config import load_profile
from maga7.common.open_lock import (
    load_multidte_lock_index,
    resolve_open_lock_contract,
    resolve_otm_rungs,
)
from maga7.common.option_trade_tpsl import simulate_trade_tpsl
from maga7.common.option_trades import load_option_trades
from maga7.common.replay import month_list, to_ny
from maga7.common.session_1s_features import features_at, prepare_day_arrays
from maga7.common.signals import load_stock_month_files
from maga7.common.stock_1s import session_dates
from maga7.tools.run_morning_sec_option_fill import _portfolio_day
from maga7.tools.scan_am_certainty_morph_tpsl import _ok, _stats
from maga7.tools.scan_am_pocket_risk_optimize import (
    _equity_stats,
    _month_compounds,
    simulate_trade_profit_protect,
)
from maga7.tools.scan_session_horizon_foresight import (
    _paths_by_ticker,
    _spot_at_arr,
    _stock_arrays,
)

DEFAULT_TRADES = Path("/mnt/s990/new_option_data_s3_trades")
PROFILE = (
    "maga7/CONFIG/strategy_profiles/"
    "single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1.json"
)
WINDOWS = (
    ("may_jul09", "2026-05-01", "2026-07-09"),
    ("jul10_23", "2026-07-10", "2026-07-23"),
)
NY = "America/New_York"
GateFn = Callable[[pd.Series], bool]


def _window_of(date: str) -> str | None:
    for name, a, b in WINDOWS:
        if a <= date <= b:
            return name
    return None


def _spot_from_1m(day: pd.DataFrame, ts: pd.Timestamp) -> float | None:
    if day is None or day.empty:
        return None
    t = to_ny(ts)
    sub = day[pd.to_datetime(day["timestamp"]) <= t]
    if sub.empty:
        return None
    px = float(sub.iloc[-1]["close"])
    return px if px > 0 else None


def _signed(row: pd.Series, feat: str) -> float:
    v = float(row.get(feat) or np.nan)
    if not np.isfinite(v):
        return float("nan")
    return v if str(row["dir"]).upper() == "UP" else -v


def _overlay_gates() -> list[tuple[str, GateFn]]:
    def none(r: pd.Series) -> bool:
        return True

    def mf100_align(r: pd.Series) -> bool:
        return _signed(r, "mf100") > 0

    def ret60_align(r: pd.Series) -> bool:
        return _signed(r, "ret_60") > 0

    def mf_ret(r: pd.Series) -> bool:
        return _signed(r, "mf100") > 0 and _signed(r, "ret_60") > 0

    def volz15(r: pd.Series) -> bool:
        v = float(r.get("vol_z") or np.nan)
        return np.isfinite(v) and v >= 1.5

    def volr12(r: pd.Series) -> bool:
        v = float(r.get("volume_ratio_60") or np.nan)
        return np.isfinite(v) and v >= 1.2

    def mf_ret_volr(r: pd.Series) -> bool:
        v = float(r.get("volume_ratio_60") or np.nan)
        return (
            _signed(r, "mf100") > 0
            and _signed(r, "ret_60") > 0
            and np.isfinite(v)
            and v >= 1.2
        )

    def volz15_mf_ret(r: pd.Series) -> bool:
        v = float(r.get("vol_z") or np.nan)
        return (
            np.isfinite(v)
            and v >= 1.5
            and _signed(r, "mf100") > 0
            and _signed(r, "ret_60") > 0
        )

    def streak3(r: pd.Series) -> bool:
        d = str(r["dir"]).upper()
        s = float(r.get("streak_up") or 0) if d == "UP" else float(r.get("streak_dn") or 0)
        return s >= 3

    def mf_ret_streak(r: pd.Series) -> bool:
        return mf_ret(r) and streak3(r)

    return [
        ("none", none),
        ("mf100", mf100_align),
        ("ret60", ret60_align),
        ("mf+ret60", mf_ret),
        ("volz15", volz15),
        ("volr12", volr12),
        ("mf+ret60+volr12", mf_ret_volr),
        ("volz15+mf+ret60", volz15_mf_ret),
        ("mf+ret60+streak3", mf_ret_streak),
    ]


def _exit_grid() -> list[dict[str, Any]]:
    return [
        {"name": "lock_tp15_sl20_h900", "mode": "tpsl", "tp": 0.15, "sl": 0.20, "max_hold": 900},
        {"name": "pp_a08_f03_tp15_sl20", "mode": "pp", "tp": 0.15, "sl": 0.20, "max_hold": 900, "arm": 0.08, "floor": 0.03},
        {"name": "tp12_sl15_h600", "mode": "tpsl", "tp": 0.12, "sl": 0.15, "max_hold": 600},
        {"name": "tp10_sl12_h300", "mode": "tpsl", "tp": 0.10, "sl": 0.12, "max_hold": 300},
        {"name": "tp08_sl15_h240", "mode": "tpsl", "tp": 0.08, "sl": 0.15, "max_hold": 240},
    ]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", default=PROFILE)
    ap.add_argument("--tag", default="research_am_pulse_activity_overlay")
    ap.add_argument("--trades-root", default=str(DEFAULT_TRADES))
    ap.add_argument("--fo-thr", type=float, default=0.008)
    ap.add_argument("--bar-delay-sec", type=int, default=60)
    ap.add_argument("--slip", type=float, default=0.01)
    ap.add_argument("--position-frac", type=float, default=0.10)
    ap.add_argument("--max-concurrent", type=int, default=2)
    ap.add_argument("--cooldown-minutes", type=float, default=10.0)
    ap.add_argument("--min-n", type=int, default=8)
    ap.add_argument("--min-day-win", type=float, default=0.55)
    ap.add_argument(
        "--lanes",
        default="A,B",
        help="A=am_pulse window, B=am_pulse_extension",
    )
    args = ap.parse_args(argv)

    prof = load_profile(args.profile)
    lanes_want = {x.strip().upper() for x in args.lanes.split(",") if x.strip()}
    lane_cfgs: list[tuple[str, dict[str, Any]]] = []
    if "A" in lanes_want:
        lane_cfgs.append(("A", load_am_pulse_lane_cfg(prof, "am_pulse")))
    if "B" in lanes_want:
        lane_cfgs.append(("B", load_am_pulse_lane_cfg(prof, "am_pulse_extension")))

    paths = prof["_paths"]
    symbols = list(prof.get("symbols") or [])
    stock_root = Path(paths["stock_root"])
    stock_1s = Path(paths.get("stock_1s_root") or "/mnt/s990/data/raw_1s/stocks").expanduser()
    trades_root = Path(args.trades_root)
    lock = load_multidte_lock_index(Path(paths["open_locked_map"]).expanduser())
    otm = resolve_otm_rungs(prof, default=3)
    out = Path(paths["results_dir"]) / args.tag
    out.mkdir(parents=True, exist_ok=True)

    start_all = min(w[1] for w in WINDOWS)
    end_all = max(w[2] for w in WINDOWS)
    dates = [d for d in session_dates(start_all, end_all) if start_all <= d <= end_all]
    months = month_list(start_all, end_all)
    print(
        f"AM pulse overlay FO@{args.fo_thr} delay={args.bar_delay_sec}s "
        f"lanes={sorted(lanes_want)} days={len(dates)}",
        flush=True,
    )

    stock_by_sym: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        sdf = load_stock_month_files(stock_root, sym, months)
        if sdf is not None and not sdf.empty:
            stock_by_sym[sym] = sdf

    # Build LOCK FO arms once (A+B)
    arms: list[dict[str, Any]] = []
    for di, date in enumerate(dates):
        if di % 10 == 0:
            print(f"[day] {date} ({di+1}/{len(dates)}) arms={len(arms)}", flush=True)
        cal = _window_of(date)
        if cal is None:
            continue
        for sym in symbols:
            sdf = stock_by_sym.get(sym)
            if sdf is None:
                continue
            day1m = sdf[sdf["date"].astype(str) == date]
            if day1m.empty:
                continue
            by_dte = lock.get((sym, date))
            if not by_dte:
                continue
            tday = load_option_trades(trades_root, sym, date)
            if tday is None or tday.empty:
                continue
            tpaths = _paths_by_ticker(tday)
            if not tpaths:
                continue
            day1s = load_stock_1s_day(stock_1s, sym, date)
            sarr = prepare_day_arrays(day1s) if day1s is not None and not day1s.empty else None
            ts_ns = px = None
            if day1s is not None and not day1s.empty:
                ts_ns, px = _stock_arrays(day1s)

            for lane_name, lane_cfg in lane_cfgs:
                dirs = {str(x).upper() for x in (lane_cfg.get("dirs") or ["DN", "UP"])}
                max_fo = float(lane_cfg.get("max_fav_from_open", 0.0) or 0.0)
                prefer_dte = int(lane_cfg.get("prefer_dte", 0) or 0)
                allowed_raw = lane_cfg.get("allowed_dte") or (prof.get("lock") or {}).get(
                    "allowed_dte"
                ) or [0, 1, 2]
                allowed_dte = [int(x) for x in allowed_raw]
                cfg = parse_am_pulse_scout(
                    {
                        "enabled": True,
                        "window_start": str(lane_cfg.get("window_start") or "09:30"),
                        "window_end": str(lane_cfg.get("window_end") or "10:30"),
                        "min_fav_from_open": float(args.fo_thr),
                        "max_fav_from_open": max_fo if max_fo > 0 else 0.015,
                        "lookback_bars": int(lane_cfg.get("lookback_bars") or 2),
                        "min_lookback_ret": 0.99,
                        "dirs": sorted(dirs),
                        "max_alerts_per_symbol": 1,
                    }
                )
                for a in scan_day(day1m, date=date, symbol=sym, cfg=cfg):
                    if a.arm != "FO" or a.dir not in dirs:
                        continue
                    arm_ts = to_ny(pd.Timestamp(a.ts))
                    decision_ts = am_pulse_decision_ts(
                        arm_ts, delay_seconds=int(args.bar_delay_sec)
                    )
                    feat = features_at(sarr, decision_ts) if sarr is not None else None
                    if feat is None:
                        continue
                    spot = None
                    if ts_ns is not None and px is not None:
                        spot = _spot_at_arr(ts_ns, px, decision_ts)
                    if spot is None:
                        spot = _spot_from_1m(day1m, decision_ts)
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
                    row = {
                        "date": date,
                        "calendar": cal,
                        "lane": lane_name,
                        "symbol": sym,
                        "dir": a.dir,
                        "arm_ts": arm_ts,
                        "decision_ts": decision_ts,
                        "fav_from_open": float(a.fav_from_open),
                        "ticker": ticker,
                        "dte": dte,
                        "pts": arr[0],
                        "plast": arr[1],
                        **feat,
                    }
                    arms.append(row)

    print(f"enriched FO arms={len(arms)}", flush=True)
    arms_df = pd.DataFrame(
        [{k: v for k, v in a.items() if k not in ("pts", "plast")} for a in arms]
    )
    arms_df.to_csv(out / "arms_enriched.csv", index=False)

    overlays = _overlay_gates()
    exits = _exit_grid()
    score_rows: list[dict[str, Any]] = []

    for gname, gfn in overlays:
        mask = [bool(gfn(pd.Series(a))) for a in arms]
        subset = [a for a, ok in zip(arms, mask) if ok]
        for ex in exits:
            raw: list[dict[str, Any]] = []
            for a in subset:
                if ex["mode"] == "pp":
                    sim = simulate_trade_profit_protect(
                        a["pts"],
                        a["plast"],
                        a["decision_ts"],
                        tp=float(ex["tp"]),
                        sl=float(ex["sl"]),
                        max_hold_sec=int(ex["max_hold"]),
                        arm_ret=float(ex["arm"]),
                        floor_ret=float(ex["floor"]),
                        slip=float(args.slip),
                    )
                else:
                    sim = simulate_trade_tpsl(
                        a["pts"],
                        a["plast"],
                        a["decision_ts"],
                        tp=float(ex["tp"]),
                        sl=float(ex["sl"]),
                        max_hold_sec=int(ex["max_hold"]),
                        slip=float(args.slip),
                    )
                if sim is None or not np.isfinite(sim.get("ret", np.nan)):
                    continue
                et = a["decision_ts"]
                raw.append(
                    {
                        "date": a["date"],
                        "symbol": a["symbol"],
                        "dir": a["dir"],
                        "lane": a["lane"],
                        "calendar": a["calendar"],
                        "entry_ts": et,
                        "exit_ts": et + pd.Timedelta(seconds=float(sim["hold_sec"])),
                        "ret": float(sim["ret"]),
                        "exit_reason": str(sim["reason"]),
                        "hold_sec": float(sim["hold_sec"]),
                    }
                )

            by_w: dict[str, list] = {w[0]: [] for w in WINDOWS}
            for t in raw:
                by_w.setdefault(str(t["calendar"]), []).append(t)

            row: dict[str, Any] = {
                "overlay": gname,
                "exit": ex["name"],
                "n_raw": len(raw),
                "frac_up": float(np.mean([t["dir"] == "UP" for t in raw])) if raw else 0.0,
                "frac_A": float(np.mean([t["lane"] == "A" for t in raw])) if raw else 0.0,
            }
            dual_ok = True
            for wname, _, _ in WINDOWS:
                bucket = by_w.get(wname) or []
                sized = _portfolio_day(
                    sorted(bucket, key=lambda x: (x["entry_ts"], x["symbol"])),
                    position_frac=float(args.position_frac),
                    max_concurrent=int(args.max_concurrent),
                    cooldown_minutes=float(args.cooldown_minutes),
                )
                st = _stats(sized)
                min_n = int(args.min_n) if wname == "may_jul09" else min(int(args.min_n), 8)
                ok = _ok(st, min_n=min_n, min_day_win=float(args.min_day_win))
                dual_ok = dual_ok and ok
                for k, v in st.items():
                    row[f"{wname}_{k}"] = v
                row[f"{wname}_ok"] = ok
                if wname == "may_jul09":
                    eq = _equity_stats(pd.DataFrame(sized))
                    row["disc_compound"] = eq.get("compound")
                    row["disc_maxdd"] = eq.get("maxdd")
                    row["disc_trade_win"] = eq.get("trade_win")
                    row["disc_n"] = eq.get("n")
            blind_bucket = by_w.get("jul10_23") or []
            sized_b = _portfolio_day(
                sorted(blind_bucket, key=lambda x: (x["entry_ts"], x["symbol"])),
                position_frac=float(args.position_frac),
                max_concurrent=int(args.max_concurrent),
                cooldown_minutes=float(args.cooldown_minutes),
            )
            eqb = _equity_stats(pd.DataFrame(sized_b))
            row["blind_compound"] = eqb.get("compound")
            row["blind_maxdd"] = eqb.get("maxdd")
            row["blind_trade_win"] = eqb.get("trade_win")
            row["blind_n"] = eqb.get("n")
            all_sized = _portfolio_day(
                sorted(raw, key=lambda x: (x["entry_ts"], x["symbol"])),
                position_frac=float(args.position_frac),
                max_concurrent=int(args.max_concurrent),
                cooldown_minutes=float(args.cooldown_minutes),
            )
            months_c = _month_compounds(pd.DataFrame(all_sized))
            row["may"] = months_c.get("2026-05")
            row["jun"] = months_c.get("2026-06")
            row["jul"] = months_c.get("2026-07")
            row["dual_pass"] = bool(dual_ok)
            score_rows.append(row)
            print(
                f"{gname:20s} {ex['name']:24s} n={len(raw):3d} "
                f"disc_cmp={row.get('disc_compound')} "
                f"blind_cmp={row.get('blind_compound')} dual={dual_ok}",
                flush=True,
            )

    sb = pd.DataFrame(score_rows)
    sb.to_csv(out / "scoreboard.csv", index=False)

    base = sb[(sb.overlay == "none") & (sb.exit == "lock_tp15_sl20_h900")]
    base_row = base.iloc[0].to_dict() if len(base) else {}
    bcmp = float(base_row.get("disc_compound") or 0)
    bdd = float(base_row.get("disc_maxdd") or -1)
    bw = float(base_row.get("disc_trade_win") or 0)
    bblind = float(base_row.get("blind_compound") or 0)

    improved = sb[
        (sb["disc_n"].fillna(0) >= 15)
        & (sb["disc_trade_win"].fillna(0) >= bw - 0.05)
        & (sb["disc_maxdd"].fillna(-1) >= min(bdd - 0.02, -0.05))
        & (sb["disc_compound"].fillna(0) > bcmp + 0.03)
        & (sb["blind_compound"].fillna(0) >= min(bblind, 0) - 0.02)
        & (sb["may"].fillna(0) > 0)
    ].sort_values(["disc_compound", "blind_compound"], ascending=[False, False])

    dual_pass = sb[sb["dual_pass"] == True]  # noqa: E712
    # same exit, overlay only
    lock_exit = sb[sb.exit == "lock_tp15_sl20_h900"].sort_values(
        "disc_compound", ascending=False
    )

    verdict = {
        "protocol": "am_pulse_fo08_activity_mf_overlay",
        "baseline": base_row,
        "improved_vs_baseline": improved.head(15).to_dict(orient="records") if len(improved) else [],
        "dual_pass": dual_pass.to_dict(orient="records") if len(dual_pass) else [],
        "lock_exit_overlays": lock_exit.to_dict(orient="records"),
        "n_arms": len(arms),
    }
    (out / "summary.json").write_text(json.dumps(verdict, indent=2, default=str), encoding="utf-8")

    cols = [
        c
        for c in [
            "overlay",
            "exit",
            "n_raw",
            "disc_n",
            "disc_trade_win",
            "disc_maxdd",
            "disc_compound",
            "blind_n",
            "blind_trade_win",
            "blind_compound",
            "dual_pass",
            "may",
            "jun",
            "jul",
        ]
        if c in sb.columns
    ]
    print("\nBASELINE", flush=True)
    print(base[cols].to_string(index=False) if len(base) else "(none)", flush=True)
    print("\nLOCK exit × overlays", flush=True)
    print(lock_exit[cols].to_string(index=False), flush=True)
    print("\nIMPROVED vs baseline", flush=True)
    print(improved[cols].head(12).to_string(index=False) if len(improved) else "(none)", flush=True)
    print("\nDUAL PASS", flush=True)
    print(dual_pass[cols].to_string(index=False) if len(dual_pass) else "(none)", flush=True)
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

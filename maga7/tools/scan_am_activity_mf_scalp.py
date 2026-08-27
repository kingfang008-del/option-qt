#!/usr/bin/env python3
"""AM activity → stock MF timing scalp (not mindless call/put flip).

Corrected thesis:
  1) Option print activity spikes (both sides) → symbol is "hot"
  2) Stock sliding MF / ret / vol windows pick UP(call) vs DN(put)
  3) Short-hold TP/SL on open-ladder contract

This is distinct from opp_first / pingpong (those flipped on option mark path).

Example:
  PYTHONPATH=. python -m maga7.tools.scan_am_activity_mf_scalp \\
    --tag research_am_activity_mf_scalp
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
from maga7.common.option_flow import _idx_at_or_before, _window_sums, prepare_option_flow_day
from maga7.common.option_trades import load_option_trades
from maga7.common.replay import to_ny
from maga7.common.session_1s_features import features_at, prepare_day_arrays
from maga7.common.stock_1s import session_dates
from maga7.tools.run_morning_sec_option_fill import _portfolio_day
from maga7.tools.scan_am_pocket_exit_design import _path_window, simulate_exit
from maga7.tools.scan_am_pocket_risk_optimize import _equity_stats, _month_compounds
from maga7.tools.scan_session_horizon_foresight import _paths_by_ticker, _spot_at_arr

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


def _window_of(date: str) -> str | None:
    for name, a, b in WINDOWS:
        if a <= date <= b:
            return name
    return None


def opt_activity_at(
    flow: dict[str, Any],
    *,
    i: int,
    window_sec: int,
    baseline_sec: int = 300,
) -> dict[str, float] | None:
    """Both-side option activity: total vol z + call/put share."""
    w = _window_sums(flow, i=i, window_sec=int(window_sec))
    if w is None:
        return None
    pv, cv = w
    tot = pv + cv
    if tot <= 0:
        return None
    base = _window_sums(flow, i=i, window_sec=max(int(baseline_sec), int(window_sec) * 2))
    if base is None:
        return None
    bp, bc = base
    base_win = max(int(baseline_sec), int(window_sec) * 2)
    base_rate = (bp + bc) / float(base_win)
    win_rate = tot / float(max(1, int(window_sec)))
    if base_rate <= 1e-9:
        z = 0.0 if win_rate <= 0 else 10.0
    else:
        z = win_rate / base_rate
    return {
        "opt_vol_z": float(z),
        "opt_tot_v": float(tot),
        "put_share": float(pv / tot),
        "call_share": float(cv / tot),
        "put_v": float(pv),
        "call_v": float(cv),
    }


def _dir_from_stock(feat: dict[str, Any], mode: str) -> str | None:
    """Causal direction from stock sliding windows. None = no trade."""
    mf = float(feat.get("mf100") or np.nan)
    mf3 = float(feat.get("mf300") or np.nan)
    r60 = float(feat.get("ret_60") or np.nan)
    r30 = float(feat.get("ret_30") or np.nan)
    volr = float(feat.get("volume_ratio_60") or np.nan)
    streak_u = float(feat.get("streak_up") or 0)
    streak_d = float(feat.get("streak_dn") or 0)

    def _sign_up(v: float) -> bool:
        return np.isfinite(v) and v > 0

    def _sign_dn(v: float) -> bool:
        return np.isfinite(v) and v < 0

    if mode == "mf100":
        if _sign_up(mf):
            return "UP"
        if _sign_dn(mf):
            return "DN"
        return None
    if mode == "ret60":
        if _sign_up(r60):
            return "UP"
        if _sign_dn(r60):
            return "DN"
        return None
    if mode == "mf100+ret60":
        if _sign_up(mf) and _sign_up(r60):
            return "UP"
        if _sign_dn(mf) and _sign_dn(r60):
            return "DN"
        return None
    if mode == "mf100+ret60+volr12":
        if not (np.isfinite(volr) and volr >= 1.2):
            return None
        if _sign_up(mf) and _sign_up(r60):
            return "UP"
        if _sign_dn(mf) and _sign_dn(r60):
            return "DN"
        return None
    if mode == "mf100+ret30+streak3":
        if _sign_up(mf) and _sign_up(r30) and streak_u >= 3:
            return "UP"
        if _sign_dn(mf) and _sign_dn(r30) and streak_d >= 3:
            return "DN"
        return None
    if mode == "mf300+ret60":
        if _sign_up(mf3) and _sign_up(r60):
            return "UP"
        if _sign_dn(mf3) and _sign_dn(r60):
            return "DN"
        return None
    return None


def _policy_grid() -> list[dict[str, Any]]:
    cfgs: list[dict[str, Any]] = []
    for win in (60, 120):
        for z in (2.0, 2.5, 3.0):
            for min_v in (150, 300):
                for dmode in (
                    "mf100+ret60",
                    "mf100+ret60+volr12",
                    "mf100",
                    "ret60",
                ):
                    for tp, sl, h in (
                        (0.08, 0.10, 30),
                        (0.12, 0.10, 45),
                        (0.08, 0.15, 60),
                    ):
                        cfgs.append(
                            {
                                "name": (
                                    f"act_w{win}_z{z:g}_v{min_v}_{dmode}"
                                    f"_tp{tp:g}_sl{sl:g}_h{h}"
                                ),
                                "flow_win": win,
                                "min_z": z,
                                "min_v": min_v,
                                "dir_mode": dmode,
                                "tp": tp,
                                "sl": sl,
                                "max_hold": h,
                            }
                        )
    return cfgs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", default=PROFILE)
    ap.add_argument("--trades-root", default=str(DEFAULT_TRADES))
    ap.add_argument("--tag", default="research_am_activity_mf_scalp")
    ap.add_argument("--window-start", default="09:30")
    ap.add_argument("--window-end", default="11:30")
    ap.add_argument("--stride-sec", type=int, default=5)
    ap.add_argument("--rearm-gap-sec", type=int, default=60)
    ap.add_argument("--max-arms-per-sym-day", type=int, default=3)
    ap.add_argument("--position-frac", type=float, default=0.20)
    ap.add_argument("--max-concurrent", type=int, default=5)
    ap.add_argument("--cooldown-minutes", type=float, default=5.0)
    ap.add_argument("--slip", type=float, default=0.01)
    ap.add_argument("--max-policies", type=int, default=0, help="0=all; debug cap")
    args = ap.parse_args(argv)

    prof = load_profile(args.profile)
    out = Path(prof["_paths"]["results_dir"]) / args.tag
    out.mkdir(parents=True, exist_ok=True)
    stock_1s = Path(prof["_paths"]["stock_1s_root"])
    trades_root = Path(args.trades_root)
    lock = load_multidte_lock_index(Path(prof["_paths"]["open_locked_map"]).expanduser())
    otm = resolve_otm_rungs(prof, default=3)
    symbols = list(prof.get("symbols") or [])

    start_all = min(w[1] for w in WINDOWS)
    end_all = max(w[2] for w in WINDOWS)
    dates = [d for d in session_dates(start_all, end_all) if start_all <= d <= end_all]
    policies = _policy_grid()
    if int(args.max_policies) > 0:
        policies = policies[: int(args.max_policies)]
    print(
        f"activity→MF scalp {args.window_start}-{args.window_end} "
        f"days={len(dates)} policies={len(policies)} syms={len(symbols)}",
        flush=True,
    )

    # Precompute events once per (flow_win, min_z, min_v, dir_mode) family to avoid
    # re-walking days per TP/SL. Group policies.
    families: dict[tuple, list[dict[str, Any]]] = {}
    for pol in policies:
        key = (pol["flow_win"], pol["min_z"], pol["min_v"], pol["dir_mode"])
        families.setdefault(key, []).append(pol)

    # Build candidate arms keyed by family
    # arm: date,sym,dir,entry_ts,feats,calendar
    family_arms: dict[tuple, list[dict[str, Any]]] = {k: [] for k in families}

    path_cache: dict[tuple[str, str], dict] = {}

    def opt_paths(date: str, sym: str):
        key = (date, sym)
        if key not in path_cache:
            tday = load_option_trades(trades_root, sym, date)
            path_cache[key] = {
                "tday": tday,
                "flow": prepare_option_flow_day(tday) if tday is not None and not tday.empty else None,
                "by_ticker": _paths_by_ticker(tday) if tday is not None and not tday.empty else {},
            }
        return path_cache[key]

    for di, date in enumerate(dates):
        if di % 10 == 0:
            print(f"[day] {date} ({di+1}/{len(dates)})", flush=True)
        cal = _window_of(date)
        if cal is None:
            continue
        for sym in symbols:
            pack = opt_paths(date, sym)
            flow = pack["flow"]
            if flow is None:
                continue
            raw = load_stock_1s_day(stock_1s, sym, date)
            if raw is None or raw.empty:
                continue
            sarr = prepare_day_arrays(raw)
            by_dte = lock.get((sym, date))
            if not by_dte:
                continue
            t0 = to_ny(pd.Timestamp(f"{date} {args.window_start}", tz=NY))
            t1 = to_ny(pd.Timestamp(f"{date} {args.window_end}", tz=NY))
            ts_ns = flow["ts_ns"]
            i0 = int(np.searchsorted(ts_ns, int(t0.value), side="left"))
            i1 = int(np.searchsorted(ts_ns, int(t1.value), side="right") - 1)
            if i1 < i0:
                continue

            # walk once; evaluate all family activity thresholds
            last_fire: dict[tuple, int] = {k: -10**9 for k in families}
            n_fire: dict[tuple, int] = {k: 0 for k in families}
            stride = max(1, int(args.stride_sec))
            for i in range(i0, i1 + 1, stride):
                t_ns = int(ts_ns[i])
                t = pd.Timestamp(t_ns, tz="UTC").tz_convert(NY)
                feat = features_at(sarr, t)
                if feat is None:
                    continue
                spot = float(feat.get("px") or np.nan)
                if not np.isfinite(spot) or spot <= 0:
                    spot_v = _spot_at_arr(sarr["ts_ns"], sarr["close"], t)
                    spot = float(spot_v) if spot_v is not None else float("nan")
                if not np.isfinite(spot):
                    continue

                for key in families:
                    flow_win, min_z, min_v, dmode = key
                    if n_fire[key] >= int(args.max_arms_per_sym_day):
                        continue
                    if (t_ns - last_fire[key]) / 1e9 < float(args.rearm_gap_sec):
                        continue
                    act = opt_activity_at(flow, i=i, window_sec=int(flow_win))
                    if act is None:
                        continue
                    if act["opt_vol_z"] < float(min_z) or act["opt_tot_v"] < float(min_v):
                        continue
                    direction = _dir_from_stock(feat, str(dmode))
                    if direction is None:
                        continue
                    # resolve contract
                    ticker, dte, _ = resolve_open_lock_contract(
                        by_dte,
                        direction=direction,
                        moneyness="ATM",
                        spot=spot,
                        prefer_dte=0,
                        allowed_dte=(0, 1, 2),
                        clear_otm_thresh=0.01,
                        ladder=True,
                        otm_rungs=otm,
                    )
                    if not ticker:
                        continue
                    path = pack["by_ticker"].get(str(ticker).replace("O:", ""))
                    if path is None:
                        continue
                    family_arms[key].append(
                        {
                            "date": date,
                            "symbol": sym,
                            "dir": direction,
                            "entry_ts": t,
                            "calendar": cal,
                            "ticker": str(ticker),
                            "dte": dte,
                            "opt_vol_z": act["opt_vol_z"],
                            "opt_tot_v": act["opt_tot_v"],
                            "put_share": act["put_share"],
                            "mf100": float(feat.get("mf100") or np.nan),
                            "ret_60": float(feat.get("ret_60") or np.nan),
                            "volume_ratio_60": float(feat.get("volume_ratio_60") or np.nan),
                            "path": path,
                        }
                    )
                    last_fire[key] = t_ns
                    n_fire[key] += 1

    # simulate exits per policy
    score_rows: list[dict[str, Any]] = []
    for key, pols in families.items():
        arms = family_arms[key]
        print(f"family {key}: arms={len(arms)} exits={len(pols)}", flush=True)
        # precompute path windows once per arm (max hold 120)
        prepared = []
        for a in arms:
            win = _path_window(
                a["path"][0],
                a["path"][1],
                a["entry_ts"],
                max_hold_sec=120,
                slip=float(args.slip),
            )
            if win is None:
                continue
            prepared.append({**a, "rets": win[0], "holds": win[1]})
        for pol in pols:
            raw = []
            for p in prepared:
                sim = simulate_exit(
                    p["rets"],
                    p["holds"],
                    mode="tpsl",
                    params={"tp": pol["tp"], "sl": pol["sl"], "max_hold": pol["max_hold"]},
                )
                if not np.isfinite(sim.get("ret", np.nan)):
                    continue
                et = p["entry_ts"]
                raw.append(
                    {
                        "date": p["date"],
                        "symbol": p["symbol"],
                        "dir": p["dir"],
                        "calendar": p["calendar"],
                        "entry_ts": et,
                        "exit_ts": et + pd.Timedelta(seconds=float(sim["hold_sec"])),
                        "ret": float(sim["ret"]),
                        "exit_reason": str(sim["reason"]),
                        "hold_sec": float(sim["hold_sec"]),
                        "opt_vol_z": p["opt_vol_z"],
                        "mf100": p["mf100"],
                        "ret_60": p["ret_60"],
                    }
                )
            disc = [t for t in raw if t["calendar"] == "may_jul09"]
            blind = [t for t in raw if t["calendar"] == "jul10_23"]
            sized_d = _portfolio_day(
                sorted(disc, key=lambda x: (x["entry_ts"], x["symbol"])),
                position_frac=float(args.position_frac),
                max_concurrent=int(args.max_concurrent),
                cooldown_minutes=float(args.cooldown_minutes),
            )
            sized_b = _portfolio_day(
                sorted(blind, key=lambda x: (x["entry_ts"], x["symbol"])),
                position_frac=float(args.position_frac),
                max_concurrent=int(args.max_concurrent),
                cooldown_minutes=float(args.cooldown_minutes),
            )
            st_d = _equity_stats(pd.DataFrame(sized_d))
            st_b = _equity_stats(pd.DataFrame(sized_b))
            months = _month_compounds(pd.DataFrame(sized_d + sized_b))
            row: dict[str, Any] = {
                "policy": pol["name"],
                "flow_win": pol["flow_win"],
                "min_z": pol["min_z"],
                "min_v": pol["min_v"],
                "dir_mode": pol["dir_mode"],
                "tp": pol["tp"],
                "sl": pol["sl"],
                "max_hold": pol["max_hold"],
                "n_raw": len(raw),
                "frac_up": float(np.mean([t["dir"] == "UP" for t in raw])) if raw else 0.0,
                "may": months.get("2026-05"),
                "jun": months.get("2026-06"),
                "jul": months.get("2026-07"),
            }
            for k, v in st_d.items():
                row[f"disc_{k}"] = v
            for k, v in st_b.items():
                row[f"blind_{k}"] = v
            score_rows.append(row)

    sb = pd.DataFrame(score_rows)
    sb.to_csv(out / "scoreboard.csv", index=False)

    soft = sb[
        (sb["disc_n"].fillna(0) >= 20)
        & (sb["disc_trade_win"].fillna(0) >= 0.55)
        & (sb["disc_maxdd"].fillna(-1) >= -0.25)
        & (sb["disc_compound"].fillna(0) > 0)
        & (sb["may"].fillna(0) > 0)
        & (sb["blind_n"].fillna(0) >= 5)
        & (sb["blind_compound"].fillna(0) > 0)
    ].copy()
    if not soft.empty:
        soft["score"] = (
            soft["disc_trade_win"] * 0.2
            + (1 + soft["disc_maxdd"]) * 0.2
            + np.clip(soft["disc_compound"], 0, 2) / 2 * 0.3
            + np.clip(soft["blind_compound"], 0, 1) * 0.3
        )
        soft = soft.sort_values("score", ascending=False)

    # also require both months positive-ish
    robust = soft[(soft["jun"].fillna(0) > -0.05) & (soft["jul"].fillna(0) > -0.05)] if len(soft) else soft

    verdict = {
        "protocol": "option_activity_then_stock_mf_direction_short_hold",
        "thesis": (
            "Detect abnormal option print activity, then time call/put via "
            "causal stock MF/ret/vol sliding windows — not mark-path flip."
        ),
        "n_policies": len(policies),
        "top_soft": soft.head(15).to_dict(orient="records") if len(soft) else [],
        "top_robust": robust.head(10).to_dict(orient="records") if len(robust) else [],
    }
    (out / "summary.json").write_text(json.dumps(verdict, indent=2, default=str), encoding="utf-8")

    cols = [
        c
        for c in [
            "policy",
            "dir_mode",
            "n_raw",
            "disc_n",
            "disc_trade_win",
            "disc_maxdd",
            "disc_compound",
            "blind_n",
            "blind_trade_win",
            "blind_compound",
            "frac_up",
            "may",
            "jun",
            "jul",
        ]
        if c in sb.columns
    ]
    print("\nTOP soft (disc+blind>0)", flush=True)
    print(soft[cols].head(12).to_string(index=False) if len(soft) else "(none)", flush=True)
    print("\nTOP robust", flush=True)
    print(robust[cols].head(8).to_string(index=False) if len(robust) else "(none)", flush=True)
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

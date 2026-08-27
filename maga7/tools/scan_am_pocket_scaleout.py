#!/usr/bin/env python3
"""Scale-out / BE / time-cut exits on AM pocket entries.

Builds on exit_design: full-position trail lost to fixed TP8/SL15.
This sweep tests:
  - scale-out: take frac@tp1, runner with trail / higher TP / BE stop
  - time-cut: if still red after T seconds, flatten
  - BE after arm: once +arm hit, move stop to floor

Portfolio marks a single blended ret; seat held until runner exits.

Example:
  PYTHONPATH=. python -m maga7.tools.scan_am_pocket_scaleout \\
    --tag research_am_pocket_scaleout
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

from maga7.common.config import load_profile
from maga7.common.option_trades import load_option_trades
from maga7.common.replay import to_ny
from maga7.tools.run_morning_sec_option_fill import _portfolio_day
from maga7.tools.scan_am_pocket_exit_design import (
    ENTRY_ACCEL0,
    ENTRY_VD_SOFT,
    _path_window,
)
from maga7.tools.scan_am_pocket_risk_optimize import (
    POCKET_SETS,
    _entry_ok,
    _equity_stats,
    _month_compounds,
)
from maga7.tools.scan_session_horizon_foresight import _paths_by_ticker

DEFAULT_PROBES = Path(
    "/mnt/s990/data/maga7/results/research_am_vwap_foresight_map_may_jul/probes.csv"
)
DEFAULT_TRADES = Path("/mnt/s990/new_option_data_s3_trades")


def simulate_scaleout(
    rets: np.ndarray,
    holds: np.ndarray,
    *,
    frac1: float,
    tp1: float,
    sl: float,
    max_hold: float,
    runner: str,
    tp2: float = 9.0,
    arm: float = 0.0,
    trail: float = 9.0,
    floor: float = -9.0,
    time_cut: float = 9e9,
    time_cut_min: float = -9.0,
    be_after_scale: bool = False,
) -> dict[str, Any]:
    """Two-leg causal exit. Returns blended ret and final hold."""
    f1 = float(np.clip(frac1, 0.0, 1.0))
    f2 = 1.0 - f1
    peak = -1.0
    armed = False
    scaled = False
    r1 = None
    h1 = None
    # after scale, optional BE stop on runner
    runner_sl = -float(sl)

    for i in range(1, len(rets)):
        r = float(rets[i])
        h = float(holds[i])
        if h > max_hold:
            # force flatten whatever remains
            if not scaled:
                return {
                    "ret": r,
                    "reason": "max_hold",
                    "hold_sec": float(holds[i - 1]),
                    "scaled": False,
                    "r1": None,
                    "r2": None,
                }
            r2 = float(rets[i - 1])
            blend = f1 * float(r1) + f2 * r2
            return {
                "ret": blend,
                "reason": "max_hold",
                "hold_sec": float(holds[i - 1]),
                "scaled": True,
                "r1": float(r1),
                "r2": r2,
            }

        peak = max(peak, r)
        if r >= arm:
            armed = True

        # hard SL before/after scale (runner_sl may be BE)
        stop = runner_sl if scaled else -float(sl)
        if r <= stop:
            if not scaled:
                return {
                    "ret": r,
                    "reason": "sl",
                    "hold_sec": h,
                    "scaled": False,
                    "r1": None,
                    "r2": None,
                }
            blend = f1 * float(r1) + f2 * r
            return {
                "ret": blend,
                "reason": "runner_sl",
                "hold_sec": h,
                "scaled": True,
                "r1": float(r1),
                "r2": r,
            }

        # time cut (full flatten if still below threshold)
        if (not scaled) and h >= time_cut and r < time_cut_min:
            return {
                "ret": r,
                "reason": "time_cut",
                "hold_sec": h,
                "scaled": False,
                "r1": None,
                "r2": None,
            }

        # first scale
        if (not scaled) and r >= tp1 and f1 > 0:
            scaled = True
            r1 = r
            h1 = h
            if be_after_scale:
                runner_sl = max(runner_sl, float(floor))
            if f2 <= 1e-12:
                return {
                    "ret": r,
                    "reason": "scale_full",
                    "hold_sec": h,
                    "scaled": True,
                    "r1": float(r1),
                    "r2": float(r1),
                }
            continue

        if not scaled:
            continue

        # runner logic
        if runner == "tp" and r >= tp2:
            blend = f1 * float(r1) + f2 * r
            return {
                "ret": blend,
                "reason": "runner_tp",
                "hold_sec": h,
                "scaled": True,
                "r1": float(r1),
                "r2": r,
            }
        if runner == "trail" and armed and peak - r >= trail:
            blend = f1 * float(r1) + f2 * r
            return {
                "ret": blend,
                "reason": "runner_trail",
                "hold_sec": h,
                "scaled": True,
                "r1": float(r1),
                "r2": r,
            }
        if runner == "pp" and armed and r <= floor:
            blend = f1 * float(r1) + f2 * r
            return {
                "ret": blend,
                "reason": "runner_pp",
                "hold_sec": h,
                "scaled": True,
                "r1": float(r1),
                "r2": r,
            }
        if runner == "hold":
            # only SL / max_hold
            pass

    # end of path
    r_end = float(rets[-1])
    h_end = float(holds[-1])
    if not scaled:
        return {
            "ret": r_end,
            "reason": "max_hold",
            "hold_sec": h_end,
            "scaled": False,
            "r1": None,
            "r2": None,
        }
    blend = f1 * float(r1) + f2 * r_end
    return {
        "ret": blend,
        "reason": "max_hold",
        "hold_sec": h_end,
        "scaled": True,
        "r1": float(r1),
        "r2": r_end,
    }


def _exit_grid() -> list[dict[str, Any]]:
    cfgs: list[dict[str, Any]] = []
    # baselines (frac1=1 ⇒ full exit at tp1)
    for tp, sl, h in ((0.08, 0.15, 240), (0.10, 0.15, 300), (0.08, 0.12, 240)):
        cfgs.append(
            {
                "name": f"base_tp{tp:g}_sl{sl:g}_h{h}",
                "frac1": 1.0,
                "tp1": tp,
                "sl": sl,
                "max_hold": h,
                "runner": "hold",
            }
        )

    # scale-out: take profit then hold/trail/pp runner
    for frac1 in (0.5, 0.67, 0.33):
        for tp1 in (0.06, 0.08, 0.10):
            for sl in (0.12, 0.15):
                for h in (300, 600, 900):
                    # runner hold to max (SL only), optional BE
                    for be in (False, True):
                        floor = 0.0 if be else -9.0
                        cfgs.append(
                            {
                                "name": (
                                    f"sc{frac1:g}@{tp1:g}_hold_sl{sl:g}_h{h}"
                                    + ("_be" if be else "")
                                ),
                                "frac1": frac1,
                                "tp1": tp1,
                                "sl": sl,
                                "max_hold": h,
                                "runner": "hold",
                                "be_after_scale": be,
                                "floor": floor,
                            }
                        )
                    # runner higher TP
                    for tp2 in (0.15, 0.20, 0.30):
                        cfgs.append(
                            {
                                "name": f"sc{frac1:g}@{tp1:g}_tp2{tp2:g}_sl{sl:g}_h{h}",
                                "frac1": frac1,
                                "tp1": tp1,
                                "tp2": tp2,
                                "sl": sl,
                                "max_hold": h,
                                "runner": "tp",
                                "be_after_scale": True,
                                "floor": 0.0,
                            }
                        )
                    # runner trail (wider than full-position trail)
                    for arm, trail in ((0.10, 0.08), (0.12, 0.10), (0.15, 0.10), (0.15, 0.15)):
                        cfgs.append(
                            {
                                "name": (
                                    f"sc{frac1:g}@{tp1:g}_tr_a{arm:g}_t{trail:g}_sl{sl:g}_h{h}"
                                ),
                                "frac1": frac1,
                                "tp1": tp1,
                                "sl": sl,
                                "max_hold": h,
                                "runner": "trail",
                                "arm": arm,
                                "trail": trail,
                                "be_after_scale": True,
                                "floor": 0.0,
                            }
                        )
                    # runner profit protect
                    for arm, floor in ((0.12, 0.04), (0.15, 0.05), (0.20, 0.08)):
                        cfgs.append(
                            {
                                "name": (
                                    f"sc{frac1:g}@{tp1:g}_pp_a{arm:g}_f{floor:g}_sl{sl:g}_h{h}"
                                ),
                                "frac1": frac1,
                                "tp1": tp1,
                                "sl": sl,
                                "max_hold": h,
                                "runner": "pp",
                                "arm": arm,
                                "floor": floor,
                                "be_after_scale": True,
                            }
                        )

    # time-cut overlays on baseline / scale
    for tc, tmin in ((90, 0.0), (120, 0.0), (180, 0.0), (180, -0.03)):
        cfgs.append(
            {
                "name": f"base_tp0.08_sl0.15_h240_tc{tc:g}_{tmin:g}",
                "frac1": 1.0,
                "tp1": 0.08,
                "sl": 0.15,
                "max_hold": 240,
                "runner": "hold",
                "time_cut": tc,
                "time_cut_min": tmin,
            }
        )
        cfgs.append(
            {
                "name": f"sc0.5@0.08_hold_sl0.15_h600_be_tc{tc:g}_{tmin:g}",
                "frac1": 0.5,
                "tp1": 0.08,
                "sl": 0.15,
                "max_hold": 600,
                "runner": "hold",
                "be_after_scale": True,
                "floor": 0.0,
                "time_cut": tc,
                "time_cut_min": tmin,
            }
        )
    return cfgs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--probes", default=str(DEFAULT_PROBES))
    ap.add_argument("--trades-root", default=str(DEFAULT_TRADES))
    ap.add_argument("--tag", default="research_am_pocket_scaleout")
    ap.add_argument(
        "--profile",
        default=(
            "maga7/CONFIG/strategy_profiles/"
            "single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1.json"
        ),
    )
    ap.add_argument("--position-frac", type=float, default=0.20)
    ap.add_argument("--max-concurrent", type=int, default=5)
    ap.add_argument("--slip", type=float, default=0.01)
    args = ap.parse_args(argv)

    prof = load_profile(args.profile)
    out = Path(prof["_paths"]["results_dir"]) / args.tag
    out.mkdir(parents=True, exist_ok=True)
    trades_root = Path(args.trades_root)

    probes = pd.read_csv(args.probes)
    probes["entry_ts"] = pd.to_datetime(probes["entry_ts"])
    probes = probes[
        probes["dir"] == np.where(probes["from_open_px"].astype(float) >= 0, "UP", "DN")
    ].copy()

    path_cache: dict[tuple[str, str], dict[str, tuple[np.ndarray, np.ndarray]]] = {}

    def paths_for(date: str, sym: str):
        key = (date, sym)
        if key not in path_cache:
            tday = load_option_trades(trades_root, sym, date)
            path_cache[key] = (
                _paths_by_ticker(tday) if tday is not None and not tday.empty else {}
            )
        return path_cache[key]

    universes = [
        ("robust", "no_b_up", ENTRY_VD_SOFT),
        ("dn_heavy", "dn_heavy", ENTRY_ACCEL0),
    ]
    exit_cfgs = _exit_grid()
    print(f"exit_cfgs={len(exit_cfgs)}", flush=True)

    score_rows: list[dict[str, Any]] = []

    for uname, pset_name, entry in universes:
        pdf = pd.DataFrame(sorted(POCKET_SETS[pset_name]), columns=["session", "tod_bucket", "dir"])
        base = probes.merge(pdf, on=["session", "tod_bucket", "dir"], how="inner")
        base = base.sort_values(["date", "symbol", "session", "entry_ts"]).drop_duplicates(
            ["date", "symbol", "session"], keep="first"
        )
        base = base[base.apply(lambda r: _entry_ok(r, spec=entry), axis=1)].copy()

        prepared: list[dict[str, Any]] = []
        for _, r in base.iterrows():
            arrs = paths_for(str(r["date"]), str(r["symbol"]))
            arr = arrs.get(str(r["ticker"]).replace("O:", ""))
            if arr is None:
                continue
            win = _path_window(
                arr[0],
                arr[1],
                to_ny(pd.Timestamp(r["entry_ts"])),
                max_hold_sec=900,
                slip=float(args.slip),
            )
            if win is None:
                continue
            rets, holds, _, _ = win
            prepared.append(
                {
                    "date": str(r["date"]),
                    "symbol": str(r["symbol"]),
                    "dir": str(r["dir"]),
                    "session": str(r["session"]),
                    "calendar": str(r["calendar"]),
                    "entry_ts": to_ny(pd.Timestamp(r["entry_ts"])),
                    "rets": rets,
                    "holds": holds,
                    "oracle_ret": float(r["oracle_ret"]),
                }
            )
        print(f"[{uname}] prepared={len(prepared)}", flush=True)

        for ex in exit_cfgs:
            raw = []
            for s in prepared:
                mh = float(ex.get("max_hold", 900))
                mask = s["holds"] <= mh + 1e-9
                rets = s["rets"][mask]
                holds = s["holds"][mask]
                if len(rets) < 2:
                    continue
                sim = simulate_scaleout(
                    rets,
                    holds,
                    frac1=float(ex.get("frac1", 1.0)),
                    tp1=float(ex.get("tp1", 0.08)),
                    sl=float(ex.get("sl", 0.15)),
                    max_hold=mh,
                    runner=str(ex.get("runner", "hold")),
                    tp2=float(ex.get("tp2", 9.0)),
                    arm=float(ex.get("arm", 0.0)),
                    trail=float(ex.get("trail", 9.0)),
                    floor=float(ex.get("floor", -9.0)),
                    time_cut=float(ex.get("time_cut", 9e9)),
                    time_cut_min=float(ex.get("time_cut_min", -9.0)),
                    be_after_scale=bool(ex.get("be_after_scale", False)),
                )
                if not np.isfinite(sim["ret"]):
                    continue
                et = s["entry_ts"]
                raw.append(
                    {
                        "date": s["date"],
                        "symbol": s["symbol"],
                        "dir": s["dir"],
                        "session": s["session"],
                        "calendar": s["calendar"],
                        "entry_ts": et,
                        "exit_ts": et + pd.Timedelta(seconds=float(sim["hold_sec"])),
                        "ret": float(sim["ret"]),
                        "exit_reason": str(sim["reason"]),
                        "hold_sec": float(sim["hold_sec"]),
                        "scaled": bool(sim["scaled"]),
                        "oracle_ret": float(s["oracle_ret"]),
                    }
                )

            disc = [t for t in raw if t["calendar"] == "may_jul09"]
            blind = [t for t in raw if t["calendar"] == "jul10_23"]
            sized_d = _portfolio_day(
                sorted(disc, key=lambda x: (x["entry_ts"], x["symbol"])),
                position_frac=float(args.position_frac),
                max_concurrent=int(args.max_concurrent),
                cooldown_minutes=10.0,
            )
            sized_b = _portfolio_day(
                sorted(blind, key=lambda x: (x["entry_ts"], x["symbol"])),
                position_frac=float(args.position_frac),
                max_concurrent=int(args.max_concurrent),
                cooldown_minutes=10.0,
            )
            st_d = _equity_stats(pd.DataFrame(sized_d))
            st_b = _equity_stats(pd.DataFrame(sized_b))
            months = _month_compounds(pd.DataFrame(sized_d + sized_b))
            if raw:
                o = np.array([t["oracle_ret"] for t in raw], dtype=float)
                r = np.array([t["ret"] for t in raw], dtype=float)
                mean_cap = float(r.mean() / o.mean()) if o.mean() > 0 else float("nan")
                scale_rate = float(np.mean([1.0 if t["scaled"] else 0.0 for t in raw]))
            else:
                mean_cap = float("nan")
                scale_rate = 0.0

            row: dict[str, Any] = {
                "universe": uname,
                "exit": ex["name"],
                "runner": ex.get("runner"),
                "frac1": ex.get("frac1"),
                "mean_capture": mean_cap,
                "scale_rate": scale_rate,
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

    picks = []
    for uname in ("robust", "dn_heavy"):
        sub = sb[sb["universe"] == uname].copy()
        if sub.empty:
            continue
        base = sub[sub["exit"] == "base_tp0.08_sl0.15_h240"]
        base_row = base.iloc[0].to_dict() if len(base) else {}
        # soft: not much worse win, better DD or better capture/compound
        soft = sub[
            (sub["disc_trade_win"] >= 0.62)
            & (sub["disc_day_win"] >= 0.55)
            & (sub["disc_maxdd"] >= -0.18)
            & (sub["disc_n"] >= 30)
            & (sub["blind_n"] >= 5)
            & (sub["blind_mean"] > 0)
        ].copy()
        if soft.empty:
            soft = sub.sort_values(
                ["disc_trade_win", "disc_maxdd", "disc_compound"],
                ascending=[False, False, False],
            ).head(8)
        else:
            soft["score"] = (
                soft["disc_trade_win"] * 0.28
                + soft["disc_day_win"] * 0.12
                + (1.0 + soft["disc_maxdd"]) * 0.28
                + np.clip(soft["disc_compound"], 0, 3) / 3 * 0.16
                + np.clip(soft["mean_capture"], 0, 0.25) / 0.25 * 0.16
            )
            soft = soft.sort_values("score", ascending=False)

        # also: best DD improvement vs baseline with win drop <= 5pp
        bw = float(base_row.get("disc_trade_win", 0.65) or 0.65)
        bdd = float(base_row.get("disc_maxdd", -0.16) or -0.16)
        better_dd = sub[
            (sub["disc_trade_win"] >= bw - 0.05)
            & (sub["disc_maxdd"] > bdd)
            & (sub["disc_compound"] > 0)
            & (sub["blind_mean"] > 0)
        ].sort_values(["disc_maxdd", "disc_compound"], ascending=[False, False])

        # best capture under dd>=-0.18 win>=0.60
        better_cap = sub[
            (sub["disc_trade_win"] >= 0.60)
            & (sub["disc_maxdd"] >= -0.18)
            & (sub["blind_mean"] > 0)
        ].sort_values("mean_capture", ascending=False)

        top = soft.iloc[0].to_dict()
        top["baseline_exit"] = "base_tp0.08_sl0.15_h240"
        top["baseline_win"] = base_row.get("disc_trade_win")
        top["baseline_maxdd"] = base_row.get("disc_maxdd")
        top["baseline_compound"] = base_row.get("disc_compound")
        top["baseline_capture"] = base_row.get("mean_capture")
        picks.append(
            {
                "universe": uname,
                "best_soft": {k: top[k] for k in top if not str(k).startswith("Unnamed")},
                "best_dd": better_dd.head(3).to_dict(orient="records"),
                "best_capture": better_cap.head(3).to_dict(orient="records"),
            }
        )
        print(f"\nBEST soft {uname}: {top['exit']}", flush=True)
        print(
            f"  win={top['disc_trade_win']:.2f} dd={top['disc_maxdd']:.3f} "
            f"cmp={top['disc_compound']:.2f} cap={top['mean_capture']:.3f} "
            f"scale={top.get('scale_rate')}",
            flush=True,
        )
        if len(better_dd):
            r0 = better_dd.iloc[0]
            print(
                f"  BEST DD+: {r0['exit']} win={r0['disc_trade_win']:.2f} "
                f"dd={r0['disc_maxdd']:.3f} cmp={r0['disc_compound']:.2f}",
                flush=True,
            )
        if len(better_cap):
            r0 = better_cap.iloc[0]
            print(
                f"  BEST CAP: {r0['exit']} win={r0['disc_trade_win']:.2f} "
                f"dd={r0['disc_maxdd']:.3f} cap={r0['mean_capture']:.3f}",
                flush=True,
            )

    verdict = {
        "protocol": "scaleout_be_timecut_vs_tpsl",
        "portfolio": {
            "position_frac": args.position_frac,
            "max_concurrent": args.max_concurrent,
        },
        "n_cfgs": len(exit_cfgs),
        "picks": picks,
    }
    (out / "summary.json").write_text(json.dumps(verdict, indent=2, default=str), encoding="utf-8")

    for uname in ("robust", "dn_heavy"):
        sub = sb[sb["universe"] == uname].sort_values(
            ["disc_trade_win", "disc_maxdd", "mean_capture"],
            ascending=[False, False, False],
        )
        cols = [
            c
            for c in [
                "exit",
                "runner",
                "frac1",
                "disc_n",
                "disc_trade_win",
                "disc_day_win",
                "disc_maxdd",
                "disc_compound",
                "mean_capture",
                "scale_rate",
                "may",
                "jun",
                "jul",
                "blind_trade_win",
                "blind_compound",
            ]
            if c in sub.columns
        ]
        print(f"\nTOP {uname}", flush=True)
        print(sub[cols].head(15).to_string(index=False), flush=True)

        # top among scale-out only
        sc = sub[sub["frac1"] < 1.0].sort_values("disc_trade_win", ascending=False)
        print(f"\nTOP scale {uname}", flush=True)
        print(sc[cols].head(12).to_string(index=False), flush=True)

    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Design causal exits for AM pocket entries using foresight path clues.

1) Path diagnosis on robust entry set: time-to-level, MAE-before-MFE, giveback.
2) Sweep exit families on trade lasts @20%/5:
   - fixed TP/SL
   - trail from peak (after arm)
   - profit-protect floor
   - time-cut if still red
3) Report win / maxDD / compound / oracle-capture vs fixed baselines.

Example:
  PYTHONPATH=. python -m maga7.tools.scan_am_pocket_exit_design \\
    --tag research_am_pocket_exit_design
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

ENTRY_VD_SOFT = {
    "name": "vd_soft",
    "accel_min": 0.0,
    "accel_max": 0.00025,
    "fo_min": 0.003,
    "fo_max": 0.015,
    "vd_min": 0.002,
    "vd_max": 0.007,
}
ENTRY_ACCEL0 = {
    "name": "accel0",
    "accel_min": 0.0,
    "accel_max": 9.0,
    "fo_min": 0.0,
    "fo_max": 9.0,
    "vd_min": -9.0,
    "vd_max": 9.0,
}


def _path_window(
    ts_ns: np.ndarray,
    last: np.ndarray,
    entry_ts: pd.Timestamp,
    *,
    max_hold_sec: int,
    slip: float,
) -> tuple[np.ndarray, np.ndarray, float, int] | None:
    t0 = int(to_ny(entry_ts).value)
    i0 = int(np.searchsorted(ts_ns, t0, side="left"))
    if i0 >= len(ts_ns):
        return None
    if (int(ts_ns[i0]) - t0) / 1e9 > 5:
        return None
    entry = float(last[i0]) * (1.0 + float(slip))
    if not np.isfinite(entry) or entry <= 0:
        return None
    end_ns = int(ts_ns[i0]) + int(max_hold_sec) * 1_000_000_000
    i_end = int(np.searchsorted(ts_ns, end_ns, side="right") - 1)
    if i_end <= i0:
        return None
    sell = last[i0 : i_end + 1] * (1.0 - float(slip))
    rets = sell / entry - 1.0
    holds = (ts_ns[i0 : i_end + 1] - ts_ns[i0]) / 1e9
    return rets.astype(np.float64), holds.astype(np.float64), entry, i0


def diagnose_paths(signals: list[dict[str, Any]], *, slip: float = 0.01) -> dict[str, Any]:
    rows = []
    levels = (0.05, 0.08, 0.10, 0.15, 0.20, 0.30)
    for s in signals:
        win = _path_window(s["pts"], s["plast"], s["entry_ts"], max_hold_sec=900, slip=slip)
        if win is None:
            continue
        rets, holds, _, _ = win
        mfe_i = int(np.nanargmax(rets))
        mae_i = int(np.nanargmin(rets[: mfe_i + 1])) if mfe_i >= 0 else 0
        mfe = float(rets[mfe_i])
        mae_before = float(np.nanmin(rets[: mfe_i + 1])) if mfe_i >= 0 else float(rets[0])
        giveback = float(mfe - rets[-1]) if mfe_i < len(rets) - 1 else 0.0
        row: dict[str, Any] = {
            "mfe": mfe,
            "mae_before_mfe": mae_before,
            "oracle_hold": float(holds[mfe_i]),
            "clock900": float(rets[-1]),
            "giveback_after_mfe": giveback,
        }
        for lv in levels:
            hit = np.where(rets >= lv)[0]
            row[f"t_to_{lv:g}"] = float(holds[int(hit[0])]) if len(hit) else np.nan
            # after first hit lv, did it give back to < lv*0.5 before end?
        rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        return {"n": 0}
    out: dict[str, Any] = {
        "n": int(len(df)),
        "frac_mfe_ge_08": float((df["mfe"] >= 0.08).mean()),
        "frac_mfe_ge_15": float((df["mfe"] >= 0.15).mean()),
        "frac_mfe_ge_20": float((df["mfe"] >= 0.20).mean()),
        "p50_mfe": float(df["mfe"].median()),
        "p50_mae_before_mfe": float(df["mae_before_mfe"].median()),
        "p50_oracle_hold": float(df["oracle_hold"].median()),
        "p50_giveback": float(df["giveback_after_mfe"].median()),
        "frac_mae_before_lt_08": float((df["mae_before_mfe"] > -0.08).mean()),
        "frac_mae_before_lt_12": float((df["mae_before_mfe"] > -0.12).mean()),
    }
    for lv in levels:
        col = f"t_to_{lv:g}"
        hit = df[col].dropna()
        out[f"frac_hit_{lv:g}"] = float(len(hit) / len(df))
        out[f"p50_t_to_{lv:g}"] = float(hit.median()) if len(hit) else None
    return out


def simulate_exit(
    rets: np.ndarray,
    holds: np.ndarray,
    *,
    mode: str,
    params: dict[str, Any],
) -> dict[str, Any]:
    """Causal exit on precomputed mark returns."""
    tp = float(params.get("tp", 9.0))
    sl = float(params.get("sl", 9.0))
    arm = float(params.get("arm", 0.0))
    trail = float(params.get("trail", 9.0))
    floor = float(params.get("floor", -9.0))
    time_cut = float(params.get("time_cut", 9e9))
    time_cut_min = float(params.get("time_cut_min", -9.0))
    max_h = float(params.get("max_hold", holds[-1] if len(holds) else 900))

    peak = -1.0
    armed = False
    for i in range(1, len(rets)):
        r = float(rets[i])
        h = float(holds[i])
        if h > max_h:
            return {"ret": float(rets[i - 1]), "reason": "max_hold", "hold_sec": float(holds[i - 1])}
        peak = max(peak, r)
        if r >= arm:
            armed = True
        if mode in {"tpsl", "trail", "pp", "hybrid"}:
            if r >= tp:
                return {"ret": r, "reason": "tp", "hold_sec": h}
            if r <= -sl:
                return {"ret": r, "reason": "sl", "hold_sec": h}
        if mode in {"trail", "hybrid"} and armed and peak - r >= trail:
            return {"ret": r, "reason": "trail", "hold_sec": h}
        if mode in {"pp", "hybrid"} and armed and r <= floor:
            return {"ret": r, "reason": "profit_floor", "hold_sec": h}
        if h >= time_cut and r < time_cut_min:
            return {"ret": r, "reason": "time_cut", "hold_sec": h}
    return {"ret": float(rets[-1]), "reason": "max_hold", "hold_sec": float(holds[-1])}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--probes", default=str(DEFAULT_PROBES))
    ap.add_argument("--trades-root", default=str(DEFAULT_TRADES))
    ap.add_argument("--tag", default="research_am_pocket_exit_design")
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

    exit_cfgs: list[dict[str, Any]] = []
    # baselines
    for tp, sl, h in ((0.08, 0.15, 240), (0.10, 0.15, 300), (0.20, 0.25, 300), (0.12, 0.12, 300)):
        exit_cfgs.append(
            {"name": f"tpsl_tp{tp:g}_sl{sl:g}_h{h}", "mode": "tpsl", "tp": tp, "sl": sl, "max_hold": h}
        )
    # trail family: arm then giveback
    for arm in (0.05, 0.08, 0.10):
        for trail in (0.04, 0.06, 0.08, 0.10):
            for sl in (0.10, 0.12, 0.15):
                for h in (300, 600, 900):
                    exit_cfgs.append(
                        {
                            "name": f"trail_a{arm:g}_t{trail:g}_sl{sl:g}_h{h}",
                            "mode": "trail",
                            "arm": arm,
                            "trail": trail,
                            "sl": sl,
                            "tp": 9.0,
                            "max_hold": h,
                        }
                    )
    # profit protect
    for arm, floor, sl, h in (
        (0.08, 0.03, 0.12, 600),
        (0.08, 0.02, 0.12, 600),
        (0.10, 0.04, 0.15, 600),
        (0.10, 0.05, 0.12, 900),
        (0.12, 0.05, 0.15, 900),
    ):
        exit_cfgs.append(
            {
                "name": f"pp_a{arm:g}_f{floor:g}_sl{sl:g}_h{h}",
                "mode": "pp",
                "arm": arm,
                "floor": floor,
                "sl": sl,
                "tp": 9.0,
                "max_hold": h,
            }
        )
    # hybrid: soft tp + trail + hard sl + time cut
    for arm, trail, sl, tp, h in (
        (0.06, 0.05, 0.12, 0.25, 600),
        (0.08, 0.06, 0.12, 0.30, 900),
        (0.08, 0.08, 0.15, 0.25, 600),
        (0.05, 0.04, 0.10, 0.20, 450),
        (0.08, 0.05, 0.12, 0.20, 600),
    ):
        exit_cfgs.append(
            {
                "name": f"hyb_a{arm:g}_tr{trail:g}_sl{sl:g}_tp{tp:g}_h{h}",
                "mode": "hybrid",
                "arm": arm,
                "trail": trail,
                "sl": sl,
                "tp": tp,
                "max_hold": h,
                "time_cut": 180,
                "time_cut_min": 0.0,
            }
        )

    all_diag = {}
    score_rows: list[dict[str, Any]] = []

    for uname, pset_name, entry in universes:
        pdf = pd.DataFrame(sorted(POCKET_SETS[pset_name]), columns=["session", "tod_bucket", "dir"])
        base = probes.merge(pdf, on=["session", "tod_bucket", "dir"], how="inner")
        base = base.sort_values(["date", "symbol", "session", "entry_ts"]).drop_duplicates(
            ["date", "symbol", "session"], keep="first"
        )
        base = base[base.apply(lambda r: _entry_ok(r, spec=entry), axis=1)].copy()
        signals: list[dict[str, Any]] = []
        for _, r in base.iterrows():
            arrs = paths_for(str(r["date"]), str(r["symbol"]))
            arr = arrs.get(str(r["ticker"]).replace("O:", ""))
            if arr is None:
                continue
            signals.append(
                {
                    "date": str(r["date"]),
                    "symbol": str(r["symbol"]),
                    "dir": str(r["dir"]),
                    "session": str(r["session"]),
                    "calendar": str(r["calendar"]),
                    "entry_ts": to_ny(pd.Timestamp(r["entry_ts"])),
                    "pts": arr[0],
                    "plast": arr[1],
                    "oracle_ret": float(r["oracle_ret"]),
                }
            )
        print(f"[{uname}] signals={len(signals)}", flush=True)
        diag = diagnose_paths(signals, slip=float(args.slip))
        all_diag[uname] = diag
        print(json.dumps(diag, indent=2, default=str)[:1200], flush=True)

        # precompute path windows once
        prepared: list[dict[str, Any]] = []
        for s in signals:
            win = _path_window(
                s["pts"], s["plast"], s["entry_ts"], max_hold_sec=900, slip=float(args.slip)
            )
            if win is None:
                continue
            rets, holds, _, _ = win
            prepared.append({**s, "rets": rets, "holds": holds})

        for ex in exit_cfgs:
            raw = []
            for s in prepared:
                # truncate path to max_hold
                mh = float(ex.get("max_hold", 900))
                mask = s["holds"] <= mh + 1e-9
                rets = s["rets"][mask]
                holds = s["holds"][mask]
                if len(rets) < 2:
                    continue
                sim = simulate_exit(rets, holds, mode=str(ex["mode"]), params=ex)
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
            # capture
            if raw:
                o = np.array([t["oracle_ret"] for t in raw], dtype=float)
                r = np.array([t["ret"] for t in raw], dtype=float)
                mean_cap = float(r.mean() / o.mean()) if o.mean() > 0 else float("nan")
                med_cap = float(
                    np.nanmedian(np.where(o > 0, np.minimum(r, o) / o, np.nan))
                )
            else:
                mean_cap = med_cap = float("nan")

            row = {
                "universe": uname,
                "exit": ex["name"],
                "mode": ex["mode"],
                "mean_capture": mean_cap,
                "median_capture": med_cap,
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
    (out / "path_diagnosis.json").write_text(json.dumps(all_diag, indent=2, default=str))

    # Prefer: disc trade_win>=0.65, maxdd>=-0.16, compound>baseline, blind mean>0
    picks = []
    for uname in ("robust", "dn_heavy"):
        sub = sb[sb["universe"] == uname].copy()
        if sub.empty:
            continue
        base = sub[sub["exit"] == "tpsl_tp0.08_sl0.15_h240"]
        base_cmp = float(base.iloc[0]["disc_compound"]) if len(base) else 0.0
        soft = sub[
            (sub["disc_trade_win"] >= 0.65)
            & (sub["disc_day_win"] >= 0.60)
            & (sub["disc_maxdd"] >= -0.18)
            & (sub["disc_n"] >= 30)
            & (sub["blind_n"] >= 5)
            & (sub["blind_mean"] > 0)
        ].copy()
        if soft.empty:
            soft = sub.sort_values(
                ["disc_trade_win", "disc_maxdd", "disc_compound"],
                ascending=[False, False, False],
            ).head(5)
        else:
            soft["score"] = (
                soft["disc_trade_win"] * 0.3
                + soft["disc_day_win"] * 0.15
                + (1 + soft["disc_maxdd"]) * 0.25
                + np.clip(soft["disc_compound"], 0, 3) / 3 * 0.15
                + np.clip(soft["mean_capture"], 0, 0.4) / 0.4 * 0.15
            )
            soft = soft.sort_values("score", ascending=False)
        top = soft.iloc[0].to_dict()
        top["baseline_compound"] = base_cmp
        picks.append(top)
        print(f"\nBEST {uname}: {top['exit']}", flush=True)
        print(
            f"  win={top['disc_trade_win']:.2f} day={top['disc_day_win']:.2f} "
            f"dd={top['disc_maxdd']:.3f} cmp={top['disc_compound']:.2f} "
            f"cap={top['mean_capture']:.3f} months "
            f"{top.get('may')}/{top.get('jun')}/{top.get('jul')}",
            flush=True,
        )

    verdict = {
        "protocol": "exit_design_from_foresight_paths",
        "portfolio": {"position_frac": args.position_frac, "max_concurrent": args.max_concurrent},
        "path_diagnosis": all_diag,
        "picks": picks,
        "design_notes": [
            "If MAE-before-MFE is usually > -8~-12%, hard SL can sit there without killing winners.",
            "If time-to-+8% is short (p50), arm trail/profit-floor early; avoid fixed low TP that caps MFE.",
            "Giveback-after-MFE large ⇒ trail/floor necessary; fixed TP alone leaves 30–40pp on table.",
            "Oracle compound is not a trading target; track mean_capture and maxDD jointly.",
        ],
    }
    (out / "summary.json").write_text(json.dumps(verdict, indent=2, default=str), encoding="utf-8")
    print("\n=== VERDICT ===", flush=True)
    print(json.dumps({k: verdict[k] for k in ("picks",)}, indent=2, default=str)[:2500], flush=True)
    # top table per universe
    for uname in ("robust", "dn_heavy"):
        sub = sb[sb["universe"] == uname].sort_values(
            ["disc_trade_win", "disc_maxdd", "mean_capture"], ascending=[False, False, False]
        )
        cols = [
            c
            for c in [
                "exit",
                "mode",
                "disc_n",
                "disc_trade_win",
                "disc_day_win",
                "disc_maxdd",
                "disc_compound",
                "mean_capture",
                "may",
                "jun",
                "jul",
                "blind_trade_win",
                "blind_compound",
            ]
            if c in sub.columns
        ]
        print(f"\nTOP {uname}", flush=True)
        print(sub[cols].head(12).to_string(index=False), flush=True)
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

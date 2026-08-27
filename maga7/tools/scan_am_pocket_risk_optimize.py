#!/usr/bin/env python3
"""Risk-optimize AM pocket sleeve: raise win-rate, cut DD, keep monthly compound.

Starts from foresight pocket probes + accel structure, then sweeps:
  - pocket subsets (drop weak TOD / B-UP)
  - entry clamps (mild accel, fo band, vwap_diff cap)
  - exits (tighter TP/SL, shorter hold, confirm-abort, profit-protect)

Scores under portfolio 20% / max 5 concurrent; select on discover, report blind.

Example:
  PYTHONPATH=. python -m maga7.tools.scan_am_pocket_risk_optimize \\
    --tag research_am_pocket_risk_opt
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
from maga7.common.option_trade_tpsl import (
    simulate_trade_tpsl,
    simulate_trade_tpsl_confirm_abort,
)
from maga7.common.option_trades import load_option_trades
from maga7.common.replay import to_ny
from maga7.tools.run_morning_sec_option_fill import _portfolio_day
from maga7.tools.scan_session_horizon_foresight import _paths_by_ticker

DEFAULT_PROBES = Path(
    "/mnt/s990/data/maga7/results/research_am_pocket_exit_distill/pocket_probes.csv"
)
DEFAULT_TRADES = Path("/mnt/s990/new_option_data_s3_trades")

# Diagnosis: drop chronically weak B-UP pockets.
POCKET_SETS: dict[str, set[tuple[str, str, str]]] = {
    "all": {
        ("AM_A_0930_1030", "09:30", "UP"),
        ("AM_A_0930_1030", "09:30", "DN"),
        ("AM_A_0930_1030", "09:35", "UP"),
        ("AM_B_1030_1130", "10:30", "DN"),
        ("AM_B_1030_1130", "10:35", "DN"),
        ("AM_B_1030_1130", "10:40", "UP"),
        ("AM_B_1030_1130", "10:45", "UP"),
        ("AM_B_1030_1130", "10:50", "UP"),
        ("AM_B_1030_1130", "10:55", "UP"),
    },
    "no_b_up": {
        ("AM_A_0930_1030", "09:30", "UP"),
        ("AM_A_0930_1030", "09:30", "DN"),
        ("AM_A_0930_1030", "09:35", "UP"),
        ("AM_B_1030_1130", "10:30", "DN"),
        ("AM_B_1030_1130", "10:35", "DN"),
    },
    "a_only": {
        ("AM_A_0930_1030", "09:30", "UP"),
        ("AM_A_0930_1030", "09:30", "DN"),
        ("AM_A_0930_1030", "09:35", "UP"),
    },
    "dn_heavy": {
        ("AM_A_0930_1030", "09:30", "DN"),
        ("AM_B_1030_1130", "10:30", "DN"),
        ("AM_B_1030_1130", "10:35", "DN"),
    },
}


def _signed(row: pd.Series, feat: str) -> float:
    v = float(row.get(feat) or np.nan)
    if not np.isfinite(v):
        return float("nan")
    return v if str(row["dir"]) == "UP" else -v


def _entry_ok(row: pd.Series, *, spec: dict[str, Any]) -> bool:
    accel = float(row.get("accel_10_30") or np.nan)
    if not np.isfinite(accel) or accel + 1e-12 < float(spec["accel_min"]):
        return False
    if accel - 1e-12 > float(spec["accel_max"]):
        return False
    fo = _signed(row, "fo_vwap30")
    if not np.isfinite(fo):
        return False
    if fo + 1e-12 < float(spec["fo_min"]) or fo - 1e-12 > float(spec["fo_max"]):
        return False
    vd = _signed(row, "vwap_diff")
    if not np.isfinite(vd):
        return False
    if vd + 1e-12 < float(spec["vd_min"]) or vd - 1e-12 > float(spec["vd_max"]):
        return False
    return True


def simulate_trade_profit_protect(
    ts_ns: np.ndarray,
    last: np.ndarray,
    entry_ts: pd.Timestamp,
    *,
    tp: float,
    sl: float,
    max_hold_sec: int,
    arm_ret: float,
    floor_ret: float,
    slip: float = 0.01,
) -> dict[str, Any] | None:
    """TP/SL with one-shot profit floor after MTM reaches arm_ret."""
    t0 = int(to_ny(entry_ts).value)
    i0 = int(np.searchsorted(ts_ns, t0, side="left"))
    if i0 >= len(ts_ns):
        return None
    lag = (int(ts_ns[i0]) - t0) / 1e9
    if lag > 5:
        return None
    entry = float(last[i0]) * (1.0 + float(slip))
    if not np.isfinite(entry) or entry <= 0:
        return None
    sell_mult = 1.0 - float(slip)
    end_ns = int(ts_ns[i0]) + int(max_hold_sec) * 1_000_000_000
    i_end = int(np.searchsorted(ts_ns, end_ns, side="right") - 1)
    if i_end < i0:
        return None
    armed = False
    exit_i = i_end
    reason = "max_hold"
    mfe = -1.0
    mae = 1.0
    for k in range(i0 + 1, i_end + 1):
        px = float(last[k])
        if not np.isfinite(px) or px <= 0:
            continue
        ret = px * sell_mult / entry - 1.0
        mfe = max(mfe, ret)
        mae = min(mae, ret)
        if ret >= float(arm_ret):
            armed = True
        if armed and ret <= float(floor_ret):
            exit_i, reason = k, "profit_floor"
            break
        if ret >= float(tp):
            exit_i, reason = k, "tp"
            break
        if ret <= -float(sl):
            exit_i, reason = k, "sl"
            break
    px_x = float(last[exit_i])
    ret = px_x * sell_mult / entry - 1.0
    hold = (int(ts_ns[exit_i]) - int(ts_ns[i0])) / 1e9
    return {
        "ret": float(ret),
        "reason": reason,
        "hold_sec": float(hold),
        "mfe": float(mfe if np.isfinite(mfe) else ret),
        "mae": float(mae if np.isfinite(mae) else ret),
    }


def _equity_stats(tr: pd.DataFrame) -> dict[str, Any]:
    if tr is None or tr.empty:
        return {
            "n": 0,
            "trade_win": None,
            "mean": None,
            "day_win": None,
            "n_days": 0,
            "compound": None,
            "maxdd": None,
            "worst_day": None,
            "sum_pnl": 0.0,
        }
    t = tr.copy()
    t["pnl_frac"] = t["ret"].astype(float) * t["size"].astype(float)
    day = t.groupby("date")["pnl_frac"].sum().sort_index()
    eq = 1.0
    peak = 1.0
    maxdd = 0.0
    for v in day.values:
        eq *= 1.0 + float(v)
        peak = max(peak, eq)
        maxdd = min(maxdd, eq / peak - 1.0 if peak > 0 else 0.0)
    return {
        "n": int(len(t)),
        "trade_win": float((t["ret"] > 0).mean()),
        "mean": float(t["ret"].mean()),
        "day_win": float((day > 0).mean()),
        "n_days": int(day.shape[0]),
        "compound": float(eq - 1.0),
        "maxdd": float(maxdd),
        "worst_day": float(day.min()) if len(day) else None,
        "sum_pnl": float(t["pnl_frac"].sum()),
    }


def _month_compounds(tr: pd.DataFrame) -> dict[str, float]:
    if tr is None or tr.empty:
        return {}
    t = tr.copy()
    t["pnl_frac"] = t["ret"].astype(float) * t["size"].astype(float)
    t["month"] = pd.to_datetime(t["date"]).dt.to_period("M").astype(str)
    out: dict[str, float] = {}
    for m, sub in t.groupby("month"):
        day = sub.groupby("date")["pnl_frac"].sum().sort_index()
        eq = 1.0
        for v in day.values:
            eq *= 1.0 + float(v)
        out[str(m)] = float(eq - 1.0)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--probes", default=str(DEFAULT_PROBES))
    ap.add_argument("--trades-root", default=str(DEFAULT_TRADES))
    ap.add_argument("--tag", default="research_am_pocket_risk_opt")
    ap.add_argument("--profile", default=(
        "maga7/CONFIG/strategy_profiles/"
        "single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1.json"
    ))
    ap.add_argument("--position-frac", type=float, default=0.20)
    ap.add_argument("--max-concurrent", type=int, default=5)
    ap.add_argument("--cooldown-minutes", type=float, default=10.0)
    ap.add_argument("--slip", type=float, default=0.01)
    ap.add_argument("--min-trade-win", type=float, default=0.62)
    ap.add_argument("--min-day-win", type=float, default=0.60)
    ap.add_argument("--max-dd", type=float, default=-0.12, help="discover maxDD floor (e.g. -0.12)")
    args = ap.parse_args(argv)

    prof = load_profile(args.profile)
    out = Path(prof["_paths"]["results_dir"]) / args.tag
    out.mkdir(parents=True, exist_ok=True)
    trades_root = Path(args.trades_root)

    probes = pd.read_csv(args.probes)
    probes["entry_ts"] = pd.to_datetime(probes["entry_ts"])
    # align
    probes = probes[
        probes["dir"] == np.where(probes["from_open_px"].astype(float) >= 0, "UP", "DN")
    ].copy()

    # Entry specs (mild extension; avoid chase)
    entry_specs = [
        {"name": "accel0", "accel_min": 0.0, "accel_max": 9.0, "fo_min": 0.0, "fo_max": 9.0, "vd_min": -9.0, "vd_max": 9.0},
        {"name": "mild_accel", "accel_min": 0.0, "accel_max": 0.0002, "fo_min": 0.003, "fo_max": 0.012, "vd_min": 0.0, "vd_max": 0.008},
        {"name": "fo_band", "accel_min": 0.0, "accel_max": 0.0003, "fo_min": 0.004, "fo_max": 0.010, "vd_min": -0.002, "vd_max": 0.008},
        {"name": "vd_soft", "accel_min": 0.0, "accel_max": 0.00025, "fo_min": 0.003, "fo_max": 0.015, "vd_min": 0.002, "vd_max": 0.007},
        {"name": "tight", "accel_min": 0.0, "accel_max": 0.00015, "fo_min": 0.004, "fo_max": 0.010, "vd_min": 0.001, "vd_max": 0.006},
    ]

    exit_specs: list[dict[str, Any]] = []
    for tp in (0.08, 0.10, 0.12, 0.15):
        for sl in (0.08, 0.10, 0.12, 0.15):
            for h in (180, 240, 300):
                exit_specs.append(
                    {"name": f"tpsl_tp{tp:g}_sl{sl:g}_h{h}", "mode": "tpsl", "tp": tp, "sl": sl, "h": h}
                )
    for tp, sl, h in ((0.10, 0.10, 240), (0.10, 0.12, 300), (0.12, 0.12, 300), (0.15, 0.12, 300)):
        for ab in (0.06, 0.08):
            exit_specs.append(
                {
                    "name": f"ca_tp{tp:g}_sl{sl:g}_h{h}_ab{ab:g}",
                    "mode": "ca",
                    "tp": tp,
                    "sl": sl,
                    "h": h,
                    "confirm_sec": 45,
                    "confirm_thr": 0.015,
                    "abort_thr": ab,
                }
            )
        exit_specs.append(
            {
                "name": f"pp_tp{tp:g}_sl{sl:g}_h{h}_a08_f03",
                "mode": "pp",
                "tp": tp,
                "sl": sl,
                "h": h,
                "arm_ret": 0.08,
                "floor_ret": 0.03,
            }
        )
    # Keep one looser baseline reference
    exit_specs.append(
        {"name": "tpsl_tp0.2_sl0.25_h300", "mode": "tpsl", "tp": 0.20, "sl": 0.25, "h": 300}
    )

    # Dedupe exit names
    seen = set()
    uniq_exits = []
    for e in exit_specs:
        if e["name"] in seen:
            continue
        seen.add(e["name"])
        uniq_exits.append(e)
    exit_specs = uniq_exits

    # Preload trade paths for all pocket probes once
    path_cache: dict[tuple[str, str], dict[str, tuple[np.ndarray, np.ndarray]]] = {}

    def paths_for(date: str, sym: str):
        key = (date, sym)
        if key not in path_cache:
            tday = load_option_trades(trades_root, sym, date)
            path_cache[key] = (
                _paths_by_ticker(tday) if tday is not None and not tday.empty else {}
            )
        return path_cache[key]

    print(
        f"risk-opt entries={len(entry_specs)} exits={len(exit_specs)} "
        f"pockets={list(POCKET_SETS)} pf={args.position_frac} mc={args.max_concurrent}",
        flush=True,
    )

    score_rows: list[dict[str, Any]] = []
    pass_rows: list[dict[str, Any]] = []

    for pset_name, pset in POCKET_SETS.items():
        pdf = pd.DataFrame(sorted(pset), columns=["session", "tod_bucket", "dir"])
        base = probes.merge(pdf, on=["session", "tod_bucket", "dir"], how="inner")
        base = base.sort_values(["date", "symbol", "session", "entry_ts"]).drop_duplicates(
            ["date", "symbol", "session"], keep="first"
        )
        print(f"[pocket {pset_name}] base n={len(base)}", flush=True)

        for ent in entry_specs:
            mask = base.apply(lambda r: _entry_ok(r, spec=ent), axis=1)
            cand = base.loc[mask].copy()
            if len(cand) < 40:
                continue
            signals: list[dict[str, Any]] = []
            for _, r in cand.iterrows():
                arrs = paths_for(str(r["date"]), str(r["symbol"]))
                ticker = str(r["ticker"]).replace("O:", "")
                arr = arrs.get(ticker)
                if arr is None:
                    continue
                signals.append(
                    {
                        "date": str(r["date"]),
                        "symbol": str(r["symbol"]),
                        "dir": str(r["dir"]),
                        "session": str(r["session"]),
                        "entry_ts": to_ny(pd.Timestamp(r["entry_ts"])),
                        "pts": arr[0],
                        "plast": arr[1],
                        "calendar": str(r["calendar"]),
                    }
                )
            if len(signals) < 40:
                continue
            print(f"  entry {ent['name']}: signals={len(signals)}", flush=True)

            for ex in exit_specs:
                raw_trades: list[dict[str, Any]] = []
                for s in signals:
                    if ex["mode"] == "tpsl":
                        sim = simulate_trade_tpsl(
                            s["pts"],
                            s["plast"],
                            s["entry_ts"],
                            tp=float(ex["tp"]),
                            sl=float(ex["sl"]),
                            max_hold_sec=int(ex["h"]),
                            slip=float(args.slip),
                        )
                    elif ex["mode"] == "ca":
                        sim = simulate_trade_tpsl_confirm_abort(
                            s["pts"],
                            s["plast"],
                            s["entry_ts"],
                            tp=float(ex["tp"]),
                            sl=float(ex["sl"]),
                            max_hold_sec=int(ex["h"]),
                            confirm_sec=int(ex["confirm_sec"]),
                            confirm_thr=float(ex["confirm_thr"]),
                            abort_thr=float(ex["abort_thr"]),
                            on_timeout="abort",
                            slip=float(args.slip),
                        )
                    else:
                        sim = simulate_trade_profit_protect(
                            s["pts"],
                            s["plast"],
                            s["entry_ts"],
                            tp=float(ex["tp"]),
                            sl=float(ex["sl"]),
                            max_hold_sec=int(ex["h"]),
                            arm_ret=float(ex["arm_ret"]),
                            floor_ret=float(ex["floor_ret"]),
                            slip=float(args.slip),
                        )
                    if sim is None or not np.isfinite(sim["ret"]):
                        continue
                    et = s["entry_ts"]
                    raw_trades.append(
                        {
                            "date": s["date"],
                            "symbol": s["symbol"],
                            "dir": s["dir"],
                            "session": s["session"],
                            "entry_ts": et,
                            "exit_ts": et + pd.Timedelta(seconds=float(sim["hold_sec"])),
                            "ret": float(sim["ret"]),
                            "exit_reason": str(sim["reason"]),
                            "hold_sec": float(sim["hold_sec"]),
                            "calendar": s["calendar"],
                        }
                    )

                disc = [t for t in raw_trades if t["calendar"] == "may_jul09"]
                blind = [t for t in raw_trades if t["calendar"] == "jul10_23"]
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

                disc_ok = (
                    st_d["n"] >= 40
                    and st_d["trade_win"] is not None
                    and st_d["trade_win"] >= float(args.min_trade_win)
                    and st_d["day_win"] is not None
                    and st_d["day_win"] >= float(args.min_day_win)
                    and st_d["mean"] is not None
                    and st_d["mean"] > 0
                    and st_d["compound"] is not None
                    and st_d["compound"] > 0
                    and st_d["maxdd"] is not None
                    and st_d["maxdd"] >= float(args.max_dd)
                )
                blind_ok = (
                    st_b["n"] >= 8
                    and st_b["mean"] is not None
                    and st_b["mean"] > 0
                    and st_b["day_win"] is not None
                    and st_b["day_win"] >= 0.50
                    and st_b["maxdd"] is not None
                    and st_b["maxdd"] >= -0.20
                )
                row = {
                    "name": f"{pset_name}|{ent['name']}|{ex['name']}",
                    "pocket": pset_name,
                    "entry": ent["name"],
                    "exit": ex["name"],
                    "mode": ex["mode"],
                    "pass": bool(disc_ok and blind_ok),
                    "disc_ok": disc_ok,
                    "blind_ok": blind_ok,
                    "may_compound": months.get("2026-05"),
                    "jun_compound": months.get("2026-06"),
                    "jul_compound": months.get("2026-07"),
                }
                for k, v in st_d.items():
                    row[f"disc_{k}"] = v
                for k, v in st_b.items():
                    row[f"blind_{k}"] = v
                score_rows.append(row)
                if row["pass"]:
                    pass_rows.append(row)
                    print(
                        f"  *** PASS {row['name']} "
                        f"win={row['disc_trade_win']:.2f} day={row['disc_day_win']:.2f} "
                        f"dd={row['disc_maxdd']:.3f} cmp={row['disc_compound']:.2f} "
                        f"blind_win={row['blind_trade_win']:.2f}",
                        flush=True,
                    )

    sb = pd.DataFrame(score_rows)
    sb.to_csv(out / "scoreboard.csv", index=False)
    (out / "pass.json").write_text(json.dumps(pass_rows, indent=2, default=str), encoding="utf-8")

    # Rank: pass first, then higher trade_win, better (less neg) maxdd, higher compound
    if len(sb):
        sb_sorted = sb.sort_values(
            ["pass", "disc_ok", "disc_trade_win", "disc_maxdd", "disc_compound"],
            ascending=[False, False, False, False, False],
        )
    else:
        sb_sorted = sb

    champ = pass_rows[0] if pass_rows else None
    if pass_rows:
        champ = max(
            pass_rows,
            key=lambda r: (
                float(r.get("disc_trade_win") or 0),
                float(r.get("disc_maxdd") or -9),
                float(r.get("disc_compound") or -9),
                float(r.get("blind_trade_win") or 0),
            ),
        )
    elif len(sb_sorted):
        # best disc_ok even if blind fails
        disc_ok_rows = [r for r in score_rows if r.get("disc_ok")]
        pool = disc_ok_rows or score_rows
        champ = max(
            pool,
            key=lambda r: (
                float(r.get("disc_trade_win") or 0),
                float(r.get("disc_maxdd") or -9),
                float(r.get("disc_compound") or -9),
            ),
        )

    verdict = {
        "protocol": "pocket_risk_optimize",
        "portfolio": {"position_frac": args.position_frac, "max_concurrent": args.max_concurrent},
        "gates": {
            "min_trade_win": args.min_trade_win,
            "min_day_win": args.min_day_win,
            "max_dd": args.max_dd,
        },
        "n_cells": len(score_rows),
        "pass_n": len(pass_rows),
        "champion": champ,
        "verdict": "RISK_PASS" if pass_rows else ("DISCOVER_ONLY" if champ and champ.get("disc_ok") else "REJECT"),
    }
    (out / "summary.json").write_text(json.dumps(verdict, indent=2, default=str), encoding="utf-8")
    print("\n=== VERDICT ===", flush=True)
    print(json.dumps(verdict, indent=2, default=str)[:3500], flush=True)
    if len(sb_sorted):
        cols = [
            c
            for c in [
                "name",
                "pass",
                "disc_n",
                "disc_trade_win",
                "disc_day_win",
                "disc_maxdd",
                "disc_compound",
                "blind_n",
                "blind_trade_win",
                "blind_compound",
                "may_compound",
                "jun_compound",
                "jul_compound",
            ]
            if c in sb_sorted.columns
        ]
        print(sb_sorted[cols].head(20).to_string(index=False), flush=True)
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

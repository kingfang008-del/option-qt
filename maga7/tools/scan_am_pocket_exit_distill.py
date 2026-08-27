#!/usr/bin/env python3
"""Distill AM sleeve rules inside foresight TOD pockets (exit-first).

Uses ``research_am_vwap_foresight_map_* / probes.csv``:
  1) keep dir-aligned probes in discover-selected TOD pockets
  2) optional VWAP *structure* entry gates (accel / sess-vwap / fo band)
  3) sweep TP/SL (+ optional confirm-abort) on trade lasts
  4) dual-window portfolio score; Jul is blind report only for selection

Example:
  PYTHONPATH=. python -m maga7.tools.scan_am_pocket_exit_distill \\
    --probes /mnt/s990/data/maga7/results/research_am_vwap_foresight_map_may_jul/probes.csv \\
    --tag research_am_pocket_exit_distill
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

from maga7.common.config import load_profile
from maga7.common.option_trade_tpsl import (
    simulate_trade_tpsl,
    simulate_trade_tpsl_confirm_abort,
)
from maga7.common.option_trades import load_option_trades
from maga7.common.replay import to_ny
from maga7.tools.run_morning_sec_option_fill import _portfolio_day
from maga7.tools.scan_am_delayed_confirm_quote_dual import _ok, _stats
from maga7.tools.scan_session_horizon_foresight import _paths_by_ticker

NY = "America/New_York"
PROFILE = (
    "maga7/CONFIG/strategy_profiles/"
    "single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1.json"
)
DEFAULT_PROBES = Path(
    "/mnt/s990/data/maga7/results/research_am_vwap_foresight_map_may_jul/probes.csv"
)
DEFAULT_TRADES = Path("/mnt/s990/new_option_data_s3_trades")

# Discover-selected pockets from aligned foresight map (edge≥0.64, n≥400-ish).
DEFAULT_POCKETS = (
    ("AM_A_0930_1030", "09:30", "UP"),
    ("AM_A_0930_1030", "09:30", "DN"),
    ("AM_A_0930_1030", "09:35", "UP"),
    ("AM_B_1030_1130", "10:30", "DN"),
    ("AM_B_1030_1130", "10:35", "DN"),
    ("AM_B_1030_1130", "10:40", "UP"),
    ("AM_B_1030_1130", "10:45", "UP"),
    ("AM_B_1030_1130", "10:50", "UP"),
    ("AM_B_1030_1130", "10:55", "UP"),
)

WINDOWS = (
    ("may_jul09", "2026-05-01", "2026-07-09"),
    ("jul10_23", "2026-07-10", "2026-07-23"),
)


EntryGate = Callable[[pd.Series], bool]


def _parse_floats(spec: str) -> list[float]:
    return [float(x) for x in str(spec).split(",") if x.strip()]


def _parse_ints(spec: str) -> list[int]:
    return [int(x) for x in str(spec).split(",") if x.strip()]


def _signed(row: pd.Series, feat: str) -> float:
    v = float(row.get(feat) or np.nan)
    if not np.isfinite(v):
        return float("nan")
    return v if str(row["dir"]) == "UP" else -v


def _gate_none(_: pd.Series) -> bool:
    return True


def _gate_accel(min_accel: float) -> EntryGate:
    def _g(r: pd.Series) -> bool:
        a = float(r.get("accel_10_30") or np.nan)
        return bool(np.isfinite(a) and a + 1e-12 >= float(min_accel))

    return _g


def _gate_vwap_diff(thr: float) -> EntryGate:
    def _g(r: pd.Series) -> bool:
        return bool(_signed(r, "vwap_diff") + 1e-12 >= float(thr))

    return _g


def _gate_fo30_band(lo: float, hi: float) -> EntryGate:
    def _g(r: pd.Series) -> bool:
        x = _signed(r, "fo_vwap30")
        return bool(np.isfinite(x) and float(lo) - 1e-12 <= x <= float(hi) + 1e-12)

    return _g


def _gate_fo30_min(thr: float) -> EntryGate:
    def _g(r: pd.Series) -> bool:
        return bool(_signed(r, "fo_vwap30") + 1e-12 >= float(thr))

    return _g


def _build_entry_gates() -> list[tuple[str, EntryGate]]:
    gates: list[tuple[str, EntryGate]] = [
        ("pocket", _gate_none),
        ("accel0", _gate_accel(0.0)),
        ("accel2bp", _gate_accel(0.0002)),
        ("vd5", _gate_vwap_diff(0.005)),
        ("vd8", _gate_vwap_diff(0.008)),
        ("vd10", _gate_vwap_diff(0.010)),
        ("fo30_4", _gate_fo30_min(0.004)),
        ("fo30_8", _gate_fo30_min(0.008)),
        ("fo30_band_4_15", _gate_fo30_band(0.004, 0.015)),
        ("accel0_vd5", lambda r: _gate_accel(0.0)(r) and _gate_vwap_diff(0.005)(r)),
        ("accel0_fo8", lambda r: _gate_accel(0.0)(r) and _gate_fo30_min(0.008)(r)),
    ]
    return gates


def _dedupe_first_per_day_symbol(df: pd.DataFrame) -> pd.DataFrame:
    """60s foresight grid → keep earliest probe per (date, symbol, session)."""
    d = df.sort_values(["date", "symbol", "session", "entry_ts"])
    return d.drop_duplicates(["date", "symbol", "session"], keep="first").reset_index(drop=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", default=PROFILE)
    ap.add_argument("--probes", default=str(DEFAULT_PROBES))
    ap.add_argument("--trades-root", default=str(DEFAULT_TRADES))
    ap.add_argument("--tag", default="research_am_pocket_exit_distill")
    ap.add_argument("--tp", default="0.10,0.15,0.20")
    ap.add_argument("--sl", default="0.15,0.20,0.25")
    ap.add_argument("--max-hold", default="300,600,900")
    ap.add_argument("--slip", type=float, default=0.01)
    ap.add_argument("--position-frac", type=float, default=0.10)
    ap.add_argument("--max-concurrent", type=int, default=2)
    ap.add_argument("--cooldown-minutes", type=float, default=10.0)
    ap.add_argument("--min-n", type=int, default=8)
    ap.add_argument("--min-day-win", type=float, default=0.55)
    ap.add_argument("--with-ca", action="store_true", help="Also sweep confirm-abort exits")
    args = ap.parse_args(argv)

    prof = load_profile(args.profile)
    paths = prof["_paths"]
    out = Path(paths["results_dir"]) / args.tag
    out.mkdir(parents=True, exist_ok=True)
    trades_root = Path(args.trades_root)

    probes = pd.read_csv(args.probes)
    probes["entry_ts"] = pd.to_datetime(probes["entry_ts"])
    # Align dir with stock from_open
    probes = probes[
        probes["dir"] == np.where(probes["from_open_px"].astype(float) >= 0, "UP", "DN")
    ].copy()
    pocket_df = pd.DataFrame(DEFAULT_POCKETS, columns=["session", "tod_bucket", "dir"])
    probes = probes.merge(pocket_df, on=["session", "tod_bucket", "dir"], how="inner")
    probes = _dedupe_first_per_day_symbol(probes)
    print(
        f"pocket probes aligned+deduped n={len(probes)} "
        f"days={probes['date'].nunique()} symbols={probes['symbol'].nunique()}",
        flush=True,
    )
    probes.to_csv(out / "pocket_probes.csv", index=False)

    # Cache trade paths by (date, symbol)
    path_cache: dict[tuple[str, str], dict[str, tuple[np.ndarray, np.ndarray]]] = {}

    def _paths(date: str, sym: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        key = (date, sym)
        if key not in path_cache:
            tday = load_option_trades(trades_root, sym, date)
            path_cache[key] = _paths_by_ticker(tday) if tday is not None and not tday.empty else {}
        return path_cache[key]

    tps = _parse_floats(args.tp)
    sls = _parse_floats(args.sl)
    holds = _parse_ints(args.max_hold)
    entry_gates = _build_entry_gates()

    exit_cfgs: list[dict[str, Any]] = []
    for tp in tps:
        for sl in sls:
            for h in holds:
                exit_cfgs.append(
                    {
                        "name": f"tpsl_tp{tp:g}_sl{sl:g}_h{h}",
                        "mode": "tpsl",
                        "tp": tp,
                        "sl": sl,
                        "max_hold_sec": h,
                    }
                )
    if args.with_ca:
        for tp in tps:
            for sl in sls:
                for h in holds:
                    for abort in (0.08, 0.10):
                        exit_cfgs.append(
                            {
                                "name": f"ca_tp{tp:g}_sl{sl:g}_h{h}_ab{abort:g}",
                                "mode": "ca",
                                "tp": tp,
                                "sl": sl,
                                "max_hold_sec": h,
                                "confirm_sec": 60,
                                "confirm_thr": 0.02,
                                "abort_thr": abort,
                                "on_timeout": "abort",
                            }
                        )

    cells: list[dict[str, Any]] = []
    for gname, _ in entry_gates:
        for ex in exit_cfgs:
            cells.append({"entry": gname, "exit": ex["name"], **ex, "entry_name": gname})

    print(f"cells={len(cells)} entries={len(entry_gates)} exits={len(exit_cfgs)}", flush=True)

    # Pre-filter candidates per entry gate once
    gated: dict[str, pd.DataFrame] = {}
    for gname, gate in entry_gates:
        sub = probes[probes.apply(gate, axis=1)].copy()
        gated[gname] = sub
        print(f"  entry {gname}: n={len(sub)}", flush=True)

    score_rows: list[dict[str, Any]] = []
    dual_pass: list[dict[str, Any]] = []

    for ci, cell in enumerate(cells):
        if ci % 25 == 0:
            print(f"[cell] {ci+1}/{len(cells)} {cell['entry']}|{cell['exit']}", flush=True)
        cand = gated[str(cell["entry_name"])]
        win_raw: dict[str, list] = {w[0]: [] for w in WINDOWS}
        for _, r in cand.iterrows():
            cal = str(r["calendar"])
            if cal not in win_raw:
                continue
            paths_sym = _paths(str(r["date"]), str(r["symbol"]))
            ticker = str(r["ticker"]).replace("O:", "")
            arr = paths_sym.get(ticker)
            if arr is None:
                continue
            et = to_ny(pd.Timestamp(r["entry_ts"]))
            if cell["mode"] == "tpsl":
                sim = simulate_trade_tpsl(
                    arr[0],
                    arr[1],
                    et,
                    tp=float(cell["tp"]),
                    sl=float(cell["sl"]),
                    max_hold_sec=int(cell["max_hold_sec"]),
                    slip=float(args.slip),
                )
            else:
                sim = simulate_trade_tpsl_confirm_abort(
                    arr[0],
                    arr[1],
                    et,
                    tp=float(cell["tp"]),
                    sl=float(cell["sl"]),
                    max_hold_sec=int(cell["max_hold_sec"]),
                    confirm_sec=int(cell["confirm_sec"]),
                    confirm_thr=float(cell["confirm_thr"]),
                    abort_thr=float(cell["abort_thr"]),
                    on_timeout=str(cell["on_timeout"]),
                    slip=float(args.slip),
                )
            if sim is None or not np.isfinite(sim["ret"]):
                continue
            xt = et + pd.Timedelta(seconds=float(sim["hold_sec"]))
            win_raw[cal].append(
                {
                    "date": str(r["date"]),
                    "symbol": str(r["symbol"]),
                    "dir": str(r["dir"]),
                    "session": str(r["session"]),
                    "entry_ts": et,
                    "exit_ts": xt,
                    "ret": float(sim["ret"]),
                    "exit_reason": str(sim["reason"]),
                    "hold_sec": float(sim["hold_sec"]),
                    "size": float(args.position_frac),
                }
            )

        win_stats: dict[str, dict[str, Any]] = {}
        both = True
        for wname, _, _ in WINDOWS:
            rows = win_raw[wname]
            if not rows:
                win_stats[wname] = _stats([])
                both = False
                continue
            sized = _portfolio_day(
                sorted(rows, key=lambda x: (x["entry_ts"], x["symbol"])),
                position_frac=float(args.position_frac),
                max_concurrent=int(args.max_concurrent),
                cooldown_minutes=float(args.cooldown_minutes),
            )
            st = _stats(sized)
            win_stats[wname] = st
            # Selection uses discover only for dual_pass flag; jul is reported.
            if wname == "may_jul09" and not _ok(
                st, min_n=int(args.min_n), min_day_win=float(args.min_day_win)
            ):
                both = False
            if wname == "jul10_23":
                # Blind: require mean>0 & n>=5 soft; not in selection gate for both
                pass
        # Dual = discover PASS and blind mean>0 with n>=5
        blind = win_stats.get("jul10_23") or {}
        blind_ok = (
            blind.get("mean") is not None
            and float(blind["mean"]) > 0
            and int(blind.get("n") or 0) >= 5
            and float(blind.get("day_win") or 0) >= 0.50
        )
        disc_ok = _ok(
            win_stats.get("may_jul09") or {},
            min_n=int(args.min_n),
            min_day_win=float(args.min_day_win),
        )
        passed = bool(disc_ok and blind_ok)
        row = {
            "name": f"{cell['entry']}|{cell['exit']}",
            "entry": cell["entry"],
            "exit": cell["exit"],
            "mode": cell["mode"],
            "tp": cell["tp"],
            "sl": cell["sl"],
            "max_hold_sec": cell["max_hold_sec"],
            "dual_pass": passed,
            "discover_pass": disc_ok,
            "blind_ok": blind_ok,
        }
        if cell["mode"] == "ca":
            row["abort_thr"] = cell["abort_thr"]
        for wname, _, _ in WINDOWS:
            for k, v in win_stats[wname].items():
                row[f"{wname}_{k}"] = v
        score_rows.append(row)
        if passed:
            dual_pass.append(row)
            print(
                f"  *** PASS {row['name']} "
                f"MJ n={row.get('may_jul09_n')} mean={row.get('may_jul09_mean'):.4f} "
                f"J10 n={row.get('jul10_23_n')} mean={row.get('jul10_23_mean'):.4f}",
                flush=True,
            )

    sb = pd.DataFrame(score_rows)
    sb.to_csv(out / "scoreboard.csv", index=False)
    (out / "dual_pass.json").write_text(
        json.dumps(dual_pass, indent=2, default=str), encoding="utf-8"
    )

    champ = None
    if dual_pass:
        champ = max(
            dual_pass,
            key=lambda r: (
                float(r.get("may_jul09_mean") or -9),
                float(r.get("jul10_23_mean") or -9),
                float(r.get("may_jul09_day_win") or 0),
            ),
        )
    elif score_rows:
        # best discover-only for reporting
        disc_rows = [r for r in score_rows if r.get("discover_pass")]
        pool = disc_rows or score_rows
        champ = max(
            pool,
            key=lambda r: (
                float(r.get("may_jul09_mean") or -9),
                float(r.get("may_jul09_day_win") or 0),
            ),
        )

    verdict = {
        "protocol": "pocket_exit_distill",
        "pockets": [list(p) for p in DEFAULT_POCKETS],
        "n_pocket_probes": int(len(probes)),
        "n_cells": len(score_rows),
        "dual_pass_n": len(dual_pass),
        "champion": champ,
        "verdict": (
            "DISTILL_PASS"
            if dual_pass
            else ("DISCOVER_ONLY" if any(r.get("discover_pass") for r in score_rows) else "REJECT")
        ),
        "note": "Selection requires discover PASS + blind mean>0 day_win≥0.50 n≥5.",
    }
    (out / "summary.json").write_text(json.dumps(verdict, indent=2, default=str), encoding="utf-8")
    print("\n=== VERDICT ===", flush=True)
    print(json.dumps(verdict, indent=2, default=str)[:3000], flush=True)
    if len(sb):
        top = sb.sort_values(
            ["dual_pass", "discover_pass", "may_jul09_mean"],
            ascending=[False, False, False],
        ).head(15)
        cols = [
            c
            for c in [
                "name",
                "dual_pass",
                "may_jul09_n",
                "may_jul09_mean",
                "may_jul09_day_win",
                "jul10_23_n",
                "jul10_23_mean",
                "jul10_23_day_win",
            ]
            if c in top.columns
        ]
        print(top[cols].to_string(index=False), flush=True)
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

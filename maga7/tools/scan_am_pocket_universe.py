#!/usr/bin/env python3
"""Pocket-universe contrast for frozen AM multi-gate + TP8.

After path-exits failed to lift capture, ask whether a different TOD pocket
set (a_only / dn_heavy / all with B-UP) beats no_b_up under the same entry/exit.

Example:
  PYTHONPATH=. python -m maga7.tools.scan_am_pocket_universe \\
    --tag research_am_pocket_universe
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
from maga7.tools.scan_am_pocket_exit_design import _path_window, simulate_exit
from maga7.tools.scan_am_pocket_multi_gate import build_gates, enrich_probes
from maga7.tools.scan_am_pocket_risk_optimize import (
    POCKET_SETS,
    _equity_stats,
    _month_compounds,
)
from maga7.tools.scan_session_horizon_foresight import _paths_by_ticker

DEFAULT_PROBES = Path(
    "/mnt/s990/data/maga7/results/research_am_vwap_foresight_map_may_jul/probes.csv"
)
DEFAULT_TRADES = Path("/mnt/s990/new_option_data_s3_trades")

ENTRY_GATES = (
    "vd_soft",
    "vd+cont60+mf100+volr12",
    "vd+volr12",
    "accel0",
)

EXIT = {"mode": "tpsl", "tp": 0.08, "sl": 0.15, "max_hold": 240}


def _align_pocket_probes(raw: pd.DataFrame, pset: set[tuple[str, str, str]]) -> pd.DataFrame:
    probes = raw[
        raw["dir"] == np.where(raw["from_open_px"].astype(float) >= 0, "UP", "DN")
    ].copy()
    pdf = pd.DataFrame(sorted(pset), columns=["session", "tod_bucket", "dir"])
    probes = probes.merge(pdf, on=["session", "tod_bucket", "dir"], how="inner")
    return probes.sort_values(["date", "symbol", "session", "entry_ts"]).drop_duplicates(
        ["date", "symbol", "session"], keep="first"
    )


def _extra_pocket_sets() -> dict[str, set[tuple[str, str, str]]]:
    """Incremental B-UP cell add-ons on top of no_b_up."""
    base = set(POCKET_SETS["no_b_up"])
    b_up = sorted(POCKET_SETS["all"] - base)
    out: dict[str, set[tuple[str, str, str]]] = {}
    for cell in b_up:
        tag = f"no_b_up+{cell[1]}_{cell[2]}"
        out[tag] = set(base) | {cell}
    out["b_up_only"] = set(b_up)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--probes", default=str(DEFAULT_PROBES))
    ap.add_argument("--trades-root", default=str(DEFAULT_TRADES))
    ap.add_argument("--tag", default="research_am_pocket_universe")
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
    ap.add_argument("--skip-enrich", action="store_true")
    args = ap.parse_args(argv)

    prof = load_profile(args.profile)
    out = Path(prof["_paths"]["results_dir"]) / args.tag
    out.mkdir(parents=True, exist_ok=True)
    stock_1s = Path(prof["_paths"]["stock_1s_root"])
    trades_root = Path(args.trades_root)
    symbols = list(prof.get("symbols") or [])

    pocket_sets = {**POCKET_SETS, **_extra_pocket_sets()}
    union: set[tuple[str, str, str]] = set()
    for s in pocket_sets.values():
        union |= set(s)

    enriched_path = out / "enriched_union.csv"
    if args.skip_enrich and enriched_path.exists():
        print(f"load {enriched_path}", flush=True)
        enriched = pd.read_csv(enriched_path)
        enriched["entry_ts"] = pd.to_datetime(enriched["entry_ts"])
    else:
        raw = pd.read_csv(args.probes)
        raw["entry_ts"] = pd.to_datetime(raw["entry_ts"])
        base = _align_pocket_probes(raw, union)
        print(f"union probes={len(base)} enriching…", flush=True)
        enriched = enrich_probes(base, stock_1s_root=stock_1s, symbols=symbols)
        enriched.to_csv(enriched_path, index=False)
        print(f"wrote {enriched_path} ok={int(enriched['enrich_ok'].sum())}", flush=True)

    enriched = enriched[enriched["enrich_ok"] == True].copy()  # noqa: E712

    path_cache: dict[tuple[str, str], dict[str, tuple[np.ndarray, np.ndarray]]] = {}

    def paths_for(date: str, sym: str):
        key = (date, sym)
        if key not in path_cache:
            tday = load_option_trades(trades_root, sym, date)
            path_cache[key] = (
                _paths_by_ticker(tday) if tday is not None and not tday.empty else {}
            )
        return path_cache[key]

    prepared: list[dict[str, Any]] = []
    for _, r in enriched.iterrows():
        arrs = paths_for(str(r["date"]), str(r["symbol"]))
        arr = arrs.get(str(r["ticker"]).replace("O:", ""))
        if arr is None:
            continue
        et = to_ny(pd.Timestamp(r["entry_ts"]))
        win = _path_window(arr[0], arr[1], et, max_hold_sec=900, slip=float(args.slip))
        if win is None:
            continue
        rets, holds, _, _ = win
        prepared.append(
            {
                "row": r,
                "date": str(r["date"]),
                "symbol": str(r["symbol"]),
                "dir": str(r["dir"]),
                "session": str(r["session"]),
                "tod_bucket": str(r["tod_bucket"]),
                "calendar": str(r["calendar"]),
                "entry_ts": et,
                "rets": rets,
                "holds": holds,
                "oracle_ret": float(r["oracle_ret"]),
                "cell": (str(r["session"]), str(r["tod_bucket"]), str(r["dir"])),
            }
        )
    print(f"prepared={len(prepared)}", flush=True)

    gate_map = dict(build_gates())
    for g in ENTRY_GATES:
        if g not in gate_map:
            raise SystemExit(f"missing gate {g}")

    score_rows: list[dict[str, Any]] = []
    for pset_name, pset in sorted(pocket_sets.items()):
        in_pocket = np.array([p["cell"] in pset for p in prepared], dtype=bool)
        for gname in ENTRY_GATES:
            gfn = gate_map[gname]
            mask = in_pocket & np.array([bool(gfn(p["row"])) for p in prepared], dtype=bool)
            subset = [p for p, ok in zip(prepared, mask) if ok]
            raw_trades: list[dict[str, Any]] = []
            for p in subset:
                sim = simulate_exit(p["rets"], p["holds"], mode="tpsl", params=EXIT)
                if sim is None or not np.isfinite(sim.get("ret", np.nan)):
                    continue
                et = p["entry_ts"]
                raw_trades.append(
                    {
                        "date": p["date"],
                        "symbol": p["symbol"],
                        "dir": p["dir"],
                        "session": p["session"],
                        "tod_bucket": p["tod_bucket"],
                        "calendar": p["calendar"],
                        "entry_ts": et,
                        "exit_ts": et + pd.Timedelta(seconds=float(sim["hold_sec"])),
                        "ret": float(sim["ret"]),
                        "exit_reason": str(sim["reason"]),
                        "hold_sec": float(sim["hold_sec"]),
                        "oracle_ret": float(p["oracle_ret"]),
                    }
                )
            disc = [t for t in raw_trades if t["calendar"] == "may_jul09"]
            blind = [t for t in raw_trades if t["calendar"] == "jul10_23"]
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
            if raw_trades:
                o = np.array([t["oracle_ret"] for t in raw_trades], dtype=float)
                rr = np.array([t["ret"] for t in raw_trades], dtype=float)
                mean_cap = float(rr.mean() / o.mean()) if o.mean() > 0 else float("nan")
                mean_oracle = float(o.mean())
                mean_ret = float(rr.mean())
            else:
                mean_cap = mean_oracle = mean_ret = float("nan")
            row: dict[str, Any] = {
                "pocket": pset_name,
                "entry": gname,
                "n_cells": len(pset),
                "n_raw": len(raw_trades),
                "mean_ret": mean_ret,
                "mean_oracle": mean_oracle,
                "mean_capture": mean_cap,
                "may": months.get("2026-05"),
                "jun": months.get("2026-06"),
                "jul": months.get("2026-07"),
            }
            for k, v in st_d.items():
                row[f"disc_{k}"] = v
            for k, v in st_b.items():
                row[f"blind_{k}"] = v
            score_rows.append(row)
            def _f(d: dict[str, Any], k: str) -> float:
                v = d.get(k, float("nan"))
                try:
                    fv = float(v)  # type: ignore[arg-type]
                except (TypeError, ValueError):
                    return float("nan")
                return fv if np.isfinite(fv) else float("nan")

            cap_print = _f({"x": mean_cap}, "x")
            print(
                f"{pset_name:22s} {gname:28s} n={len(raw_trades):3d} "
                f"win={_f(st_d, 'trade_win'):.3f} "
                f"dd={_f(st_d, 'maxdd'):.3f} "
                f"cmp={_f(st_d, 'compound'):.3f} "
                f"cap={cap_print:.3f}",
                flush=True,
            )

    sb = pd.DataFrame(score_rows)
    sb.to_csv(out / "scoreboard.csv", index=False)

    champ = "vd+cont60+mf100+volr12"
    base = sb[(sb.pocket == "no_b_up") & (sb.entry == champ)]
    base_row = base.iloc[0].to_dict() if len(base) else {}
    bw = float(base_row.get("disc_trade_win") or 0.74)
    bdd = float(base_row.get("disc_maxdd") or -0.12)
    bcmp = float(base_row.get("disc_compound") or 0.44)
    bcap = float(base_row.get("mean_capture") or 0.13)

    # same entry, other pockets that beat on soft criteria
    same = sb[sb.entry == champ].copy()
    better = same[
        (same["disc_n"] >= 15)
        & (same["disc_trade_win"] >= bw - 0.03)
        & (same["disc_maxdd"] >= -0.20)
        & (same["may"] > 0)
        & (
            (same["mean_capture"] > bcap + 0.015)
            | (same["disc_maxdd"] > bdd + 0.02)
            | (same["disc_compound"] > bcmp + 0.05)
        )
    ].sort_values(["mean_capture", "disc_compound"], ascending=[False, False])

    # risk-preferred: DD better, compound not crushed
    safer = same[
        (same["disc_n"] >= 15)
        & (same["disc_maxdd"] > bdd + 0.01)
        & (same["disc_compound"] >= bcmp - 0.08)
        & (same["disc_trade_win"] >= 0.65)
        & (same["may"] > 0)
    ].sort_values("disc_maxdd", ascending=False)

    # per pocket best entry under soft risk
    soft = sb[
        (sb["disc_n"] >= 15)
        & (sb["disc_trade_win"] >= 0.65)
        & (sb["disc_maxdd"] >= -0.20)
        & (sb["disc_compound"] > 0)
        & (sb["may"] > 0)
    ].copy()
    if not soft.empty:
        soft["score"] = (
            soft["disc_trade_win"] * 0.2
            + (1 + soft["disc_maxdd"]) * 0.25
            + np.clip(soft["disc_compound"], 0, 1.5) / 1.5 * 0.25
            + np.clip(soft["mean_capture"], 0, 0.35) / 0.35 * 0.3
        )
        soft = soft.sort_values("score", ascending=False)

    verdict = {
        "protocol": "pocket_universe_x_champ_gates_tp8",
        "baseline_no_b_up_champ": base_row,
        "better_same_entry": better.head(12).to_dict(orient="records") if len(better) else [],
        "safer_same_entry": safer.head(12).to_dict(orient="records") if len(safer) else [],
        "top_soft_any": soft.head(15).to_dict(orient="records") if len(soft) else [],
        "pocket_sets": {k: sorted(list(v)) for k, v in pocket_sets.items()},
    }
    (out / "summary.json").write_text(json.dumps(verdict, indent=2, default=str), encoding="utf-8")

    cols = [
        c
        for c in [
            "pocket",
            "entry",
            "n_raw",
            "disc_n",
            "disc_trade_win",
            "disc_maxdd",
            "disc_compound",
            "mean_capture",
            "mean_oracle",
            "may",
            "jun",
            "jul",
            "blind_trade_win",
            "blind_compound",
        ]
        if c in sb.columns
    ]
    print("\nBASELINE no_b_up + champ", flush=True)
    print(base[cols].to_string(index=False) if len(base) else "(none)", flush=True)
    print("\nCHAMP entry × pockets", flush=True)
    print(same[cols].sort_values("disc_compound", ascending=False).to_string(index=False), flush=True)
    print("\nBETTER same entry", flush=True)
    print(better[cols].head(10).to_string(index=False) if len(better) else "(none)", flush=True)
    print("\nSAFER same entry", flush=True)
    print(safer[cols].head(10).to_string(index=False) if len(safer) else "(none)", flush=True)
    print("\nTOP soft any", flush=True)
    print(soft[cols].head(12).to_string(index=False) if len(soft) else "(none)", flush=True)
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

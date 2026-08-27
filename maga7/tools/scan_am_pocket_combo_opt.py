#!/usr/bin/env python3
"""Combine multi-gate AM entries with scale-out / path exits.

Takes champion entry gates from multi_gate research and sweeps a focused exit
family to see if capture/DD jump together.

Example:
  PYTHONPATH=. python -m maga7.tools.scan_am_pocket_combo_opt \\
    --tag research_am_pocket_combo_opt
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
from maga7.tools.scan_am_pocket_multi_gate import build_gates
from maga7.tools.scan_am_pocket_risk_optimize import _equity_stats, _month_compounds
from maga7.tools.scan_am_pocket_scaleout import simulate_scaleout
from maga7.tools.scan_session_horizon_foresight import _paths_by_ticker

DEFAULT_ENRICHED = Path(
    "/mnt/s990/data/maga7/results/research_am_pocket_multi_gate/enriched_probes.csv"
)
DEFAULT_TRADES = Path("/mnt/s990/new_option_data_s3_trades")

# Champion / contrast entry gates from multi_gate scoreboard.
ENTRY_GATES = (
    "vd_soft",
    "vd+volr12",
    "vd+streak3",
    "vd+mf100+volr12",
    "vd+cont60+mf100+volr12",
    "vd+agree+cont60+mf100+volr12",
)


def _exit_grid() -> list[dict[str, Any]]:
    cfgs: list[dict[str, Any]] = []
    for tp, sl, h in (
        (0.08, 0.15, 240),
        (0.08, 0.12, 240),
        (0.10, 0.15, 300),
        (0.12, 0.15, 300),
        (0.15, 0.15, 300),
        (0.20, 0.20, 300),
    ):
        cfgs.append(
            {
                "name": f"tpsl_tp{tp:g}_sl{sl:g}_h{h}",
                "kind": "tpsl",
                "tp": tp,
                "sl": sl,
                "max_hold": h,
            }
        )
    # scale-out champions + neighbors
    for frac1, tp1, sl, h, runner, extra in (
        (0.67, 0.06, 0.15, 600, "trail", {"arm": 0.15, "trail": 0.15, "be_after_scale": True, "floor": 0.0}),
        (0.67, 0.06, 0.12, 600, "trail", {"arm": 0.15, "trail": 0.15, "be_after_scale": True, "floor": 0.0}),
        (0.67, 0.08, 0.15, 600, "trail", {"arm": 0.15, "trail": 0.15, "be_after_scale": True, "floor": 0.0}),
        (0.50, 0.06, 0.15, 600, "trail", {"arm": 0.15, "trail": 0.15, "be_after_scale": True, "floor": 0.0}),
        (0.67, 0.06, 0.15, 600, "tp", {"tp2": 0.20, "be_after_scale": True, "floor": 0.0}),
        (0.67, 0.06, 0.15, 900, "tp", {"tp2": 0.30, "be_after_scale": True, "floor": 0.0}),
        (0.67, 0.06, 0.15, 600, "hold", {"be_after_scale": True, "floor": 0.0}),
        (0.50, 0.08, 0.15, 600, "hold", {"be_after_scale": True, "floor": 0.0}),
        # tighter first scale + wider runner trail
        (0.67, 0.05, 0.12, 600, "trail", {"arm": 0.12, "trail": 0.10, "be_after_scale": True, "floor": 0.0}),
        (0.50, 0.05, 0.12, 900, "trail", {"arm": 0.15, "trail": 0.12, "be_after_scale": True, "floor": 0.02}),
        # time-cut overlay on tpsl
    ):
        name = f"sc{frac1:g}@{tp1:g}_{runner}_sl{sl:g}_h{h}"
        if extra.get("trail") is not None:
            name += f"_a{extra['arm']:g}_t{extra['trail']:g}"
        if extra.get("tp2") is not None:
            name += f"_tp2{extra['tp2']:g}"
        if extra.get("floor") not in (None, -9.0) and runner == "hold":
            name += f"_be{extra['floor']:g}"
        cfg = {
            "name": name,
            "kind": "scale",
            "frac1": frac1,
            "tp1": tp1,
            "sl": sl,
            "max_hold": h,
            "runner": runner,
            **extra,
        }
        cfgs.append(cfg)

    for tc in (90, 120, 180):
        cfgs.append(
            {
                "name": f"tpsl_tp0.08_sl0.15_h240_tc{tc}",
                "kind": "tpsl",
                "tp": 0.08,
                "sl": 0.15,
                "max_hold": 240,
                "time_cut": tc,
                "time_cut_min": 0.0,
            }
        )
        cfgs.append(
            {
                "name": f"sc0.67@0.06_trail_sl0.15_h600_a0.15_t0.15_tc{tc}",
                "kind": "scale",
                "frac1": 0.67,
                "tp1": 0.06,
                "sl": 0.15,
                "max_hold": 600,
                "runner": "trail",
                "arm": 0.15,
                "trail": 0.15,
                "be_after_scale": True,
                "floor": 0.0,
                "time_cut": tc,
                "time_cut_min": 0.0,
            }
        )
    return cfgs


def _run_exit(rets: np.ndarray, holds: np.ndarray, ex: dict[str, Any]) -> dict[str, Any] | None:
    mh = float(ex.get("max_hold", 900))
    mask = holds <= mh + 1e-9
    rets = rets[mask]
    holds = holds[mask]
    if len(rets) < 2:
        return None
    if ex["kind"] == "tpsl":
        return simulate_exit(
            rets,
            holds,
            mode="tpsl",
            params={
                "tp": float(ex["tp"]),
                "sl": float(ex["sl"]),
                "max_hold": mh,
                "time_cut": float(ex.get("time_cut", 9e9)),
                "time_cut_min": float(ex.get("time_cut_min", -9.0)),
            },
        )
    return simulate_scaleout(
        rets,
        holds,
        frac1=float(ex["frac1"]),
        tp1=float(ex["tp1"]),
        sl=float(ex["sl"]),
        max_hold=mh,
        runner=str(ex["runner"]),
        tp2=float(ex.get("tp2", 9.0)),
        arm=float(ex.get("arm", 0.0)),
        trail=float(ex.get("trail", 9.0)),
        floor=float(ex.get("floor", -9.0)),
        time_cut=float(ex.get("time_cut", 9e9)),
        time_cut_min=float(ex.get("time_cut_min", -9.0)),
        be_after_scale=bool(ex.get("be_after_scale", False)),
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--enriched", default=str(DEFAULT_ENRICHED))
    ap.add_argument("--trades-root", default=str(DEFAULT_TRADES))
    ap.add_argument("--tag", default="research_am_pocket_combo_opt")
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

    probes = pd.read_csv(args.enriched)
    probes["entry_ts"] = pd.to_datetime(probes["entry_ts"])
    probes = probes[probes["enrich_ok"] == True].copy()  # noqa: E712

    gate_map = dict(build_gates())
    for g in ENTRY_GATES:
        if g not in gate_map:
            raise SystemExit(f"missing gate {g}")

    path_cache: dict[tuple[str, str], dict[str, tuple[np.ndarray, np.ndarray]]] = {}

    def paths_for(date: str, sym: str):
        key = (date, sym)
        if key not in path_cache:
            tday = load_option_trades(trades_root, sym, date)
            path_cache[key] = (
                _paths_by_ticker(tday) if tday is not None and not tday.empty else {}
            )
        return path_cache[key]

    # prepare once
    prepared: list[dict[str, Any]] = []
    for _, r in probes.iterrows():
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
                "row": r,
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
    print(f"prepared={len(prepared)}", flush=True)

    entry_masks = {
        g: np.array([bool(gate_map[g](p["row"])) for p in prepared], dtype=bool)
        for g in ENTRY_GATES
    }
    for g, m in entry_masks.items():
        print(f"  entry {g}: n={int(m.sum())}", flush=True)

    exits = _exit_grid()
    print(f"exits={len(exits)}", flush=True)

    score_rows: list[dict[str, Any]] = []
    for gname, mask in entry_masks.items():
        subset = [p for p, ok in zip(prepared, mask) if ok]
        for ex in exits:
            raw = []
            for p in subset:
                sim = _run_exit(p["rets"], p["holds"], ex)
                if sim is None or not np.isfinite(sim["ret"]):
                    continue
                et = p["entry_ts"]
                raw.append(
                    {
                        "date": p["date"],
                        "symbol": p["symbol"],
                        "dir": p["dir"],
                        "session": p["session"],
                        "calendar": p["calendar"],
                        "entry_ts": et,
                        "exit_ts": et + pd.Timedelta(seconds=float(sim["hold_sec"])),
                        "ret": float(sim["ret"]),
                        "exit_reason": str(sim["reason"]),
                        "hold_sec": float(sim["hold_sec"]),
                        "oracle_ret": float(p["oracle_ret"]),
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
                rr = np.array([t["ret"] for t in raw], dtype=float)
                mean_cap = float(rr.mean() / o.mean()) if o.mean() > 0 else float("nan")
            else:
                mean_cap = float("nan")
            row: dict[str, Any] = {
                "entry": gname,
                "exit": ex["name"],
                "kind": ex["kind"],
                "n_raw": len(raw),
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

    sb = pd.DataFrame(score_rows)
    sb.to_csv(out / "scoreboard.csv", index=False)

    # baselines
    base = sb[(sb.entry == "vd_soft") & (sb.exit == "tpsl_tp0.08_sl0.15_h240")]
    champ_entry = "vd+cont60+mf100+volr12"
    base_row = base.iloc[0].to_dict() if len(base) else {}

    soft = sb[
        (sb["disc_n"] >= 18)
        & (sb["disc_trade_win"] >= 0.68)
        & (sb["disc_maxdd"] >= -0.16)
        & (sb["disc_compound"] > 0)
        & (sb["blind_n"] >= 3)
        & (sb["may"] > 0)
        & (sb["jul"] > -0.05)
    ].copy()
    if not soft.empty:
        soft["score"] = (
            soft["disc_trade_win"] * 0.22
            + (1 + soft["disc_maxdd"]) * 0.28
            + np.clip(soft["disc_compound"], 0, 1.5) / 1.5 * 0.2
            + np.clip(soft["mean_capture"], 0, 0.25) / 0.25 * 0.3
        )
        soft = soft.sort_values("score", ascending=False)

    # best for champion entry
    sub = sb[sb.entry == champ_entry].sort_values(
        ["disc_trade_win", "disc_maxdd", "mean_capture"], ascending=[False, False, False]
    )

    # Pareto vs vd_soft baseline: better DD and win not worse, or better capture with DD ok
    bw = float(base_row.get("disc_trade_win") or 0.69)
    bdd = float(base_row.get("disc_maxdd") or -0.16)
    bcap = float(base_row.get("mean_capture") or 0.09)
    bcmp = float(base_row.get("disc_compound") or 0.33)
    pareto = sb[
        (sb["disc_trade_win"] >= bw - 0.01)
        & (sb["disc_maxdd"] >= bdd)
        & (sb["disc_compound"] >= bcmp * 0.8)
        & (sb["blind_mean"] > -0.02)
        & (sb["may"] > 0)
    ].sort_values(["disc_maxdd", "mean_capture", "disc_compound"], ascending=[False, False, False])

    verdict = {
        "protocol": "multi_gate_entry_x_focused_exit",
        "portfolio": {"position_frac": args.position_frac, "max_concurrent": args.max_concurrent},
        "baseline": base_row,
        "champ_entry": champ_entry,
        "top_soft": soft.head(12).to_dict(orient="records") if len(soft) else [],
        "pareto_vs_baseline": pareto.head(12).to_dict(orient="records") if len(pareto) else [],
        "champ_entry_top": sub.head(10).to_dict(orient="records"),
        "delta_note": {
            "baseline_win": bw,
            "baseline_dd": bdd,
            "baseline_cap": bcap,
            "baseline_cmp": bcmp,
        },
    }
    (out / "summary.json").write_text(json.dumps(verdict, indent=2, default=str), encoding="utf-8")

    cols = [
        c
        for c in [
            "entry",
            "exit",
            "kind",
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
        if c in sb.columns
    ]
    print("\nBASELINE", flush=True)
    print(base[cols].to_string(index=False), flush=True)
    print(f"\nTOP soft", flush=True)
    print(soft[cols].head(15).to_string(index=False) if len(soft) else "(none)", flush=True)
    print(f"\nPARETO vs vd_soft+TP8", flush=True)
    print(pareto[cols].head(12).to_string(index=False) if len(pareto) else "(none)", flush=True)
    print(f"\nCHAMP ENTRY {champ_entry} top exits", flush=True)
    print(sub[cols].head(12).to_string(index=False), flush=True)
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Overlay rare 1s impact gates on frozen AM pocket champ entry.

Champ (freeze): no_b_up + vd_soft ∩ cont60 ∩ mf100+ ∩ volr12
Exit: TP8/SL15/h240 @20%/5

Impact cuts come from research_buyer_impact_1s (AM stride percentiles),
applied as hard AND filters — ask whether rare impact lifts capture /
dual without killing the sleeve.

Example:
  PYTHONPATH=. python -m maga7.tools.scan_am_pocket_impact_overlay \\
    --tag research_am_pocket_impact_overlay
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

from maga7.common.bar_agg import load_stock_1s_day
from maga7.common.config import load_profile
from maga7.common.option_trades import load_option_trades
from maga7.common.replay import to_ny
from maga7.common.session_1s_features import features_at, prepare_day_arrays
from maga7.tools.run_morning_sec_option_fill import _portfolio_day
from maga7.tools.scan_am_pocket_exit_design import ENTRY_VD_SOFT, _path_window, simulate_exit
from maga7.tools.scan_am_pocket_multi_gate import build_gates
from maga7.tools.scan_am_pocket_risk_optimize import _entry_ok, _equity_stats, _month_compounds
from maga7.tools.scan_buyer_impact_1s import _impact_row
from maga7.tools.scan_session_horizon_foresight import _paths_by_ticker

DEFAULT_ENRICHED = Path(
    "/mnt/s990/data/maga7/results/research_am_pocket_multi_gate/enriched_probes.csv"
)
DEFAULT_TRADES = Path("/mnt/s990/new_option_data_s3_trades")
# Absolute cuts from research_buyer_impact_1s lift_table (AM 09:30–11:30 stride).
IMPACT_CUTS = {"p90": 1.4157, "p95": 1.8700, "p98": 2.7359}
EXIT_BASE = {"name": "tpsl_tp0.08_sl0.15_h240", "mode": "tpsl", "tp": 0.08, "sl": 0.15, "max_hold": 240}
GateFn = Callable[[pd.Series], bool]


def _attach_impact(probes: pd.DataFrame, stock_1s_root: Path) -> pd.DataFrame:
    """Add ret_15 + impact_score from causal 1s (enriched may lack ret_15)."""
    cache: dict[tuple[str, str], dict[str, np.ndarray] | None] = {}
    rows: list[dict[str, Any]] = []
    for idx, r in probes.iterrows():
        key = (str(r["date"]), str(r["symbol"]).upper())
        if key not in cache:
            raw = load_stock_1s_day(stock_1s_root, key[1], key[0])
            cache[key] = (
                None
                if raw is None or getattr(raw, "empty", True)
                else prepare_day_arrays(raw)
            )
        arr = cache[key]
        feat = features_at(arr, to_ny(pd.Timestamp(r["entry_ts"]))) if arr is not None else None
        if feat is None:
            rows.append({"_i": idx, "impact_ok": False})
            continue
        imp = _impact_row(feat)
        rows.append(
            {
                "_i": idx,
                "impact_ok": True,
                "ret_15": imp.get("ret15"),
                "abs_ret15": imp.get("abs_ret15"),
                "abs_ret30": imp.get("abs_ret30"),
                "impact_score": imp.get("impact_score"),
            }
        )
    extra = pd.DataFrame(rows).set_index("_i")
    out = probes.copy()
    for c in extra.columns:
        out[c] = extra[c]
    return out


def _champ_fn() -> GateFn:
    gmap = dict(build_gates())
    return gmap["vd+cont60+mf100+volr12"]


def _overlay_gates(champ: GateFn) -> list[tuple[str, GateFn]]:
    def _and(name: str, extra: GateFn) -> tuple[str, GateFn]:
        def g(r: pd.Series, _c=champ, _e=extra) -> bool:
            return bool(_c(r)) and bool(_e(r))

        return name, g

    def volr(thr: float) -> GateFn:
        def g(r: pd.Series, _t=thr) -> bool:
            v = float(r.get("volume_ratio_60") or np.nan)
            return np.isfinite(v) and v >= _t

        return g

    def volz(thr: float) -> GateFn:
        def g(r: pd.Series, _t=thr) -> bool:
            v = float(r.get("vol_z") or np.nan)
            return np.isfinite(v) and v >= _t

        return g

    def abs_ret30(bp: float) -> GateFn:
        thr = bp / 10000.0

        def g(r: pd.Series, _t=thr) -> bool:
            v = float(r.get("abs_ret30") or np.nan)
            if not np.isfinite(v):
                v = abs(float(r.get("ret_30") or np.nan))
            return np.isfinite(v) and v >= _t

        return g

    def impact_ge(cut: float) -> GateFn:
        def g(r: pd.Series, _c=cut) -> bool:
            v = float(r.get("impact_score") or np.nan)
            return np.isfinite(v) and v >= _c

        return g

    def ret30_volr(bp: float, vr: float) -> GateFn:
        a = abs_ret30(bp)
        b = volr(vr)

        def g(r: pd.Series) -> bool:
            return a(r) and b(r)

        return g

    gates: list[tuple[str, GateFn]] = [("champ", champ)]
    for name, fn in [
        ("+volr15", volr(1.5)),
        ("+volr20", volr(2.0)),
        ("+volz2", volz(2.0)),
        ("+volz3", volz(3.0)),
        ("+abs_ret30_10bp", abs_ret30(10)),
        ("+abs_ret30_20bp", abs_ret30(20)),
        ("+ret30_20bp_volr15", ret30_volr(20, 1.5)),
        ("+impact_p90", impact_ge(IMPACT_CUTS["p90"])),
        ("+impact_p95", impact_ge(IMPACT_CUTS["p95"])),
        ("+impact_p98", impact_ge(IMPACT_CUTS["p98"])),
    ]:
        gates.append(_and(f"champ{name}", fn))
    return gates


def _score_cell(
    raw: list[dict[str, Any]],
    *,
    position_frac: float,
    max_concurrent: int,
) -> dict[str, Any]:
    disc = [t for t in raw if t["calendar"] == "may_jul09"]
    blind = [t for t in raw if t["calendar"] == "jul10_23"]
    sized_d = _portfolio_day(
        sorted(disc, key=lambda x: (x["entry_ts"], x["symbol"])),
        position_frac=position_frac,
        max_concurrent=max_concurrent,
        cooldown_minutes=10.0,
    )
    sized_b = _portfolio_day(
        sorted(blind, key=lambda x: (x["entry_ts"], x["symbol"])),
        position_frac=position_frac,
        max_concurrent=max_concurrent,
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
    dual = bool(
        st_d.get("compound", 0) is not None
        and st_b.get("compound", 0) is not None
        and float(st_d.get("compound") or -1) > 0
        and float(st_b.get("compound") or -1) > 0
    )
    row: dict[str, Any] = {
        "n_raw": len(raw),
        "mean_capture": mean_cap,
        "may": months.get("2026-05"),
        "jun": months.get("2026-06"),
        "jul": months.get("2026-07"),
        "dual_pass": dual,
    }
    for k, v in st_d.items():
        row[f"disc_{k}"] = v
    for k, v in st_b.items():
        row[f"blind_{k}"] = v
    return row


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--enriched", default=str(DEFAULT_ENRICHED))
    ap.add_argument("--trades-root", default=str(DEFAULT_TRADES))
    ap.add_argument("--tag", default="research_am_pocket_impact_overlay")
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
    stock_1s = Path(prof["_paths"]["stock_1s_root"])
    trades_root = Path(args.trades_root)

    probes = pd.read_csv(args.enriched)
    probes["entry_ts"] = pd.to_datetime(probes["entry_ts"])
    probes = probes[probes["enrich_ok"] == True].copy()  # noqa: E712
    print(f"enriched={len(probes)} attaching impact…", flush=True)
    probes = _attach_impact(probes, stock_1s)
    probes = probes[probes["impact_ok"] == True].copy()  # noqa: E712
    probes.to_csv(out / "enriched_with_impact.csv", index=False)
    print(f"impact_ok={len(probes)}", flush=True)

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
    for _, r in probes.iterrows():
        if not _entry_ok(r, spec=ENTRY_VD_SOFT):
            # still keep; champ includes vd_soft
            pass
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

    gates = _overlay_gates(_champ_fn())
    score_rows: list[dict[str, Any]] = []
    ex = EXIT_BASE
    for gname, gfn in gates:
        raw: list[dict[str, Any]] = []
        for p in prepared:
            if not gfn(p["row"]):
                continue
            mh = float(ex["max_hold"])
            m = p["holds"] <= mh + 1e-9
            rets = p["rets"][m]
            holds = p["holds"][m]
            if len(rets) < 2:
                continue
            sim = simulate_exit(rets, holds, mode=str(ex["mode"]), params=ex)
            if not np.isfinite(sim["ret"]):
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
        cell = _score_cell(
            raw,
            position_frac=float(args.position_frac),
            max_concurrent=int(args.max_concurrent),
        )
        cell["gate"] = gname
        score_rows.append(cell)
        print(
            f"  {gname}: n={cell['n_raw']} disc={cell.get('disc_compound')} "
            f"blind={cell.get('blind_compound')} cap={cell.get('mean_capture')} "
            f"dual={cell['dual_pass']}",
            flush=True,
        )

    sb = pd.DataFrame(score_rows)
    sb.to_csv(out / "scoreboard.csv", index=False)
    base = sb[sb["gate"] == "champ"].iloc[0].to_dict() if len(sb) else {}
    improved = []
    for _, r in sb.iterrows():
        if r["gate"] == "champ":
            continue
        if (
            float(r.get("disc_compound") or -9) >= float(base.get("disc_compound") or 0)
            and float(r.get("mean_capture") or -9) >= float(base.get("mean_capture") or 0) - 1e-9
            and float(r.get("blind_compound") or -9) > 0
        ):
            improved.append(r["gate"])

    summary = {
        "protocol": "champ_pocket_plus_rare_impact_AND",
        "champ": "vd+cont60+mf100+volr12 on no_b_up enriched probes",
        "exit": EXIT_BASE,
        "impact_cuts": IMPACT_CUTS,
        "n_prepared": len(prepared),
        "champ_row": base,
        "improved_vs_champ": improved,
        "dual_pass_gates": [r["gate"] for _, r in sb.iterrows() if r["dual_pass"]],
        "l2_obi_note": (
            "No order-book/OBI dataset under /mnt/s990; repo 'L2' = hunter watchdog layer."
        ),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps({"improved": improved, "dual": summary["dual_pass_gates"]}, indent=2))
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

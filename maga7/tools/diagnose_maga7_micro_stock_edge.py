#!/usr/bin/env python3
"""Mag7 stock-only diagnostic for micro-state edges (no options).

Reads ``events_maga7.parquet`` from micro-state scan, attaches causal 1s
context (from_open, vol_z, |ret_short|), scores signed stock forward returns
at 30/60/90s under filter grids. Goal: find pockets with win≥55% and sparse
fires (≤5/day/symbol), before any option book.

Example:
  PYTHONPATH=. python -m maga7.tools.diagnose_maga7_micro_stock_edge \\
    --events-tag research_micro_state_quote_scalp_dual \\
    --tag research_maga7_micro_stock_diag
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
from maga7.common.replay import to_ny

NY = "America/New_York"
FREEZE = (
    "maga7/CONFIG/strategy_profiles/"
    "single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1.json"
)
HORIZONS = (30, 60, 90)


def _day_arrays(stock_1s: Path, sym: str, date: str) -> dict[str, Any] | None:
    day = load_stock_1s_day(stock_1s, sym, date)
    if day is None or day.empty:
        return None
    ts = pd.to_datetime(day["timestamp"])
    ts = ts.dt.tz_localize(NY) if ts.dt.tz is None else ts.dt.tz_convert(NY)
    c = day["close"].to_numpy(dtype=np.float64)
    v = day["volume"].to_numpy(dtype=np.float64) if "volume" in day.columns else np.ones(len(c))
    # open = first RTH bar ≥ 09:30
    t0930 = pd.Timestamp(f"{date} 09:30:00", tz=NY)
    i0 = int(ts.searchsorted(t0930, side="left"))
    if i0 >= len(c):
        return None
    open_px = float(c[i0])
    # causal vol ma 300s
    vol_ma = np.full(len(v), np.nan)
    csum = np.cumsum(np.concatenate([[0.0], v]))
    w = 300
    for i in range(len(v)):
        lo = max(0, i + 1 - w)
        n = i + 1 - lo
        if n >= 30:
            vol_ma[i] = (csum[i + 1] - csum[lo]) / n
    with np.errstate(divide="ignore", invalid="ignore"):
        vol_z = v / vol_ma
    return {"ts": pd.DatetimeIndex(ts), "c": c, "vol_z": vol_z, "open": open_px}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", default=FREEZE)
    ap.add_argument("--events-tag", default="research_micro_state_quote_scalp_dual")
    ap.add_argument("--tag", default="research_maga7_micro_stock_diag")
    ap.add_argument("--cell", default="s20_l60_snr2.5", help="base cell to diagnose")
    ap.add_argument("--max-events", type=int, default=0, help="0=all unique signals")
    args = ap.parse_args(argv)

    prof = load_profile(args.profile)
    paths = prof["_paths"]
    stock_1s = Path(paths["stock_1s_root"])
    results = Path(paths["results_dir"])
    ev_path = results / args.events_tag / "events_maga7.parquet"
    if not ev_path.is_file():
        raise SystemExit(f"missing {ev_path}")
    out = results / args.tag
    out.mkdir(parents=True, exist_ok=True)

    ev = pd.read_parquet(ev_path)
    ev = ev[ev["cell"].astype(str) == str(args.cell)].copy()
    ev = ev.drop_duplicates(["date", "symbol", "dir", "ts"]).reset_index(drop=True)
    if args.max_events and len(ev) > args.max_events:
        ev = ev.sample(n=int(args.max_events), random_state=0).reset_index(drop=True)
    print(f"base cell={args.cell} unique_sigs={len(ev)}", flush=True)

    cache: dict[tuple[str, str], dict[str, Any] | None] = {}
    rows: list[dict[str, Any]] = []
    for i, r in enumerate(ev.itertuples(index=False)):
        if i % 500 == 0:
            print(f"[label] {i}/{len(ev)}", flush=True)
        date, sym = str(r.date), str(r.symbol)
        key = (sym, date)
        if key not in cache:
            cache[key] = _day_arrays(stock_1s, sym, date)
        day = cache[key]
        if day is None:
            continue
        t0 = to_ny(r.ts)
        idx = int(day["ts"].searchsorted(t0, side="left"))
        if idx >= len(day["c"]) - 1:
            continue
        px = float(day["c"][idx])
        if px <= 0 or day["open"] <= 0:
            continue
        from_open = px / day["open"] - 1.0
        vz = float(day["vol_z"][idx]) if np.isfinite(day["vol_z"][idx]) else float("nan")
        # |move| over short window ending at signal
        ws = int(getattr(r, "short_sec", 20) or 20)
        j0 = max(0, idx - ws)
        ret_s = float(day["c"][idx] / day["c"][j0] - 1.0) if day["c"][j0] > 0 else 0.0
        lab: dict[str, float] = {}
        ok = True
        for h in HORIZONS:
            j = min(idx + h, len(day["c"]) - 1)
            signed = (day["c"][j] / px - 1.0) if str(r.dir) == "UP" else (1.0 - day["c"][j] / px)
            if not np.isfinite(signed):
                ok = False
                break
            lab[f"fwd{h}"] = float(signed)
        if not ok:
            continue
        rows.append(
            {
                "date": date,
                "symbol": sym,
                "dir": str(r.dir),
                "ts": str(t0),
                "snr": float(r.snr),
                "slope_s": float(r.slope_s),
                "slope_l": float(r.slope_l),
                "accel": float(r.accel),
                "from_open": float(from_open),
                "vol_z": vz,
                "abs_ret_s": abs(ret_s),
                "abs_slope_s": abs(float(r.slope_s)),
                **lab,
            }
        )
    lab_df = pd.DataFrame(rows)
    lab_df.to_parquet(out / "labeled_base.parquet", index=False)
    print(f"labeled={len(lab_df)}", flush=True)

    # Filter grid
    snr_mins = [2.5, 3.5, 5.0, 7.0]
    slope_mins = [15.0, 25.0, 40.0, 60.0]  # bp/min
    fo_mins = [0.0, 0.002, 0.005]
    vz_mins = [0.0, 1.0, 1.5, 2.0]
    ret_s_mins = [0.0, 0.001, 0.002]
    first_only_flags = [0, 1]  # 1 = first signal per date×symbol×dir

    score_rows: list[dict[str, Any]] = []
    for snr_m in snr_mins:
        for sl_m in slope_mins:
            for fo_m in fo_mins:
                for vz_m in vz_mins:
                    for rs_m in ret_s_mins:
                        for first_only in first_only_flags:
                            sub = lab_df[
                                (lab_df["snr"] >= snr_m)
                                & (lab_df["abs_slope_s"] >= sl_m)
                                & (lab_df["from_open"].abs() >= fo_m)
                                & ((lab_df["vol_z"].isna()) | (lab_df["vol_z"] >= vz_m))
                                & (lab_df["abs_ret_s"] >= rs_m)
                            ].copy()
                            if first_only:
                                sub = sub.sort_values("ts").drop_duplicates(
                                    ["date", "symbol", "dir"], keep="first"
                                )
                            if len(sub) < 30:
                                continue
                            n_dates = int(sub["date"].nunique())
                            tpd = float(len(sub) / max(n_dates, 1) / 8.0)  # per symbol-day
                            row: dict[str, Any] = {
                                "snr_min": snr_m,
                                "slope_bp_min": sl_m,
                                "from_open_min": fo_m,
                                "vol_z_min": vz_m,
                                "abs_ret_s_min": rs_m,
                                "first_only": first_only,
                                "n": int(len(sub)),
                                "n_dates": n_dates,
                                "tpd_per_sym": tpd,
                            }
                            for h in HORIZONS:
                                col = f"fwd{h}"
                                a = sub[col].to_numpy(dtype=np.float64)
                                row[f"mean{h}"] = float(a.mean())
                                row[f"win{h}"] = float((a > 0).mean())
                                row[f"med{h}"] = float(np.median(a))
                            # accept-ish: sparse + 60s edge
                            row["pass60"] = int(
                                row["win60"] >= 0.55
                                and row["mean60"] > 0
                                and row["tpd_per_sym"] <= 5.0
                                and row["n"] >= 40
                            )
                            score_rows.append(row)

    score = pd.DataFrame(score_rows)
    if len(score):
        score = score.sort_values(["pass60", "mean60", "win60"], ascending=[False, False, False])
    score.to_csv(out / "scoreboard.csv", index=False)
    passes = score[score["pass60"] == 1] if len(score) else score
    passes.to_csv(out / "pockets_pass60.csv", index=False)

    # Baseline (no extra filters)
    base = {
        "n": int(len(lab_df)),
        "tpd_per_sym": float(len(lab_df) / max(lab_df["date"].nunique(), 1) / 8.0),
    }
    for h in HORIZONS:
        a = lab_df[f"fwd{h}"].to_numpy(dtype=np.float64)
        base[f"mean{h}"] = float(a.mean())
        base[f"win{h}"] = float((a > 0).mean())

    summary = {
        "base_cell": args.cell,
        "baseline_no_extra_filter": base,
        "n_grid": int(len(score)),
        "n_pass60": int(len(passes)) if len(score) else 0,
        "top_pass60": passes.head(20).to_dict(orient="records") if len(passes) else [],
        "top_by_mean60_any": score.head(20).to_dict(orient="records") if len(score) else [],
        "verdict": "STOCK_EDGE_FOUND" if len(passes) else "NO_STOCK_POCKET",
        "note": (
            "Stock-only. pass60 = win60≥55% ∧ mean60>0 ∧ tpd/sym≤5 ∧ n≥40. "
            "If NO_STOCK_POCKET, Mag7 micro-state needs new features (VWAP/NQ/OBI), "
            "not tighter option exits."
        ),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in summary if k not in {"top_pass60", "top_by_mean60_any"}}, indent=2))
    print("\n=== baseline ===", flush=True)
    print(base, flush=True)
    print(f"\n=== pass60 pockets ({len(passes)}) ===", flush=True)
    if len(passes):
        cols = [
            "snr_min",
            "slope_bp_min",
            "from_open_min",
            "vol_z_min",
            "abs_ret_s_min",
            "first_only",
            "n",
            "tpd_per_sym",
            "mean60",
            "win60",
            "mean90",
            "win90",
        ]
        print(passes[cols].head(15).to_string(index=False), flush=True)
    else:
        print(score[["snr_min","slope_bp_min","from_open_min","vol_z_min","abs_ret_s_min","first_only","n","tpd_per_sym","mean60","win60"]].head(15).to_string(index=False), flush=True)
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

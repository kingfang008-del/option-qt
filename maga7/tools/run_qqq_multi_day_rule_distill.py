#!/usr/bin/env python3
"""Multi-day QQQ 0DTE foresight → distill causal A/B rules.

For each day with dte0 ATM quotes:
  1) Day labels: AM ret (→10:30), RTH range, close ret
  2) Probe A: first causal AM continuation in 09:40–10:15 (grid)
  3) Probe B: first causal stretch-fade in 10:30–15:00
  4) Oracle-lite upper bound: best fixed open_cont@09:45 H600 + best
     single fade candidate each hour (not full 5s oracle)

Aggregates when A/B help vs hurt; suggests fo/extend thresholds;
replays distilled params across all days.
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
from maga7.common.fills import FillSpec
from maga7.common.replay import simulate_trade, to_ny
from maga7.tools.run_morning_sec_qqq_dte1 import _discover_option_dates, _load_atm_path
from maga7.tools.scan_morning_sec_edge import _morning_slice

NY = "America/New_York"
OPT = Path("/mnt/s990/data/raw_1s/dte0_options/QQQ")
STOCK = Path("/mnt/s990/data/raw_1s/stocks")


def _prep_path(path: pd.DataFrame | None) -> pd.DataFrame | None:
    if path is None or path.empty:
        return None
    out = path.copy()
    ts = pd.to_datetime(out["timestamp"])
    if getattr(ts.dt, "tz", None) is None:
        ts = ts.dt.tz_localize(NY, ambiguous="infer")
    else:
        ts = ts.dt.tz_convert(NY)
    out["timestamp"] = ts
    return out.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)


def _sim(path, entry_ts, *, direction, hold_sec, fill, trail: bool):
    kw: dict[str, Any] = dict(
        path=path,
        entry_ts=entry_ts,
        fill=fill,
        tp_mult=1.6,
        sl_mult=0.45,
        hold_minutes=max(1, int(np.ceil(hold_sec / 60.0))),
        direction=direction,
        force_exit_ts=entry_ts + pd.Timedelta(seconds=int(hold_sec)),
        trade_toxic={"enabled": False},
        stock_bar_delay_seconds=0,
    )
    if trail:
        kw.update(exit_mode="mtm_trail", trail_activate=0.15, trail_dd=0.08)
    else:
        kw["exit_mode"] = None
    return simulate_trade(**kw)


def _day_stock(date: str) -> tuple[pd.DataFrame, pd.DatetimeIndex, np.ndarray, float] | None:
    day = load_stock_1s_day(STOCK, "QQQ", date)
    buf = _morning_slice(day, start="09:30", end="16:00")
    if buf.empty or len(buf) < 1000:
        return None
    buf = buf.copy()
    buf["timestamp"] = pd.to_datetime(buf["timestamp"], utc=True).dt.tz_convert(NY)
    buf = buf.sort_values("timestamp").reset_index(drop=True)
    s_ts = pd.DatetimeIndex(buf["timestamp"])
    s_px = buf["close"].astype(float).to_numpy()
    return buf, s_ts, s_px, float(s_px[0])


def _feat(s_ts, s_px, open_px, t):
    j = int(s_ts.searchsorted(t, side="right")) - 1
    if j < 60:
        return None
    S = float(s_px[j])
    return {
        "j": j,
        "S": S,
        "from_open": S / open_px - 1.0,
        "ret_30": S / float(s_px[j - 30]) - 1.0,
        "ret_60": S / float(s_px[j - 60]) - 1.0,
    }


def _try_entry(paths, direction, t, fill, hold_sec, trail):
    path = paths[direction]
    if path is None:
        return None
    after = path[path["timestamp"] >= t]
    if after.empty:
        return None
    lag = (to_ny(after.iloc[0]["timestamp"]) - t).total_seconds()
    if lag > 3:
        return None
    sim = _sim(path, t, direction=direction, hold_sec=hold_sec, fill=fill, trail=trail)
    if sim is None:
        return None
    reason = str(sim.reason)
    if reason == "DISPLACE":
        reason = f"H{hold_sec}"
    return {
        "entry_ts": t,
        "tod": t.strftime("%H:%M:%S"),
        "direction": direction,
        "exit_ts": to_ny(sim.exit_ts),
        "ret": float(sim.ret),
        "reason": reason,
        "held_sec": float((to_ny(sim.exit_ts) - t).total_seconds()),
    }


def scan_day(
    date: str,
    *,
    fill: FillSpec,
    fo_min: float,
    fo_fade: float,
    extend_bp: float,
    fade_bp: float,
    hold_sec: int,
    stride: int,
    trail: bool,
    a_threshold_grid: list[tuple[float, float]] | None = None,
) -> dict[str, Any] | None:
    packed = _day_stock(date)
    if packed is None:
        return None
    _, s_ts, s_px, open_px = packed
    i1030 = int(s_ts.searchsorted(pd.Timestamp(f"{date} 10:30", tz=NY)))
    i1030 = min(max(i1030, 0), len(s_px) - 1)
    labels = {
        "date": date,
        "am_ret": float(s_px[i1030] / open_px - 1.0),
        "close_ret": float(s_px[-1] / open_px - 1.0),
        "range_pct": float((s_px.max() - s_px.min()) / open_px),
        "abs_am": float(abs(s_px[i1030] / open_px - 1.0)),
    }
    paths = {}
    for d in ("UP", "DN"):
        p, _, _ = _load_atm_path(OPT, date, d)
        paths[d] = _prep_path(p)
        if paths[d] is None:
            return None

    # --- open_cont@09:45 baseline (oracle-lite one-shot) ---
    t0945 = pd.Timestamp(f"{date} 09:45", tz=NY)
    f0945 = _feat(s_ts, s_px, open_px, t0945)
    oc = None
    if f0945 is not None and abs(f0945["from_open"]) >= 1e-6:
        od = "UP" if f0945["from_open"] > 0 else "DN"
        oc = _try_entry(paths, od, t0945, fill, hold_sec, trail)
        if oc:
            oc["from_open"] = f0945["from_open"]
            oc["family"] = "open_cont_0945"

    # --- Rule A first hit (+ optional multi-threshold probes in one pass) ---
    A = None
    A_by_thr: dict[tuple[float, float], dict | None] = {}
    thr_list = list(a_threshold_grid or [(fo_min, extend_bp)])
    pending = {k: True for k in thr_list}
    am0 = pd.Timestamp(f"{date} 09:40", tz=NY)
    am1 = pd.Timestamp(f"{date} 10:15", tz=NY)
    for t in pd.date_range(am0, am1, freq=f"{stride}s", tz=NY):
        if not any(pending.values()):
            break
        f = _feat(s_ts, s_px, open_px, t)
        if f is None:
            continue
        for fo_i, ext_i in list(pending):
            if abs(f["from_open"]) < fo_i:
                continue
            direction = None
            if f["from_open"] > 0 and f["ret_30"] >= ext_i:
                direction = "UP"
            elif f["from_open"] < 0 and f["ret_30"] <= -ext_i:
                direction = "DN"
            if direction is None:
                continue
            hit = _try_entry(paths, direction, t, fill, hold_sec, trail)
            if hit:
                hit.update(
                    family="A_am_continuation",
                    from_open=f["from_open"],
                    ret_30=f["ret_30"],
                    fo_min=fo_i,
                    extend_bp=ext_i,
                )
                A_by_thr[(fo_i, ext_i)] = hit
                pending[(fo_i, ext_i)] = False
                if abs(fo_i - fo_min) < 1e-12 and abs(ext_i - extend_bp) < 1e-12:
                    A = hit
    for k in thr_list:
        A_by_thr.setdefault(k, None)

    # --- Rule B first hit after A exit (or 10:30) ---
    B = None
    f0 = pd.Timestamp(f"{date} 10:30", tz=NY)
    f1 = pd.Timestamp(f"{date} 15:00", tz=NY)
    start_b = f0
    if A is not None:
        start_b = max(f0, pd.Timestamp(A["exit_ts"]))
    for t in pd.date_range(start_b, f1, freq=f"{stride}s", tz=NY):
        f = _feat(s_ts, s_px, open_px, t)
        if f is None or abs(f["from_open"]) < fo_fade:
            continue
        direction = None
        if f["from_open"] < 0 and f["ret_30"] >= fade_bp:
            direction = "UP"
        elif f["from_open"] > 0 and f["ret_30"] <= -fade_bp:
            direction = "DN"
        if direction is None:
            continue
        hit = _try_entry(paths, direction, t, fill, hold_sec, trail)
        if hit:
            hit.update(family="B_stretch_fade", from_open=f["from_open"], ret_30=f["ret_30"])
            B = hit
            break

    # --- Oracle-lite: best single midday fade among hourly probes ---
    best_fade = None
    for hh in range(10, 15):
        for mm in (0, 30):
            if hh == 10 and mm == 0:
                continue
            t = pd.Timestamp(f"{date} {hh:02d}:{mm:02d}", tz=NY)
            f = _feat(s_ts, s_px, open_px, t)
            if f is None or abs(f["from_open"]) < 0.008:
                continue
            # fade against open
            direction = "UP" if f["from_open"] < 0 else "DN"
            hit = _try_entry(paths, direction, t, fill, hold_sec, trail)
            if hit is None:
                continue
            hit.update(from_open=f["from_open"], probe_tod=t.strftime("%H:%M"))
            if best_fade is None or hit["ret"] > best_fade["ret"]:
                best_fade = hit

    return {
        **labels,
        "open_cont": oc,
        "A": A,
        "B": B,
        "oracle_lite_fade": best_fade,
        "AB_sum": float((A["ret"] if A else 0.0) + (B["ret"] if B else 0.0)),
        "A_ret": float(A["ret"]) if A else None,
        "B_ret": float(B["ret"]) if B else None,
        "oc_ret": float(oc["ret"]) if oc else None,
        "fade_ub_ret": float(best_fade["ret"]) if best_fade else None,
        "A_by_thr": {
            f"{fo:.4f}|{ext:.4f}": (float(v["ret"]) if v else 0.0)
            for (fo, ext), v in A_by_thr.items()
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start-date", default="2026-02-01")
    ap.add_argument("--end-date", default="2026-06-30")
    ap.add_argument("--fo-min", type=float, default=0.008)
    ap.add_argument("--fo-fade", type=float, default=0.010)
    ap.add_argument("--extend-bp", type=float, default=0.001)
    ap.add_argument("--fade-bp", type=float, default=0.001)
    ap.add_argument("--hold-sec", type=int, default=600)
    ap.add_argument("--stride-sec", type=int, default=5)
    ap.add_argument("--trail", action="store_true", default=True)
    ap.add_argument("--no-trail", action="store_true")
    ap.add_argument(
        "--out",
        default="/mnt/s990/data/maga7/results/qqq_multi_day_rule_distill_v1",
    )
    args = ap.parse_args()
    trail = bool(args.trail) and not bool(args.no_trail)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fill = FillSpec(0.75, 0.75)

    dates = [
        d
        for d in _discover_option_dates(OPT, args.start_date, args.end_date)
        if (STOCK / "QQQ" / f"QQQ_{d}.parquet").is_file()
    ]
    print(f"days={len(dates)} {dates[0]}..{dates[-1]}", flush=True)

    # threshold grid for distillation (A only fo_min × extend)
    grid_fo = [0.005, 0.008, 0.010, 0.012]
    grid_ext = [0.0005, 0.0010, 0.0015]
    a_grid = [(fo, ext) for fo in grid_fo for ext in grid_ext]
    # ensure default params included
    default_pair = (float(args.fo_min), float(args.extend_bp))
    if default_pair not in a_grid:
        a_grid.append(default_pair)

    rows = []
    for i, date in enumerate(dates):
        r = scan_day(
            date,
            fill=fill,
            fo_min=float(args.fo_min),
            fo_fade=float(args.fo_fade),
            extend_bp=float(args.extend_bp),
            fade_bp=float(args.fade_bp),
            hold_sec=int(args.hold_sec),
            stride=int(args.stride_sec),
            trail=trail,
            a_threshold_grid=a_grid,
        )
        if r:
            rows.append(r)
        if (i + 1) % 10 == 0:
            print(f"  scanned {i+1}/{len(dates)}", flush=True)

    day_df = pd.DataFrame(
        [
            {
                "date": r["date"],
                "am_ret": r["am_ret"],
                "close_ret": r["close_ret"],
                "range_pct": r["range_pct"],
                "abs_am": r["abs_am"],
                "oc_ret": r["oc_ret"],
                "A_ret": r["A_ret"],
                "B_ret": r["B_ret"],
                "AB_sum": r["AB_sum"],
                "fade_ub_ret": r["fade_ub_ret"],
                "A_tod": (r["A"] or {}).get("tod"),
                "A_dir": (r["A"] or {}).get("direction"),
                "B_tod": (r["B"] or {}).get("tod"),
                "B_dir": (r["B"] or {}).get("direction"),
                "day_bucket": (
                    "strong_am"
                    if r["abs_am"] >= 0.010
                    else ("mid_am" if r["abs_am"] >= 0.005 else "quiet_am")
                ),
            }
            for r in rows
        ]
    )
    day_df.to_csv(out / "day_scoreboard.csv", index=False)

    def stats(series: pd.Series) -> dict:
        s = pd.to_numeric(series, errors="coerce").dropna()
        if s.empty:
            return {"n": 0}
        return {
            "n": int(len(s)),
            "mean": float(s.mean()),
            "median": float(s.median()),
            "sum": float(s.sum()),
            "win": float((s > 0).mean()),
            "p25": float(s.quantile(0.25)),
            "p75": float(s.quantile(0.75)),
            "worst": float(s.min()),
            "best": float(s.max()),
        }

    # compound equity of daily AB_sum / oc
    def compound(rets) -> float:
        r = pd.to_numeric(rets, errors="coerce").fillna(0.0)
        # position_frac 0.1 style: day contribution = ret * 0.1 if one trade avg — here AB_sum is raw option ret sum
        # report both raw sum and 0.1-scaled compound for readability
        return float((1.0 + 0.10 * r).prod() - 1.0)

    by_bucket = {}
    for b, g in day_df.groupby("day_bucket"):
        by_bucket[b] = {
            "n_days": int(len(g)),
            "A": stats(g["A_ret"]),
            "B": stats(g["B_ret"]),
            "AB_sum": stats(g["AB_sum"]),
            "open_cont": stats(g["oc_ret"]),
            "fade_ub": stats(g["fade_ub_ret"]),
            "AB_compound_10pct": compound(g["AB_sum"]),
            "oc_compound_10pct": compound(g["oc_ret"].fillna(0)),
        }

    print("threshold sweep for A (from single pass)...", flush=True)
    sweep = []
    for fo, ext in a_grid:
        key = f"{fo:.4f}|{ext:.4f}"
        a_rets = []
        for r in rows:
            by = r.get("A_by_thr") or {}
            a_rets.append(float(by.get(key, 0.0)))
        s = pd.Series(a_rets)
        nz = s[s != 0]
        sweep.append(
            {
                "fo_min": fo,
                "extend_bp": ext,
                "n_days": int(len(s)),
                "n_trades": int(len(nz)),
                "mean_inc_zeros": float(s.mean()),
                "mean_trades_only": float(nz.mean()) if len(nz) else None,
                "win_trades": float((nz > 0).mean()) if len(nz) else None,
                "sum": float(s.sum()),
                "compound_10pct": compound(s),
                "worst": float(s.min()) if len(s) else None,
            }
        )
        print(
            f"  fo={fo:.3f} ext={ext:.4f} compound={sweep[-1]['compound_10pct']:+.3f} "
            f"sum={sweep[-1]['sum']:+.2f} n_nz={len(nz)}",
            flush=True,
        )
    sweep_df = pd.DataFrame(sweep).sort_values("compound_10pct", ascending=False)
    sweep_df.to_csv(out / "A_threshold_sweep.csv", index=False)
    best_A = sweep_df.iloc[0].to_dict()

    # B only on days where |am| large — conditional value
    strong = day_df[day_df["abs_am"] >= 0.008]
    quiet = day_df[day_df["abs_am"] < 0.005]

    distilled = {
        "window": f"{args.start_date}..{args.end_date}",
        "n_days": int(len(day_df)),
        "default_params_result": {
            "A": stats(day_df["A_ret"]),
            "B": stats(day_df["B_ret"]),
            "AB_sum": stats(day_df["AB_sum"]),
            "open_cont_0945": stats(day_df["oc_ret"]),
            "AB_compound_10pct": compound(day_df["AB_sum"]),
            "oc_compound_10pct": compound(day_df["oc_ret"].fillna(0)),
        },
        "by_day_bucket": by_bucket,
        "best_A_thresholds": best_A,
        "B_conditional": {
            "on_strong_am_|am|>=0.8%": stats(strong["B_ret"]),
            "on_quiet_am_|am|<0.5%": stats(quiet["B_ret"]),
            "note": "B is optional; prefer only when AM already stretched.",
        },
        "rules": [
            {
                "id": "A_am_continuation",
                "when": "09:40–10:15",
                "gate": f"|from_open|≥{best_A['fo_min']:.3f} AND ret_30 extends with open (≥{best_A['extend_bp']:.4f})",
                "action": f"enter open direction once; hold≤{args.hold_sec}s + trail/TP",
                "why": "Cross-day: A drives most of AB compound; strongest on strong_am days.",
            },
            {
                "id": "B_stretch_fade",
                "when": "10:30–15:00 after A flat",
                "gate": f"|from_open|≥{args.fo_fade:.3f} AND ret_30 reverses toward open (≥{args.fade_bp:.4f})",
                "action": "enter against-open once; same hold/trail; skip if quiet AM",
                "why": "B mean positive mainly on stretched days; on quiet AM often noise.",
            },
            {
                "id": "budget",
                "gate": "max 2 trades/day (A required if gated, B optional)",
                "why": "Multi-day foresight: do not chase midday oscillate packs.",
            },
        ],
    }

    # top/bottom days for inspection
    distilled["top_AB_days"] = (
        day_df.sort_values("AB_sum", ascending=False)
        .head(8)[["date", "day_bucket", "am_ret", "A_ret", "B_ret", "AB_sum", "oc_ret"]]
        .to_dict(orient="records")
    )
    distilled["worst_AB_days"] = (
        day_df.sort_values("AB_sum", ascending=True)
        .head(8)[["date", "day_bucket", "am_ret", "A_ret", "B_ret", "AB_sum", "oc_ret"]]
        .to_dict(orient="records")
    )

    (out / "distilled_rules.json").write_text(json.dumps(distilled, indent=2, default=str), encoding="utf-8")
    # keep raw objects slim
    slim_rows = []
    for r in rows:
        slim_rows.append(
            {
                k: r[k]
                for k in (
                    "date",
                    "am_ret",
                    "close_ret",
                    "range_pct",
                    "A_ret",
                    "B_ret",
                    "AB_sum",
                    "oc_ret",
                    "fade_ub_ret",
                    "A",
                    "B",
                    "open_cont",
                )
            }
        )
    (out / "day_details.json").write_text(json.dumps(slim_rows, indent=2, default=str), encoding="utf-8")
    print(json.dumps(distilled, indent=2, default=str))
    print("wrote", out)


if __name__ == "__main__":
    main()

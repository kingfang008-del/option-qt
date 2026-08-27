#!/usr/bin/env python3
"""Dual-window: Hunt-only STOCK_REV (06-24 AMD family) vs peer3 spine.

Does not change baseline CORE exits. Variants scope STOCK_REV via
``trade.stock_rev_exit.routes=["hunt"]`` (or hunt_only=true).
"""
from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from maga7.common.config import load_profile
from maga7.common.replay import run_offline_replay

PEER3 = (
    "maga7/CONFIG/strategy_profiles/"
    "single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1.json"
)
OUT = Path("/mnt/s990/data/maga7/results/hunt_stock_rev_dual_v1")

WINDOWS = {
    "may_jul": ("2026-05-01", "2026-07-24"),
    "jan_mar": ("2026-01-02", "2026-03-31"),
}

FOCUS = {
    ("2026-06-24", "AMD"),
    ("2026-05-28", "META"),
    ("2026-06-03", "AMD"),
    ("2026-06-11", "TSLA"),
    ("2026-06-26", "AMD"),
    ("2026-07-01", "META"),
}


def _hunt_srev(
    *,
    min_hold: float,
    stock_max: float,
    opt_mtm_max: float = 0.0,
) -> dict[str, Any]:
    return {
        "stock_rev_exit": {
            "enabled": True,
            "when": "always",
            "hunt_only": True,
            "min_hold_minutes": float(min_hold),
            "stock_max": float(stock_max),
            "opt_mtm_max": float(opt_mtm_max),
        }
    }


VARIANTS: dict[str, dict[str, Any]] = {
    "baseline": {},
    # CF shortlist (Hunt mid-path): uw_m5_h10 best net; looser m0/m2 hurt META
    "hunt_uw_m5_h10": _hunt_srev(min_hold=10, stock_max=-0.005),
    "hunt_uw_m5_h15": _hunt_srev(min_hold=15, stock_max=-0.005),
    "hunt_uw_m3_h10": _hunt_srev(min_hold=10, stock_max=-0.003),
    "hunt_uw_m3_h15": _hunt_srev(min_hold=15, stock_max=-0.003),
    "hunt_uw_m2_h10": _hunt_srev(min_hold=10, stock_max=-0.002),
    "hunt_uw_m0_h10": _hunt_srev(min_hold=10, stock_max=0.0),
    # Control: same params but all routes (should regress CORE)
    "all_uw_m5_h10": {
        "stock_rev_exit": {
            "enabled": True,
            "when": "always",
            "min_hold_minutes": 10,
            "stock_max": -0.005,
            "opt_mtm_max": 0.0,
        }
    },
}


def _metrics(summary: dict, trades: pd.DataFrame) -> dict[str, Any]:
    t = trades.copy() if trades is not None else pd.DataFrame()
    reasons = (
        {str(k): int(v) for k, v in t["reason"].value_counts().items()}
        if len(t) and "reason" in t.columns
        else {}
    )
    hunt = t[t["route"].astype(str).str.lower() == "hunt"] if len(t) and "route" in t.columns else t.iloc[0:0]
    core = t[t["route"].astype(str).str.lower() != "hunt"] if len(t) and "route" in t.columns else t
    focus_rows = []
    if len(t):
        t["date"] = t["date"].astype(str)
        for d, s in sorted(FOCUS):
            sub = t[(t["date"] == d) & (t["symbol"] == s)]
            if len(sub):
                r = sub.iloc[0]
                focus_rows.append(
                    {
                        "date": d,
                        "symbol": s,
                        "route": str(r.get("route")),
                        "ret": float(r["ret"]),
                        "reason": str(r["reason"]),
                    }
                )
    return {
        "total_ret": float(summary.get("total_ret") or 0),
        "maxdd": float(summary.get("maxdd") or 0),
        "n_trades": int(summary.get("n_trades") or 0),
        "trade_win": float(summary.get("trade_win") or 0),
        "n_stock_rev": int(reasons.get("STOCK_REV", 0)),
        "hunt_sum_ret": float(hunt["ret"].sum()) if len(hunt) else 0.0,
        "hunt_n": int(len(hunt)),
        "core_sum_ret": float(core["ret"].sum()) if len(core) else 0.0,
        "core_n": int(len(core)),
        "reasons": reasons,
        "focus": focus_rows,
    }


def run_one(window: str, variant: str, overlay: dict[str, Any]) -> dict[str, Any]:
    start, end = WINDOWS[window]
    prof = deepcopy(load_profile(PEER3))
    prof["date_range"] = {"start": start, "end": end}
    trade = prof.setdefault("trade", {})
    for k, v in overlay.items():
        if isinstance(v, dict) and isinstance(trade.get(k), dict):
            trade[k] = {**trade[k], **v}
        else:
            trade[k] = v
    print(f"=== {window} / {variant} {start}..{end} ===", flush=True)
    result = run_offline_replay(prof, scheme="single")
    summary, trades = result["summary"], result["trades"]
    daily = result.get("daily")
    tag = OUT / window / f"replay__{variant}"
    tag.mkdir(parents=True, exist_ok=True)
    (tag / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    trades.to_csv(tag / "trades.csv", index=False)
    if daily is not None and len(daily):
        daily.to_csv(tag / "daily.csv", index=False)
    m = _metrics(summary, trades)
    row = {"window": window, "variant": variant, "start": start, "end": end, **m}
    print(
        f"  ret={row['total_ret']:+.3f} dd={row['maxdd']:.3f} "
        f"hunt_sum={row['hunt_sum_ret']:+.3f} (n={row['hunt_n']}) "
        f"core_sum={row['core_sum_ret']:+.3f} REV={row['n_stock_rev']}",
        flush=True,
    )
    for fr in row["focus"]:
        if str(fr.get("route", "")).lower() == "hunt" or fr["date"] == "2026-06-24":
            print(
                f"    focus {fr['date']} {fr['symbol']} {fr['route']}: "
                f"{fr['ret']:+.3f} {fr['reason']}",
                flush=True,
            )
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument(
        "--variants",
        default=",".join(VARIANTS),
        help="Comma list of variant names",
    )
    ap.add_argument("--windows", default="may_jul,jan_mar")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    variants = [v.strip() for v in str(args.variants).split(",") if v.strip()]
    windows = [w.strip() for w in str(args.windows).split(",") if w.strip()]
    rows: list[dict[str, Any]] = []
    for window in windows:
        base_ret = None
        for variant in variants:
            if variant not in VARIANTS:
                raise SystemExit(f"unknown variant {variant}")
            row = run_one(window, variant, VARIANTS[variant])
            if variant == "baseline":
                base_ret = float(row["total_ret"])
            if base_ret is not None and abs(base_ret) > 1e-12:
                row["ret_retention"] = float(row["total_ret"]) / float(base_ret)
            else:
                row["ret_retention"] = None
            rows.append(row)
    # flatten focus for csv
    flat = []
    for r in rows:
        base = {k: v for k, v in r.items() if k not in {"focus", "reasons"}}
        flat.append(base)
    sb = pd.DataFrame(flat)
    sb.to_csv(out / "scoreboard.csv", index=False)
    (out / "scoreboard.json").write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")

    # Dual verdict vs baseline
    verdict: dict[str, Any] = {"rule": "hunt_only STOCK_REV vs peer3", "candidates": []}
    for variant in variants:
        if variant == "baseline":
            continue
        mj = next((r for r in rows if r["window"] == "may_jul" and r["variant"] == variant), None)
        jm = next((r for r in rows if r["window"] == "jan_mar" and r["variant"] == variant), None)
        if not mj:
            continue
        keep_mj = float(mj.get("ret_retention") or 0)
        keep_jm = float(jm.get("ret_retention") or 0) if jm else None
        amd = next(
            (f for f in mj.get("focus") or [] if f["date"] == "2026-06-24" and f["symbol"] == "AMD"),
            None,
        )
        ok = keep_mj >= 0.95 and (keep_jm is None or keep_jm >= 0.95)
        verdict["candidates"].append(
            {
                "variant": variant,
                "may_jul_ret": mj["total_ret"],
                "may_jul_keep": keep_mj,
                "jan_mar_keep": keep_jm,
                "hunt_sum_delta_vs_base": None,
                "amd_0624": amd,
                "dual_pass": ok,
            }
        )
    base_mj = next(r for r in rows if r["window"] == "may_jul" and r["variant"] == "baseline")
    for c in verdict["candidates"]:
        mj = next(r for r in rows if r["window"] == "may_jul" and r["variant"] == c["variant"])
        c["hunt_sum_delta_vs_base"] = float(mj["hunt_sum_ret"]) - float(base_mj["hunt_sum_ret"])
        c["core_sum_delta_vs_base"] = float(mj["core_sum_ret"]) - float(base_mj["core_sum_ret"])
    (out / "verdict.json").write_text(json.dumps(verdict, indent=2, default=str), encoding="utf-8")
    print("\n=== VERDICT ===", flush=True)
    for c in verdict["candidates"]:
        print(
            f"{c['variant']:20s} keep_mj={c['may_jul_keep']:.3f} keep_jm={c['jan_mar_keep']} "
            f"huntΔ={c['hunt_sum_delta_vs_base']:+.3f} coreΔ={c['core_sum_delta_vs_base']:+.3f} "
            f"amd={c['amd_0624']} dual={'PASS' if c['dual_pass'] else 'FAIL'}",
            flush=True,
        )
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

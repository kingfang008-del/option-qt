#!/usr/bin/env python3
"""CORE (Rule-A, 10:30+) exit stress gate — offline PASS/FAIL before live exit changes.

**Not AM sleeve.** Freezes peer3 research_baseline entries via ``run_offline_replay``
and overlays exit-only trade knobs (giveback extend, mtm trail, stale cut, …).

Dual windows: may_jul + jan_mar (same convention as giveback/lock-profit ablations).
Promote only if both windows keep ret and maxDD does not worsen beyond slack.

Example:
  PYTHONPATH=. python -m maga7.tools.run_exit_stress_gate \\
    --tag research_core_exit_stress_gate_20260728 \\
    --variants baseline,strip_gb,gb08_p10,trail30_dd20
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

WINDOWS = {
    "may_jul": ("2026-05-01", "2026-07-20"),
    "jan_mar": ("2026-01-02", "2026-03-31"),
}

# Exit-only overlays on current CORE profile (already ships gb12_p15).
VARIANTS: dict[str, dict[str, Any]] = {
    "baseline": {},
    # Control: remove giveback extend gate (prove current default matters).
    "strip_gb": {
        "hold_extend_max_giveback": None,
        "hold_extend_giveback_min_peak": None,
    },
    # Tighter giveback vs current gb12/p15.
    "gb08_p10": {
        "hold_extend_max_giveback": 0.08,
        "hold_extend_giveback_min_peak": 0.10,
    },
    "gb08_p12": {
        "hold_extend_max_giveback": 0.08,
        "hold_extend_giveback_min_peak": 0.12,
    },
    "gb10_p15": {
        "hold_extend_max_giveback": 0.10,
        "hold_extend_giveback_min_peak": 0.15,
    },
    "gb12_p10": {
        "hold_extend_max_giveback": 0.12,
        "hold_extend_giveback_min_peak": 0.10,
    },
    # Soft lock after peak (keep hold_extend outer rails).
    "trail30_dd20": {
        "exit_mode": "hold_extend+mtm_trail",
        "trail_activate": 0.30,
        "trail_dd": 0.20,
    },
    "trail25_dd15": {
        "exit_mode": "hold_extend+mtm_trail",
        "trail_activate": 0.25,
        "trail_dd": 0.15,
    },
    "trail20_dd12": {
        "exit_mode": "hold_extend+mtm_trail",
        "trail_activate": 0.20,
        "trail_dd": 0.12,
    },
    # Require green MTM to earn T30→T45.
    "ext_mtm10": {"hold_extend_mtm_min": 0.10},
    "ext_mtm15": {"hold_extend_mtm_min": 0.15},
    # Mid-hold stale loser cut.
    "stale15": {
        "stale_cut_minutes": 15,
        "stale_cut_mtm_max": 0.0,
        "stale_cut_stock_max": 0.0,
    },
}


def _reason_counts(trades: pd.DataFrame) -> dict[str, int]:
    if trades is None or trades.empty or "reason" not in trades.columns:
        return {}
    return {str(k): int(v) for k, v in trades["reason"].value_counts().items()}


def _loss_stats(trades: pd.DataFrame) -> dict[str, float]:
    if trades is None or trades.empty:
        return {
            "n_loss": 0,
            "loss_sum_ret": 0.0,
            "n_sl": 0,
            "sl_sum_ret": 0.0,
            "n_deep": 0,
            "deep_sum_ret": 0.0,
        }
    ret = pd.to_numeric(trades["ret"], errors="coerce")
    reason = trades["reason"].astype(str) if "reason" in trades.columns else pd.Series([""] * len(trades))
    loss = ret <= 0
    sl = reason.str.upper().eq("SL")
    deep = ret <= -0.15
    return {
        "n_loss": int(loss.sum()),
        "loss_sum_ret": float(ret[loss].sum()) if loss.any() else 0.0,
        "n_sl": int(sl.sum()),
        "sl_sum_ret": float(ret[sl].sum()) if sl.any() else 0.0,
        "n_deep": int(deep.sum()),
        "deep_sum_ret": float(ret[deep].sum()) if deep.any() else 0.0,
    }


def _apply_overlay(trade: dict[str, Any], overlay: dict[str, Any]) -> None:
    for k, v in overlay.items():
        if v is None:
            trade[k] = None
        elif isinstance(v, dict) and isinstance(trade.get(k), dict):
            trade[k] = {**trade[k], **v}
        else:
            trade[k] = v


def run_one(
    window: str,
    variant: str,
    overlay: dict[str, Any],
    *,
    out_root: Path,
    profile_path: str,
) -> dict[str, Any]:
    start, end = WINDOWS[window]
    prof = deepcopy(load_profile(profile_path))
    # CORE only: drop AM satellite sleeves so replay mirrors Rule-A book.
    for sleeve_key in ("am_pulse", "am_pulse_extension", "morning_sleeve", "pm_fade"):
        if sleeve_key in prof:
            block = prof.get(sleeve_key)
            if isinstance(block, dict):
                block = dict(block)
                block["enabled"] = False
                prof[sleeve_key] = block
    prof["date_range"] = {"start": start, "end": end}
    trade = prof.setdefault("trade", {})
    _apply_overlay(trade, overlay)

    print(f"=== CORE {window} / {variant} {start}..{end} ===", flush=True)
    result = run_offline_replay(prof, scheme="single")
    summary = result["summary"]
    trades = result["trades"]
    daily = result.get("daily")
    worst_day = None
    if daily is not None and len(daily) and "day_ret" in daily.columns:
        worst_day = float(pd.to_numeric(daily["day_ret"], errors="coerce").min())

    out = out_root / window / f"replay__{variant}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    trades.to_csv(out / "trades.csv", index=False)
    if daily is not None and len(daily):
        daily.to_csv(out / "daily.csv", index=False)

    reasons = _reason_counts(trades)
    loss = _loss_stats(trades)
    row: dict[str, Any] = {
        "sleeve": "CORE",
        "window": window,
        "variant": variant,
        "start": start,
        "end": end,
        "total_ret": float(summary["total_ret"]),
        "maxdd": float(summary["maxdd"]),
        "n_trades": int(summary["n_trades"]),
        "trade_win": float(summary.get("trade_win") or 0.0),
        "worst_day": worst_day,
        "n_tp": int(reasons.get("TP", 0)),
        "n_sl": int(reasons.get("SL", 0)),
        "n_t30": int(reasons.get("T+30", 0)),
        "n_t45": int(reasons.get("T+45", 0)),
        "n_trail": int(reasons.get("TRAIL", 0) + reasons.get("TRAIL_LADDER", 0)),
        **loss,
        "reasons": reasons,
    }
    print(
        f"  ret={row['total_ret']:+.3f} dd={row['maxdd']:.3f} n={row['n_trades']} "
        f"win={row['trade_win']:.3f} "
        f"TP/SL/T30/T45={row['n_tp']}/{row['n_sl']}/{row['n_t30']}/{row['n_t45']} "
        f"deep_sum={row['deep_sum_ret']:+.3f}",
        flush=True,
    )
    return row


def _verdict_dual(
    edf: pd.DataFrame,
    *,
    wins: list[str],
    min_retain_pass: float,
    min_retain_weak: float,
    maxdd_slack: float,
    deep_improve: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    base = {
        r["window"]: r
        for _, r in edf[edf["variant"] == "baseline"].iterrows()
    }
    score: list[dict[str, Any]] = []
    for variant, g in edf.groupby("variant"):
        g = g.set_index("window")
        if not set(wins).issubset(set(g.index)):
            continue
        retains = []
        dd_deltas = []
        deep_improves = []
        dual_ok_ret = True
        for w in wins:
            br = base[w]
            vr = g.loc[w]
            b_ret = float(br["total_ret"])
            v_ret = float(vr["total_ret"])
            keep = (1.0 + v_ret) / (1.0 + b_ret) if (1.0 + b_ret) != 0 else float("nan")
            retains.append(keep)
            dd_deltas.append(float(vr["maxdd"]) - float(br["maxdd"]))
            b_deep = float(br["deep_sum_ret"])
            v_deep = float(vr["deep_sum_ret"])
            if b_deep < -1e-9:
                deep_improves.append((v_deep - b_deep) / abs(b_deep))
            else:
                deep_improves.append(0.0)
            if v_ret <= 0:
                dual_ok_ret = False

        min_ret = float(min(retains))
        min_dd_delta = float(min(dd_deltas))  # worst (most negative) dd change
        min_deep_imp = float(min(deep_improves))
        dd_ok_pass = min_dd_delta >= -1e-9
        dd_ok_weak = min_dd_delta >= -abs(maxdd_slack)
        deep_ok = min_deep_imp >= deep_improve - 1e-9 or variant == "baseline"

        if variant == "baseline":
            verdict = "PASS"
        elif not dual_ok_ret:
            verdict = "FAIL"
        elif min_ret >= min_retain_pass and dd_ok_pass and deep_ok:
            verdict = "PASS"
        elif min_ret >= min_retain_weak and dd_ok_weak:
            verdict = "WEAK"
        else:
            verdict = "FAIL"

        score.append(
            {
                "variant": variant,
                "verdict": verdict,
                "min_retention": min_ret,
                "mean_retention": float(sum(retains) / len(retains)),
                "min_dd_delta": min_dd_delta,
                "min_deep_improve": min_deep_imp,
                **{
                    f"{w}_ret": float(g.loc[w, "total_ret"])
                    for w in wins
                },
                **{
                    f"{w}_maxdd": float(g.loc[w, "maxdd"])
                    for w in wins
                },
                **{
                    f"{w}_deep_sum": float(g.loc[w, "deep_sum_ret"])
                    for w in wins
                },
            }
        )
    score.sort(
        key=lambda r: (
            {"PASS": 0, "WEAK": 1, "FAIL": 2}.get(r["verdict"], 9),
            -r["min_retention"],
            -r.get("may_jul_ret", 0.0),
        )
    )
    summary = {
        "promote": [r["variant"] for r in score if r["verdict"] == "PASS" and r["variant"] != "baseline"],
        "weak_candidates": [r["variant"] for r in score if r["verdict"] == "WEAK"],
        "fail": [r["variant"] for r in score if r["verdict"] == "FAIL"],
    }
    return score, summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", default=PEER3)
    ap.add_argument("--tag", default="research_core_exit_stress_gate_20260728")
    ap.add_argument("--windows", default="may_jul,jan_mar")
    ap.add_argument(
        "--variants",
        default=(
            "baseline,strip_gb,gb08_p10,gb08_p12,gb10_p15,gb12_p10,"
            "trail30_dd20,trail25_dd15,ext_mtm10,stale15"
        ),
    )
    ap.add_argument("--min-retain-pass", type=float, default=0.95)
    ap.add_argument("--min-retain-weak", type=float, default=0.85)
    ap.add_argument("--maxdd-slack", type=float, default=0.02)
    ap.add_argument(
        "--deep-improve",
        type=float,
        default=0.05,
        help="Min relative improvement on deep-loss sum (ret<=-15%) for PASS",
    )
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)

    wins = [w.strip() for w in args.windows.split(",") if w.strip()]
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    for w in wins:
        if w not in WINDOWS:
            raise SystemExit(f"unknown window {w}; choose from {list(WINDOWS)}")
    for v in variants:
        if v not in VARIANTS:
            raise SystemExit(f"unknown variant {v}; choose from {list(VARIANTS)}")

    prof0 = load_profile(args.profile)
    out = Path(args.out) if args.out else Path(prof0["_paths"]["results_dir"]) / args.tag
    out.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for window in wins:
        for variant in variants:
            rows.append(
                run_one(
                    window,
                    variant,
                    VARIANTS[variant],
                    out_root=out,
                    profile_path=args.profile,
                )
            )

    # retention vs baseline per window
    base_ret = {r["window"]: float(r["total_ret"]) for r in rows if r["variant"] == "baseline"}
    enriched = []
    for r in rows:
        b = base_ret.get(r["window"])
        keep = None
        if b is not None and (1.0 + b) != 0:
            keep = (1.0 + float(r["total_ret"])) / (1.0 + b)
        enriched.append(
            {
                **{k: v for k, v in r.items() if k != "reasons"},
                "ret_retention": keep,
                "ret_vs_baseline": (float(r["total_ret"]) - b) if b is not None else None,
                "reasons": r["reasons"],
            }
        )

    edf = pd.DataFrame([{k: v for k, v in r.items() if k != "reasons"} for r in enriched])
    edf.to_csv(out / "scoreboard.csv", index=False)
    (out / "scoreboard.json").write_text(
        json.dumps(enriched, indent=2, default=str), encoding="utf-8"
    )

    score, promo = _verdict_dual(
        edf,
        wins=wins,
        min_retain_pass=args.min_retain_pass,
        min_retain_weak=args.min_retain_weak,
        maxdd_slack=args.maxdd_slack,
        deep_improve=args.deep_improve,
    )
    pd.DataFrame(score).to_csv(out / "verdicts.csv", index=False)

    summary = {
        "tag": args.tag,
        "sleeve": "CORE",
        "note": (
            "CORE Rule-A 10:30+ exit stress gate (NOT AM pulse). "
            "Promote only PASS; WEAK may be research overlay only."
        ),
        "profile": args.profile,
        "windows": {w: list(WINDOWS[w]) for w in wins},
        "gates": {
            "min_retain_pass": args.min_retain_pass,
            "min_retain_weak": args.min_retain_weak,
            "maxdd_slack": args.maxdd_slack,
            "deep_improve": args.deep_improve,
        },
        "verdicts": score,
        **promo,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    print("\n=== CORE exit stress gate ===", flush=True)
    print(pd.DataFrame(score).to_string(index=False, float_format=lambda x: f"{x:.3f}"), flush=True)
    print(f"\npromote={promo['promote']} weak={promo['weak_candidates']}", flush=True)
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

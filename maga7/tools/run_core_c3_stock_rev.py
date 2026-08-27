#!/usr/bin/env python3
"""CORE C3 — overlay L3 STOCK_REV on current research_baseline (path failure exit).

Does **not** invent a new FailureDetector. Re-accepts the July L3 champion
``wash_m3_uw_h10`` (and optional alt ``uw_m3_h15``) on today's S1+morph+gb12
spine. Same fires as Rule-A; only hold can flatten earlier via STOCK_REV.

Corpus / windows match C1–C2:
  weak   2026-01-02 .. 2026-03-31
  strong 2026-04-01 .. 2026-07-21

Control is a **fresh current-profile replay** (empty overlay), not the older
S1 accept book — morph/event stack has moved since S1, so S1 keep>1 is not
path adaptation.

PASS (adaptability C3 bar, tighter than L3's 0.85):
  strong keep >= 0.95 vs current baseline
  weak   keep >= 0.95 vs current baseline
  tail (deep_sum_ret) improved vs current baseline
  n_stock_rev >= 1 (not fake adapt)
  n_trades within ``max_n_delta`` of current baseline (exit-only)

FAIL → keep T30 rails; do not wire; do not add morph BLOCKs.

Example:
  PYTHONPATH=. python -m maga7.tools.run_core_c3_stock_rev \\
    --tag research_core_c3_stock_rev
"""
from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from maga7.common.config import load_profile
from maga7.common.replay import run_offline_replay

PROFILE = (
    "maga7/CONFIG/strategy_profiles/"
    "single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1.json"
)
S1_ROOT = Path(
    "/mnt/s990/data/maga7/results/s1_research_baseline_accept_apr_jul_jan_mar_v1"
)
S1_ARMS = {"weak": "weak_S1", "strong": "strong_S1"}
WINDOWS = {
    "weak": ("2026-01-02", "2026-03-31"),
    "strong": ("2026-04-01", "2026-07-21"),
}
CLOCK_REASONS = {"T+30", "T+45", "TIME", "HOLD", "HOLD_EXTEND", "EXTEND"}
CHAMPION = "wash_m3_uw_h10"
VARIANTS: dict[str, dict[str, Any]] = {
    CHAMPION: {
        "stock_rev_exit": {
            "enabled": True,
            "when": "mixed_wash_up",
            "min_hold_minutes": 10,
            "stock_max": -0.003,
            "opt_mtm_max": 0.0,
            "washout_breadth_min": 3,
            "note": "L3 champion overlay on current research_baseline (C3).",
        }
    },
    "uw_m3_h15": {
        "stock_rev_exit": {
            "enabled": True,
            "when": "always",
            "min_hold_minutes": 15,
            "stock_max": -0.003,
            "opt_mtm_max": 0.0,
            "note": "L3 alt overlay (always / 15m). C3 sensitivity.",
        }
    },
}
AM_SLEEVES = ("am_pulse", "am_pulse_extension", "morning_sleeve", "pm_fade", "am_v2")


def _apply_overlay(trade: dict[str, Any], overlay: dict[str, Any]) -> None:
    for k, v in overlay.items():
        if v is None:
            trade[k] = None
        elif isinstance(v, dict) and isinstance(trade.get(k), dict):
            trade[k] = {**trade[k], **v}
        else:
            trade[k] = v


def _disable_am(prof: dict[str, Any]) -> None:
    for key in AM_SLEEVES:
        block = prof.get(key)
        if isinstance(block, dict):
            block = dict(block)
            block["enabled"] = False
            prof[key] = block


def reason_stats(trades: pd.DataFrame) -> dict[str, Any]:
    if trades is None or trades.empty or "reason" not in trades.columns:
        return {
            "reasons": {},
            "n_clock": 0,
            "clock_share": None,
            "n_stock_rev": 0,
            "n_sl": 0,
            "n_tp": 0,
            "n_trade_tox": 0,
            "n_deep": 0,
            "deep_sum_ret": 0.0,
            "loss_sum_ret": 0.0,
        }
    vc = {str(k): int(v) for k, v in trades["reason"].value_counts().items()}
    n = int(sum(vc.values()))
    n_clock = sum(v for k, v in vc.items() if k in CLOCK_REASONS or str(k).startswith("T+"))
    ret = pd.to_numeric(trades["ret"], errors="coerce")
    deep = ret <= -0.15
    loss = ret <= 0
    return {
        "reasons": vc,
        "n_clock": int(n_clock),
        "clock_share": float(n_clock / n) if n else None,
        "n_stock_rev": int(vc.get("STOCK_REV", 0)),
        "n_sl": int(vc.get("SL", 0)),
        "n_tp": int(vc.get("TP", 0)),
        "n_trade_tox": int(vc.get("TRADE_TOX", 0) + vc.get("TOXIC", 0)),
        "n_deep": int(deep.sum()),
        "deep_sum_ret": float(ret[deep].sum()) if deep.any() else 0.0,
        "loss_sum_ret": float(ret[loss].sum()) if loss.any() else 0.0,
    }


def equity_from_trades(trades: pd.DataFrame) -> dict[str, Any]:
    if trades is None or trades.empty:
        return {"n_trades": 0, "total_ret": 0.0, "maxdd": 0.0, "trade_win": None}
    eq = 100.0
    peak = 100.0
    maxdd = 0.0
    rets = []
    for row in trades.itertuples(index=False):
        r = float(pd.to_numeric(getattr(row, "ret"), errors="coerce"))
        if not np.isfinite(r):
            continue
        sf = float(pd.to_numeric(getattr(row, "size_frac", 0.2), errors="coerce") or 0.2)
        eq *= 1.0 + sf * r
        peak = max(peak, eq)
        if peak > 0:
            maxdd = min(maxdd, eq / peak - 1.0)
        rets.append(r)
    rr = np.asarray(rets, dtype=float)
    return {
        "n_trades": int(len(rr)),
        "total_ret": float(eq / 100.0 - 1.0),
        "maxdd": float(maxdd),
        "trade_win": float((rr > 0).mean()) if len(rr) else None,
    }


def keep_ratio(variant_ret: float, base_ret: float) -> float:
    den = 1.0 + float(base_ret)
    if den == 0:
        return 0.0
    return float((1.0 + float(variant_ret)) / den)


def verdict_c3(
    *,
    strong_keep: float,
    weak_keep: float,
    strong_deep: float,
    weak_deep: float,
    base_strong_deep: float,
    base_weak_deep: float,
    n_stock_rev: int,
    n_delta_max: int = 0,
    max_n_delta: int = 2,
    min_keep: float = 0.95,
) -> dict[str, Any]:
    """PASS only if both windows keep, tail improves, and STOCK_REV actually fires."""
    tail_ok = (float(strong_deep) + float(weak_deep)) > (
        float(base_strong_deep) + float(base_weak_deep)
    ) + 1e-12
    # deep_sum is negative; "improved" = algebraically larger (less negative).
    fired = int(n_stock_rev) >= 1
    keep_ok = float(strong_keep) >= float(min_keep) and float(weak_keep) >= float(min_keep)
    n_ok = int(n_delta_max) <= int(max_n_delta)
    passed = bool(keep_ok and tail_ok and fired and n_ok)
    reason = "pass"
    if not n_ok:
        reason = "entry_set_drift"
    elif not fired:
        reason = "no_stock_rev_fires"
    elif not keep_ok:
        reason = "keep_below_bar"
    elif not tail_ok:
        reason = "tail_not_improved"
    return {
        "pass": passed,
        "reason": reason,
        "keep_ok": keep_ok,
        "tail_ok": tail_ok,
        "fired": fired,
        "n_ok": n_ok,
    }


def load_s1_baseline(window: str) -> dict[str, Any]:
    arm = S1_ARMS[window]
    tpath = S1_ROOT / arm / "trades.csv"
    spath = S1_ROOT / arm / "summary.json"
    if not tpath.exists():
        raise SystemExit(f"missing S1 baseline {tpath}")
    trades = pd.read_csv(tpath)
    rs = reason_stats(trades)
    if spath.exists():
        summary = json.loads(spath.read_text(encoding="utf-8"))
        total_ret = float(summary["total_ret"])
        maxdd = float(summary["maxdd"])
        n_trades = int(summary["n_trades"])
        trade_win = float(summary.get("trade_win") or 0.0)
    else:
        eq = equity_from_trades(trades)
        total_ret = float(eq["total_ret"])
        maxdd = float(eq["maxdd"])
        n_trades = int(eq["n_trades"])
        trade_win = float(eq["trade_win"] or 0.0)
    start, end = WINDOWS[window]
    return {
        "window": window,
        "variant": "baseline_s1",
        "start": start,
        "end": end,
        "total_ret": total_ret,
        "maxdd": maxdd,
        "n_trades": n_trades,
        "trade_win": trade_win,
        **{k: v for k, v in rs.items() if k != "reasons"},
        "reasons": rs["reasons"],
        "keep": 1.0,
    }


def run_overlay(
    window: str,
    variant: str,
    overlay: dict[str, Any],
    *,
    out_root: Path,
    profile_path: str,
) -> dict[str, Any]:
    start, end = WINDOWS[window]
    prof = deepcopy(load_profile(profile_path))
    _disable_am(prof)
    prof["date_range"] = {"start": start, "end": end}
    trade = prof.setdefault("trade", {})
    _apply_overlay(trade, overlay)
    print(f"=== C3 {window} / {variant} {start}..{end} ===", flush=True)
    result = run_offline_replay(prof, scheme="single")
    summary = result["summary"]
    trades = result["trades"]
    daily = result.get("daily")
    tag = out_root / window / f"replay__{variant}"
    tag.mkdir(parents=True, exist_ok=True)
    (tag / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    trades.to_csv(tag / "trades.csv", index=False)
    if daily is not None and len(daily):
        daily.to_csv(tag / "daily.csv", index=False)
    rs = reason_stats(trades)
    row = {
        "window": window,
        "variant": variant,
        "start": start,
        "end": end,
        "total_ret": float(summary["total_ret"]),
        "maxdd": float(summary["maxdd"]),
        "n_trades": int(summary["n_trades"]),
        "n_signals_topk": int(summary.get("n_signals_topk") or 0),
        "trade_win": float(summary.get("trade_win") or 0.0),
        **{k: v for k, v in rs.items() if k != "reasons"},
        "reasons": rs["reasons"],
    }
    print(
        f"  ret={row['total_ret']:+.3f} dd={row['maxdd']:.3f} n={row['n_trades']} "
        f"clock={row['clock_share']} REV={row['n_stock_rev']} "
        f"deep={row['deep_sum_ret']:+.3f}",
        flush=True,
    )
    return row


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", default=PROFILE)
    ap.add_argument("--tag", default="research_core_c3_stock_rev")
    ap.add_argument(
        "--variants",
        default=CHAMPION,
        help="comma list or 'all' (default: L3 champion wash_m3_uw_h10)",
    )
    ap.add_argument("--windows", default="weak,strong")
    ap.add_argument("--min-keep", type=float, default=0.95)
    ap.add_argument(
        "--max-n-delta",
        type=int,
        default=2,
        help="max |n_trades overlay − current baseline|; larger = entry-set drift",
    )
    args = ap.parse_args(argv)

    profile = load_profile(args.profile)
    out = Path(profile["_paths"]["results_dir"]) / args.tag
    out.mkdir(parents=True, exist_ok=True)

    wins = [w.strip() for w in args.windows.split(",") if w.strip()]
    if args.variants.strip() == "all":
        variants = list(VARIANTS.keys())
    else:
        variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    unknown = [v for v in variants if v not in VARIANTS]
    if unknown:
        raise SystemExit(f"unknown variants {unknown}; known={list(VARIANTS)}")

    rows: list[dict[str, Any]] = []
    s1_ref: dict[str, dict[str, Any]] = {}
    for w in wins:
        ref = load_s1_baseline(w)
        s1_ref[w] = ref
        rows.append(ref)
        print(
            f"=== C3 {w} / baseline_s1 (reference only) ret={ref['total_ret']:+.3f} "
            f"n={ref['n_trades']} clock={ref['clock_share']} ===",
            flush=True,
        )

    current: dict[str, dict[str, Any]] = {}
    for w in wins:
        row = run_overlay(w, "baseline", {}, out_root=out, profile_path=args.profile)
        row["keep"] = 1.0
        current[w] = row
        rows.append(row)

    for variant in variants:
        for w in wins:
            row = run_overlay(
                w, variant, VARIANTS[variant], out_root=out, profile_path=args.profile
            )
            b = current[w]
            row["keep"] = keep_ratio(row["total_ret"], b["total_ret"])
            row["maxdd_delta"] = float(row["maxdd"] - b["maxdd"])
            row["clock_delta"] = (
                None
                if row["clock_share"] is None or b["clock_share"] is None
                else float(row["clock_share"] - b["clock_share"])
            )
            row["deep_delta"] = float(row["deep_sum_ret"] - b["deep_sum_ret"])
            row["n_delta"] = int(row["n_trades"] - b["n_trades"])
            rows.append(row)

    sb = pd.DataFrame(rows)
    sb.to_csv(out / "scoreboard.csv", index=False)

    adopted: list[dict[str, Any]] = []
    for variant in variants:
        by_w = {
            r["window"]: r
            for r in rows
            if r["variant"] == variant
        }
        if any(w not in by_w for w in wins):
            continue
        n_rev = sum(int(by_w[w]["n_stock_rev"] or 0) for w in wins)
        n_delta_max = max(abs(int(by_w[w].get("n_delta") or 0)) for w in wins)
        strong = by_w.get("strong") or by_w[wins[-1]]
        weak = by_w.get("weak") or by_w[wins[0]]
        b_strong = current.get("strong") or current[wins[-1]]
        b_weak = current.get("weak") or current[wins[0]]
        v = verdict_c3(
            strong_keep=float(strong["keep"]),
            weak_keep=float(weak["keep"]),
            strong_deep=float(strong["deep_sum_ret"]),
            weak_deep=float(weak["deep_sum_ret"]),
            base_strong_deep=float(b_strong["deep_sum_ret"]),
            base_weak_deep=float(b_weak["deep_sum_ret"]),
            n_stock_rev=n_rev,
            n_delta_max=int(n_delta_max),
            max_n_delta=int(args.max_n_delta),
            min_keep=float(args.min_keep),
        )
        adopted.append(
            {
                "variant": variant,
                **v,
                "strong_keep": float(strong["keep"]),
                "weak_keep": float(weak["keep"]),
                "strong_ret": float(strong["total_ret"]),
                "weak_ret": float(weak["total_ret"]),
                "strong_clock": strong.get("clock_share"),
                "weak_clock": weak.get("clock_share"),
                "n_stock_rev": int(n_rev),
                "n_delta_max": int(n_delta_max),
                "strong_n": int(strong["n_trades"]),
                "weak_n": int(weak["n_trades"]),
                "base_strong_n": int(b_strong["n_trades"]),
                "base_weak_n": int(b_weak["n_trades"]),
                "strong_deep": float(strong["deep_sum_ret"]),
                "weak_deep": float(weak["deep_sum_ret"]),
                "base_strong_deep": float(b_strong["deep_sum_ret"]),
                "base_weak_deep": float(b_weak["deep_sum_ret"]),
                "weak_n_stock_rev": int(weak["n_stock_rev"] or 0),
                "strong_n_stock_rev": int(strong["n_stock_rev"] or 0),
            }
        )
    passed = [a for a in adopted if a["pass"]]
    promote = f"C3_{passed[0]['variant']}" if passed else "NONE"
    summary = {
        "protocol": "core_c3_stock_rev",
        "promotion_mark": "path_failure_exit_stock_rev_overlay",
        "control": "current_profile_replay_baseline",
        "s1_reference": "s1_research_baseline_accept_apr_jul_jan_mar_v1",
        "profile": args.profile,
        "pass_rule": (
            f"vs current baseline: strong keep>={args.min_keep} AND weak keep>={args.min_keep} "
            f"AND combined deep_sum_ret improved AND n_stock_rev>=1 "
            f"AND |n_trades delta|<={args.max_n_delta}"
        ),
        "note": (
            "Do not invent FailureDetector. Overlay existing L3 STOCK_REV. "
            "Calendar is eval-only. AM sleeves forced off. "
            "S1 book is reference only — verdict uses current-profile baseline. "
            "L3 Jul-22 retain~0.91 vs older peer3 is NOT this accept."
        ),
        "s1_reference_rows": {
            k: {kk: vv for kk, vv in v.items() if kk != "reasons"} for k, v in s1_ref.items()
        },
        "baseline": {
            k: {kk: vv for kk, vv in v.items() if kk != "reasons"} for k, v in current.items()
        },
        "variants": adopted,
        "promote": promote,
        "pass": bool(promote != "NONE"),
        "next_step": (
            "wire_stock_rev_on_research_baseline"
            if promote != "NONE"
            else "keep_t30_rails_no_new_morph"
        ),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    lines = [
        "# CORE C3 — STOCK_REV path-failure overlay",
        "",
        "- action: **hold-path overlay only** (no Rule-A / no new BLOCK / no FailureDetector)",
        f"- champion: `{CHAMPION}` on current `extend_mtm_full_day_peer3_v1`",
        f"- promote: **{promote}**",
        f"- pass: **{summary['pass']}**",
        "",
        "## Scoreboard",
        "",
    ]
    show = sb.drop(columns=["reasons"], errors="ignore")
    try:
        lines.append(show.to_markdown(index=False))
    except Exception:
        lines.append(show.to_string(index=False))
    if promote != "NONE":
        best = passed[0]
        lines += [
            "",
            "## 结论",
            "",
            f"**C3 PASS** → `{best['variant']}` vs **current-profile baseline** "
            f"strong keep={best['strong_keep']:.3f} · weak keep={best['weak_keep']:.3f} · "
            f"STOCK_REV n={best['n_stock_rev']}（弱窗 {best['weak_n_stock_rev']} / 强窗 {best['strong_n_stock_rev']}）。",
            "可写入 research_baseline `trade.stock_rev_exit`（enabled）；生产 freeze 另闸。",
        ]
    else:
        why = adopted[0]["reason"] if adopted else "no_variant"
        lines += [
            "",
            "## 结论",
            "",
            f"**C3 FAIL** ({why}) — 保持 T30 rails；不接线；不加 morph BLOCK。",
            "对照是当前 profile 重放 baseline，不是更旧的 S1 成交簿。",
        ]
    (out / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")

    print(show.to_string(index=False), flush=True)
    print(json.dumps({"promote": promote, "pass": summary["pass"], "variants": adopted}, indent=2))
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""CORE C4 — entry validator bakeoff on actual CORE TopK seats.

Does **not** train or wire ML (tcn_gate / lgbm_bouncer / Top2 LightGBM).
Stock-funnel Top2 validator already FAIL (0/3 OOS). Seat-score gate already
REJECT_FOR_BASELINE. This step asks: on the **current CORE option TopK book**,
is there a causal, entry-visible skip that rejects obvious FA without
killing strong-window fat?

Corpus: current-profile baseline trades from C3
  ``research_core_c3_stock_rev/{weak,strong}/replay__baseline/trades.csv``
Action: post-hoc skip (same fires; no backfill). Calendar is eval-only.

PASS:
  strong keep >= 0.95
  weak keep >= 1.0 (improve) OR (keep>=0.95 AND MaxDD Δ>=50bp)
  n_skip_weak >= 2
  reject_prec_weak >= 0.67
  true_loss_strong <= 0.10

FAIL → do not connect ML router; do not add morph BLOCKs.

Example:
  PYTHONPATH=. python -m maga7.tools.run_core_c4_entry_validator \\
    --tag research_core_c4_entry_validator
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

PROFILE = (
    "maga7/CONFIG/strategy_profiles/"
    "single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1.json"
)
C3_ROOT = Path("/mnt/s990/data/maga7/results/research_core_c3_stock_rev")
WINDOWS = (("weak", "2026-01-02", "2026-03-31"), ("strong", "2026-04-01", "2026-07-21"))
NY = "America/New_York"


def _prep(trades: pd.DataFrame) -> pd.DataFrame:
    t = trades.copy()
    t["date"] = t["date"].astype(str)
    t["ret"] = pd.to_numeric(t["ret"], errors="coerce")
    t["size_frac"] = pd.to_numeric(t.get("size_frac"), errors="coerce").fillna(0.2)
    t["entry_ts"] = pd.to_datetime(t["entry_ts"], utc=True, errors="coerce")
    tod = t["entry_ts"].dt.tz_convert(NY)
    t["tod"] = tod.dt.strftime("%H:%M")
    t["abs_fo"] = pd.to_numeric(t.get("sig_from_open"), errors="coerce").abs()
    s1 = t.get("stock_path_confirm_reason")
    t["s1"] = s1.fillna("").astype(str).str.strip().str.lower() if s1 is not None else ""
    t["route"] = t.get("route", "baseline").astype(str).str.strip().str.lower()
    t["dir"] = t.get("dir", "").astype(str).str.strip().str.upper()
    return t.sort_values(["date", "entry_ts", "symbol"]).reset_index(drop=True)


def load_core_topk_book(c3_root: Path = C3_ROOT) -> pd.DataFrame:
    parts = []
    for wname, _, _ in WINDOWS:
        p = c3_root / wname / "replay__baseline" / "trades.csv"
        if not p.exists():
            raise SystemExit(f"missing current CORE book {p} (run C3 first)")
        t = pd.read_csv(p)
        t["window"] = wname
        parts.append(t)
    return _prep(pd.concat(parts, ignore_index=True))


def equity(trades: pd.DataFrame) -> dict[str, Any]:
    if trades is None or trades.empty:
        return {"n": 0, "total_ret": 0.0, "maxdd": 0.0, "win": None, "n_loss": 0}
    eq = 100.0
    peak = 100.0
    maxdd = 0.0
    rets: list[float] = []
    for row in trades.itertuples(index=False):
        r = float(row.ret)
        if not np.isfinite(r):
            continue
        eq *= 1.0 + float(row.size_frac) * r
        peak = max(peak, eq)
        if peak > 0:
            maxdd = min(maxdd, eq / peak - 1.0)
        rets.append(r)
    rr = np.asarray(rets, dtype=float)
    return {
        "n": int(len(rr)),
        "total_ret": float(eq / 100.0 - 1.0),
        "maxdd": float(maxdd),
        "win": float((rr > 0).mean()) if len(rr) else None,
        "n_loss": int((rr <= 0).sum()) if len(rr) else 0,
    }


def keep_ratio(variant_ret: float, base_ret: float) -> float:
    den = 1.0 + float(base_ret)
    if den == 0:
        return 0.0
    return float((1.0 + float(variant_ret)) / den)


def skip_fo_lt(thr: float) -> Callable[[pd.DataFrame], pd.Series]:
    def _m(t: pd.DataFrame) -> pd.Series:
        return pd.to_numeric(t["abs_fo"], errors="coerce") < float(thr)

    return _m


def skip_s1_none(t: pd.DataFrame) -> pd.Series:
    return t["s1"].isin(["", "none", "nan"])


def skip_hunt(t: pd.DataFrame) -> pd.Series:
    return t["route"].eq("hunt")


def skip_tod_ge(hhmm: str) -> Callable[[pd.DataFrame], pd.Series]:
    def _m(t: pd.DataFrame) -> pd.Series:
        return t["tod"] >= str(hhmm)

    return _m


def skip_fo_and_s1(thr: float) -> Callable[[pd.DataFrame], pd.Series]:
    inner = skip_fo_lt(thr)

    def _m(t: pd.DataFrame) -> pd.Series:
        return inner(t) & skip_s1_none(t)

    return _m


def skip_rebound(t: pd.DataFrame) -> pd.Series:
    return t["route"].str.contains("rebound", na=False)


VARIANTS: list[dict[str, Any]] = [
    {"name": "skip_fo_lt_1pct", "mask": skip_fo_lt(0.01)},
    {"name": "skip_fo_lt_05pct", "mask": skip_fo_lt(0.005)},
    {"name": "skip_s1_none", "mask": skip_s1_none},
    {"name": "skip_hunt", "mask": skip_hunt},
    {"name": "skip_tod_ge_1130", "mask": skip_tod_ge("11:30")},
    {"name": "skip_fo1_and_s1none", "mask": skip_fo_and_s1(0.01)},
    {"name": "skip_rebound_trap", "mask": skip_rebound},
]


def verdict_c4(
    *,
    strong_keep: float,
    weak_keep: float,
    weak_maxdd_delta: float,
    n_skip_weak: int,
    reject_prec_weak: float | None,
    true_loss_strong: float,
    min_keep: float = 0.95,
    min_prec: float = 0.67,
    max_true_loss: float = 0.10,
    min_skip_weak: int = 2,
) -> dict[str, Any]:
    weak_ok = float(weak_keep) >= 1.0 - 1e-12 or (
        float(weak_keep) >= float(min_keep) and float(weak_maxdd_delta) >= 0.005
    )
    fired = int(n_skip_weak) >= int(min_skip_weak)
    prec_ok = reject_prec_weak is not None and float(reject_prec_weak) >= float(min_prec)
    true_ok = float(true_loss_strong) <= float(max_true_loss) + 1e-12
    keep_ok = float(strong_keep) >= float(min_keep) and weak_ok
    passed = bool(keep_ok and fired and prec_ok and true_ok)
    reason = "pass"
    if not fired:
        reason = "no_weak_rejects"
    elif not prec_ok:
        reason = "reject_precision_low"
    elif not true_ok:
        reason = "true_loss_too_high"
    elif not keep_ok:
        reason = "keep_below_bar"
    return {
        "pass": passed,
        "reason": reason,
        "keep_ok": keep_ok,
        "fired": fired,
        "prec_ok": prec_ok,
        "true_ok": true_ok,
    }


def _window_stats(sub: pd.DataFrame, skipped: pd.DataFrame, kept: pd.DataFrame, base: dict[str, Any]) -> dict[str, Any]:
    st = equity(kept)
    n_skip = int(len(skipped))
    n_loss_skip = int((skipped["ret"] <= 0).sum()) if n_skip else 0
    n_win_skip = int((skipped["ret"] > 0).sum()) if n_skip else 0
    n_win = int((sub["ret"] > 0).sum())
    return {
        **st,
        "n_skip": n_skip,
        "n_loss_skip": n_loss_skip,
        "reject_prec": (float(n_loss_skip / n_skip) if n_skip else None),
        "true_loss": (float(n_win_skip / n_win) if n_win else 0.0),
        "keep": keep_ratio(st["total_ret"], base["total_ret"]),
        "maxdd_delta": float(st["maxdd"] - base["maxdd"]),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", default=PROFILE)
    ap.add_argument("--tag", default="research_core_c4_entry_validator")
    ap.add_argument("--c3-root", default=str(C3_ROOT))
    args = ap.parse_args(argv)

    profile = load_profile(args.profile)
    out = Path(profile["_paths"]["results_dir"]) / args.tag
    out.mkdir(parents=True, exist_ok=True)

    trades = load_core_topk_book(Path(args.c3_root))
    trades.to_csv(out / "seats.csv", index=False)

    rows: list[dict[str, Any]] = []
    baseline: dict[str, dict[str, Any]] = {}
    for wname, w0, w1 in WINDOWS:
        sub = trades[trades["window"] == wname]
        st = equity(sub)
        baseline[wname] = st
        rows.append({"variant": "baseline", "window": wname, **st, "n_skip": 0, "keep": 1.0})

    adopted: list[dict[str, Any]] = []
    for var in VARIANTS:
        mask = var["mask"](trades)
        for wname, _, _ in WINDOWS:
            sub = trades[trades["window"] == wname]
            m = mask.loc[sub.index]
            skipped = sub[m]
            kept = sub[~m]
            st = _window_stats(sub, skipped, kept, baseline[wname])
            rows.append({"variant": var["name"], "window": wname, **st})

        by = {r["window"]: r for r in rows if r["variant"] == var["name"]}
        weak, strong = by["weak"], by["strong"]
        v = verdict_c4(
            strong_keep=float(strong["keep"]),
            weak_keep=float(weak["keep"]),
            weak_maxdd_delta=float(weak["maxdd_delta"]),
            n_skip_weak=int(weak["n_skip"]),
            reject_prec_weak=weak.get("reject_prec"),
            true_loss_strong=float(strong["true_loss"]),
        )
        adopted.append(
            {
                "variant": var["name"],
                **v,
                "strong_keep": float(strong["keep"]),
                "weak_keep": float(weak["keep"]),
                "n_skip_weak": int(weak["n_skip"]),
                "n_skip_strong": int(strong["n_skip"]),
                "reject_prec_weak": weak.get("reject_prec"),
                "reject_prec_strong": strong.get("reject_prec"),
                "true_loss_strong": float(strong["true_loss"]),
                "true_loss_weak": float(weak["true_loss"]),
                "weak_maxdd_delta": float(weak["maxdd_delta"]),
            }
        )

    sb = pd.DataFrame(rows)
    sb.to_csv(out / "scoreboard.csv", index=False)
    passed = [a for a in adopted if a["pass"]]
    promote = f"C4_{passed[0]['variant']}" if passed else "NONE"
    summary = {
        "protocol": "core_c4_entry_validator",
        "promotion_mark": "causal_skip_on_actual_topk_seats",
        "corpus": "research_core_c3_stock_rev current-profile baseline",
        "n_seats": int(len(trades)),
        "pass_rule": (
            "strong keep>=0.95 AND (weak keep>=1.0 OR keep>=0.95+MaxDDΔ>=50bp) "
            "AND n_skip_weak>=2 AND reject_prec_weak>=0.67 AND true_loss_strong<=0.10"
        ),
        "note": (
            "Post-hoc skip only. Do not wire tcn_gate / lgbm_bouncer / seat_score. "
            "Stock Top2 LightGBM validator already FAIL 0/3 OOS. "
            "Calendar is eval-only."
        ),
        "baseline": baseline,
        "variants": adopted,
        "promote": promote,
        "pass": bool(promote != "NONE"),
        "next_step": (
            "wire_entry_validator_on_research_baseline"
            if promote != "NONE"
            else "do_not_connect_ml_router"
        ),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    lines = [
        "# CORE C4 — Entry validator on actual TopK seats",
        "",
        "- action: **post-hoc skip** of causal 'obvious FA' (no Rule-A change, no ML)",
        "- object: current CORE option TopK fills (C3 baseline book)",
        f"- promote: **{promote}**",
        f"- pass: **{summary['pass']}**",
        "",
        "## Scoreboard",
        "",
    ]
    try:
        lines.append(sb.to_markdown(index=False))
    except Exception:
        lines.append(sb.to_string(index=False))
    if promote != "NONE":
        best = passed[0]
        lines += [
            "",
            "## 结论",
            "",
            f"**C4 PASS** → `{best['variant']}` "
            f"strong keep={best['strong_keep']:.3f} · weak keep={best['weak_keep']:.3f}。",
            "可研究接线；仍不接 TCN/LGBM router。",
        ]
    else:
        lines += [
            "",
            "## 结论",
            "",
            "**C4 FAIL** — 能在弱窗标出假信号的入场特征，在强窗同样切到肥单。",
            "不接 ML router；不加 morph BLOCK。seat_score / tcn_gate / lgbm_bouncer 保持 off。",
        ]
    (out / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(sb.to_string(index=False), flush=True)
    print(json.dumps({"promote": promote, "pass": summary["pass"], "variants": adopted}, indent=2))
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

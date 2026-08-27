#!/usr/bin/env python3
"""Baseline T30→T45 + feature risk gates (keep clock, add indicators).

Runs on research baseline profile. Does **not** abolish fixed hold_minutes;
adds causal feature gates at extend / mid-hold stale cut.
"""
from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

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

WINDOWS = [
    ("may_jul", "2026-05-01", "2026-07-17"),
    ("jan_mar", "2026-01-02", "2026-03-31"),
]

VARIANTS: dict[str, dict] = {
    "baseline": {},
    # T30 extend: require underlying still favorable
    "ext_stock0": {
        "hold_extend_require_stock": True,
        "hold_extend_stock_min": 0.0,
    },
    "ext_stock15bp": {
        "hold_extend_require_stock": True,
        "hold_extend_stock_min": 0.0015,
    },
    # Must have printed option MFE before earning T45
    "ext_mfe10": {"hold_extend_min_peak_mfe": 0.10},
    "ext_mfe20": {"hold_extend_min_peak_mfe": 0.20},
    # Deny extend if QQQ already adverse
    "ext_qqq50bp": {"hold_extend_max_qqq_adverse": 0.005},
    "ext_qqq80bp": {"hold_extend_max_qqq_adverse": 0.008},
    # Combos
    "ext_stock0_mfe10": {
        "hold_extend_require_stock": True,
        "hold_extend_stock_min": 0.0,
        "hold_extend_min_peak_mfe": 0.10,
    },
    "ext_stock15_mfe10": {
        "hold_extend_require_stock": True,
        "hold_extend_stock_min": 0.0015,
        "hold_extend_min_peak_mfe": 0.10,
    },
    "ext_stock0_qqq50": {
        "hold_extend_require_stock": True,
        "hold_extend_stock_min": 0.0,
        "hold_extend_max_qqq_adverse": 0.005,
    },
    # Mid-hold stale loser cut (T15): MTM≤0 and stock≤0
    "stale15": {
        "stale_cut_minutes": 15,
        "stale_cut_mtm_max": 0.0,
        "stale_cut_stock_max": 0.0,
    },
    "stale15_stock_m15": {
        "stale_cut_minutes": 15,
        "stale_cut_mtm_max": 0.0,
        "stale_cut_stock_max": -0.0015,
    },
    # Best-guess combo
    "combo_stock_mfe_stale": {
        "hold_extend_require_stock": True,
        "hold_extend_stock_min": 0.0,
        "hold_extend_min_peak_mfe": 0.10,
        "stale_cut_minutes": 15,
        "stale_cut_mtm_max": 0.0,
        "stale_cut_stock_max": 0.0,
    },
}


def _metrics(summary: dict, trades: pd.DataFrame) -> dict:
    reasons = {}
    if trades is not None and not trades.empty and "reason" in trades.columns:
        reasons = {str(k): int(v) for k, v in trades["reason"].value_counts().items()}
    return {
        "total_ret": float(summary["total_ret"]),
        "maxdd": float(summary["maxdd"]),
        "n_trades": int(summary["n_trades"]),
        "trade_win": float(summary["trade_win"]),
        "trade_exp": float(summary.get("trade_exp") or 0),
        "n_t30": int(reasons.get("T+30", 0)),
        "n_t45": int(reasons.get("T+45", 0)),
        "n_stale_cut": int(reasons.get("STALE_CUT", 0)),
        "n_trade_tox": int(
            reasons.get("TRADE_TOX", 0) + reasons.get("TRADE_TOX_RECONNECT", 0)
        ),
        "n_sl": int(reasons.get("SL", 0)),
        "reason_top": dict(list(sorted(reasons.items(), key=lambda kv: -kv[1])[:8])),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", default=PROFILE)
    ap.add_argument("--windows", default="may_jul,jan_mar")
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument(
        "--out",
        default="/mnt/s990/data/maga7/results/hold_feature_risk_ablation_v1",
    )
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    want_w = {x.strip() for x in args.windows.split(",") if x.strip()}
    want_v = [x.strip() for x in args.variants.split(",") if x.strip()]

    rows = []
    for wname, start, end in WINDOWS:
        if wname not in want_w:
            continue
        for vname in want_v:
            if vname not in VARIANTS:
                raise SystemExit(f"unknown variant {vname}")
            prof = load_profile(args.profile)
            prof = deepcopy(prof)
            prof["date_range"] = {"start": start, "end": end}
            trade = prof.setdefault("trade", {})
            for k, v in VARIANTS[vname].items():
                trade[k] = v
            tag = f"{wname}__{vname}"
            print(f"=== {tag} ===", flush=True)
            result = run_offline_replay(prof, scheme="single")
            summary = result["summary"]
            trades = result["trades"]
            wdir = out / tag
            wdir.mkdir(parents=True, exist_ok=True)
            (wdir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
            trades.to_csv(wdir / "trades.csv", index=False)
            m = _metrics(summary, trades)
            m["window"] = wname
            m["variant"] = vname
            m["start"] = start
            m["end"] = end
            rows.append(m)
            print(
                f"  ret={m['total_ret']:+.1%} dd={m['maxdd']:+.1%} "
                f"win={m['trade_win']:.1%} n={m['n_trades']} "
                f"T30/T45/STALE={m['n_t30']}/{m['n_t45']}/{m['n_stale_cut']}",
                flush=True,
            )

    bdf = pd.DataFrame(rows)
    bdf.to_csv(out / "scoreboard.csv", index=False)

    # Relative to baseline per window
    cmp_rows = []
    for wname, _, _ in WINDOWS:
        if wname not in want_w:
            continue
        base = bdf[(bdf.window == wname) & (bdf.variant == "baseline")]
        if base.empty:
            continue
        br = float(base.iloc[0]["total_ret"])
        bd = float(base.iloc[0]["maxdd"])
        for r in bdf[bdf.window == wname].itertuples():
            cmp_rows.append(
                {
                    "window": wname,
                    "variant": r.variant,
                    "total_ret": r.total_ret,
                    "vs_base_ret": r.total_ret - br,
                    "maxdd": r.maxdd,
                    "vs_base_dd": r.maxdd - bd,  # less negative is better
                    "trade_win": r.trade_win,
                    "n_trades": r.n_trades,
                    "n_t45": r.n_t45,
                    "n_stale_cut": r.n_stale_cut,
                    "keep_ret_95": r.total_ret >= 0.95 * br if br > 0 else r.total_ret >= br,
                    "dd_not_worse": r.maxdd >= bd - 0.02,
                }
            )
    cdf = pd.DataFrame(cmp_rows)
    cdf.to_csv(out / "vs_baseline.csv", index=False)

    # Promote: both windows keep ≥95% ret, DD not worse >2pp, and material
    # lift on at least one window (exclude no-op clones of baseline).
    promote = []
    for v in want_v:
        if v == "baseline":
            continue
        sub = cdf[cdf.variant == v]
        if len(sub) < len(want_w):
            continue
        # Exclude no-op clones that leave PnL identical to baseline.
        material = bool((sub["vs_base_ret"].abs() > 1e-6).any())
        if (
            bool(sub["keep_ret_95"].all())
            and bool(sub["dd_not_worse"].all())
            and material
        ):
            promote.append(v)
    # Prefer positive ret lift on weak window
    best = None
    if promote:
        weak = cdf[(cdf.window == "jan_mar") & (cdf.variant.isin(promote))]
        if len(weak):
            best = weak.sort_values("vs_base_ret", ascending=False).iloc[0]["variant"]
        else:
            best = promote[0]

    verdict = (
        "PROMOTE"
        if best
        else (
            "INTERESTING"
            if any(cdf.loc[cdf.variant != "baseline", "vs_base_ret"] > 0)
            else "NOT_USEFUL"
        )
    )
    summary = {
        "verdict": verdict,
        "promote_candidates": promote,
        "best": best,
        "note": (
            "Keep T30 clock; feature gates only affect extend/stale early cut. "
            "PROMOTE = all selected windows ≥95% baseline ret and MaxDD not worse >2pp."
        ),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    lines = [
        "# Hold Feature Risk Ablation (baseline T30→T45)",
        "",
        f"**Verdict: `{verdict}`** · best=`{best}` · candidates=`{promote}`",
        "",
        "Keeps research baseline entry + `hold_minutes=30` / extend45. "
        "Adds stock / MFE / QQQ gates at T30 and optional STALE_CUT.",
        "",
        "## Scoreboard",
        "",
        "```",
        bdf.to_string(index=False),
        "```",
        "",
        "## vs baseline",
        "",
        "```",
        cdf.to_string(index=False),
        "```",
        "",
    ]
    (out / "REPORT.md").write_text("\n".join(lines))
    print(json.dumps(summary, indent=2), flush=True)
    print("wrote", out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

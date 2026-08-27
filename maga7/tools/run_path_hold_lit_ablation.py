#!/usr/bin/env python3
"""Literature-aligned path hold ablation (risk-first).

Three rails vs unconditional T30/T45:
  1. MTM trail after peak arm (path take-profit / giveback)
  2. STOCK_REV always (thesis invalidation — abolish unconditional hold)
  3. Clock only as undeveloped safety (no hold_extend gift)

Profile default: path_hold_lit_v1
Docs: maga7/docs/path_adaptive_hold_research.md
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

PEER3 = (
    "maga7/CONFIG/strategy_profiles/"
    "single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1.json"
)
LIT = (
    "maga7/CONFIG/strategy_profiles/"
    "single_qqq_open_ladder_atm5otm_path_hold_lit_v1.json"
)

WINDOWS = [
    ("may_jul", "2026-05-01", "2026-07-17"),
    ("jan_mar", "2026-01-02", "2026-03-31"),
]

REV10 = {
    "enabled": True,
    "when": "always",
    "min_hold_minutes": 10,
    "stock_max": 0.0,
    "opt_mtm_max": 0.10,
}

VARIANTS: dict[str, dict] = {
    # peer3 freeze scaffold
    "baseline_t30": {},
    # Rail 1 only
    "trail20_t45": {
        "exit_mode": "mtm_trail",
        "hold_minutes": 45,
        "hold_extend_minutes": None,
        "trail_activate": 0.20,
        "trail_dd": 0.12,
    },
    # Rail 2 only (no extend)
    "rev10_t45": {
        "exit_mode": "none",
        "hold_minutes": 45,
        "hold_extend_minutes": None,
        "stock_rev_exit": dict(REV10),
    },
    # Full lit (matches path_hold_lit_v1)
    "lit_trail_rev": {
        "exit_mode": "mtm_trail",
        "hold_minutes": 45,
        "hold_extend_minutes": None,
        "trail_activate": 0.20,
        "trail_dd": 0.12,
        "stock_rev_exit": dict(REV10),
    },
    # Tighter undeveloped safety
    "lit_trail_rev_t30": {
        "exit_mode": "mtm_trail",
        "hold_minutes": 30,
        "hold_extend_minutes": None,
        "trail_activate": 0.20,
        "trail_dd": 0.12,
        "stock_rev_exit": dict(REV10),
    },
    # Keep peer3 extend but add path rails (soft migration)
    "peer3_plus_trail_rev": {
        "exit_mode": "hold_extend+mtm_trail",
        "trail_activate": 0.20,
        "trail_dd": 0.12,
        "stock_rev_exit": dict(REV10),
    },
}


def _hold_sec(trades: pd.DataFrame) -> pd.Series:
    if trades is None or trades.empty:
        return pd.Series(dtype=float)
    et = pd.to_datetime(trades["entry_ts"], utc=True, errors="coerce")
    xt = pd.to_datetime(trades["exit_ts"], utc=True, errors="coerce")
    return (xt - et).dt.total_seconds()


def _metrics(summary: dict, trades: pd.DataFrame) -> dict:
    hs = _hold_sec(trades)
    reasons: dict[str, int] = {}
    if trades is not None and not trades.empty and "reason" in trades.columns:
        reasons = {str(k): int(v) for k, v in trades["reason"].value_counts().items()}
    t30 = trades[trades["reason"].isin(["T+30", "T+45"])] if (
        trades is not None and not trades.empty and "reason" in trades.columns
    ) else pd.DataFrame()
    losers = (
        trades[pd.to_numeric(trades["ret"], errors="coerce") < 0]
        if trades is not None and not trades.empty
        else pd.DataFrame()
    )
    loser_hs = _hold_sec(losers) if len(losers) else pd.Series(dtype=float)
    return {
        "total_ret": float(summary["total_ret"]),
        "maxdd": float(summary["maxdd"]),
        "n_trades": int(summary["n_trades"]),
        "trade_win": float(summary["trade_win"]),
        "worst_day": float(summary.get("worst_day") or 0.0),
        "med_hold_sec": float(hs.median()) if len(hs) else None,
        "med_loser_hold_sec": float(loser_hs.median()) if len(loser_hs) else None,
        "t30_sum_ret": float(pd.to_numeric(t30["ret"], errors="coerce").sum())
        if len(t30)
        else 0.0,
        "n_t30": int(reasons.get("T+30", 0)),
        "n_t45": int(reasons.get("T+45", 0)),
        "n_tp": int(reasons.get("TP", 0)),
        "n_trail": int(reasons.get("TRAIL", 0)),
        "n_stock_rev": int(reasons.get("STOCK_REV", 0)),
        "n_sl": int(reasons.get("SL", 0)),
        "reason_top": dict(list(sorted(reasons.items(), key=lambda kv: -kv[1])[:12])),
    }


def _risk_rank(df: pd.DataFrame) -> pd.DataFrame:
    """Prefer better worst_day / MaxDD, then less T30 bleed; ret secondary."""
    out = df.copy()
    out["risk_score"] = (
        out["worst_day"].rank(ascending=False)
        + out["maxdd"].rank(ascending=False)
        + (-out["t30_sum_ret"]).rank(ascending=True) * 0.5
        + out["total_ret"].rank(ascending=False) * 0.25
    )
    out = out.sort_values(["window", "risk_score"], ascending=[True, False])
    out["risk_rank"] = out.groupby("window").cumcount() + 1
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", default=PEER3, help="Base profile (variants overlay trade)")
    ap.add_argument("--windows", default="may_jul,jan_mar")
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument(
        "--out",
        default="/mnt/s990/data/maga7/results/path_hold_lit_v1",
    )
    ap.add_argument(
        "--lit-profile-check",
        action="store_true",
        help="Also run LIT profile as-is (sanity vs lit_trail_rev overlay)",
    )
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    want_w = {x.strip() for x in args.windows.split(",") if x.strip()}
    want_v = [x.strip() for x in args.variants.split(",") if x.strip()]

    rows: list[dict] = []
    jobs: list[tuple[str, str, str, str, dict | None]] = []
    for wname, start, end in WINDOWS:
        if wname not in want_w:
            continue
        for vname in want_v:
            if vname not in VARIANTS:
                raise SystemExit(f"unknown variant {vname}")
            jobs.append((wname, start, end, vname, VARIANTS[vname]))
        if args.lit_profile_check:
            jobs.append((wname, start, end, "lit_profile_file", None))

    for wname, start, end, vname, overlay in jobs:
        if overlay is None:
            prof = load_profile(LIT)
            base_tag = "lit_file"
        else:
            prof = load_profile(args.profile)
            base_tag = vname
        prof = deepcopy(prof)
        prof["date_range"] = {"start": start, "end": end}
        if overlay is not None:
            trade = prof.setdefault("trade", {})
            for k, v in overlay.items():
                trade[k] = v
        tag = f"{wname}__{base_tag}"
        print(f"=== {tag} ===", flush=True)
        result = run_offline_replay(prof, scheme="single")
        summary = result["summary"]
        trades = result["trades"]
        # worst_day from daily if present
        daily = result.get("daily")
        if daily is not None and len(daily) and "day_ret" in daily.columns:
            summary = dict(summary)
            summary["worst_day"] = float(pd.to_numeric(daily["day_ret"]).min())
        wdir = out / tag
        wdir.mkdir(parents=True, exist_ok=True)
        (wdir / "summary.json").write_text(
            json.dumps(summary, indent=2, default=str), encoding="utf-8"
        )
        trades.to_csv(wdir / "trades.csv", index=False)
        if daily is not None and len(daily):
            daily.to_csv(wdir / "daily.csv", index=False)
        m = _metrics(summary, trades)
        m["window"] = wname
        m["variant"] = base_tag
        m["start"] = start
        m["end"] = end
        rows.append(m)
        print(
            f"  ret={m['total_ret']:+.1%} dd={m['maxdd']:+.1%} "
            f"worst={m['worst_day']:+.1%} n={m['n_trades']} "
            f"TP/TRAIL/REV/T30/T45="
            f"{m['n_tp']}/{m['n_trail']}/{m['n_stock_rev']}/{m['n_t30']}/{m['n_t45']}",
            flush=True,
        )

    bdf = pd.DataFrame(rows)
    bdf.to_csv(out / "scoreboard.csv", index=False)
    ranked = _risk_rank(bdf)
    ranked.to_csv(out / "risk_ranked.csv", index=False)
    (out / "scoreboard.json").write_text(
        json.dumps(rows, indent=2, default=str), encoding="utf-8"
    )

    # Pick promote: best risk_rank on jan_mar among full-rail variants, dual-window ok
    full = {"lit_trail_rev", "lit_trail_rev_t30", "peer3_plus_trail_rev", "lit_profile_file"}
    promote = None
    sub = ranked[ranked.variant.isin(full)]
    if len(sub):
        # average risk_rank across windows
        avg = sub.groupby("variant")["risk_rank"].mean().sort_values()
        promote = str(avg.index[0])

    summary = {
        "verdict": "PROMOTE_RESEARCH" if promote else "RESEARCH",
        "best_risk": promote,
        "thesis": (
            "Path take-profit (trail) + thesis cut (STOCK_REV) + clock as "
            "undeveloped safety only. Fixed T30/T45 extend is not the edge."
        ),
        "rails": ["TRAIL", "STOCK_REV", "T_safety"],
        "profile": LIT,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = [
        "# Path Hold Literature Ablation",
        "",
        f"**Verdict: `{summary['verdict']}`** · best_risk=`{promote}`",
        "",
        "## Rails",
        "",
        "1. TRAIL — path take-profit / giveback lock",
        "2. STOCK_REV — abolish unconditional hold when stock thesis fails",
        "3. Clock — undeveloped safety only (no hold_extend)",
        "",
        "## Scoreboard",
        "",
        "```",
        bdf.to_string(index=False),
        "```",
        "",
        "## Risk-ranked",
        "",
        "```",
        ranked.to_string(index=False),
        "```",
        "",
    ]
    (out / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print("wrote", out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Freeze peer3 entries; ablate profit-lock exits only (dual window).

Compares:
  - baseline hold_extend (mtm_min=0)
  - raise hold_extend_mtm_min
  - stack soft/hard mtm_trail on hold_extend

Windows follow L3 / path-hold lit: May–Jul + Jan–Mar.
Does not change live/shadow session.
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
OUT = Path("/mnt/s990/data/maga7/results/lock_profit_exit_ablation_v1")

WINDOWS = {
    "may_jul": ("2026-05-01", "2026-07-20"),
    "jan_mar": ("2026-01-02", "2026-03-31"),
}

VARIANTS: dict[str, dict[str, Any]] = {
    "baseline": {},
    "ext_mtm05": {"hold_extend_mtm_min": 0.05},
    "ext_mtm10": {"hold_extend_mtm_min": 0.10},
    "ext_mtm15": {"hold_extend_mtm_min": 0.15},
    # Soft trail: only arm after larger peak; wider giveback — less right-tail cut.
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
    # Harder lit-style trail (known to cut May–Jul hard; included as control).
    "trail20_dd12": {
        "exit_mode": "hold_extend+mtm_trail",
        "trail_activate": 0.20,
        "trail_dd": 0.12,
    },
    "trail15_dd08": {
        "early_exit_mode": "mtm_trail",
        "trail_activate": 0.15,
        "trail_dd": 0.08,
    },
    # Combine soft extend gate + soft trail.
    "ext05_trail30": {
        "hold_extend_mtm_min": 0.05,
        "exit_mode": "hold_extend+mtm_trail",
        "trail_activate": 0.30,
        "trail_dd": 0.20,
    },
}


def _reason_counts(trades: pd.DataFrame) -> dict[str, int]:
    if trades is None or trades.empty or "reason" not in trades.columns:
        return {}
    return {str(k): int(v) for k, v in trades["reason"].value_counts().items()}


def _clock_bleed(trades: pd.DataFrame) -> dict[str, float]:
    if trades is None or trades.empty or "reason" not in trades.columns:
        return {"n_clock": 0, "clock_sum_ret": 0.0, "clock_mean_ret": 0.0}
    clock = trades[trades["reason"].astype(str).str.startswith("T+")]
    rets = pd.to_numeric(clock["ret"], errors="coerce")
    return {
        "n_clock": int(len(clock)),
        "clock_sum_ret": float(rets.sum()) if len(rets) else 0.0,
        "clock_mean_ret": float(rets.mean()) if len(rets) else 0.0,
    }


def run_one(window: str, variant: str, overlay: dict[str, Any]) -> dict[str, Any]:
    start, end = WINDOWS[window]
    prof = deepcopy(load_profile(PEER3))
    prof["date_range"] = {"start": start, "end": end}
    trade = prof.setdefault("trade", {})
    for k, v in overlay.items():
        trade[k] = v
    print(f"=== {window} / {variant} {start}..{end} ===", flush=True)
    result = run_offline_replay(prof, scheme="single")
    summary = result["summary"]
    trades = result["trades"]
    daily = result.get("daily")
    worst_day = None
    if daily is not None and len(daily) and "day_ret" in daily.columns:
        worst_day = float(pd.to_numeric(daily["day_ret"], errors="coerce").min())
    out = OUT / window / f"replay__{variant}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    trades.to_csv(out / "trades.csv", index=False)
    if daily is not None and len(daily):
        daily.to_csv(out / "daily.csv", index=False)
    reasons = _reason_counts(trades)
    clock = _clock_bleed(trades)
    row = {
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
        "n_trail": int(reasons.get("TRAIL", 0)),
        **clock,
        "reasons": reasons,
    }
    print(
        f"  ret={row['total_ret']:+.3f} dd={row['maxdd']:.3f} n={row['n_trades']} "
        f"win={row['trade_win']:.3f} "
        f"TP/TRAIL/T30/T45={row['n_tp']}/{row['n_trail']}/{row['n_t30']}/{row['n_t45']} "
        f"clock_sum={row['clock_sum_ret']:+.3f}",
        flush=True,
    )
    return row


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--windows", default="may_jul,jan_mar")
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    wins = [w.strip() for w in args.windows.split(",") if w.strip()]
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]

    rows: list[dict[str, Any]] = []
    for window in wins:
        if window not in WINDOWS:
            raise SystemExit(f"unknown window {window}")
        for variant in variants:
            if variant not in VARIANTS:
                raise SystemExit(f"unknown variant {variant}")
            rows.append(run_one(window, variant, VARIANTS[variant]))

    df = pd.DataFrame(rows)
    base = {
        r["window"]: float(r["total_ret"])
        for r in rows
        if r["variant"] == "baseline"
    }
    enriched = []
    for r in rows:
        b = base.get(r["window"])
        keep = None
        if b is not None and (1.0 + b) != 0:
            keep = (1.0 + float(r["total_ret"])) / (1.0 + b)
        enriched.append(
            {
                **{k: v for k, v in r.items() if k != "reasons"},
                "ret_retention": keep,
                "ret_vs_baseline": (
                    float(r["total_ret"]) - b if b is not None else None
                ),
                "reasons": r["reasons"],
            }
        )
    edf = pd.DataFrame([{k: v for k, v in r.items() if k != "reasons"} for r in enriched])
    edf.to_csv(out / "scoreboard.csv", index=False)
    (out / "scoreboard.json").write_text(
        json.dumps(enriched, indent=2, default=str), encoding="utf-8"
    )

    # Dual-window promote: retain >=0.85 both windows, prefer less clock bleed.
    retain = edf[edf["variant"] != "baseline"].copy()
    dual_ok = []
    for variant, g in retain.groupby("variant"):
        if set(g["window"]) >= set(wins) and (g["ret_retention"] >= 0.85).all():
            dual_ok.append(
                {
                    "variant": variant,
                    "min_retention": float(g["ret_retention"].min()),
                    "mean_retention": float(g["ret_retention"].mean()),
                    "may_jul_ret": float(g.loc[g.window == "may_jul", "total_ret"].iloc[0])
                    if "may_jul" in set(g.window)
                    else None,
                    "jan_mar_ret": float(g.loc[g.window == "jan_mar", "total_ret"].iloc[0])
                    if "jan_mar" in set(g.window)
                    else None,
                    "clock_sum_total": float(g["clock_sum_ret"].sum()),
                    "n_trail_total": int(g["n_trail"].sum()),
                }
            )
    dual_df = pd.DataFrame(dual_ok).sort_values(
        ["min_retention", "may_jul_ret"], ascending=[False, False]
    ) if dual_ok else pd.DataFrame()
    if len(dual_df):
        dual_df.to_csv(out / "dual_ok.csv", index=False)

    # Today's live NVDA giveback case: what-if commentary only in verdict.
    verdict = {
        "verdict": "PROMOTE_RESEARCH" if len(dual_df) else "REJECT_FOR_BASELINE",
        "dual_ok_variants": dual_df["variant"].tolist() if len(dual_df) else [],
        "best": dual_df.iloc[0].to_dict() if len(dual_df) else None,
        "live_case": {
            "date": "2026-07-27",
            "nvda": {
                "entry": "11:16",
                "exit": "12:01 T+45",
                "peak_mfe": 0.328,
                "exit_ret": -0.1688,
                "note": "peak +33% never hit TP(+60%); extend_mtm_min=0 allowed hold through giveback",
            },
            "amd": {
                "entry": "11:15",
                "exit": "11:45 T+30",
                "peak_mfe": 0.0,
                "exit_ret": -0.0453,
                "note": "OMS sell-mark never green; wide entry spread; clock flat",
            },
        },
        "thesis": (
            "Locking giveback (trail / raise extend_mtm_min) must retain >=0.85 "
            "of peer3 total_ret on BOTH may_jul and jan_mar before baseline wire."
        ),
    }
    (out / "verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")
    lines = [
        "# Lock-profit exit ablation (peer3 freeze entry)",
        "",
        f"**Verdict: `{verdict['verdict']}`**",
        "",
        "## Scoreboard",
        "",
        "```",
        edf.to_string(index=False),
        "```",
        "",
        "## Dual-window retain>=0.85",
        "",
        "```",
        dual_df.to_string(index=False) if len(dual_df) else "(none)",
        "```",
        "",
        "## Live 2026-07-27 after 10:30",
        "",
        "- AMD 11:15→11:45 **T+30** peak_mfe=0 exit=-4.5%",
        "- NVDA 11:16→12:01 **T+45** peak_mfe=+32.8% exit=-16.9%",
        "",
    ]
    (out / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(verdict, indent=2), flush=True)
    print("wrote", out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Dual-window: peer3 entries × giveback extend gate + profit_stall ladder.

Freeze signal/entry. Compare:
  1) baseline hold_extend
  2) hold_extend_max_giveback (refuse T30→T45 after peak fade)
  3) second-level profit_stall via ladder_active (always / wash-conditional)

Windows: May–Jul + Jan–Mar.
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
OUT = Path("/mnt/s990/data/maga7/results/giveback_stall_exit_ablation_v1")

WINDOWS = {
    "may_jul": ("2026-05-01", "2026-07-20"),
    "jan_mar": ("2026-01-02", "2026-03-31"),
}


def _lac(
    *,
    max_hold: int,
    stall_peak: float,
    stall_sec: int,
    when: str = "always",
    tp_trail: float = 0.20,
    trail_dd: float = 0.08,
    tp_exit: float = 0.40,
    sl: list[float] | None = None,
    mf_flip: bool = False,
) -> dict[str, Any]:
    return {
        "enabled": True,
        "when": when,
        "max_hold_seconds": int(max_hold),
        "keep_outer_rails": True,
        "sl_rails": [{"ret": float(x)} for x in (sl or [-0.20, -0.35])],
        "tp_rails": [
            {"ret": float(tp_trail), "action": "trail", "trail_dd": float(trail_dd)},
            {"ret": float(tp_exit), "action": "exit"},
        ],
        "profit_stall": {
            "min_peak": float(stall_peak),
            "stall_seconds": int(stall_sec),
        },
        "mf_flip": bool(mf_flip),
        "mf_grace_seconds": 30 if mf_flip else 9999,
    }


VARIANTS: dict[str, dict[str, Any]] = {
    "baseline": {},
    # Giveback-aware extend: refuse gift time after peak fade.
    "gb12_p15": {
        "hold_extend_max_giveback": 0.12,
        "hold_extend_giveback_min_peak": 0.15,
    },
    "gb15_p15": {
        "hold_extend_max_giveback": 0.15,
        "hold_extend_giveback_min_peak": 0.15,
    },
    "gb15_p20": {
        "hold_extend_max_giveback": 0.15,
        "hold_extend_giveback_min_peak": 0.20,
    },
    "gb20_p20": {
        "hold_extend_max_giveback": 0.20,
        "hold_extend_giveback_min_peak": 0.20,
    },
    "gb12_p20": {
        "hold_extend_max_giveback": 0.12,
        "hold_extend_giveback_min_peak": 0.20,
    },
    # Soft profit_stall always-on (replace clock with sec ladder book).
    "stall20_60_always": {
        "exit_mode": "ladder_active",
        "hold_minutes": 15,
        "hold_extend_minutes": None,
        "ladder_fallback_exit_mode": "hold_extend",
        "ladder_active": _lac(
            max_hold=900,
            stall_peak=0.20,
            stall_sec=60,
            when="always",
            mf_flip=False,
        ),
        "delta_time_stop": {"enabled": False},
        "roi_time_stop": {"enabled": False},
    },
    # Isolate stall: loose ladder rails + 45m SEC_MAX so stall is the main add-on.
    "stall_only_20_60": {
        "exit_mode": "ladder_active",
        "hold_minutes": 30,
        "hold_extend_minutes": None,
        "ladder_fallback_exit_mode": "hold_extend",
        "ladder_active": {
            "enabled": True,
            "when": "always",
            "max_hold_seconds": 2700,
            "keep_outer_rails": True,
            "sl_rails": [{"ret": -0.99}],
            "tp_rails": [{"ret": 9.0, "action": "exit"}],
            "profit_stall": {"min_peak": 0.20, "stall_seconds": 60},
            "mf_flip": False,
            "mf_grace_seconds": 9999,
        },
        "delta_time_stop": {"enabled": False},
        "roi_time_stop": {"enabled": False},
    },
    "stall_only_25_90": {
        "exit_mode": "ladder_active",
        "hold_minutes": 30,
        "hold_extend_minutes": None,
        "ladder_fallback_exit_mode": "hold_extend",
        "ladder_active": {
            "enabled": True,
            "when": "always",
            "max_hold_seconds": 2700,
            "keep_outer_rails": True,
            "sl_rails": [{"ret": -0.99}],
            "tp_rails": [{"ret": 9.0, "action": "exit"}],
            "profit_stall": {"min_peak": 0.25, "stall_seconds": 90},
            "mf_flip": False,
            "mf_grace_seconds": 9999,
        },
        "delta_time_stop": {"enabled": False},
        "roi_time_stop": {"enabled": False},
    },
    "stall_only_20_60_wash": {
        "exit_mode": "hold_extend",
        "ladder_fallback_exit_mode": "hold_extend",
        "ladder_active": {
            "enabled": True,
            "when": "mixed_wash_up",
            "max_hold_seconds": 2700,
            "keep_outer_rails": True,
            "sl_rails": [{"ret": -0.99}],
            "tp_rails": [{"ret": 9.0, "action": "exit"}],
            "profit_stall": {"min_peak": 0.20, "stall_seconds": 60},
            "mf_flip": False,
            "mf_grace_seconds": 9999,
        },
        "delta_time_stop": {"enabled": False},
        "roi_time_stop": {"enabled": False},
    },
    "stall15_45_always": {
        "exit_mode": "ladder_active",
        "hold_minutes": 15,
        "hold_extend_minutes": None,
        "ladder_fallback_exit_mode": "hold_extend",
        "ladder_active": _lac(
            max_hold=900,
            stall_peak=0.15,
            stall_sec=45,
            when="always",
            mf_flip=False,
        ),
        "delta_time_stop": {"enabled": False},
        "roi_time_stop": {"enabled": False},
    },
    "stall25_90_always": {
        "exit_mode": "ladder_active",
        "hold_minutes": 20,
        "hold_extend_minutes": None,
        "ladder_fallback_exit_mode": "hold_extend",
        "ladder_active": _lac(
            max_hold=1200,
            stall_peak=0.25,
            stall_sec=90,
            when="always",
            mf_flip=False,
        ),
        "delta_time_stop": {"enabled": False},
        "roi_time_stop": {"enabled": False},
    },
    # Conditional: wash days use stall ladder; clean days keep peer3 extend.
    "stall20_60_wash": {
        "exit_mode": "hold_extend",
        "ladder_fallback_exit_mode": "hold_extend",
        "ladder_active": _lac(
            max_hold=900,
            stall_peak=0.20,
            stall_sec=60,
            when="mixed_wash_up",
            mf_flip=False,
        ),
        "delta_time_stop": {"enabled": False},
        "roi_time_stop": {"enabled": False},
    },
    "stall15_45_wash": {
        "exit_mode": "hold_extend",
        "ladder_fallback_exit_mode": "hold_extend",
        "ladder_active": _lac(
            max_hold=900,
            stall_peak=0.15,
            stall_sec=45,
            when="mixed_wash_up",
            mf_flip=False,
        ),
        "delta_time_stop": {"enabled": False},
        "roi_time_stop": {"enabled": False},
    },
    # Combine soft giveback + wash stall.
    "gb15_p20__stall20_wash": {
        "hold_extend_max_giveback": 0.15,
        "hold_extend_giveback_min_peak": 0.20,
        "exit_mode": "hold_extend",
        "ladder_fallback_exit_mode": "hold_extend",
        "ladder_active": _lac(
            max_hold=900,
            stall_peak=0.20,
            stall_sec=60,
            when="mixed_wash_up",
            mf_flip=False,
        ),
        "delta_time_stop": {"enabled": False},
        "roi_time_stop": {"enabled": False},
    },
}


def _reason_counts(trades: pd.DataFrame) -> dict[str, int]:
    if trades is None or trades.empty or "reason" not in trades.columns:
        return {}
    return {str(k): int(v) for k, v in trades["reason"].value_counts().items()}


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
        "n_t30": int(reasons.get("T+30", 0)),
        "n_t45": int(reasons.get("T+45", 0)),
        "n_trail": int(reasons.get("TRAIL", 0) + reasons.get("TRAIL_LADDER", 0)),
        "n_stall": int(reasons.get("PROFIT_STALL", 0)),
        "n_sec_max": int(reasons.get("SEC_MAX", 0)),
        "reasons": reasons,
    }
    print(
        f"  ret={row['total_ret']:+.3f} dd={row['maxdd']:.3f} n={row['n_trades']} "
        f"win={row['trade_win']:.3f} "
        f"TP/T30/T45/STALL/SEC="
        f"{row['n_tp']}/{row['n_t30']}/{row['n_t45']}/{row['n_stall']}/{row['n_sec_max']}",
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

    base = {r["window"]: float(r["total_ret"]) for r in rows if r["variant"] == "baseline"}
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
    edf = pd.DataFrame(
        [{k: v for k, v in r.items() if k != "reasons"} for r in enriched]
    )
    edf.to_csv(out / "scoreboard.csv", index=False)
    (out / "scoreboard.json").write_text(
        json.dumps(enriched, indent=2, default=str), encoding="utf-8"
    )

    dual_ok = []
    for variant, g in edf[edf.variant != "baseline"].groupby("variant"):
        if set(g["window"]) >= set(wins) and (g["ret_retention"] >= 0.85).all():
            dual_ok.append(
                {
                    "variant": variant,
                    "min_retention": float(g["ret_retention"].min()),
                    "mean_retention": float(g["ret_retention"].mean()),
                    "may_jul_ret": float(
                        g.loc[g.window == "may_jul", "total_ret"].iloc[0]
                    )
                    if "may_jul" in set(g.window)
                    else None,
                    "jan_mar_ret": float(
                        g.loc[g.window == "jan_mar", "total_ret"].iloc[0]
                    )
                    if "jan_mar" in set(g.window)
                    else None,
                    "n_stall_total": int(g["n_stall"].sum()),
                    "n_t30_total": int(g["n_t30"].sum()),
                }
            )
    dual_df = (
        pd.DataFrame(dual_ok).sort_values(
            ["min_retention", "may_jul_ret"], ascending=[False, False]
        )
        if dual_ok
        else pd.DataFrame()
    )
    if len(dual_df):
        dual_df.to_csv(out / "dual_ok.csv", index=False)

    verdict = {
        "verdict": "PROMOTE_RESEARCH" if len(dual_df) else "REJECT_FOR_BASELINE",
        "dual_ok_variants": dual_df["variant"].tolist() if len(dual_df) else [],
        "best": dual_df.iloc[0].to_dict() if len(dual_df) else None,
        "thesis": (
            "Giveback-aware T30 extend refuses gift time after peak fade; "
            "profit_stall is a second-level path lock. Promote only if both "
            "windows retain >=0.85 of peer3 total_ret."
        ),
        "live_nvda_case": {
            "note": (
                "2026-07-27 NVDA peak_mfe=+32.8% then T+45 −16.9%: "
                "giveback gate fires at T+30 if peak−mtm already large; "
                "stall fires earlier if peak stalls before clock."
            )
        },
    }
    (out / "verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")
    lines = [
        "# Giveback extend + profit_stall ablation",
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
    ]
    (out / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(verdict, indent=2), flush=True)
    print("wrote", out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

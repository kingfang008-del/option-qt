#!/usr/bin/env python3
"""CORE C7 — hold-path fail-fast on the current spine (weak-window primary).

C3 STOCK_REV needed Mag7 wash breadth and fired 0 times in the weak window —
exactly the chop the live book now resembles. This step does **not** invent a
new detector. It overlays unused hold arms already in replay:

  - delta_time_stop: stock never confirms + option still red
  - mtm_floor: option still red after a min-hold (don't grind to T30)

PASS bar is **reweighted** (user: current tape ≈ weak window):
  PRIMARY weak: early-cut fires, clock-loss sum or MaxDD improves, keep>=0.85
  COST     strong: keep>=0.70 and TP count not gutted
Not the C2–C5 dual keep>=0.95 bar. Strong keep is a cost, not the objective.

No Rule-A change, no morph BLOCK, no AM densify.

Example:
  PYTHONPATH=. python -m maga7.tools.run_core_c7_hold_failfast \\
    --tag research_core_c7_hold_failfast
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
from maga7.tools.run_core_c3_stock_rev import (
    WINDOWS,
    _apply_overlay,
    _disable_am,
    keep_ratio,
    reason_stats,
)

PROFILE = (
    "maga7/CONFIG/strategy_profiles/"
    "single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1.json"
)
EARLY = {"DELTA_STOP", "MTM_FLOOR", "MAE_CUT", "ADVERSE_SOFT", "HOLD_SHOCK"}
CLOCK = {"T+30", "T+45", "TIME", "HOLD", "HOLD_EXTEND", "EXTEND"}
VARIANTS: dict[str, dict[str, Any]] = {
    "delta_stall_5m": {
        "delta_time_stop": {
            "enabled": True,
            "check_seconds": 300,
            "max_seconds": 900,
            "min_stock_move": 0.0015,
            "opt_mtm_max": 0.0,
            "note": "C7: stock stall + opt red. Existing arm, default was off.",
        }
    },
    "floor_red_h15": {
        "exit_mode": "hold_extend+mtm_floor",
        "mtm_floor_ret": 0.0,
        "exit_min_hold_minutes": 15,
        "note": "C7: still red at 15m → flatten. Don't grind to T30.",
    },
    "floor_m10_h10": {
        "exit_mode": "hold_extend+mtm_floor",
        "mtm_floor_ret": -0.10,
        "exit_min_hold_minutes": 10,
        "note": "C7: opt MTM<=-10% after 10m. Gap between toxic -25% and clock.",
    },
}


def _clock_loss_sum(trades: pd.DataFrame) -> float:
    if trades is None or trades.empty:
        return 0.0
    ret = pd.to_numeric(trades["ret"], errors="coerce")
    reason = trades["reason"].astype(str)
    mask = reason.isin(CLOCK) & (ret <= 0)
    return float(ret[mask].sum()) if mask.any() else 0.0


def _n_early(trades: pd.DataFrame) -> int:
    if trades is None or trades.empty:
        return 0
    return int(trades["reason"].astype(str).isin(EARLY).sum())


def _n_tp(trades: pd.DataFrame) -> int:
    if trades is None or trades.empty:
        return 0
    return int(trades["reason"].astype(str).eq("TP").sum())


def verdict_c7(
    *,
    weak_keep: float,
    strong_keep: float,
    weak_maxdd_delta: float,
    weak_clock_loss_delta: float,
    n_early_weak: int,
    n_tp_strong: int,
    n_tp_strong_base: int,
    min_weak_keep: float = 0.85,
    min_strong_keep: float = 0.70,
    min_early_weak: int = 2,
    min_tp_keep: float = 0.70,
) -> dict[str, Any]:
    """Weak-primary hold fail-fast. Strong keep is a cost ceiling."""
    if int(n_early_weak) < int(min_early_weak):
        return {"pass": False, "reason": "no_weak_early_cut"}
    grind_ok = float(weak_clock_loss_delta) > 1e-12 or float(weak_maxdd_delta) >= 0.005
    if not grind_ok:
        return {"pass": False, "reason": "weak_grind_not_improved"}
    if float(weak_keep) < float(min_weak_keep):
        return {"pass": False, "reason": "weak_keep_below_bar"}
    if float(strong_keep) < float(min_strong_keep):
        return {"pass": False, "reason": "strong_cost_too_high"}
    tp_floor = float(min_tp_keep) * max(int(n_tp_strong_base), 1)
    if int(n_tp_strong) + 1e-12 < tp_floor:
        return {"pass": False, "reason": "strong_tp_gutted"}
    return {"pass": True, "reason": "pass"}


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
    _disable_am(prof)
    prof["date_range"] = {"start": start, "end": end}
    trade = prof.setdefault("trade", {})
    _apply_overlay(trade, overlay)
    print(f"=== C7 {window} / {variant} {start}..{end} ===", flush=True)
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
        "trade_win": float(summary.get("trade_win") or 0.0),
        "n_early": _n_early(trades),
        "n_tp": _n_tp(trades),
        "clock_loss_sum": _clock_loss_sum(trades),
        "n_delta_stop": int(summary.get("n_delta_stop") or rs.get("reasons", {}).get("DELTA_STOP", 0) or 0),
        **{k: v for k, v in rs.items() if k != "reasons"},
        "reasons": rs["reasons"],
    }
    print(
        f"  ret={row['total_ret']:+.3f} n={row['n_trades']} dd={row['maxdd']:.3f} "
        f"early={row['n_early']} clock_loss={row['clock_loss_sum']:.3f} tp={row['n_tp']}",
        flush=True,
    )
    return row


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", default=PROFILE)
    ap.add_argument("--tag", default="research_core_c7_hold_failfast")
    ap.add_argument("--variants", default="all")
    args = ap.parse_args(argv)

    profile = load_profile(args.profile)
    out = Path(profile["_paths"]["results_dir"]) / args.tag
    out.mkdir(parents=True, exist_ok=True)

    names = list(VARIANTS)
    if args.variants not in {"all", "*", ""}:
        names = [x.strip() for x in args.variants.split(",") if x.strip()]
        missing = [n for n in names if n not in VARIANTS]
        if missing:
            raise SystemExit(f"unknown variants {missing}")

    rows: list[dict[str, Any]] = []
    baseline: dict[str, dict[str, Any]] = {}
    for window in WINDOWS:
        base = run_one(window, "baseline", {}, out_root=out, profile_path=args.profile)
        base["keep"] = 1.0
        baseline[window] = base
        rows.append(base)
        for name in names:
            row = run_one(window, name, VARIANTS[name], out_root=out, profile_path=args.profile)
            row["keep"] = keep_ratio(row["total_ret"], base["total_ret"])
            row["maxdd_delta"] = float(row["maxdd"] - base["maxdd"])
            row["clock_loss_delta"] = float(row["clock_loss_sum"] - base["clock_loss_sum"])
            rows.append(row)

    sb = pd.DataFrame(rows)
    sb.to_csv(out / "scoreboard.csv", index=False)

    adopted = []
    for name in names:
        weak = next(r for r in rows if r["window"] == "weak" and r["variant"] == name)
        strong = next(r for r in rows if r["window"] == "strong" and r["variant"] == name)
        bw, bs = baseline["weak"], baseline["strong"]
        v = verdict_c7(
            weak_keep=float(weak["keep"]),
            strong_keep=float(strong["keep"]),
            weak_maxdd_delta=float(weak["maxdd"] - bw["maxdd"]),
            weak_clock_loss_delta=float(weak["clock_loss_sum"] - bw["clock_loss_sum"]),
            n_early_weak=int(weak["n_early"]),
            n_tp_strong=int(strong["n_tp"]),
            n_tp_strong_base=int(bs["n_tp"]),
        )
        adopted.append(
            {
                "variant": name,
                "pass": bool(v["pass"]),
                "reason": v["reason"],
                "weak_keep": float(weak["keep"]),
                "strong_keep": float(strong["keep"]),
                "weak_maxdd_delta": float(weak["maxdd"] - bw["maxdd"]),
                "weak_clock_loss_delta": float(weak["clock_loss_sum"] - bw["clock_loss_sum"]),
                "n_early_weak": int(weak["n_early"]),
                "n_early_strong": int(strong["n_early"]),
                "n_tp_strong": int(strong["n_tp"]),
                "n_tp_strong_base": int(bs["n_tp"]),
                "weak_total_ret": float(weak["total_ret"]),
                "strong_total_ret": float(strong["total_ret"]),
            }
        )
    passed = [a for a in adopted if a["pass"]]
    passed.sort(
        key=lambda x: (x["weak_clock_loss_delta"], x["weak_maxdd_delta"], x["strong_keep"]),
        reverse=True,
    )
    promote = f"C7_{passed[0]['variant']}" if passed else "NONE"
    summary = {
        "protocol": "core_c7_hold_failfast",
        "promotion_mark": "weak_primary_hold_cut",
        "pass_rule": (
            "n_early_weak>=2 AND (clock_loss_sum↑ OR weak MaxDD Δ>=50bp) "
            "AND weak keep>=0.85 AND strong keep>=0.70 AND strong TP keep>=0.70"
        ),
        "note": (
            "Bar reweighted: weak window is the live analogue. "
            "Strong keep is a cost ceiling, not C2–C5 keep>=0.95. "
            "Do not add morph BLOCKs. Production freeze unchanged."
        ),
        "variants": adopted,
        "promote": promote,
        "pass": bool(promote != "NONE"),
        "baseline": {
            w: {k: baseline[w][k] for k in ("total_ret", "maxdd", "n_trades", "n_tp", "clock_loss_sum", "n_early")}
            for w in WINDOWS
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    lines = [
        "# CORE C7 — 持仓失效快退（弱窗主门）",
        "",
        "- 问题：红单磨满 T+30（硬抗），不是再加 morph",
        "- 对照：当前 research_baseline（含 C5 drop3 + C6 DD 预算）",
        "- 臂：已有 `delta_time_stop` / `mtm_floor`（默认曾 off）",
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
            f"**C7 PASS** → `{best['variant']}` weak keep={best['weak_keep']:.3f} · "
            f"strong keep={best['strong_keep']:.3f} · "
            f"weak clock_loss Δ={best['weak_clock_loss_delta']:+.3f} · "
            f"early weak/strong={best['n_early_weak']}/{best['n_early_strong']}。",
            "弱窗主门。生产 freeze 不动。",
        ]
    else:
        lines += [
            "",
            "## 结论",
            "",
            "**C7 FAIL** — 现有持仓臂仍分不开弱窗硬抗与强窗延迟 TP。不接线。不把 Jul-22 STOCK_REV 再拧一遍。",
        ]
    (out / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"promote": promote, "pass": summary["pass"], "variants": adopted}, indent=2, default=str))
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

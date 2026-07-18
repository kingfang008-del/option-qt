#!/usr/bin/env python3
"""P0 hold-out: freeze params, score L0 / L1 / L2 on unseen windows.

No knob search. Writes ``results/watchdog/holdout_p0/``.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from maga7.common.config import load_profile
from maga7.common.replay import run_offline_replay

# Locked research params — do not edit during this run.
WINDOWS = [
    {
        "name": "holdout_2026_01",
        "start": "2026-01-02",
        "end": "2026-01-30",
        "baseline_profile": "maga7/CONFIG/strategy_profiles/single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1.json",
        "note": "Before Feb–Apr / May–Jul tuning windows",
    },
    {
        "name": "holdout_2025_h2",
        "start": "2025-07-01",
        "end": "2025-12-31",
        "baseline_profile": "maga7/CONFIG/strategy_profiles/single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_2025h2_v1.json",
        "note": "Prior-year H2 OOS; dedicated 2025h2 lock map",
    },
]

L1_PROFILE = "maga7/CONFIG/strategy_profiles/single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_watchdog_v1.json"
L2_PROFILE = "maga7/CONFIG/strategy_profiles/single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_watchdog_hunter_washout_reclaim_v1.json"
GATE_VS_BASE = 0.90


def _tot(daily: pd.DataFrame) -> float:
    eq = 1.0
    for r in daily["day_ret"].astype(float):
        eq *= 1.0 + float(r)
    return eq - 1.0


def _attach_overlay(base: dict, overlay_src: dict, *, hunter: bool) -> dict:
    """Copy paths/date/symbols from baseline window profile; attach watchdog from research."""
    p = copy.deepcopy(base)
    wd = copy.deepcopy(overlay_src.get("watchdog") or {})
    if not hunter:
        hunt = dict(wd.get("hunter") or {})
        hunt["enabled"] = False
        wd["hunter"] = hunt
    p["watchdog"] = wd
    # keep baseline paths (esp. 2025h2 lock map)
    return p


def _run(prof: dict, *, start: str, end: str, tag: str, out: Path) -> dict:
    p = copy.deepcopy(prof)
    p["date_range"] = {"start": start, "end": end}
    res = run_offline_replay(p, scheme="single")
    s = res["summary"]
    daily = res["daily"].copy()
    trades = res["trades"].copy()
    sub = out / tag
    sub.mkdir(parents=True, exist_ok=True)
    total = _tot(daily)
    s = dict(s)
    s["total_ret"] = total
    (sub / "summary.json").write_text(json.dumps(s, indent=2, default=str), encoding="utf-8")
    daily.to_csv(sub / "daily.csv", index=False)
    trades.to_csv(sub / "trades.csv", index=False)
    return {"summary": s, "daily": daily, "trades": trades, "total_ret": total}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="maga7/results/watchdog/holdout_p0")
    ap.add_argument("--gate", type=float, default=GATE_VS_BASE)
    args = ap.parse_args()

    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)

    l1_src = load_profile(L1_PROFILE)
    l2_src = load_profile(L2_PROFILE)

    board: list[dict] = []
    for w in WINDOWS:
        print(f"== {w['name']} {w['start']}..{w['end']}", flush=True)
        base = load_profile(w["baseline_profile"])

        print("  L0 baseline", flush=True)
        b = _run(base, start=w["start"], end=w["end"], tag=f"{w['name']}_L0", out=out)
        b_ret = b["total_ret"]

        print("  L1 watchdog (hunter off)", flush=True)
        p1 = _attach_overlay(base, l1_src, hunter=False)
        r1 = _run(p1, start=w["start"], end=w["end"], tag=f"{w['name']}_L1", out=out)

        print("  L2 + washout_reclaim v2", flush=True)
        p2 = _attach_overlay(base, l2_src, hunter=True)
        r2 = _run(p2, start=w["start"], end=w["end"], tag=f"{w['name']}_L2", out=out)

        for variant, rr in (("L0", b), ("L1", r1), ("L2", r2)):
            s = rr["summary"]
            tr = rr["total_ret"]
            vs = (tr / b_ret) if abs(b_ret) > 1e-12 else None
            board.append(
                {
                    "window": w["name"],
                    "start": w["start"],
                    "end": w["end"],
                    "note": w["note"],
                    "variant": variant,
                    "total_ret": tr,
                    "maxdd": s.get("maxdd"),
                    "n_trades": s.get("n_trades"),
                    "n_hunt_trades": s.get("n_hunt_trades"),
                    "watchdog_state_counts": s.get("watchdog_state_counts") or {},
                    "vs_L0": 1.0 if variant == "L0" else vs,
                    "pass_gate": True
                    if variant == "L0"
                    else (vs is not None and vs >= float(args.gate)),
                    "trade_exp": s.get("trade_exp"),
                    "trade_win": s.get("trade_win"),
                }
            )

    df = pd.DataFrame(board)
    df.to_csv(out / "scoreboard.csv", index=False)
    (out / "scoreboard.json").write_text(json.dumps(board, indent=2, default=str), encoding="utf-8")

    # Verdict: every L1/L2 row must pass gate
    checks = [r for r in board if r["variant"] in {"L1", "L2"}]
    all_pass = all(bool(r["pass_gate"]) for r in checks)
    # Soft note if L2 fails but L1 passes
    l1_ok = all(r["pass_gate"] for r in checks if r["variant"] == "L1")
    l2_ok = all(r["pass_gate"] for r in checks if r["variant"] == "L2")
    if all_pass:
        verdict = "PASS_P0"
    elif l1_ok and not l2_ok:
        verdict = "PASS_L1_ONLY"
    else:
        verdict = "FAIL_P0"

    summary = {
        "verdict": verdict,
        "gate_vs_L0": float(args.gate),
        "rule": "No retune. L1/L2 must keep >= gate * L0 total_ret on each hold-out window.",
        "params_locked": {
            "L1": "degrade reclaim_disp55 + halt washout_and_reclaim",
            "L2": "washout_reclaim wd=0.015 mutex_scope=symbol_dir allow_baseline_opposite=true",
        },
        "l1_pass": l1_ok,
        "l2_pass": l2_ok,
        "scoreboard": board,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    md = [
        "# P0 Hold-out scoreboard",
        "",
        f"**Verdict: `{verdict}`** (gate vs L0 ≥ {args.gate:.0%})",
        "",
        "Params frozen — no search.",
        "",
        "| window | variant | total_ret | MaxDD | n_trades | n_hunt | vs_L0 | pass |",
        "|--------|---------|-----------|-------|----------|--------|-------|------|",
    ]
    for r in board:
        md.append(
            f"| {r['window']} | {r['variant']} | {r['total_ret']:.3f} | {r['maxdd']} | "
            f"{r['n_trades']} | {r['n_hunt_trades']} | {r['vs_L0']} | {r['pass_gate']} |"
        )
    md += [
        "",
        "## Notes",
        "",
        "- `holdout_2026_01`: before Feb–Apr / May–Jul tuning windows.",
        "- `holdout_2025_h2`: prior-year OOS with 2025h2 lock map.",
        "- `PASS_L1_ONLY`: keep L1 research path; demote L2 to observe.",
        "- `FAIL_P0`: do not promote overlays; stay L0 default.",
        "",
    ]
    (out / "HOLDOUT.md").write_text("\n".join(md), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

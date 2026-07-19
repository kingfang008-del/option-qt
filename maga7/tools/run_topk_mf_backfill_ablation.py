#!/usr/bin/env python3
"""Ablate TopK backfill variants on research baseline (dual window).

Variants:
  - baseline: earliest TopK (freeze default)
  - time_backfill: topk_backfill_on_block (all_first time order)
  - mf_backfill: topk_mf_backfill_on_block (TopK first, then MF-ranked remainder)
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

PROFILE = (
    "maga7/CONFIG/strategy_profiles/"
    "single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1.json"
)
WINDOWS = {
    "strong": ("2026-05-01", "2026-07-17"),
    "weak": ("2026-02-01", "2026-04-30"),
    "focus": ("2026-07-07", "2026-07-10"),
}


def _tot(daily: pd.DataFrame) -> float:
    eq = 1.0
    for r in daily["day_ret"].astype(float):
        eq *= 1.0 + float(r)
    return eq - 1.0


def _run(prof: dict, *, start: str, end: str) -> dict:
    p = copy.deepcopy(prof)
    p["date_range"] = {"start": start, "end": end}
    res = run_offline_replay(p, scheme="single")
    s = res["summary"]
    daily = res["daily"]
    trades = res["trades"]
    total = _tot(daily) if not daily.empty else float(s["total_ret"])
    focus = {}
    if not trades.empty and "date" in trades.columns:
        t = trades[trades["date"].astype(str).between("2026-07-07", "2026-07-10")]
        if len(t):
            cols = [c for c in ["date", "symbol", "dir", "ret", "reason", "topk_backfill"] if c in t.columns]
            focus = t[cols].to_dict(orient="records")
    return {
        "total_ret": total,
        "maxdd": float(s["maxdd"]),
        "n_trades": int(s["n_trades"]),
        "n_topk_backfill": int(s.get("n_topk_backfill") or 0),
        "n_hunt_trades": int(s.get("n_hunt_trades") or 0),
        "focus_trades": focus,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", default=PROFILE)
    ap.add_argument("--out", default="maga7/results/topk_mf_backfill_ablation")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    base = load_profile(args.profile)
    variants = {
        "baseline": {},
        "time_backfill": {"topk_backfill_on_block": True},
        "mf_backfill": {"topk_mf_backfill_on_block": True},
    }

    rows = []
    focus_dump = {}
    for vname, patch in variants.items():
        for wname, (start, end) in WINDOWS.items():
            print(f"run {vname} {wname}...")
            prof = copy.deepcopy(base)
            trade = prof.setdefault("trade", {})
            trade["topk_backfill_on_block"] = False
            trade["topk_mf_backfill_on_block"] = False
            trade.update(patch)
            r = _run(prof, start=start, end=end)
            rows.append({"variant": vname, "window": wname, **{k: r[k] for k in r if k != "focus_trades"}})
            if wname == "focus":
                focus_dump[vname] = r["focus_trades"]
            print(
                f"  ret={r['total_ret']:.4f} dd={r['maxdd']:.4f} n={r['n_trades']} "
                f"backfill={r['n_topk_backfill']}"
            )

    df = pd.DataFrame(rows)
    # vs baseline per window
    for w in df["window"].unique():
        b = float(df[(df.window == w) & (df.variant == "baseline")]["total_ret"].iloc[0])
        m = df["window"] == w
        df.loc[m, "vs_baseline"] = df.loc[m, "total_ret"].map(
            lambda x, b=b: (1.0 + float(x)) / (1.0 + b) if (1.0 + b) != 0 else float("nan")
        )
    df.to_csv(out / "scoreboard.csv", index=False)
    summary = {
        "profile": args.profile,
        "gate_note": "mf_backfill = TopK-first then MF-ranked remainder; slots only on clears",
        "scoreboard": df.to_dict(orient="records"),
        "focus_trades": focus_dump,
    }
    # promote heuristic: strong vs baseline >=0.95 and weak vs baseline >=0.95
    piv = df.pivot(index="variant", columns="window", values="vs_baseline")
    promote = []
    for v in ("time_backfill", "mf_backfill"):
        if v not in piv.index:
            continue
        ok = True
        for w in ("strong", "weak"):
            if w in piv.columns and float(piv.loc[v, w]) < 0.95:
                ok = False
        if ok:
            promote.append(v)
    summary["promote_candidates"] = promote
    summary["verdict"] = "PASS_CANDIDATE" if promote else "REJECT_FOR_BASELINE"
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(json.dumps({"verdict": summary["verdict"], "promote": promote}, indent=2))
    print(df.to_string(index=False))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

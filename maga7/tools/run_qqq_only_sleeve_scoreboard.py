#!/usr/bin/env python3
"""QQQ-only sleeve: L0 vs L0+Hunt on available short-DTE window."""
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

L0 = "maga7/CONFIG/strategy_profiles/qqq_only_open_ladder_atm5otm_extend_mtm_v1.json"
L2 = "maga7/CONFIG/strategy_profiles/qqq_only_open_ladder_atm5otm_extend_mtm_watchdog_hunter_v1.json"


def _tot(daily: pd.DataFrame) -> float:
    eq = 1.0
    for r in daily["day_ret"].astype(float):
        eq *= 1.0 + float(r)
    return eq - 1.0


def _run(path: str, *, start: str, end: str) -> dict:
    prof = load_profile(path)
    p = copy.deepcopy(prof)
    p["date_range"] = {"start": start, "end": end}
    res = run_offline_replay(p, scheme="single")
    s = res["summary"]
    daily = res["daily"]
    trades = res["trades"]
    total = _tot(daily) if not daily.empty else float(s["total_ret"])
    n_hunt = 0
    hunt_mean = float("nan")
    if not trades.empty and "route" in trades.columns:
        h = trades[trades["route"].astype(str) == "hunt"]
        n_hunt = int(len(h))
        if n_hunt:
            hunt_mean = float(h["ret"].mean())
    return {
        "profile": Path(path).stem,
        "total_ret": total,
        "maxdd": float(s["maxdd"]),
        "n_trades": int(s["n_trades"]),
        "n_hunt_trades": n_hunt,
        "hunt_mean_ret": hunt_mean,
        "end_equity": float(s.get("end_equity") or 100.0 * (1.0 + total)),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", default="2026-05-01")
    ap.add_argument("--end", default="2026-07-13")
    ap.add_argument("--out", default="maga7/results/qqq_only_sleeve_may_jul")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for label, path in (("L0", L0), ("L2_hunt", L2)):
        print(f"run {label}...")
        r = _run(path, start=args.start, end=args.end)
        r["label"] = label
        rows.append(r)
        print(
            f"  ret={r['total_ret']:.4f} n={r['n_trades']} hunt={r['n_hunt_trades']} "
            f"maxdd={r['maxdd']:.4f}"
        )

    df = pd.DataFrame(rows)
    l0 = float(df.loc[df["label"] == "L0", "total_ret"].iloc[0])
    df["vs_L0"] = df["total_ret"].map(lambda x: (1.0 + float(x)) / (1.0 + l0) if (1.0 + l0) != 0 else float("nan"))
    df.to_csv(out / "scoreboard.csv", index=False)
    summary = {
        "window": f"{args.start}..{args.end}",
        "note": "Feb–Apr skipped: QQQ open_lock + 1s quotes currently May–Jul only (→07-13).",
        "rows": rows,
        "L2_vs_L0": float(df.loc[df["label"] == "L2_hunt", "vs_L0"].iloc[0]),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

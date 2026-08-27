#!/usr/bin/env python3
"""Dual-window: UP-only overnight gap-up degrade vs peer3 OFF.

Rule: gap ≥ thr and direction=UP → size × scale (DN untouched).

Windows (standard):
  weak    2026-02-01 .. 2026-04-30
  strong  2026-05-01 .. 2026-07-23

Pass wire: strong vs_OFF>1 and weak keep≥0.95
Pass research: strong vs_OFF>1 and weak keep≥0.85
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from maga7.common.config import load_profile
from maga7.common.replay import run_offline_replay

BASE = (
    "maga7/CONFIG/strategy_profiles/"
    "single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1.json"
)

WINDOWS = (
    ("weak_feb_apr", "2026-02-01", "2026-04-30"),
    ("strong_may_jul23", "2026-05-01", "2026-07-23"),
)


def _gate(**kw: Any) -> dict[str, Any]:
    cfg = {
        "enabled": True,
        "up_only": True,
        "mode": "scale",
        "scale": 0.5,
        "max_fav_gap": 0.04,
    }
    cfg.update(kw)
    return cfg


ARMS: dict[str, dict[str, Any] | None] = {
    "OFF": {"enabled": False},
    "UP04_X50": _gate(max_fav_gap=0.04, scale=0.5),
    "UP04_X25": _gate(max_fav_gap=0.04, scale=0.25),
    "UP035_X50": _gate(max_fav_gap=0.035, scale=0.5),
    "UP05_X50": _gate(max_fav_gap=0.05, scale=0.5),
}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", default=BASE)
    ap.add_argument("--out", default="maga7/results/overnight_gap_up_degrade_dual_v1")
    ap.add_argument("--arms", default=",".join(ARMS.keys()))
    args = ap.parse_args(argv)

    base = load_profile(args.profile)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    arm_names = [a.strip() for a in args.arms.split(",") if a.strip()]

    rows: list[dict[str, Any]] = []
    for arm in arm_names:
        if arm not in ARMS:
            print(f"skip {arm}", flush=True)
            continue
        for wname, start, end in WINDOWS:
            p = copy.deepcopy(base)
            p["date_range"] = {"start": start, "end": end}
            p.setdefault("trade", {})["overnight_gap_gate"] = copy.deepcopy(ARMS[arm])
            tag = f"{arm}__{wname}"
            print(f"=== {tag} ===", flush=True)
            res = run_offline_replay(p, scheme="single")
            sub = out / tag
            sub.mkdir(parents=True, exist_ok=True)
            (sub / "summary.json").write_text(
                json.dumps(res["summary"], indent=2, default=str), encoding="utf-8"
            )
            res["daily"].to_csv(sub / "daily.csv", index=False)
            res["trades"].to_csv(sub / "trades.csv", index=False)
            s = res["summary"]
            d = res["daily"].copy()
            d["date"] = pd.to_datetime(d["date"]).dt.strftime("%Y-%m-%d")
            jul = d[(d.date >= "2026-07-10") & (d.date <= "2026-07-23")]
            jul_ret = (
                float((1 + jul["day_ret"].astype(float)).prod() - 1) if len(jul) else None
            )
            r = {
                "arm": arm,
                "window": wname,
                "total_ret": float(s.get("total_ret") or 0),
                "maxdd": float(s.get("maxdd") or 0),
                "n_trades": int(s.get("n_trades") or 0),
                "trade_win": s.get("trade_win"),
                "day_win": s.get("day_win"),
                "n_overnight_gap_block": s.get("n_overnight_gap_block"),
                "n_overnight_gap_scale": s.get("n_overnight_gap_scale"),
                "jul10_23_ret": jul_ret,
            }
            rows.append(r)
            print(
                f"  ret={r['total_ret']:.3f} maxdd={r['maxdd']:.3f} n={r['n_trades']} "
                f"scale={r['n_overnight_gap_scale']} jul={jul_ret}",
                flush=True,
            )

    board = pd.DataFrame(rows)
    off = {r["window"]: float(r["total_ret"]) for r in rows if r["arm"] == "OFF"}
    board["vs_OFF"] = board.apply(
        lambda r: float(r["total_ret"]) / off[r["window"]] if off.get(r["window"]) else None,
        axis=1,
    )
    board.to_csv(out / "scoreboard.csv", index=False)

    verdict = "DUAL_FAIL"
    best = None
    for arm in arm_names:
        if arm == "OFF":
            continue
        sub = board[board.arm == arm]
        if len(sub) < 2:
            continue
        weak = float(sub[sub.window == "weak_feb_apr"]["vs_OFF"].iloc[0])
        strong = float(sub[sub.window == "strong_may_jul23"]["vs_OFF"].iloc[0])
        print(f"{arm}: weak={weak:.3f} strong={strong:.3f}", flush=True)
        if strong > 1.0 and weak >= 0.95:
            verdict, best = "DUAL_PASS_WIRE", arm
            break
        if strong > 1.0 and weak >= 0.85 and verdict == "DUAL_FAIL":
            verdict, best = "DUAL_PASS_RESEARCH", arm

    summary = {
        "verdict": verdict,
        "best": best,
        "rule": "up_only + mode=scale: gap≥thr & UP → size×scale",
        "windows": [list(w) for w in WINDOWS],
        "scoreboard": board.to_dict(orient="records"),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print("\n=== scoreboard ===", flush=True)
    print(board.to_string(index=False), flush=True)
    print(f"\nverdict={verdict} best={best}", flush=True)
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

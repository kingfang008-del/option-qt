#!/usr/bin/env python3
"""Dual-window: range_stall_gate vs wired spine (og+peer_gap).

Rule: chase≥thr & pre5≤0 → block; optional peer_pre5 arm / max_peer / min_ffo.
Windows: weak Feb–Apr / strong May–Jul23
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
FOCUS = [
    "2026-02-06",
    "2026-02-17",
    "2026-02-25",
    "2026-02-26",
    "2026-03-12",
    "2026-03-16",
    "2026-03-18",
    "2026-04-06",
    "2026-04-23",
    "2026-06-11",
]


def _gate(**kw: Any) -> dict[str, Any]:
    cfg = {
        "enabled": True,
        "mode": "block",
        "min_chase": 0.9,
        # Near-flat pre5 (bp): 0 alone misses 03-18 AMZN ≈+0bp noise.
        "max_pre5": 0.0002,
        "pre_seconds": 300,
    }
    cfg.update(kw)
    return cfg


ARMS = {
    "OFF": {"enabled": False},
    # Measure at final entry clock (after confirms) — see replay wiring.
    "RS90": _gate(max_pre5=0.0),
    "RS90_FLAT": _gate(max_pre5=0.0002),
    "RS90_P5": _gate(max_peer=5, max_pre5=0.0002),
    "RS90_FFO2_P5": _gate(max_peer=5, min_fav_from_open=0.02, max_pre5=0.0002),
    "RS90_P3PRE": _gate(peer_pre5_max_peer=3, max_pre5=0.0),
    "RS90_UNI": _gate(max_peer=5, peer_pre5_max_peer=3, max_pre5=0.0002),
}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", default=BASE)
    ap.add_argument("--out", default="maga7/results/range_stall_dual_v1")
    ap.add_argument("--arms", default=",".join(ARMS.keys()))
    args = ap.parse_args(argv)

    base = load_profile(args.profile)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for arm in [a.strip() for a in args.arms.split(",") if a.strip()]:
        if arm not in ARMS:
            continue
        for wname, start, end in WINDOWS:
            p = copy.deepcopy(base)
            p["date_range"] = {"start": start, "end": end}
            p.setdefault("trade", {})["range_stall_gate"] = copy.deepcopy(ARMS[arm])
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
            focus = d[d.date.isin(FOCUS)]
            r = {
                "arm": arm,
                "window": wname,
                "total_ret": float(s.get("total_ret") or 0),
                "maxdd": float(s.get("maxdd") or 0),
                "n_trades": int(s.get("n_trades") or 0),
                "n_range_stall_block": s.get("n_range_stall_block"),
                "focus_day_ret_sum": float(focus["day_ret"].astype(float).sum())
                if len(focus)
                else None,
            }
            rows.append(r)
            print(
                f"  ret={r['total_ret']:.3f} maxdd={r['maxdd']:.3f} "
                f"blk={r['n_range_stall_block']} focus={r['focus_day_ret_sum']}",
                flush=True,
            )

    board = pd.DataFrame(rows)
    off = {r["window"]: r["total_ret"] for r in rows if r["arm"] == "OFF"}
    board["vs_OFF"] = board.apply(
        lambda r: r["total_ret"] / off[r["window"]] if off.get(r["window"]) else None,
        axis=1,
    )
    board.to_csv(out / "scoreboard.csv", index=False)

    verdict, best = "DUAL_FAIL", None
    for arm in [a for a in ARMS if a != "OFF"]:
        if arm not in set(board.arm):
            continue
        sub = board[board.arm == arm]
        weak = float(sub[sub.window == "weak_feb_apr"].vs_OFF.iloc[0])
        strong = float(sub[sub.window == "strong_may_jul23"].vs_OFF.iloc[0])
        print(f"{arm}: weak={weak:.3f} strong={strong:.3f}", flush=True)
        if strong > 1.0 and weak >= 0.95:
            verdict, best = "DUAL_PASS_WIRE", arm
            break
        if strong > 1.0 and weak >= 0.85 and verdict == "DUAL_FAIL":
            verdict, best = "DUAL_PASS_RESEARCH", arm

    summary = {
        "verdict": verdict,
        "best": best,
        "rule": "chase>=min & pre5<=max → block (+ optional peer_pre5 arm)",
        "scoreboard": board.to_dict(orient="records"),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(board.to_string(index=False), flush=True)
    print(f"verdict={verdict} best={best}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

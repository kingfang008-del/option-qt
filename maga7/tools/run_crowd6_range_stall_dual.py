#!/usr/bin/env python3
"""Dual-window: range_stall crowd_min_peer 7→6 (06-24 GOOGL peer=6 hole).

Keeps wired RS otherwise. Target: block GOOGL 06-24 tox, keep AMZN same-day TP.
Windows: weak Feb–Apr / strong May–Jul24
Wire: strong vs_OFF≥1 and weak keep≥0.95 (survive-first 0.97 preferred)
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
    ("strong_may_jul24", "2026-05-01", "2026-07-24"),
)
FOCUS = ["2026-06-24"]


def _rs(crowd_min: int) -> dict[str, Any]:
    return {
        "enabled": True,
        "mode": "block",
        "min_chase": 0.9,
        "max_pre5": 0.00025,
        "pre_seconds": 300,
        "max_peer": 5,
        "min_fav_from_open": 0.012,
        "peer_pre5_max_peer": 3,
        "crowd_min_peer": int(crowd_min),
        "crowd_max_pre5": 0.0025,
        "crowd_min_fav_from_open": 0.01,
    }


ARMS = {
    "OFF": _rs(7),  # wired spine
    "CROWD6": _rs(6),
}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", default=BASE)
    ap.add_argument(
        "--out", default="/mnt/s990/data/maga7/results/crowd6_range_stall_dual_v1"
    )
    args = ap.parse_args(argv)

    base = load_profile(args.profile)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    for arm, cfg in ARMS.items():
        for wname, start, end in WINDOWS:
            p = copy.deepcopy(base)
            p["date_range"] = {"start": start, "end": end}
            p.setdefault("trade", {})["range_stall_gate"] = copy.deepcopy(cfg)
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
            tr = res["trades"].copy()
            if len(tr):
                tr["date"] = pd.to_datetime(tr["date"]).dt.strftime("%Y-%m-%d")
            g = tr[
                (tr["date"] == "2026-06-24")
                & (tr["symbol"].astype(str).str.upper() == "GOOGL")
            ] if len(tr) else tr
            a = tr[
                (tr["date"] == "2026-06-24")
                & (tr["symbol"].astype(str).str.upper() == "AMZN")
            ] if len(tr) else tr
            r = {
                "arm": arm,
                "window": wname,
                "total_ret": float(s.get("total_ret") or 0),
                "maxdd": float(s.get("maxdd") or 0),
                "n_trades": int(s.get("n_trades") or 0),
                "n_range_stall_block": s.get("n_range_stall_block"),
                "n_googl_0624": int(len(g)),
                "n_amzn_0624": int(len(a)),
                "amzn_0624_ret": float(a["ret"].iloc[0]) if len(a) else None,
            }
            rows.append(r)
            print(
                f"  ret={r['total_ret']:.3f} maxdd={r['maxdd']:.3f} blk={r['n_range_stall_block']} "
                f"GOOGL0624={r['n_googl_0624']} AMZN0624={r['n_amzn_0624']}",
                flush=True,
            )

    board = pd.DataFrame(rows)
    off = {r["window"]: r["total_ret"] for r in rows if r["arm"] == "OFF"}
    board["vs_OFF"] = board.apply(
        lambda r: (r["total_ret"] / off[r["window"]])
        if off.get(r["window"]) not in (None, 0)
        else None,
        axis=1,
    )
    board.to_csv(out / "scoreboard.csv", index=False)

    verdict, best = "DUAL_FAIL", None
    for arm in [a for a in ARMS if a != "OFF"]:
        sub = board[board.arm == arm]
        weak = float(sub[sub.window == "weak_feb_apr"].vs_OFF.iloc[0])
        strong = float(sub[sub.window == "strong_may_jul24"].vs_OFF.iloc[0])
        srow = sub[sub.window == "strong_may_jul24"].iloc[0]
        clears = int(srow["n_googl_0624"]) == 0
        keeps_amzn = int(srow["n_amzn_0624"]) >= 1
        print(
            f"{arm}: weak={weak:.3f} strong={strong:.3f} "
            f"clear_GOOGL={clears} keep_AMZN={keeps_amzn}",
            flush=True,
        )
        if not (clears and keeps_amzn):
            continue
        if strong >= 1.0 and weak >= 0.97:
            verdict, best = "DUAL_PASS_WIRE", arm
            break
        if strong >= 1.0 and weak >= 0.95:
            verdict, best = "DUAL_PASS_WIRE", arm
        if strong >= 0.95 and weak >= 0.85 and verdict == "DUAL_FAIL":
            verdict, best = "DUAL_PASS_RESEARCH", arm

    summary = {
        "verdict": verdict,
        "best": best,
        "rule": "range_stall crowd_min_peer: 7 → 6 (peer=6 hole on 06-24 GOOGL)",
        "scoreboard": board.to_dict(orient="records"),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    (out / "verdict.json").write_text(
        json.dumps({"verdict": verdict, "best": best}, indent=2), encoding="utf-8"
    )
    print(board.to_string(index=False), flush=True)
    print(f"verdict={verdict} best={best}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

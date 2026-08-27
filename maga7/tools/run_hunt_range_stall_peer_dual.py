#!/usr/bin/env python3
"""Dual-window: Hunt uses peer_n for range_stall (06-24 AMD).

Does NOT change CORE peer thresholds. Arms only flip
``trade.range_stall_gate.hunt_peer_align`` on the wired spine gate.

Windows: weak Feb–Apr / strong May–Jul24
Pass wire: strong vs_OFF ≥ 1 and weak keep ≥ 0.95 (survive-first 0.97 ok)
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
FOCUS = [
    "2026-06-24",  # AMD Hunt washout reclaim −26%
    "2026-06-03",
    "2026-05-28",
]


def _spine_rs(*, hunt_peer: bool, hunt_asof: str | None = None) -> dict[str, Any]:
    """Wired C7_FFO012_P25 + CROWD25, optional Hunt-only RS peer/asof."""
    cfg: dict[str, Any] = {
        "enabled": True,
        "mode": "block",
        "min_chase": 0.9,
        "max_pre5": 0.00025,
        "pre_seconds": 300,
        "max_peer": 5,
        "min_fav_from_open": 0.012,
        "peer_pre5_max_peer": 3,
        "crowd_min_peer": 7,
        "crowd_max_pre5": 0.0025,
        "crowd_min_fav_from_open": 0.01,
        "hunt_peer_align": bool(hunt_peer),
    }
    if hunt_asof:
        cfg["hunt_asof"] = hunt_asof
    return cfg


ARMS = {
    # Current: Hunt peer_n=None for RS; peer_gap also sees None.
    "OFF": _spine_rs(hunt_peer=False),
    # Peer for RS only (no peer_gap side-effect); decision clock (misses 06-24).
    "HUNT_RS_PEER": _spine_rs(hunt_peer=True),
    # Peer for RS + measure at hunter signal_deadline (~10:00 fill clock).
    "HUNT_RS_PEER_T10": _spine_rs(hunt_peer=True, hunt_asof="signal_deadline"),
}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", default=BASE)
    ap.add_argument(
        "--out",
        default="/mnt/s990/data/maga7/results/hunt_range_stall_peer_dual_v1",
    )
    ap.add_argument("--arms", default=",".join(ARMS.keys()))
    args = ap.parse_args(argv)

    base = load_profile(args.profile)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

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
            trades = res["trades"].copy()
            if len(trades) and "date" in trades.columns:
                trades["date"] = pd.to_datetime(trades["date"]).dt.strftime("%Y-%m-%d")
            amd_0624 = None
            if len(trades):
                hit = trades[
                    (trades["date"] == "2026-06-24")
                    & (trades["symbol"].astype(str).str.upper() == "AMD")
                ]
                if "route" in hit.columns:
                    hit = hit[hit["route"].astype(str).str.lower() == "hunt"]
                amd_0624 = int(len(hit))
            r = {
                "arm": arm,
                "window": wname,
                "total_ret": float(s.get("total_ret") or 0),
                "maxdd": float(s.get("maxdd") or 0),
                "n_trades": int(s.get("n_trades") or 0),
                "n_hunt_trades": s.get("n_hunt_trades"),
                "n_range_stall_block": s.get("n_range_stall_block"),
                "focus_day_ret_sum": float(focus["day_ret"].astype(float).sum())
                if len(focus)
                else None,
                "n_amd_hunt_0624": amd_0624,
            }
            rows.append(r)
            print(
                f"  ret={r['total_ret']:.3f} maxdd={r['maxdd']:.3f} "
                f"blk={r['n_range_stall_block']} hunt={r['n_hunt_trades']} "
                f"amd0624={amd_0624}",
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
        if arm not in set(board.arm):
            continue
        sub = board[board.arm == arm]
        weak = float(sub[sub.window == "weak_feb_apr"].vs_OFF.iloc[0])
        strong = float(sub[sub.window == "strong_may_jul24"].vs_OFF.iloc[0])
        # No-op arms (identical to OFF) are not promotions.
        strong_row = sub[sub.window == "strong_may_jul24"].iloc[0]
        clears_0624 = int(strong_row.get("n_amd_hunt_0624") or 0) == 0 and int(
            board[(board.arm == "OFF") & (board.window == "strong_may_jul24")].iloc[0][
                "n_amd_hunt_0624"
            ]
            or 0
        ) == 1
        delta_trades = int(strong_row["n_trades"]) - int(
            board[(board.arm == "OFF") & (board.window == "strong_may_jul24")].iloc[0][
                "n_trades"
            ]
        )
        print(
            f"{arm}: weak={weak:.3f} strong={strong:.3f} "
            f"clears_amd0624={clears_0624} d_trades={delta_trades}",
            flush=True,
        )
        if abs(strong - 1.0) < 1e-9 and abs(weak - 1.0) < 1e-9 and delta_trades == 0:
            continue
        if strong >= 1.0 and weak >= 0.97 and clears_0624:
            verdict, best = "DUAL_PASS_WIRE", arm
            break
        if strong >= 0.95 and weak >= 0.95 and clears_0624 and verdict == "DUAL_FAIL":
            verdict, best = "DUAL_PASS_WIRE", arm
        if strong >= 0.85 and weak >= 0.85 and clears_0624 and verdict == "DUAL_FAIL":
            verdict, best = "DUAL_PASS_RESEARCH", arm

    summary = {
        "verdict": verdict,
        "best": best,
        "rule": "hunt_peer_align (+ optional hunt_asof=signal_deadline). "
        "Peer is for range_stall only — not peer_gap. "
        "T10 floor needed because Hunt decision≪fill when quotes start ~10:00.",
        "focus": FOCUS,
        "scoreboard": board.to_dict(orient="records"),
        "note_v2": (
            "HUNT_RS_PEER is no-op at decision clock (09:52 pre5 still hot). "
            "HUNT_RS_PEER_T10 clears 06-24 AMD but also cuts 07-01 META Hunt TP; "
            "strong keep≈0.92 → not wire."
        ),
    }
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    (out / "verdict.json").write_text(
        json.dumps({"verdict": verdict, "best": best}, indent=2), encoding="utf-8"
    )
    print(board.to_string(index=False), flush=True)
    print(f"verdict={verdict} best={best}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

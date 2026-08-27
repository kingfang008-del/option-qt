#!/usr/bin/env python3
"""Dual-window: peer_gap_gate (weak peer + overnight gap) vs spine OFF overlay.

Spine already has overnight_gap BLOCK (4%+adv). This gate targets residual
SL/TOX: peer==3 & fav_gap≥1.5% (04-08 AAPL, 02-18 NVDA).

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


def _gate(**kw: Any) -> dict[str, Any]:
    cfg = {
        "enabled": True,
        "mode": "block",
        "min_fav_gap": 0.015,
        "max_peer": 3,
    }
    cfg.update(kw)
    return cfg


ARMS = {
    "OFF": {"enabled": False},
    "P3_G15": _gate(min_fav_gap=0.015, max_peer=3),
    "P3_G15_UP": _gate(min_fav_gap=0.015, max_peer=3, up_only=True),
    "P3_G15_FFO": _gate(min_fav_gap=0.015, max_peer=3, max_fav_from_open=0.01),
    "P3_G18": _gate(min_fav_gap=0.018, max_peer=3),
}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", default=BASE)
    ap.add_argument("--out", default="maga7/results/peer_gap_dual_v1")
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
            p.setdefault("trade", {})["peer_gap_gate"] = copy.deepcopy(ARMS[arm])
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
            focus = d[d.date.isin(["2026-02-17", "2026-02-18", "2026-04-08", "2026-06-11"])]
            r = {
                "arm": arm,
                "window": wname,
                "total_ret": float(s.get("total_ret") or 0),
                "maxdd": float(s.get("maxdd") or 0),
                "n_trades": int(s.get("n_trades") or 0),
                "n_peer_gap_block": s.get("n_peer_gap_block"),
                "focus_day_ret_sum": float(focus["day_ret"].astype(float).sum())
                if len(focus)
                else None,
            }
            rows.append(r)
            print(
                f"  ret={r['total_ret']:.3f} maxdd={r['maxdd']:.3f} "
                f"blk={r['n_peer_gap_block']} focus_sum={r['focus_day_ret_sum']}",
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
        "rule": "peer<=max_peer & fav_gap>=min_fav_gap → block",
        "scoreboard": board.to_dict(orient="records"),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(board.to_string(index=False), flush=True)
    print(f"verdict={verdict} best={best}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

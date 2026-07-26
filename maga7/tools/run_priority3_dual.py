#!/usr/bin/env python3
"""Dual-window: clear priority residual days 02-18 / 03-12 / 06-11.

Baseline OFF = current wired research spine.

Arms:
  CROWD25 — range_stall crowd: peer≥7 & chase≥0.9 & pre5≤25bp & crowd_ffo≥1%
            (02-18 AMZN + 03-12 AAPL/NVDA; 20bp misses NVDA at minute-clock
            pre5≈24.9bp; Arm A ffo stays 1.2%)
  UGS15   — up_gap_stall @ feature_ts: UP & gap≥1.5% & |fo|≤0.1% & chase≥0.9
            & sess≤40m (06-11 TSLA)
  BOTH    — CROWD25 + UGS15
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
FOCUS = ["2026-02-18", "2026-03-12", "2026-06-11", "2026-05-11", "2026-03-05"]

RS_WIRED = {
    "enabled": True,
    "mode": "block",
    "min_chase": 0.9,
    "max_pre5": 0.00025,
    "pre_seconds": 300,
    "max_peer": 5,
    "min_fav_from_open": 0.012,
    "peer_pre5_max_peer": 3,
    "crowd_min_peer": 7,
    "crowd_max_pre5": 0.001,
}

RS_CROWD25 = {
    **RS_WIRED,
    "crowd_max_pre5": 0.0025,
    "crowd_min_fav_from_open": 0.01,
}

UGS15 = {
    "enabled": True,
    "mode": "block",
    "min_fav_gap": 0.015,
    "max_abs_from_open": 0.001,
    "min_chase": 0.9,
    "max_sess_min": 40.0,
}


def _apply(arm: str, profile: dict[str, Any]) -> dict[str, Any]:
    p = copy.deepcopy(profile)
    trade = p.setdefault("trade", {})
    if arm == "OFF":
        return p
    if arm in {"CROWD25", "BOTH"}:
        trade["range_stall_gate"] = copy.deepcopy(RS_CROWD25)
    if arm in {"UGS15", "BOTH"}:
        trade["up_gap_stall_gate"] = copy.deepcopy(UGS15)
    return p


ARMS = ("OFF", "CROWD25", "UGS15", "BOTH")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", default=BASE)
    ap.add_argument("--out", default="maga7/results/priority3_dual_v1")
    ap.add_argument("--arms", default=",".join(ARMS))
    args = ap.parse_args(argv)

    base = load_profile(args.profile)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    for arm in [a.strip() for a in args.arms.split(",") if a.strip()]:
        if arm not in ARMS:
            continue
        for wname, start, end in WINDOWS:
            p = _apply(arm, base)
            p["date_range"] = {"start": start, "end": end}
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
            tr = res["trades"]
            focus_tr = []
            if len(tr):
                for dt in FOCUS:
                    hit = tr[tr["date"].astype(str) == dt]
                    for r in hit.itertuples(index=False):
                        focus_tr.append(
                            {
                                "date": dt,
                                "symbol": r.symbol,
                                "ret": float(r.ret),
                                "reason": r.reason,
                            }
                        )
            (sub / "focus.json").write_text(
                json.dumps(focus_tr, indent=2), encoding="utf-8"
            )
            r = {
                "arm": arm,
                "window": wname,
                "total_ret": float(s.get("total_ret") or 0),
                "maxdd": float(s.get("maxdd") or 0),
                "n_trades": int(s.get("n_trades") or 0),
                "n_range_stall_block": s.get("n_range_stall_block"),
                "n_up_gap_stall_block": s.get("n_up_gap_stall_block"),
                "focus_day_ret_sum": float(focus["day_ret"].astype(float).sum())
                if len(focus)
                else None,
            }
            rows.append(r)
            print(
                f"  ret={r['total_ret']:.3f} maxdd={r['maxdd']:.3f} "
                f"rs={r['n_range_stall_block']} ugs={r['n_up_gap_stall_block']} "
                f"focus_sum={r['focus_day_ret_sum']}",
                flush=True,
            )

    board = pd.DataFrame(rows)
    off = {r["window"]: r["total_ret"] for r in rows if r["arm"] == "OFF"}
    board["vs_OFF"] = board.apply(
        lambda r: r["total_ret"] / off[r["window"]] if off.get(r["window"]) else None,
        axis=1,
    )
    board.to_csv(out / "scoreboard.csv", index=False)

    verdict, best, best_score = "DUAL_FAIL", None, -1.0
    for arm in [a for a in ARMS if a != "OFF"]:
        if arm not in set(board.arm):
            continue
        sub = board[board.arm == arm]
        weak = float(sub[sub.window == "weak_feb_apr"].vs_OFF.iloc[0])
        strong = float(sub[sub.window == "strong_may_jul23"].vs_OFF.iloc[0])
        score = weak * strong
        print(f"{arm}: weak={weak:.3f} strong={strong:.3f} score={score:.3f}", flush=True)
        if strong >= 0.97 and weak >= 0.95 and score > best_score:
            verdict, best, best_score = "DUAL_PASS_WIRE", arm, score
        elif strong >= 0.97 and weak >= 0.85 and verdict == "DUAL_FAIL" and score > best_score:
            verdict, best, best_score = "DUAL_PASS_RESEARCH", arm, score

    summary = {
        "verdict": verdict,
        "best": best,
        "rule": "CROWD25 and/or UGS15 vs wired spine",
        "scoreboard": board.to_dict(orient="records"),
    }
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    print(board.to_string(index=False), flush=True)
    print(f"verdict={verdict} best={best}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

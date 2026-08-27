#!/usr/bin/env python3
"""Ablation: QQQ+VIXY chop gate + session cumflow leaders vs research baseline.

Windows:
  weak   Jan–Mar
  mid    May1–Jul9
  jul    Jul10–23   ← target pocket (index chop / single-name trends)

Arms (same peer3_v1 trade stack; only ``session_flow_gate`` toggled).

suite=v1 (block/scale; REJECT 2026-07-25):
  PRE / CHOP_BLOCK / LEADER_ALWAYS / CHOP_AND_LEADER / CHOP_LEADER_SOFT

suite=v2 (narrow chop + leader boost; default):
  PRE
  NARROW_BLOCK     q_am≤0.3% chop → Top-K block
  NARROW_SOFT      q_am≤0.3% chop → non-leader ×0.5
  BOOST_CHOP       q_am≤0.5% chop → leader ×1.5 (others 1.0)
  BOOST_NARROW     q_am≤0.3% chop → leader ×1.5
  BOOST_TILT       q_am≤0.3% chop → leader ×1.5 + non-leader ×0.75
  BOOST_ALWAYS     always leader ×1.25 (no chop gate)

Pass hint: jul ret↑ vs PRE and weak/mid keep≥0.85 (not wired on fail).

Example:
  PYTHONPATH=. python -m maga7.tools.run_session_flow_gate_ablation \\
    --suite v2 --out maga7/results/session_flow_gate_ablation_v2
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
    ("weak_jan_mar", "2026-01-02", "2026-03-31"),
    ("mid_may_jul9", "2026-05-01", "2026-07-09"),
    ("jul10_23", "2026-07-10", "2026-07-23"),
)


def _gate(**kw: Any) -> dict[str, Any]:
    cfg = {
        "enabled": True,
        "asof": "10:30",
        "when": "chop_only",
        "mode": "block",
        "scale": 0.5,
        "boost": 1.5,
        "non_leader_scale": 1.0,
        "q_am_max": 0.005,
        "vixy_am_max": 0.015,
        "top_k": 2,
        "require_sign_align": True,
        "min_abs_cum": 0.0,
    }
    cfg.update(kw)
    return cfg


ARMS_V1: dict[str, dict[str, Any] | None] = {
    "PRE": None,
    "CHOP_BLOCK": _gate(when="chop_block"),
    "LEADER_ALWAYS": _gate(when="always"),
    "CHOP_AND_LEADER": _gate(when="chop_only", mode="block"),
    "CHOP_LEADER_SOFT": _gate(when="chop_only", mode="scale", scale=0.5),
}

ARMS_V2: dict[str, dict[str, Any] | None] = {
    "PRE": None,
    "NARROW_BLOCK": _gate(q_am_max=0.003, vixy_am_max=0.010, mode="block", top_k=2),
    "NARROW_SOFT": _gate(q_am_max=0.003, vixy_am_max=0.010, mode="scale", scale=0.5, top_k=2),
    "BOOST_CHOP": _gate(mode="boost", boost=1.5, non_leader_scale=1.0, top_k=2),
    "BOOST_NARROW": _gate(
        q_am_max=0.003, vixy_am_max=0.010, mode="boost", boost=1.5, non_leader_scale=1.0, top_k=2
    ),
    "BOOST_TILT": _gate(
        q_am_max=0.003,
        vixy_am_max=0.010,
        mode="boost",
        boost=1.5,
        non_leader_scale=0.75,
        top_k=2,
    ),
    "BOOST_ALWAYS": _gate(when="always", mode="boost", boost=1.25, non_leader_scale=1.0, top_k=2),
}

SUITES = {"v1": ARMS_V1, "v2": ARMS_V2}
ARMS = ARMS_V2


def _apply(base: dict[str, Any], arm: str, arms: dict[str, dict[str, Any] | None]) -> dict[str, Any]:
    p = copy.deepcopy(base)
    cfg = arms[arm]
    if cfg is None:
        p.pop("session_flow_gate", None)
        # ensure off if present
        p["session_flow_gate"] = {"enabled": False}
    else:
        p["session_flow_gate"] = dict(cfg)
    return p


def _run(prof: dict[str, Any], *, start: str, end: str, tag: str, out: Path) -> dict[str, Any]:
    p = copy.deepcopy(prof)
    p["date_range"] = {"start": start, "end": end}
    res = run_offline_replay(p, scheme="single")
    sub = out / tag
    sub.mkdir(parents=True, exist_ok=True)
    (sub / "summary.json").write_text(
        json.dumps(res["summary"], indent=2, default=str), encoding="utf-8"
    )
    res["daily"].to_csv(sub / "daily.csv", index=False)
    res["trades"].to_csv(sub / "trades.csv", index=False)
    s = res["summary"]
    return {
        "tag": tag,
        "total_ret": float(s.get("total_ret") or 0.0),
        "maxdd": float(s.get("maxdd") or 0.0),
        "n_trades": int(s.get("n_trades") or 0),
        "trade_win": s.get("trade_win"),
        "day_win": s.get("day_win"),
        "n_session_flow_block": s.get("n_session_flow_block"),
        "n_session_flow_scale": s.get("n_session_flow_scale"),
        "session_flow_day_counts": s.get("session_flow_day_counts"),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", default=BASE)
    ap.add_argument("--suite", default="v2", choices=sorted(SUITES.keys()))
    ap.add_argument("--out", default="maga7/results/session_flow_gate_ablation_v2")
    ap.add_argument(
        "--arms",
        default="",
        help="subset of arms (default: all in suite)",
    )
    ap.add_argument(
        "--windows",
        default=",".join(w[0] for w in WINDOWS),
        help="subset of window names",
    )
    args = ap.parse_args(argv)

    arms = SUITES[str(args.suite)]
    base = load_profile(args.profile)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    arm_names = (
        [a.strip() for a in args.arms.split(",") if a.strip()]
        if str(args.arms).strip()
        else list(arms.keys())
    )
    want_w = {w.strip() for w in args.windows.split(",") if w.strip()}
    windows = [w for w in WINDOWS if w[0] in want_w]

    rows: list[dict[str, Any]] = []
    for arm in arm_names:
        if arm not in arms:
            print(f"skip unknown arm {arm}", flush=True)
            continue
        prof = _apply(base, arm, arms)
        for wname, start, end in windows:
            tag = f"{arm}__{wname}"
            print(f"=== {tag} ===", flush=True)
            r = _run(prof, start=start, end=end, tag=tag, out=out)
            r["arm"] = arm
            r["window"] = wname
            rows.append(r)
            print(
                f"  ret={r['total_ret']:.3f} maxdd={r['maxdd']:.3f} n={r['n_trades']} "
                f"block={r['n_session_flow_block']} scale={r['n_session_flow_scale']} "
                f"days={r['session_flow_day_counts']}",
                flush=True,
            )

    board = pd.DataFrame(rows)
    pre = {
        r["window"]: float(r["total_ret"])
        for r in rows
        if r["arm"] == "PRE"
    }
    board["vs_PRE"] = board.apply(
        lambda r: (float(r["total_ret"]) / pre[r["window"]]) if pre.get(r["window"]) else None,
        axis=1,
    )
    board.to_csv(out / "scoreboard.csv", index=False)

    # pick best on jul with keep on weak/mid
    jul = board[board.window == "jul10_23"].copy()
    best = None
    verdict = "NO_LIFT"
    for _, r in jul.sort_values("total_ret", ascending=False).iterrows():
        if r["arm"] == "PRE":
            continue
        weak = board[(board.arm == r["arm"]) & (board.window == "weak_jan_mar")]
        mid = board[(board.arm == r["arm"]) & (board.window == "mid_may_jul9")]
        if weak.empty or mid.empty:
            continue
        kw = float(weak.iloc[0]["vs_PRE"] or 0)
        km = float(mid.iloc[0]["vs_PRE"] or 0)
        if float(r["vs_PRE"] or 0) > 1.0 and kw >= 0.85 and km >= 0.85:
            best = r.to_dict()
            verdict = "SESSION_FLOW_LIFT"
            break
    if best is None and len(jul):
        # partial: jul improves but hurts a prior window
        cand = jul[jul.arm != "PRE"].sort_values("total_ret", ascending=False)
        if len(cand) and float(cand.iloc[0]["vs_PRE"] or 0) > 1.0:
            best = cand.iloc[0].to_dict()
            verdict = "JUL_ONLY_PARTIAL"

    summary = {
        "verdict": verdict,
        "best": best,
        "suite": str(args.suite),
        "windows": [list(w) for w in windows],
        "arms": arm_names,
        "scoreboard": board.to_dict(orient="records"),
        "note": (
            "Proactive QQQ+VIXY chop + session cumflow leaders. "
            "Research only; do not wire without SESSION_FLOW_LIFT."
        ),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print("\n=== scoreboard ===", flush=True)
    cols = [
        c
        for c in [
            "arm",
            "window",
            "total_ret",
            "maxdd",
            "n_trades",
            "vs_PRE",
            "n_session_flow_block",
            "session_flow_day_counts",
        ]
        if c in board.columns
    ]
    print(board[cols].to_string(index=False), flush=True)
    print(f"\nverdict={verdict} best={best.get('arm') if best else None}", flush=True)
    print(f"wrote {out}", flush=True)
    # Always 0 so pipelines with `| tee` don't mask the printed verdict.
    # Callers should read summary.json["verdict"].
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

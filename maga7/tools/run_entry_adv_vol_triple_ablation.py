#!/usr/bin/env python3
"""Three-window ablation: entry adverse-vol gate vs peer3_v1 (Jul toxic rescue).

Windows (same as session_flow_gate):
  weak   Jan–Mar
  mid    May1–Jul9
  jul    Jul10–23

Probe note (2026-07-25): Jul META/MSFT losers enter on *favorable* stock path with
cool adverse-vol share; AMD UP tox is the main share-hot pocket. Expect partial
rescue at best — PASS still requires jul↑ and weak/mid keep≥0.85.

Example:
  PYTHONPATH=. python -m maga7.tools.run_entry_adv_vol_triple_ablation \\
    --out maga7/results/entry_adv_vol_triple_v1
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


def _eav(**kw: Any) -> dict[str, Any]:
    cfg = {
        "enabled": True,
        "action": "scale",
        "window_seconds": 120,
        "max_share": 0.55,
        "scale": 0.5,
        "on_missing": "allow",
    }
    cfg.update(kw)
    return {"entry_adv_vol": cfg}


ARMS: dict[str, dict[str, Any] | None] = {
    "PRE": None,
    "SCALE55_W120": _eav(action="scale", max_share=0.55, window_seconds=120, scale=0.5),
    "BLOCK55_W120": _eav(action="block", max_share=0.55, window_seconds=120),
    "SCALE50_W120": _eav(action="scale", max_share=0.50, window_seconds=120, scale=0.5),
    "SCALE50_W60": _eav(action="scale", max_share=0.50, window_seconds=60, scale=0.5),
    "SCALE50_UP": _eav(
        action="scale", max_share=0.50, window_seconds=120, scale=0.5, dirs=["UP"]
    ),
    "SCALE55_LAG60": _eav(
        action="scale",
        max_share=0.55,
        window_seconds=120,
        scale=0.5,
        lag_seconds=60,
    ),
    "BLOCK55_LAG60": _eav(
        action="block",
        max_share=0.55,
        window_seconds=120,
        lag_seconds=60,
    ),
}


def _apply(base: dict[str, Any], arm: str) -> dict[str, Any]:
    p = copy.deepcopy(base)
    trade = p.setdefault("trade", {})
    cfg = ARMS[arm]
    if cfg is None:
        trade["entry_adv_vol"] = {"enabled": False}
    else:
        trade.update(copy.deepcopy(cfg))
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
        "n_entry_adv_vol_block": s.get("n_entry_adv_vol_block"),
        "n_entry_adv_vol_scale": s.get("n_entry_adv_vol_scale"),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", default=BASE)
    ap.add_argument("--out", default="maga7/results/entry_adv_vol_triple_v1")
    ap.add_argument("--arms", default=",".join(ARMS.keys()))
    ap.add_argument("--windows", default=",".join(w[0] for w in WINDOWS))
    args = ap.parse_args(argv)

    base = load_profile(args.profile)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    arm_names = [a.strip() for a in args.arms.split(",") if a.strip()]
    want_w = {w.strip() for w in args.windows.split(",") if w.strip()}
    windows = [w for w in WINDOWS if w[0] in want_w]

    rows: list[dict[str, Any]] = []
    for arm in arm_names:
        if arm not in ARMS:
            print(f"skip unknown arm {arm}", flush=True)
            continue
        prof = _apply(base, arm)
        for wname, start, end in windows:
            tag = f"{arm}__{wname}"
            print(f"=== {tag} ===", flush=True)
            r = _run(prof, start=start, end=end, tag=tag, out=out)
            r["arm"] = arm
            r["window"] = wname
            rows.append(r)
            print(
                f"  ret={r['total_ret']:.3f} maxdd={r['maxdd']:.3f} n={r['n_trades']} "
                f"block={r['n_entry_adv_vol_block']} scale={r['n_entry_adv_vol_scale']}",
                flush=True,
            )

    board = pd.DataFrame(rows)
    pre = {r["window"]: float(r["total_ret"]) for r in rows if r["arm"] == "PRE"}
    board["vs_PRE"] = board.apply(
        lambda r: (float(r["total_ret"]) / pre[r["window"]]) if pre.get(r["window"]) else None,
        axis=1,
    )
    board.to_csv(out / "scoreboard.csv", index=False)

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
            verdict = "ENTRY_ADV_VOL_LIFT"
            break
    if best is None and len(jul):
        cand = jul[jul.arm != "PRE"].sort_values("total_ret", ascending=False)
        if len(cand) and float(cand.iloc[0]["vs_PRE"] or 0) > 1.0:
            best = cand.iloc[0].to_dict()
            verdict = "JUL_ONLY_PARTIAL"

    summary = {
        "verdict": verdict,
        "best": best,
        "windows": [list(w) for w in windows],
        "arms": arm_names,
        "scoreboard": board.to_dict(orient="records"),
        "note": (
            "Entry adverse-vol share gate. Jul META/MSFT often cool-share / "
            "favorable-into-entry; AMD is the main hot pocket. Do not wire without LIFT."
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
            "n_entry_adv_vol_block",
            "n_entry_adv_vol_scale",
        ]
        if c in board.columns
    ]
    print(board[cols].to_string(index=False), flush=True)
    print(f"\nverdict={verdict} best={best.get('arm') if best else None}", flush=True)
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Dual-window scoreboard: open_washout breadth gate vs freeze / reclaim.

Research only. Does not enable freeze ``regime_router``.
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


def _run(prof: dict, *, start: str, end: str, tag: str, out: Path) -> dict:
    p = copy.deepcopy(prof)
    p["date_range"] = {"start": start, "end": end}
    res = run_offline_replay(p, scheme="single")
    s = res["summary"]
    sub = out / tag
    sub.mkdir(parents=True, exist_ok=True)
    (sub / "summary.json").write_text(json.dumps(s, indent=2, default=str), encoding="utf-8")
    res["daily"].to_csv(sub / "daily.csv", index=False)
    res["trades"].to_csv(sub / "trades.csv", index=False)
    d0717 = None
    hit = res["daily"][res["daily"]["date"].astype(str) == "2026-07-17"]
    if len(hit):
        d0717 = float(hit.iloc[0]["day_ret"])
    return {
        "tag": tag,
        "total_ret": float(s["total_ret"]),
        "maxdd": float(s["maxdd"]),
        "n_trades": int(s["n_trades"]),
        "trade_win": s.get("trade_win"),
        "n_router_expert_days": s.get("n_router_expert_days"),
        "router_day_counts": s.get("router_day_counts"),
        "day_ret_0717": d0717,
    }


def _with_router(base: dict, router: dict) -> dict:
    p = copy.deepcopy(base)
    p["regime_router"] = router
    return p


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--profile",
        default="maga7/CONFIG/strategy_profiles/single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1.json",
    )
    ap.add_argument("--experts", default="maga7/CONFIG/regime_router/experts_v1.json")
    ap.add_argument("--out", default="maga7/results/regime_router/washout_gate_scoreboard")
    ap.add_argument("--strong-start", default="2026-05-01")
    ap.add_argument("--strong-end", default="2026-07-17")
    ap.add_argument("--weak-start", default="2026-02-01")
    ap.add_argument("--weak-end", default="2026-04-30")
    args = ap.parse_args()

    base = load_profile(args.profile)
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)

    experts_path = args.experts
    variants = {
        "baseline": None,  # freeze: router off
        "reclaim_disp55": {
            "enabled": True,
            "mode": "rule",
            "rule": "reclaim_disp55",
            "asof": "10:30",
            "experts_path": experts_path,
        },
        "washout_b3_dn": {
            "enabled": True,
            "mode": "rule",
            "rule": "washout_breadth3",
            "asof": "10:30",
            "wash_window_end": "10:00",
            "wash_drop_min": 0.003,
            "washout_breadth_min": 3,
            "washout_expert": "washout_gate_dn",
            "experts_path": experts_path,
        },
        "washout_b3_both": {
            "enabled": True,
            "mode": "rule",
            "rule": "washout_breadth3",
            "asof": "10:30",
            "wash_window_end": "10:00",
            "wash_drop_min": 0.003,
            "washout_breadth_min": 3,
            "washout_expert": "washout_gate_both",
            "experts_path": experts_path,
        },
        "washout_b3_halt": {
            "enabled": True,
            "mode": "rule",
            "rule": "washout_breadth3",
            "asof": "10:30",
            "wash_window_end": "10:00",
            "wash_drop_min": 0.003,
            "washout_breadth_min": 3,
            "washout_expert": "washout_gate_halt",
            "experts_path": experts_path,
        },
        "washout_or_reclaim_dn": {
            "enabled": True,
            "mode": "rule",
            "rule": "washout_or_reclaim",
            "asof": "10:30",
            "wash_window_end": "10:00",
            "wash_drop_min": 0.003,
            "washout_breadth_min": 3,
            "washout_expert": "washout_gate_dn",
            "experts_path": experts_path,
        },
        "washout_or_reclaim_both": {
            "enabled": True,
            "mode": "rule",
            "rule": "washout_or_reclaim",
            "asof": "10:30",
            "wash_window_end": "10:00",
            "wash_drop_min": 0.003,
            "washout_breadth_min": 3,
            "washout_expert": "washout_gate_both",
            "experts_path": experts_path,
        },
    }

    windows = [
        ("strong_may_jul", args.strong_start, args.strong_end),
        ("weak_feb_apr", args.weak_start, args.weak_end),
    ]
    rows = []
    for vname, router in variants.items():
        prof = base if router is None else _with_router(base, router)
        # ensure baseline keeps router disabled even if profile has a block
        if router is None:
            prof = copy.deepcopy(base)
            rr = dict(prof.get("regime_router") or {})
            rr["enabled"] = False
            prof["regime_router"] = rr
        for wname, start, end in windows:
            tag = f"{vname}__{wname}"
            print(f"=== {tag} ===", flush=True)
            r = _run(prof, start=start, end=end, tag=tag, out=out)
            r["variant"] = vname
            r["window"] = wname
            rows.append(r)
            print(
                f"  ret={r['total_ret']:.3f} maxdd={r['maxdd']:.3f} "
                f"n={r['n_trades']} expert_days={r['n_router_expert_days']} "
                f"0717={r['day_ret_0717']}",
                flush=True,
            )

    board = pd.DataFrame(rows)
    # relative to baseline per window
    base_ret = {
        r["window"]: r["total_ret"] for r in rows if r["variant"] == "baseline"
    }
    board["vs_baseline"] = board.apply(
        lambda r: (r["total_ret"] / base_ret[r["window"]]) if base_ret.get(r["window"]) else None,
        axis=1,
    )
    board_out = board.drop(columns=["tag"], errors="ignore")
    board_out.to_csv(out / "scoreboard.csv", index=False)
    summary = {
        "profile": args.profile,
        "windows": windows,
        "variants": list(variants.keys()),
        "scoreboard": board_out.to_dict(orient="records"),
        "note": "washout breadth gate research; freeze untouched",
    }
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    print(board_out.to_string(index=False))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

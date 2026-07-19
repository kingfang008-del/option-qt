#!/usr/bin/env python3
"""Dual-window ablation: split entry + pullback scale-in with factor confirm.

Stacks ``trade.scale_in`` on current research baseline (L2+TT1_05). Does not
mutate the profile file.
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

PROFILE = (
    "maga7/CONFIG/strategy_profiles/"
    "single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1.json"
)


def _run(prof: dict, *, start: str, end: str, tag: str, out: Path) -> dict:
    p = copy.deepcopy(prof)
    p["date_range"] = {"start": start, "end": end}
    res = run_offline_replay(p, scheme="single")
    s = res["summary"]
    trades = res["trades"]
    daily = res["daily"]
    sub = out / tag
    sub.mkdir(parents=True, exist_ok=True)
    (sub / "summary.json").write_text(json.dumps(s, indent=2, default=str), encoding="utf-8")
    trades.to_csv(sub / "trades.csv", index=False)
    daily.to_csv(sub / "daily.csv", index=False)
    n_add = int(trades["scale_in_added"].sum()) if len(trades) and "scale_in_added" in trades.columns else 0
    worst_trade = float(trades["ret"].min()) if len(trades) else None
    worst_day = float(daily["day_ret"].min()) if len(daily) else None
    focus = ["2026-05-06", "2026-05-11", "2026-06-24"]
    focus_rows = []
    if len(trades):
        for d in focus:
            sub_t = trades[trades["date"].astype(str) == d]
            for r in sub_t.itertuples(index=False):
                focus_rows.append(
                    {
                        "date": d,
                        "symbol": getattr(r, "symbol", None),
                        "dir": getattr(r, "dir", None),
                        "ret": float(getattr(r, "ret", float("nan"))),
                        "reason": getattr(r, "reason", None),
                        "added": bool(getattr(r, "scale_in_added", False)),
                        "deployed": float(getattr(r, "scale_in_deployed_frac", 1.0) or 1.0),
                    }
                )
    return {
        "tag": tag,
        "total_ret": float(s["total_ret"]),
        "maxdd": float(s["maxdd"]),
        "n_trades": int(s["n_trades"]),
        "trade_win": s.get("trade_win"),
        "trade_exp": s.get("trade_exp"),
        "n_scale_in_added": n_add,
        "worst_trade": worst_trade,
        "worst_day": worst_day,
        "focus": focus_rows,
        "end_equity": float(s["end_equity"]),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", default=PROFILE)
    ap.add_argument("--out", default="maga7/results/scale_in_ablation_peer3_dual_window")
    ap.add_argument("--strong-start", default="2026-05-01")
    ap.add_argument("--strong-end", default="2026-07-17")
    ap.add_argument("--weak-start", default="2026-02-01")
    ap.add_argument("--weak-end", default="2026-04-30")
    args = ap.parse_args()

    base = load_profile(args.profile)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    def si(**kw):
        return {"scale_in": {"enabled": True, "first_frac": 0.5, "add_frac": 0.5, "min_hold_seconds": 120, **kw}}

    variants: list[tuple[str, dict]] = [
        ("00_baseline", {}),
        ("01_half_only", si(confirm_mode="never", pullback_ret=0.30)),
        ("02_pb20_always", si(confirm_mode="always", pullback_ret=0.20)),
        ("03_pb30_always", si(confirm_mode="always", pullback_ret=0.30)),
        ("04_pb20_mf", si(confirm_mode="mf", pullback_ret=0.20)),
        ("05_pb30_mf", si(confirm_mode="mf", pullback_ret=0.30)),
        ("06_pb30_mf_streak", si(confirm_mode="mf_streak", pullback_ret=0.30)),
        ("07_pb25_mf", si(confirm_mode="mf", pullback_ret=0.25)),
    ]

    windows = [
        ("strong_may_jul", args.strong_start, args.strong_end),
        ("weak_feb_apr", args.weak_start, args.weak_end),
    ]
    board = []
    for wname, start, end in windows:
        base_ret = None
        for vname, patch in variants:
            p = copy.deepcopy(base)
            p.setdefault("trade", {}).update(patch)
            print(f"==> {wname} / {vname}", flush=True)
            row = _run(p, start=start, end=end, tag=f"{wname}__{vname}", out=out)
            if vname == "00_baseline":
                base_ret = row["total_ret"]
            row["window"] = wname
            row["variant"] = vname
            row["vs_baseline"] = (row["total_ret"] / base_ret) if base_ret else None
            board.append(row)
            print(
                f"    ret={row['total_ret']:+.2%} dd={row['maxdd']:.2%} "
                f"vs={row['vs_baseline']:.1%} add={row['n_scale_in_added']} "
                f"worst_t={row['worst_trade']:.2%} worst_d={row['worst_day']:.2%}",
                flush=True,
            )

    (out / "scoreboard.json").write_text(json.dumps(board, indent=2, default=str), encoding="utf-8")
    flat = []
    for r in board:
        flat.append({k: v for k, v in r.items() if k != "focus"})
    pd.DataFrame(flat).to_csv(out / "scoreboard.csv", index=False)
    cols = [
        "window",
        "variant",
        "total_ret",
        "maxdd",
        "n_trades",
        "vs_baseline",
        "n_scale_in_added",
        "worst_trade",
        "worst_day",
        "trade_win",
        "trade_exp",
    ]
    print(pd.DataFrame(flat)[cols].to_string(index=False))
    print("--- focus trades ---")
    for r in board:
        if r.get("focus"):
            print(r["window"], r["variant"], r["focus"])
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

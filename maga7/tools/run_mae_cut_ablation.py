#!/usr/bin/env python3
"""Dual-window ablation: hold_extend + option MAE early cut (toxic path).

Stacks ``early_exit_mode=mae_cut`` on freeze ``exit_mode=hold_extend``.
Does not mutate the freeze profile (research switch only).
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
    trades = res["trades"]
    sub = out / tag
    sub.mkdir(parents=True, exist_ok=True)
    (sub / "summary.json").write_text(json.dumps(s, indent=2, default=str), encoding="utf-8")
    trades.to_csv(sub / "trades.csv", index=False)
    res["daily"].to_csv(sub / "daily.csv", index=False)
    n_mae = int((trades["reason"] == "MAE_CUT").sum()) if not trades.empty and "reason" in trades.columns else 0
    day_0717 = None
    daily = res["daily"]
    if not daily.empty:
        hit = daily[daily["date"].astype(str) == "2026-07-17"]
        if not hit.empty:
            day_0717 = float(hit.iloc[0]["day_ret"])
    # 07-17 DN trade rets if any
    t0717 = []
    if not trades.empty:
        sub_t = trades[trades["date"].astype(str) == "2026-07-17"]
        for r in sub_t.itertuples(index=False):
            t0717.append(
                {
                    "symbol": getattr(r, "symbol", None),
                    "dir": getattr(r, "dir", None),
                    "ret": float(getattr(r, "ret", float("nan"))),
                    "reason": getattr(r, "reason", None),
                }
            )
    return {
        "tag": tag,
        "total_ret": float(s["total_ret"]),
        "maxdd": float(s["maxdd"]),
        "n_trades": int(s["n_trades"]),
        "trade_win": s.get("trade_win"),
        "n_mae_cut": n_mae,
        "day_ret_0717": day_0717,
        "trades_0717": t0717,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--profile",
        default="maga7/CONFIG/strategy_profiles/single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1.json",
    )
    ap.add_argument("--out", default="maga7/results/mae_cut_ablation_dual_window")
    ap.add_argument("--strong-start", default="2026-05-01")
    ap.add_argument("--strong-end", default="2026-07-17")
    ap.add_argument("--weak-start", default="2026-02-01")
    ap.add_argument("--weak-end", default="2026-04-30")
    args = ap.parse_args()

    base = load_profile(args.profile)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    variants: list[tuple[str, dict]] = [
        ("baseline", {}),
        (
            "mae25_bypass20",
            {
                "early_exit_mode": "mae_cut",
                "mae_cut_ret": 0.25,
                "mae_cut_mfe_bypass": 0.20,
                "mae_cut_min_hold_minutes": 5,
                "mae_cut_only_dn": False,
            },
        ),
        (
            "mae25_bypass20_dn",
            {
                "early_exit_mode": "mae_cut",
                "mae_cut_ret": 0.25,
                "mae_cut_mfe_bypass": 0.20,
                "mae_cut_min_hold_minutes": 5,
                "mae_cut_only_dn": True,
            },
        ),
        (
            "mae30_bypass20",
            {
                "early_exit_mode": "mae_cut",
                "mae_cut_ret": 0.30,
                "mae_cut_mfe_bypass": 0.20,
                "mae_cut_min_hold_minutes": 5,
                "mae_cut_only_dn": False,
            },
        ),
        (
            "mae25_bypass40",
            {
                "early_exit_mode": "mae_cut",
                "mae_cut_ret": 0.25,
                "mae_cut_mfe_bypass": 0.40,
                "mae_cut_min_hold_minutes": 5,
                "mae_cut_only_dn": False,
            },
        ),
        (
            "mae20_bypass15_dn",
            {
                "early_exit_mode": "mae_cut",
                "mae_cut_ret": 0.20,
                "mae_cut_mfe_bypass": 0.15,
                "mae_cut_min_hold_minutes": 3,
                "mae_cut_only_dn": True,
            },
        ),
        (
            "mae25_dn_mf",
            {
                "early_exit_mode": "mae_cut",
                "mae_cut_ret": 0.25,
                "mae_cut_mfe_bypass": 0.20,
                "mae_cut_min_hold_minutes": 5,
                "mae_cut_only_dn": True,
                "mae_cut_require_mf_against": True,
            },
        ),
        (
            "mae30_dn_mf",
            {
                "early_exit_mode": "mae_cut",
                "mae_cut_ret": 0.30,
                "mae_cut_mfe_bypass": 0.20,
                "mae_cut_min_hold_minutes": 5,
                "mae_cut_only_dn": True,
                "mae_cut_require_mf_against": True,
            },
        ),
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
            row = _run(p, start=start, end=end, tag=f"{wname}__{vname}", out=out)
            if vname == "baseline":
                base_ret = row["total_ret"]
            row["window"] = wname
            row["variant"] = vname
            row["vs_baseline"] = (row["total_ret"] / base_ret) if base_ret else None
            board.append(row)

    (out / "scoreboard.json").write_text(json.dumps(board, indent=2, default=str), encoding="utf-8")
    flat = [{k: v for k, v in r.items() if k != "trades_0717"} for r in board]
    pd.DataFrame(flat).to_csv(out / "scoreboard.csv", index=False)
    cols = [
        "window",
        "variant",
        "total_ret",
        "maxdd",
        "n_trades",
        "vs_baseline",
        "n_mae_cut",
        "day_ret_0717",
        "trade_win",
    ]
    print(pd.DataFrame(flat)[cols].to_string(index=False))
    print("--- 07-17 trades ---")
    for r in board:
        if r["window"] == "strong_may_jul" and r.get("trades_0717"):
            print(r["variant"], r["trades_0717"])
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
